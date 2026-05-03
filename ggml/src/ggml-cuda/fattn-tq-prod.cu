#include "common.cuh"
#include "fattn-tq-prod.cuh"

void ggml_cuda_flash_attn_ext_tq_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q->type  == GGML_TYPE_F32);
    GGML_ASSERT(K->type  == GGML_TYPE_TQ_PROD);
    GGML_ASSERT(V->type  == GGML_TYPE_TQ_PROD);
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(Q->ne[0] == 128);

    float logit_softcap, max_bias, m0, m1, scale;
    uint32_t n_head_log2;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int ne02 = Q->ne[2];
    {
        const uint32_t v = 1u << (uint32_t) floorf(log2f((float) ne02));
        n_head_log2 = v;
        m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
        m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    }

    const int ne01 = Q->ne[1];
    const int ne03 = Q->ne[3];
    const int nb01 = Q->nb[1];
    const int nb02 = Q->nb[2];
    const int nb03 = Q->nb[3];

    const int ne11 = K->ne[1];
    const int ne12 = K->ne[2];
    const int nb11 = K->nb[1];
    const int nb12 = K->nb[2];
    const int64_t nb13 = K->nb[3];

    const int nb21 = V->nb[1];
    const int nb22 = V->nb[2];
    const int64_t nb23 = V->nb[3];

    const int ne32 = mask ? mask->ne[3] : 1;
    const int nb31 = mask ? mask->nb[1] : 0;
    const int64_t nb33 = mask ? mask->nb[3] : 0;

    constexpr int D = 128;
    constexpr int dim_idx = 1;
    GGML_ASSERT(tq_d_Pi[dim_idx]   != nullptr && "TQ Pi not initialised");
    GGML_ASSERT(tq_d_Pi_T[dim_idx] != nullptr && "TQ Pi_T not initialised");
    GGML_ASSERT(tq_d_S[dim_idx]    != nullptr && "TQ S not initialised");
    GGML_ASSERT(tq_d_S_T[dim_idx]  != nullptr && "TQ S_T not initialised");

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int base_blocks    = ne01 * ne02 * ne03;
    const int min_splits     = (4 * nsm + base_blocks - 1) / base_blocks;
    const int tks = (ne11 > 16384) ? 1024 : 256;  // larger splits at long context to fit VKQ_parts in L2
    const int splits_by_size = (ne11 + tks - 1) / tks;
    int n_splits = min(ne11, max(min_splits, splits_by_size));
    if (n_splits < 1) n_splits = 1;

    const int64_t n_queries = (int64_t)ne01 * ne02 * ne03;

    // Step 1: rotate Q once → q_pi_buf, q_s_buf
    ggml_cuda_pool_alloc<float> q_pi_buf(ctx.pool(), n_queries * D);
    ggml_cuda_pool_alloc<float> q_s_buf (ctx.pool(), n_queries * D);
    {
        const dim3 grid_rot(ne01, ne02 * ne03);
        const dim3 block_rot(D);
        const size_t smem_rot = D * sizeof(float);
        tq_prod_rotate_q_kernel<D><<<grid_rot, block_rot, smem_rot, ctx.stream()>>>(
            (const char *) Q->data,
            q_pi_buf.ptr, q_s_buf.ptr,
            tq_d_Pi_T[dim_idx], tq_d_S_T[dim_idx],
            scale,
            ne01, ne02, ne03,
            nb01, nb02, nb03
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: split-K partial attention (stores raw [acc_rot|acc_sqjl], 2D per split)
    ggml_cuda_pool_alloc<float>  dst_tmp     (ctx.pool(), n_queries * n_splits * 2 * D);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(ctx.pool(), n_queries * n_splits);

    // warp-parallel layout: [2*NWARPS*D acc | 3*NWARPS softmax scalars | 2 global scalars]
    constexpr int NWARPS = D / 32;
    const size_t smem_partial = (2 * NWARPS * D + 3 * NWARPS + 2) * sizeof(float);
    const dim3 grid_partial(ne01, n_splits, ne02 * ne03);
    const dim3 block_dim(D, 1, 1);

    flash_attn_tq_prod_partial_kernel<D><<<grid_partial, block_dim, smem_partial, ctx.stream()>>>(
        q_pi_buf.ptr, q_s_buf.ptr,
        (const char *) K->data,
        (const char *) V->data,
        mask ? (const char *) mask->data : nullptr,
        dst_tmp.ptr, dst_tmp_meta.ptr,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne01, ne02, ne03,
        ne11, ne12,
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        ne32, nb31, nb33,
        n_splits
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 3: combine splits + single Pi^T/S^T unrotation
    const size_t smem_combine = (2 * n_splits + 2 * D + 1) * sizeof(float);
    const dim3 grid_combine(ne01, ne02, ne03);

    flash_attn_tq_prod_combine<D><<<grid_combine, block_dim, smem_combine, ctx.stream()>>>(
        dst_tmp.ptr, dst_tmp_meta.ptr,
        (float *) dst->data,
        tq_d_Pi[dim_idx], tq_d_S[dim_idx],
        n_splits
    );
    CUDA_CHECK(cudaGetLastError());
}
