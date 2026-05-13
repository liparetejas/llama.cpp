#include "common.cuh"
#include "fattn-tq-mse.cuh"
#include "fattn-tq-mse-vec.cuh"

void ggml_cuda_flash_attn_ext_tq_mse_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q->type   == GGML_TYPE_F32);
    GGML_ASSERT(K->type   == GGML_TYPE_TQ_MSE);
    GGML_ASSERT(V->type   == GGML_TYPE_TQ_MSE);
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(Q->ne[0]  == 128);

    float logit_softcap, max_bias, scale;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int ne02 = Q->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) ne02));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

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

    constexpr int D       = 128;
    constexpr int dim_idx = 1;
    GGML_ASSERT(tq_d_Pi[dim_idx]   != nullptr && "TQ Pi not initialised");
    GGML_ASSERT(tq_d_Pi_T[dim_idx] != nullptr && "TQ Pi_T not initialised");

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int base_blocks    = ne01 * ne02 * ne03;
    const int min_splits     = (4 * nsm + base_blocks - 1) / base_blocks;
    const int tks = (ne11 > 16384) ? 1024 : 256;
    const int splits_by_size = (ne11 + tks - 1) / tks;
    int n_splits = min(ne11, max(min_splits, splits_by_size));
    if (n_splits < 1) n_splits = 1;

    const int64_t n_queries = (int64_t)ne01 * ne02 * ne03;

    // VEC partial kernel: no separate rotate_q step; d_Pi_T passed directly
    ggml_cuda_pool_alloc<float>  dst_tmp     (ctx.pool(), n_queries * n_splits * D);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(ctx.pool(), n_queries * n_splits);

    constexpr int NWARPS = D / 32;
    const size_t smem_vec = (D + NWARPS*D + 3*NWARPS + 2) * sizeof(float);  // 2616 bytes
    const dim3 grid_vec(ne01, n_splits, ne02 * ne03);
    const dim3 block_dim(D);

    flash_attn_tq_mse_vec_kernel<D><<<grid_vec, block_dim, smem_vec, ctx.stream()>>>(
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? (const char *) mask->data : nullptr,
        dst_tmp.ptr,
        dst_tmp_meta.ptr,
        tq_d_Pi_T[dim_idx],
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne01, ne02, ne03,
        ne11, ne12,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        ne32, nb31, nb33,
        n_splits
    );
    CUDA_CHECK(cudaGetLastError());

    // Combine splits + Pi^T unrotation (reuse existing kernel unchanged)
    const size_t smem_combine = (2 * n_splits + D + 1) * sizeof(float);
    const dim3 grid_combine(ne01, ne02, ne03);

    flash_attn_tq_mse_combine<D><<<grid_combine, block_dim, smem_combine, ctx.stream()>>>(
        dst_tmp.ptr,
        dst_tmp_meta.ptr,
        (float *) dst->data,
        tq_d_Pi[dim_idx],
        n_splits
    );
    CUDA_CHECK(cudaGetLastError());
}
