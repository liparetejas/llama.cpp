#include "common.cuh"
#include "fattn-tq-mse-fused.cuh"

void ggml_cuda_flash_attn_ext_tq_mse_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q->type  == GGML_TYPE_F32);
    GGML_ASSERT(K->type  == GGML_TYPE_TQ_MSE);
    GGML_ASSERT(V->type  == GGML_TYPE_TQ_MSE);
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(Q->ne[0] == 128);

    float scale, max_bias, logit_softcap;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int ne02 = Q->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) ne02));
    const float m0 = powf(2.0f, -(max_bias)        / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int ne01 = Q->ne[1];
    const int ne03 = Q->ne[3];

    constexpr int D = 128;
    constexpr int dim_idx = 1;
    GGML_ASSERT(tq_d_Pi[dim_idx]   != nullptr && "TQ Pi not initialised");
    GGML_ASSERT(tq_d_Pi_T[dim_idx] != nullptr && "TQ Pi_T not initialised");

    // smem: D + NWARPS*D + 3*NWARPS + 2 floats
    constexpr int NWARPS = D / 32;
    const size_t smem_sz = (D + NWARPS*D + 3*NWARPS + 2) * sizeof(float);

    const dim3 grid (ne01, ne02 * ne03);
    const dim3 block(D);

    flash_attn_tq_mse_fused_kernel<D><<<grid, block, smem_sz, ctx.stream()>>>(
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? (const char *) mask->data : nullptr,
        (float *) dst->data,
        tq_d_Pi_T[dim_idx],
        tq_d_Pi[dim_idx],
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        (int32_t)Q->ne[1], (int32_t)Q->ne[2], (int32_t)Q->ne[3],
        (int32_t)K->ne[1], (int32_t)K->ne[2],
        (int32_t)Q->nb[1], (int32_t)Q->nb[2], (int64_t)Q->nb[3],
        (int32_t)K->nb[1], (int32_t)K->nb[2], (int64_t)K->nb[3],
        (int32_t)V->nb[1], (int32_t)V->nb[2], (int64_t)V->nb[3],
        mask ? (int32_t)mask->ne[3] : 1,
        mask ? (int32_t)mask->nb[1] : 0,
        mask ? (int64_t)mask->nb[3] : 0LL
    );
    CUDA_CHECK(cudaGetLastError());
}
