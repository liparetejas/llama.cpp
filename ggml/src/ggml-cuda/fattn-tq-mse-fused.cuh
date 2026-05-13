#pragma once
#include "common.cuh"
#include "fattn-common.cuh"
#include "turbo-quant.cuh"

// ---------------------------------------------------------------------------
// Fused single-kernel TQ_MSE flash-attention.
// Replaces the 3-kernel pipeline (rotate_q → partial → combine) with ONE
// kernel that does everything inline:
//   Phase 1 — rotate Q with Pi^T (smem matmul, coalesced Pi_T reads)
//   Phase 2 — flash-attention hot loop over ALL K tokens (warp-stride)
//   Phase 3 — merge 4 warp accumulators, normalize, rotate output with Pi
//
// Grid : (ne01, ne02*ne03)  — one block per (query, head×seq)
// Block: D = 128 threads (NWARPS=4 warps of 32 lanes each)
// Smem : D + NWARPS*D + 3*NWARPS + 2 floats  ≈ 2616 bytes
//
// Memory traffic vs 3-kernel approach (d=16384, n_splits=64):
//   Eliminated: 786 KB dst_tmp write + 786 KB read, 12 KB meta write/read,
//               2 extra kernel launches (rotate_q + combine).
//   Added     : one Pi_T read (64 KB) and one Pi read (64 KB) per block,
//               both cached in L2 after the first block warms them.
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_mse_fused_kernel(
        const char  * __restrict__ Q_data,
        const char  * __restrict__ K_data,
        const char  * __restrict__ V_data,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        const float * __restrict__ d_Pi_T,   // Pi^T row-major: d_Pi_T[j*D+i] coalesced
        const float * __restrict__ d_Pi,     // Pi  row-major: d_Pi[j*D+i]   coalesced for unrotation
        const float  scale,
        const float  max_bias,
        const float  m0,
        const float  m1,
        const uint32_t n_head_log2,
        const float  logit_softcap,
        const int32_t ne01,
        const int32_t ne02,
        const int32_t ne03,
        const int32_t ne11,
        const int32_t ne12,
        const int32_t nb01,
        const int32_t nb02,
        const int64_t nb03,
        const int32_t nb11,
        const int32_t nb12,
        const int64_t nb13,
        const int32_t nb21,
        const int32_t nb22,
        const int64_t nb23,
        const int32_t ne32,
        const int32_t nb31,
        const int64_t nb33
) {
    static_assert(D == 128, "fattn-tq-mse-fused: only D=128 is supported");
    constexpr int NWARPS  = D / WARP_SIZE;   // 4
    constexpr int dim_idx = 1;               // index for D=128 codebook/matrices

    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int ic0      = blockIdx.x;
    const int head     = blockIdx.y % ne02;
    const int sequence = blockIdx.y / ne02;
    const int gqa_ratio = ne02 / ne12;

    // ------------------------------------------------------------------
    // Smem layout (654 floats = 2616 bytes):
    //   [q_rot:    D      floats]  reused as merged output in phase 3
    //   [wacc:     NWARPS*D floats]
    //   [kq_max:   NWARPS floats]
    //   [kq_sum:   NWARPS floats]
    //   [scale:    NWARPS floats]
    //   [gmax:     1      float ]
    //   [fsum:     1      float ]
    // ------------------------------------------------------------------
    extern __shared__ float smem[];
    float * const q_smem     = smem;
    float * const wacc_smem  = smem + D;
    float * const kq_max_arr = wacc_smem  + NWARPS * D;
    float * const kq_sum_arr = kq_max_arr + NWARPS;
    float * const scale_arr  = kq_sum_arr + NWARPS;
    float * const gmax_slot  = scale_arr  + NWARPS;
    float * const fsum_slot  = gmax_slot  + 1;

    // ------------------------------------------------------------------
    // Phase 1 : Q → smem, then q_rot[tid] = scale * Pi @ Q[tid]
    // ------------------------------------------------------------------
    const float * Q_row = (const float *)(Q_data
                        + (int64_t)nb03 * sequence
                        + (int64_t)nb02 * head
                        + (int64_t)nb01 * ic0);
    q_smem[tid] = Q_row[tid];
    __syncthreads();

    // Each thread tid computes one element of Pi @ Q.
    // d_Pi_T[j*D+tid] = Pi^T[j,tid] = Pi[tid,j]: reading j-stride accesses
    // is coalesced because for fixed j all 32 threads read consecutive addresses.
    float qr0=0.0f, qr1=0.0f, qr2=0.0f, qr3=0.0f;
    for (int j = 0; j < D; j += 4) {
        qr0 = fmaf(d_Pi_T[(j+0)*D + tid], q_smem[j+0], qr0);
        qr1 = fmaf(d_Pi_T[(j+1)*D + tid], q_smem[j+1], qr1);
        qr2 = fmaf(d_Pi_T[(j+2)*D + tid], q_smem[j+2], qr2);
        qr3 = fmaf(d_Pi_T[(j+3)*D + tid], q_smem[j+3], qr3);
    }
    __syncthreads();
    q_smem[tid] = (qr0 + qr1 + qr2 + qr3) * scale;
    __syncthreads();

    // Each thread loads 4 elements of q_rot (strided by WARP_SIZE=32)
    const float qp0 = q_smem[lane_id];
    const float qp1 = q_smem[lane_id + 32];
    const float qp2 = q_smem[lane_id + 64];
    const float qp3 = q_smem[lane_id + 96];

    // ------------------------------------------------------------------
    // Phase 2 : flash-attention hot loop (same dot product + softmax as
    //           the existing partial kernel, but over ALL K tokens).
    // ------------------------------------------------------------------
    const char * K_head = K_data + (int64_t)nb13*sequence + (int64_t)nb12*(head/gqa_ratio);
    const char * V_head = V_data + (int64_t)nb23*sequence + (int64_t)nb22*(head/gqa_ratio);
    const half * maskh  = mask ? (const half *)(mask + (int64_t)nb33*(sequence % ne32)
                                                      + (int64_t)nb31 * ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // 2-bit bit-extraction offsets (same layout as partial kernel)
    const int bs0 = (lane_id & 3) << 1;
    const int bi0 = lane_id >> 2;
    const int bi1 = bi0 + (D/4/4);    // = bi0 + 8
    const int bi2 = bi0 + (2*D/4/4);  // = bi0 + 16
    const int bi3 = bi0 + (3*D/4/4);  // = bi0 + 24

    float w_kq_max = -FLT_MAX / 2.0f;
    float w_kq_sum = 0.0f;
    float w_acc0 = 0.0f, w_acc1 = 0.0f, w_acc2 = 0.0f, w_acc3 = 0.0f;

    for (int k = warp_id; k < ne11; k += NWARPS) {
        const uint8_t * K_blk = (const uint8_t *)(K_head + (int64_t)k * nb11);
        const uint8_t * V_blk = (const uint8_t *)(V_head + (int64_t)k * nb21);

        // TQ_MSE dequant: codebook lookup for 4 elements per thread
        const float cb_k0 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi0] >> bs0) & 3];
        const float cb_k1 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi1] >> bs0) & 3];
        const float cb_k2 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi2] >> bs0) & 3];
        const float cb_k3 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi3] >> bs0) & 3];

        // Partial dot product then full-warp reduce (5 shuffles → broadcasts to all lanes)
        float partial = qp0*cb_k0 + qp1*cb_k1 + qp2*cb_k2 + qp3*cb_k3;
        for (int off = 16; off >= 1; off >>= 1)
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);

        // Online softmax (lane 0 only — result broadcast below)
        float a_w = 0.0f, s_old = 1.0f;
        if (lane_id == 0) {
            const float norm_k    = *((const float *) K_blk);
            float kq              = partial * norm_k;
            if (logit_softcap != 0.0f) kq = logit_softcap * tanhf(kq);
            if (maskh)                 kq += slope * __half2float(maskh[k]);
            const float kq_max_new = fmaxf(w_kq_max, kq + FATTN_KQ_MAX_OFFSET);
            s_old      = expf(w_kq_max - kq_max_new);
            a_w        = expf(kq - kq_max_new);
            w_kq_max   = kq_max_new;
            w_kq_sum   = w_kq_sum * s_old + a_w;
        }
        a_w   = __shfl_sync(0xFFFFFFFF, a_w,   0);
        s_old = __shfl_sync(0xFFFFFFFF, s_old, 0);

        // V accumulation (same 4-element stride pattern)
        const float norm_v = *((const float *) V_blk);
        const float aw_nv  = a_w * norm_v;
        const float cb_v0  = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi0] >> bs0) & 3];
        const float cb_v1  = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi1] >> bs0) & 3];
        const float cb_v2  = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi2] >> bs0) & 3];
        const float cb_v3  = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi3] >> bs0) & 3];
        w_acc0 = fmaf(aw_nv, cb_v0, w_acc0 * s_old);
        w_acc1 = fmaf(aw_nv, cb_v1, w_acc1 * s_old);
        w_acc2 = fmaf(aw_nv, cb_v2, w_acc2 * s_old);
        w_acc3 = fmaf(aw_nv, cb_v3, w_acc3 * s_old);
    }

    // ------------------------------------------------------------------
    // Phase 3 : merge warp accumulators → normalize → Pi^T unrotate → output
    // ------------------------------------------------------------------
    // Store per-warp acc to smem (stride-32 layout mirrors partial kernel)
    wacc_smem[warp_id*D + lane_id]      = w_acc0;
    wacc_smem[warp_id*D + lane_id + 32] = w_acc1;
    wacc_smem[warp_id*D + lane_id + 64] = w_acc2;
    wacc_smem[warp_id*D + lane_id + 96] = w_acc3;
    if (lane_id == 0) {
        kq_max_arr[warp_id] = w_kq_max;
        kq_sum_arr[warp_id] = w_kq_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float gmax = kq_max_arr[0];
        for (int w = 1; w < NWARPS; w++) gmax = fmaxf(gmax, kq_max_arr[w]);
        float fsum = 0.0f;
        for (int w = 0; w < NWARPS; w++) {
            const float sc = expf(kq_max_arr[w] - gmax);
            scale_arr[w] = sc;
            fsum += sc * kq_sum_arr[w];
        }
        *gmax_slot = gmax;
        *fsum_slot = fsum;
    }
    __syncthreads();

    const float inv_fsum = 1.0f / *fsum_slot;

    // Merge all warp accumulators (4-way ILP)
    float m0v=0.0f, m1v=0.0f, m2v=0.0f, m3v=0.0f;
    for (int w = 0; w < NWARPS; w++) {
        const float sc = scale_arr[w];
        m0v = fmaf(sc, wacc_smem[w*D + lane_id],      m0v);
        m1v = fmaf(sc, wacc_smem[w*D + lane_id + 32], m1v);
        m2v = fmaf(sc, wacc_smem[w*D + lane_id + 64], m2v);
        m3v = fmaf(sc, wacc_smem[w*D + lane_id + 96], m3v);
    }

    // Store normalized merged output back into q_smem[0..D-1] for unrotation
    q_smem[lane_id]      = m0v * inv_fsum;
    q_smem[lane_id + 32] = m1v * inv_fsum;
    q_smem[lane_id + 64] = m2v * inv_fsum;
    q_smem[lane_id + 96] = m3v * inv_fsum;
    __syncthreads();

    // Pi^T unrotation: out[tid] = sum_j Pi[j*D+tid] * q_smem[j]
    // = (Pi^T @ merged_output)[tid] — identical access pattern to Phase 1.
    float r0=0.0f, r1=0.0f, r2=0.0f, r3=0.0f;
    for (int j = 0; j < D; j += 4) {
        r0 = fmaf(d_Pi[(j+0)*D + tid], q_smem[j+0], r0);
        r1 = fmaf(d_Pi[(j+1)*D + tid], q_smem[j+1], r1);
        r2 = fmaf(d_Pi[(j+2)*D + tid], q_smem[j+2], r2);
        r3 = fmaf(d_Pi[(j+3)*D + tid], q_smem[j+3], r3);
    }

    const int64_t j_dst = ((int64_t)(sequence * ne01 + ic0) * ne02 + head) * D;
    ((float *)dst)[j_dst + tid] = r0 + r1 + r2 + r3;
}

void ggml_cuda_flash_attn_ext_tq_mse_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
