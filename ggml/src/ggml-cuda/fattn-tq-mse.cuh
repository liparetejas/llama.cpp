#pragma once
#include "common.cuh"
#include "fattn-common.cuh"
#include "turbo-quant.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: rotate Q once per (query, head, seq).
// Grid: (ne01, ne02*ne03), Block: (D).
// Output q_pi_out[(seq*ne01+ic0)*ne02+head, D] = scale * Pi @ Q_row.
// Pi_T[j*D+tid] = Pi[tid*D+j]: coalesced read across warp.
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D)
__global__ void tq_mse_rotate_q_kernel(
        const char  * __restrict__ Q_data,
        float       * __restrict__ q_pi_out,
        const float * __restrict__ d_Pi_T,
        const float  scale,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03
) {
    const int tid      = threadIdx.x;
    const int ic0      = blockIdx.x;
    const int head     = blockIdx.y % ne02;
    const int sequence = blockIdx.y / ne02;

    const float * Q_row = (const float *)(Q_data + (int64_t)nb03*sequence
                                                  + (int64_t)nb02*head
                                                  + (int64_t)nb01*ic0);
    extern __shared__ float smem[];  // D floats for Q broadcast
    smem[tid] = Q_row[tid];
    __syncthreads();

    float qr0=0,qr1=0,qr2=0,qr3=0;
    for (int j = 0; j < D; j += 4) {
        qr0 = fmaf(d_Pi_T[(j+0)*D+tid], smem[j+0], qr0);
        qr1 = fmaf(d_Pi_T[(j+1)*D+tid], smem[j+1], qr1);
        qr2 = fmaf(d_Pi_T[(j+2)*D+tid], smem[j+2], qr2);
        qr3 = fmaf(d_Pi_T[(j+3)*D+tid], smem[j+3], qr3);
    }

    const int64_t out_idx = ((int64_t)(sequence * ne01 + ic0) * ne02 + head) * D;
    q_pi_out[out_idx + tid] = (qr0 + qr1 + qr2 + qr3) * scale;
}

// ---------------------------------------------------------------------------
// Kernel 2: Split-K partial attention — warp-parallel K tokens.
// Grid: (ne01, n_splits, ne02*ne03), Block: (D = 128 = 4 warps × 32 lanes).
//
// Design: each warp independently processes every 4th K token (stride=NWARPS).
// No __syncthreads in the hot loop — only 2 syncs total per block to merge
// warp-level accumulators at the end.
//
// Smem layout (≈2104 bytes per block, 4 blocks/SM → 8416 bytes < 64 KB):
//   [warp_acc_rot: NWARPS*D floats] [kq_max: NWARPS] [kq_sum: NWARPS]
//   [scale: NWARPS] [global_max: 1] [final_sum: 1]
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_mse_partial_kernel(
        const float  * __restrict__ q_pi_buf,
        const char   * __restrict__ K_data,
        const char   * __restrict__ V_data,
        const char   * __restrict__ mask,
        float        * __restrict__ dst_parts,
        float2       * __restrict__ dst_meta,
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
        const int32_t nb11,
        const int32_t nb12,
        const int64_t nb13,
        const int32_t nb21,
        const int32_t nb22,
        const int64_t nb23,
        const int32_t ne32,
        const int32_t nb31,
        const int64_t nb33,
        const int32_t n_splits
) {
    static_assert(D == 128, "fattn-tq-mse: only D=128 is supported");
    static_assert(D % WARP_SIZE == 0, "D must be divisible by WARP_SIZE");
    constexpr int NWARPS = D / WARP_SIZE;  // 4 warps per block
    constexpr int dim_idx = 1;

    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int ic0      = blockIdx.x;
    const int split    = blockIdx.y;
    const int head     = blockIdx.z % ne02;
    const int sequence = blockIdx.z / ne02;
    const int gqa_ratio = ne02 / ne12;

    const int tokens_per_split = (ne11 + n_splits - 1) / n_splits;
    const int k_start = split * tokens_per_split;
    const int k_end   = min(k_start + tokens_per_split, ne11);

    // Each thread is responsible for 4 elements of D=128 (stride WARP_SIZE=32):
    //   elem0=lane_id, elem1=lane_id+32, elem2=lane_id+64, elem3=lane_id+96
    const int64_t q_idx = ((int64_t)(sequence * ne01 + ic0) * ne02 + head) * D;
    const float qp0 = q_pi_buf[q_idx + lane_id];
    const float qp1 = q_pi_buf[q_idx + lane_id + 32];
    const float qp2 = q_pi_buf[q_idx + lane_id + 64];
    const float qp3 = q_pi_buf[q_idx + lane_id + 96];

    const char * K_head = K_data + (int64_t)nb13*sequence + (int64_t)nb12*(head/gqa_ratio);
    const char * V_head = V_data + (int64_t)nb23*sequence + (int64_t)nb22*(head/gqa_ratio);
    const half * maskh  = mask ? (const half *)(mask + (int64_t)nb33*(sequence%ne32)
                                                      + (int64_t)nb31*ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // Per-element bit extraction offsets for the 4 elements per thread
    const int bs0 = (lane_id & 3) << 1;       // bit_shift for elem0,1,2,3 (same shift)
    const int bi0 = lane_id >> 2;              // byte_idx for elem0
    const int bi1 = bi0 + (D/4/4);            // byte_idx for elem1  (= bi0 + 8)
    const int bi2 = bi0 + (2*D/4/4);          // byte_idx for elem2  (= bi0 + 16)
    const int bi3 = bi0 + (3*D/4/4);          // byte_idx for elem3  (= bi0 + 24)

    // Smem layout: [warp_acc_rot: NWARPS*D | kq_max_arr: NWARPS | kq_sum_arr: NWARPS |
    //               scale_arr: NWARPS | global_max_slot: 1 | final_sum_slot: 1]
    extern __shared__ float smem[];
    float * const warp_acc_smem = smem;                      // NWARPS*D floats
    float * const kq_max_arr    = smem + NWARPS * D;         // NWARPS floats
    float * const kq_sum_arr    = kq_max_arr + NWARPS;       // NWARPS floats
    float * const scale_arr     = kq_sum_arr + NWARPS;       // NWARPS floats
    float * const global_slot   = scale_arr + NWARPS;        // 1 float
    float * const finsum_slot   = global_slot + 1;           // 1 float

    // Per-warp registers: running softmax state + 4-element accumulator
    float w_kq_max = -FLT_MAX / 2.0f;
    float w_kq_sum = 0.0f;
    float w_acc0 = 0.0f, w_acc1 = 0.0f, w_acc2 = 0.0f, w_acc3 = 0.0f;

    // Hot loop: NO __syncthreads; only warp-level shuffles
    for (int k = k_start + warp_id; k < k_end; k += NWARPS) {
        const uint8_t * K_blk = (const uint8_t *)(K_head + (int64_t)k * nb11);
        const uint8_t * V_blk = (const uint8_t *)(V_head + (int64_t)k * nb21);

        // Decode 2-bit indices for 4 elements per thread
        const float cb_k0 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi0] >> bs0) & 3];
        const float cb_k1 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi1] >> bs0) & 3];
        const float cb_k2 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi2] >> bs0) & 3];
        const float cb_k3 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi3] >> bs0) & 3];

        // Dot product: 4 partial contributions, then warp reduce
        float partial = qp0*cb_k0 + qp1*cb_k1 + qp2*cb_k2 + qp3*cb_k3;
        for (int off = 16; off >= 1; off >>= 1) {
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
        }

        // Lane 0: compute kq and run online softmax
        float a_w = 0.0f, s_old = 1.0f;
        if (lane_id == 0) {
            const float norm_k = *((const float *) K_blk);
            float kq = partial * norm_k;
            if (logit_softcap != 0.0f) kq = logit_softcap * tanhf(kq);
            if (maskh) kq += slope * __half2float(maskh[k]);
            const float kq_max_new = fmaxf(w_kq_max, kq + FATTN_KQ_MAX_OFFSET);
            s_old      = expf(w_kq_max - kq_max_new);
            a_w        = expf(kq       - kq_max_new);
            w_kq_max   = kq_max_new;
            w_kq_sum   = w_kq_sum * s_old + a_w;
        }
        // Broadcast a_w and s_old from lane 0 to all lanes (no syncthreads!)
        a_w   = __shfl_sync(0xFFFFFFFF, a_w,   0);
        s_old = __shfl_sync(0xFFFFFFFF, s_old, 0);

        // V accumulation for 4 elements
        const float norm_v = *((const float *) V_blk);
        const float aw_nv  = a_w * norm_v;
        const float cb_v0 = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi0] >> bs0) & 3];
        const float cb_v1 = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi1] >> bs0) & 3];
        const float cb_v2 = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi2] >> bs0) & 3];
        const float cb_v3 = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi3] >> bs0) & 3];
        w_acc0 = fmaf(aw_nv, cb_v0, w_acc0 * s_old);
        w_acc1 = fmaf(aw_nv, cb_v1, w_acc1 * s_old);
        w_acc2 = fmaf(aw_nv, cb_v2, w_acc2 * s_old);
        w_acc3 = fmaf(aw_nv, cb_v3, w_acc3 * s_old);
    }

    // Store each warp's acc_rot to smem — stride-32 layout for the merge step
    warp_acc_smem[warp_id * D + lane_id]      = w_acc0;
    warp_acc_smem[warp_id * D + lane_id + 32] = w_acc1;
    warp_acc_smem[warp_id * D + lane_id + 64] = w_acc2;
    warp_acc_smem[warp_id * D + lane_id + 96] = w_acc3;
    if (lane_id == 0) {
        kq_max_arr[warp_id] = w_kq_max;
        kq_sum_arr[warp_id] = w_kq_sum;
    }
    __syncthreads();

    // Thread 0: find global max, compute per-warp rescale factors and final sum
    if (tid == 0) {
        float gmax = kq_max_arr[0];
        for (int w = 1; w < NWARPS; w++) gmax = fmaxf(gmax, kq_max_arr[w]);
        float fsum = 0.0f;
        for (int w = 0; w < NWARPS; w++) {
            const float sc = expf(kq_max_arr[w] - gmax);
            scale_arr[w] = sc;
            fsum += sc * kq_sum_arr[w];
        }
        *global_slot = gmax;
        *finsum_slot = fsum;
    }
    __syncthreads();

    const float final_sum = *finsum_slot;

    // Merge: combine acc_rots from all 4 warps weighted by scale_arr
    // Each thread handles its 4 elements (indices lane_id, +32, +64, +96)
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    for (int w = 0; w < NWARPS; w++) {
        const float sc = scale_arr[w];
        out0 = fmaf(sc, warp_acc_smem[w * D + lane_id],      out0);
        out1 = fmaf(sc, warp_acc_smem[w * D + lane_id + 32], out1);
        out2 = fmaf(sc, warp_acc_smem[w * D + lane_id + 64], out2);
        out3 = fmaf(sc, warp_acc_smem[w * D + lane_id + 96], out3);
    }

    // Store raw acc_rot (Pi-rotated domain); unrotation deferred to combine kernel.
    const int j_dst = ((sequence * ne01 + ic0) * ne02 + head);
    const int64_t base = ((int64_t)j_dst * n_splits + split) * D;
    if (warp_id == 0) {
        dst_parts[base + lane_id]      = out0;
        dst_parts[base + lane_id + 32] = out1;
        dst_parts[base + lane_id + 64] = out2;
        dst_parts[base + lane_id + 96] = out3;
    }
    if (tid == 0) {
        dst_meta[(int64_t)j_dst * n_splits + split] = make_float2(*global_slot, final_sum);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: combine splits + single Pi^T unrotation.
// Grid: (ne01, ne02, ne03), Block: (D).
// Smem: [meta: 2*parallel_blocks floats | acc_combined: D floats | denom: 1 float]
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_mse_combine(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float        * __restrict__ dst,
        const float  * __restrict__ d_Pi,
        const int parallel_blocks
) {
    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;
    const int j_dst    = (sequence * ne01 + col) * ne02 + head;
    const int tid      = threadIdx.x;

    VKQ_parts += (int64_t)j_dst * parallel_blocks * D;
    VKQ_meta  += (int64_t)j_dst * parallel_blocks;
    dst       += (int64_t)j_dst * D;

    // smem: [meta: 2*parallel_blocks | acc_combined: D | denom_slot: 1]
    extern __shared__ float smem[];
    float2 * meta     = (float2 *) smem;
    float  * acc_smem = smem + 2 * parallel_blocks;
    // acc_smem[D] is the denom slot

    for (int i = tid; i < 2 * parallel_blocks; i += D) {
        ((float *)meta)[i] = ((const float *)VKQ_meta)[i];
    }
    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; l++) kqmax = fmaxf(kqmax, meta[l].x);

    float acc_t = 0.0f;
    for (int l = 0; l < parallel_blocks; l++) {
        const float sc = expf(meta[l].x - kqmax);
        acc_t = fmaf(sc, VKQ_parts[(int64_t)l * D + tid], acc_t);
    }
    acc_smem[tid] = acc_t;

    if (tid == 0) {
        float denom = 0.0f;
        for (int l = 0; l < parallel_blocks; l++) denom += expf(meta[l].x - kqmax) * meta[l].y;
        acc_smem[D] = denom;
    }
    __syncthreads();
    const float inv_denom = 1.0f / acc_smem[D];

    // 4-way unrolled Pi^T matvec: 4 independent FMA chains for ILP=4
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    for (int j = 0; j < D; j += 4) {
        out0 = fmaf(d_Pi[(j+0) * D + tid], acc_smem[j+0], out0);
        out1 = fmaf(d_Pi[(j+1) * D + tid], acc_smem[j+1], out1);
        out2 = fmaf(d_Pi[(j+2) * D + tid], acc_smem[j+2], out2);
        out3 = fmaf(d_Pi[(j+3) * D + tid], acc_smem[j+3], out3);
    }

    dst[tid] = (out0 + out1 + out2 + out3) * inv_denom;
}

void ggml_cuda_flash_attn_ext_tq_mse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
