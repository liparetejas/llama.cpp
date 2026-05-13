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

    // Codebook into scalar registers — 2 SEL instructions per lookup, no memory in hot loop
    const float cb_r0 = tq_d_cb_2bit[dim_idx][0];
    const float cb_r1 = tq_d_cb_2bit[dim_idx][1];
    const float cb_r2 = tq_d_cb_2bit[dim_idx][2];
    const float cb_r3 = tq_d_cb_2bit[dim_idx][3];
    #define CB_SEL_P(idx) (((idx) & 2) ? (((idx) & 1) ? cb_r3 : cb_r2) : (((idx) & 1) ? cb_r1 : cb_r0))

    // Per-warp registers: running softmax state + 4-element accumulator
    float w_kq_max = -FLT_MAX / 2.0f;
    float w_kq_sum = 0.0f;
    float w_acc0 = 0.0f, w_acc1 = 0.0f, w_acc2 = 0.0f, w_acc3 = 0.0f;

    // Hot loop: NO __syncthreads; only warp-level shuffles
    for (int k = k_start + warp_id; k < k_end; k += NWARPS) {
        const uint8_t * K_blk = (const uint8_t *)(K_head + (int64_t)k * nb11);
        const uint8_t * V_blk = (const uint8_t *)(V_head + (int64_t)k * nb21);

        // Decode 2-bit indices for 4 elements per thread using register select
        const float cb_k0 = CB_SEL_P((K_blk[4 + bi0] >> bs0) & 3);
        const float cb_k1 = CB_SEL_P((K_blk[4 + bi1] >> bs0) & 3);
        const float cb_k2 = CB_SEL_P((K_blk[4 + bi2] >> bs0) & 3);
        const float cb_k3 = CB_SEL_P((K_blk[4 + bi3] >> bs0) & 3);

        // Dot product: 4 partial contributions, then warp reduce
        float partial = qp0*cb_k0 + qp1*cb_k1 + qp2*cb_k2 + qp3*cb_k3;
        #pragma unroll
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

        // V accumulation for 4 elements using register select
        const float norm_v = *((const float *) V_blk);
        const float aw_nv  = a_w * norm_v;
        const float cb_v0 = CB_SEL_P((V_blk[4 + bi0] >> bs0) & 3);
        const float cb_v1 = CB_SEL_P((V_blk[4 + bi1] >> bs0) & 3);
        const float cb_v2 = CB_SEL_P((V_blk[4 + bi2] >> bs0) & 3);
        const float cb_v3 = CB_SEL_P((V_blk[4 + bi3] >> bs0) & 3);
        w_acc0 = fmaf(aw_nv, cb_v0, w_acc0 * s_old);
        w_acc1 = fmaf(aw_nv, cb_v1, w_acc1 * s_old);
        w_acc2 = fmaf(aw_nv, cb_v2, w_acc2 * s_old);
        w_acc3 = fmaf(aw_nv, cb_v3, w_acc3 * s_old);
    }
    #undef CB_SEL_P

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
// Kernel 2b: Batched-softmax split-K partial attention.
//
// Same Q/K/V layout as flash_attn_tq_mse_partial_kernel. Key difference:
//   OLD: per K token → 2 expf at lane 0 + 2 shfl_sync broadcasts = 64 expf/32 tokens
//   NEW: per BATCH K tokens → 1 s_old + BATCH a_w at lane 0 = 33 expf/32 tokens (1.94×)
//        a_w values written to smem, __syncwarp(), all threads read for V accumulation.
//        Eliminates the 2 shfl_sync broadcasts per token entirely.
//
// Extra smem over partial_kernel: NWARPS*(BATCH+1) floats = 132 floats = 528 bytes.
// Total smem: (NWARPS*D + 3*NWARPS + 2 + NWARPS*(BATCH+1)) * sizeof(float) = 2616 bytes.
// ---------------------------------------------------------------------------
template<int D, int BATCH>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_mse_batch_partial_kernel(
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
    static_assert(D == 128,      "flash_attn_tq_mse_batch: only D=128 supported");
    static_assert(BATCH >= 1,    "BATCH must be >= 1");
    constexpr int NWARPS = D / WARP_SIZE;  // 4
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

    const int bs0 = (lane_id & 3) << 1;
    const int bi0 = lane_id >> 2;
    const int bi1 = bi0 + (D/4/4);
    const int bi2 = bi0 + (2*D/4/4);
    const int bi3 = bi0 + (3*D/4/4);

    // Smem layout:
    // [warp_acc_rot: NWARPS*D | kq_max_arr: NWARPS | kq_sum_arr: NWARPS |
    //  scale_arr: NWARPS | global_slot: 1 | finsum_slot: 1 |
    //  aw_smem: NWARPS*BATCH | sold_smem: NWARPS]
    extern __shared__ float smem[];
    float * const warp_acc_smem = smem;
    float * const kq_max_arr    = smem + NWARPS * D;
    float * const kq_sum_arr    = kq_max_arr + NWARPS;
    float * const scale_arr     = kq_sum_arr + NWARPS;
    float * const global_slot   = scale_arr  + NWARPS;
    float * const finsum_slot   = global_slot + 1;
    float * const aw_smem       = finsum_slot + 1;           // NWARPS*BATCH floats
    float * const sold_smem     = aw_smem + NWARPS * BATCH;  // NWARPS floats

    // Load 4-entry codebook into scalar registers — eliminates global/L1 lookup in hot loop.
    // Uses 2 SEL (predicated register select) instructions per lookup instead of a load.
    const float cb_r0 = tq_d_cb_2bit[dim_idx][0];
    const float cb_r1 = tq_d_cb_2bit[dim_idx][1];
    const float cb_r2 = tq_d_cb_2bit[dim_idx][2];
    const float cb_r3 = tq_d_cb_2bit[dim_idx][3];

    // 2-way predicated select (2 cycles, no memory): compiles to 2 FSEL instructions.
    #define CB_SEL(idx) (((idx) & 2) ? (((idx) & 1) ? cb_r3 : cb_r2) : (((idx) & 1) ? cb_r1 : cb_r0))

    float w_kq_max = -FLT_MAX / 2.0f;
    float w_kq_sum = 0.0f;
    float w_acc0 = 0.0f, w_acc1 = 0.0f, w_acc2 = 0.0f, w_acc3 = 0.0f;

    // Outer loop: each warp starts at k_start + warp_id*BATCH, strides by NWARPS*BATCH.
    // Inner loops use BATCH as compile-time bound (enables #pragma unroll).
    // Edge case (last batch with fewer than BATCH tokens) uses predication via early break.
    for (int k_batch = k_start + warp_id * BATCH; k_batch < k_end; k_batch += NWARPS * BATCH) {
        // -- Phase A: compute BATCH dot products (all 32 threads participate) ---------
        // KQ_buf written only by lane 0; predicated dead at other lanes by compiler.
        float KQ_buf[BATCH];
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const int k = k_batch + b;
            if (k >= k_end) break;
            const uint8_t * K_blk = (const uint8_t *)(K_head + (int64_t)k * nb11);
            const float cb_k0 = CB_SEL((K_blk[4 + bi0] >> bs0) & 3);
            const float cb_k1 = CB_SEL((K_blk[4 + bi1] >> bs0) & 3);
            const float cb_k2 = CB_SEL((K_blk[4 + bi2] >> bs0) & 3);
            const float cb_k3 = CB_SEL((K_blk[4 + bi3] >> bs0) & 3);
            float partial = qp0*cb_k0 + qp1*cb_k1 + qp2*cb_k2 + qp3*cb_k3;
            #pragma unroll
            for (int off = 16; off >= 1; off >>= 1)
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
            if (lane_id == 0) {
                const float norm_k = *((const float *) K_blk);
                float kq = partial * norm_k;
                if (logit_softcap != 0.0f) kq = logit_softcap * tanhf(kq);
                if (maskh) kq += slope * __half2float(maskh[k]);
                KQ_buf[b] = kq;
            }
        }

        // -- Phase B: lane 0 only — batch softmax, write a_w & s_old to smem ----------
        if (lane_id == 0) {
            const int batch_size = min(BATCH, k_end - k_batch);
            float batch_max = KQ_buf[0];
            #pragma unroll
            for (int b = 1; b < BATCH; b++) {
                if (b < batch_size) batch_max = fmaxf(batch_max, KQ_buf[b]);
            }
            const float kq_max_new = fmaxf(w_kq_max, batch_max + FATTN_KQ_MAX_OFFSET);
            const float s_old = expf(w_kq_max - kq_max_new);
            w_kq_max = kq_max_new;
            sold_smem[warp_id] = s_old;
            float batch_sum = 0.0f;
            #pragma unroll
            for (int b = 0; b < BATCH; b++) {
                if (b < batch_size) {
                    const float a_w = expf(KQ_buf[b] - kq_max_new);
                    aw_smem[warp_id * BATCH + b] = a_w;
                    batch_sum += a_w;
                }
            }
            w_kq_sum = w_kq_sum * s_old + batch_sum;
        }
        __syncwarp();

        // -- Phase C: all threads read s_old & a_w, accumulate V ---------------------
        const float s_old_val = sold_smem[warp_id];
        w_acc0 *= s_old_val;
        w_acc1 *= s_old_val;
        w_acc2 *= s_old_val;
        w_acc3 *= s_old_val;
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const int k = k_batch + b;
            if (k >= k_end) break;
            const uint8_t * V_blk = (const uint8_t *)(V_head + (int64_t)k * nb21);
            const float a_w   = aw_smem[warp_id * BATCH + b];
            const float norm_v = *((const float *) V_blk);
            const float aw_nv  = a_w * norm_v;
            const float cb_v0 = CB_SEL((V_blk[4 + bi0] >> bs0) & 3);
            const float cb_v1 = CB_SEL((V_blk[4 + bi1] >> bs0) & 3);
            const float cb_v2 = CB_SEL((V_blk[4 + bi2] >> bs0) & 3);
            const float cb_v3 = CB_SEL((V_blk[4 + bi3] >> bs0) & 3);
            w_acc0 += aw_nv * cb_v0;
            w_acc1 += aw_nv * cb_v1;
            w_acc2 += aw_nv * cb_v2;
            w_acc3 += aw_nv * cb_v3;
        }
    }

    #undef CB_SEL

    // -- Merge phase (identical to flash_attn_tq_mse_partial_kernel) -----------------
    warp_acc_smem[warp_id * D + lane_id]      = w_acc0;
    warp_acc_smem[warp_id * D + lane_id + 32] = w_acc1;
    warp_acc_smem[warp_id * D + lane_id + 64] = w_acc2;
    warp_acc_smem[warp_id * D + lane_id + 96] = w_acc3;
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
        *global_slot = gmax;
        *finsum_slot = fsum;
    }
    __syncthreads();

    const float final_sum = *finsum_slot;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    for (int w = 0; w < NWARPS; w++) {
        const float sc = scale_arr[w];
        out0 = fmaf(sc, warp_acc_smem[w * D + lane_id],      out0);
        out1 = fmaf(sc, warp_acc_smem[w * D + lane_id + 32], out1);
        out2 = fmaf(sc, warp_acc_smem[w * D + lane_id + 64], out2);
        out3 = fmaf(sc, warp_acc_smem[w * D + lane_id + 96], out3);
    }

    const int j_dst = (sequence * ne01 + ic0) * ne02 + head;
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
// Kernel 2c: All-thread softmax — eliminates smem Phase B entirely.
//
// Key insight: after the 5-step shfl_xor warp-reduce, ALL 32 threads hold the
// same full dot product `partial`. Since norm_k/maskh are broadcast loads (same
// address for all threads), every thread can independently compute kq, s_old,
// a_w, and update w_kq_max/w_kq_sum with zero inter-thread communication.
//
// Eliminates vs BATCH kernel:
//   - sold_smem writes/reads  (NWARPS floats per outer iter)
//   - aw_smem writes/reads    (NWARPS*BATCH floats per outer iter)
//   - __syncwarp per outer iter
// Cost: 2 expf per token on all 32 threads — but SFU processes all 32 threads
// in the same wall-clock time as 1 thread (8 SFU units × 4 passes = same cycles).
// Net: strictly fewer ops in the hot loop while expf cost is unchanged.
//
// #pragma unroll 4 lets the compiler overlap K loads of token k+1 with the
// shfl chain of token k, hiding global memory latency.
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_mse_allthread_kernel(
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
    static_assert(D == 128, "flash_attn_tq_mse_allthread: only D=128 supported");
    constexpr int NWARPS  = D / WARP_SIZE;  // 4
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

    const int bs0 = (lane_id & 3) << 1;
    const int bi0 = lane_id >> 2;
    const int bi1 = bi0 + (D/4/4);
    const int bi2 = bi0 + (2*D/4/4);
    const int bi3 = bi0 + (3*D/4/4);

    // Smem layout (same as OLD partial kernel — no aw/sold smem needed):
    // [warp_acc_rot: NWARPS*D | kq_max_arr: NWARPS | kq_sum_arr: NWARPS |
    //  scale_arr: NWARPS | global_slot: 1 | finsum_slot: 1]
    extern __shared__ float smem[];
    float * const warp_acc_smem = smem;
    float * const kq_max_arr    = smem + NWARPS * D;
    float * const kq_sum_arr    = kq_max_arr + NWARPS;
    float * const scale_arr     = kq_sum_arr + NWARPS;
    float * const global_slot   = scale_arr  + NWARPS;
    float * const finsum_slot   = global_slot + 1;

    // Load 4-entry codebook into scalar registers
    const float cb_r0 = tq_d_cb_2bit[dim_idx][0];
    const float cb_r1 = tq_d_cb_2bit[dim_idx][1];
    const float cb_r2 = tq_d_cb_2bit[dim_idx][2];
    const float cb_r3 = tq_d_cb_2bit[dim_idx][3];
    #define CB_SEL_AT(idx) (((idx) & 2) ? (((idx) & 1) ? cb_r3 : cb_r2) : (((idx) & 1) ? cb_r1 : cb_r0))

    // All threads maintain identical w_kq_max, w_kq_sum (no sync needed).
    float w_kq_max = -FLT_MAX / 2.0f;
    float w_kq_sum = 0.0f;
    float w_acc0 = 0.0f, w_acc1 = 0.0f, w_acc2 = 0.0f, w_acc3 = 0.0f;

    for (int k = k_start + warp_id; k < k_end; k += NWARPS) {
        const uint8_t * K_blk = (const uint8_t *)(K_head + (int64_t)k * nb11);

        // Dot product (all threads, 5-step shfl_xor reduce)
        const float cb_k0 = CB_SEL_AT((K_blk[4 + bi0] >> bs0) & 3);
        const float cb_k1 = CB_SEL_AT((K_blk[4 + bi1] >> bs0) & 3);
        const float cb_k2 = CB_SEL_AT((K_blk[4 + bi2] >> bs0) & 3);
        const float cb_k3 = CB_SEL_AT((K_blk[4 + bi3] >> bs0) & 3);
        float partial = qp0*cb_k0 + qp1*cb_k1 + qp2*cb_k2 + qp3*cb_k3;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1)
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);

        // All threads compute kq — norm_k/maskh are broadcast loads (1 L1 txn for all 32)
        const float norm_k = *((const float *) K_blk);
        float kq = partial * norm_k;
        if (logit_softcap != 0.0f) kq = logit_softcap * tanhf(kq);
        if (maskh) kq += slope * __half2float(maskh[k]);

        // All threads update running softmax state (identical across all lanes, no sync needed)
        const float kq_max_new = fmaxf(w_kq_max, kq + FATTN_KQ_MAX_OFFSET);
        const float s_old = expf(w_kq_max - kq_max_new);
        const float a_w   = expf(kq        - kq_max_new);
        w_kq_max = kq_max_new;
        w_kq_sum = w_kq_sum * s_old + a_w;

        // Scale existing accumulator and accumulate V (per-thread dims diverge here)
        w_acc0 *= s_old;
        w_acc1 *= s_old;
        w_acc2 *= s_old;
        w_acc3 *= s_old;

        const uint8_t * V_blk = (const uint8_t *)(V_head + (int64_t)k * nb21);
        const float norm_v = *((const float *) V_blk);
        const float aw_nv  = a_w * norm_v;
        const float cb_v0  = CB_SEL_AT((V_blk[4 + bi0] >> bs0) & 3);
        const float cb_v1  = CB_SEL_AT((V_blk[4 + bi1] >> bs0) & 3);
        const float cb_v2  = CB_SEL_AT((V_blk[4 + bi2] >> bs0) & 3);
        const float cb_v3  = CB_SEL_AT((V_blk[4 + bi3] >> bs0) & 3);
        w_acc0 = fmaf(aw_nv, cb_v0, w_acc0);
        w_acc1 = fmaf(aw_nv, cb_v1, w_acc1);
        w_acc2 = fmaf(aw_nv, cb_v2, w_acc2);
        w_acc3 = fmaf(aw_nv, cb_v3, w_acc3);
    }
    #undef CB_SEL_AT

    // -- Merge phase (same as other partial kernels) --------------------------
    warp_acc_smem[warp_id * D + lane_id]      = w_acc0;
    warp_acc_smem[warp_id * D + lane_id + 32] = w_acc1;
    warp_acc_smem[warp_id * D + lane_id + 64] = w_acc2;
    warp_acc_smem[warp_id * D + lane_id + 96] = w_acc3;
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
        *global_slot = gmax;
        *finsum_slot = fsum;
    }
    __syncthreads();

    const float final_sum = *finsum_slot;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    for (int w = 0; w < NWARPS; w++) {
        const float sc = scale_arr[w];
        out0 = fmaf(sc, warp_acc_smem[w * D + lane_id],      out0);
        out1 = fmaf(sc, warp_acc_smem[w * D + lane_id + 32], out1);
        out2 = fmaf(sc, warp_acc_smem[w * D + lane_id + 64], out2);
        out3 = fmaf(sc, warp_acc_smem[w * D + lane_id + 96], out3);
    }

    const int j_dst = (sequence * ne01 + ic0) * ne02 + head;
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
// Kernel 2d: Half-warp attention — 4-step warp-reduce instead of 5-step.
//
// Key insight: 5 shfl_xor steps (off=16,8,4,2,1) are needed to reduce 32 threads.
// If we split the 128-thread block into 8 "half-warps" of 16 threads, each
// handling a different K token, the reduction only needs 4 steps (off=8,4,2,1).
// shfl_xor with off<=8 never crosses a 16-thread boundary → both half-warps
// of a 32-thread warp stay fully isolated. off=16 is eliminated.
//
// Trade-off: 8 elements/thread (stride 16) instead of 4 (stride 32).
//   +  1 fewer shfl step per token → ~12% fewer serial latency cycles
//   +  ALL-THREAD softmax (no smem/syncwarp for aw/sold)
//   -  8 FMA per token (vs 4) — hidden behind shfl chain latency
//   -  8 K/V byte loads per thread (vs 4) — ~same L1 traffic total
//   -  8 acc registers (vs 4) — higher register pressure
//
// Per-token: 8 FMA + 4 shfl = 64 cycle latency vs 4 FMA + 5 shfl = 80 cycles.
// Theoretical: 20% speedup in the shfl-dominated inner loop.
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_mse_halfwarp_kernel(
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
    static_assert(D == 128, "flash_attn_tq_mse_halfwarp: only D=128 supported");
    constexpr int NHALF_WARPS = D / 16;  // 8
    constexpr int dim_idx = 1;

    const int tid      = threadIdx.x;
    const int hw_id    = tid / 16;   // 0..7 — which half-warp
    const int hw_lane  = tid % 16;   // 0..15 — position within half-warp
    const int warp_id  = tid / 32;   // 0..3 — for final dst_parts write
    const int lane_id  = tid % 32;   // 0..31 — for final dst_parts write
    const int ic0      = blockIdx.x;
    const int split    = blockIdx.y;
    const int head     = blockIdx.z % ne02;
    const int sequence = blockIdx.z / ne02;
    const int gqa_ratio = ne02 / ne12;

    const int tokens_per_split = (ne11 + n_splits - 1) / n_splits;
    const int k_start = split * tokens_per_split;
    const int k_end   = min(k_start + tokens_per_split, ne11);

    const int64_t q_idx = ((int64_t)(sequence * ne01 + ic0) * ne02 + head) * D;
    // Each thread holds 8 Q dims at stride 16 (hw_lane, hw_lane+16, ..., hw_lane+112)
    const float qp0 = q_pi_buf[q_idx + hw_lane];
    const float qp1 = q_pi_buf[q_idx + hw_lane + 16];
    const float qp2 = q_pi_buf[q_idx + hw_lane + 32];
    const float qp3 = q_pi_buf[q_idx + hw_lane + 48];
    const float qp4 = q_pi_buf[q_idx + hw_lane + 64];
    const float qp5 = q_pi_buf[q_idx + hw_lane + 80];
    const float qp6 = q_pi_buf[q_idx + hw_lane + 96];
    const float qp7 = q_pi_buf[q_idx + hw_lane + 112];

    const char * K_head = K_data + (int64_t)nb13*sequence + (int64_t)nb12*(head/gqa_ratio);
    const char * V_head = V_data + (int64_t)nb23*sequence + (int64_t)nb22*(head/gqa_ratio);
    const half * maskh  = mask ? (const half *)(mask + (int64_t)nb33*(sequence%ne32)
                                                      + (int64_t)nb31*ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // K/V index extraction with stride-16 layout:
    // thread hw_lane accesses dims hw_lane + {0,16,...,112} = byte (hw_lane+k*16)/4 + 4,
    // bit-shift (hw_lane % 4)*2 (same for all k since hw_lane+k*16 ≡ hw_lane mod 4).
    const int bs  = (hw_lane & 3) << 1;   // bit shift: 0,2,4,6
    const int bb  = hw_lane >> 2;          // byte_base: 0..3
    const int bi0 = bb;                    // byte offset for dim hw_lane
    const int bi1 = bb + 4;               // byte offset for dim hw_lane+16
    const int bi2 = bb + 8;
    const int bi3 = bb + 12;
    const int bi4 = bb + 16;
    const int bi5 = bb + 20;
    const int bi6 = bb + 24;
    const int bi7 = bb + 28;

    // Smem layout (NHALF_WARPS=8 groups):
    // [hw_acc_rot: NHALF_WARPS*D | kq_max_arr: NHALF_WARPS | kq_sum_arr: NHALF_WARPS |
    //  scale_arr: NHALF_WARPS | global_slot: 1 | finsum_slot: 1]
    extern __shared__ float smem[];
    float * const hw_acc_smem = smem;
    float * const kq_max_arr  = smem + NHALF_WARPS * D;
    float * const kq_sum_arr  = kq_max_arr + NHALF_WARPS;
    float * const scale_arr   = kq_sum_arr + NHALF_WARPS;
    float * const global_slot = scale_arr  + NHALF_WARPS;
    float * const finsum_slot = global_slot + 1;

    const float cb_r0 = tq_d_cb_2bit[dim_idx][0];
    const float cb_r1 = tq_d_cb_2bit[dim_idx][1];
    const float cb_r2 = tq_d_cb_2bit[dim_idx][2];
    const float cb_r3 = tq_d_cb_2bit[dim_idx][3];
    #define CB_SEL_HW(idx) (((idx) & 2) ? (((idx) & 1) ? cb_r3 : cb_r2) : (((idx) & 1) ? cb_r1 : cb_r0))

    // All half-warp threads independently maintain the running softmax state.
    // Since partial is identical across all 16 hw_lane threads after the 4-step reduce,
    // and norm_k/maskh are broadcast loads, kq is identical → no sync needed.
    float w_kq_max = -FLT_MAX / 2.0f;
    float w_kq_sum = 0.0f;
    float w_acc0 = 0.0f, w_acc1 = 0.0f, w_acc2 = 0.0f, w_acc3 = 0.0f;
    float w_acc4 = 0.0f, w_acc5 = 0.0f, w_acc6 = 0.0f, w_acc7 = 0.0f;

    for (int k = k_start + hw_id; k < k_end; k += NHALF_WARPS) {
        const uint8_t * K_blk = (const uint8_t *)(K_head + (int64_t)k * nb11);

        // 8-term dot product + 4-step shfl_xor (off=8,4,2,1 stays within 16-thread half-warp)
        const float cb_k0 = CB_SEL_HW((K_blk[4 + bi0] >> bs) & 3);
        const float cb_k1 = CB_SEL_HW((K_blk[4 + bi1] >> bs) & 3);
        const float cb_k2 = CB_SEL_HW((K_blk[4 + bi2] >> bs) & 3);
        const float cb_k3 = CB_SEL_HW((K_blk[4 + bi3] >> bs) & 3);
        const float cb_k4 = CB_SEL_HW((K_blk[4 + bi4] >> bs) & 3);
        const float cb_k5 = CB_SEL_HW((K_blk[4 + bi5] >> bs) & 3);
        const float cb_k6 = CB_SEL_HW((K_blk[4 + bi6] >> bs) & 3);
        const float cb_k7 = CB_SEL_HW((K_blk[4 + bi7] >> bs) & 3);
        float partial = qp0*cb_k0 + qp1*cb_k1 + qp2*cb_k2 + qp3*cb_k3 +
                        qp4*cb_k4 + qp5*cb_k5 + qp6*cb_k6 + qp7*cb_k7;
        #pragma unroll
        for (int off = 8; off >= 1; off >>= 1)
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);

        // Broadcast loads — all 16 threads load the same address, 1 L1 transaction each
        const float norm_k = *((const float *) K_blk);
        float kq = partial * norm_k;
        if (logit_softcap != 0.0f) kq = logit_softcap * tanhf(kq);
        if (maskh) kq += slope * __half2float(maskh[k]);

        const float kq_max_new = fmaxf(w_kq_max, kq + FATTN_KQ_MAX_OFFSET);
        const float s_old = expf(w_kq_max - kq_max_new);
        const float a_w   = expf(kq        - kq_max_new);
        w_kq_max = kq_max_new;
        w_kq_sum = w_kq_sum * s_old + a_w;

        w_acc0 *= s_old; w_acc1 *= s_old; w_acc2 *= s_old; w_acc3 *= s_old;
        w_acc4 *= s_old; w_acc5 *= s_old; w_acc6 *= s_old; w_acc7 *= s_old;

        const uint8_t * V_blk = (const uint8_t *)(V_head + (int64_t)k * nb21);
        const float norm_v = *((const float *) V_blk);
        const float aw_nv  = a_w * norm_v;
        const float cb_v0  = CB_SEL_HW((V_blk[4 + bi0] >> bs) & 3);
        const float cb_v1  = CB_SEL_HW((V_blk[4 + bi1] >> bs) & 3);
        const float cb_v2  = CB_SEL_HW((V_blk[4 + bi2] >> bs) & 3);
        const float cb_v3  = CB_SEL_HW((V_blk[4 + bi3] >> bs) & 3);
        const float cb_v4  = CB_SEL_HW((V_blk[4 + bi4] >> bs) & 3);
        const float cb_v5  = CB_SEL_HW((V_blk[4 + bi5] >> bs) & 3);
        const float cb_v6  = CB_SEL_HW((V_blk[4 + bi6] >> bs) & 3);
        const float cb_v7  = CB_SEL_HW((V_blk[4 + bi7] >> bs) & 3);
        w_acc0 = fmaf(aw_nv, cb_v0, w_acc0);
        w_acc1 = fmaf(aw_nv, cb_v1, w_acc1);
        w_acc2 = fmaf(aw_nv, cb_v2, w_acc2);
        w_acc3 = fmaf(aw_nv, cb_v3, w_acc3);
        w_acc4 = fmaf(aw_nv, cb_v4, w_acc4);
        w_acc5 = fmaf(aw_nv, cb_v5, w_acc5);
        w_acc6 = fmaf(aw_nv, cb_v6, w_acc6);
        w_acc7 = fmaf(aw_nv, cb_v7, w_acc7);
    }
    #undef CB_SEL_HW

    // -- Merge: write half-warp acc to smem, combine across 8 groups --------------
    // Thread hw_lane of hw_id writes dims hw_lane+{0,16,...,112}.
    hw_acc_smem[hw_id * D + hw_lane]       = w_acc0;
    hw_acc_smem[hw_id * D + hw_lane + 16]  = w_acc1;
    hw_acc_smem[hw_id * D + hw_lane + 32]  = w_acc2;
    hw_acc_smem[hw_id * D + hw_lane + 48]  = w_acc3;
    hw_acc_smem[hw_id * D + hw_lane + 64]  = w_acc4;
    hw_acc_smem[hw_id * D + hw_lane + 80]  = w_acc5;
    hw_acc_smem[hw_id * D + hw_lane + 96]  = w_acc6;
    hw_acc_smem[hw_id * D + hw_lane + 112] = w_acc7;
    if (hw_lane == 0) {
        kq_max_arr[hw_id] = w_kq_max;
        kq_sum_arr[hw_id] = w_kq_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float gmax = kq_max_arr[0];
        for (int hw = 1; hw < NHALF_WARPS; hw++) gmax = fmaxf(gmax, kq_max_arr[hw]);
        float fsum = 0.0f;
        for (int hw = 0; hw < NHALF_WARPS; hw++) {
            const float sc = expf(kq_max_arr[hw] - gmax);
            scale_arr[hw] = sc;
            fsum += sc * kq_sum_arr[hw];
        }
        *global_slot = gmax;
        *finsum_slot = fsum;
    }
    __syncthreads();

    // Combine: thread lane_id accumulates all 8 half-warps for its 4 output dims
    const float final_sum = *finsum_slot;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    for (int hw = 0; hw < NHALF_WARPS; hw++) {
        const float sc = scale_arr[hw];
        out0 = fmaf(sc, hw_acc_smem[hw * D + lane_id],      out0);
        out1 = fmaf(sc, hw_acc_smem[hw * D + lane_id + 32], out1);
        out2 = fmaf(sc, hw_acc_smem[hw * D + lane_id + 64], out2);
        out3 = fmaf(sc, hw_acc_smem[hw * D + lane_id + 96], out3);
    }

    const int j_dst = (sequence * ne01 + ic0) * ne02 + head;
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

    // 4-way unrolled accumulation (requires parallel_blocks % 4 == 0)
    float acc_t = 0.0f;
    if (parallel_blocks % 4 == 0) {
        float a0=0,a1=0,a2=0,a3=0;
        for (int l = 0; l < parallel_blocks; l += 4) {
            const float sc0 = expf(meta[l+0].x - kqmax);
            const float sc1 = expf(meta[l+1].x - kqmax);
            const float sc2 = expf(meta[l+2].x - kqmax);
            const float sc3 = expf(meta[l+3].x - kqmax);
            a0 = fmaf(sc0, VKQ_parts[(int64_t)(l+0)*D+tid], a0);
            a1 = fmaf(sc1, VKQ_parts[(int64_t)(l+1)*D+tid], a1);
            a2 = fmaf(sc2, VKQ_parts[(int64_t)(l+2)*D+tid], a2);
            a3 = fmaf(sc3, VKQ_parts[(int64_t)(l+3)*D+tid], a3);
        }
        acc_t = a0+a1+a2+a3;
    } else {
        for (int l = 0; l < parallel_blocks; l++) {
            acc_t = fmaf(expf(meta[l].x - kqmax), VKQ_parts[(int64_t)l*D+tid], acc_t);
        }
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
