#pragma once
#include "common.cuh"
#include "fattn-common.cuh"
#include "turbo-quant.cuh"

// ---------------------------------------------------------------------------
// VEC-style TQ_MSE partial kernel — v2.
//
// Critical fixes vs first attempt:
//
//  (1) Phase A inner loop has NO inter-iteration dependency:
//      Removed the serial `KQ_max_new = fmaxf(KQ_max_new, kq)` chain from
//      inside the 32-step KQ loop.  Max is found in ONE 5-shuffle warp-reduce
//      AFTER the loop instead of 32 sequential fmaxf calls.  With no shared
//      mutable state in the inner loop the compiler can fully pipeline the
//      32 dot-product iterations.
//
//  (2) Batch softmax weight sum via warp-reduce (5 shuffles) instead of 32
//      sequential adds in the V loop.  Eliminates the w_kq_sum serial chain.
//
//  (3) a_w values stored to smem (warp-private region in q_rot area) before
//      the V loop.  Phase B reads a_w[i_v] with a constant smem offset (after
//      #pragma unroll), avoiding variable-lane __shfl_sync which prevents
//      instruction-level pipelining.
//
//  (4) Phase A fused Q rotation (same as v1) — eliminates separate rotate_q
//      kernel launch and q_pi_buf global allocation.
//
// Expected improvement vs old 3-kernel pipeline at d=16384:
//   Phase A: ~160 shuffle-cycles (throughput-limited) vs ~1056 (latency-limited)
//   Batch max: 25 cycles (5 shuffles) vs embedded in 32-step serial chain
//   Phase B: ~128 FMA-cycles with smem vs ~288 (serial variable-lane shuffles)
//   → roughly 3-5× less critical-path latency per K token
//
// Grid : (ne01, n_splits, ne02*ne03)
// Block: D = 128 threads (NWARPS=4 warps × WARP_SIZE=32 lanes)
// Smem : (D + NWARPS*D + 3*NWARPS + 2) floats = 654 floats = 2616 bytes
//        (q_rot area D floats is reused as warp-private a_w buffer in hot loop)
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_mse_vec_kernel(
        const char   * __restrict__ Q_data,
        const char   * __restrict__ K_data,
        const char   * __restrict__ V_data,
        const char   * __restrict__ mask,
        float        * __restrict__ dst_parts,
        float2       * __restrict__ dst_meta,
        const float  * __restrict__ d_Pi_T,
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
        const int64_t nb33,
        const int32_t n_splits
) {
    static_assert(D == 128, "fattn-tq-mse-vec: only D=128 is supported");
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

    // Smem layout: [q_rot: D | warp_acc: NWARPS*D | kq_max: NWARPS | kq_sum: NWARPS |
    //               scale: NWARPS | global_max: 1 | final_sum: 1]   = 654 floats
    // q_rot area (D floats) is REUSED as warp-private a_w buffer during hot loop.
    extern __shared__ float smem[];
    float * const q_rot_smem   = smem;           // [0..127] — reused as a_w buffer
    float * const warp_acc_smem = smem + D;       // [128..639]
    float * const kq_max_arr   = warp_acc_smem + NWARPS * D;
    float * const kq_sum_arr   = kq_max_arr + NWARPS;
    float * const scale_arr    = kq_sum_arr + NWARPS;
    float * const global_slot  = scale_arr  + NWARPS;
    float * const finsum_slot  = global_slot + 1;

    // ------------------------------------------------------------------
    // Phase 0: Q rotation fused inline — all D threads cooperate once.
    // After the sync, q_rot_smem[0..127] holds rotated Q temporarily.
    // ------------------------------------------------------------------
    const float * Q_row = (const float *)(Q_data
                        + (int64_t)nb03 * sequence
                        + (int64_t)nb02 * head
                        + (int64_t)nb01 * ic0);
    q_rot_smem[tid] = Q_row[tid];
    __syncthreads();

    float qr0=0.0f, qr1=0.0f, qr2=0.0f, qr3=0.0f;
    for (int j = 0; j < D; j += 4) {
        qr0 = fmaf(d_Pi_T[(j+0)*D + tid], q_rot_smem[j+0], qr0);
        qr1 = fmaf(d_Pi_T[(j+1)*D + tid], q_rot_smem[j+1], qr1);
        qr2 = fmaf(d_Pi_T[(j+2)*D + tid], q_rot_smem[j+2], qr2);
        qr3 = fmaf(d_Pi_T[(j+3)*D + tid], q_rot_smem[j+3], qr3);
    }
    __syncthreads();
    q_rot_smem[tid] = (qr0 + qr1 + qr2 + qr3) * scale;
    __syncthreads();

    // Load rotated Q into per-thread registers (stride-32 layout)
    const float qp0 = q_rot_smem[lane_id];
    const float qp1 = q_rot_smem[lane_id + 32];
    const float qp2 = q_rot_smem[lane_id + 64];
    const float qp3 = q_rot_smem[lane_id + 96];
    // q_rot_smem is now free; we'll reuse it as a_w scratch below

    // Per-thread bit-extraction offsets (same layout as old partial kernel)
    const int bs0 = (lane_id & 3) << 1;
    const int bi0 = lane_id >> 2;
    const int bi1 = bi0 + (D/4/4);    // = bi0 + 8
    const int bi2 = bi0 + (2*D/4/4);  // = bi0 + 16
    const int bi3 = bi0 + (3*D/4/4);  // = bi0 + 24

    const char * K_head = K_data + (int64_t)nb13*sequence + (int64_t)nb12*(head/gqa_ratio);
    const char * V_head = V_data + (int64_t)nb23*sequence + (int64_t)nb22*(head/gqa_ratio);
    const half * maskh  = mask ? (const half *)(mask + (int64_t)nb33*(sequence%ne32)
                                                      + (int64_t)nb31*ic0) : nullptr;
    const float slope   = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    float w_kq_max = -FLT_MAX / 2.0f;
    float w_kq_sum = 0.0f;
    float w_acc0=0.0f, w_acc1=0.0f, w_acc2=0.0f, w_acc3=0.0f;

    // Warp-private a_w buffer: use [warp_id*WARP_SIZE .. (warp_id+1)*WARP_SIZE)
    // of q_rot_smem. Each warp writes its own region; no cross-warp conflict.
    float * const warp_a_smem = q_rot_smem + warp_id * WARP_SIZE;

    // ------------------------------------------------------------------
    // Phase 1+2: VEC-style hot loop.
    // Warp w handles K tokens [k_start + w*WARP_SIZE, k_start + (w+NWARPS)*WARP_SIZE, ...]
    //
    // Phase A — KQ dot products, NO serial max chain:
    //   Each of 32 inner iterations computes ONE K token's full dot product
    //   (all 32 lanes contribute, warp_reduce broadcasts result).
    //   Thread lane_id==i_kq "owns" that KQ value (register KQ_reg).
    //   NO shared state mutated — compiler can pipeline iterations.
    //   After inner loop: ONE 5-shuffle warp-reduce finds batch max.
    //
    // Phase B — V accumulation via smem-cached weights (no variable-lane shuffle):
    //   a_w stored to warp_a_smem; reads are constant-offset after #pragma unroll.
    //   Softmax weight sum computed via warp-reduce, not loop accumulation.
    // ------------------------------------------------------------------
    for (int k_outer = k_start + warp_id*WARP_SIZE; k_outer < k_end; k_outer += NWARPS*WARP_SIZE) {
        const int batch_size = min(WARP_SIZE, k_end - k_outer);

        // ---- Phase A: KQ dot products, no serial dependency across iterations ----
        // KQ_reg starts at -inf so out-of-bounds threads contribute 0 to softmax
        float KQ_reg = -FLT_MAX / 2.0f;

        // Unrolled to WARP_SIZE; compiler sees constant i_kq per iteration →
        // fully pipelined memory loads and constant-lane SETP/SELP for ownership.
        #pragma unroll
        for (int i_kq = 0; i_kq < WARP_SIZE; i_kq++) {
            if (i_kq >= batch_size) break;
            const int k = k_outer + i_kq;
            const uint8_t * K_blk = (const uint8_t *)(K_head + (int64_t)k * nb11);

            const float cb_k0 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi0] >> bs0) & 3];
            const float cb_k1 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi1] >> bs0) & 3];
            const float cb_k2 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi2] >> bs0) & 3];
            const float cb_k3 = tq_d_cb_2bit[dim_idx][(K_blk[4 + bi3] >> bs0) & 3];

            float partial = qp0*cb_k0 + qp1*cb_k1 + qp2*cb_k2 + qp3*cb_k3;
            #pragma unroll
            for (int off = 16; off >= 1; off >>= 1)
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, off);
            // partial is broadcast to all 32 lanes by warp reduce

            const float norm_k = *((const float *)K_blk);
            float kq = partial * norm_k;
            if (logit_softcap != 0.0f) kq = logit_softcap * tanhf(kq);
            if (maskh) kq += slope * __half2float(maskh[k]);

            // Compile-time-constant predicate (SETP/SELP, no divergence)
            KQ_reg = (lane_id == i_kq) ? kq : KQ_reg;
            // NOTE: NO fmaxf update here — eliminates 32-step serial chain
        }

        // Warp-reduce to find batch max (5 shuffles, parallel across all 32 threads)
        float batch_max = KQ_reg;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1)
            batch_max = fmaxf(batch_max, __shfl_xor_sync(0xFFFFFFFF, batch_max, off));
        // batch_max = max(KQ_reg[0..31]) = max over all K tokens in this batch

        const float KQ_max_new = fmaxf(w_kq_max, batch_max + FATTN_KQ_MAX_OFFSET);
        const float s_old = expf(w_kq_max - KQ_max_new);  // all threads, parallel
        w_kq_max = KQ_max_new;

        // Rescale running V accumulator by exp(old_max - new_max)
        w_acc0 *= s_old; w_acc1 *= s_old; w_acc2 *= s_old; w_acc3 *= s_old;
        w_kq_sum *= s_old;

        // ONE expf per thread for its owned KQ weight (parallel across all 32 threads)
        const float a_w = expf(KQ_reg - KQ_max_new);

        // Batch softmax sum via warp-reduce (5 shuffles, replaces 32 serial adds)
        float batch_sum = a_w;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1)
            batch_sum += __shfl_xor_sync(0xFFFFFFFF, batch_sum, off);
        w_kq_sum += batch_sum;

        // ---- Phase B: V accumulation — smem-cached a_w, constant-offset reads ----
        // Store a_w to warp-private smem region; all writes go to distinct addresses.
        warp_a_smem[lane_id] = a_w;
        __syncwarp();

        // With #pragma unroll, i_v is compile-time → warp_a_smem[i_v] is constant
        // offset (no variable-lane shuffle needed, compiler can issue all loads early).
        #pragma unroll
        for (int i_v = 0; i_v < WARP_SIZE; i_v++) {
            if (i_v >= batch_size) break;
            const int k = k_outer + i_v;
            const float a_w_k = warp_a_smem[i_v];  // constant-offset smem read

            const uint8_t * V_blk = (const uint8_t *)(V_head + (int64_t)k * nb21);
            const float norm_v = *((const float *)V_blk);
            const float aw_nv  = a_w_k * norm_v;
            const float cb_v0 = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi0] >> bs0) & 3];
            const float cb_v1 = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi1] >> bs0) & 3];
            const float cb_v2 = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi2] >> bs0) & 3];
            const float cb_v3 = tq_d_cb_2bit[dim_idx][(V_blk[4 + bi3] >> bs0) & 3];
            w_acc0 += aw_nv * cb_v0;
            w_acc1 += aw_nv * cb_v1;
            w_acc2 += aw_nv * cb_v2;
            w_acc3 += aw_nv * cb_v3;
        }
    }

    // ------------------------------------------------------------------
    // Phase 3: merge 4 warp accumulators → store partial result + meta
    // (identical to flash_attn_tq_mse_partial_kernel epilog)
    // ------------------------------------------------------------------
    warp_acc_smem[warp_id*D + lane_id]      = w_acc0;
    warp_acc_smem[warp_id*D + lane_id + 32] = w_acc1;
    warp_acc_smem[warp_id*D + lane_id + 64] = w_acc2;
    warp_acc_smem[warp_id*D + lane_id + 96] = w_acc3;
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

    float out0=0.0f, out1=0.0f, out2=0.0f, out3=0.0f;
    for (int w = 0; w < NWARPS; w++) {
        const float sc = scale_arr[w];
        out0 = fmaf(sc, warp_acc_smem[w*D + lane_id],      out0);
        out1 = fmaf(sc, warp_acc_smem[w*D + lane_id + 32], out1);
        out2 = fmaf(sc, warp_acc_smem[w*D + lane_id + 64], out2);
        out3 = fmaf(sc, warp_acc_smem[w*D + lane_id + 96], out3);
    }

    const int j_dst = (sequence*ne01 + ic0)*ne02 + head;
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

void ggml_cuda_flash_attn_ext_tq_mse_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
