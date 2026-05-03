#pragma once
#include "common.cuh"
#include "fattn-common.cuh"
#include "turbo-quant.cuh"

// ---------------------------------------------------------------------------
// TQ_PROD block layout (D=128, 56 bytes):
//   [0..3]   float norm
//   [4..7]   float gamma
//   [8..39]  128 × 2-bit MSE indices  (4 per byte, offset 8)
//   [40..55] 128 × 1-bit QJL signs   (8 per byte, offset 40)
//
// Two-pass design:
//   Pass 1: tq_prod_rotate_q_kernel  →  q_pi_buf, q_s_buf  [n_queries, D]
//   Pass 2: flash_attn_tq_prod_partial_kernel  (stores raw [acc_rot|acc_sqjl])
//   Pass 3: flash_attn_tq_prod_combine  (combine + Pi^T/S^T unrotation once)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Kernel 1: rotate Q once per (query, head, seq).
// Grid: (ne01, ne02*ne03), Block: (D).
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D)
__global__ void tq_prod_rotate_q_kernel(
        const char  * __restrict__ Q_data,
        float       * __restrict__ q_pi_out,
        float       * __restrict__ q_s_out,
        const float * __restrict__ d_Pi_T,
        const float * __restrict__ d_S_T,
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
    extern __shared__ float smem[];
    smem[tid] = Q_row[tid];
    __syncthreads();

    float qr0=0,qr1=0,qr2=0,qr3=0;
    float qs0=0,qs1=0,qs2=0,qs3=0;
    for (int j = 0; j < D; j += 4) {
        qr0 = fmaf(d_Pi_T[(j+0)*D+tid], smem[j+0], qr0);
        qr1 = fmaf(d_Pi_T[(j+1)*D+tid], smem[j+1], qr1);
        qr2 = fmaf(d_Pi_T[(j+2)*D+tid], smem[j+2], qr2);
        qr3 = fmaf(d_Pi_T[(j+3)*D+tid], smem[j+3], qr3);
        qs0 = fmaf(d_S_T [(j+0)*D+tid], smem[j+0], qs0);
        qs1 = fmaf(d_S_T [(j+1)*D+tid], smem[j+1], qs1);
        qs2 = fmaf(d_S_T [(j+2)*D+tid], smem[j+2], qs2);
        qs3 = fmaf(d_S_T [(j+3)*D+tid], smem[j+3], qs3);
    }

    const int64_t out_idx = ((int64_t)(sequence * ne01 + ic0) * ne02 + head) * D;
    q_pi_out[out_idx + tid] = (qr0+qr1+qr2+qr3) * scale;
    q_s_out [out_idx + tid] = (qs0+qs1+qs2+qs3) * scale;
}

// ---------------------------------------------------------------------------
// Kernel 2: split-K partial attention — warp-parallel K tokens.
// Grid: (ne01, n_splits, ne02*ne03), Block: (D = 128 = 4 warps × 32 lanes).
//
// Each warp independently processes every 4th K token (stride=NWARPS).
// No __syncthreads in the hot loop — only 2 syncs total per block.
//
// TQ_PROD block layout (56 bytes):
//   [0..3]  float norm,  [4..7]  float gamma
//   [8..39] 128×2-bit MSE indices, [40..55] 128×1-bit QJL signs
//
// Each thread handles 4 elements (stride WARP_SIZE=32).
// MSE: bs = (lane_id&3)<<1, bi0=lane_id>>2, bi1=bi0+8, bi2=bi0+16, bi3=bi0+24
// QJL: qs = lane_id&7,     qi0=lane_id>>3, qi1=qi0+4, qi2=qi0+8,  qi3=qi0+12
//
// Smem (≈4104 bytes per block, 4 blocks/SM → 16416B < 64KB):
//   [warp_acc_rot:  NWARPS*D | warp_acc_sqjl: NWARPS*D |
//    kq_max:NWARPS | kq_sum:NWARPS | scale:NWARPS | global_max:1 | final_sum:1]
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_prod_partial_kernel(
        const float  * __restrict__ q_pi_buf,
        const float  * __restrict__ q_s_buf,
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
    static_assert(D == 128, "fattn-tq-prod: only D=128 is supported");
    static_assert(D % WARP_SIZE == 0, "D must be divisible by WARP_SIZE");
    constexpr int NWARPS    = D / WARP_SIZE;   // 4
    constexpr int dim_idx   = 1;
    constexpr int qjl_start = 8 + D / 4;       // 8 + 32 = 40

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

    // Each thread covers 4 elements: {lane_id, lane_id+32, lane_id+64, lane_id+96}
    const int64_t q_idx = ((int64_t)(sequence * ne01 + ic0) * ne02 + head) * D;
    const float qp0 = q_pi_buf[q_idx + lane_id];
    const float qp1 = q_pi_buf[q_idx + lane_id + 32];
    const float qp2 = q_pi_buf[q_idx + lane_id + 64];
    const float qp3 = q_pi_buf[q_idx + lane_id + 96];
    const float qs0 = q_s_buf [q_idx + lane_id];
    const float qs1 = q_s_buf [q_idx + lane_id + 32];
    const float qs2 = q_s_buf [q_idx + lane_id + 64];
    const float qs3 = q_s_buf [q_idx + lane_id + 96];

    const char * K_head = K_data + (int64_t)nb13*sequence + (int64_t)nb12*(head/gqa_ratio);
    const char * V_head = V_data + (int64_t)nb23*sequence + (int64_t)nb22*(head/gqa_ratio);
    const half * maskh  = mask ? (const half *)(mask + (int64_t)nb33*(sequence%ne32)
                                                      + (int64_t)nb31*ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // MSE bit-extraction offsets (2-bit, 4 per byte, starting at byte 8)
    const int bs0 = (lane_id & 3) << 1;    // bit shift (same for all 4 elements)
    const int bi0 = lane_id >> 2;           // byte idx for elem0 (0..7)
    const int bi1 = bi0 + (D/4/4);         // byte idx for elem1 (= bi0 + 8)
    const int bi2 = bi0 + (2*D/4/4);       // byte idx for elem2 (= bi0 + 16)
    const int bi3 = bi0 + (3*D/4/4);       // byte idx for elem3 (= bi0 + 24)

    // QJL bit-extraction offsets (1-bit, 8 per byte, starting at byte 40)
    const int qs_shift = lane_id & 7;       // bit shift (same for all 4 elements)
    const int qi0 = lane_id >> 3;           // byte idx for elem0 (0..3)
    const int qi1 = qi0 + (D/8/4);         // byte idx for elem1 (= qi0 + 4)
    const int qi2 = qi0 + (2*D/8/4);       // byte idx for elem2 (= qi0 + 8)
    const int qi3 = qi0 + (3*D/8/4);       // byte idx for elem3 (= qi0 + 12)

    // Smem layout
    extern __shared__ float smem[];
    float * const warp_rot_smem  = smem;                        // NWARPS*D
    float * const warp_sqjl_smem = smem + NWARPS * D;           // NWARPS*D
    float * const kq_max_arr     = smem + 2 * NWARPS * D;       // NWARPS
    float * const kq_sum_arr     = kq_max_arr + NWARPS;         // NWARPS
    float * const scale_arr      = kq_sum_arr + NWARPS;         // NWARPS
    float * const global_slot    = scale_arr + NWARPS;          // 1
    float * const finsum_slot    = global_slot + 1;             // 1

    const float qjl_coeff = sqrtf(3.14159265f / 2.0f) / (float)D;

    // Per-warp registers
    float w_kq_max = -FLT_MAX / 2.0f;
    float w_kq_sum = 0.0f;
    float wr0 = 0.0f, wr1 = 0.0f, wr2 = 0.0f, wr3 = 0.0f;  // acc_rot
    float wq0 = 0.0f, wq1 = 0.0f, wq2 = 0.0f, wq3 = 0.0f;  // acc_sqjl

    // Hot loop — NO __syncthreads; only warp-level shuffles
    for (int k = k_start + warp_id; k < k_end; k += NWARPS) {
        const uint8_t * K_blk = (const uint8_t *)(K_head + (int64_t)k * nb11);
        const uint8_t * V_blk = (const uint8_t *)(V_head + (int64_t)k * nb21);

        // KQ MSE: decode 4 cb_k values and partial dot product
        const float cb_k0 = tq_d_cb_2bit[dim_idx][(K_blk[8 + bi0] >> bs0) & 3];
        const float cb_k1 = tq_d_cb_2bit[dim_idx][(K_blk[8 + bi1] >> bs0) & 3];
        const float cb_k2 = tq_d_cb_2bit[dim_idx][(K_blk[8 + bi2] >> bs0) & 3];
        const float cb_k3 = tq_d_cb_2bit[dim_idx][(K_blk[8 + bi3] >> bs0) & 3];
        float pm = qp0*cb_k0 + qp1*cb_k1 + qp2*cb_k2 + qp3*cb_k3;
        for (int off = 16; off >= 1; off >>= 1) pm += __shfl_xor_sync(0xFFFFFFFF, pm, off);

        // KQ QJL: decode 4 sign values and partial dot product
        const float sk0 = ((K_blk[qjl_start + qi0] >> qs_shift) & 1) ? 1.0f : -1.0f;
        const float sk1 = ((K_blk[qjl_start + qi1] >> qs_shift) & 1) ? 1.0f : -1.0f;
        const float sk2 = ((K_blk[qjl_start + qi2] >> qs_shift) & 1) ? 1.0f : -1.0f;
        const float sk3 = ((K_blk[qjl_start + qi3] >> qs_shift) & 1) ? 1.0f : -1.0f;
        float pq = qs0*sk0 + qs1*sk1 + qs2*sk2 + qs3*sk3;
        for (int off = 16; off >= 1; off >>= 1) pq += __shfl_xor_sync(0xFFFFFFFF, pq, off);

        // Lane 0: combine, apply softmax
        float a_w = 0.0f, s_old = 1.0f;
        if (lane_id == 0) {
            const float norm_k  = *((const float *) K_blk);
            const float gamma_k = *((const float *)(K_blk + 4));
            float kq = norm_k * (pm + qjl_coeff * gamma_k * pq);
            if (logit_softcap != 0.0f) kq = logit_softcap * tanhf(kq);
            if (maskh) kq += slope * __half2float(maskh[k]);
            const float kq_max_new = fmaxf(w_kq_max, kq + FATTN_KQ_MAX_OFFSET);
            s_old      = expf(w_kq_max - kq_max_new);
            a_w        = expf(kq       - kq_max_new);
            w_kq_max   = kq_max_new;
            w_kq_sum   = w_kq_sum * s_old + a_w;
        }
        // Broadcast from lane 0 (no syncthreads!)
        a_w   = __shfl_sync(0xFFFFFFFF, a_w,   0);
        s_old = __shfl_sync(0xFFFFFFFF, s_old, 0);

        // V accumulation for acc_rot (MSE) and acc_sqjl (QJL)
        const float norm_v  = *((const float *) V_blk);
        const float gamma_v = *((const float *)(V_blk + 4));
        const float aw_nv   = a_w * norm_v;
        const float aw_nvg  = aw_nv * gamma_v;

        const float cb_v0 = tq_d_cb_2bit[dim_idx][(V_blk[8 + bi0] >> bs0) & 3];
        const float cb_v1 = tq_d_cb_2bit[dim_idx][(V_blk[8 + bi1] >> bs0) & 3];
        const float cb_v2 = tq_d_cb_2bit[dim_idx][(V_blk[8 + bi2] >> bs0) & 3];
        const float cb_v3 = tq_d_cb_2bit[dim_idx][(V_blk[8 + bi3] >> bs0) & 3];
        wr0 = fmaf(aw_nv, cb_v0, wr0 * s_old);
        wr1 = fmaf(aw_nv, cb_v1, wr1 * s_old);
        wr2 = fmaf(aw_nv, cb_v2, wr2 * s_old);
        wr3 = fmaf(aw_nv, cb_v3, wr3 * s_old);

        const float sv0 = ((V_blk[qjl_start + qi0] >> qs_shift) & 1) ? 1.0f : -1.0f;
        const float sv1 = ((V_blk[qjl_start + qi1] >> qs_shift) & 1) ? 1.0f : -1.0f;
        const float sv2 = ((V_blk[qjl_start + qi2] >> qs_shift) & 1) ? 1.0f : -1.0f;
        const float sv3 = ((V_blk[qjl_start + qi3] >> qs_shift) & 1) ? 1.0f : -1.0f;
        wq0 = fmaf(aw_nvg, sv0, wq0 * s_old);
        wq1 = fmaf(aw_nvg, sv1, wq1 * s_old);
        wq2 = fmaf(aw_nvg, sv2, wq2 * s_old);
        wq3 = fmaf(aw_nvg, sv3, wq3 * s_old);
    }

    // Store warp accumulators to smem (stride-32 layout)
    warp_rot_smem [warp_id * D + lane_id]      = wr0;
    warp_rot_smem [warp_id * D + lane_id + 32] = wr1;
    warp_rot_smem [warp_id * D + lane_id + 64] = wr2;
    warp_rot_smem [warp_id * D + lane_id + 96] = wr3;
    warp_sqjl_smem[warp_id * D + lane_id]      = wq0;
    warp_sqjl_smem[warp_id * D + lane_id + 32] = wq1;
    warp_sqjl_smem[warp_id * D + lane_id + 64] = wq2;
    warp_sqjl_smem[warp_id * D + lane_id + 96] = wq3;
    if (lane_id == 0) {
        kq_max_arr[warp_id] = w_kq_max;
        kq_sum_arr[warp_id] = w_kq_sum;
    }
    __syncthreads();

    // Thread 0: compute global max, per-warp scales, and final sum
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

    // Merge warp accumulators; only warp_id==0 writes output
    float out_r0 = 0.0f, out_r1 = 0.0f, out_r2 = 0.0f, out_r3 = 0.0f;
    float out_q0 = 0.0f, out_q1 = 0.0f, out_q2 = 0.0f, out_q3 = 0.0f;
    for (int w = 0; w < NWARPS; w++) {
        const float sc = scale_arr[w];
        out_r0 = fmaf(sc, warp_rot_smem [w * D + lane_id],      out_r0);
        out_r1 = fmaf(sc, warp_rot_smem [w * D + lane_id + 32], out_r1);
        out_r2 = fmaf(sc, warp_rot_smem [w * D + lane_id + 64], out_r2);
        out_r3 = fmaf(sc, warp_rot_smem [w * D + lane_id + 96], out_r3);
        out_q0 = fmaf(sc, warp_sqjl_smem[w * D + lane_id],      out_q0);
        out_q1 = fmaf(sc, warp_sqjl_smem[w * D + lane_id + 32], out_q1);
        out_q2 = fmaf(sc, warp_sqjl_smem[w * D + lane_id + 64], out_q2);
        out_q3 = fmaf(sc, warp_sqjl_smem[w * D + lane_id + 96], out_q3);
    }

    // Store raw [acc_rot | acc_sqjl]; unrotation deferred to combine kernel.
    const int j_dst = ((sequence * ne01 + ic0) * ne02 + head);
    const int64_t base = ((int64_t)j_dst * n_splits + split) * 2 * D;
    if (warp_id == 0) {
        dst_parts[base + lane_id]           = out_r0;
        dst_parts[base + lane_id + 32]      = out_r1;
        dst_parts[base + lane_id + 64]      = out_r2;
        dst_parts[base + lane_id + 96]      = out_r3;
        dst_parts[base + D + lane_id]       = out_q0;
        dst_parts[base + D + lane_id + 32]  = out_q1;
        dst_parts[base + D + lane_id + 64]  = out_q2;
        dst_parts[base + D + lane_id + 96]  = out_q3;
    }
    if (tid == 0) {
        dst_meta[(int64_t)j_dst * n_splits + split] = make_float2(*global_slot, final_sum);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: combine splits + single Pi^T/S^T unrotation.
// Grid: (ne01, ne02, ne03), Block: (D).
// Smem: [meta: 2*parallel_blocks | acc_rot: D | acc_sqjl: D | denom: 1]
// ---------------------------------------------------------------------------
template<int D>
__launch_bounds__(D, 4)
__global__ void flash_attn_tq_prod_combine(
        const float  * __restrict__ VKQ_parts,  // [n_q * n_splits * 2*D]
        const float2 * __restrict__ VKQ_meta,   // [n_q * n_splits]
        float        * __restrict__ dst,         // [n_q * D]
        const float  * __restrict__ d_Pi,        // Pi[D][D] row-major
        const float  * __restrict__ d_S,         // S[D][D] row-major
        const int parallel_blocks
) {
    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;
    const int j_dst    = (sequence * ne01 + col) * ne02 + head;
    const int tid      = threadIdx.x;

    VKQ_parts += (int64_t)j_dst * parallel_blocks * 2 * D;
    VKQ_meta  += (int64_t)j_dst * parallel_blocks;
    dst       += (int64_t)j_dst * D;

    // smem: [meta: 2*parallel_blocks | acc_rot: D | acc_sqjl: D | denom_slot: 1]
    extern __shared__ float smem[];
    float2 * meta       = (float2 *) smem;
    float  * rot_smem   = smem + 2 * parallel_blocks;
    float  * sqjl_smem  = rot_smem + D;
    // sqjl_smem[D] is the denom slot

    // Load meta
    for (int i = tid; i < 2 * parallel_blocks; i += D) {
        ((float *)meta)[i] = ((const float *)VKQ_meta)[i];
    }
    __syncthreads();

    // Global max
    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; l++) {
        kqmax = fmaxf(kqmax, meta[l].x);
    }

    // Combine acc_rot and acc_sqjl across splits
    float rot_t = 0.0f, sqjl_t = 0.0f;
    for (int l = 0; l < parallel_blocks; l++) {
        const float sc = expf(meta[l].x - kqmax);
        const int64_t base = (int64_t)l * 2 * D;
        rot_t  = fmaf(sc, VKQ_parts[base + tid],     rot_t);
        sqjl_t = fmaf(sc, VKQ_parts[base + D + tid], sqjl_t);
    }
    rot_smem [tid] = rot_t;
    sqjl_smem[tid] = sqjl_t;

    // Denominator on thread 0
    if (tid == 0) {
        float denom = 0.0f;
        for (int l = 0; l < parallel_blocks; l++) {
            denom += expf(meta[l].x - kqmax) * meta[l].y;
        }
        sqjl_smem[D] = denom;
    }
    __syncthreads();
    const float inv_denom = 1.0f / sqjl_smem[D];

    const float qjl_coeff = sqrtf(3.14159265f / 2.0f) / (float)D;

    // Pi^T and S^T matvecs: 4-way unrolled for ILP=4 on both chains
    float mse0=0,mse1=0,mse2=0,mse3=0;
    float qjl0=0,qjl1=0,qjl2=0,qjl3=0;
    for (int j = 0; j < D; j += 4) {
        mse0 = fmaf(d_Pi[(j+0)*D+tid], rot_smem [j+0], mse0);
        mse1 = fmaf(d_Pi[(j+1)*D+tid], rot_smem [j+1], mse1);
        mse2 = fmaf(d_Pi[(j+2)*D+tid], rot_smem [j+2], mse2);
        mse3 = fmaf(d_Pi[(j+3)*D+tid], rot_smem [j+3], mse3);
        qjl0 = fmaf(d_S [(j+0)*D+tid], sqjl_smem[j+0], qjl0);
        qjl1 = fmaf(d_S [(j+1)*D+tid], sqjl_smem[j+1], qjl1);
        qjl2 = fmaf(d_S [(j+2)*D+tid], sqjl_smem[j+2], qjl2);
        qjl3 = fmaf(d_S [(j+3)*D+tid], sqjl_smem[j+3], qjl3);
    }

    dst[tid] = ((mse0+mse1+mse2+mse3) + qjl_coeff*(qjl0+qjl1+qjl2+qjl3)) * inv_denom;
}

void ggml_cuda_flash_attn_ext_tq_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
