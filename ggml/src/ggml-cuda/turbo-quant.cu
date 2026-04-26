// Define the constant memory here; suppress extern decl in the header.
#define TQ_DEFINING_TU
#include "turbo-quant.cuh"
#undef TQ_DEFINING_TU

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// Minimal CUDA error check that does not depend on ggml internals.
// When compiled as part of the ggml-cuda library, common.cuh's CUDA_CHECK
// will take precedence via the translation units that include it; this
// fallback is only active when turbo-quant.cu is compiled standalone.
#ifndef CUDA_CHECK
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t _e = (err);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   \
                    #err, __FILE__, __LINE__, cudaGetErrorString(_e));         \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#endif

// ---------------------------------------------------------------------------
// Constant memory definitions (single definition; other TUs use extern decl)
// ---------------------------------------------------------------------------

__constant__ float tq_d_cb_2bit[TURBO_QUANT_NUM_DIMS][4];
__constant__ float tq_d_cb_1bit[TURBO_QUANT_NUM_DIMS][2];

// ---------------------------------------------------------------------------
// Device pointer storage
// ---------------------------------------------------------------------------

float * tq_d_Pi[TURBO_QUANT_NUM_DIMS] = {nullptr, nullptr, nullptr};
float * tq_d_S [TURBO_QUANT_NUM_DIMS] = {nullptr, nullptr, nullptr};

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

static const int tq_host_dims[TURBO_QUANT_NUM_DIMS] = {64, 128, 256};

static std::once_flag tq_cuda_init_flag;

static void _tq_cuda_do_init() {
    // 1. Ensure host-side matrices are ready.
    tq_init_rotations();

    // 2. Upload codebooks to CUDA constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(tq_d_cb_2bit,
                                  tq_codebook_2bit,
                                  sizeof(tq_codebook_2bit)));
    CUDA_CHECK(cudaMemcpyToSymbol(tq_d_cb_1bit,
                                  tq_codebook_1bit,
                                  sizeof(tq_codebook_1bit)));

    // 3. Allocate and upload Pi and S matrices for each supported dimension.
    for (int di = 0; di < TURBO_QUANT_NUM_DIMS; ++di) {
        int d = tq_host_dims[di];
        size_t bytes = (size_t)d * d * sizeof(float);

        // Pi
        CUDA_CHECK(cudaMalloc(&tq_d_Pi[di], bytes));
        CUDA_CHECK(cudaMemcpy(tq_d_Pi[di],
                              tq_get_Pi_row(d, 0),   // row-major: row 0 = start of matrix
                              bytes,
                              cudaMemcpyHostToDevice));

        // S
        CUDA_CHECK(cudaMalloc(&tq_d_S[di], bytes));
        CUDA_CHECK(cudaMemcpy(tq_d_S[di],
                              tq_get_S_row(d, 0),
                              bytes,
                              cudaMemcpyHostToDevice));
    }
}

void tq_cuda_init(void) {
    std::call_once(tq_cuda_init_flag, _tq_cuda_do_init);
}

// ---------------------------------------------------------------------------
// TQ_MSE quantization kernel
// One CUDA block per row; blockDim.x = dim (64 / 128 / 256).
// Shared memory layout:
//   smem[0..dim-1]   : input vector, normalized in-place
//   smem[dim..dim+7] : per-warp partial sums for the block reduce
// ---------------------------------------------------------------------------

__global__ void tq_mse_quantize_kernel(
    const float * __restrict__ x,
    uint8_t      * __restrict__ y,
    int   dim,
    int   dim_idx,
    int   row_stride_bytes,
    const float * __restrict__ d_pi)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    extern __shared__ float smem[];

    // Load input row
    float xi = x[(int64_t)row * dim + tid];
    smem[tid] = xi;
    __syncthreads();

    // Block reduce: sum of squares
    float sq = xi * xi;
    for (int mask = 16; mask > 0; mask >>= 1)
        sq += __shfl_xor_sync(0xffffffffu, sq, mask);
    if ((tid & 31) == 0)
        smem[dim + (tid >> 5)] = sq;
    __syncthreads();

    float norm_val, inv_norm;
    if (tid == 0) {
        float total = 0.0f;
        int n_warps = blockDim.x >> 5;
        for (int w = 0; w < n_warps; w++) total += smem[dim + w];
        norm_val = sqrtf(total);
        smem[dim]     = norm_val;
        smem[dim + 1] = (norm_val > 1e-20f) ? 1.0f / norm_val : 0.0f;
    }
    __syncthreads();

    norm_val = smem[dim];
    inv_norm = smem[dim + 1];

    // Normalize in-place
    smem[tid] *= inv_norm;
    __syncthreads();

    // Rotate and quantize: each thread computes one output coordinate
    float y_rot = 0.0f;
    for (int k = 0; k < dim; k++)
        y_rot += d_pi[tid * dim + k] * smem[k];

    // Nearest-centroid lookup
    float best_dist = fabsf(y_rot - tq_d_cb_2bit[dim_idx][0]);
    int best = 0;
    for (int c = 1; c < 4; c++) {
        float d = fabsf(y_rot - tq_d_cb_2bit[dim_idx][c]);
        if (d < best_dist) { best_dist = d; best = c; }
    }

    // Store index into shared (overwrite; x_norm no longer needed)
    smem[tid] = (float)best;
    __syncthreads();

    // Write output
    uint8_t * out_row = y + (int64_t)row * row_stride_bytes;

    if (tid == 0)
        *((float *)out_row) = norm_val;

    // Pack 4 x 2-bit indices per output byte, LSB-first
    if ((tid & 3) == 0) {
        uint8_t packed = (uint8_t)(
            ((int)smem[tid    ]      ) |
            ((int)smem[tid + 1] << 2) |
            ((int)smem[tid + 2] << 4) |
            ((int)smem[tid + 3] << 6));
        out_row[4 + (tid >> 2)] = packed;
    }
}

// ---------------------------------------------------------------------------
// Host launcher (C ABI)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// TQ_MSE dequantization kernel
// One CUDA block per row; blockDim.x = dim (64 / 128 / 256).
// Shared memory layout:
//   smem[0..dim-1] : y_tilde (dequantized centroid values)
//   smem[dim]      : stored norm (written by thread 0)
// ---------------------------------------------------------------------------

__global__ void tq_mse_dequantize_kernel(
    const uint8_t * __restrict__ x,
    float         * __restrict__ y,
    int   dim,
    int   dim_idx,
    int   row_stride_bytes,
    const float * __restrict__ d_pi)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    extern __shared__ float smem[];

    const uint8_t *in_row = x + (int64_t)row * row_stride_bytes;

    // Thread 0 reads stored norm into smem[dim]
    if (tid == 0)
        smem[dim] = *((const float *)in_row);

    // Unpack 2-bit index and load centroid for this coordinate
    uint8_t byte = in_row[4 + tid / 4];
    int idx = (byte >> ((tid & 3) * 2)) & 3;
    smem[tid] = tq_d_cb_2bit[dim_idx][idx];
    __syncthreads();

    float norm_val = smem[dim];

    // Inverse rotation: x_hat[tid] = sum_k( Pi[k, tid] * y_tilde[k] )
    //                              = sum_k( d_pi[k*dim + tid] * smem[k] )
    // Access pattern: at iteration k all threads read d_pi[k*dim .. k*dim+dim-1]
    // This is fully coalesced across threads.
    float x_hat = 0.0f;
    for (int k = 0; k < dim; k++)
        x_hat += d_pi[k * dim + tid] * smem[k];

    y[(int64_t)row * dim + tid] = x_hat * norm_val;
}

// ---------------------------------------------------------------------------
// Host dequantize launcher (C ABI)
// ---------------------------------------------------------------------------

extern "C" void ggml_cuda_tq_mse_dequantize(
    const void * x, float * y, int dim, int n_rows, cudaStream_t stream)
{
    tq_cuda_init();

    int dim_idx;
    switch (dim) {
        case  64: dim_idx = 0; break;
        case 128: dim_idx = 1; break;
        case 256: dim_idx = 2; break;
        default:
            fprintf(stderr, "ggml_cuda_tq_mse_dequantize: unsupported dim %d\n", dim);
            return;
    }

    int row_stride = (int)tq_mse_block_size(dim);
    size_t shmem   = (size_t)(dim + 2) * sizeof(float);

    tq_mse_dequantize_kernel<<<n_rows, dim, shmem, stream>>>(
        (const uint8_t *)x, y, dim, dim_idx, row_stride, tq_d_Pi[dim_idx]);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Host quantize launcher (C ABI)
// ---------------------------------------------------------------------------

extern "C" void ggml_cuda_tq_mse_quantize(
    const float * x, void * y, int dim, int n_rows, cudaStream_t stream)
{
    tq_cuda_init();

    int dim_idx;
    switch (dim) {
        case  64: dim_idx = 0; break;
        case 128: dim_idx = 1; break;
        case 256: dim_idx = 2; break;
        default:
            fprintf(stderr, "ggml_cuda_tq_mse_quantize: unsupported dim %d\n", dim);
            return;
    }

    int row_stride = (int)tq_mse_block_size(dim);
    size_t shmem   = (size_t)(dim + 8) * sizeof(float);

    tq_mse_quantize_kernel<<<n_rows, dim, shmem, stream>>>(
        x, (uint8_t *)y, dim, dim_idx, row_stride, tq_d_Pi[dim_idx]);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// TQ_PROD quantization kernel
// One CUDA block per row; blockDim.x = dim.
// Shared memory layout (2*dim + 19 floats):
//   smem[0..dim-1]           : x_norm, then overwritten with residual r
//   smem[dim..dim+7]         : warp partial sums for norm
//   smem[dim+8]              : norm_val
//   smem[dim+9]              : inv_norm
//   smem[dim+10..2*dim+9]    : y_tilde (centroid values in rotated domain, then Pi^T@y_tilde)
//   smem[2*dim+10..2*dim+17] : warp partial sums for gamma
//   smem[2*dim+18]           : gamma_val
// ---------------------------------------------------------------------------

__global__ void tq_prod_quantize_kernel(
    const float * __restrict__ x,
    uint8_t      * __restrict__ y,
    int   dim,
    int   dim_idx,
    int   row_stride_bytes,
    const float * __restrict__ d_pi,
    const float * __restrict__ d_s)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    extern __shared__ float smem[];

    float * x_norm   = smem;               // [0..dim-1]
    float * warp_buf = smem + dim;          // [dim..dim+7]
    float * y_tilde  = smem + dim + 10;    // [dim+10..2*dim+9]
    float * gamma_buf = smem + 2*dim + 10; // [2*dim+10..2*dim+17]

    // Load and square for norm
    float xi = x[(int64_t)row * dim + tid];
    x_norm[tid] = xi;
    __syncthreads();

    // Block reduce: sum of squares
    float sq = xi * xi;
    for (int mask = 16; mask > 0; mask >>= 1)
        sq += __shfl_xor_sync(0xffffffffu, sq, mask);
    if ((tid & 31) == 0)
        warp_buf[tid >> 5] = sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int n_warps = blockDim.x >> 5;
        for (int w = 0; w < n_warps; w++) total += warp_buf[w];
        float nv = sqrtf(total);
        smem[dim + 8] = nv;
        smem[dim + 9] = (nv > 1e-20f) ? 1.0f / nv : 0.0f;
    }
    __syncthreads();

    float norm_val = smem[dim + 8];
    float inv_norm = smem[dim + 9];

    // Normalize in-place
    x_norm[tid] *= inv_norm;
    __syncthreads();

    // Stage 1: MSE quantization
    // Each thread computes one rotated coordinate, finds nearest centroid
    float y_rot = 0.0f;
    for (int k = 0; k < dim; k++)
        y_rot += d_pi[tid * dim + k] * x_norm[k];

    float best_dist = fabsf(y_rot - tq_d_cb_2bit[dim_idx][0]);
    int best = 0;
    for (int c = 1; c < 4; c++) {
        float d = fabsf(y_rot - tq_d_cb_2bit[dim_idx][c]);
        if (d < best_dist) { best_dist = d; best = c; }
    }

    // Store centroid value for inverse rotation
    y_tilde[tid] = tq_d_cb_2bit[dim_idx][best];
    __syncthreads();

    // Inverse rotation: x_mse[tid] = sum_k( Pi[k, tid] * y_tilde[k] )
    // Access d_pi[k*dim + tid] — coalesced across threads
    float x_mse = 0.0f;
    for (int k = 0; k < dim; k++)
        x_mse += d_pi[k * dim + tid] * y_tilde[k];

    // Residual r = x_norm - x_mse; overwrite x_norm slot
    float r_val = x_norm[tid] - x_mse;
    x_norm[tid] = r_val;
    __syncthreads();

    // Gamma = ||r||: block reduce
    float rsq = r_val * r_val;
    for (int mask = 16; mask > 0; mask >>= 1)
        rsq += __shfl_xor_sync(0xffffffffu, rsq, mask);
    if ((tid & 31) == 0)
        gamma_buf[tid >> 5] = rsq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int n_warps = blockDim.x >> 5;
        for (int w = 0; w < n_warps; w++) total += gamma_buf[w];
        smem[2*dim + 18] = sqrtf(total);
    }
    __syncthreads();

    float gamma_val = smem[2*dim + 18];

    // QJL: each thread computes Sr_j = dot(S[j], r) and extracts sign bit
    float sr = 0.0f;
    for (int k = 0; k < dim; k++)
        sr += d_s[tid * dim + k] * x_norm[k];   // x_norm now holds r

    int qjl_bit = (sr >= 0.0f) ? 1 : 0;

    // Pack QJL bits using __ballot_sync (warp mask, 32 bits per warp)
    uint32_t ballot = __ballot_sync(0xffffffffu, qjl_bit);

    // Write output
    uint8_t * out_row = y + (int64_t)row * row_stride_bytes;
    int n_mse_bytes = (2 * dim + 7) / 8;

    if (tid == 0) {
        *((float *)out_row)       = norm_val;
        *((float *)(out_row + 4)) = gamma_val;
    }

    // Pack 2-bit MSE indices (same as TQ_MSE: 4 indices per byte)
    // Reuse y_tilde to store int indices temporarily
    // best is a register value; pack directly
    // Store best index into y_tilde (now free after inverse rotation)
    y_tilde[tid] = (float)best;
    __syncthreads();

    if ((tid & 3) == 0) {
        uint8_t packed = (uint8_t)(
            ((int)y_tilde[tid    ]      ) |
            ((int)y_tilde[tid + 1] << 2) |
            ((int)y_tilde[tid + 2] << 4) |
            ((int)y_tilde[tid + 3] << 6));
        out_row[8 + (tid >> 2)] = packed;
    }

    // Write QJL bits: one uint32 per warp (lane 0 of each warp writes)
    if ((tid & 31) == 0) {
        int warp_id = tid >> 5;
        uint8_t * qjl_buf = out_row + 8 + n_mse_bytes;
        // Copy 4 bytes of ballot to qjl_buf at warp_id * 4
        uint32_t * dst = (uint32_t *)(qjl_buf + warp_id * 4);
        *dst = ballot;
    }
}

// ---------------------------------------------------------------------------
// Host launcher (C ABI)
// ---------------------------------------------------------------------------

extern "C" void ggml_cuda_tq_prod_quantize(
    const float * x, void * y, int dim, int n_rows, cudaStream_t stream)
{
    tq_cuda_init();

    int dim_idx;
    switch (dim) {
        case  64: dim_idx = 0; break;
        case 128: dim_idx = 1; break;
        case 256: dim_idx = 2; break;
        default:
            fprintf(stderr, "ggml_cuda_tq_prod_quantize: unsupported dim %d\n", dim);
            return;
    }

    int row_stride = (int)tq_prod_block_size(dim);
    // smem: (2*dim + 19) floats
    size_t shmem = (size_t)(2 * dim + 19) * sizeof(float);

    tq_prod_quantize_kernel<<<n_rows, dim, shmem, stream>>>(
        x, (uint8_t *)y, dim, dim_idx, row_stride,
        tq_d_Pi[dim_idx], tq_d_S[dim_idx]);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// TQ_PROD dequantization kernel
// One CUDA block per row; blockDim.x = dim.
// Shared memory layout (dim + 2 floats):
//   smem[0..dim-1] : y_tilde (centroid values), then overwritten with qjl_signs
//   smem[dim]      : norm
//   smem[dim+1]    : gamma
// ---------------------------------------------------------------------------

__global__ void tq_prod_dequantize_kernel(
    const uint8_t * __restrict__ x,
    float         * __restrict__ y,
    int   dim,
    int   dim_idx,
    int   row_stride_bytes,
    const float * __restrict__ d_pi,
    const float * __restrict__ d_s)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    extern __shared__ float smem[];

    const uint8_t *in_row = x + (int64_t)row * row_stride_bytes;
    int n_mse_bytes = (2 * dim + 7) / 8;

    if (tid == 0) {
        smem[dim]     = *((const float *)in_row);         // norm
        smem[dim + 1] = *((const float *)(in_row + 4));   // gamma
    }

    // Unpack 2-bit MSE index for this thread; load centroid into shared
    uint8_t mse_byte = in_row[8 + tid / 4];
    int mse_idx = (mse_byte >> ((tid & 3) * 2)) & 3;
    smem[tid] = tq_d_cb_2bit[dim_idx][mse_idx];
    __syncthreads();

    float norm_val  = smem[dim];
    float gamma_val = smem[dim + 1];

    // x_mse[tid] = Pi^T row tid · y_tilde = sum_k d_pi[k*dim + tid] * smem[k]
    // Coalesced: all threads read d_pi[k*dim .. k*dim+dim-1] together
    float x_mse = 0.0f;
    for (int k = 0; k < dim; k++)
        x_mse += d_pi[k * dim + tid] * smem[k];

    // Sync before overwriting smem: all threads must finish reading y_tilde
    // before any thread overwrites its slot with the QJL sign.
    __syncthreads();

    // Overwrite shared with QJL signs for this thread
    uint8_t qjl_byte = in_row[8 + n_mse_bytes + tid / 8];
    int qjl_bit = (qjl_byte >> (tid & 7)) & 1;
    smem[tid] = (qjl_bit == 1) ? 1.0f : -1.0f;
    __syncthreads();

    // x_qjl[tid] = coeff * gamma * S^T row tid · qjl_signs
    // S^T row tid = column tid of S = d_s[k*dim + tid] (coalesced)
    float coeff = sqrtf(3.14159265f / 2.0f) / (float)dim;
    float x_qjl = 0.0f;
    for (int k = 0; k < dim; k++)
        x_qjl += d_s[k * dim + tid] * smem[k];
    x_qjl *= coeff * gamma_val;

    y[(int64_t)row * dim + tid] = (x_mse + x_qjl) * norm_val;
}

// ---------------------------------------------------------------------------
// Host dequantize launcher (C ABI)
// ---------------------------------------------------------------------------

extern "C" void ggml_cuda_tq_prod_dequantize(
    const void * x, float * y, int dim, int n_rows, cudaStream_t stream)
{
    tq_cuda_init();

    int dim_idx;
    switch (dim) {
        case  64: dim_idx = 0; break;
        case 128: dim_idx = 1; break;
        case 256: dim_idx = 2; break;
        default:
            fprintf(stderr, "ggml_cuda_tq_prod_dequantize: unsupported dim %d\n", dim);
            return;
    }

    int row_stride = (int)tq_prod_block_size(dim);
    size_t shmem   = (size_t)(dim + 2) * sizeof(float);

    tq_prod_dequantize_kernel<<<n_rows, dim, shmem, stream>>>(
        (const uint8_t *)x, y, dim, dim_idx, row_stride,
        tq_d_Pi[dim_idx], tq_d_S[dim_idx]);
    CUDA_CHECK(cudaGetLastError());
}
