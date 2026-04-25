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
