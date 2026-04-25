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
