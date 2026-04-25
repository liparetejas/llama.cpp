#pragma once
#include "../ggml-turbo-quant.h"
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Constant memory: Lloyd-Max codebooks (small, read-only, all threads share)
// Defined (without extern) in turbo-quant.cu; declared here for all other TUs.
// TQ_DEFINING_TU is set by turbo-quant.cu before including this header so
// that the definition and declaration don't collide in the same TU.
// ---------------------------------------------------------------------------

#ifndef TQ_DEFINING_TU
extern __constant__ float tq_d_cb_2bit[TURBO_QUANT_NUM_DIMS][4];
extern __constant__ float tq_d_cb_1bit[TURBO_QUANT_NUM_DIMS][2];
#endif

// ---------------------------------------------------------------------------
// Device pointers: Pi and S matrices (one per supported dim 64/128/256)
// Set at init; read-only during inference.
// ---------------------------------------------------------------------------

extern float * tq_d_Pi[TURBO_QUANT_NUM_DIMS];
extern float * tq_d_S [TURBO_QUANT_NUM_DIMS];

// ---------------------------------------------------------------------------
// Initialise device-side TurboQuant data.
// Idempotent; safe to call from multiple host threads.
// Must be called while a CUDA device is active (i.e. inside the CUDA backend
// init path).
// ---------------------------------------------------------------------------
void tq_cuda_init(void);

// ---------------------------------------------------------------------------
// Quantise n_rows rows of dim F32 values to TQ_MSE format.
// x         : device pointer to n_rows * dim floats (row-major, contiguous)
// y         : device pointer to output buffer (n_rows * tq_mse_block_size(dim) bytes)
// dim       : head dimension; must be 64, 128, or 256
// n_rows    : number of vectors to quantise
// stream    : CUDA stream
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
#endif
void ggml_cuda_tq_mse_quantize(
    const float * x, void * y, int dim, int n_rows, cudaStream_t stream);
