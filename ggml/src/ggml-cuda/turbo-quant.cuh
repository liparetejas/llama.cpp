#pragma once
#include "../ggml-turbo-quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
// tq_d_Pi[di]   : Pi  stored row-major Pi[i*D+j]      → Pi[i][j]
// tq_d_Pi_T[di] : Pi^T stored row-major Pi_T[j*D+i]   → used for COALESCED Phase-1 Q rotation
// tq_d_S[di]    : S   stored row-major S[i*D+j]
// tq_d_S_T[di]  : S^T stored row-major S_T[j*D+i]     → used for COALESCED Phase-1 Q→S rotation
// ---------------------------------------------------------------------------

extern float * tq_d_Pi  [TURBO_QUANT_NUM_DIMS];
extern float * tq_d_S   [TURBO_QUANT_NUM_DIMS];
extern float * tq_d_Pi_T[TURBO_QUANT_NUM_DIMS];
extern float * tq_d_S_T [TURBO_QUANT_NUM_DIMS];

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

// ---------------------------------------------------------------------------
// Dequantise n_rows rows of TQ_MSE data to F32.
// x         : device pointer to n_rows * tq_mse_block_size(dim) bytes
// y         : device pointer to output n_rows * dim floats
// dim       : head dimension; must be 64, 128, or 256
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
#endif
void ggml_cuda_tq_mse_dequantize(
    const void * x, float * y, int dim, int n_rows, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Quantise n_rows rows to TQ_PROD format (2-bit MSE + 1-bit QJL, b=3).
// x         : device pointer to n_rows * dim floats
// y         : device pointer to output buffer (n_rows * tq_prod_block_size(dim) bytes)
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
#endif
void ggml_cuda_tq_prod_quantize(
    const float * x, void * y, int dim, int n_rows, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Dequantise n_rows rows of TQ_PROD data to F32.
// x         : device pointer to n_rows * tq_prod_block_size(dim) bytes
// y         : device pointer to output n_rows * dim floats
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
#endif
void ggml_cuda_tq_prod_dequantize(
    const void * x, float * y, int dim, int n_rows, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Convert TQ_MSE / TQ_PROD → F16 for flash-attention (used by ggml_get_to_fp16_cuda).
// k = total element count (nelements of the tensor = n_rows * 128).
// ---------------------------------------------------------------------------
void ggml_cuda_tq_mse_to_f16(const void * x, half * y, int64_t k, cudaStream_t stream);
void ggml_cuda_tq_prod_to_f16(const void * x, half * y, int64_t k, cudaStream_t stream);

// ---------------------------------------------------------------------------
// SET_ROWS: write indexed rows from F32 src into TQ_MSE/TQ_PROD dst.
// src0_d : float input  [ne01 x ne00], strides s01/s02/s03 in floats
// src1_d : row indices  [ne01 x ne02 x ne03], strides s10/s11/s12 in idx_t
// dst_d  : TQ output, byte strides nb1/nb2/nb3
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
void ggml_cuda_tq_mse_set_rows_i32(
    const float * src0_d, const int32_t * src1_d, void * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int head_dim, cudaStream_t stream);

void ggml_cuda_tq_mse_set_rows_i64(
    const float * src0_d, const int64_t * src1_d, void * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int head_dim, cudaStream_t stream);

void ggml_cuda_tq_prod_set_rows_i32(
    const float * src0_d, const int32_t * src1_d, void * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int head_dim, cudaStream_t stream);

void ggml_cuda_tq_prod_set_rows_i64(
    const float * src0_d, const int64_t * src1_d, void * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int head_dim, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
