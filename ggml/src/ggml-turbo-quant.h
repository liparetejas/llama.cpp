#pragma once
#include "ggml.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  TurboQuant constants                                               */
/* ------------------------------------------------------------------ */

/* Maximum head dimension supported (must be a power of 2). */
#define TURBO_QUANT_MAX_DIM  256

/* Default bit-width for TurboQuantMSE stage (b-1 for TurboQuantProd). */
#define TURBO_QUANT_BITS_MSE    2    /* 2-bit MSE stage                */
#define TURBO_QUANT_BITS_PROD   3    /* 3-bit total = 2 MSE + 1 QJL    */

/* Codebook sizes */
#define TURBO_QUANT_K_MSE  (1 << TURBO_QUANT_BITS_MSE)           /* 4 centroids */
#define TURBO_QUANT_K_PROD (1 << (TURBO_QUANT_BITS_PROD - 1))    /* 4 MSE       */

/* ------------------------------------------------------------------ */
/*  Codebook tables (precomputed Lloyd-Max for Beta(d) distribution)   */
/*  Indexed as tq_codebook_2bit[dim_index][centroid_index]             */
/*  dim_index maps: 64->0, 128->1, 256->2                             */
/* ------------------------------------------------------------------ */
#define TURBO_QUANT_NUM_DIMS 3   /* support d=64, 128, 256 */

/* 2-bit codebook (4 centroids) for d in {64, 128, 256}               */
extern const float tq_codebook_2bit[TURBO_QUANT_NUM_DIMS][4];

/* 1-bit codebook (2 centroids) for d in {64, 128, 256}               */
extern const float tq_codebook_1bit[TURBO_QUANT_NUM_DIMS][2];

/* ------------------------------------------------------------------ */
/*  Rotation matrix seed helper                                        */
/*  Returns a deterministic uint64 seed derived from (dim, quant_type)*/
/* ------------------------------------------------------------------ */
uint64_t tq_rotation_seed(int dim, int quant_type_id);

/* ------------------------------------------------------------------ */
/*  Per-dim rotation matrices (Pi)                                     */
/*  Stored in row-major order: Pi[dim][dim] as float32.               */
/*  Must be initialised by calling tq_init_rotations() once at        */
/*  program startup (or lazily on first use).                         */
/* ------------------------------------------------------------------ */
void tq_init_rotations(void);   /* call once; idempotent (thread-safe with once_flag) */
bool tq_rotations_ready(void);  /* returns true after init */

/* Pi matrix accessor: returns pointer to row i of the d×d Pi matrix  */
const float * tq_get_Pi_row(int dim, int row);

/* S matrix accessor (for QJL stage): row i of d×d S matrix           */
const float * tq_get_S_row(int dim, int row);

/* ------------------------------------------------------------------ */
/*  Block size helpers                                                  */
/* ------------------------------------------------------------------ */

/* Number of bytes to store one quantised head vector of length dim    */
/*  TQ_MSE:  4 (norm) + ceil(2*dim/8)                                 */
/*  TQ_PROD: 4 (norm) + 4 (gamma) + ceil(2*dim/8) + ceil(dim/8)      */
size_t tq_mse_block_size(int dim);
size_t tq_prod_block_size(int dim);

/* ------------------------------------------------------------------ */
/*  CPU reference quantization (used by type_traits.from_float_ref    */
/*  and by tests; not performance-critical)                            */
/* ------------------------------------------------------------------ */

/* Quantise k floats from x into one TQ_MSE row at y.                 */
/* k must be 64, 128, or 256 (the head dimension).                    */
/* tq_init_rotations() must have been called first.                   */
void quantize_row_tq_mse_ref(const float * x, void * y, int64_t k);

/* Dequantise one TQ_MSE row from x back into k floats at y.          */
/* k must match the dim used during quantisation (64, 128, or 256).   */
void dequantize_row_tq_mse(const void * x, float * y, int64_t k);

/* Quantise k floats from x into one TQ_PROD row at y.                */
/* 2-bit MSE stage + 1-bit QJL residual stage (b=3 total bits).       */
void quantize_row_tq_prod_ref(const float * x, void * y, int64_t k);

/* Dequantise one TQ_PROD row from x back into k floats at y.         */
void dequantize_row_tq_prod(const void * x, float * y, int64_t k);

#ifdef __cplusplus
}
#endif
