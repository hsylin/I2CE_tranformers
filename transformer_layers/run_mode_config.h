#pragma once

/*
 * Centralized compile-time run-mode configuration.
 *
 * Each CFG_* macro below normalizes a build flag to either 1 or 0, so the rest
 * of the transformer code can use simple #if checks without repeating
 * #ifdef/#else blocks. The source build flags, such as RELOAD_WEIGHT or SIMD,
 * are supplied by the Makefile/SCons command line.
 */

// --------------------------------------------------
// Base runtime/data-source switches
// --------------------------------------------------

#ifdef RELOAD_WEIGHT
#define CFG_RELOAD_WEIGHT 1
#else
#define CFG_RELOAD_WEIGHT 0
#endif

// Use notebook-generated Dense .bin weights when runtime weight reloading is
// enabled. These weights are needed by Dense fallback/reference execution.
#if CFG_RELOAD_WEIGHT && defined(USE_NOTEBOOK_GENERATED_WEIGHTS)
#define CFG_USE_NOTEBOOK_GENERATED_WEIGHTS 1
#else
#define CFG_USE_NOTEBOOK_GENERATED_WEIGHTS 0
#endif

// Use generated CodebookDense layers instead of the default Dense kernels when
// weights are being reloaded and USE_CODEBOOK_GEMM is supplied.
#if CFG_RELOAD_WEIGHT && defined(USE_CODEBOOK_GEMM)
#define CFG_USE_CODEBOOK_GEMM 1
#else
#define CFG_USE_CODEBOOK_GEMM 0
#endif

// --------------------------------------------------
// Validation / profiling switches
// --------------------------------------------------

// Enable Dense reference path and numerical comparison.
// Default: OFF unless explicitly requested with ENABLE_CODEBOOK_REFERENCE.
#if CFG_USE_CODEBOOK_GEMM && defined(ENABLE_CODEBOOK_REFERENCE)
#define CFG_USE_CODEBOOK_REFERENCE 1
#else
#define CFG_USE_CODEBOOK_REFERENCE 0
#endif

// Enable debug prints.
// Default: OFF unless explicitly requested.
#ifdef ENABLE_DEBUG_PRINT
#define CFG_ENABLE_DEBUG_PRINT 1
#else
#define CFG_ENABLE_DEBUG_PRINT 0
#endif

// Optional: a dedicated profiling mode.
// If enabled, force-disable reference and debug printing so the measured region
// is the CodebookDense GEMM path instead of validation or logging overhead.
#ifdef PROFILE_GEMM_ONLY
#define CFG_PROFILE_GEMM_ONLY 1
#else
#define CFG_PROFILE_GEMM_ONLY 0
#endif

// Emit named gem5 checkpoint dumps for coarse transformer regions.
// Post-processing can subtract adjacent dumpstats snapshots to get each
// region while the last snapshot remains the whole transformer block total.
#ifdef GEM5_PROFILE_REGIONS
#define CFG_GEM5_PROFILE_REGIONS 1
#else
#define CFG_GEM5_PROFILE_REGIONS 0
#endif

// Enable SIMD/SVE implementations when the build defines SIMD.
#ifdef SIMD
#define CFG_SIMD 1
#else
#define CFG_SIMD 0
#endif

// Run the default Dense transformer as a no-SIMD, sequential multi-learner
// baseline. The learner count comes directly from N_LEARNERS in codebooks_def.h.
#ifdef DENSE_NO_SIMD_BASELINE
#define CFG_DENSE_NO_SIMD_BASELINE 1
#else
#define CFG_DENSE_NO_SIMD_BASELINE 0
#endif

// Keep the whole 4-learner transformer block in [seq][feature][learner]
// layout between grouped CodebookDense, attention, softmax, AddNorm, and FFN.
#ifdef FULL_INTERLEAVED_PIPELINE
#define CFG_FULL_INTERLEAVED_PIPELINE 1
#else
#define CFG_FULL_INTERLEAVED_PIPELINE 0
#endif

// Run the Transformer block with float32 activations/output accumulation.
// This is intentionally independent from USE_F16, which is kept for a future
// half-precision implementation.
#ifdef USE_FP32_TRANSFORMER
#define CFG_USE_FP32_TRANSFORMER 1
#else
#define CFG_USE_FP32_TRANSFORMER 0
#endif

#if CFG_PROFILE_GEMM_ONLY || CFG_GEM5_PROFILE_REGIONS
#undef CFG_USE_CODEBOOK_REFERENCE
#define CFG_USE_CODEBOOK_REFERENCE 0

#undef CFG_ENABLE_DEBUG_PRINT
#define CFG_ENABLE_DEBUG_PRINT 0
#endif

// Pure codebook mode means the run should rely entirely on the generated
// CodebookDense registry for layer weights. In this mode we do not build Dense
// fallback/reference weights unless reference validation is explicitly enabled.
#if CFG_USE_CODEBOOK_GEMM && !CFG_USE_CODEBOOK_REFERENCE
#define CFG_CODEBOOK_ONLY_MODE 1
#else
#define CFG_CODEBOOK_ONLY_MODE 0
#endif

// --------------------------------------------------
// Sanity checks
// --------------------------------------------------

// Dense fallback/reference runs still need notebook-generated per-learner .bin
// weights. Pure codebook runs skip Dense weights entirely, so they may use the
// generated registry without notebook-generated .bin files.
#if CFG_USE_CODEBOOK_GEMM && !CFG_PROFILE_GEMM_ONLY && \
    !CFG_CODEBOOK_ONLY_MODE && !CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#error "Non-profile Codebook GEMM requires USE_NOTEBOOK_GENERATED_WEIGHTS when Dense fallback/reference is enabled."
#endif

// Reference comparison only makes sense when codebook GEMM is enabled.
#if CFG_USE_CODEBOOK_REFERENCE && !CFG_USE_CODEBOOK_GEMM
#error "ENABLE_CODEBOOK_REFERENCE requires USE_CODEBOOK_GEMM."
#endif

#if CFG_DENSE_NO_SIMD_BASELINE && CFG_SIMD
#error "DENSE_NO_SIMD_BASELINE requires SIMD_FLAG=0."
#endif

#if CFG_DENSE_NO_SIMD_BASELINE && CFG_USE_CODEBOOK_GEMM
#error "DENSE_NO_SIMD_BASELINE requires USE_CODEBOOK_GEMM_FLAG=0."
#endif

#if CFG_DENSE_NO_SIMD_BASELINE && CFG_FULL_INTERLEAVED_PIPELINE
#error "DENSE_NO_SIMD_BASELINE cannot be combined with FULL_INTERLEAVED_PIPELINE."
#endif

#if CFG_FULL_INTERLEAVED_PIPELINE && !CFG_USE_CODEBOOK_GEMM
#error "FULL_INTERLEAVED_PIPELINE requires USE_CODEBOOK_GEMM."
#endif

#if CFG_FULL_INTERLEAVED_PIPELINE && !CFG_SIMD && !CFG_USE_FP32_TRANSFORMER
#error "FULL_INTERLEAVED_PIPELINE requires SIMD_FLAG=1 for the int8 pipeline."
#endif

#if CFG_USE_FP32_TRANSFORMER && !CFG_USE_CODEBOOK_GEMM
#error "USE_FP32_TRANSFORMER requires USE_CODEBOOK_GEMM because the default Dense path is int8-only."
#endif

#if CFG_USE_FP32_TRANSFORMER && CFG_USE_CODEBOOK_REFERENCE
#error "USE_FP32_TRANSFORMER cannot be combined with ENABLE_CODEBOOK_REFERENCE; use the notebook FP32 comparison instead."
#endif
