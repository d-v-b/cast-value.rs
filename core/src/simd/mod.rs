//! SIMD-accelerated conversion kernels.
//!
//! These are internal fast paths called from the slice conversion functions
//! when the configuration allows vectorization (empty scalar_map, clamp mode,
//! supported rounding mode). The public API is unchanged.
//!
//! Dispatch priority per function:
//! 1. **x86_64 + AVX2** (`pulp::x86::V3`) — raw intrinsics; best x86_64 perf.
//! 2. **x86_64 + AVX1** — 256-bit FP operations for CPUs without AVX2.
//! 3. **AArch64** — NEON intrinsics; always available on AArch64 targets.
//! 4. **WASM32 + SIMD128** — 128-bit SIMD for WebAssembly.
//! 5. **Everything else** — [`pulp::Arch::new().dispatch`][pulp::Arch] with the
//!    generic [`WithSimd`][pulp::WithSimd] kernel: vectorizes NaN detection and
//!    clamping on any arch pulp supports, with per-lane scalar rounding as the
//!    only non-vectorized step.

use crate::RoundingMode;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "x86_64")]
mod avx;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "wasm32")]
mod wasm32;

mod generic;

/// Try to convert f64 slice to u8 slice using SIMD with clamping.
///
/// Always returns `Ok(true)` on success or `Err` if a NaN or ±Inf was detected.
///
/// # Preconditions
///
/// * `src.len() == dst.len()`
/// * `rounding` is not `NearestAway` (caller must check)
pub fn try_f64_to_u8_clamp(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    dispatch_f64_to_u8(src, dst, rounding)
}

/// Try to convert f64 slice to i32 slice using SIMD with clamping.
///
/// Always returns `Ok(true)` on success or `Err` if a NaN or ±Inf was detected.
///
/// # Preconditions
///
/// * `src.len() == dst.len()`
/// * `rounding` is not `NearestAway` (caller must check)
pub fn try_f64_to_i32_clamp(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    dispatch_f64_to_i32(src, dst, rounding)
}

/// Try to convert f32 slice to u8 slice using SIMD with clamping.
///
/// Always returns `Ok(true)` on success or `Err` if a NaN or ±Inf was detected.
///
/// # Preconditions
///
/// * `src.len() == dst.len()`
/// * `rounding` is not `NearestAway` (caller must check)
pub fn try_f32_to_u8_clamp(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    dispatch_f32_to_u8(src, dst, rounding)
}

/// Try to convert f64 slice to f32 slice using SIMD with nearest-even rounding.
///
/// When `error_on_overflow` is true, returns `Err(OutOfRange)` if any finite
/// f64 overflows to ±Inf in f32. When false, overflow to ±Inf is accepted.
///
/// # Preconditions
///
/// * `src.len() == dst.len()`
pub fn try_f64_to_f32_nearest(
    src: &[f64],
    dst: &mut [f32],
    error_on_overflow: bool,
) -> Result<bool, crate::CastError> {
    dispatch_f64_to_f32(src, dst, error_on_overflow)
}

/// Try to convert f64 slice to i32 slice using SIMD, returning an error
/// if any value is out of range (no clamping).
///
/// # Preconditions
///
/// * `src.len() == dst.len()`
/// * `rounding` is not `NearestAway` (caller must check)
pub fn try_f64_to_i32_check(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    dispatch_f64_to_i32_check(src, dst, rounding)
}

/// Convert f64 to i32 element by element on the scalar path, with no
/// out-of-range mode, stopping at the first error.
///
/// The `f64_to_i32_check` kernels hand over to this for their scalar tail and
/// for the rest of the slice once a vector holds a NaN/Inf or out-of-range
/// lane, so their errors match the scalar path in both variant and order.
///
/// # Errors
///
/// Returns the first element's `CastError`, as `convert_float_to_int` would.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
fn f64_to_i32_check_scalar(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let config = crate::FloatToIntConfig {
        map_entries: Vec::new(), // `Vec::new` does not allocate
        rounding,
        out_of_range: None,
    };
    for (&val, out) in src.iter().zip(dst.iter_mut()) {
        *out = crate::convert_float_to_int(val, &config)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Architecture dispatch — each function selects the optimal kernel.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn dispatch_f64_to_u8(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    if let pulp::x86::Arch::V3(simd) = pulp::x86::Arch::new() {
        // SAFETY: V3 guarantees AVX2 + SSE4.1 are available.
        return unsafe { avx2::f64_to_u8_clamp(simd, src, dst, rounding) }.map(|()| true);
    }
    // Check for AVX1 (without AVX2)
    if is_x86_feature_detected!("avx") {
        // SAFETY: Runtime check confirmed AVX is available.
        return unsafe { avx::f64_to_u8_clamp(src, dst, rounding) }.map(|()| true);
    }
    // Fall back to the generic WithSimd kernel.
    generic::f64_to_u8_clamp(src, dst, rounding)
}

#[cfg(target_arch = "aarch64")]
fn dispatch_f64_to_u8(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    // SAFETY: NEON is always available on AArch64.
    unsafe { aarch64::f64_to_u8_clamp(src, dst, rounding) }.map(|()| true)
}

#[cfg(target_arch = "wasm32")]
fn dispatch_f64_to_u8(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    #[cfg(target_feature = "simd128")]
    {
        // SAFETY: simd128 feature is enabled at compile time.
        return unsafe { wasm32::f64_to_u8_clamp(src, dst, rounding) }.map(|()| true);
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        generic::f64_to_u8_clamp(src, dst, rounding)
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn dispatch_f64_to_u8(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    generic::f64_to_u8_clamp(src, dst, rounding)
}

#[cfg(target_arch = "x86_64")]
fn dispatch_f64_to_i32(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    if let pulp::x86::Arch::V3(simd) = pulp::x86::Arch::new() {
        return unsafe { avx2::f64_to_i32_clamp(simd, src, dst, rounding) }.map(|()| true);
    }
    if is_x86_feature_detected!("avx") {
        return unsafe { avx::f64_to_i32_clamp(src, dst, rounding) }.map(|()| true);
    }
    generic::f64_to_i32_clamp(src, dst, rounding)
}

#[cfg(target_arch = "aarch64")]
fn dispatch_f64_to_i32(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    unsafe { aarch64::f64_to_i32_clamp(src, dst, rounding) }.map(|()| true)
}

#[cfg(target_arch = "wasm32")]
fn dispatch_f64_to_i32(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    #[cfg(target_feature = "simd128")]
    {
        return unsafe { wasm32::f64_to_i32_clamp(src, dst, rounding) }.map(|()| true);
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        generic::f64_to_i32_clamp(src, dst, rounding)
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn dispatch_f64_to_i32(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    generic::f64_to_i32_clamp(src, dst, rounding)
}

#[cfg(target_arch = "x86_64")]
fn dispatch_f32_to_u8(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    if let pulp::x86::Arch::V3(simd) = pulp::x86::Arch::new() {
        return unsafe { avx2::f32_to_u8_clamp(simd, src, dst, rounding) }.map(|()| true);
    }
    if is_x86_feature_detected!("avx") {
        return unsafe { avx::f32_to_u8_clamp(src, dst, rounding) }.map(|()| true);
    }
    generic::f32_to_u8_clamp(src, dst, rounding)
}

#[cfg(target_arch = "aarch64")]
fn dispatch_f32_to_u8(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    unsafe { aarch64::f32_to_u8_clamp(src, dst, rounding) }.map(|()| true)
}

#[cfg(target_arch = "wasm32")]
fn dispatch_f32_to_u8(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    #[cfg(target_feature = "simd128")]
    {
        return unsafe { wasm32::f32_to_u8_clamp(src, dst, rounding) }.map(|()| true);
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        generic::f32_to_u8_clamp(src, dst, rounding)
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn dispatch_f32_to_u8(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    generic::f32_to_u8_clamp(src, dst, rounding)
}

// ---------------------------------------------------------------------------
// Dispatch: f64 → f32 (nearest-even)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn dispatch_f64_to_f32(
    src: &[f64],
    dst: &mut [f32],
    error_on_overflow: bool,
) -> Result<bool, crate::CastError> {
    if let pulp::x86::Arch::V3(simd) = pulp::x86::Arch::new() {
        return unsafe { avx2::f64_to_f32_nearest(simd, src, dst, error_on_overflow) }
            .map(|()| true);
    }
    if is_x86_feature_detected!("avx") {
        return unsafe { avx::f64_to_f32_nearest(src, dst, error_on_overflow) }.map(|()| true);
    }
    Ok(false)
}

#[cfg(target_arch = "aarch64")]
fn dispatch_f64_to_f32(
    src: &[f64],
    dst: &mut [f32],
    error_on_overflow: bool,
) -> Result<bool, crate::CastError> {
    unsafe { aarch64::f64_to_f32_nearest(src, dst, error_on_overflow) }.map(|()| true)
}

#[cfg(target_arch = "wasm32")]
fn dispatch_f64_to_f32(
    src: &[f64],
    dst: &mut [f32],
    error_on_overflow: bool,
) -> Result<bool, crate::CastError> {
    #[cfg(target_feature = "simd128")]
    {
        return unsafe { wasm32::f64_to_f32_nearest(src, dst, error_on_overflow) }.map(|()| true);
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        let _ = (src, dst, error_on_overflow);
        Ok(false)
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn dispatch_f64_to_f32(
    _src: &[f64],
    _dst: &mut [f32],
    _error_on_overflow: bool,
) -> Result<bool, crate::CastError> {
    Ok(false)
}

// ---------------------------------------------------------------------------
// Dispatch: f64 → i32 (range-check, no clamp)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn dispatch_f64_to_i32_check(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    if let pulp::x86::Arch::V3(simd) = pulp::x86::Arch::new() {
        return unsafe { avx2::f64_to_i32_check(simd, src, dst, rounding) }.map(|()| true);
    }
    if is_x86_feature_detected!("avx") {
        return unsafe { avx::f64_to_i32_check(src, dst, rounding) }.map(|()| true);
    }
    Ok(false)
}

#[cfg(target_arch = "aarch64")]
fn dispatch_f64_to_i32_check(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    unsafe { aarch64::f64_to_i32_check(src, dst, rounding) }.map(|()| true)
}

#[cfg(target_arch = "wasm32")]
fn dispatch_f64_to_i32_check(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    #[cfg(target_feature = "simd128")]
    {
        return unsafe { wasm32::f64_to_i32_check(src, dst, rounding) }.map(|()| true);
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        let _ = (src, dst, rounding);
        Ok(false)
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn dispatch_f64_to_i32_check(
    _src: &[f64],
    _dst: &mut [i32],
    _rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    Ok(false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{convert_float_to_int, FloatToIntConfig, OutOfRangeMode};

    /// Reference scalar conversion for comparison.
    fn scalar_f64_to_u8_clamp(
        src: &[f64],
        rounding: RoundingMode,
    ) -> Result<Vec<u8>, crate::CastError> {
        let config = FloatToIntConfig::<f64, u8> {
            map_entries: vec![],
            rounding,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        let mut dst = vec![0u8; src.len()];
        for (i, &val) in src.iter().enumerate() {
            dst[i] = convert_float_to_int(val, &config)?;
        }
        Ok(dst)
    }

    fn scalar_f32_to_u8_clamp(
        src: &[f32],
        rounding: RoundingMode,
    ) -> Result<Vec<u8>, crate::CastError> {
        let config = FloatToIntConfig::<f32, u8> {
            map_entries: vec![],
            rounding,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        let mut dst = vec![0u8; src.len()];
        for (i, &val) in src.iter().enumerate() {
            dst[i] = convert_float_to_int(val, &config)?;
        }
        Ok(dst)
    }

    /// Test SIMD matches scalar for all rounding modes with in-range values.
    #[test]
    fn test_f64_to_u8_all_rounding_modes() {
        let src: Vec<f64> = (0..1000).map(|i| (i % 256) as f64 + 0.7).collect();

        for mode in [
            RoundingMode::NearestEven,
            RoundingMode::TowardsZero,
            RoundingMode::TowardsPositive,
            RoundingMode::TowardsNegative,
        ] {
            let expected = scalar_f64_to_u8_clamp(&src, mode).unwrap();
            let mut dst = vec![0u8; src.len()];
            let result = try_f64_to_u8_clamp(&src, &mut dst, mode);
            match result {
                Ok(true) => assert_eq!(dst, expected, "mismatch for {mode:?}"),
                Ok(false) => unreachable!("SIMD path should never return Ok(false)"),
                Err(e) => panic!("unexpected error for {mode:?}: {e}"),
            }
        }
    }

    /// Test boundary values: 0, 255, negative, >255.
    #[test]
    fn test_f64_to_u8_boundaries() {
        let src = vec![0.0, 255.0, -0.0, -100.0, 500.0, 0.5, 254.5, 127.3];
        let expected = scalar_f64_to_u8_clamp(&src, RoundingMode::NearestEven).unwrap();
        let mut dst = vec![0u8; src.len()];
        let result = try_f64_to_u8_clamp(&src, &mut dst, RoundingMode::NearestEven)
            .expect("unexpected error");
        assert!(result, "expected Ok(true)");
        assert_eq!(dst, expected);
    }

    /// Test that NaN is detected and returns an error.
    #[test]
    fn test_f64_to_u8_nan_error() {
        let src = vec![1.0, 2.0, f64::NAN, 4.0];
        let mut dst = vec![0u8; src.len()];
        let result = try_f64_to_u8_clamp(&src, &mut dst, RoundingMode::NearestEven);
        assert!(
            matches!(result, Err(crate::CastError::NanOrInf { .. })),
            "expected NanOrInf error, got {result:?}"
        );
    }

    /// Test various slice lengths including tail handling.
    #[test]
    fn test_f64_to_u8_various_lengths() {
        for len in [0, 1, 3, 15, 16, 17, 31, 32, 33, 100, 1000] {
            let src: Vec<f64> = (0..len).map(|i| (i % 256) as f64 + 0.1).collect();
            let expected = scalar_f64_to_u8_clamp(&src, RoundingMode::NearestEven).unwrap();
            let mut dst = vec![0u8; len];
            let result = try_f64_to_u8_clamp(&src, &mut dst, RoundingMode::NearestEven)
                .unwrap_or_else(|e| panic!("unexpected error for len={len}: {e}"));
            assert!(result, "expected Ok(true) for len={len}");
            assert_eq!(dst, expected, "mismatch for len={len}");
        }
    }

    /// Test f32→u8 SIMD matches scalar.
    #[test]
    fn test_f32_to_u8_all_rounding_modes() {
        let src: Vec<f32> = (0..1000).map(|i| (i % 256) as f32 + 0.7).collect();

        for mode in [
            RoundingMode::NearestEven,
            RoundingMode::TowardsZero,
            RoundingMode::TowardsPositive,
            RoundingMode::TowardsNegative,
        ] {
            let expected = scalar_f32_to_u8_clamp(&src, mode).unwrap();
            let mut dst = vec![0u8; src.len()];
            match try_f32_to_u8_clamp(&src, &mut dst, mode) {
                Ok(true) => assert_eq!(dst, expected, "mismatch for {mode:?}"),
                Ok(false) => unreachable!("SIMD path should never return Ok(false)"),
                Err(e) => panic!("unexpected error for {mode:?}: {e}"),
            }
        }
    }

    /// Test f32→u8 NaN detection.
    #[test]
    fn test_f32_to_u8_nan_error() {
        let src = vec![1.0f32, f32::NAN, 3.0];
        let mut dst = vec![0u8; src.len()];
        let result = try_f32_to_u8_clamp(&src, &mut dst, RoundingMode::NearestEven);
        assert!(
            matches!(result, Err(crate::CastError::NanOrInf { .. })),
            "expected NanOrInf error, got {result:?}"
        );
    }

    /// ±Inf is not representable in an integer type, so every float→int kernel
    /// must reject it rather than clamp it. Lengths cover both the vector body
    /// and the scalar tail of every architecture's kernel.
    #[test]
    fn test_float_to_int_kernels_reject_inf() {
        for len in [1, 2, 3, 8, 9, 16, 17, 33] {
            for inf in [f64::INFINITY, f64::NEG_INFINITY] {
                for pos in [0, len / 2, len - 1] {
                    let mut src = vec![1.0_f64; len];
                    src[pos] = inf;
                    let src_f32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
                    let mut dst_u8 = vec![0u8; len];
                    let mut dst_i32 = vec![0i32; len];
                    let results = [
                        try_f64_to_u8_clamp(&src, &mut dst_u8, RoundingMode::NearestEven),
                        try_f64_to_i32_clamp(&src, &mut dst_i32, RoundingMode::NearestEven),
                        try_f64_to_i32_check(&src, &mut dst_i32, RoundingMode::NearestEven),
                        try_f32_to_u8_clamp(&src_f32, &mut dst_u8, RoundingMode::NearestEven),
                    ];
                    // Only the check kernel may be unavailable on a CPU (`Ok(false)`,
                    // meaning the caller falls back to the scalar path); the clamp
                    // kernels always run.
                    for (kernel, result) in results.into_iter().enumerate() {
                        let fallback_ok = kernel == 2 && matches!(result, Ok(false));
                        assert!(
                            fallback_ok || matches!(result, Err(crate::CastError::NanOrInf { .. })),
                            "kernel {kernel}, len={len}, pos={pos}, {inf}: got {result:?}"
                        );
                    }
                }
            }
        }
    }

    /// Empty slice should succeed with Ok(true).
    #[test]
    fn test_empty_slice() {
        let src: Vec<f64> = vec![];
        let mut dst: Vec<u8> = vec![];
        let result = try_f64_to_u8_clamp(&src, &mut dst, RoundingMode::NearestEven);
        assert!(matches!(result, Ok(true)));
    }
}
