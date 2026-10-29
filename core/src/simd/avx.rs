//! AVX1 SIMD kernels for x86_64 CPUs with AVX but not AVX2.
//!
//! AMD FX-8350 (Piledriver) supports AVX1, FMA4, XOP but not AVX2.
//! These kernels use 256-bit AVX registers (4 × f64, 8 × f32) which is
//! significantly faster than scalar but slower than AVX2 due to lack of
//! integer SIMD for packing operations.

use crate::RoundingMode;
use core::arch::x86_64::*;

/// Select the AVX1 rounding mode constant for `_mm256_round_pd`/`_mm256_round_ps`.
fn avx_round_mode(rounding: RoundingMode) -> i32 {
    match rounding {
        RoundingMode::NearestEven => 0x08,
        RoundingMode::TowardsZero => 0x0B,
        RoundingMode::TowardsPositive => 0x0A,
        RoundingMode::TowardsNegative => 0x09,
        RoundingMode::NearestAway => unreachable!(),
    }
}

/// Convert f64 slice to u8 slice with rounding and clamping (AVX1).
///
/// Processes 4 f64 at a time using 256-bit AVX registers.
/// Slower than AVX2 version due to lack of AVX2 integer packing instructions.
///
/// # Safety
///
/// Caller must ensure AVX is available.
#[target_feature(enable = "avx")]
pub(super) unsafe fn f64_to_u8_clamp(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let round_mode = avx_round_mode(rounding);
    let n = src.len();
    let simd_len = n / 4 * 4;

    let lo = _mm256_set1_pd(0.0);
    let hi = _mm256_set1_pd(255.0);

    // Temporary buffer for converting f64 → u8 (AVX1 lacks integer pack ops)
    let mut temp = [0.0f64; 4];

    for i in (0..simd_len).step_by(4) {
        let v = _mm256_loadu_pd(src.as_ptr().add(i));

        // NaN/Inf check (`x - x` is NaN exactly when x is NaN or ±Inf)
        let nan_mask = _mm256_cmp_pd(_mm256_sub_pd(v, v), _mm256_setzero_pd(), _CMP_UNORD_Q);
        if _mm256_movemask_pd(nan_mask) != 0 {
            for &val in &src[i..std::cmp::min(i + 4, n)] {
                if !val.is_finite() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        // Round
        let r = round_f64_avx(v, round_mode);

        // Clamp
        let c = _mm256_min_pd(_mm256_max_pd(r, lo), hi);

        // Store to temp buffer and convert to u8
        _mm256_storeu_pd(temp.as_mut_ptr(), c);
        for j in 0..4 {
            dst[i + j] = temp[j] as u8;
        }
    }

    // Scalar tail
    for i in simd_len..n {
        let val = src[i];
        if !val.is_finite() {
            return Err(crate::CastError::NanOrInf { value: val });
        }
        let rounded = match rounding {
            RoundingMode::NearestEven => val.round_ties_even(),
            RoundingMode::TowardsZero => val.trunc(),
            RoundingMode::TowardsPositive => val.ceil(),
            RoundingMode::TowardsNegative => val.floor(),
            RoundingMode::NearestAway => unreachable!(),
        };
        dst[i] = rounded.clamp(0.0, 255.0) as u8;
    }

    Ok(())
}

/// Convert f64 slice to i32 slice with rounding and clamping (AVX1).
///
/// # Safety
///
/// Caller must ensure AVX is available.
#[target_feature(enable = "avx")]
pub(super) unsafe fn f64_to_i32_clamp(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let round_mode = avx_round_mode(rounding);
    let n = src.len();
    let simd_len = n / 4 * 4;

    let lo = _mm256_set1_pd(i32::MIN as f64);
    let hi = _mm256_set1_pd(i32::MAX as f64);

    for i in (0..simd_len).step_by(4) {
        let v = _mm256_loadu_pd(src.as_ptr().add(i));

        // NaN/Inf check (`x - x` is NaN exactly when x is NaN or ±Inf)
        let nan_mask = _mm256_cmp_pd(_mm256_sub_pd(v, v), _mm256_setzero_pd(), _CMP_UNORD_Q);
        if _mm256_movemask_pd(nan_mask) != 0 {
            for &val in &src[i..std::cmp::min(i + 4, n)] {
                if !val.is_finite() {
                    return Err(crate::CastError::NanOrInf { value: val });
                }
            }
        }

        let r = round_f64_avx(v, round_mode);
        let c = _mm256_min_pd(_mm256_max_pd(r, lo), hi);

        // Convert f64x4 → i32x4 (returns 128-bit)
        let converted = _mm256_cvtpd_epi32(c);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, converted);
    }

    // Scalar tail
    for i in simd_len..n {
        let val = src[i];
        if !val.is_finite() {
            return Err(crate::CastError::NanOrInf { value: val });
        }
        let rounded = match rounding {
            RoundingMode::NearestEven => val.round_ties_even(),
            RoundingMode::TowardsZero => val.trunc(),
            RoundingMode::TowardsPositive => val.ceil(),
            RoundingMode::TowardsNegative => val.floor(),
            RoundingMode::NearestAway => unreachable!(),
        };
        dst[i] = rounded.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    }

    Ok(())
}

/// Convert f32 slice to u8 slice with rounding and clamping (AVX1).
///
/// Processes 8 f32 at a time using 256-bit AVX registers.
///
/// # Safety
///
/// Caller must ensure AVX is available.
#[target_feature(enable = "avx")]
pub(super) unsafe fn f32_to_u8_clamp(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let round_mode = avx_round_mode(rounding);
    let n = src.len();
    let simd_len = n / 8 * 8;

    let lo = _mm256_set1_ps(0.0);
    let hi = _mm256_set1_ps(255.0);

    let mut temp = [0.0f32; 8];

    for i in (0..simd_len).step_by(8) {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));

        // NaN/Inf check (`x - x` is NaN exactly when x is NaN or ±Inf)
        let nan_mask = _mm256_cmp_ps(_mm256_sub_ps(v, v), _mm256_setzero_ps(), _CMP_UNORD_Q);
        if _mm256_movemask_ps(nan_mask) != 0 {
            for &val in &src[i..std::cmp::min(i + 8, n)] {
                if !val.is_finite() {
                    return Err(crate::CastError::NanOrInf { value: val as f64 });
                }
            }
        }

        let r = round_f32_avx(v, round_mode);
        let c = _mm256_min_ps(_mm256_max_ps(r, lo), hi);

        // Store to temp buffer and convert to u8
        _mm256_storeu_ps(temp.as_mut_ptr(), c);
        for j in 0..8 {
            dst[i + j] = temp[j] as u8;
        }
    }

    // Scalar tail
    for i in simd_len..n {
        let val = src[i];
        if !val.is_finite() {
            return Err(crate::CastError::NanOrInf { value: val as f64 });
        }
        let rounded = match rounding {
            RoundingMode::NearestEven => val.round_ties_even(),
            RoundingMode::TowardsZero => val.trunc(),
            RoundingMode::TowardsPositive => val.ceil(),
            RoundingMode::TowardsNegative => val.floor(),
            RoundingMode::NearestAway => unreachable!(),
        };
        dst[i] = rounded.clamp(0.0, 255.0) as u8;
    }

    Ok(())
}

/// Convert f64 slice to f32 slice using nearest-even (two-pass, AVX1).
///
/// `_mm256_cvtpd_ps` is an AVX1 instruction: 4 f64 → 4 f32.
///
/// # Safety
///
/// Caller must ensure AVX is available.
#[target_feature(enable = "avx")]
pub(super) unsafe fn f64_to_f32_nearest(
    src: &[f64],
    dst: &mut [f32],
    error_on_overflow: bool,
) -> Result<(), crate::CastError> {
    let n = src.len();
    let simd_len = n / 4 * 4;

    // Pass 1: branch-free narrowing
    for i in (0..simd_len).step_by(4) {
        let v = _mm256_loadu_pd(src.as_ptr().add(i));
        let narrowed = _mm256_cvtpd_ps(v);
        _mm_storeu_ps(dst.as_mut_ptr().add(i), narrowed);
    }
    for i in simd_len..n {
        dst[i] = src[i] as f32;
    }

    // Pass 2: overflow check
    if error_on_overflow {
        for i in 0..n {
            if src[i].is_finite() && dst[i].is_infinite() {
                return Err(crate::CastError::FloatOutOfRange {
                    value: src[i],
                    lo: f32::MIN as f64,
                    hi: f32::MAX as f64,
                });
            }
        }
    }

    Ok(())
}

/// Convert f64 slice to i32 slice with rounding, error on out-of-range (AVX1).
///
/// # Safety
///
/// Caller must ensure AVX is available.
#[target_feature(enable = "avx")]
pub(super) unsafe fn f64_to_i32_check(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<(), crate::CastError> {
    let round_mode = avx_round_mode(rounding);
    let n = src.len();
    let simd_len = n / 4 * 4;

    let lo = _mm256_set1_pd(i32::MIN as f64);
    let hi = _mm256_set1_pd(i32::MAX as f64);

    for i in (0..simd_len).step_by(4) {
        let v = _mm256_loadu_pd(src.as_ptr().add(i));
        let r = round_f64_avx(v, round_mode);

        // `x - x` is NaN exactly when x is NaN or ±Inf; NaN also fails both
        // range comparisons, so it needs this separate mask.
        let non_finite = _mm256_cmp_pd(_mm256_sub_pd(v, v), _mm256_setzero_pd(), _CMP_UNORD_Q);
        let below = _mm256_cmp_pd(r, lo, _CMP_LT_OQ);
        let above = _mm256_cmp_pd(r, hi, _CMP_GT_OQ);
        let invalid = _mm256_or_pd(non_finite, _mm256_or_pd(below, above));
        if _mm256_movemask_pd(invalid) != 0 {
            // A NaN/Inf or out-of-range lane: finish on the scalar path, which
            // reports the first error in element order, exactly as the
            // non-SIMD path does.
            return super::f64_to_i32_check_scalar(&src[i..], &mut dst[i..], rounding);
        }

        let converted = _mm256_cvtpd_epi32(r);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, converted);
    }

    // Scalar tail
    super::f64_to_i32_check_scalar(&src[simd_len..], &mut dst[simd_len..], rounding)
}

// ---------------------------------------------------------------------------
// Rounding helpers
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn round_f64_avx(v: __m256d, mode: i32) -> __m256d {
    match mode {
        0x08 => _mm256_round_pd(v, 0x08),
        0x09 => _mm256_round_pd(v, 0x09),
        0x0A => _mm256_round_pd(v, 0x0A),
        0x0B => _mm256_round_pd(v, 0x0B),
        _ => unreachable!(),
    }
}

#[inline(always)]
unsafe fn round_f32_avx(v: __m256, mode: i32) -> __m256 {
    match mode {
        0x08 => _mm256_round_ps(v, 0x08),
        0x09 => _mm256_round_ps(v, 0x09),
        0x0A => _mm256_round_ps(v, 0x0A),
        0x0B => _mm256_round_ps(v, 0x0B),
        _ => unreachable!(),
    }
}
