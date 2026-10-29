//! zarr-cast-value: Pure Rust implementation of the cast_value codec's
//! per-element conversion logic.
//!
//! This crate is independent of Python/PyO3 and can be used by any Rust consumer
//! (e.g. zarrs, zarr-python bindings).

use num_traits::{Float, FromPrimitive, One, ToPrimitive};
use std::ops::{Add, Sub};

mod simd;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during element conversion.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CastError {
    /// A NaN or Infinity value was encountered when casting to an integer type.
    #[error("Cannot cast {value} to integer type without scalar_map")]
    NanOrInf { value: f64 },
    /// A value is outside an integer target's range and no out_of_range mode
    /// was set. `lo` and `hi` are the target type's exact inclusive bounds.
    #[error(
        "Value {value} out of range [{lo}, {hi}]. \
         Set out_of_range='clamp' or out_of_range='wrap' to handle this."
    )]
    OutOfRange { value: f64, lo: i128, hi: i128 },
    /// A finite value rounds to a magnitude beyond a float target's finite
    /// range `[lo, hi]`, and out_of_range is not `clamp` (`wrap` is not
    /// permitted for float targets).
    #[error(
        "Value {value} out of range [{lo}, {hi}]. \
         Set out_of_range='clamp' to map it to ±Infinity."
    )]
    FloatOutOfRange { value: f64, lo: f64, hi: f64 },
    /// The source and destination slices have different lengths.
    #[error("Source has {src_len} elements but destination has {dst_len}")]
    LengthMismatch { src_len: usize, dst_len: usize },
}

// ---------------------------------------------------------------------------
// Rounding modes
// ---------------------------------------------------------------------------

/// How to round floating-point values when casting to integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoundingMode {
    NearestEven,
    TowardsZero,
    TowardsPositive,
    TowardsNegative,
    NearestAway,
}

impl std::str::FromStr for RoundingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nearest-even" => Ok(Self::NearestEven),
            "towards-zero" => Ok(Self::TowardsZero),
            "towards-positive" => Ok(Self::TowardsPositive),
            "towards-negative" => Ok(Self::TowardsNegative),
            "nearest-away" => Ok(Self::NearestAway),
            _ => Err(format!("Unknown rounding mode: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Out-of-range modes
// ---------------------------------------------------------------------------

/// How to handle values outside the target integer type's range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutOfRangeMode {
    Clamp,
    Wrap,
}

impl std::str::FromStr for OutOfRangeMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "clamp" => Ok(Self::Clamp),
            "wrap" => Ok(Self::Wrap),
            _ => Err(format!("Unknown out_of_range mode: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar map entry
// ---------------------------------------------------------------------------

/// A pre-parsed scalar map entry, typed on Src and Dst.
#[derive(Debug, Clone, Copy)]
pub struct MapEntry<Src, Dst> {
    pub src: Src,
    pub tgt: Dst,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Common base for all numeric types that participate in cast_value conversions.
/// Provides only the operations shared by both integers and floats.
///
/// Uses `num_traits::ToPrimitive` for f64 conversion (used in error reporting).
pub trait CastNum: Copy + PartialEq + PartialOrd + std::fmt::Debug + ToPrimitive {}

/// An integer numeric type (i8..i64, u8..u64).
///
/// The type's bit width `N` is `8 * size_of::<Self>()`; `wrap` reduces modulo
/// `2^N`. `num_traits::FromPrimitive` builds the wrapped value from its
/// residue.
pub trait CastInt: CastNum + FromPrimitive {}

/// A floating-point numeric type (f32, f64).
/// Carries float-specific operations not covered by `num_traits::Float`:
/// mode-aware rounding and IEEE 754 nextUp/nextDown.
///
/// `num_traits::Float` provides: `is_nan`, `is_infinite`, `is_finite`, `abs`.
/// `num_traits::One` provides: `one()`.
pub trait CastFloat: CastNum + Float + One + Add<Output = Self> + Sub<Output = Self> {
    /// Round to integer according to the given rounding mode.
    fn round_with_mode(self, mode: RoundingMode) -> Self;
    /// The smallest representable value greater than self (IEEE 754 nextUp).
    fn next_up(self) -> Self;
    /// The largest representable value less than self (IEEE 754 nextDown).
    fn next_down(self) -> Self;
}

/// Conversion from Src to Dst. Implemented per (Src, Dst) pair.
pub trait CastInto<Dst: CastNum>: CastNum {
    /// The minimum Dst value, represented in Src's type.
    fn dst_min() -> Self;
    /// The maximum Dst value, represented in Src's type.
    fn dst_max() -> Self;
    /// Convert self to Dst. Caller must ensure value is in range.
    fn cast_into(self) -> Dst;
}

// For constructing MapEntry values from f64 pairs (e.g. at the Python boundary),
// use `num_traits::FromPrimitive::from_f64()` which returns `Option<Self>`.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a numeric value to f64 for error reporting (not on the hot path).
/// Returns `f64::NAN` if the conversion is not possible.
#[inline]
fn to_f64_lossy<T: ToPrimitive>(val: T) -> f64 {
    val.to_f64().unwrap_or(f64::NAN)
}

// ---------------------------------------------------------------------------
// Scalar map lookup
// ---------------------------------------------------------------------------

/// Apply scalar_map lookup for float sources. Returns Some(tgt) on match.
#[inline]
fn apply_scalar_map_float<Src: CastFloat, Dst: CastNum>(
    val: Src,
    map_entries: &[MapEntry<Src, Dst>],
) -> Option<Dst> {
    for entry in map_entries {
        if entry.src.is_nan() {
            if val.is_nan() {
                return Some(entry.tgt);
            }
        } else if val == entry.src {
            return Some(entry.tgt);
        }
    }
    None
}

/// Apply scalar_map lookup for integer sources. Returns Some(tgt) on match.
/// Integers can never be NaN, so only exact equality is checked.
#[inline]
fn apply_scalar_map_int<Src: CastInt, Dst: CastNum>(
    val: Src,
    map_entries: &[MapEntry<Src, Dst>],
) -> Option<Dst> {
    for entry in map_entries {
        if val == entry.src {
            return Some(entry.tgt);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Conversion configs (per-path)
// ---------------------------------------------------------------------------

/// Configuration for float→int conversion.
pub struct FloatToIntConfig<Src, Dst> {
    pub map_entries: Vec<MapEntry<Src, Dst>>,
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for int→int conversion.
pub struct IntToIntConfig<Src, Dst> {
    pub map_entries: Vec<MapEntry<Src, Dst>>,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for float→float conversion.
pub struct FloatToFloatConfig<Src, Dst> {
    pub map_entries: Vec<MapEntry<Src, Dst>>,
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for int→float conversion.
///
/// `out_of_range` only matters for targets narrower than the integer range
/// (e.g. `int32` → `float16`, whose finite range ends at 65504).
pub struct IntToFloatConfig<Src, Dst> {
    pub map_entries: Vec<MapEntry<Src, Dst>>,
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// 2^63 and 2^64: the `i64`/`u64` domain limits, exact in `f64`.
const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

/// Handle a float result whose rounded magnitude exceeds the target's finite
/// range, according to `out_of_range`.
///
/// Per the spec, `clamp` maps such values to ±Infinity for targets that have
/// infinities (all supported float types do). `wrap` is not permitted for
/// float targets, and no mode means overflow is an error.
///
/// # Arguments
///
/// * `negative` - Whether the source value is negative (selects -Infinity)
/// * `value` - The source value as `f64`, for error reporting
/// * `out_of_range` - The configured out-of-range mode
///
/// # Errors
///
/// Returns `CastError::FloatOutOfRange` unless `out_of_range` is `clamp`.
#[cold]
#[inline(never)]
fn float_overflow<Dst: CastFloat>(
    negative: bool,
    value: f64,
    out_of_range: Option<OutOfRangeMode>,
) -> Result<Dst, CastError> {
    match out_of_range {
        Some(OutOfRangeMode::Clamp) => Ok(if negative {
            Dst::neg_infinity()
        } else {
            Dst::infinity()
        }),
        Some(OutOfRangeMode::Wrap) | None => Err(CastError::FloatOutOfRange {
            value,
            lo: to_f64_lossy(Dst::min_value()),
            hi: to_f64_lossy(Dst::max_value()),
        }),
    }
}

/// The number of significand bits of a float type, including the implicit
/// leading bit (24 for `f32`, 53 for `f64`, 11 for `f16`).
///
/// Derived from the exponent of `epsilon() = 2^(1 - p)`, which constant-folds
/// after monomorphization.
#[inline]
fn mantissa_digits<F: CastFloat>() -> u32 {
    let epsilon_bits = to_f64_lossy(F::epsilon()).to_bits();
    // The f64 exponent field is biased by 1023 (`as` here reinterprets small
    // bit-field values, never truncating).
    let exponent = ((epsilon_bits >> 52) & 0x7ff) as i32 - 1023;
    (1 - exponent) as u32
}

/// The magnitude and sign of an integer, exact for every `CastInt` type.
#[inline]
fn int_magnitude<Src: CastInt>(val: Src) -> (u64, bool) {
    match val.to_i64() {
        Some(v) => (v.unsigned_abs(), v < 0),
        None => (
            val.to_u64()
                .expect("an integer outside the i64 range is a u64"),
            false,
        ),
    }
}

/// Round an integer (given as `magnitude` and sign) to `mantissa_digits`
/// significant bits with the given rounding mode, returning the exact rounded
/// value as `f64`.
///
/// The rounding happens on the integer's bits, so it is a single rounding
/// even for `i64`/`u64` values that `f64` cannot hold exactly. The result has
/// at most `mantissa_digits` (≤ 53) significant bits and a magnitude of at
/// most 2^64, so it is exact in `f64`.
fn round_int_to_precision(
    magnitude: u64,
    negative: bool,
    mantissa_digits: u32,
    rounding: RoundingMode,
) -> f64 {
    let significant_bits = u64::BITS - magnitude.leading_zeros();
    let rounded = if significant_bits <= mantissa_digits {
        magnitude as f64
    } else {
        // `unit` is the weight of the lowest kept bit; `remainder` is what
        // truncation to `mantissa_digits` bits discards.
        let shift = significant_bits - mantissa_digits;
        let unit = 1_u64 << shift;
        let remainder = magnitude & (unit - 1);
        let truncated = magnitude - remainder;
        let half = unit >> 1;
        let round_up = match rounding {
            RoundingMode::NearestEven => {
                remainder > half || (remainder == half && (magnitude >> shift) & 1 == 1)
            }
            RoundingMode::TowardsZero => false,
            RoundingMode::TowardsPositive => !negative && remainder != 0,
            RoundingMode::TowardsNegative => negative && remainder != 0,
            RoundingMode::NearestAway => remainder >= half,
        };
        // Adding in f64 rather than u64 lets a carry reach 2^64; the sum has at
        // most `mantissa_digits` significant bits, so it is exact.
        truncated as f64 + if round_up { unit as f64 } else { 0.0 }
    };
    if negative {
        -rounded
    } else {
        rounded
    }
}

/// The bit width `N` and signedness of an integer type.
///
/// Both constant-fold after monomorphization.
#[inline]
fn int_width<T: CastInt>() -> (u32, bool) {
    // Only signed types can represent -1.
    (
        8 * std::mem::size_of::<T>() as u32,
        T::from_i64(-1).is_some(),
    )
}

/// The exact inclusive bounds of an integer type, for error reporting.
fn int_bounds<T: CastInt>() -> (i128, i128) {
    match int_width::<T>() {
        (bits, true) => (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1),
        (bits, false) => (0, (1_i128 << bits) - 1),
    }
}

/// Reduce a finite, integer-valued `f64` modulo `2^N` into the `N`-bit integer
/// type `Dst`, using two's complement for signed targets.
///
/// Exact for every input: values in `[-2^63, 2^63)` convert exactly to `i64`,
/// whose two's complement bits are the residue modulo 2^64. Larger magnitudes
/// are multiples of 2^11 in `f64`, so `rem_euclid(2^64)` cannot round.
fn wrap_integral_f64<Dst: CastInt>(val: f64) -> Dst {
    // `as` between integer types reinterprets the bits (two's complement).
    let residue = if (-TWO_POW_63..TWO_POW_63).contains(&val) {
        val as i64 as u64
    } else {
        val.rem_euclid(TWO_POW_64) as u64
    };
    // Shifting the residue's low N bits to the top and back keeps exactly
    // those bits; the arithmetic shift of an `i64` also sign-extends them.
    let (bits, signed) = int_width::<Dst>();
    let shift = u64::BITS - bits;
    let wrapped = if signed {
        Dst::from_i64(((residue << shift) as i64) >> shift)
    } else {
        Dst::from_u64((residue << shift) >> shift)
    };
    wrapped.expect("a residue modulo 2^N fits in an N-bit integer")
}

// ---------------------------------------------------------------------------
// Layer 1: Four per-element conversion functions
// ---------------------------------------------------------------------------

/// Convert a single float element to an integer.
///
/// Pipeline: scalar_map → NaN/Inf check → round → range check/clamp/wrap → cast.
///
/// # Errors
///
/// Returns `CastError::NanOrInf` for NaN or ±Infinity not covered by the
/// scalar map (integers cannot represent them, whatever `out_of_range` is).
/// Returns `CastError::OutOfRange` for an out-of-range value when
/// `out_of_range` is not set.
#[inline]
pub fn convert_float_to_int<Src, Dst>(
    val: Src,
    config: &FloatToIntConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastInt,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_float(val, &config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: reject NaN and ±Inf. The spec requires an error for values the
    // target cannot represent even with `out_of_range` set.
    if !val.is_finite() {
        return Err(CastError::NanOrInf {
            value: to_f64_lossy(val),
        });
    }

    // Step 3: round
    let rounded = val.round_with_mode(config.rounding);

    // Step 4: range check. `lo` (0 or -2^(N-1)) is exact in every float type,
    // but `hi` (2^k - 1) may round up to 2^k (e.g. `i32::MAX as f32` is 2^31),
    // so `val <= hi` would accept 2^k. `hi + 1` is exactly 2^k either way (or
    // +Inf for f16, which still bounds every finite value), making
    // `[lo, hi_exclusive)` the exact set of in-range values.
    let lo = Src::dst_min();
    let hi = Src::dst_max();
    let hi_exclusive = hi + Src::one();
    if rounded >= lo && rounded < hi_exclusive {
        return Ok(rounded.cast_into());
    }

    // Step 5: out-of-range handling
    match config.out_of_range {
        // When `hi` rounded up to 2^k, `as` saturates it to the true maximum.
        Some(OutOfRangeMode::Clamp) => Ok(if rounded < lo { lo } else { hi }.cast_into()),
        Some(OutOfRangeMode::Wrap) => Ok(wrap_integral_f64(to_f64_lossy(rounded))),
        None => {
            let (lo, hi) = int_bounds::<Dst>();
            Err(CastError::OutOfRange {
                value: to_f64_lossy(val),
                lo,
                hi,
            })
        }
    }
}

/// Convert a single integer element to another integer type.
///
/// Pipeline: scalar_map → range check/clamp/wrap → cast.
///
/// # Errors
///
/// Returns `CastError::OutOfRange` for an out-of-range value when
/// `out_of_range` is not set.
#[inline]
pub fn convert_int_to_int<Src, Dst>(
    val: Src,
    config: &IntToIntConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastInt,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_int(val, &config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: range check + out-of-range handling
    let lo = Src::dst_min();
    let hi = Src::dst_max();
    match config.out_of_range {
        Some(OutOfRangeMode::Clamp) => {
            let clamped = if val < lo {
                lo
            } else if val > hi {
                hi
            } else {
                val
            };
            Ok(clamped.cast_into())
        }
        Some(OutOfRangeMode::Wrap) => {
            // Int→int wrap: Rust's `as` cast truncates to the target width,
            // which is exactly modular arithmetic.
            Ok(val.cast_into())
        }
        None => {
            if val < lo || val > hi {
                // `lo`/`hi` above are clipped to the source range; report the
                // target type's own bounds.
                let (dst_lo, dst_hi) = int_bounds::<Dst>();
                Err(CastError::OutOfRange {
                    value: to_f64_lossy(val),
                    lo: dst_lo,
                    hi: dst_hi,
                })
            } else {
                Ok(val.cast_into())
            }
        }
    }
}

/// Convert a single float element to another float type.
///
/// Pipeline: scalar_map → NaN/Inf propagation → cast → rounding adjustment → range check.
///
/// Per the zarr cast_value spec:
/// - NaN values MUST be propagated (if both types support IEEE 754).
/// - Signed zero MUST be preserved.
/// - Rounding applies when the target type has insufficient precision.
/// - Out-of-range handling applies when the *rounded* value exceeds the
///   target's finite range. `clamp` maps it to ±Infinity; `wrap` is not
///   permitted for float targets.
///
/// # Errors
///
/// Returns `CastError::OutOfRange` for a finite value whose rounded magnitude
/// exceeds the target's finite range, unless `out_of_range` is `clamp`.
#[inline]
pub fn convert_float_to_float<Src, Dst>(
    val: Src,
    config: &FloatToFloatConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastFloat,
{
    float_to_float_with_rounding(val, config, config.rounding)
}

/// [`convert_float_to_float`] with the rounding mode passed separately, so the slice
/// function can pass a constant per mode and let the per-element mode checks
/// fold away.
#[inline(always)]
fn float_to_float_with_rounding<Src, Dst>(
    val: Src,
    config: &FloatToFloatConfig<Src, Dst>,
    rounding: RoundingMode,
) -> Result<Dst, CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastFloat,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_float(val, &config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: NaN and Inf propagate naturally through the cast (IEEE 754).

    // Step 3: cast (Rust `as` uses nearest-even rounding by default)
    let nearest = val.cast_into();

    // Step 4: if the cast was not exact and a non-nearest-even rounding mode is
    // requested, adjust the result. Every supported float converts to f64
    // exactly, so comparing in f64 detects inexactness exactly.
    let val_f64 = to_f64_lossy(val);
    let nearest_f64 = to_f64_lossy(nearest);
    if rounding == RoundingMode::NearestEven || val_f64 == nearest_f64 {
        // Exact or nearest-even: the only possible overflow is a finite value
        // that nearest-even rounded to ±Inf.
        if nearest.is_infinite() && val.is_finite() {
            return float_overflow(val_f64 < 0.0, val_f64, config.out_of_range);
        }
        return Ok(nearest);
    }

    // Inexact with a directed or nearest-away mode, or NaN (NaN != NaN). ±Inf
    // converts exactly, so `val` is finite or NaN; every comparison below is
    // false for NaN, which therefore passes through unchanged.
    let result = match rounding {
        RoundingMode::NearestEven => nearest, // unreachable, but keep exhaustive
        RoundingMode::TowardsZero => {
            // Need the value closer to zero
            if nearest_f64.abs() > val_f64.abs() {
                if val_f64 >= 0.0 {
                    nearest.next_down()
                } else {
                    nearest.next_up()
                }
            } else {
                nearest
            }
        }
        RoundingMode::TowardsPositive => {
            // Need the value >= original
            if nearest_f64 < val_f64 {
                nearest.next_up()
            } else {
                nearest
            }
        }
        RoundingMode::TowardsNegative => {
            // Need the value <= original
            if nearest_f64 > val_f64 {
                nearest.next_down()
            } else {
                nearest
            }
        }
        RoundingMode::NearestAway => {
            // Nearest, ties away from zero. The `as` cast already gave
            // us nearest-even. We only need to adjust on ties where
            // nearest-even went towards zero but nearest-away should
            // go away from zero. Check if the midpoint between the result
            // and the adjacent value equals the original.
            let candidate = if val_f64 > nearest_f64 {
                nearest.next_up()
            } else {
                nearest.next_down()
            };
            let mid = (nearest_f64 + to_f64_lossy(candidate)) / 2.0;
            if mid == val_f64 && candidate.abs() > nearest.abs() {
                candidate // a tie: pick the one farther from zero
            } else {
                nearest // not a tie, nearest-even already correct
            }
        }
    };

    // Step 5: out-of-range check on the rounded value. Overflow needs the
    // result or the nearest-even value to be ±Inf; `|` (not `||`) keeps this
    // hot test branch-free, and the exact decision is made off the hot path.
    if result.is_infinite() | nearest.is_infinite() {
        return directed_overflow_check(val_f64, nearest, result, config.out_of_range);
    }
    Ok(result)
}

/// Decide whether an inexact float→float result that is, or came from, ±Inf
/// overflowed, and handle it per `out_of_range`.
///
/// It overflows when the result is ±Inf, or when nearest-even overflowed and
/// a towards-zero adjustment stepped back to ±MAX: with an unbounded exponent,
/// that adjustment only lands on MAX while |val| < MAX + ulp(MAX) =
/// 2^(emax + 1). (Only narrowing casts get here, so the threshold never
/// overflows f64.)
///
/// # Errors
///
/// Returns `CastError::FloatOutOfRange` on overflow unless `out_of_range` is
/// `clamp`.
#[cold]
#[inline(never)]
fn directed_overflow_check<Dst: CastFloat>(
    val_f64: f64,
    nearest: Dst,
    result: Dst,
    out_of_range: Option<OutOfRangeMode>,
) -> Result<Dst, CastError> {
    let max = to_f64_lossy(Dst::max_value());
    let ulp_of_max = max - to_f64_lossy(Dst::max_value().next_down());
    let overflowed =
        result.is_infinite() || (nearest.is_infinite() && val_f64.abs() >= max + ulp_of_max);
    if overflowed {
        return float_overflow(val_f64 < 0.0, val_f64, out_of_range);
    }
    Ok(result)
}

/// Convert a single integer element to a float type.
///
/// Pipeline: scalar_map → round to the target precision → range check.
///
/// Per the zarr cast_value spec, rounding applies when "casting from integer
/// types to floating-point types with insufficient mantissa bits (e.g. int64
/// to float32)". Directed rounding is done on the integer's bits, so large
/// `i64`/`u64` values are rounded once rather than first through `f64`.
/// Out-of-range handling only applies to narrow targets such as `f16`.
///
/// # Errors
///
/// Returns `CastError::OutOfRange` when the rounded value exceeds the
/// target's finite range, unless `out_of_range` is `clamp`.
#[inline]
pub fn convert_int_to_float<Src, Dst>(
    val: Src,
    config: &IntToFloatConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastFloat,
{
    int_to_float_with_rounding(val, config, config.rounding)
}

/// [`convert_int_to_float`] with the rounding mode passed separately, so the slice
/// function can pass a constant per mode and let the per-element mode checks
/// fold away.
#[inline(always)]
fn int_to_float_with_rounding<Src, Dst>(
    val: Src,
    config: &IntToFloatConfig<Src, Dst>,
    rounding: RoundingMode,
) -> Result<Dst, CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastFloat,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_int(val, &config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: round. Rust's `as` rounds integers to the nearest float, ties to
    // even, and is exact (so right in every mode) when the bits from the
    // highest to the lowest set bit fit in the target's significand. Only the
    // remaining values are rounded on the integer's bits.
    let nearest = val.cast_into();
    let (magnitude, negative) = int_magnitude(val);
    let digits = mantissa_digits::<Dst>();
    let significant_span =
        (u64::BITS - magnitude.leading_zeros()).saturating_sub(magnitude.trailing_zeros());
    let result = if rounding == RoundingMode::NearestEven || significant_span <= digits {
        nearest
    } else {
        let rounded = round_int_to_precision(magnitude, negative, digits, rounding);
        if rounded.abs() > to_f64_lossy(Dst::max_value()) {
            Dst::infinity() // overflow marker, handled below
        } else {
            // `rounded` is exactly representable in Dst, so this cannot fail.
            <Dst as num_traits::NumCast>::from(rounded)
                .expect("a value rounded to the target precision is representable")
        }
    };

    // Step 3: an integer is always finite, so an infinite result is overflow.
    // Only targets whose finite range ends below 2^64 (f16) can overflow; the
    // range test is a constant, so wide targets skip the check entirely.
    if to_f64_lossy(Dst::max_value()) < TWO_POW_64 && result.is_infinite() {
        let val_f64 = to_f64_lossy(val);
        return float_overflow(val_f64 < 0.0, val_f64, config.out_of_range);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Layer 2: Four slice conversion functions
// ---------------------------------------------------------------------------

/// Convert a slice of float values to integer values. Returns early on first error.
///
/// When the configuration allows it (empty scalar_map, clamp mode,
/// rounding mode other than NearestAway), uses SIMD-accelerated kernels
/// for supported type pairs (f64→u8, f64→i32, f32→u8).
///
/// # Errors
///
/// Returns `CastError::LengthMismatch` if `src` and `dst` differ in length
/// (checked before anything is written), or the first element's error.
pub fn convert_slice_float_to_int<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &FloatToIntConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastFloat + CastInto<Dst> + 'static,
    Dst: CastInt + 'static,
{
    if src.len() != dst.len() {
        return Err(CastError::LengthMismatch {
            src_len: src.len(),
            dst_len: dst.len(),
        });
    }
    // SIMD fast path: empty scalar_map + clamp + supported rounding mode.
    // Uses TypeId to dispatch to concrete SIMD kernels at runtime.
    // The `'static` bound is satisfied by all primitive numeric types.
    if config.map_entries.is_empty()
        && config.out_of_range == Some(OutOfRangeMode::Clamp)
        && config.rounding != RoundingMode::NearestAway
    {
        use std::any::TypeId;

        // f64 → u8
        if TypeId::of::<Src>() == TypeId::of::<f64>() && TypeId::of::<Dst>() == TypeId::of::<u8>() {
            // SAFETY: We just verified Src == f64 and Dst == u8 via TypeId.
            let src_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
            let dst_u8: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len()) };
            if simd::try_f64_to_u8_clamp(src_f64, dst_u8, config.rounding)? {
                return Ok(());
            }
        }

        // f64 → i32
        if TypeId::of::<Src>() == TypeId::of::<f64>() && TypeId::of::<Dst>() == TypeId::of::<i32>()
        {
            let src_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
            let dst_i32: &mut [i32] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len()) };
            if simd::try_f64_to_i32_clamp(src_f64, dst_i32, config.rounding)? {
                return Ok(());
            }
        }

        // f32 → u8
        if TypeId::of::<Src>() == TypeId::of::<f32>() && TypeId::of::<Dst>() == TypeId::of::<u8>() {
            let src_f32: &[f32] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len()) };
            let dst_u8: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len()) };
            if simd::try_f32_to_u8_clamp(src_f32, dst_u8, config.rounding)? {
                return Ok(());
            }
        }
    }

    // SIMD fast path: empty scalar_map + no out_of_range (error mode) + supported rounding.
    // Range-checking variant: errors if any value is out of range.
    if config.map_entries.is_empty()
        && config.out_of_range.is_none()
        && config.rounding != RoundingMode::NearestAway
    {
        use std::any::TypeId;

        // f64 → i32
        if TypeId::of::<Src>() == TypeId::of::<f64>() && TypeId::of::<Dst>() == TypeId::of::<i32>()
        {
            let src_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
            let dst_i32: &mut [i32] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i32, dst.len()) };
            if simd::try_f64_to_i32_check(src_f64, dst_i32, config.rounding)? {
                return Ok(());
            }
        }
    }

    // Scalar fallback
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_float_to_int(*in_val, config)?;
    }
    Ok(())
}

/// Convert a slice of integer values to integer values. Returns early on first error.
///
/// # Errors
///
/// Returns `CastError::LengthMismatch` if `src` and `dst` differ in length
/// (checked before anything is written), or the first element's error.
pub fn convert_slice_int_to_int<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &IntToIntConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastInt,
{
    if src.len() != dst.len() {
        return Err(CastError::LengthMismatch {
            src_len: src.len(),
            dst_len: dst.len(),
        });
    }
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_int_to_int(*in_val, config)?;
    }
    Ok(())
}

/// Convert a slice of float values to float values. Returns early on first error.
///
/// When the configuration allows it (empty scalar_map, nearest-even rounding),
/// uses SIMD-accelerated kernels for supported type pairs (f64->f32).
///
/// # Errors
///
/// Returns `CastError::LengthMismatch` if `src` and `dst` differ in length
/// (checked before anything is written), or the first element's error.
pub fn convert_slice_float_to_float<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &FloatToFloatConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastFloat + CastInto<Dst> + 'static,
    Dst: CastFloat + 'static,
{
    if src.len() != dst.len() {
        return Err(CastError::LengthMismatch {
            src_len: src.len(),
            dst_len: dst.len(),
        });
    }
    // SIMD fast path: empty scalar_map + nearest-even rounding + f64→f32.
    if config.map_entries.is_empty() && config.rounding == RoundingMode::NearestEven {
        use std::any::TypeId;

        if TypeId::of::<Src>() == TypeId::of::<f64>() && TypeId::of::<Dst>() == TypeId::of::<f32>()
        {
            // SAFETY: We just verified Src == f64 and Dst == f32 via TypeId.
            let src_f64: &[f64] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
            let dst_f32: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len()) };
            let error_on_overflow = config.out_of_range != Some(OutOfRangeMode::Clamp);
            if simd::try_f64_to_f32_nearest(src_f64, dst_f32, error_on_overflow)? {
                return Ok(());
            }
        }
    }

    // Dispatch on the rounding mode once, outside the loop: each arm passes a
    // constant, so the per-element mode checks fold away.
    macro_rules! convert_with {
        ($mode:expr) => {
            for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
                *out_slot = float_to_float_with_rounding(*in_val, config, $mode)?;
            }
        };
    }
    match config.rounding {
        RoundingMode::NearestEven => convert_with!(RoundingMode::NearestEven),
        RoundingMode::TowardsZero => convert_with!(RoundingMode::TowardsZero),
        RoundingMode::TowardsPositive => convert_with!(RoundingMode::TowardsPositive),
        RoundingMode::TowardsNegative => convert_with!(RoundingMode::TowardsNegative),
        RoundingMode::NearestAway => convert_with!(RoundingMode::NearestAway),
    }
    Ok(())
}

/// Convert a slice of integer values to float values. Returns early on first error.
///
/// # Errors
///
/// Returns `CastError::LengthMismatch` if `src` and `dst` differ in length
/// (checked before anything is written), or the first element's error.
pub fn convert_slice_int_to_float<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &IntToFloatConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastFloat,
{
    if src.len() != dst.len() {
        return Err(CastError::LengthMismatch {
            src_len: src.len(),
            dst_len: dst.len(),
        });
    }
    // Dispatch on the rounding mode once, outside the loop: each arm passes a
    // constant, so the per-element mode checks fold away.
    macro_rules! convert_with {
        ($mode:expr) => {
            for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
                *out_slot = int_to_float_with_rounding(*in_val, config, $mode)?;
            }
        };
    }
    match config.rounding {
        RoundingMode::NearestEven => convert_with!(RoundingMode::NearestEven),
        RoundingMode::TowardsZero => convert_with!(RoundingMode::TowardsZero),
        RoundingMode::TowardsPositive => convert_with!(RoundingMode::TowardsPositive),
        RoundingMode::TowardsNegative => convert_with!(RoundingMode::TowardsNegative),
        RoundingMode::NearestAway => convert_with!(RoundingMode::NearestAway),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Trait implementations for primitive types
// ---------------------------------------------------------------------------

// ---- CastNum impls ----

macro_rules! impl_cast_num {
    ($($ty:ty),*) => {
        $( impl CastNum for $ty {} )*
    };
}

impl_cast_num!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

// ---- CastInt impls ----

macro_rules! impl_cast_int {
    ($($ty:ty),*) => {
        $( impl CastInt for $ty {} )*
    };
}

impl_cast_int!(i8, i16, i32, i64, u8, u16, u32, u64);

// ---- CastFloat impls ----

macro_rules! impl_cast_float {
    ($($ty:ty),*) => {
        $(
            impl CastFloat for $ty {
                #[inline]
                fn round_with_mode(self, mode: RoundingMode) -> Self {
                    match mode {
                        RoundingMode::NearestEven => <$ty>::round_ties_even(self),
                        RoundingMode::TowardsZero => <$ty>::trunc(self),
                        RoundingMode::TowardsPositive => <$ty>::ceil(self),
                        RoundingMode::TowardsNegative => <$ty>::floor(self),
                        // `round` is exactly half-away-from-zero. (The `floor(|x| + 0.5)`
                        // idiom is not: the addition itself rounds, e.g. for
                        // 0.49999999999999994 or odd integers above 2^52.)
                        RoundingMode::NearestAway => <$ty>::round(self),
                    }
                }
                #[inline]
                fn next_up(self) -> Self { <$ty>::next_up(self) }
                #[inline]
                fn next_down(self) -> Self { <$ty>::next_down(self) }
            }
        )*
    };
}

impl_cast_float!(f32, f64);

// ---- CastInto impls ----
// We need N×N implementations. Use a macro to generate them.

macro_rules! impl_cast_into {
    ($src:ty => $dst:ty, min: $min:expr, max: $max:expr) => {
        impl CastInto<$dst> for $src {
            #[inline]
            fn dst_min() -> Self {
                $min
            }
            #[inline]
            fn dst_max() -> Self {
                $max
            }
            #[inline]
            fn cast_into(self) -> $dst {
                self as $dst
            }
        }
    };
}

// Helper: for float→int, the min/max are the integer bounds as floats.
// For int→int, the min/max are the integer bounds clamped to the source range.
// For any→float, range checking is skipped (target is float, no range check),
// so min/max values don't matter, but we still need the impl.

// -- int→int --
macro_rules! impl_int_to_int {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: {
                let dst_min = <$dst>::MIN as i128;
                let src_min = <$src>::MIN as i128;
                (if dst_min < src_min { src_min } else { dst_min }) as $src
            },
            max: {
                let dst_max = <$dst>::MAX as i128;
                let src_max = <$src>::MAX as i128;
                (if dst_max > src_max { src_max } else { dst_max }) as $src
            }
        );
    };
}

// -- float→int --
macro_rules! impl_float_to_int {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: <$dst>::MIN as $src,
            max: <$dst>::MAX as $src
        );
    };
}

// -- int→float (no range issue — float range always covers integer range) --
macro_rules! impl_int_to_float {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: 0 as $src,  // unused — no range check for int→float
            max: 0 as $src
        );
    };
}

// -- float→float (needs real range bounds for out_of_range support) --
macro_rules! impl_float_to_float {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: <$dst>::MIN as $src,
            max: <$dst>::MAX as $src
        );
    };
}

// Generate all combinations for the 10 types: i8,i16,i32,i64,u8,u16,u32,u64,f32,f64

// int sources → int targets
macro_rules! impl_all_int_to_int {
    ($src:ty => $($dst:ty),*) => {
        $( impl_int_to_int!($src => $dst); )*
    };
}

impl_all_int_to_int!(i8 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(i16 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(i32 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(i64 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(u8 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(u16 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(u32 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_int_to_int!(u64 => i8, i16, i32, i64, u8, u16, u32, u64);

// float sources → int targets
macro_rules! impl_all_float_to_int {
    ($src:ty => $($dst:ty),*) => {
        $( impl_float_to_int!($src => $dst); )*
    };
}

impl_all_float_to_int!(f32 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_all_float_to_int!(f64 => i8, i16, i32, i64, u8, u16, u32, u64);

// int sources → float targets
macro_rules! impl_all_int_to_float {
    ($src:ty => $($dst:ty),*) => {
        $( impl_int_to_float!($src => $dst); )*
    };
}

impl_all_int_to_float!(i8 => f32, f64);
impl_all_int_to_float!(i16 => f32, f64);
impl_all_int_to_float!(i32 => f32, f64);
impl_all_int_to_float!(i64 => f32, f64);
impl_all_int_to_float!(u8 => f32, f64);
impl_all_int_to_float!(u16 => f32, f64);
impl_all_int_to_float!(u32 => f32, f64);
impl_all_int_to_float!(u64 => f32, f64);

// float sources → float targets
macro_rules! impl_all_float_to_float {
    ($src:ty => $($dst:ty),*) => {
        $( impl_float_to_float!($src => $dst); )*
    };
}

impl_all_float_to_float!(f32 => f32, f64);
impl_all_float_to_float!(f64 => f32, f64);

// ---------------------------------------------------------------------------
// float16 support (behind the "float16" feature flag)
// ---------------------------------------------------------------------------

#[cfg(feature = "float16")]
mod float16_impls {
    use super::*;
    use half::f16;

    // ---- CastNum for f16 ----
    impl CastNum for f16 {}

    // ---- CastFloat for f16 ----
    // f16 lacks native arithmetic, so all operations are performed by
    // promoting to f32, computing there, and converting back. This is
    // lossless for inputs that are representable as f16.
    impl CastFloat for f16 {
        #[inline]
        fn round_with_mode(self, mode: RoundingMode) -> Self {
            f16::from_f32(self.to_f32().round_with_mode(mode))
        }
        #[inline]
        fn next_up(self) -> Self {
            // IEEE 754 nextUp implemented directly on f16 bits.
            // Avoids round-tripping through f32 which could skip f16 values.
            let bits = self.to_bits();
            if self.is_nan() {
                self
            } else if bits == 0x8000 {
                // -0.0 → smallest positive subnormal
                f16::from_bits(0x0001)
            } else if (bits & 0x8000) == 0 {
                // Positive: increment bit pattern
                f16::from_bits(bits + 1)
            } else {
                // Negative: decrement bit pattern (moves towards zero)
                f16::from_bits(bits - 1)
            }
        }
        #[inline]
        fn next_down(self) -> Self {
            // nextDown(x) = -nextUp(-x)
            -(-self).next_up()
        }
    }

    // ---- CastInto impls for f16 ----
    // Cannot use the `as` cast macro since Rust has no `as f16` or `as T`
    // from f16. All conversions go through f32.

    // -- f16 → int targets --
    macro_rules! impl_f16_to_int {
        ($($dst:ty),*) => {
            $(
                impl CastInto<$dst> for f16 {
                    #[inline]
                    fn dst_min() -> Self { f16::from_f32(<$dst>::MIN as f32) }
                    #[inline]
                    fn dst_max() -> Self { f16::from_f32(<$dst>::MAX as f32) }
                    #[inline]
                    fn cast_into(self) -> $dst { self.to_f32() as $dst }
                }
            )*
        };
    }
    impl_f16_to_int!(i8, i16, i32, i64, u8, u16, u32, u64);

    // -- int → f16 targets --
    macro_rules! impl_int_to_f16 {
        ($($src:ty),*) => {
            $(
                impl CastInto<f16> for $src {
                    #[inline]
                    fn dst_min() -> Self { 0 as $src } // unused — no range check
                    #[inline]
                    fn dst_max() -> Self { 0 as $src }
                    #[inline]
                    fn cast_into(self) -> f16 { f16::from_f32(self as f32) }
                }
            )*
        };
    }
    impl_int_to_f16!(i8, i16, i32, i64, u8, u16, u32, u64);

    // -- f16 → float targets --
    impl CastInto<f32> for f16 {
        #[inline]
        fn dst_min() -> Self {
            f16::from_f32(f32::MIN)
        }
        #[inline]
        fn dst_max() -> Self {
            f16::from_f32(f32::MAX)
        }
        #[inline]
        fn cast_into(self) -> f32 {
            self.to_f32()
        }
    }

    impl CastInto<f64> for f16 {
        #[inline]
        fn dst_min() -> Self {
            f16::from_f32(f64::MIN as f32)
        }
        #[inline]
        fn dst_max() -> Self {
            f16::from_f32(f64::MAX as f32)
        }
        #[inline]
        fn cast_into(self) -> f64 {
            self.to_f32() as f64
        }
    }

    // -- float → f16 targets --
    impl CastInto<f16> for f32 {
        #[inline]
        fn dst_min() -> Self {
            f16::MIN.to_f32()
        }
        #[inline]
        fn dst_max() -> Self {
            f16::MAX.to_f32()
        }
        #[inline]
        fn cast_into(self) -> f16 {
            f16::from_f32(self)
        }
    }

    impl CastInto<f16> for f64 {
        #[inline]
        fn dst_min() -> Self {
            f16::MIN.to_f32() as f64
        }
        #[inline]
        fn dst_max() -> Self {
            f16::MAX.to_f32() as f64
        }
        #[inline]
        fn cast_into(self) -> f16 {
            // Convert directly: going through f32 would round twice.
            f16::from_f64(self)
        }
    }

    // -- f16 → f16 (identity) --
    impl CastInto<f16> for f16 {
        #[inline]
        fn dst_min() -> Self {
            f16::MIN
        }
        #[inline]
        fn dst_max() -> Self {
            f16::MAX
        }
        #[inline]
        fn cast_into(self) -> f16 {
            self
        }
    }
}

// Re-export f16 when the feature is enabled so consumers can use it
// through this crate without depending on `half` directly.
#[cfg(feature = "float16")]
pub use half::f16;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build FloatToIntConfig
    fn f2i_cfg<Src: CastFloat, Dst: CastInt>(
        map: Vec<MapEntry<Src, Dst>>,
        rounding: RoundingMode,
        oor: Option<OutOfRangeMode>,
    ) -> FloatToIntConfig<Src, Dst> {
        FloatToIntConfig {
            map_entries: map,
            rounding,
            out_of_range: oor,
        }
    }

    // Helper to build IntToIntConfig
    fn i2i_cfg<Src: CastInt, Dst: CastInt>(
        map: Vec<MapEntry<Src, Dst>>,
        oor: Option<OutOfRangeMode>,
    ) -> IntToIntConfig<Src, Dst> {
        IntToIntConfig {
            map_entries: map,
            out_of_range: oor,
        }
    }

    #[test]
    fn test_float64_to_uint8_basic() {
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(42.0_f64, &c).unwrap(), 42_u8);
    }

    #[test]
    fn test_float64_to_uint8_rounding() {
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        // 2.5 rounds to 2 (banker's rounding)
        assert_eq!(convert_float_to_int(2.5_f64, &c).unwrap(), 2_u8);
        // 3.5 rounds to 4
        assert_eq!(convert_float_to_int(3.5_f64, &c).unwrap(), 4_u8);
    }

    #[test]
    fn test_float64_to_uint8_clamp() {
        let c = f2i_cfg::<f64, u8>(
            vec![],
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Clamp),
        );
        assert_eq!(convert_float_to_int(300.0_f64, &c).unwrap(), 255_u8);
        assert_eq!(convert_float_to_int(-10.0_f64, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_float64_to_int8_wrap() {
        let c = f2i_cfg::<f64, i8>(
            vec![],
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Wrap),
        );
        // 200 wraps: (200 - (-128)) % 256 + (-128) = 328 % 256 + (-128) = 72 + (-128) = -56
        assert_eq!(convert_float_to_int(200.0_f64, &c).unwrap(), -56_i8);
    }

    #[test]
    fn test_int32_to_int8_wrap() {
        let c = i2i_cfg::<i32, i8>(vec![], Some(OutOfRangeMode::Wrap));
        // 200_i32 as i8 = -56 (bit truncation = modular wrap)
        assert_eq!(convert_int_to_int(200_i32, &c).unwrap(), -56_i8);
        assert_eq!(convert_int_to_int(-200_i32, &c).unwrap(), 56_i8);
    }

    #[test]
    fn test_int32_to_uint8_wrap() {
        let c = i2i_cfg::<i32, u8>(vec![], Some(OutOfRangeMode::Wrap));
        assert_eq!(convert_int_to_int(300_i32, &c).unwrap(), 44_u8);
        assert_eq!(convert_int_to_int(-1_i32, &c).unwrap(), 255_u8);
    }

    #[test]
    fn test_out_of_range_error() {
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(300.0_f64, &c).is_err());
    }

    #[test]
    fn test_nan_to_int_error() {
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(f64::NAN, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_error() {
        // Without out_of_range, Inf is caught by the range check (OutOfRange error)
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(f64::INFINITY, &c).is_err());
        assert!(convert_float_to_int(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_clamp_errors() {
        // Integers cannot represent ±Inf, so the spec requires an error even
        // with clamp (only scalar_map can map them).
        let c = f2i_cfg::<f64, u8>(
            vec![],
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Clamp),
        );
        for inf in [f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                convert_float_to_int(inf, &c),
                Err(CastError::NanOrInf { .. })
            ));
            let mut dst = [0_u8];
            assert!(matches!(
                convert_slice_float_to_int(&[inf], &mut dst, &c),
                Err(CastError::NanOrInf { .. })
            ));
        }
    }

    #[test]
    fn test_scalar_map_inf_to_int() {
        let entries = vec![MapEntry {
            src: f64::INFINITY,
            tgt: 255_u8,
        }];
        let c = f2i_cfg(
            entries,
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Clamp),
        );
        assert_eq!(convert_float_to_int(f64::INFINITY, &c).unwrap(), 255_u8);
    }

    #[test]
    fn test_inf_to_int_wrap_errors() {
        // With wrap, Inf is rejected: integers cannot represent it
        let c = f2i_cfg::<f64, i8>(
            vec![],
            RoundingMode::NearestEven,
            Some(OutOfRangeMode::Wrap),
        );
        assert!(convert_float_to_int(f64::INFINITY, &c).is_err());
        assert!(convert_float_to_int(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_scalar_map_nan() {
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 0_u8,
        }];
        let c = f2i_cfg(entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(f64::NAN, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_scalar_map_nan_payloads() {
        // Any NaN should match a NaN map entry, regardless of payload bits.
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 42_u8,
        }];
        let c = f2i_cfg(entries, RoundingMode::NearestEven, None);

        // Standard NaN
        assert_eq!(convert_float_to_int(f64::NAN, &c).unwrap(), 42_u8);

        // NaN with custom payload (different bit pattern, still NaN)
        let nan_payload = f64::from_bits(0x7FF8_0000_0000_0001);
        assert!(nan_payload.is_nan());
        assert_eq!(convert_float_to_int(nan_payload, &c).unwrap(), 42_u8);

        // Negative NaN (sign bit set)
        let neg_nan = f64::from_bits(0xFFF8_0000_0000_0000);
        assert!(neg_nan.is_nan());
        assert_eq!(convert_float_to_int(neg_nan, &c).unwrap(), 42_u8);

        // Signaling NaN (quiet bit clear, payload nonzero)
        let snan = f64::from_bits(0x7FF0_0000_0000_0001);
        assert!(snan.is_nan());
        assert_eq!(convert_float_to_int(snan, &c).unwrap(), 42_u8);
    }

    #[test]
    fn test_scalar_map_exact() {
        let entries = vec![MapEntry {
            src: 42.0_f64,
            tgt: 99_u8,
        }];
        let c = f2i_cfg(entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(42.0_f64, &c).unwrap(), 99_u8);
        // Non-matching value goes through normal path
        assert_eq!(convert_float_to_int(10.0_f64, &c).unwrap(), 10_u8);
    }

    #[test]
    fn test_int32_to_uint8_range_check() {
        let c = i2i_cfg::<i32, u8>(vec![], None);
        assert_eq!(convert_int_to_int(100_i32, &c).unwrap(), 100_u8);
        assert!(convert_int_to_int(300_i32, &c).is_err());
        assert!(convert_int_to_int(-1_i32, &c).is_err());
    }

    #[test]
    fn test_int32_to_float64() {
        let c = IntToFloatConfig::<i32, f64> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert_eq!(convert_int_to_float(42_i32, &c).unwrap(), 42.0_f64);
    }

    #[test]
    fn test_convert_slice_basic() {
        let src = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        convert_slice_float_to_int(&src, &mut dst, &c).unwrap();
        assert_eq!(dst, [1, 2, 3, 4]);
    }

    #[test]
    fn test_convert_slice_early_termination() {
        let src = [1.0_f64, 2.0, 300.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = f2i_cfg::<f64, u8>(vec![], RoundingMode::NearestEven, None);
        let result = convert_slice_float_to_int(&src, &mut dst, &c);
        assert!(result.is_err());
        // First two should have been written
        assert_eq!(dst[0], 1);
        assert_eq!(dst[1], 2);
    }

    #[test]
    fn test_all_rounding_modes() {
        // towards-zero
        let c = f2i_cfg::<f64, i8>(vec![], RoundingMode::TowardsZero, None);
        assert_eq!(convert_float_to_int(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_float_to_int(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-positive
        let c = f2i_cfg::<f64, i8>(vec![], RoundingMode::TowardsPositive, None);
        assert_eq!(convert_float_to_int(2.1_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_float_to_int(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-negative
        let c = f2i_cfg::<f64, i8>(vec![], RoundingMode::TowardsNegative, None);
        assert_eq!(convert_float_to_int(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_float_to_int(-2.1_f64, &c).unwrap(), -3_i8);

        // nearest-away
        let c = f2i_cfg::<f64, i8>(vec![], RoundingMode::NearestAway, None);
        assert_eq!(convert_float_to_int(2.5_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_float_to_int(-2.5_f64, &c).unwrap(), -3_i8);
    }

    #[test]
    fn test_int64_to_int32_clamp() {
        let c = i2i_cfg::<i64, i32>(vec![], Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_int_to_int(i64::MAX, &c).unwrap(), i32::MAX);
        assert_eq!(convert_int_to_int(i64::MIN, &c).unwrap(), i32::MIN);
    }

    #[test]
    fn test_float32_to_float64() {
        let c = FloatToFloatConfig::<f32, f64> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert_eq!(
            convert_float_to_float(1.25_f32, &c).unwrap(),
            1.25_f32 as f64
        );
    }

    // -- float→float rounding tests --

    #[test]
    fn test_float64_to_float32_nearest_even() {
        // A value that is exactly between two f32 representable values
        // should round to even (the default `as` behavior).
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        // 1.0 is exactly representable — no rounding
        assert_eq!(convert_float_to_float(1.0_f64, &c).unwrap(), 1.0_f32);
    }

    #[test]
    fn test_float64_to_float32_towards_zero() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsZero,
            out_of_range: None,
        };
        // Pick a value not exactly representable in f32.
        // f32 has ~7 decimal digits of precision.
        let val: f64 = 1.0000001_f64; // slightly above 1.0
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards zero: result should be <= val (since val > 0)
        assert!(
            to_f64_lossy(result) <= val,
            "towards-zero: {result} > {val}"
        );
        assert!(to_f64_lossy(result) > 0.0);

        // Negative: towards zero means closer to 0 (larger, i.e. less negative)
        let val_neg: f64 = -1.0000001_f64;
        let result_neg = convert_float_to_float(val_neg, &c).unwrap();
        assert!(
            to_f64_lossy(result_neg) >= val_neg,
            "towards-zero negative: {result_neg} < {val_neg}"
        );
    }

    #[test]
    fn test_float64_to_float32_towards_positive() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsPositive,
            out_of_range: None,
        };
        let val: f64 = 1.0000001_f64;
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards positive: result should be >= val
        assert!(
            to_f64_lossy(result) >= val,
            "towards-positive: {result} < {val}"
        );
    }

    #[test]
    fn test_float64_to_float32_towards_negative() {
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsNegative,
            out_of_range: None,
        };
        let val: f64 = 1.0000001_f64;
        let result = convert_float_to_float(val, &c).unwrap();
        // Towards negative: result should be <= val
        assert!(
            to_f64_lossy(result) <= val,
            "towards-negative: {result} > {val}"
        );
    }

    // -- float→float out-of-range tests --

    #[test]
    fn test_float64_to_float32_overflow_error() {
        // A value exceeding f32::MAX should error when no out_of_range is set
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        let val = f64::from(f32::MAX) * 2.0;
        assert!(convert_float_to_float(val, &c).is_err());
    }

    #[test]
    fn test_float64_to_float32_overflow_clamp() {
        // With clamp, overflow maps to ±Infinity
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: Some(OutOfRangeMode::Clamp),
        };
        let val = f64::from(f32::MAX) * 2.0;
        let result = convert_float_to_float(val, &c).unwrap();
        assert!(result.is_infinite());
        assert!(result > 0.0_f32);

        let val_neg = -f64::from(f32::MAX) * 2.0;
        let result_neg = convert_float_to_float(val_neg, &c).unwrap();
        assert!(result_neg.is_infinite());
        assert!(result_neg < 0.0_f32);
    }

    #[test]
    fn test_float64_to_float32_inf_propagates() {
        // ±Inf from source should propagate to target (not trigger out-of-range error)
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert!(convert_float_to_float(f64::INFINITY, &c)
            .unwrap()
            .is_infinite());
        assert!(convert_float_to_float(f64::NEG_INFINITY, &c)
            .unwrap()
            .is_infinite());
    }

    #[test]
    fn test_float64_to_float32_nan_propagates() {
        // NaN should propagate without error
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        assert!(convert_float_to_float(f64::NAN, &c).unwrap().is_nan());
    }

    #[test]
    fn test_float64_to_float32_signed_zero() {
        // Signed zero must be preserved
        let c = FloatToFloatConfig::<f64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        let result = convert_float_to_float(-0.0_f64, &c).unwrap();
        assert!(result == 0.0_f32);
        assert!(result.is_sign_negative());
    }

    // -- int→float rounding tests --

    #[test]
    fn test_int64_to_float32_rounding() {
        // i64 values near 2^24 can't all be exactly represented in f32
        // (f32 has 24 bits of mantissa). Pick a value that requires rounding.
        let val: i64 = (1_i64 << 24) + 1; // 16777217 — not exactly representable in f32

        // nearest-even: default behavior
        let c_ne = IntToFloatConfig::<i64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::NearestEven,
            out_of_range: None,
        };
        let result_ne = convert_int_to_float(val, &c_ne).unwrap();
        assert_eq!(result_ne, val as f32); // same as Rust `as`

        // towards-positive: result should be >= val
        let c_pos = IntToFloatConfig::<i64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsPositive,
            out_of_range: None,
        };
        let result_pos = convert_int_to_float(val, &c_pos).unwrap();
        assert!(
            to_f64_lossy(result_pos) >= val as f64,
            "towards-positive: {result_pos} < {val}"
        );

        // towards-zero: result should be <= val (since val > 0)
        let c_tz = IntToFloatConfig::<i64, f32> {
            map_entries: vec![],
            rounding: RoundingMode::TowardsZero,
            out_of_range: None,
        };
        let result_tz = convert_int_to_float(val, &c_tz).unwrap();
        assert!(
            to_f64_lossy(result_tz) <= val as f64,
            "towards-zero: {result_tz} > {val}"
        );
    }

    // ---- Regression tests for exactness bugs found by comparison with zarrs ----

    fn f2f<Src, Dst>(
        val: Src,
        rounding: RoundingMode,
        oor: Option<OutOfRangeMode>,
    ) -> Result<Dst, CastError>
    where
        Src: CastFloat + CastInto<Dst> + 'static,
        Dst: CastFloat + 'static,
    {
        let c = FloatToFloatConfig {
            map_entries: vec![],
            rounding,
            out_of_range: oor,
        };
        // Through the slice API so SIMD fast paths are covered too.
        let mut dst = [Dst::zero()];
        convert_slice_float_to_float(&[val], &mut dst, &c).map(|()| dst[0])
    }

    fn i2f<Src, Dst>(
        val: Src,
        rounding: RoundingMode,
        oor: Option<OutOfRangeMode>,
    ) -> Result<Dst, CastError>
    where
        Src: CastInt + CastInto<Dst>,
        Dst: CastFloat,
    {
        let c = IntToFloatConfig {
            map_entries: vec![],
            rounding,
            out_of_range: oor,
        };
        convert_int_to_float(val, &c)
    }

    fn f2i<Src, Dst>(
        val: Src,
        rounding: RoundingMode,
        oor: Option<OutOfRangeMode>,
    ) -> Result<Dst, CastError>
    where
        Src: CastFloat + CastInto<Dst> + 'static,
        Dst: CastInt + 'static,
    {
        let c = f2i_cfg(vec![], rounding, oor);
        let mut dst = [Dst::from_u8(0).expect("0 fits every integer type")];
        convert_slice_float_to_int(&[val], &mut dst, &c).map(|()| dst[0])
    }

    /// Values at and beyond the integer limits are handled exactly, even where
    /// the limit itself is not representable in the source float type.
    #[test]
    fn test_float_to_int_exact_limits() {
        use OutOfRangeMode::{Clamp, Wrap};
        use RoundingMode::NearestEven as NE;
        let two_31 = 2_147_483_648.0;
        // 2^63 - 2^10: the largest f64 below 2^63, in range for i64.
        let below_2_63 = 9_223_372_036_854_774_784.0_f64;
        assert_eq!(
            f2i::<f64, i64>(below_2_63, NE, None).unwrap(),
            i64::MAX - 1023
        );
        assert_eq!(f2i::<f32, i32>(-two_31 as f32, NE, None).unwrap(), i32::MIN);
        // Exactly 2^k, one past the maximum:
        assert_eq!(
            f2i::<f32, i32>(two_31 as f32, NE, Some(Wrap)).unwrap(),
            i32::MIN
        );
        assert_eq!(
            f2i::<f32, i32>(two_31 as f32, NE, Some(Clamp)).unwrap(),
            i32::MAX
        );
        assert_eq!(f2i::<f64, u64>(TWO_POW_64, NE, Some(Wrap)).unwrap(), 0);
        assert_eq!(
            f2i::<f64, u64>(TWO_POW_64, NE, Some(Clamp)).unwrap(),
            u64::MAX
        );
        assert_eq!(
            f2i::<f64, i64>(TWO_POW_63, NE, Some(Wrap)).unwrap(),
            i64::MIN
        );
        // Wrap of large magnitudes, where `val - lo` would round:
        assert_eq!(
            f2i::<f64, i64>(TWO_POW_63 + 2048.0, NE, Some(Wrap)).unwrap(),
            i64::MIN + 2048
        );
        assert_eq!(
            f2i::<f32, i32>(3_000_000_256.0, NE, Some(Wrap)).unwrap(),
            (3_000_000_256_i64 - (1 << 32)) as i32
        );
        assert_eq!(f2i::<f64, u32>(-1.0, NE, Some(Wrap)).unwrap(), u32::MAX);
        assert_eq!(f2i::<f64, u64>(-1.0, NE, Some(Wrap)).unwrap(), u64::MAX);
        assert_eq!(
            f2i::<f64, u8>(-TWO_POW_64 - 4096.0, NE, Some(Wrap)).unwrap(),
            0
        );
        assert_eq!(
            f2i::<f64, i16>(1e20, NE, Some(Wrap)).unwrap(),
            1e20_f64 as i128 as i16
        );
        // The spec's int16 wrap examples, from a float source:
        assert_eq!(f2i::<f64, i16>(32768.0, NE, Some(Wrap)).unwrap(), -32768);
        assert_eq!(f2i::<f64, i16>(32769.0, NE, Some(Wrap)).unwrap(), -32767);
        assert_eq!(f2i::<f64, i16>(-32769.0, NE, Some(Wrap)).unwrap(), 32767);
    }

    /// Each one-past-the-maximum value is an error when `out_of_range` is unset.
    #[test]
    fn test_float_to_int_one_past_max_errors() {
        use RoundingMode::NearestEven as NE;
        let is_oor = |r: Result<(), CastError>| matches!(r, Err(CastError::OutOfRange { .. }));
        assert!(is_oor(f2i::<f32, i32>(2_147_483_648.0, NE, None).map(drop)));
        assert!(is_oor(f2i::<f32, u32>(4_294_967_296.0, NE, None).map(drop)));
        assert!(is_oor(f2i::<f64, i64>(TWO_POW_63, NE, None).map(drop)));
        assert!(is_oor(f2i::<f64, u64>(TWO_POW_64, NE, None).map(drop)));
        assert!(is_oor(
            f2i::<f32, u64>(TWO_POW_64 as f32, NE, None).map(drop)
        ));
    }

    /// nearest-away rounds exactly, including where `floor(|x| + 0.5)` fails.
    #[test]
    fn test_float_to_int_nearest_away_exact() {
        use RoundingMode::NearestAway as NA;
        assert_eq!(f2i::<f64, i8>(0.49999999999999994, NA, None).unwrap(), 0);
        assert_eq!(f2i::<f64, i8>(-0.49999999999999994, NA, None).unwrap(), 0);
        assert_eq!(f2i::<f64, i8>(-2.5, NA, None).unwrap(), -3);
        assert_eq!(
            f2i::<f64, i64>(4_503_599_627_370_497.0, NA, None).unwrap(),
            (1 << 52) + 1
        );
        assert_eq!(
            f2i::<f32, i32>(8_388_609.0, NA, None).unwrap(),
            (1 << 23) + 1
        );
    }

    /// Integers wider than the target mantissa round once, from their exact value.
    #[test]
    fn test_int_to_float_rounds_once() {
        use RoundingMode::*;
        // u64: 2^63 + 1 sits just above an f32 value (and rounds to it in f64).
        let just_above: u64 = (1 << 63) + 1;
        let lower = 0x5f00_0000_u32; // 2^63 as f32
        let upper = 0x5f00_0001_u32; // next f32 above 2^63
        let bits = |mode| i2f::<u64, f32>(just_above, mode, None).unwrap().to_bits();
        assert_eq!(bits(NearestEven), lower);
        assert_eq!(bits(TowardsZero), lower);
        assert_eq!(bits(TowardsNegative), lower);
        assert_eq!(bits(TowardsPositive), upper);
        assert_eq!(bits(NearestAway), lower);
        // One below an f32 midpoint: f64 rounds it onto the midpoint, but it is
        // not a tie.
        let halfway: u64 = (1 << 63) + (1 << 39);
        assert_eq!(
            i2f::<u64, f32>(halfway - 1, NearestAway, None)
                .unwrap()
                .to_bits(),
            lower
        );
        assert_eq!(
            i2f::<u64, f32>(halfway, NearestAway, None)
                .unwrap()
                .to_bits(),
            upper
        );
        assert_eq!(
            i2f::<u64, f32>(halfway, NearestEven, None)
                .unwrap()
                .to_bits(),
            lower
        );
        // i64 → f64 beyond 2^53, where f64 itself cannot hold the source:
        let p53 = 1_i64 << 53;
        assert_eq!(
            i2f::<i64, f64>(p53 + 1, TowardsPositive, None).unwrap(),
            (p53 + 2) as f64
        );
        assert_eq!(
            i2f::<i64, f64>(p53 + 3, TowardsZero, None).unwrap(),
            (p53 + 2) as f64
        );
        assert_eq!(
            i2f::<i64, f64>(-p53 - 3, TowardsZero, None).unwrap(),
            (-p53 - 2) as f64
        );
        assert_eq!(
            i2f::<i64, f64>(-p53 - 1, TowardsNegative, None).unwrap(),
            (-p53 - 2) as f64
        );
        assert_eq!(
            i2f::<i64, f64>(p53 + 1, NearestAway, None).unwrap(),
            (p53 + 2) as f64
        );
        // Extremes, including a carry to 2^64:
        assert_eq!(
            i2f::<u64, f64>(u64::MAX, TowardsPositive, None).unwrap(),
            TWO_POW_64
        );
        assert_eq!(
            i2f::<i64, f32>(i64::MIN, TowardsZero, None).unwrap(),
            -TWO_POW_63 as f32
        );
        // Exact values are untouched by every mode:
        for mode in [
            NearestEven,
            TowardsZero,
            TowardsPositive,
            TowardsNegative,
            NearestAway,
        ] {
            assert_eq!(
                i2f::<i32, f32>(-16_777_216, mode, None).unwrap(),
                -16_777_216.0
            );
        }
    }

    /// A value whose *rounded* magnitude exceeds the finite range is out of
    /// range in every rounding mode; one that rounds onto ±MAX is not.
    #[test]
    fn test_float_to_float_overflow_after_rounding() {
        use OutOfRangeMode::Clamp;
        use RoundingMode::*;
        let max = f64::from(f32::MAX);
        let ulp = max - f64::from(f32::MAX.next_down());
        for mode in [TowardsZero, TowardsNegative] {
            assert_eq!(
                f2f::<f64, f32>(1e39, mode, Some(Clamp)).unwrap(),
                f32::INFINITY
            );
        }
        for mode in [TowardsZero, TowardsPositive] {
            assert_eq!(
                f2f::<f64, f32>(-1e39, mode, Some(Clamp)).unwrap(),
                f32::NEG_INFINITY
            );
        }
        // Between MAX + ulp/2 (nearest-even overflows) and MAX + ulp: towards
        // zero lands on MAX, which is in range.
        let above_max = max + 0.75 * ulp;
        assert_eq!(
            f2f::<f64, f32>(above_max, TowardsZero, None).unwrap(),
            f32::MAX
        );
        assert_eq!(
            f2f::<f64, f32>(-above_max, TowardsPositive, None).unwrap(),
            f32::MIN
        );
        assert_eq!(
            f2f::<f64, f32>(above_max, NearestEven, Some(Clamp)).unwrap(),
            f32::INFINITY
        );
        // Slightly above MAX: nearest rounds to MAX, towards-positive overflows.
        let slightly_above = max + 0.25 * ulp;
        assert_eq!(
            f2f::<f64, f32>(slightly_above, NearestEven, None).unwrap(),
            f32::MAX
        );
        assert_eq!(
            f2f::<f64, f32>(slightly_above, NearestAway, None).unwrap(),
            f32::MAX
        );
        assert_eq!(
            f2f::<f64, f32>(slightly_above, TowardsPositive, Some(Clamp)).unwrap(),
            f32::INFINITY
        );
    }

    /// Overflow after directed rounding is an error when `out_of_range` is unset.
    #[test]
    fn test_float_to_float_directed_overflow_errors() {
        use RoundingMode::*;
        for mode in [TowardsZero, TowardsNegative, NearestAway, TowardsPositive] {
            assert!(matches!(
                f2f::<f64, f32>(1e39, mode, None),
                Err(CastError::FloatOutOfRange { .. })
            ));
        }
        assert!(matches!(
            f2f::<f64, f32>(
                f64::from(f32::MAX) * 1.5,
                TowardsZero,
                Some(OutOfRangeMode::Wrap)
            ),
            Err(CastError::FloatOutOfRange { .. })
        ));
    }

    /// Out-of-range errors report the original input and the target type's
    /// exact bounds, even where the bound is not representable in the source.
    #[test]
    fn test_out_of_range_error_reports_exact_bounds() {
        use RoundingMode::{NearestEven, TowardsNegative};
        let err = f2i::<f32, i32>(2_147_483_648.0, NearestEven, None).unwrap_err();
        assert!(matches!(
            err,
            CastError::OutOfRange { value, lo, hi }
                if value == 2_147_483_648.0
                    && lo == i128::from(i32::MIN)
                    && hi == i128::from(i32::MAX)
        ));
        assert_eq!(
            err.to_string(),
            "Value 2147483648 out of range [-2147483648, 2147483647]. \
             Set out_of_range='clamp' or out_of_range='wrap' to handle this."
        );
        // The unrounded input is reported.
        let err = f2i::<f64, u8>(-0.25, TowardsNegative, None).unwrap_err();
        assert!(matches!(err, CastError::OutOfRange { value, lo: 0, hi: 255 } if value == -0.25));
        let err = f2i::<f64, u64>(TWO_POW_64, NearestEven, None).unwrap_err();
        assert!(matches!(err, CastError::OutOfRange { hi, .. } if hi == i128::from(u64::MAX)));
        // Int→int reports the target's bounds, not the source-clipped ones.
        let c = i2i_cfg::<u8, i8>(vec![], None);
        let err = convert_int_to_int(200_u8, &c).unwrap_err();
        assert!(matches!(
            err,
            CastError::OutOfRange {
                lo: -128,
                hi: 127,
                ..
            }
        ));
    }

    /// Every slice function rejects mismatched lengths before writing,
    /// including configurations that take a SIMD fast path.
    #[test]
    fn test_slice_length_mismatch_errors() {
        let is_mismatch = |r: Result<(), CastError>, src_len: usize, dst_len: usize| {
            matches!(r, Err(CastError::LengthMismatch { src_len: s, dst_len: d })
                if s == src_len && d == dst_len)
        };
        let src_f64 = [7.0_f64; 16];
        let src_i32 = [7_i32; 16];
        for dst_len in [0, 4, 17] {
            let mut dst_u8 = vec![0_u8; dst_len];
            let mut dst_i32 = vec![0_i32; dst_len];
            let mut dst_f32 = vec![0_f32; dst_len];
            let clamp = f2i_cfg::<f64, u8>(
                vec![],
                RoundingMode::NearestEven,
                Some(OutOfRangeMode::Clamp),
            );
            let check = f2i_cfg::<f64, i32>(vec![], RoundingMode::NearestEven, None);
            let f2f_cfg = FloatToFloatConfig::<f64, f32> {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let i2f_cfg = IntToFloatConfig::<i32, f32> {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let i2i = i2i_cfg::<i32, u8>(vec![], Some(OutOfRangeMode::Wrap));
            assert!(is_mismatch(
                convert_slice_float_to_int(&src_f64, &mut dst_u8, &clamp),
                16,
                dst_len
            ));
            assert!(is_mismatch(
                convert_slice_float_to_int(&src_f64, &mut dst_i32, &check),
                16,
                dst_len
            ));
            assert!(is_mismatch(
                convert_slice_float_to_float(&src_f64, &mut dst_f32, &f2f_cfg),
                16,
                dst_len
            ));
            assert!(is_mismatch(
                convert_slice_int_to_float(&src_i32, &mut dst_f32, &i2f_cfg),
                16,
                dst_len
            ));
            assert!(is_mismatch(
                convert_slice_int_to_int(&src_i32, &mut dst_u8, &i2i),
                16,
                dst_len
            ));
            // Nothing was written.
            assert!(dst_u8.iter().all(|&v| v == 0) && dst_f32.iter().all(|&v| v == 0.0));
        }
    }

    /// The slice API (SIMD fast paths) returns the same values and the same
    /// first error as converting element by element, for every mode a fast
    /// path accepts. Lengths cover vector bodies and scalar tails.
    #[test]
    fn test_slice_matches_per_element() {
        use OutOfRangeMode::Clamp;
        use RoundingMode::*;
        let specials = [
            0.5,
            1.5,
            2.5,
            -0.5,
            -2.5,
            254.5,
            255.5,
            -0.0,
            3e9,
            -3e9,
            2_147_483_647.5,
            -2_147_483_648.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1e300,
            7.0,
        ];
        // The slice result, `Ok(values)` or the first error, must equal the
        // per-element one exactly (compared via Debug, since CastError has no
        // PartialEq).
        fn assert_matches<S, D>(src: &[S], rounding: RoundingMode, oor: Option<OutOfRangeMode>)
        where
            S: CastFloat + CastInto<D> + 'static,
            D: CastInt + 'static,
        {
            let cfg = f2i_cfg::<S, D>(vec![], rounding, oor);
            let mut dst = vec![D::from_u8(0).expect("0 fits every integer type"); src.len()];
            let slice = convert_slice_float_to_int(src, &mut dst, &cfg).map(|()| dst);
            let expected: Result<Vec<D>, _> =
                src.iter().map(|&v| convert_float_to_int(v, &cfg)).collect();
            assert_eq!(
                format!("{slice:?}"),
                format!("{expected:?}"),
                "{src:?} {rounding:?} {oor:?}"
            );
        }
        // Stride 1 places each special next to its neighbour (e.g. an
        // out-of-range value just before ±Inf); stride 7 mixes them.
        for len in 1..=35 {
            for start in 0..specials.len() {
                for stride in [1, 7] {
                    let src: Vec<f64> = (0..len)
                        .map(|i| specials[(start + i * stride) % specials.len()])
                        .collect();
                    let src_f32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
                    for rounding in [NearestEven, TowardsZero, TowardsPositive, TowardsNegative] {
                        for oor in [None, Some(Clamp)] {
                            assert_matches::<f64, i32>(&src, rounding, oor);
                            assert_matches::<f64, u8>(&src, rounding, oor);
                            assert_matches::<f32, u8>(&src_f32, rounding, oor);
                        }
                    }
                }
            }
        }
    }

    // ---- float16 tests ----

    #[cfg(feature = "float16")]
    mod float16_tests {
        use super::*;
        use half::f16;

        #[test]
        fn test_f16_to_u8_basic() {
            let c = f2i_cfg::<f16, u8>(vec![], RoundingMode::NearestEven, None);
            assert_eq!(
                convert_float_to_int(f16::from_f32(42.0), &c).unwrap(),
                42_u8
            );
        }

        #[test]
        fn test_f16_to_u8_clamp() {
            let c = f2i_cfg::<f16, u8>(
                vec![],
                RoundingMode::NearestEven,
                Some(OutOfRangeMode::Clamp),
            );
            assert_eq!(
                convert_float_to_int(f16::from_f32(300.0), &c).unwrap(),
                255_u8
            );
            assert_eq!(
                convert_float_to_int(f16::from_f32(-10.0), &c).unwrap(),
                0_u8
            );
        }

        #[test]
        fn test_f16_nan_error() {
            let c = f2i_cfg::<f16, u8>(vec![], RoundingMode::NearestEven, None);
            assert!(convert_float_to_int(f16::NAN, &c).is_err());
        }

        #[test]
        fn test_f16_scalar_map_nan() {
            let c = f2i_cfg::<f16, u8>(
                vec![MapEntry {
                    src: f16::NAN,
                    tgt: 0_u8,
                }],
                RoundingMode::NearestEven,
                None,
            );
            assert_eq!(convert_float_to_int(f16::NAN, &c).unwrap(), 0_u8);
        }

        #[test]
        fn test_f32_to_f16_basic() {
            let c = FloatToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let result: f16 = convert_float_to_float(1.5_f32, &c).unwrap();
            assert_eq!(result, f16::from_f32(1.5));
        }

        #[test]
        fn test_f64_to_f16_overflow_clamp() {
            let c = FloatToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: Some(OutOfRangeMode::Clamp),
            };
            let result: f16 = convert_float_to_float(1.0e10_f64, &c).unwrap();
            assert!(result.is_infinite());
        }

        #[test]
        fn test_f16_to_f32_lossless() {
            let c = FloatToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let val = f16::from_f32(1.5);
            let result: f32 = convert_float_to_float(val, &c).unwrap();
            assert_eq!(result, 1.5_f32);
        }

        #[test]
        fn test_f16_nan_propagates() {
            let c = FloatToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let result: f32 = convert_float_to_float(f16::NAN, &c).unwrap();
            assert!(result.is_nan());
        }

        #[test]
        fn test_i32_to_f16() {
            let c = IntToFloatConfig {
                map_entries: vec![],
                rounding: RoundingMode::NearestEven,
                out_of_range: None,
            };
            let result: f16 = convert_int_to_float(42_i32, &c).unwrap();
            assert_eq!(result, f16::from_f32(42.0));
        }

        #[test]
        fn test_f16_to_i32() {
            let c = f2i_cfg::<f16, i32>(vec![], RoundingMode::NearestEven, None);
            assert_eq!(
                convert_float_to_int(f16::from_f32(-7.0), &c).unwrap(),
                -7_i32
            );
        }

        #[test]
        fn test_f16_slice_conversion() {
            let src = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
            let mut dst = [0_u8; 3];
            let c = f2i_cfg::<f16, u8>(vec![], RoundingMode::NearestEven, None);
            convert_slice_float_to_int(&src, &mut dst, &c).unwrap();
            assert_eq!(dst, [1, 2, 3]);
        }

        /// f64 → f16 rounds once, directly from the f64 value.
        #[test]
        fn test_f64_to_f16_rounds_once() {
            // Just above a tie between two f16 values; f32 would round it onto
            // the tie, and ties-to-even would then pick the lower value.
            let val = 1.0 + 2f64.powi(-11) + 2f64.powi(-40);
            let expected = f16::from_bits(f16::ONE.to_bits() + 1);
            let result: f16 = f2f(val, RoundingMode::NearestEven, None).unwrap();
            assert_eq!(result, expected);
        }

        /// Integers beyond f16's finite range clamp to ±Infinity.
        #[test]
        fn test_int_to_f16_overflow_clamp() {
            use RoundingMode::*;
            let clamp = Some(OutOfRangeMode::Clamp);
            assert_eq!(
                i2f::<i32, f16>(70_000, NearestEven, clamp).unwrap(),
                f16::INFINITY
            );
            assert_eq!(
                i2f::<i32, f16>(-70_000, TowardsZero, clamp).unwrap(),
                f16::NEG_INFINITY
            );
            // 65519 rounds to MAX with nearest-even, but up past it towards +Inf.
            assert_eq!(
                i2f::<u16, f16>(65_519, NearestEven, None).unwrap(),
                f16::MAX
            );
            assert_eq!(
                i2f::<u16, f16>(65_519, TowardsPositive, clamp).unwrap(),
                f16::INFINITY
            );
            // 65535 overflows nearest-even, but towards zero lands on MAX.
            assert_eq!(
                i2f::<u16, f16>(65_535, TowardsZero, None).unwrap(),
                f16::MAX
            );
        }

        /// Integers beyond f16's finite range are an error without `out_of_range`.
        #[test]
        fn test_int_to_f16_overflow_errors() {
            for mode in [RoundingMode::NearestEven, RoundingMode::TowardsZero] {
                assert!(matches!(
                    i2f::<i32, f16>(70_000, mode, None),
                    Err(CastError::FloatOutOfRange { .. })
                ));
            }
        }

        #[test]
        fn test_f16_next_up_next_down() {
            let one = f16::from_f32(1.0);
            let up = one.next_up();
            let down = one.next_down();
            assert!(up > one);
            assert!(down < one);
            assert_eq!(up.next_down(), one);
            assert_eq!(down.next_up(), one);
        }
    }
}
