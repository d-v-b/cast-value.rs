//! zarr-cast-value-core: Pure Rust implementation of the cast_value codec's
//! per-element conversion logic.
//!
//! This crate is independent of Python/PyO3 and can be used by any Rust consumer
//! (e.g. zarrs, zarr-python bindings).

use std::ops::{Add, Sub};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during element conversion.
#[derive(Debug, Clone)]
pub enum CastError {
    /// A NaN or Infinity value was encountered when casting to an integer type.
    NanOrInf { value: f64 },
    /// A value is out of range for the target type and no out_of_range mode was set.
    OutOfRange { value: f64, lo: f64, hi: f64 },
}

impl std::fmt::Display for CastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CastError::NanOrInf { value } => {
                write!(f, "Cannot cast {value} to integer type without scalar_map")
            }
            CastError::OutOfRange { value, lo, hi } => {
                write!(
                    f,
                    "Value {value} out of range [{lo}, {hi}]. \
                     Set out_of_range='clamp' or out_of_range='wrap' to handle this."
                )
            }
        }
    }
}

impl std::error::Error for CastError {}

// ---------------------------------------------------------------------------
// Rounding modes
// ---------------------------------------------------------------------------

/// How to round floating-point values when casting to integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub trait CastNum: Copy + PartialEq + PartialOrd + std::fmt::Debug {
    /// Convert to f64 for error reporting only (not in the conversion hot path).
    fn to_f64_lossy(self) -> f64;
}

/// An integer numeric type (i8..i64, u8..u64).
/// Marker trait — integers need no special operations beyond `CastNum`.
pub trait CastInt: CastNum {}

/// A floating-point numeric type (f32, f64).
/// Carries all float-specific operations: NaN/Inf checks, rounding, and
/// arithmetic needed for float→int wrap mode.
pub trait CastFloat: CastNum + Add<Output = Self> + Sub<Output = Self> {
    /// Returns true if the value is NaN.
    fn is_nan(self) -> bool;
    /// Returns true if the value is infinite.
    fn is_infinite(self) -> bool;
    /// Round according to the given rounding mode.
    fn round(self, mode: RoundingMode) -> Self;
    /// Euclidean remainder (used in float→int wrap mode).
    fn rem_euclid(self, rhs: Self) -> Self;
    /// The value 1 in this type.
    fn one() -> Self;
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

/// Helper trait for constructing a value from f64 (needed for constructing
/// MapEntry values from f64 pairs at the Python boundary).
pub trait FromF64 {
    fn from_f64(val: f64) -> Self;
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
pub struct FloatToIntConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
    pub rounding: RoundingMode,
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for int→int conversion.
pub struct IntToIntConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
    pub out_of_range: Option<OutOfRangeMode>,
}

/// Configuration for float→float conversion.
pub struct FloatToFloatConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
}

/// Configuration for int→float conversion.
pub struct IntToFloatConfig<'a, Src, Dst> {
    pub map_entries: &'a [MapEntry<Src, Dst>],
}

// ---------------------------------------------------------------------------
// Layer 1: Four per-element conversion functions
// ---------------------------------------------------------------------------

/// Convert a single float element to an integer.
///
/// Pipeline: scalar_map → NaN check → round → range check/clamp/wrap → cast.
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
        if let Some(tgt) = apply_scalar_map_float(val, config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: reject NaN (NaN comparisons are all false, would slip through range check)
    if val.is_nan() {
        return Err(CastError::NanOrInf {
            value: val.to_f64_lossy(),
        });
    }

    // Step 3: round
    let val = val.round(config.rounding);

    // Step 4: range check + out-of-range handling
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
            // Inf can't be wrapped (rem_euclid(Inf) is NaN)
            if val.is_infinite() {
                return Err(CastError::NanOrInf {
                    value: val.to_f64_lossy(),
                });
            }
            if val < lo || val > hi {
                let range = hi - lo + Src::one();
                let wrapped = (val - lo).rem_euclid(range) + lo;
                Ok(wrapped.cast_into())
            } else {
                Ok(val.cast_into())
            }
        }
        None => {
            if val < lo || val > hi {
                Err(CastError::OutOfRange {
                    value: val.to_f64_lossy(),
                    lo: lo.to_f64_lossy(),
                    hi: hi.to_f64_lossy(),
                })
            } else {
                Ok(val.cast_into())
            }
        }
    }
}

/// Convert a single integer element to another integer type.
///
/// Pipeline: scalar_map → range check/clamp/wrap → cast.
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
        if let Some(tgt) = apply_scalar_map_int(val, config.map_entries) {
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
                Err(CastError::OutOfRange {
                    value: val.to_f64_lossy(),
                    lo: lo.to_f64_lossy(),
                    hi: hi.to_f64_lossy(),
                })
            } else {
                Ok(val.cast_into())
            }
        }
    }
}

/// Convert a single float element to another float type.
///
/// Pipeline: scalar_map → cast. No rounding or range check needed.
#[inline]
pub fn convert_float_to_float<Src, Dst>(
    val: Src,
    config: &FloatToFloatConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastFloat,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_float(val, config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: cast
    Ok(val.cast_into())
}

/// Convert a single integer element to a float type.
///
/// Pipeline: scalar_map → cast. No rounding or range check needed.
#[inline]
pub fn convert_int_to_float<Src, Dst>(
    val: Src,
    config: &IntToFloatConfig<Src, Dst>,
) -> Result<Dst, CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastFloat,
{
    // Step 1: scalar_map lookup
    if !config.map_entries.is_empty() {
        if let Some(tgt) = apply_scalar_map_int(val, config.map_entries) {
            return Ok(tgt);
        }
    }

    // Step 2: cast
    Ok(val.cast_into())
}

// ---------------------------------------------------------------------------
// Layer 2: Four slice conversion functions
// ---------------------------------------------------------------------------

/// Convert a slice of float values to integer values. Returns early on first error.
pub fn convert_slice_float_to_int<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &FloatToIntConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastInt,
{
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_float_to_int(*in_val, config)?;
    }
    Ok(())
}

/// Convert a slice of integer values to integer values. Returns early on first error.
pub fn convert_slice_int_to_int<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &IntToIntConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastInt,
{
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_int_to_int(*in_val, config)?;
    }
    Ok(())
}

/// Convert a slice of float values to float values. Returns early on first error.
pub fn convert_slice_float_to_float<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &FloatToFloatConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastFloat + CastInto<Dst>,
    Dst: CastFloat,
{
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_float_to_float(*in_val, config)?;
    }
    Ok(())
}

/// Convert a slice of integer values to float values. Returns early on first error.
pub fn convert_slice_int_to_float<Src, Dst>(
    src: &[Src],
    dst: &mut [Dst],
    config: &IntToFloatConfig<Src, Dst>,
) -> Result<(), CastError>
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastFloat,
{
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()) {
        *out_slot = convert_int_to_float(*in_val, config)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Trait implementations for primitive types
// ---------------------------------------------------------------------------

// ---- CastNum impls ----

macro_rules! impl_cast_num {
    ($($ty:ty),*) => {
        $(
            impl CastNum for $ty {
                #[inline]
                fn to_f64_lossy(self) -> f64 { self as f64 }
            }
        )*
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
                fn is_nan(self) -> bool { <$ty>::is_nan(self) }
                #[inline]
                fn is_infinite(self) -> bool { <$ty>::is_infinite(self) }
                #[inline]
                fn round(self, mode: RoundingMode) -> Self {
                    match mode {
                        RoundingMode::NearestEven => <$ty>::round_ties_even(self),
                        RoundingMode::TowardsZero => <$ty>::trunc(self),
                        RoundingMode::TowardsPositive => <$ty>::ceil(self),
                        RoundingMode::TowardsNegative => <$ty>::floor(self),
                        RoundingMode::NearestAway => {
                            <$ty>::copysign((<$ty>::abs(self) + 0.5).floor(), self)
                        }
                    }
                }
                #[inline]
                fn rem_euclid(self, rhs: Self) -> Self { <$ty>::rem_euclid(self, rhs) }
                #[inline]
                fn one() -> Self { 1.0 }
            }
        )*
    };
}

impl_cast_float!(f32, f64);

// ---- FromF64 impls ----

macro_rules! impl_from_f64 {
    ($($ty:ty),*) => {
        $(
            impl FromF64 for $ty {
                #[inline]
                fn from_f64(val: f64) -> Self { val as $ty }
            }
        )*
    };
}

impl_from_f64!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

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

// -- any→float (range check not applied, but impl needed) --
macro_rules! impl_to_float {
    ($src:ty => $dst:ty) => {
        impl_cast_into!(
            $src => $dst,
            min: 0 as $src,  // unused — no range check for float targets
            max: 0 as $src
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

// all sources → float targets
macro_rules! impl_all_to_float {
    ($src:ty => $($dst:ty),*) => {
        $( impl_to_float!($src => $dst); )*
    };
}

impl_all_to_float!(i8 => f32, f64);
impl_all_to_float!(i16 => f32, f64);
impl_all_to_float!(i32 => f32, f64);
impl_all_to_float!(i64 => f32, f64);
impl_all_to_float!(u8 => f32, f64);
impl_all_to_float!(u16 => f32, f64);
impl_all_to_float!(u32 => f32, f64);
impl_all_to_float!(u64 => f32, f64);
impl_all_to_float!(f32 => f32, f64);
impl_all_to_float!(f64 => f32, f64);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build FloatToIntConfig
    fn f2i_cfg<Src: CastFloat, Dst: CastInt>(
        map: &[MapEntry<Src, Dst>],
        rounding: RoundingMode,
        oor: Option<OutOfRangeMode>,
    ) -> FloatToIntConfig<'_, Src, Dst> {
        FloatToIntConfig {
            map_entries: map,
            rounding,
            out_of_range: oor,
        }
    }

    // Helper to build IntToIntConfig
    fn i2i_cfg<Src: CastInt, Dst: CastInt>(
        map: &[MapEntry<Src, Dst>],
        oor: Option<OutOfRangeMode>,
    ) -> IntToIntConfig<'_, Src, Dst> {
        IntToIntConfig {
            map_entries: map,
            out_of_range: oor,
        }
    }

    #[test]
    fn test_float64_to_uint8_basic() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(42.0_f64, &c).unwrap(), 42_u8);
    }

    #[test]
    fn test_float64_to_uint8_rounding() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        // 2.5 rounds to 2 (banker's rounding)
        assert_eq!(convert_float_to_int(2.5_f64, &c).unwrap(), 2_u8);
        // 3.5 rounds to 4
        assert_eq!(convert_float_to_int(3.5_f64, &c).unwrap(), 4_u8);
    }

    #[test]
    fn test_float64_to_uint8_clamp() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_float_to_int(300.0_f64, &c).unwrap(), 255_u8);
        assert_eq!(convert_float_to_int(-10.0_f64, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_float64_to_int8_wrap() {
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Wrap));
        // 200 wraps: (200 - (-128)) % 256 + (-128) = 328 % 256 + (-128) = 72 + (-128) = -56
        assert_eq!(convert_float_to_int(200.0_f64, &c).unwrap(), -56_i8);
    }

    #[test]
    fn test_int32_to_int8_wrap() {
        let c = i2i_cfg::<i32, i8>(&[], Some(OutOfRangeMode::Wrap));
        // 200_i32 as i8 = -56 (bit truncation = modular wrap)
        assert_eq!(convert_int_to_int(200_i32, &c).unwrap(), -56_i8);
        assert_eq!(convert_int_to_int(-200_i32, &c).unwrap(), 56_i8);
    }

    #[test]
    fn test_int32_to_uint8_wrap() {
        let c = i2i_cfg::<i32, u8>(&[], Some(OutOfRangeMode::Wrap));
        assert_eq!(convert_int_to_int(300_i32, &c).unwrap(), 44_u8);
        assert_eq!(convert_int_to_int(-1_i32, &c).unwrap(), 255_u8);
    }

    #[test]
    fn test_out_of_range_error() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(300.0_f64, &c).is_err());
    }

    #[test]
    fn test_nan_to_int_error() {
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(f64::NAN, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_error() {
        // Without out_of_range, Inf is caught by the range check (OutOfRange error)
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        assert!(convert_float_to_int(f64::INFINITY, &c).is_err());
        assert!(convert_float_to_int(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_inf_to_int_clamp() {
        // With clamp, +Inf clamps to max, -Inf clamps to min
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_float_to_int(f64::INFINITY, &c).unwrap(), 255_u8);
        assert_eq!(convert_float_to_int(f64::NEG_INFINITY, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_inf_to_int_wrap_errors() {
        // With wrap, Inf is rejected (rem_euclid(Inf) is NaN)
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::NearestEven, Some(OutOfRangeMode::Wrap));
        assert!(convert_float_to_int(f64::INFINITY, &c).is_err());
        assert!(convert_float_to_int(f64::NEG_INFINITY, &c).is_err());
    }

    #[test]
    fn test_scalar_map_nan() {
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 0_u8,
        }];
        let c = f2i_cfg(&entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(f64::NAN, &c).unwrap(), 0_u8);
    }

    #[test]
    fn test_scalar_map_nan_payloads() {
        // Any NaN should match a NaN map entry, regardless of payload bits.
        let entries = vec![MapEntry {
            src: f64::NAN,
            tgt: 42_u8,
        }];
        let c = f2i_cfg(&entries, RoundingMode::NearestEven, None);

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
        let c = f2i_cfg(&entries, RoundingMode::NearestEven, None);
        assert_eq!(convert_float_to_int(42.0_f64, &c).unwrap(), 99_u8);
        // Non-matching value goes through normal path
        assert_eq!(convert_float_to_int(10.0_f64, &c).unwrap(), 10_u8);
    }

    #[test]
    fn test_int32_to_uint8_range_check() {
        let c = i2i_cfg::<i32, u8>(&[], None);
        assert_eq!(convert_int_to_int(100_i32, &c).unwrap(), 100_u8);
        assert!(convert_int_to_int(300_i32, &c).is_err());
        assert!(convert_int_to_int(-1_i32, &c).is_err());
    }

    #[test]
    fn test_int32_to_float64() {
        let c = IntToFloatConfig::<i32, f64> { map_entries: &[] };
        assert_eq!(convert_int_to_float(42_i32, &c).unwrap(), 42.0_f64);
    }

    #[test]
    fn test_convert_slice_basic() {
        let src = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        convert_slice_float_to_int(&src, &mut dst, &c).unwrap();
        assert_eq!(dst, [1, 2, 3, 4]);
    }

    #[test]
    fn test_convert_slice_early_termination() {
        let src = [1.0_f64, 2.0, 300.0, 4.0];
        let mut dst = [0_u8; 4];
        let c = f2i_cfg::<f64, u8>(&[], RoundingMode::NearestEven, None);
        let result = convert_slice_float_to_int(&src, &mut dst, &c);
        assert!(result.is_err());
        // First two should have been written
        assert_eq!(dst[0], 1);
        assert_eq!(dst[1], 2);
    }

    #[test]
    fn test_all_rounding_modes() {
        // towards-zero
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::TowardsZero, None);
        assert_eq!(convert_float_to_int(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_float_to_int(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-positive
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::TowardsPositive, None);
        assert_eq!(convert_float_to_int(2.1_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_float_to_int(-2.7_f64, &c).unwrap(), -2_i8);

        // towards-negative
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::TowardsNegative, None);
        assert_eq!(convert_float_to_int(2.7_f64, &c).unwrap(), 2_i8);
        assert_eq!(convert_float_to_int(-2.1_f64, &c).unwrap(), -3_i8);

        // nearest-away
        let c = f2i_cfg::<f64, i8>(&[], RoundingMode::NearestAway, None);
        assert_eq!(convert_float_to_int(2.5_f64, &c).unwrap(), 3_i8);
        assert_eq!(convert_float_to_int(-2.5_f64, &c).unwrap(), -3_i8);
    }

    #[test]
    fn test_int64_to_int32_clamp() {
        let c = i2i_cfg::<i64, i32>(&[], Some(OutOfRangeMode::Clamp));
        assert_eq!(convert_int_to_int(i64::MAX, &c).unwrap(), i32::MAX);
        assert_eq!(convert_int_to_int(i64::MIN, &c).unwrap(), i32::MIN);
    }

    #[test]
    fn test_float32_to_float64() {
        let c = FloatToFloatConfig::<f32, f64> { map_entries: &[] };
        assert_eq!(
            convert_float_to_float(3.14_f32, &c).unwrap(),
            3.14_f32 as f64
        );
    }
}
