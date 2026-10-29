//! Exact reference implementations of the cast_value conversions.
//!
//! Every oracle works on the exact value of its input, written as
//! `(-1)^negative × magnitude × 2^exponent` and manipulated with `u128`/`i128`
//! arithmetic, so the only rounding is the one the spec asks for. None of them
//! call the crate's conversion code.

use std::cmp::Ordering;
use std::fmt::Debug;

use zarr_cast_value::{CastError, OutOfRangeMode, RoundingMode};

/// Every rounding mode, ordered so that each mode's result is at least the
/// previous one's at the first index and at most the last one's.
pub const ROUNDING_MODES: [RoundingMode; 5] = [
    RoundingMode::TowardsNegative,
    RoundingMode::TowardsZero,
    RoundingMode::NearestEven,
    RoundingMode::NearestAway,
    RoundingMode::TowardsPositive,
];

/// Every out-of-range setting, including none.
pub const OUT_OF_RANGE_MODES: [Option<OutOfRangeMode>; 3] = [
    None,
    Some(OutOfRangeMode::Clamp),
    Some(OutOfRangeMode::Wrap),
];

/// Every (rounding, out-of-range) configuration.
pub fn all_modes() -> impl Iterator<Item = (RoundingMode, Option<OutOfRangeMode>)> {
    OUT_OF_RANGE_MODES
        .into_iter()
        .flat_map(|out_of_range| ROUNDING_MODES.map(|rounding| (rounding, out_of_range)))
}

/// The exact power of two `2^exponent` as `f64`, for `-1074 <= exponent <= 1023`.
pub fn pow2(exponent: i32) -> f64 {
    assert!(
        (-1074..=1023).contains(&exponent),
        "2^{exponent} is not an f64"
    );
    // Build the bits directly: `powi` would round (or underflow) on the way.
    if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (exponent + 1074))
    }
}

/// An IEEE 754 binary interchange format.
#[derive(Clone, Copy, Debug)]
pub struct FloatFormat {
    /// Significand bits, including the implicit leading bit.
    pub precision: u32,
    /// Exponent of the smallest normal value.
    pub min_exponent: i32,
    /// Exponent of the largest finite value.
    pub max_exponent: i32,
}

pub const F16: FloatFormat = FloatFormat {
    precision: 11,
    min_exponent: -14,
    max_exponent: 15,
};
pub const F32: FloatFormat = FloatFormat {
    precision: 24,
    min_exponent: -126,
    max_exponent: 127,
};
pub const F64: FloatFormat = FloatFormat {
    precision: 53,
    min_exponent: -1022,
    max_exponent: 1023,
};

impl FloatFormat {
    /// The largest finite value, `(2^p - 1) × 2^(emax - p + 1)`.
    pub fn max(self) -> f64 {
        ((1_u64 << self.precision) - 1) as f64 * pow2(self.max_exponent - self.precision as i32 + 1)
    }
}

/// A finite value `(-1)^negative × magnitude × 2^exponent` with
/// `magnitude < 2^64`.
#[derive(Clone, Copy, Debug)]
pub struct Exact {
    pub negative: bool,
    pub magnitude: u128,
    pub exponent: i32,
}

impl Exact {
    /// Decompose a finite `f64` from its bit fields.
    pub fn from_f64(x: f64) -> Self {
        assert!(x.is_finite(), "{x} has no exact value");
        let bits = x.to_bits();
        let biased_exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = u128::from(bits & ((1 << 52) - 1));
        let (magnitude, exponent) = if biased_exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | 1 << 52, biased_exponent - 1075)
        };
        Self {
            negative: bits >> 63 == 1,
            magnitude,
            exponent,
        }
    }

    /// Wrap an integer (any value of a 64-bit-or-narrower type).
    pub fn from_int(value: i128) -> Self {
        Self {
            negative: value < 0,
            magnitude: value.unsigned_abs(),
            exponent: 0,
        }
    }
}

/// Divide `magnitude` by `2^shift`, rounding the quotient to an integer in
/// `mode`. `negative` is the sign of the value whose magnitude this is, which
/// the directed modes depend on.
fn shift_round(magnitude: u128, shift: u32, negative: bool, mode: RoundingMode) -> u128 {
    assert!(magnitude < 1 << 64, "magnitude {magnitude} is too wide");
    if shift == 0 {
        return magnitude;
    }
    // A magnitude below 2^64 shifted by more than 64 leaves quotient 0 and a
    // remainder below one half.
    let (quotient, remainder_vs_half, inexact) = if shift > 64 {
        (0, Ordering::Less, magnitude != 0)
    } else {
        let remainder = magnitude & ((1 << shift) - 1);
        (
            magnitude >> shift,
            remainder.cmp(&(1 << (shift - 1))),
            remainder != 0,
        )
    };
    let round_up = match mode {
        RoundingMode::NearestEven => {
            remainder_vs_half == Ordering::Greater
                || (remainder_vs_half == Ordering::Equal && quotient & 1 == 1)
        }
        RoundingMode::NearestAway => remainder_vs_half != Ordering::Less,
        RoundingMode::TowardsZero => false,
        RoundingMode::TowardsPositive => inexact && !negative,
        RoundingMode::TowardsNegative => inexact && negative,
    };
    quotient + u128::from(round_up)
}

/// The inclusive bounds of a `bits`-wide integer type.
pub fn int_bounds(bits: u32, signed: bool) -> (i128, i128) {
    if signed {
        (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    } else {
        (0, (1 << bits) - 1)
    }
}

/// Reduce `value` modulo `2^bits` into a `bits`-wide integer type, as two's
/// complement for signed types.
fn wrap(value: i128, bits: u32, signed: bool) -> i128 {
    let modulus = 1_i128 << bits;
    let residue = value.rem_euclid(modulus);
    if signed && residue >= modulus / 2 {
        residue - modulus
    } else {
        residue
    }
}

/// The outcome of one conversion, comparable between the crate and an oracle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outcome {
    Int(i128),
    /// A non-NaN float result, as the bits of its (exact) `f64` widening, so
    /// that signed zeros differ.
    Float(u64),
    Nan,
    NanOrInf,
    OutOfRange {
        lo: i128,
        hi: i128,
    },
    FloatOutOfRange {
        lo: f64,
        hi: f64,
    },
}

impl Outcome {
    /// A float result, widened exactly to `f64`.
    pub fn from_float(x: f64) -> Self {
        if x.is_nan() {
            Self::Nan
        } else {
            Self::Float(x.to_bits())
        }
    }

    /// Flatten a crate result whose `Ok` side is already an `Outcome`.
    pub fn from_result(result: Result<Self, CastError>) -> Self {
        result.unwrap_or_else(|error| match error {
            CastError::NanOrInf { .. } => Self::NanOrInf,
            CastError::OutOfRange { lo, hi, .. } => Self::OutOfRange { lo, hi },
            CastError::FloatOutOfRange { lo, hi, .. } => Self::FloatOutOfRange { lo, hi },
            CastError::LengthMismatch { .. } => unreachable!("element casts check no lengths"),
        })
    }

    /// The numeric value of a successful, non-NaN result, for ordering.
    /// (`i128 as f64` rounds, but monotonically, so `<=` survives it.)
    fn ordered(&self) -> Option<f64> {
        match *self {
            Self::Int(value) => Some(value as f64),
            Self::Float(bits) => Some(f64::from_bits(bits)),
            _ => None,
        }
    }
}

/// Reference int→int cast: range-check, clamp or wrap the exact value.
pub fn int_to_int(
    value: i128,
    (bits, signed): (u32, bool),
    out_of_range: Option<OutOfRangeMode>,
) -> Outcome {
    let (lo, hi) = int_bounds(bits, signed);
    if (lo..=hi).contains(&value) {
        return Outcome::Int(value);
    }
    match out_of_range {
        Some(OutOfRangeMode::Clamp) => Outcome::Int(value.clamp(lo, hi)),
        Some(OutOfRangeMode::Wrap) => Outcome::Int(wrap(value, bits, signed)),
        None => Outcome::OutOfRange { lo, hi },
    }
}

/// Reference float→int cast: round the exact value, then range-check, clamp
/// or wrap it as an integer.
pub fn float_to_int(
    x: f64,
    target: (u32, bool),
    rounding: RoundingMode,
    out_of_range: Option<OutOfRangeMode>,
) -> Outcome {
    if !x.is_finite() {
        return Outcome::NanOrInf;
    }
    let exact = Exact::from_f64(x);
    // A non-negative exponent means `x` is already an integer. Beyond 2^64 it
    // only matters that `x` is out of every range and a multiple of 2^65 (so
    // 0 modulo 2^N), which 2^65 itself also is.
    let magnitude = match exact.exponent {
        65.. => 1 << 65,
        0..=64 => exact.magnitude << exact.exponent,
        ..0 => shift_round(
            exact.magnitude,
            exact.exponent.unsigned_abs(),
            exact.negative,
            rounding,
        ),
    };
    let value = if exact.negative {
        -(magnitude as i128)
    } else {
        magnitude as i128
    };
    int_to_int(value, target, out_of_range)
}

/// Reference cast of an exact value to a float format: round with an
/// unbounded exponent (but the format's subnormal spacing), then treat a
/// rounded magnitude beyond the finite range as overflow.
pub fn to_float(
    value: Exact,
    format: FloatFormat,
    rounding: RoundingMode,
    out_of_range: Option<OutOfRangeMode>,
) -> Outcome {
    let sign = if value.negative { -1.0 } else { 1.0 };
    if value.magnitude == 0 {
        return Outcome::from_float(sign * 0.0);
    }
    let precision = format.precision as i32;
    let leading_exponent = (127 - value.magnitude.leading_zeros()) as i32 + value.exponent;
    // The weight of the last significand bit at this magnitude, never below
    // the subnormal spacing.
    let quantum = (leading_exponent - precision + 1).max(format.min_exponent - precision + 1);
    let digits = if value.exponent >= quantum {
        value.magnitude << (value.exponent - quantum)
    } else {
        shift_round(
            value.magnitude,
            (quantum - value.exponent) as u32,
            value.negative,
            rounding,
        )
    };
    // `digits <= 2^p`, so the result is at least 2^(emax + 1) exactly when its
    // leading bit lies above emax.
    if digits != 0 && (127 - digits.leading_zeros()) as i32 + quantum > format.max_exponent {
        return match out_of_range {
            Some(OutOfRangeMode::Clamp) => Outcome::from_float(sign * f64::INFINITY),
            Some(OutOfRangeMode::Wrap) | None => Outcome::FloatOutOfRange {
                lo: -format.max(),
                hi: format.max(),
            },
        };
    }
    // Both factors and the product are exact in f64.
    Outcome::from_float(sign * digits as f64 * pow2(quantum))
}

/// Reference float→float cast: NaN and ±Inf propagate, finite values round.
pub fn float_to_float(
    x: f64,
    format: FloatFormat,
    rounding: RoundingMode,
    out_of_range: Option<OutOfRangeMode>,
) -> Outcome {
    if x.is_finite() {
        to_float(Exact::from_f64(x), format, rounding, out_of_range)
    } else {
        Outcome::from_float(x)
    }
}

/// Compare a conversion with its oracle under every configuration, and check
/// that the rounding modes are ordered (towards-negative ≤ every other mode ≤
/// towards-positive) wherever they all succeed without wrapping.
///
/// # Arguments
///
/// * `input` - The value being converted, for failure messages
/// * `target` - The target type's name, for failure messages
/// * `actual` - Runs the crate's conversion under a configuration
/// * `expected` - Runs the oracle under a configuration
pub fn check_conversion(
    input: &dyn Debug,
    target: &str,
    actual: impl Fn(RoundingMode, Option<OutOfRangeMode>) -> Outcome,
    expected: impl Fn(RoundingMode, Option<OutOfRangeMode>) -> Outcome,
) {
    for out_of_range in OUT_OF_RANGE_MODES {
        let outcomes = ROUNDING_MODES.map(|rounding| {
            let outcome = actual(rounding, out_of_range);
            assert_eq!(
                outcome,
                expected(rounding, out_of_range),
                "{input:?} -> {target} with {rounding:?}, {out_of_range:?}"
            );
            outcome
        });
        if out_of_range == Some(OutOfRangeMode::Wrap) {
            continue;
        }
        let values = outcomes.map(|outcome| outcome.ordered());
        let (Some(lower), Some(upper)) = (values[0], values[4]) else {
            continue;
        };
        for value in values.into_iter().flatten() {
            assert!(
                lower <= value && value <= upper,
                "{input:?} -> {target} with {out_of_range:?}: rounding is not monotone: {values:?}"
            );
        }
    }
}
