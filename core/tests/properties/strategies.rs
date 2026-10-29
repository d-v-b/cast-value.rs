//! Input generators weighted towards the values where casts go wrong: integer
//! range limits, rounding ties, float format limits and subnormals.

use proptest::prelude::*;

use crate::oracle::{pow2, FloatFormat, F16, F32};

/// Move `x` by `steps` units in the last place (down for negative `steps`).
fn step_ulps<T: Copy>(x: T, steps: i32, up: fn(T) -> T, down: fn(T) -> T) -> T {
    let step = if steps > 0 { up } else { down };
    (0..steps.unsigned_abs()).fold(x, |value, _| step(value))
}

/// Powers of two where integer and float ranges, precisions and subnormals
/// end, plus the largest finite values of each float format.
fn limit_anchors() -> Vec<f64> {
    [
        0, 7, 8, 11, 15, 16, 24, 31, 32, 52, 53, 63, 64, 127, 128, 1023, -14, -24, -25, -126, -149,
        -150, -1022, -1074,
    ]
    .into_iter()
    .map(pow2)
    .chain([F16.max(), F32.max(), f64::MAX])
    .collect()
}

/// A limit anchor offset by a small integer or half-integer.
fn near_limits() -> impl Strategy<Value = f64> {
    (
        prop::sample::select(limit_anchors()),
        prop::sample::select(vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
    )
        .prop_map(|(anchor, delta)| anchor + delta)
}

/// Finite values of `format`, or the points halfway between them and their
/// successors (including halfway past MAX, where rounding to nearest starts
/// to overflow). Exact in `f64` for `F16` and `F32`.
fn format_neighborhood(format: FloatFormat) -> impl Strategy<Value = f64> {
    let precision = format.precision as i32;
    let max_biased_exponent = 2 * format.max_exponent;
    (
        0..=max_biased_exponent,
        0_u64..1 << (precision - 1),
        any::<bool>(),
    )
        .prop_map(move |(biased_exponent, fraction, halfway)| {
            let implicit_bit = if biased_exponent == 0 {
                0
            } else {
                1 << (precision - 1)
            };
            let exponent = biased_exponent.max(1) - format.max_exponent - precision + 1;
            // One extra bit below the last significand bit holds the half.
            let digits = ((fraction | implicit_bit) << 1) | u64::from(halfway);
            digits as f64 * pow2(exponent - 1)
        })
}

/// A half-integer, exact up to 2^52 in magnitude.
fn half_integer() -> impl Strategy<Value = f64> {
    (any::<i64>(), 0_u32..64).prop_map(|(value, shift)| (value >> shift) as f64 + 0.5)
}

/// `f64` inputs: arbitrary bit patterns plus the edge cases above, nudged by
/// a few ulps and given either sign.
pub fn f64_values() -> impl Strategy<Value = f64> {
    let edges = prop_oneof![
        near_limits(),
        format_neighborhood(F16),
        format_neighborhood(F32),
        half_integer(),
    ];
    prop_oneof![
        prop::num::f64::ANY,
        (edges, -3_i32..=3, any::<bool>()).prop_map(|(x, steps, negative)| {
            let x = step_ulps(x, steps, f64::next_up, f64::next_down);
            if negative {
                -x
            } else {
                x
            }
        }),
    ]
}

/// `f32` inputs: arbitrary bit patterns plus the edge cases above (rounded to
/// `f32`), nudged by a few `f32` ulps and given either sign.
pub fn f32_values() -> impl Strategy<Value = f32> {
    let edges = prop_oneof![near_limits(), format_neighborhood(F16), half_integer()];
    prop_oneof![
        prop::num::f32::ANY,
        (edges, -3_i32..=3, any::<bool>()).prop_map(|(x, steps, negative)| {
            let x = step_ulps(x as f32, steps, f32::next_up, f32::next_down);
            if negative {
                -x
            } else {
                x
            }
        }),
    ]
}

/// Slices of `values` for the SIMD kernels: either arbitrary, or drawn from a
/// small range so that whole slices succeed without clamping.
pub fn slices<T: Copy + std::fmt::Debug + 'static>(
    values: impl Strategy<Value = T> + 'static,
    in_range: impl Strategy<Value = T> + 'static,
) -> impl Strategy<Value = Vec<T>> {
    prop_oneof![
        prop::collection::vec(values, 0..=70),
        prop::collection::vec(in_range, 0..=70),
    ]
}

/// Bit patterns of a `width`-bit integer: uniform, at every magnitude, with a
/// rounding tie in the low bits, and next to powers of two.
fn int_bits(width: u32) -> impl Strategy<Value = u64> {
    let mask = u64::MAX >> (64 - width);
    let uniform = any::<u64>().prop_map(move |bits| bits & mask);
    prop_oneof![
        uniform.clone(),
        (uniform.clone(), 0..width).prop_map(|(bits, shift)| bits >> shift),
        (uniform, 0..width, 1..=width.min(41)).prop_map(|(bits, shift, tie_bit)| {
            ((bits >> shift) & !((1 << tie_bit) - 1)) | (1 << (tie_bit - 1))
        }),
        (0..width, -3_i64..=3).prop_map(move |(exponent, delta)| {
            (1_u64 << exponent).wrapping_add_signed(delta) & mask
        }),
    ]
}

/// Unsigned integer inputs from `int_bits`.
macro_rules! unsigned_values {
    ($ty:ty) => {
        int_bits(<$ty>::BITS).prop_map(|bits| bits as $ty)
    };
}

/// Signed integer inputs from `int_bits`, with a random sign. (`as` between
/// integer types reinterprets the low bits as two's complement.)
macro_rules! signed_values {
    ($ty:ty) => {
        (int_bits(<$ty>::BITS), any::<bool>()).prop_map(|(bits, negative)| {
            let value = bits as $ty;
            if negative {
                value.wrapping_neg()
            } else {
                value
            }
        })
    };
}

pub fn i32_values() -> impl Strategy<Value = i32> {
    signed_values!(i32)
}

pub fn i64_values() -> impl Strategy<Value = i64> {
    signed_values!(i64)
}

pub fn u32_values() -> impl Strategy<Value = u32> {
    unsigned_values!(u32)
}

pub fn u64_values() -> impl Strategy<Value = u64> {
    unsigned_values!(u64)
}
