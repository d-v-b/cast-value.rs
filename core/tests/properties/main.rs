//! Property-based tests of the per-element and slice casts against exact
//! reference implementations (`oracle`).
//!
//! Small input spaces (every `f16`, every 8- and 16-bit integer) are checked
//! exhaustively; the rest use `proptest`. Set `PROPTEST_CASES` to raise the
//! number of cases per property from the default 256, e.g.
//! `PROPTEST_CASES=100000 cargo test -p zarr-cast-value --test properties`.

mod float_to_int;
mod int_to_int;
mod oracle;
mod slices;
mod strategies;
mod to_float;

use std::fmt::Debug;

#[cfg(feature = "float16")]
use zarr_cast_value::f16;
use zarr_cast_value::MapEntry;

use oracle::Outcome;

/// Operations every source and target type needs, described independently
/// of the crate.
pub trait Scalar: Copy + Debug {
    /// Reinterpret the low bits of `bits` as a value (for random map targets).
    fn from_raw(bits: u64) -> Self;
    /// This value as a conversion result.
    fn outcome(self) -> Outcome;
    /// Whether this scalar-map key matches `input`. Pins the crate's current
    /// semantics: numeric equality (so `0.0` and `-0.0` match each other),
    /// and any NaN matches any NaN whatever its sign or payload.
    fn key_matches(self, input: Self) -> bool;
}

/// An integer type.
pub trait IntType: Scalar {
    const BITS: u32;
    const SIGNED: bool;
    fn to_i128(self) -> i128;
}

macro_rules! impl_int_type {
    ($($ty:ty),*) => {
        $(
            impl Scalar for $ty {
                fn from_raw(bits: u64) -> Self {
                    // `as` between integer types keeps the low bits.
                    bits as $ty
                }
                fn outcome(self) -> Outcome {
                    Outcome::Int(self.to_i128())
                }
                fn key_matches(self, input: Self) -> bool {
                    self == input
                }
            }

            impl IntType for $ty {
                const BITS: u32 = <$ty>::BITS;
                const SIGNED: bool = <$ty>::MIN != 0;
                fn to_i128(self) -> i128 {
                    i128::from(self)
                }
            }
        )*
    };
}

impl_int_type!(i8, i16, i32, i64, u8, u16, u32, u64);

/// A float type.
pub trait FloatType: Scalar {
    const FORMAT: oracle::FloatFormat;
    /// Widen exactly to `f64`.
    fn to_f64(self) -> f64;
}

macro_rules! impl_float_type {
    ($ty:ty, $bits:ty, $format:expr, |$x:ident| $to_f64:expr) => {
        impl Scalar for $ty {
            fn from_raw(bits: u64) -> Self {
                <$ty>::from_bits(bits as $bits)
            }
            fn outcome(self) -> Outcome {
                Outcome::from_float(self.to_f64())
            }
            fn key_matches(self, input: Self) -> bool {
                (self.is_nan() && input.is_nan()) || self == input
            }
        }

        impl FloatType for $ty {
            const FORMAT: oracle::FloatFormat = $format;
            fn to_f64(self) -> f64 {
                let $x = self;
                $to_f64
            }
        }
    };
}

impl_float_type!(f32, u32, oracle::F32, |x| f64::from(x));
impl_float_type!(f64, u64, oracle::F64, |x| x);
#[cfg(feature = "float16")]
impl_float_type!(f16, u16, oracle::F16, |x| f16::to_f64(x));

/// A scalar map, as keys and raw target bits that each target type reads
/// through `Scalar::from_raw` (so one map serves every target type).
#[derive(Clone, Copy, Debug)]
pub struct ScalarMap<'a, S> {
    pub keys: &'a [S],
    pub targets: &'a [u64],
}

impl<S: Scalar> ScalarMap<'_, S> {
    /// No entries.
    pub const EMPTY: Self = Self {
        keys: &[],
        targets: &[],
    };

    /// The entries for target type `D`, in order.
    pub fn entries<D: Scalar>(self) -> Vec<MapEntry<S, D>> {
        self.keys
            .iter()
            .zip(self.targets)
            .map(|(&src, &bits)| MapEntry {
                src,
                tgt: D::from_raw(bits),
            })
            .collect()
    }
}

/// The reference scalar-map lookup: the target of the first key matching
/// `input`, if any (the spec says repeated keys resolve to the first).
pub fn scalar_map_lookup<S: Scalar, D: Scalar>(
    entries: &[MapEntry<S, D>],
    input: S,
) -> Option<Outcome> {
    entries
        .iter()
        .find(|entry| entry.src.key_matches(input))
        .map(|entry| entry.tgt.outcome())
}

/// Every `f16` bit pattern, including NaNs, infinities and subnormals.
#[cfg(feature = "float16")]
pub fn all_f16() -> impl Iterator<Item = f16> {
    (0..=u16::MAX).map(f16::from_bits)
}

/// Call `$check::<$src, T>($input)` for every integer type `T`.
#[macro_export]
macro_rules! for_each_int_target {
    ($check:ident::<$src:ty>($input:expr)) => {{
        let input = $input;
        $check::<$src, i8>(input);
        $check::<$src, i16>(input);
        $check::<$src, i32>(input);
        $check::<$src, i64>(input);
        $check::<$src, u8>(input);
        $check::<$src, u16>(input);
        $check::<$src, u32>(input);
        $check::<$src, u64>(input);
    }};
}

/// Call `$check::<$src, T>($input)` for every float type `T`.
#[macro_export]
macro_rules! for_each_float_target {
    ($check:ident::<$src:ty>($input:expr)) => {{
        let input = $input;
        $check::<$src, f32>(input);
        $check::<$src, f64>(input);
        #[cfg(feature = "float16")]
        $check::<$src, zarr_cast_value::f16>(input);
    }};
}

/// Define a property that checks inputs from `$values` with
/// `$for_each!($check::<$ty>(..))`, first without and then with a scalar map
/// whose keys come from `$related`.
#[macro_export]
macro_rules! mapped_property {
    ($name:ident: $for_each:ident!($check:ident::<$ty:ty>), $values:ident, $related:ident) => {
        proptest::proptest! {
            #[test]
            fn $name(
                (x, keys, targets) in $crate::strategies::mapped_inputs($values, $related),
            ) {
                let map = $crate::ScalarMap { keys: &keys, targets: &targets };
                $for_each!($check::<$ty>((x, $crate::ScalarMap::EMPTY)));
                $for_each!($check::<$ty>((x, map)));
            }
        }
    };
}
