//! Property-based tests of the per-element and slice casts against exact
//! reference implementations (`oracle`).
//!
//! Small input spaces (every `f16`, every 8- and 16-bit integer) are checked
//! exhaustively; the rest use `proptest`. Set `PROPTEST_CASES` to raise the
//! number of cases per property from the default 256, e.g.
//! `PROPTEST_CASES=100000 cargo test -p zarr-cast-value --test properties`.

mod float_to_int;
mod oracle;
mod slices;
mod strategies;
mod to_float;

#[cfg(feature = "float16")]
use zarr_cast_value::f16;

/// An integer target or source, described independently of the crate.
pub trait IntType: Copy {
    const BITS: u32;
    const SIGNED: bool;
    fn to_i128(self) -> i128;
}

macro_rules! impl_int_type {
    ($($ty:ty),*) => {
        $(
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

/// A float target or source, described independently of the crate.
pub trait FloatType: Copy {
    const FORMAT: oracle::FloatFormat;
    /// Widen exactly to `f64`.
    fn to_f64(self) -> f64;
}

impl FloatType for f32 {
    const FORMAT: oracle::FloatFormat = oracle::F32;
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl FloatType for f64 {
    const FORMAT: oracle::FloatFormat = oracle::F64;
    fn to_f64(self) -> f64 {
        self
    }
}

#[cfg(feature = "float16")]
impl FloatType for f16 {
    const FORMAT: oracle::FloatFormat = oracle::F16;
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
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
