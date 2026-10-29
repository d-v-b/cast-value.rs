//! `convert_float_to_float` and `convert_int_to_float` against
//! `oracle::to_float`.

use std::any::type_name;

use proptest::prelude::*;
use zarr_cast_value::{
    convert_float_to_float, convert_int_to_float, CastFloat, CastInt, CastInto, FloatToFloatConfig,
    IntToFloatConfig,
};

use crate::oracle::{self, Exact, Outcome};
use crate::strategies::{f32_values, f64_values, i32_values, i64_values, u32_values, u64_values};
use crate::{for_each_float_target, FloatType, IntType};

/// Check one float→float cast under every configuration.
fn check_float<Src, Dst>(x: Src)
where
    Src: FloatType + CastFloat + CastInto<Dst>,
    Dst: FloatType + CastFloat,
{
    oracle::check_conversion(
        &x,
        type_name::<Dst>(),
        |rounding, out_of_range| {
            let config = FloatToFloatConfig {
                map_entries: Vec::new(),
                rounding,
                out_of_range,
            };
            Outcome::from_result(
                convert_float_to_float(x, &config).map(|y: Dst| Outcome::from_float(y.to_f64())),
            )
        },
        |rounding, out_of_range| {
            oracle::float_to_float(x.to_f64(), Dst::FORMAT, rounding, out_of_range)
        },
    );
}

/// Check one int→float cast under every configuration.
fn check_int<Src, Dst>(x: Src)
where
    Src: IntType + CastInt + CastInto<Dst>,
    Dst: FloatType + CastFloat,
{
    oracle::check_conversion(
        &x,
        type_name::<Dst>(),
        |rounding, out_of_range| {
            let config = IntToFloatConfig {
                map_entries: Vec::new(),
                rounding,
                out_of_range,
            };
            Outcome::from_result(
                convert_int_to_float(x, &config).map(|y: Dst| Outcome::from_float(y.to_f64())),
            )
        },
        |rounding, out_of_range| {
            oracle::to_float(
                Exact::from_int(x.to_i128()),
                Dst::FORMAT,
                rounding,
                out_of_range,
            )
        },
    );
}

proptest! {
    #[test]
    fn f64_to_float_matches_oracle(x in f64_values()) {
        for_each_float_target!(check_float::<f64>(x));
    }

    #[test]
    fn f32_to_float_matches_oracle(x in f32_values()) {
        for_each_float_target!(check_float::<f32>(x));
    }

    #[test]
    fn i32_to_float_matches_oracle(x in i32_values()) {
        for_each_float_target!(check_int::<i32>(x));
    }

    #[test]
    fn i64_to_float_matches_oracle(x in i64_values()) {
        for_each_float_target!(check_int::<i64>(x));
    }

    #[test]
    fn u32_to_float_matches_oracle(x in u32_values()) {
        for_each_float_target!(check_int::<u32>(x));
    }

    #[test]
    fn u64_to_float_matches_oracle(x in u64_values()) {
        for_each_float_target!(check_int::<u64>(x));
    }
}

#[test]
fn every_narrow_int_to_float_matches_oracle() {
    (i8::MIN..=i8::MAX).for_each(|x| for_each_float_target!(check_int::<i8>(x)));
    (u8::MIN..=u8::MAX).for_each(|x| for_each_float_target!(check_int::<u8>(x)));
    (i16::MIN..=i16::MAX).for_each(|x| for_each_float_target!(check_int::<i16>(x)));
    (u16::MIN..=u16::MAX).for_each(|x| for_each_float_target!(check_int::<u16>(x)));
}

#[cfg(feature = "float16")]
#[test]
fn every_f16_to_float_matches_oracle() {
    use zarr_cast_value::f16;

    for x in crate::all_f16() {
        for_each_float_target!(check_float::<f16>(x));
    }
}
