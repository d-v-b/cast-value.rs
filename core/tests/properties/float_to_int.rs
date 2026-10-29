//! `convert_float_to_int` against `oracle::float_to_int`.

use std::any::type_name;

use proptest::prelude::*;
use zarr_cast_value::{convert_float_to_int, CastFloat, CastInt, CastInto, FloatToIntConfig};

use crate::oracle::{self, Outcome};
use crate::strategies::{f32_values, f64_values};
use crate::{for_each_int_target, FloatType, IntType};

/// Check one float→int cast under every configuration.
fn check<Src, Dst>(x: Src)
where
    Src: FloatType + CastFloat + CastInto<Dst>,
    Dst: IntType + CastInt,
{
    oracle::check_conversion(
        &x,
        type_name::<Dst>(),
        |rounding, out_of_range| {
            let config = FloatToIntConfig {
                map_entries: Vec::new(),
                rounding,
                out_of_range,
            };
            Outcome::from_result(
                convert_float_to_int(x, &config).map(|y: Dst| Outcome::Int(y.to_i128())),
            )
        },
        |rounding, out_of_range| {
            oracle::float_to_int(x.to_f64(), (Dst::BITS, Dst::SIGNED), rounding, out_of_range)
        },
    );
}

proptest! {
    #[test]
    fn f64_to_int_matches_oracle(x in f64_values()) {
        for_each_int_target!(check::<f64>(x));
    }

    #[test]
    fn f32_to_int_matches_oracle(x in f32_values()) {
        for_each_int_target!(check::<f32>(x));
    }
}

#[cfg(feature = "float16")]
#[test]
fn every_f16_to_int_matches_oracle() {
    use zarr_cast_value::f16;

    for x in crate::all_f16() {
        for_each_int_target!(check::<f16>(x));
    }
}
