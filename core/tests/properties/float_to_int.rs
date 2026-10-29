//! `convert_float_to_int` against `oracle::float_to_int`.

use std::any::type_name;

use zarr_cast_value::{convert_float_to_int, CastFloat, CastInt, CastInto, FloatToIntConfig};

use crate::oracle::{self, Outcome};
use crate::strategies::{f32_values, f64_values, related_f32, related_f64};
use crate::{
    for_each_int_target, mapped_property, scalar_map_lookup, FloatType, IntType, ScalarMap,
};

/// Check one float→int cast under every configuration.
fn check<Src, Dst>((x, map): (Src, ScalarMap<Src>))
where
    Src: FloatType + CastFloat + CastInto<Dst>,
    Dst: IntType + CastInt,
{
    let entries = map.entries::<Dst>();
    let mapped = scalar_map_lookup(&entries, x);
    oracle::check_conversion(
        &(x, map),
        type_name::<Dst>(),
        |rounding, out_of_range| {
            let config = FloatToIntConfig {
                map_entries: entries.clone(),
                rounding,
                out_of_range,
            };
            Outcome::from_result(convert_float_to_int(x, &config).map(Dst::outcome))
        },
        |rounding, out_of_range| {
            mapped.unwrap_or_else(|| {
                oracle::float_to_int(x.to_f64(), (Dst::BITS, Dst::SIGNED), rounding, out_of_range)
            })
        },
    );
}

mapped_property!(f64_to_int_matches_oracle: for_each_int_target!(check::<f64>), f64_values, related_f64);
mapped_property!(f32_to_int_matches_oracle: for_each_int_target!(check::<f32>), f32_values, related_f32);

#[cfg(feature = "float16")]
#[test]
fn every_f16_to_int_matches_oracle() {
    use zarr_cast_value::f16;

    for x in crate::all_f16() {
        for_each_int_target!(check::<f16>((x, ScalarMap::EMPTY)));
    }
}
