//! `convert_float_to_float` and `convert_int_to_float` against
//! `oracle::to_float`.

use std::any::type_name;

use zarr_cast_value::{
    convert_float_to_float, convert_int_to_float, CastFloat, CastInt, CastInto, FloatToFloatConfig,
    IntToFloatConfig,
};

use crate::oracle::{self, Exact, Outcome};
use crate::strategies::{
    f32_values, f64_values, i32_values, i64_values, related_f32, related_f64, related_i32,
    related_i64, related_u32, related_u64, u32_values, u64_values,
};
use crate::{
    for_each_float_target, mapped_property, scalar_map_lookup, FloatType, IntType, ScalarMap,
};

/// Check one float→float cast under every configuration.
fn check_float<Src, Dst>((x, map): (Src, ScalarMap<Src>))
where
    Src: FloatType + CastFloat + CastInto<Dst>,
    Dst: FloatType + CastFloat,
{
    let entries = map.entries::<Dst>();
    let mapped = scalar_map_lookup(&entries, x);
    oracle::check_conversion(
        &(x, map),
        type_name::<Dst>(),
        |rounding, out_of_range| {
            let config = FloatToFloatConfig {
                map_entries: entries.clone(),
                rounding,
                out_of_range,
            };
            Outcome::from_result(convert_float_to_float(x, &config).map(Dst::outcome))
        },
        |rounding, out_of_range| {
            mapped.unwrap_or_else(|| {
                oracle::float_to_float(x.to_f64(), Dst::FORMAT, rounding, out_of_range)
            })
        },
    );
}

/// Check one int→float cast under every configuration.
fn check_int<Src, Dst>((x, map): (Src, ScalarMap<Src>))
where
    Src: IntType + CastInt + CastInto<Dst>,
    Dst: FloatType + CastFloat,
{
    let entries = map.entries::<Dst>();
    let mapped = scalar_map_lookup(&entries, x);
    oracle::check_conversion(
        &(x, map),
        type_name::<Dst>(),
        |rounding, out_of_range| {
            let config = IntToFloatConfig {
                map_entries: entries.clone(),
                rounding,
                out_of_range,
            };
            Outcome::from_result(convert_int_to_float(x, &config).map(Dst::outcome))
        },
        |rounding, out_of_range| {
            mapped.unwrap_or_else(|| {
                oracle::to_float(
                    Exact::from_int(x.to_i128()),
                    Dst::FORMAT,
                    rounding,
                    out_of_range,
                )
            })
        },
    );
}

mapped_property!(f64_to_float_matches_oracle: for_each_float_target!(check_float::<f64>), f64_values, related_f64);
mapped_property!(f32_to_float_matches_oracle: for_each_float_target!(check_float::<f32>), f32_values, related_f32);
mapped_property!(i32_to_float_matches_oracle: for_each_float_target!(check_int::<i32>), i32_values, related_i32);
mapped_property!(i64_to_float_matches_oracle: for_each_float_target!(check_int::<i64>), i64_values, related_i64);
mapped_property!(u32_to_float_matches_oracle: for_each_float_target!(check_int::<u32>), u32_values, related_u32);
mapped_property!(u64_to_float_matches_oracle: for_each_float_target!(check_int::<u64>), u64_values, related_u64);

#[test]
fn every_narrow_int_to_float_matches_oracle() {
    (i8::MIN..=i8::MAX)
        .for_each(|x| for_each_float_target!(check_int::<i8>((x, ScalarMap::EMPTY))));
    (u8::MIN..=u8::MAX)
        .for_each(|x| for_each_float_target!(check_int::<u8>((x, ScalarMap::EMPTY))));
    (i16::MIN..=i16::MAX)
        .for_each(|x| for_each_float_target!(check_int::<i16>((x, ScalarMap::EMPTY))));
    (u16::MIN..=u16::MAX)
        .for_each(|x| for_each_float_target!(check_int::<u16>((x, ScalarMap::EMPTY))));
}

#[cfg(feature = "float16")]
#[test]
fn every_f16_to_float_matches_oracle() {
    use zarr_cast_value::f16;

    for x in crate::all_f16() {
        for_each_float_target!(check_float::<f16>((x, ScalarMap::EMPTY)));
    }
}
