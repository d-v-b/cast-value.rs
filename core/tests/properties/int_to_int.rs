//! `convert_int_to_int` against `oracle::int_to_int`.

use std::any::type_name;

use zarr_cast_value::{convert_int_to_int, CastInt, CastInto, IntToIntConfig};

use crate::oracle::{self, Outcome, OUT_OF_RANGE_MODES};
use crate::strategies::{
    i32_values, i64_values, related_i32, related_i64, related_u32, related_u64, u32_values,
    u64_values,
};
use crate::{for_each_int_target, mapped_property, scalar_map_lookup, IntType, ScalarMap};

/// Check one int→int cast under every out-of-range mode (int→int casts do
/// not round).
fn check<Src, Dst>((x, map): (Src, ScalarMap<Src>))
where
    Src: IntType + CastInt + CastInto<Dst>,
    Dst: IntType + CastInt,
{
    let entries = map.entries::<Dst>();
    let mapped = scalar_map_lookup(&entries, x);
    for out_of_range in OUT_OF_RANGE_MODES {
        let config = IntToIntConfig {
            map_entries: entries.clone(),
            out_of_range,
        };
        assert_eq!(
            Outcome::from_result(convert_int_to_int(x, &config).map(Dst::outcome)),
            mapped.unwrap_or_else(|| {
                oracle::int_to_int(x.to_i128(), (Dst::BITS, Dst::SIGNED), out_of_range)
            }),
            "{:?} -> {} with {out_of_range:?}",
            (x, map),
            type_name::<Dst>()
        );
    }
}

mapped_property!(i32_to_int_matches_oracle: for_each_int_target!(check::<i32>), i32_values, related_i32);
mapped_property!(i64_to_int_matches_oracle: for_each_int_target!(check::<i64>), i64_values, related_i64);
mapped_property!(u32_to_int_matches_oracle: for_each_int_target!(check::<u32>), u32_values, related_u32);
mapped_property!(u64_to_int_matches_oracle: for_each_int_target!(check::<u64>), u64_values, related_u64);

#[test]
fn every_narrow_int_to_int_matches_oracle() {
    (i8::MIN..=i8::MAX).for_each(|x| for_each_int_target!(check::<i8>((x, ScalarMap::EMPTY))));
    (u8::MIN..=u8::MAX).for_each(|x| for_each_int_target!(check::<u8>((x, ScalarMap::EMPTY))));
    (i16::MIN..=i16::MAX).for_each(|x| for_each_int_target!(check::<i16>((x, ScalarMap::EMPTY))));
    (u16::MIN..=u16::MAX).for_each(|x| for_each_int_target!(check::<u16>((x, ScalarMap::EMPTY))));
}
