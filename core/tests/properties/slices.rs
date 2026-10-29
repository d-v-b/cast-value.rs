//! The slice casts (and their SIMD kernels) against mapping the per-element
//! cast over the slice, including which error comes first.

use std::any::type_name;
use std::fmt::Debug;

use proptest::prelude::*;
use zarr_cast_value::{
    convert_float_to_float, convert_float_to_int, convert_int_to_float, convert_int_to_int,
    convert_slice_float_to_float, convert_slice_float_to_int, convert_slice_int_to_float,
    convert_slice_int_to_int, CastError, CastFloat, CastInt, CastInto, FloatToFloatConfig,
    FloatToIntConfig, IntToFloatConfig, IntToIntConfig,
};

use crate::oracle::{all_modes, OUT_OF_RANGE_MODES};
use crate::strategies::{f32_values, f64_values, i64_values, keys_from, mapped_slices};
use crate::{for_each_float_target, for_each_int_target, FloatType, IntType, ScalarMap};

/// A slice with the scalar map to cast it under.
type MappedSlice<'a, S> = (&'a [S], ScalarMap<'a, S>);

/// Assert that a slice cast returns what mapping `element` over `src` does,
/// stopping at the first error. `CastError` has no `PartialEq`, so results
/// are compared through `Debug`.
fn assert_slice_matches<Src: Copy + Debug, Dst: Copy + Default + Debug>(
    src: &[Src],
    slice: impl Fn(&[Src], &mut [Dst]) -> Result<(), CastError>,
    element: impl Fn(Src) -> Result<Dst, CastError>,
    context: &dyn Debug,
) {
    let mut dst = vec![Dst::default(); src.len()];
    let actual = slice(src, &mut dst).map(|()| dst);
    let expected: Result<Vec<Dst>, CastError> = src.iter().map(|&x| element(x)).collect();
    assert_eq!(
        format!("{actual:?}"),
        format!("{expected:?}"),
        "{src:?} with {context:?}"
    );
}

fn check_float_to_int<Src, Dst>((src, map): MappedSlice<Src>)
where
    Src: FloatType + CastFloat + CastInto<Dst> + 'static,
    Dst: IntType + CastInt + Default + 'static,
{
    for (rounding, out_of_range) in all_modes() {
        let config = FloatToIntConfig {
            map_entries: map.entries(),
            rounding,
            out_of_range,
        };
        assert_slice_matches(
            src,
            |src, dst| convert_slice_float_to_int(src, dst, &config),
            |x| convert_float_to_int(x, &config),
            &(type_name::<Dst>(), map, rounding, out_of_range),
        );
    }
}

fn check_int_to_int<Src, Dst>((src, map): MappedSlice<Src>)
where
    Src: IntType + CastInt + CastInto<Dst>,
    Dst: IntType + CastInt + Default,
{
    for out_of_range in OUT_OF_RANGE_MODES {
        let config = IntToIntConfig {
            map_entries: map.entries(),
            out_of_range,
        };
        assert_slice_matches(
            src,
            |src, dst| convert_slice_int_to_int(src, dst, &config),
            |x| convert_int_to_int(x, &config),
            &(type_name::<Dst>(), map, out_of_range),
        );
    }
}

fn check_float_to_float<Src, Dst>((src, map): MappedSlice<Src>)
where
    Src: FloatType + CastFloat + CastInto<Dst> + 'static,
    Dst: FloatType + CastFloat + Default + 'static,
{
    for (rounding, out_of_range) in all_modes() {
        let config = FloatToFloatConfig {
            map_entries: map.entries(),
            rounding,
            out_of_range,
        };
        assert_slice_matches(
            src,
            |src, dst| convert_slice_float_to_float(src, dst, &config),
            |x| convert_float_to_float(x, &config),
            &(type_name::<Dst>(), map, rounding, out_of_range),
        );
    }
}

fn check_int_to_float<Src, Dst>((src, map): MappedSlice<Src>)
where
    Src: IntType + CastInt + CastInto<Dst>,
    Dst: FloatType + CastFloat + Default,
{
    for (rounding, out_of_range) in all_modes() {
        let config = IntToFloatConfig {
            map_entries: map.entries(),
            rounding,
            out_of_range,
        };
        assert_slice_matches(
            src,
            |src, dst| convert_slice_int_to_float(src, dst, &config),
            |x| convert_int_to_float(x, &config),
            &(type_name::<Dst>(), map, rounding, out_of_range),
        );
    }
}

// Each property runs without a scalar map (where the SIMD kernels apply) and
// with one whose keys are drawn from the slice itself.
proptest! {
    #[test]
    fn f64_slices_match_elementwise(
        (src, picks, targets) in mapped_slices(f64_values(), -300.0..300.0),
    ) {
        let keys = keys_from(&src, &picks);
        for map in [ScalarMap::EMPTY, ScalarMap { keys: &keys, targets: &targets }] {
            for_each_int_target!(check_float_to_int::<f64>((src.as_slice(), map)));
            for_each_float_target!(check_float_to_float::<f64>((src.as_slice(), map)));
        }
    }

    #[test]
    fn f32_slices_match_elementwise(
        (src, picks, targets) in mapped_slices(f32_values(), -300.0_f32..300.0),
    ) {
        let keys = keys_from(&src, &picks);
        for map in [ScalarMap::EMPTY, ScalarMap { keys: &keys, targets: &targets }] {
            for_each_int_target!(check_float_to_int::<f32>((src.as_slice(), map)));
            for_each_float_target!(check_float_to_float::<f32>((src.as_slice(), map)));
        }
    }

    #[test]
    fn i64_slices_match_elementwise(
        (src, picks, targets) in mapped_slices(i64_values(), -300_i64..300),
    ) {
        let keys = keys_from(&src, &picks);
        for map in [ScalarMap::EMPTY, ScalarMap { keys: &keys, targets: &targets }] {
            for_each_int_target!(check_int_to_int::<i64>((src.as_slice(), map)));
            for_each_float_target!(check_int_to_float::<i64>((src.as_slice(), map)));
        }
    }
}
