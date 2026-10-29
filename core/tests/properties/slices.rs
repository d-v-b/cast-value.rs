//! The slice casts (and their SIMD kernels) against mapping the per-element
//! cast over the slice, including which error comes first.

use std::any::type_name;
use std::fmt::Debug;

use proptest::prelude::*;
use zarr_cast_value::{
    convert_float_to_float, convert_float_to_int, convert_int_to_float,
    convert_slice_float_to_float, convert_slice_float_to_int, convert_slice_int_to_float,
    CastError, CastFloat, CastInt, CastInto, FloatToFloatConfig, FloatToIntConfig,
    IntToFloatConfig,
};

use crate::oracle::all_modes;
use crate::strategies::{f32_values, f64_values, i64_values, slices};
use crate::{for_each_float_target, for_each_int_target};

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

fn check_float_to_int<Src, Dst>(src: &[Src])
where
    Src: CastFloat + CastInto<Dst> + 'static,
    Dst: CastInt + Default + 'static,
{
    for (rounding, out_of_range) in all_modes() {
        let config = FloatToIntConfig {
            map_entries: Vec::new(),
            rounding,
            out_of_range,
        };
        assert_slice_matches(
            src,
            |src, dst| convert_slice_float_to_int(src, dst, &config),
            |x| convert_float_to_int(x, &config),
            &(type_name::<Dst>(), rounding, out_of_range),
        );
    }
}

fn check_float_to_float<Src, Dst>(src: &[Src])
where
    Src: CastFloat + CastInto<Dst> + 'static,
    Dst: CastFloat + Default + 'static,
{
    for (rounding, out_of_range) in all_modes() {
        let config = FloatToFloatConfig {
            map_entries: Vec::new(),
            rounding,
            out_of_range,
        };
        assert_slice_matches(
            src,
            |src, dst| convert_slice_float_to_float(src, dst, &config),
            |x| convert_float_to_float(x, &config),
            &(type_name::<Dst>(), rounding, out_of_range),
        );
    }
}

fn check_int_to_float<Src, Dst>(src: &[Src])
where
    Src: CastInt + CastInto<Dst>,
    Dst: CastFloat + Default,
{
    for (rounding, out_of_range) in all_modes() {
        let config = IntToFloatConfig {
            map_entries: Vec::new(),
            rounding,
            out_of_range,
        };
        assert_slice_matches(
            src,
            |src, dst| convert_slice_int_to_float(src, dst, &config),
            |x| convert_int_to_float(x, &config),
            &(type_name::<Dst>(), rounding, out_of_range),
        );
    }
}

proptest! {
    #[test]
    fn f64_slices_match_elementwise(src in slices(f64_values(), -300.0..300.0)) {
        for_each_int_target!(check_float_to_int::<f64>(&src));
        for_each_float_target!(check_float_to_float::<f64>(&src));
    }

    #[test]
    fn f32_slices_match_elementwise(src in slices(f32_values(), -300.0_f32..300.0)) {
        for_each_int_target!(check_float_to_int::<f32>(&src));
        for_each_float_target!(check_float_to_float::<f32>(&src));
    }

    #[test]
    fn i64_slices_match_elementwise(src in slices(i64_values(), -300_i64..300)) {
        for_each_float_target!(check_int_to_float::<i64>(&src));
    }
}
