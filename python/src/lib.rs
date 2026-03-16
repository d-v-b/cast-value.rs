//! zarr-cast-value: PyO3 bindings for the cast_value codec.
//!
//! Exposes `cast_array` and `cast_array_into` to Python, dispatching on
//! numpy dtype pairs to monomorphized `convert_slice` calls from the core crate.

use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyList;
use zarr_cast_value_core::{
    CastError, ConvertConfig, FromF64, MapEntry, OutOfRangeMode, RoundingMode,
};

// ---------------------------------------------------------------------------
// Helpers: parse Python arguments into Rust types
// ---------------------------------------------------------------------------

fn parse_rounding(s: &str) -> PyResult<RoundingMode> {
    RoundingMode::from_str(s).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

fn parse_out_of_range(s: Option<&str>) -> PyResult<Option<OutOfRangeMode>> {
    match s {
        None => Ok(None),
        Some(s) => OutOfRangeMode::from_str(s)
            .map(Some)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
    }
}

fn cast_error_to_pyerr(e: CastError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

// ---------------------------------------------------------------------------
// Parse scalar_map_entries from Python list[list[float]] into typed MapEntry
// ---------------------------------------------------------------------------

fn parse_map_entries<Src, Dst>(entries: Option<&Bound<'_, PyList>>) -> PyResult<Vec<MapEntry<Src, Dst>>>
where
    Src: zarr_cast_value_core::CastType + FromF64,
    Dst: zarr_cast_value_core::CastType + FromF64,
{
    let Some(list) = entries else {
        return Ok(Vec::new());
    };
    let mut result = Vec::with_capacity(list.len());
    for item in list.iter() {
        let pair: Vec<f64> = item.extract()?;
        if pair.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Each scalar_map entry must be a [source, target] pair",
            ));
        }
        let src_f64 = pair[0];
        let tgt_f64 = pair[1];
        let src_is_nan = src_f64.is_nan();
        result.push(MapEntry {
            src: Src::from_f64(src_f64),
            src_is_nan,
            tgt: Dst::from_f64(tgt_f64),
        });
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Dtype key extraction
// ---------------------------------------------------------------------------

fn dtype_key(kind: char, itemsize: usize) -> PyResult<&'static str> {
    match (kind, itemsize) {
        ('i', 1) => Ok("int8"),
        ('i', 2) => Ok("int16"),
        ('i', 4) => Ok("int32"),
        ('i', 8) => Ok("int64"),
        ('u', 1) => Ok("uint8"),
        ('u', 2) => Ok("uint16"),
        ('u', 4) => Ok("uint32"),
        ('u', 8) => Ok("uint64"),
        ('f', 4) => Ok("float32"),
        ('f', 8) => Ok("float64"),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            format!("Unsupported dtype kind={kind} itemsize={itemsize}"),
        )),
    }
}

fn array_dtype_key(arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<&'static str> {
    let dtype = arr.getattr("dtype")?;
    let kind: char = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    dtype_key(kind, itemsize)
}

// ---------------------------------------------------------------------------
// N×N dispatch: allocating variant
// ---------------------------------------------------------------------------

/// Dispatch on (src_dtype, tgt_dtype) to call convert_slice with concrete types.
/// Allocates a new output numpy array.
fn dispatch_alloc<'py>(
    py: Python<'py>,
    arr: &Bound<'py, pyo3::types::PyAny>,
    src_dtype: &str,
    tgt_dtype: &str,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyList>>,
) -> PyResult<PyObject> {
    // Use a macro to generate all 100 match arms
    macro_rules! do_convert {
        ($src_ty:ty, $dst_ty:ty) => {{
            let input_arr: PyReadonlyArrayDyn<'_, $src_ty> = arr.extract()?;
            let src_slice = input_arr.as_slice().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array must be contiguous")
            })?;
            let map_entries = parse_map_entries::<$src_ty, $dst_ty>(map_entries_py)?;
            let config = ConvertConfig {
                map_entries: &map_entries,
                rounding,
                out_of_range: oor,
            };
            let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
            let output = PyArrayDyn::<$dst_ty>::zeros(py, &shape[..], false);
            {
                let mut output_rw = unsafe { output.as_array_mut() };
                let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Failed to get mutable slice from output array",
                    )
                })?;
                zarr_cast_value_core::convert_slice(src_slice, dst_slice, &config)
                    .map_err(cast_error_to_pyerr)?;
            }
            Ok(output.into_pyobject(py)?.into_any().unbind())
        }};
    }

    macro_rules! dispatch_dst {
        ($src_ty:ty, $tgt_dtype:expr) => {
            match $tgt_dtype {
                "int8" => do_convert!($src_ty, i8),
                "int16" => do_convert!($src_ty, i16),
                "int32" => do_convert!($src_ty, i32),
                "int64" => do_convert!($src_ty, i64),
                "uint8" => do_convert!($src_ty, u8),
                "uint16" => do_convert!($src_ty, u16),
                "uint32" => do_convert!($src_ty, u32),
                "uint64" => do_convert!($src_ty, u64),
                "float32" => do_convert!($src_ty, f32),
                "float64" => do_convert!($src_ty, f64),
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    format!("Unsupported target dtype: {}", $tgt_dtype),
                )),
            }
        };
    }

    match src_dtype {
        "int8" => dispatch_dst!(i8, tgt_dtype),
        "int16" => dispatch_dst!(i16, tgt_dtype),
        "int32" => dispatch_dst!(i32, tgt_dtype),
        "int64" => dispatch_dst!(i64, tgt_dtype),
        "uint8" => dispatch_dst!(u8, tgt_dtype),
        "uint16" => dispatch_dst!(u16, tgt_dtype),
        "uint32" => dispatch_dst!(u32, tgt_dtype),
        "uint64" => dispatch_dst!(u64, tgt_dtype),
        "float32" => dispatch_dst!(f32, tgt_dtype),
        "float64" => dispatch_dst!(f64, tgt_dtype),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            format!("Unsupported source dtype: {src_dtype}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// N×N dispatch: into variant
// ---------------------------------------------------------------------------

fn dispatch_into<'py>(
    py: Python<'py>,
    arr: &Bound<'py, pyo3::types::PyAny>,
    out: &Bound<'py, pyo3::types::PyAny>,
    src_dtype: &str,
    tgt_dtype: &str,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_entries_py: Option<&Bound<'py, PyList>>,
) -> PyResult<PyObject> {
    macro_rules! do_convert_into {
        ($src_ty:ty, $dst_ty:ty) => {{
            let input_arr: PyReadonlyArrayDyn<'_, $src_ty> = arr.extract()?;
            let src_slice = input_arr.as_slice().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array must be contiguous")
            })?;
            let map_entries = parse_map_entries::<$src_ty, $dst_ty>(map_entries_py)?;
            let config = ConvertConfig {
                map_entries: &map_entries,
                rounding,
                out_of_range: oor,
            };
            let out_arr: &Bound<'_, PyArrayDyn<$dst_ty>> = out.downcast()?;
            {
                let mut output_rw = unsafe { out_arr.as_array_mut() };
                let dst_slice = output_rw.as_slice_mut().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Output array must be contiguous and writeable",
                    )
                })?;
                zarr_cast_value_core::convert_slice(src_slice, dst_slice, &config)
                    .map_err(cast_error_to_pyerr)?;
            }
            Ok(py.None())
        }};
    }

    macro_rules! dispatch_dst_into {
        ($src_ty:ty, $tgt_dtype:expr) => {
            match $tgt_dtype {
                "int8" => do_convert_into!($src_ty, i8),
                "int16" => do_convert_into!($src_ty, i16),
                "int32" => do_convert_into!($src_ty, i32),
                "int64" => do_convert_into!($src_ty, i64),
                "uint8" => do_convert_into!($src_ty, u8),
                "uint16" => do_convert_into!($src_ty, u16),
                "uint32" => do_convert_into!($src_ty, u32),
                "uint64" => do_convert_into!($src_ty, u64),
                "float32" => do_convert_into!($src_ty, f32),
                "float64" => do_convert_into!($src_ty, f64),
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    format!("Unsupported target dtype: {}", $tgt_dtype),
                )),
            }
        };
    }

    match src_dtype {
        "int8" => dispatch_dst_into!(i8, tgt_dtype),
        "int16" => dispatch_dst_into!(i16, tgt_dtype),
        "int32" => dispatch_dst_into!(i32, tgt_dtype),
        "int64" => dispatch_dst_into!(i64, tgt_dtype),
        "uint8" => dispatch_dst_into!(u8, tgt_dtype),
        "uint16" => dispatch_dst_into!(u16, tgt_dtype),
        "uint32" => dispatch_dst_into!(u32, tgt_dtype),
        "uint64" => dispatch_dst_into!(u64, tgt_dtype),
        "float32" => dispatch_dst_into!(f32, tgt_dtype),
        "float64" => dispatch_dst_into!(f64, tgt_dtype),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            format!("Unsupported source dtype: {src_dtype}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// cast_array: allocating variant
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (arr, target_dtype, rounding_mode, out_of_range_mode=None, scalar_map_entries=None))]
fn cast_array<'py>(
    py: Python<'py>,
    arr: &Bound<'py, pyo3::types::PyAny>,
    target_dtype: &str,
    rounding_mode: &str,
    out_of_range_mode: Option<&str>,
    scalar_map_entries: Option<&Bound<'py, PyList>>,
) -> PyResult<PyObject> {
    let rounding = parse_rounding(rounding_mode)?;
    let oor = parse_out_of_range(out_of_range_mode)?;
    let src_dtype = array_dtype_key(arr)?;

    dispatch_alloc(py, arr, src_dtype, target_dtype, rounding, oor, scalar_map_entries)
}

// ---------------------------------------------------------------------------
// cast_array_into: pre-allocated variant
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (arr, out, rounding_mode, out_of_range_mode=None, scalar_map_entries=None))]
fn cast_array_into<'py>(
    py: Python<'py>,
    arr: &Bound<'py, pyo3::types::PyAny>,
    out: &Bound<'py, pyo3::types::PyAny>,
    rounding_mode: &str,
    out_of_range_mode: Option<&str>,
    scalar_map_entries: Option<&Bound<'py, PyList>>,
) -> PyResult<PyObject> {
    let rounding = parse_rounding(rounding_mode)?;
    let oor = parse_out_of_range(out_of_range_mode)?;
    let src_dtype = array_dtype_key(arr)?;
    let tgt_dtype = array_dtype_key(out)?;

    // Validate shapes match
    let src_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let dst_shape: Vec<usize> = out.getattr("shape")?.extract()?;
    if src_shape != dst_shape {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Shape mismatch: input shape {:?} != output shape {:?}",
            src_shape, dst_shape
        )));
    }

    dispatch_into(py, arr, out, src_dtype, tgt_dtype, rounding, oor, scalar_map_entries)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn zarr_cast_value(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cast_array, m)?)?;
    m.add_function(wrap_pyfunction!(cast_array_into, m)?)?;
    Ok(())
}
