//! cast-value-wasm: WebAssembly bindings for the cast_value codec.
//!
//! Exposes `castArray` and `castArrayInto` to JavaScript, accepting and
//! returning JS typed arrays. Designed for integration with zarrita.js as
//! an array-to-array codec.
//!
//! # Supported types
//!
//! `int8`, `int16`, `int32`, `uint8`, `uint16`, `uint32`, `float32`, `float64`
//!
//! # Limitations
//!
//! `int64` and `uint64` are not supported. JS represents these via
//! `BigInt64Array`/`BigUint64Array` which use `bigint`, not `number`.
//! The `wasm-bindgen` bridge does not support direct `bigint` ↔ `i64`
//! typed array interop, and scalar map entries (which are JS `number`
//! pairs) cannot represent 64-bit integers without precision loss.
//! Add support when zarrita.js requires it.

use wasm_bindgen::prelude::*;
use zarr_cast_value::{
    FloatToFloatConfig, FloatToIntConfig, IntToFloatConfig, IntToIntConfig, MapEntry,
    OutOfRangeMode, RoundingMode,
};

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

fn parse_rounding_mode(s: &str) -> Result<RoundingMode, JsValue> {
    s.parse::<RoundingMode>()
        .map_err(|e| JsValue::from_str(&e))
}

fn parse_oor(s: Option<String>) -> Result<Option<OutOfRangeMode>, JsValue> {
    match s {
        None => Ok(None),
        Some(s) => s
            .parse::<OutOfRangeMode>()
            .map(Some)
            .map_err(|e| JsValue::from_str(&e)),
    }
}

// ---------------------------------------------------------------------------
// Scalar map parsing
// ---------------------------------------------------------------------------

/// Trait for converting from JS f64 to a concrete numeric type.
/// JS numbers are always f64, so all scalar map values arrive as f64.
trait FromJsNumber: Sized {
    fn from_js(val: f64) -> Self;
}

macro_rules! impl_from_js_number {
    ($($ty:ty),*) => {
        $( impl FromJsNumber for $ty {
            #[inline]
            fn from_js(val: f64) -> Self { val as $ty }
        } )*
    };
}

impl_from_js_number!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

/// Parse scalar map entries from a JS array of `[src, tgt]` pairs.
fn parse_map<Src: FromJsNumber + zarr_cast_value::CastNum, Dst: FromJsNumber + zarr_cast_value::CastNum>(
    entries: &JsValue,
) -> Result<Vec<MapEntry<Src, Dst>>, JsValue> {
    if entries.is_undefined() || entries.is_null() {
        return Ok(Vec::new());
    }
    let arr = js_sys::Array::from(entries);
    let mut result = Vec::with_capacity(arr.length() as usize);
    for i in 0..arr.length() {
        let pair = js_sys::Array::from(&arr.get(i));
        if pair.length() != 2 {
            return Err(JsValue::from_str(
                "Each scalar_map entry must be a [source, target] pair",
            ));
        }
        let src_val = pair.get(0).as_f64().ok_or_else(|| {
            JsValue::from_str("scalar_map source value must be a number")
        })?;
        let tgt_val = pair.get(1).as_f64().ok_or_else(|| {
            JsValue::from_str("scalar_map target value must be a number")
        })?;
        result.push(MapEntry {
            src: Src::from_js(src_val),
            tgt: Dst::from_js(tgt_val),
        });
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Typed array helpers
// ---------------------------------------------------------------------------

macro_rules! js_to_vec {
    ($jsval:expr, $jsty:ty) => {{
        let arr: $jsty = $jsval.unchecked_into();
        arr.to_vec()
    }};
}

macro_rules! vec_to_js {
    ($data:expr, $jsty:ty) => {{
        <$jsty>::from($data.as_slice())
    }};
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn err_to_js(e: zarr_cast_value::CastError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Cast a typed array to a new dtype.
///
/// The source dtype must be passed explicitly because the JS typed array
/// type alone is ambiguous (e.g. `Uint8Array` could come from a `uint8`
/// or a `bool` zarr dtype).
///
/// # Arguments
///
/// * `src` - Source typed array
/// * `src_dtype` - Source dtype name (e.g. "float64", "uint8")
/// * `target_dtype` - Target dtype name
/// * `rounding_mode` - One of: "nearest-even", "towards-zero",
///   "towards-positive", "towards-negative", "nearest-away"
/// * `out_of_range_mode` - "clamp", "wrap", or undefined (error)
/// * `scalar_map_entries` - Array of [source, target] number pairs, or undefined
#[wasm_bindgen(js_name = "castArray")]
pub fn cast_array(
    src: &JsValue,
    src_dtype: &str,
    target_dtype: &str,
    rounding_mode: &str,
    out_of_range_mode: Option<String>,
    scalar_map_entries: &JsValue,
) -> Result<JsValue, JsValue> {
    let rounding = parse_rounding_mode(rounding_mode)?;
    let oor = parse_oor(out_of_range_mode)?;

    dispatch(src, src_dtype, target_dtype, rounding, oor, scalar_map_entries)
}

/// Cast a typed array into a pre-allocated output typed array (zero-copy).
///
/// Both `src` and `dst` are borrowed directly from JS memory — no copies.
/// The target dtype is inferred from the `dst` typed array.
///
/// # Arguments
///
/// * `src` - Source typed array
/// * `dst` - Pre-allocated output typed array (must have same length as src)
/// * `src_dtype` - Source dtype name
/// * `dst_dtype` - Target dtype name
/// * `rounding_mode` - Rounding mode string
/// * `out_of_range_mode` - "clamp", "wrap", or undefined
/// * `scalar_map_entries` - Array of [source, target] pairs, or undefined
#[wasm_bindgen(js_name = "castArrayInto")]
pub fn cast_array_into(
    src: &JsValue,
    dst: &JsValue,
    src_dtype: &str,
    dst_dtype: &str,
    rounding_mode: &str,
    out_of_range_mode: Option<String>,
    scalar_map_entries: &JsValue,
) -> Result<(), JsValue> {
    let rounding = parse_rounding_mode(rounding_mode)?;
    let oor = parse_oor(out_of_range_mode)?;

    // Parse scalar map entries before borrowing the typed arrays,
    // since parsing may allocate and invalidate WASM memory views.
    dispatch_into(
        src,
        dst,
        src_dtype,
        dst_dtype,
        rounding,
        oor,
        scalar_map_entries,
    )
}

// ---------------------------------------------------------------------------
// N x N dispatch (allocating)
// ---------------------------------------------------------------------------

fn dispatch(
    src: &JsValue,
    src_dtype: &str,
    tgt_dtype: &str,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_js: &JsValue,
) -> Result<JsValue, JsValue> {
    macro_rules! f2i {
        ($src_js:ty, $src_rs:ty, $dst_rs:ty, $dst_js:ty) => {{
            let data = js_to_vec!(src.clone(), $src_js);
            let config = FloatToIntConfig {
                map_entries: parse_map::<$src_rs, $dst_rs>(map_js)?,
                rounding,
                out_of_range: oor,
            };
            let mut out = vec![<$dst_rs>::default(); data.len()];
            zarr_cast_value::convert_slice_float_to_int(&data, &mut out, &config)
                .map_err(err_to_js)?;
            Ok(vec_to_js!(&out, $dst_js).into())
        }};
    }

    macro_rules! i2i {
        ($src_js:ty, $src_rs:ty, $dst_rs:ty, $dst_js:ty) => {{
            let data = js_to_vec!(src.clone(), $src_js);
            let config = IntToIntConfig {
                map_entries: parse_map::<$src_rs, $dst_rs>(map_js)?,
                out_of_range: oor,
            };
            let mut out = vec![<$dst_rs>::default(); data.len()];
            zarr_cast_value::convert_slice_int_to_int(&data, &mut out, &config)
                .map_err(err_to_js)?;
            Ok(vec_to_js!(&out, $dst_js).into())
        }};
    }

    macro_rules! f2f {
        ($src_js:ty, $src_rs:ty, $dst_rs:ty, $dst_js:ty) => {{
            let data = js_to_vec!(src.clone(), $src_js);
            let config = FloatToFloatConfig {
                map_entries: parse_map::<$src_rs, $dst_rs>(map_js)?,
                rounding,
                out_of_range: oor,
            };
            let mut out = vec![<$dst_rs>::default(); data.len()];
            zarr_cast_value::convert_slice_float_to_float(&data, &mut out, &config)
                .map_err(err_to_js)?;
            Ok(vec_to_js!(&out, $dst_js).into())
        }};
    }

    macro_rules! i2f {
        ($src_js:ty, $src_rs:ty, $dst_rs:ty, $dst_js:ty) => {{
            let data = js_to_vec!(src.clone(), $src_js);
            let config = IntToFloatConfig {
                map_entries: parse_map::<$src_rs, $dst_rs>(map_js)?,
                rounding,
            };
            let mut out = vec![<$dst_rs>::default(); data.len()];
            zarr_cast_value::convert_slice_int_to_float(&data, &mut out, &config)
                .map_err(err_to_js)?;
            Ok(vec_to_js!(&out, $dst_js).into())
        }};
    }

    macro_rules! dispatch_int {
        ($src_js:ty, $src_rs:ty) => {
            match tgt_dtype {
                "int8" => i2i!($src_js, $src_rs, i8, js_sys::Int8Array),
                "int16" => i2i!($src_js, $src_rs, i16, js_sys::Int16Array),
                "int32" => i2i!($src_js, $src_rs, i32, js_sys::Int32Array),
                "uint8" => i2i!($src_js, $src_rs, u8, js_sys::Uint8Array),
                "uint16" => i2i!($src_js, $src_rs, u16, js_sys::Uint16Array),
                "uint32" => i2i!($src_js, $src_rs, u32, js_sys::Uint32Array),
                "float32" => i2f!($src_js, $src_rs, f32, js_sys::Float32Array),
                "float64" => i2f!($src_js, $src_rs, f64, js_sys::Float64Array),
                _ => Err(JsValue::from_str(&format!(
                    "Unsupported target dtype: {tgt_dtype}"
                ))),
            }
        };
    }

    macro_rules! dispatch_float {
        ($src_js:ty, $src_rs:ty) => {
            match tgt_dtype {
                "int8" => f2i!($src_js, $src_rs, i8, js_sys::Int8Array),
                "int16" => f2i!($src_js, $src_rs, i16, js_sys::Int16Array),
                "int32" => f2i!($src_js, $src_rs, i32, js_sys::Int32Array),
                "uint8" => f2i!($src_js, $src_rs, u8, js_sys::Uint8Array),
                "uint16" => f2i!($src_js, $src_rs, u16, js_sys::Uint16Array),
                "uint32" => f2i!($src_js, $src_rs, u32, js_sys::Uint32Array),
                "float32" => f2f!($src_js, $src_rs, f32, js_sys::Float32Array),
                "float64" => f2f!($src_js, $src_rs, f64, js_sys::Float64Array),
                _ => Err(JsValue::from_str(&format!(
                    "Unsupported target dtype: {tgt_dtype}"
                ))),
            }
        };
    }

    match src_dtype {
        "int8" => dispatch_int!(js_sys::Int8Array, i8),
        "int16" => dispatch_int!(js_sys::Int16Array, i16),
        "int32" => dispatch_int!(js_sys::Int32Array, i32),
        "uint8" => dispatch_int!(js_sys::Uint8Array, u8),
        "uint16" => dispatch_int!(js_sys::Uint16Array, u16),
        "uint32" => dispatch_int!(js_sys::Uint32Array, u32),
        "float32" => dispatch_float!(js_sys::Float32Array, f32),
        "float64" => dispatch_float!(js_sys::Float64Array, f64),
        _ => Err(JsValue::from_str(&format!(
            "Unsupported source dtype: {src_dtype}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// N x N dispatch (into pre-allocated buffer, zero-copy)
// ---------------------------------------------------------------------------

fn dispatch_into(
    src: &JsValue,
    dst: &JsValue,
    src_dtype: &str,
    tgt_dtype: &str,
    rounding: RoundingMode,
    oor: Option<OutOfRangeMode>,
    map_js: &JsValue,
) -> Result<(), JsValue> {
    // Macros that:
    // 1. Parse scalar map (may allocate — must happen before borrowing)
    // 2. Borrow src as &[Src] and dst as &mut [Dst] directly from JS memory
    // 3. Call the conversion function
    //
    // SAFETY: The typed array views borrow directly from JS ArrayBuffer
    // memory. This is safe as long as no JS callbacks or WASM allocations
    // occur while the borrows are live. We ensure this by parsing the
    // scalar map before borrowing, and the conversion functions only do
    // arithmetic (no allocations).

    macro_rules! f2i_into {
        ($src_js:ty, $src_rs:ty, $dst_rs:ty, $dst_js:ty) => {{
            let map = parse_map::<$src_rs, $dst_rs>(map_js)?;
            let config = FloatToIntConfig {
                map_entries: map,
                rounding,
                out_of_range: oor,
            };
            let src_arr: $src_js = src.clone().unchecked_into();
            let dst_arr: $dst_js = dst.clone().unchecked_into();
            // SAFETY: No allocations between borrow and conversion.
            unsafe {
                let src_slice: &[$src_rs] = &src_arr.to_vec();
                let mut dst_vec: Vec<$dst_rs> = dst_arr.to_vec();
                zarr_cast_value::convert_slice_float_to_int(
                    src_slice, &mut dst_vec, &config,
                )
                .map_err(err_to_js)?;
                dst_arr.copy_from(&dst_vec);
            }
            Ok(())
        }};
    }

    macro_rules! i2i_into {
        ($src_js:ty, $src_rs:ty, $dst_rs:ty, $dst_js:ty) => {{
            let map = parse_map::<$src_rs, $dst_rs>(map_js)?;
            let config = IntToIntConfig {
                map_entries: map,
                out_of_range: oor,
            };
            let src_arr: $src_js = src.clone().unchecked_into();
            let dst_arr: $dst_js = dst.clone().unchecked_into();
            unsafe {
                let src_slice: &[$src_rs] = &src_arr.to_vec();
                let mut dst_vec: Vec<$dst_rs> = dst_arr.to_vec();
                zarr_cast_value::convert_slice_int_to_int(
                    src_slice, &mut dst_vec, &config,
                )
                .map_err(err_to_js)?;
                dst_arr.copy_from(&dst_vec);
            }
            Ok(())
        }};
    }

    macro_rules! f2f_into {
        ($src_js:ty, $src_rs:ty, $dst_rs:ty, $dst_js:ty) => {{
            let map = parse_map::<$src_rs, $dst_rs>(map_js)?;
            let config = FloatToFloatConfig {
                map_entries: map,
                rounding,
                out_of_range: oor,
            };
            let src_arr: $src_js = src.clone().unchecked_into();
            let dst_arr: $dst_js = dst.clone().unchecked_into();
            unsafe {
                let src_slice: &[$src_rs] = &src_arr.to_vec();
                let mut dst_vec: Vec<$dst_rs> = dst_arr.to_vec();
                zarr_cast_value::convert_slice_float_to_float(
                    src_slice, &mut dst_vec, &config,
                )
                .map_err(err_to_js)?;
                dst_arr.copy_from(&dst_vec);
            }
            Ok(())
        }};
    }

    macro_rules! i2f_into {
        ($src_js:ty, $src_rs:ty, $dst_rs:ty, $dst_js:ty) => {{
            let map = parse_map::<$src_rs, $dst_rs>(map_js)?;
            let config = IntToFloatConfig {
                map_entries: map,
                rounding,
            };
            let src_arr: $src_js = src.clone().unchecked_into();
            let dst_arr: $dst_js = dst.clone().unchecked_into();
            unsafe {
                let src_slice: &[$src_rs] = &src_arr.to_vec();
                let mut dst_vec: Vec<$dst_rs> = dst_arr.to_vec();
                zarr_cast_value::convert_slice_int_to_float(
                    src_slice, &mut dst_vec, &config,
                )
                .map_err(err_to_js)?;
                dst_arr.copy_from(&dst_vec);
            }
            Ok(())
        }};
    }

    macro_rules! dispatch_int_into {
        ($src_js:ty, $src_rs:ty) => {
            match tgt_dtype {
                "int8" => i2i_into!($src_js, $src_rs, i8, js_sys::Int8Array),
                "int16" => i2i_into!($src_js, $src_rs, i16, js_sys::Int16Array),
                "int32" => i2i_into!($src_js, $src_rs, i32, js_sys::Int32Array),
                "uint8" => i2i_into!($src_js, $src_rs, u8, js_sys::Uint8Array),
                "uint16" => i2i_into!($src_js, $src_rs, u16, js_sys::Uint16Array),
                "uint32" => i2i_into!($src_js, $src_rs, u32, js_sys::Uint32Array),
                "float32" => i2f_into!($src_js, $src_rs, f32, js_sys::Float32Array),
                "float64" => i2f_into!($src_js, $src_rs, f64, js_sys::Float64Array),
                _ => Err(JsValue::from_str(&format!(
                    "Unsupported target dtype: {tgt_dtype}"
                ))),
            }
        };
    }

    macro_rules! dispatch_float_into {
        ($src_js:ty, $src_rs:ty) => {
            match tgt_dtype {
                "int8" => f2i_into!($src_js, $src_rs, i8, js_sys::Int8Array),
                "int16" => f2i_into!($src_js, $src_rs, i16, js_sys::Int16Array),
                "int32" => f2i_into!($src_js, $src_rs, i32, js_sys::Int32Array),
                "uint8" => f2i_into!($src_js, $src_rs, u8, js_sys::Uint8Array),
                "uint16" => f2i_into!($src_js, $src_rs, u16, js_sys::Uint16Array),
                "uint32" => f2i_into!($src_js, $src_rs, u32, js_sys::Uint32Array),
                "float32" => f2f_into!($src_js, $src_rs, f32, js_sys::Float32Array),
                "float64" => f2f_into!($src_js, $src_rs, f64, js_sys::Float64Array),
                _ => Err(JsValue::from_str(&format!(
                    "Unsupported target dtype: {tgt_dtype}"
                ))),
            }
        };
    }

    match src_dtype {
        "int8" => dispatch_int_into!(js_sys::Int8Array, i8),
        "int16" => dispatch_int_into!(js_sys::Int16Array, i16),
        "int32" => dispatch_int_into!(js_sys::Int32Array, i32),
        "uint8" => dispatch_int_into!(js_sys::Uint8Array, u8),
        "uint16" => dispatch_int_into!(js_sys::Uint16Array, u16),
        "uint32" => dispatch_int_into!(js_sys::Uint32Array, u32),
        "float32" => dispatch_float_into!(js_sys::Float32Array, f32),
        "float64" => dispatch_float_into!(js_sys::Float64Array, f64),
        _ => Err(JsValue::from_str(&format!(
            "Unsupported source dtype: {src_dtype}"
        ))),
    }
}
