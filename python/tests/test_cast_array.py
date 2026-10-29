"""Tests for cast_array (allocating variant)."""

from math import inf, nan

import numpy as np
import pytest

from cast_value_rs import cast_array, cast_array_into

from .conftest import LAYOUT_DTYPE_PATHS, Expect, ExpectFail, layout_arrays, nan_eq

# ---------------------------------------------------------------------------
# float -> int
# ---------------------------------------------------------------------------

FLOAT_TO_INT_CASES = [
    Expect(
        input=dict(
            arr=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([1, 2, 3], dtype=np.uint8),
        id="basic",
    ),
    Expect(
        input=dict(
            arr=np.array([2.5, 3.5], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([2, 4], dtype=np.uint8),
        id="nearest-even",
    ),
    Expect(
        input=dict(
            arr=np.array([2.7, -2.7], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="towards-zero",
        ),
        expected=np.array([2, -2], dtype=np.int8),
        id="towards-zero",
    ),
    Expect(
        input=dict(
            arr=np.array([2.1, -2.7], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="towards-positive",
        ),
        expected=np.array([3, -2], dtype=np.int8),
        id="towards-positive",
    ),
    Expect(
        input=dict(
            arr=np.array([2.7, -2.1], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="towards-negative",
        ),
        expected=np.array([2, -3], dtype=np.int8),
        id="towards-negative",
    ),
    Expect(
        input=dict(
            arr=np.array([2.5, -2.5], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="nearest-away",
        ),
        expected=np.array([3, -3], dtype=np.int8),
        id="nearest-away",
    ),
    Expect(
        input=dict(
            arr=np.array([300.0, -10.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="clamp",
        ),
        expected=np.array([255, 0], dtype=np.uint8),
        id="clamp",
    ),
    Expect(
        input=dict(
            arr=np.array([200.0], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="nearest-even",
            out_of_range_mode="wrap",
        ),
        expected=np.array([-56], dtype=np.int8),
        id="wrap",
    ),
    Expect(
        input=dict(
            arr=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            target_dtype="int16",
            rounding_mode="nearest-even",
        ),
        expected=np.array([1, 2, 3], dtype=np.int16),
        id="f32-source",
    ),
]


@pytest.mark.parametrize(
    "case",
    FLOAT_TO_INT_CASES,
    ids=[c.id for c in FLOAT_TO_INT_CASES],
)
def test_float_to_int(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


FLOAT_TO_INT_FAIL_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([300.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        exception=ValueError, match="out of range",
        id="out-of-range",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([nan], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        exception=ValueError, match="NaN",
        id="nan",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([inf], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="nearest-even",
            out_of_range_mode="wrap",
        ),
        exception=ValueError, match="cast",
        id="inf-wrap",
    ),
    # Integers cannot represent ±Inf, so even clamp must fail (spec).
    ExpectFail(
        input=dict(
            arr=np.array([inf], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="clamp",
        ),
        exception=ValueError, match="cast",
        id="inf-clamp",
    ),
]


@pytest.mark.parametrize(
    "case",
    FLOAT_TO_INT_FAIL_CASES,
    ids=[c.id for c in FLOAT_TO_INT_FAIL_CASES],
)
def test_float_to_int_errors(case: ExpectFail):
    case.check(cast_array)


# ---------------------------------------------------------------------------
# int -> int
# ---------------------------------------------------------------------------

INT_TO_INT_CASES = [
    Expect(
        input=dict(
            arr=np.array([1, 2, 3], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([1, 2, 3], dtype=np.uint8),
        id="basic",
    ),
    Expect(
        input=dict(
            arr=np.array([300, -1], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="clamp",
        ),
        expected=np.array([255, 0], dtype=np.uint8),
        id="clamp",
    ),
    Expect(
        input=dict(
            arr=np.array([300, -1], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="wrap",
        ),
        expected=np.array([44, 255], dtype=np.uint8),
        id="wrap",
    ),
    Expect(
        input=dict(
            arr=np.array([0, 127, -128], dtype=np.int8),
            target_dtype="int32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([0, 127, -128], dtype=np.int32),
        id="widening",
    ),
]


@pytest.mark.parametrize(
    "case", INT_TO_INT_CASES, ids=[c.id for c in INT_TO_INT_CASES]
)
def test_int_to_int(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


INT_TO_INT_FAIL_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([300], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        exception=ValueError, match="out of range",
        id="out-of-range",
    ),
]


@pytest.mark.parametrize(
    "case", INT_TO_INT_FAIL_CASES, ids=[c.id for c in INT_TO_INT_FAIL_CASES]
)
def test_int_to_int_errors(case: ExpectFail):
    case.check(cast_array)


# ---------------------------------------------------------------------------
# float -> float
# ---------------------------------------------------------------------------

FLOAT_TO_FLOAT_CASES = [
    Expect(
        input=dict(
            arr=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        id="f64-to-f32",
    ),
    Expect(
        input=dict(
            arr=np.array([nan], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([nan], dtype=np.float32),
        eq=nan_eq,
        id="nan-propagates",
    ),
    Expect(
        input=dict(
            arr=np.array([inf, -inf], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([inf, -inf], dtype=np.float32),
        id="inf-propagates",
    ),
    Expect(
        input=dict(
            arr=np.array([-0.0], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([-0.0], dtype=np.float32),
        eq=lambda a, b: bool(
            np.array_equal(a, b) and np.signbit(a[0]) == np.signbit(b[0])
        ),
        id="signed-zero",
    ),
]


@pytest.mark.parametrize(
    "case", FLOAT_TO_FLOAT_CASES, ids=[c.id for c in FLOAT_TO_FLOAT_CASES]
)
def test_float_to_float(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


_overflow_val = float(np.finfo(np.float32).max) * 1.5

FLOAT_TO_FLOAT_FAIL_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([_overflow_val], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        exception=ValueError, match="out of range",
        id="overflow",
    ),
]


@pytest.mark.parametrize(
    "case",
    FLOAT_TO_FLOAT_FAIL_CASES,
    ids=[c.id for c in FLOAT_TO_FLOAT_FAIL_CASES],
)
def test_float_to_float_errors(case: ExpectFail):
    case.check(cast_array)


def test_float_to_float_overflow_clamp():
    result = cast_array(
        np.array([_overflow_val], dtype=np.float64),
        target_dtype="float32",
        rounding_mode="nearest-even",
        out_of_range_mode="clamp",
    )
    assert np.isinf(result[0]) and result[0] > 0


# ---------------------------------------------------------------------------
# int -> float
# ---------------------------------------------------------------------------

INT_TO_FLOAT_CASES = [
    Expect(
        input=dict(
            arr=np.array([0, -1, 32767, -32768], dtype=np.int16),
            target_dtype="float64",
            rounding_mode="nearest-even",
        ),
        expected=np.array([0.0, -1.0, 32767.0, -32768.0], dtype=np.float64),
        id="lossless-widening",
    ),
    Expect(
        input=dict(
            arr=np.array([70000, -70000], dtype=np.int32),
            target_dtype="float16",
            rounding_mode="nearest-even",
            out_of_range_mode="clamp",
        ),
        expected=np.array([inf, -inf], dtype=np.float16),
        id="float16-overflow-clamp",
    ),
]


@pytest.mark.parametrize(
    ("value", "dtype", "rounding_mode"),
    [
        (70000, "int32", "nearest-even"),
        # 65519 rounds to float16's max (65504) with nearest-even, but up
        # past it towards +Inf, so it is out of range only in this mode.
        (65519, "uint16", "towards-positive"),
    ],
)
def test_int_to_float16_overflow_error(value, dtype, rounding_mode):
    """Integers rounding beyond float16's finite range fail without
    out_of_range, in both the allocating and the `into` variant."""
    arr = np.array([value], dtype=dtype)
    with pytest.raises(ValueError, match="out of range"):
        cast_array(arr, target_dtype="float16", rounding_mode=rounding_mode)
    with pytest.raises(ValueError, match="out of range"):
        cast_array_into(
            arr, np.empty(1, dtype=np.float16), rounding_mode=rounding_mode
        )


def test_int_to_float16_directed_rounding_near_max():
    """65535 overflows float16 with nearest-even, but towards-zero rounds it
    onto the maximum, which is in range."""
    arr = np.array([65535], dtype=np.uint16)
    result = cast_array(arr, target_dtype="float16", rounding_mode="towards-zero")
    np.testing.assert_array_equal(result, np.array([65504], dtype=np.float16))
    out = np.empty(1, dtype=np.float16)
    cast_array_into(arr, out, rounding_mode="towards-zero")
    np.testing.assert_array_equal(out, np.array([65504], dtype=np.float16))


@pytest.mark.parametrize(
    "case", INT_TO_FLOAT_CASES, ids=[c.id for c in INT_TO_FLOAT_CASES]
)
def test_int_to_float(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


def test_int_to_float_rounding():
    """i64 values near 2^24 can't all be exactly represented in f32."""
    val = (1 << 24) + 1  # 16777217
    arr = np.array([val], dtype=np.int64)

    result_pos = cast_array(
        arr, target_dtype="float32", rounding_mode="towards-positive"
    )
    assert float(result_pos[0]) >= val

    result_tz = cast_array(
        arr, target_dtype="float32", rounding_mode="towards-zero"
    )
    assert float(result_tz[0]) <= val


# ---------------------------------------------------------------------------
# shape / misc
# ---------------------------------------------------------------------------

SHAPE_CASES = [
    Expect(
        input=dict(
            arr=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([[1, 2], [3, 4]], dtype=np.uint8),
        id="2d",
    ),
    Expect(
        input=dict(
            arr=np.array([], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([], dtype=np.uint8),
        id="empty",
    ),
]


@pytest.mark.parametrize("case", SHAPE_CASES, ids=[c.id for c in SHAPE_CASES])
def test_shape(case: Expect):
    result = cast_array(**case.input)
    assert result.shape == case.expected.shape
    case.check(result)


# ---------------------------------------------------------------------------
# Memory layout
# ---------------------------------------------------------------------------

# A numpy array's buffer only matches logical (row-major) order when the array
# is row-major. Casting must follow the logical order for every layout, not the
# order the elements happen to sit in memory.
MEMORY_LAYOUT_CASES = [
    Expect(
        input=dict(arr=arr, target_dtype=tgt, rounding_mode="nearest-even"),
        expected=arr.astype(tgt),
        id=f"{layout}-{src}-to-{tgt}",
    )
    for src, tgt in LAYOUT_DTYPE_PATHS
    for layout, arr in layout_arrays(src)
]


@pytest.mark.parametrize(
    "case", MEMORY_LAYOUT_CASES, ids=[c.id for c in MEMORY_LAYOUT_CASES]
)
def test_memory_layout(case: Expect):
    """Casting depends on logical element order, never on memory layout."""
    result = cast_array(**case.input)
    assert result.shape == case.input["arr"].shape
    case.check(result)


def test_memory_layout_clamp():
    """Non-contiguous input through the clamp fast path.

    Clamp with a supported rounding mode selects the SIMD kernels; make sure
    layout normalization holds on that path too, not just the scalar one.
    """
    arr = np.array([[-1.0, 300.0, 5.0], [7.0, 260.0, -9.0]], dtype=np.float32).T
    assert not arr.flags["C_CONTIGUOUS"]
    result = cast_array(
        arr,
        target_dtype="uint8",
        rounding_mode="nearest-even",
        out_of_range_mode="clamp",
    )
    assert np.array_equal(result, np.clip(arr, 0, 255).astype(np.uint8))


def test_memory_layout_scalar_map():
    """Scalar map entries match values, not memory positions, on a view."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4).T
    result = cast_array(
        arr,
        target_dtype="uint16",
        rounding_mode="nearest-even",
        scalar_map_entries=[(5.0, 500)],
    )
    expected = arr.astype(np.uint16)
    expected[arr == 5.0] = 500
    assert np.array_equal(result, expected)
