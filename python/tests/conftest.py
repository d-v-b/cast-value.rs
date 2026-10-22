from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest


def default_eq(a: Any, b: Any) -> bool:
    """Default equality: element-wise comparison via numpy."""
    return bool(np.array_equal(a, b))


def nan_eq(a: Any, b: Any) -> bool:
    """Equality that treats NaN == NaN as True."""
    return bool(np.array_equal(a, b, equal_nan=True))


@dataclass
class Expect:
    """A test case: call the function on `input`, check it matches `expected`."""

    input: Any
    expected: Any
    eq: Callable[[Any, Any], bool] = field(default=default_eq)
    id: str | None = None

    def check(self, actual: Any) -> None:
        assert self.eq(actual, self.expected), f"expected {self.expected!r}, got {actual!r}"


@dataclass
class ExpectFail:
    """A test case that should raise an exception."""

    input: Any
    exception: type[Exception]
    match: str
    id: str | None = None

    def check(self, fn: Callable[..., Any]) -> None:
        with pytest.raises(self.exception, match=self.match):
            fn(**self.input)


# One (src, tgt) pair per conversion path in the bindings, so a layout
# regression in any of the four helpers is caught, plus float16 for its
# special handling.
LAYOUT_DTYPE_PATHS = [
    ("float64", "uint16"),  # float -> int
    ("int32", "uint8"),  # int -> int
    ("float64", "float32"),  # float -> float
    ("int32", "float32"),  # int -> float
    ("float16", "uint8"),  # f16 source
]


def layout_arrays(dtype: str) -> list[tuple[str, np.ndarray]]:
    """Integer-valued arrays of `dtype` in every memory layout we support.

    Integer values in [0, 20) are exactly representable in every dtype in
    LAYOUT_DTYPE_PATHS, so the expected result is independent of rounding
    and clamping.
    """
    flat = np.arange(20, dtype=dtype)
    grid = np.arange(12, dtype=dtype).reshape(3, 4)
    cube = np.arange(20, dtype=dtype).reshape(5, 2, 2)
    return [
        ("row-major", grid),
        ("column-major", grid.T),
        ("transposed-3d", cube.transpose(1, 2, 0)),
        ("strided", flat[::2]),
        ("negative-stride", grid[::-1]),
        ("sliced-view", grid[:, 1:3]),
        ("zero-dim", np.array(7, dtype=dtype)),
        ("empty", np.zeros((0, 3), dtype=dtype)),
    ]
