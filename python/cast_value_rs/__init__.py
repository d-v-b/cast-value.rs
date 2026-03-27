from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cast_value_rs._cast_value_rs import cast_array as _cast_array
from cast_value_rs._cast_value_rs import cast_array_into

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from cast_value_rs._cast_value_rs import (
        DTypeName,
        OutOfRangeMode,
        RoundingMode,
    )

# Supported dtype name strings, used for validation.
_SUPPORTED_DTYPES = frozenset(
    (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
)


def _resolve_dtype(target_dtype: DTypeName | np.dtype[np.generic] | type[np.generic]) -> str:
    """Normalize a dtype argument to a string name.

    Accepts:
    - A string like ``"uint8"``, ``"float32"``, etc.
    - A numpy dtype object like ``np.dtype("uint8")``
    - A numpy scalar type like ``np.uint8``
    """
    if isinstance(target_dtype, str):
        name = target_dtype
    elif isinstance(target_dtype, np.dtype):
        name = target_dtype.name
    elif isinstance(target_dtype, type) and issubclass(target_dtype, np.generic):
        name = np.dtype(target_dtype).name
    else:
        msg = (
            f"target_dtype must be a string, numpy dtype, or numpy type, "
            f"got {type(target_dtype).__name__}"
        )
        raise TypeError(msg)

    if name not in _SUPPORTED_DTYPES:
        msg = f"Unsupported target dtype: {name}"
        raise TypeError(msg)

    return name


def cast_array(
    arr: npt.NDArray[np.generic],
    *,
    target_dtype: DTypeName | np.dtype[np.generic] | type[np.generic],
    rounding_mode: RoundingMode,
    out_of_range_mode: OutOfRangeMode | None = None,
    scalar_map_entries: (
        dict[float, float] | Iterable[tuple[float, float]] | None
    ) = None,
) -> npt.NDArray[np.generic]:
    """Cast a numpy array to a new dtype, allocating a new output array.

    Parameters
    ----------
    arr
        Input numpy array.
    target_dtype
        Target dtype. Accepts a string name (e.g. ``"uint8"``), a numpy
        dtype object (e.g. ``np.dtype("uint8")``), or a numpy scalar type
        (e.g. ``np.uint8``).
    rounding_mode
        How to round values during conversion.
    out_of_range_mode
        How to handle values outside the target type's range.
        ``None`` means out-of-range values raise an error.
    scalar_map_entries
        Mapping of special source values to target values.

    Returns
    -------
    npt.NDArray[np.generic]
        A new numpy array with the target dtype.
    """
    return _cast_array(
        arr,
        target_dtype=_resolve_dtype(target_dtype),
        rounding_mode=rounding_mode,
        out_of_range_mode=out_of_range_mode,
        scalar_map_entries=scalar_map_entries,
    )


__all__ = ["cast_array", "cast_array_into"]
