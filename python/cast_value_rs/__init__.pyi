from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from cast_value_rs._cast_value_rs import (
    DTypeName as DTypeName,
    OutOfRangeMode as OutOfRangeMode,
    RoundingMode as RoundingMode,
    cast_array_into as cast_array_into,
)

def cast_array(
    arr: npt.NDArray[np.generic],
    *,
    target_dtype: DTypeName | np.dtype[np.generic] | type[np.generic],
    rounding_mode: RoundingMode,
    out_of_range_mode: OutOfRangeMode | None = None,
    scalar_map_entries: dict[float, float] | Iterable[tuple[float, float]] | None = None,
) -> npt.NDArray[np.generic]: ...
