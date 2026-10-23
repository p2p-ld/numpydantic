from typing import Any, TypeVar

import numpy as np

from numpydantic.types import DtypeType

_T_Shape = TypeVar("_T_Shape", bound=tuple, default=Any)
_T_Dtype = TypeVar("_T_Dtype", bound=DtypeType, default=Any)

NDArray = np.ndarray[_T_Shape, np.dtype[_T_Dtype]]
