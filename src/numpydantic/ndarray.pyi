from typing import Any, TypeVar

import numpy as np

from numpydantic.types import DtypeType
from numpydantic.validation.shape import Shape

_T_Shape = TypeVar("_T_Shape", bound=Shape | tuple, default=Any)
_T_Dtype = TypeVar("_T_Dtype", bound=DtypeType, default=Any)

NDArray = np.ndarray[_T_Shape | tuple, np.dtype[_T_Dtype]]
