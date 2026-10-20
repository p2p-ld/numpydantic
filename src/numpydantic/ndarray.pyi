from typing import Any, TypeVar

import numpy as np

from numpydantic.types import DtypeType
from numpydantic.shape import Shape

_T_Shape = TypeVar("_T_Shape", bound=Shape | tuple, default=Any, contravariant=True)
_T_Dtype = TypeVar("_T_Dtype", bound=DtypeType, default=Any)

NDArray = np.ndarray[_T_Shape, np.dtype[_T_Dtype]]
