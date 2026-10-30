from typing import Literal

import numpy as np

from numpydantic import NDArray


def g(frame: NDArray[tuple[Literal[1080], Literal[1920]], np.uint8]) -> None: ...


rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
g(rgb)  # E: incompatible type
