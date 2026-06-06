from typing import reveal_type

import dask.array as da
import numpy as np
import zarr

x = np.zeros((3, 4, 5), dtype=np.uint8)
y = da.zeros((3, 4, 5), dtype=np.uint8)
z = zarr.zeros((3, 4, 5), another=int, dtype=np.uint8)

reveal_type(x)
reveal_type(y)
reveal_type(z)
