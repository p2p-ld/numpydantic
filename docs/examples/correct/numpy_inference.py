from typing import reveal_type

import numpy as np

x = np.ones((1, 2, 3), dtype=np.float32)
reveal_type(x)
