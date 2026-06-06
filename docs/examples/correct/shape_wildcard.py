from typing import cast, reveal_type

import numpy as np

from numpydantic import NDArray, Shape

AT_LEAST_2D = NDArray[Shape["*, *, ..."]]

# two or more dimensions!
x: AT_LEAST_2D = np.ones((2, 3))
y: AT_LEAST_2D = np.ones((2, 3, 4))
z_pre: AT_LEAST_2D = np.ones((2, 3, 4, 5))
z = cast(NDArray[Shape[2, 3, 4, 5]], z_pre)

# The LHS types will lose their constructor enrichment
reveal_type(x)
reveal_type(y)

# casting can rescue it
reveal_type(z)
