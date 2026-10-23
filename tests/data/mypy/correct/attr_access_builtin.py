import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray, Shape


class GroundTruth(BaseModel):
    A: NDArray[Shape[int, int, int], float]


def use(gt: GroundTruth) -> tuple[int, ...]:
    # some standard array function
    gt.A.sum()
    return gt.A.shape


inst = GroundTruth(A=np.zeros((3, 3, 3)))
use(inst)
