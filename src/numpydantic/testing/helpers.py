from typing import Any, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, computed_field

from numpydantic import NDArray, Shape
from numpydantic.dtype import Float


class ValidationCase(BaseModel):
    """
    Test case for validating an array.

    Contains both the validating model and the parameterization for an array to
    test in a given interface
    """

    annotation: Any = NDArray[Shape["10, 10, *"], Float]
    """
    Array annotation used in the validating model
    Any typed because the types of type annotations are weird
    """
    shape: Tuple[int, ...] = (10, 10, 10)
    """Shape of the array to validate"""
    dtype: Union[Type, np.dtype] = float
    """Dtype of the array to validate"""
    passes: bool
    """Whether the validation should pass or not"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field()
    def model(self) -> Type[BaseModel]:
        """A model with a field ``array`` with the given annotation"""
        annotation = self.annotation

        class Model(BaseModel):
            array: annotation

        return Model
