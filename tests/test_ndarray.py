import pdb

import pytest

from typing import Union, Optional, Any
import json

import numpy as np
from pydantic import BaseModel, ValidationError, Field
from nptyping import Shape, Number

from numpydantic import NDArray
from numpydantic.exceptions import ShapeError, DtypeError
from numpydantic import dtype


# from .fixtures import tmp_output_dir_func


def test_ndarray_type():
    class Model(BaseModel):
        array: NDArray[Shape["2 x, * y"], Number]
        array_any: Optional[NDArray[Any, Any]] = None

    schema = Model.model_json_schema()
    assert schema["properties"]["array"]["items"] == {
        "items": {"type": "number"},
        "type": "array",
    }
    assert schema["properties"]["array"]["maxItems"] == 2
    assert schema["properties"]["array"]["minItems"] == 2

    # models should instantiate correctly!
    instance = Model(array=np.zeros((2, 3)))

    with pytest.raises(ValidationError):
        instance = Model(array=np.zeros((4, 6)))

    with pytest.raises(DtypeError):
        instance = Model(array=np.ones((2, 3), dtype=bool))

    instance = Model(array=np.zeros((2, 3)), array_any=np.ones((3, 4, 5)))


def test_ndarray_union():
    class Model(BaseModel):
        array: Optional[
            Union[
                NDArray[Shape["* x, * y"], Number],
                NDArray[Shape["* x, * y, 3 r_g_b"], Number],
                NDArray[Shape["* x, * y, 3 r_g_b, 4 r_g_b_a"], Number],
            ]
        ] = Field(None)

    instance = Model()
    instance = Model(array=np.random.random((5, 10)))
    instance = Model(array=np.random.random((5, 10, 3)))
    instance = Model(array=np.random.random((5, 10, 3, 4)))

    with pytest.raises(ValidationError):
        instance = Model(array=np.random.random((5,)))

    with pytest.raises(ValidationError):
        instance = Model(array=np.random.random((5, 10, 4)))

    with pytest.raises(ValidationError):
        instance = Model(array=np.random.random((5, 10, 3, 6)))

    with pytest.raises(ValidationError):
        instance = Model(array=np.random.random((5, 10, 4, 6)))


def test_ndarray_coercion():
    """
    Coerce lists to arrays
    """

    class Model(BaseModel):
        array: NDArray[Shape["* x"], Number]

    amod = Model(array=[1, 2, 3, 4.5])
    assert np.allclose(amod.array, np.array([1, 2, 3, 4.5]))
    with pytest.raises(DtypeError):
        amod = Model(array=["a", "b", "c"])


def test_ndarray_serialize():
    """
    Arrays should be dumped to a list when using json, but kept as ndarray otherwise
    """

    class Model(BaseModel):
        array: NDArray[Any, Number]

    mod = Model(array=np.random.random((3, 3)))
    mod_str = mod.model_dump_json()
    mod_json = json.loads(mod_str)
    assert isinstance(mod_json["array"], list)

    # but when we just dump to a dict we don't coerce
    mod_dict = mod.model_dump()
    assert isinstance(mod_dict["array"], np.ndarray)


_json_schema_types = [
    *[(t, float) for t in dtype.Float],
    *[(t, int) for t in dtype.Integer],
]


def test_json_schema_basic(array_model):
    """
    NDArray types should correctly generate a list of lists JSON schema
    """
    shape = (15, 10)
    dtype = float
    model = array_model(shape, dtype)
    schema = model.model_json_schema()
    field = schema["properties"]["array"]

    # outer shape
    assert field["maxItems"] == shape[0]
    assert field["minItems"] == shape[0]
    assert field["type"] == "array"

    # inner shape
    inner = field["items"]
    assert inner["minItems"] == shape[1]
    assert inner["maxItems"] == shape[1]
    assert inner["items"]["type"] == "number"


@pytest.mark.parametrize("dtype", [*dtype.Integer, *dtype.Float])
def test_json_schema_dtype_single(dtype, array_model):
    """
    dtypes should have correct mins and maxes set, and store the source dtype
    """
    if issubclass(dtype, np.floating):
        info = np.finfo(dtype)
        min_val = info.min
        max_val = info.max
        schema_type = "number"
    elif issubclass(dtype, np.integer):
        info = np.iinfo(dtype)
        min_val = info.min
        max_val = info.max
        schema_type = "integer"
    else:
        raise ValueError("These should all be numpy types!")

    shape = (15, 10)
    model = array_model(shape, dtype)
    schema = model.model_json_schema()
    inner_type = schema["properties"]["array"]["items"]["items"]
    assert inner_type["minimum"] == min_val
    assert inner_type["maximum"] == max_val
    assert inner_type["type"] == schema_type
    assert schema["properties"]["array"]["dtype"] == ".".join(
        [dtype.__module__, dtype.__name__]
    )


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (dtype.Integer, "integer"),
        (dtype.Float, "number"),
        (dtype.Bool, "boolean"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
        (complex, "any"),
    ],
)
def test_json_schema_dtype_builtin(dtype, expected, array_model):
    """
    Using builtin or generic (eg. `dtype.Integer` ) dtypes should
    make a simple json schema without mins/maxes/dtypes.
    """
    model = array_model(dtype=dtype)
    schema = model.model_json_schema()
    inner_type = schema["properties"]["array"]["items"]["items"]
    if expected == "any":
        assert inner_type == {}
    else:
        assert inner_type["type"] == expected


@pytest.mark.skip("Not implemented yet")
def test_json_schema_wildcard():
    """
    NDarray types should generate a JSON schema without shape constraints
    """
    pass


@pytest.mark.skip("Not implemented yet")
def test_json_schema_ellipsis():
    """
    NDArray types should create a recursive JSON schema for any-shaped arrays
    """
    pass
