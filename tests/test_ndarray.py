import json
from typing import Any, Optional, Union

import numpy as np
import pytest
from pydantic import BaseModel, Field, ValidationError

from numpydantic import NDArray, Shape, dtype
from numpydantic.dtype import Number
from numpydantic.exceptions import DtypeError


@pytest.mark.json_schema
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
    _ = Model(array=np.zeros((2, 3)))

    with pytest.raises(ValidationError):
        _ = Model(array=np.zeros((4, 6)))

    with pytest.raises(ValidationError):
        _ = Model(array=np.ones((2, 3), dtype=bool))

    _ = Model(array=np.zeros((2, 3)), array_any=np.ones((3, 4, 5)))


@pytest.mark.dtype
@pytest.mark.json_schema
def test_schema_unsupported_type():
    """
    Complex numbers should just be made with an `any` schema
    """

    class Model(BaseModel):
        array: NDArray[Shape["2 x, * y"], complex]

    schema = Model.model_json_schema()
    assert schema["properties"]["array"]["items"] == {
        "items": {},
        "type": "array",
    }


@pytest.mark.dtype
@pytest.mark.json_schema
def test_schema_tuple():
    """
    Types specified as tuples should have their schemas as a union
    """

    class Model(BaseModel):
        array: NDArray[Shape["2 x, * y"], (np.uint8, np.uint16)]

    schema = Model.model_json_schema()
    assert "anyOf" in schema["properties"]["array"]["items"]["items"]
    conditions = schema["properties"]["array"]["items"]["items"]["anyOf"]

    assert all([i["type"] == "integer" for i in conditions])
    assert sorted([i["maximum"] for i in conditions]) == [255, 65535]
    assert all([i["minimum"] == 0 for i in conditions])


@pytest.mark.dtype
@pytest.mark.json_schema
def test_schema_number():
    """
    np.numeric should just be the float schema
    """

    class Model(BaseModel):
        array: NDArray[Shape["2 x, * y"], np.number]

    schema = Model.model_json_schema()
    assert schema["properties"]["array"]["items"] == {
        "items": {"type": "number"},
        "type": "array",
    }


def test_ndarray_union():
    generator = np.random.default_rng()

    class Model(BaseModel):
        array: Optional[
            Union[
                NDArray[Shape["* x, * y"], Number],
                NDArray[Shape["* x, * y, 3 r_g_b"], Number],
                NDArray[Shape["* x, * y, 3 r_g_b, 4 r_g_b_a"], Number],
            ]
        ] = Field(None)

    _ = Model()
    _ = Model(array=generator.random((5, 10)))
    _ = Model(array=generator.random((5, 10, 3)))
    _ = Model(array=generator.random((5, 10, 3, 4)))

    with pytest.raises(ValidationError):
        _ = Model(array=generator.random((5,)))

    with pytest.raises(ValidationError):
        _ = Model(array=generator.random((5, 10, 4)))

    with pytest.raises(ValidationError):
        _ = Model(array=generator.random((5, 10, 3, 6)))

    with pytest.raises(ValidationError):
        _ = Model(array=generator.random((5, 10, 4, 6)))


@pytest.mark.shape
@pytest.mark.dtype
@pytest.mark.parametrize("dtype", dtype.Number)
def test_ndarray_unparameterized(dtype):
    """
    NDArray without any parameters is any shape, any type
    """
    generator = np.random.default_rng()

    class Model(BaseModel):
        array: NDArray

    # not very sophisticated fuzzing of "any shape"
    test_cases = 10
    for _ in range(test_cases):
        n_dimensions = generator.integers(1, 8)
        dim_sizes = generator.integers(1, 7, size=n_dimensions)
        _ = Model(array=np.zeros(dim_sizes, dtype=dtype))


@pytest.mark.shape
def test_ndarray_any():
    """
    using :class:`typing.Any` in for the shape means any shape
    """
    generator = np.random.default_rng()

    class Model(BaseModel):
        array: NDArray[Any, np.uint8]

    # not very sophisticated fuzzing of "any shape"
    test_cases = 100
    for _ in range(test_cases):
        n_dimensions = generator.integers(1, 8)
        dim_sizes = generator.integers(1, 16, size=n_dimensions)
        _ = Model(array=np.zeros(dim_sizes, dtype=np.uint8))


def test_ndarray_coercion():
    """
    Coerce lists to arrays
    """

    class Model(BaseModel):
        array: NDArray[Shape["* x"], Number]

    amod = Model(array=[1, 2, 3, 4.5])
    assert np.allclose(amod.array, np.array([1, 2, 3, 4.5]))
    with pytest.raises(ValidationError):
        amod = Model(array=["a", "b", "c"])


@pytest.mark.shape
def test_shape_ellipsis():
    """
    Test that ellipsis is a wildcard, rather than "repeat the last index"
    """

    class MyModel(BaseModel):
        array: NDArray[Shape["1, 2, ..."], Number]

    _ = MyModel(array=np.zeros((1, 2, 3, 4, 5)))


@pytest.mark.serialization
def test_ndarray_serialize():
    """
    Arrays should be dumped to a list when using json, but kept as ndarray otherwise
    """

    class Model(BaseModel):
        array: NDArray[Any, Number]

    mod = Model(array=np.random.default_rng().random((3, 3)))
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


@pytest.mark.json_schema
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


@pytest.mark.dtype
@pytest.mark.json_schema
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


@pytest.mark.dtype
@pytest.mark.json_schema
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


@pytest.mark.dtype
@pytest.mark.json_schema
def test_json_schema_dtype_model():
    """
    Pydantic models can be used in arrays as dtypes
    """

    class TestModel(BaseModel):
        x: int
        y: int
        z: int

    class MyModel(BaseModel):
        array: NDArray[Shape["*, *"], TestModel]

    schema = MyModel.model_json_schema()
    # we should have a "$defs" with TestModel in it,
    # and our array should be objects of that type
    assert schema["properties"]["array"]["items"]["items"] == {
        "$ref": "#/$defs/TestModel"
    }
    # we don't test pydantic' generic json schema model generation,
    # just that one was defined
    assert "TestModel" in schema["$defs"]


def _recursive_array(schema):
    assert "$defs" in schema
    # get the key uses for the array
    array_key = list(schema["$defs"].keys())[0]

    # the array property should be a ref to the recursive array
    # get the innermost part of the field schema
    field_schema = schema["properties"]["array"]
    while "items" in field_schema:
        field_schema = field_schema["items"]
    assert field_schema["$ref"] == f"#/$defs/{array_key}"

    # and the recursive array should indeed be recursive...
    # specifically it should be an array whose items can be itself or
    # of the type specified by the dtype
    any_of = schema["$defs"][array_key]["anyOf"]
    assert any_of[0]["items"]["$ref"] == f"#/$defs/{array_key}"
    assert any_of[0]["type"] == "array"
    # here we are just assuming that it's a uint8 array..
    assert any_of[1]["type"] == "integer"
    assert any_of[1]["maximum"] == 255
    assert any_of[1]["minimum"] == 0


@pytest.mark.shape
@pytest.mark.json_schema
def test_json_schema_ellipsis():
    """
    NDArray types should create a recursive JSON schema for any-shaped arrays
    """

    class AnyShape(BaseModel):
        array: NDArray[Shape["*, ..."], np.uint8]

    schema = AnyShape.model_json_schema()
    _recursive_array(schema)

    class ConstrainedAnyShape(BaseModel):
        array: NDArray[Shape["3, 4, ..."], np.uint8]

    schema = ConstrainedAnyShape.model_json_schema()
    _recursive_array(schema)


def test_instancecheck():
    """
    NDArray should handle ``isinstance()`` s.t. valid arrays are ``True``
    and invalid arrays are ``False``

    We don't make this test exhaustive because correctness of validation
    is tested elsewhere. We are just testing that the type checking works
    """
    array_type = NDArray[Shape["1, 2, 3"], int]

    assert isinstance(np.zeros((1, 2, 3), dtype=int), array_type)
    assert not isinstance(np.zeros((2, 2, 3), dtype=int), array_type)
    assert not isinstance(np.zeros((1, 2, 3), dtype=float), array_type)

    def my_function(array: NDArray[Shape["1, 2, 3"], int]):
        return array

    my_function(np.zeros((1, 2, 3), int))


def test_callable():
    """
    NDArray objects are callable to validate and cast
    Don't test validation here, just that we can be called
    """
    annotation = NDArray[Shape["3"], int]
    array = np.array([1, 2, 3], dtype=int)
    validated = annotation(array)
    assert validated is array

    with pytest.raises(DtypeError):
        _ = annotation(np.zeros((1, 2, 3)))
