import pytest

from typing import Union, Optional, Any
import json

import numpy as np
from pydantic import BaseModel, ValidationError, Field
from nptyping import Shape, Number

from numpydantic import NDArray
from numpydantic.proxy import NDArrayProxy


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

    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
        amod = Model(array=["a", "b", "c"])


def test_ndarray_serialize():
    """
    Large arrays should get compressed with blosc, otherwise just to list
    """

    class Model(BaseModel):
        large_array: NDArray[Any, Number]
        small_array: NDArray[Any, Number]

    mod = Model(
        large_array=np.random.random((1024, 1024)), small_array=np.random.random((3, 3))
    )
    mod_str = mod.model_dump_json()
    mod_json = json.loads(mod_str)
    for a in ("array", "shape", "dtype", "unpack_fns"):
        assert a in mod_json["large_array"].keys()
    assert isinstance(mod_json["large_array"]["array"], str)
    assert isinstance(mod_json["small_array"], list)

    # but when we just dump to a dict we don't compress
    mod_dict = mod.model_dump()
    assert isinstance(mod_dict["large_array"], np.ndarray)


# def test_ndarray_proxy(tmp_output_dir_func):
#     h5f_source = tmp_output_dir_func / 'test.h5'
#     with h5py.File(h5f_source, 'w') as h5f:
#         dset_good = h5f.create_dataset('/data', data=np.random.random((1024,1024,3)))
#         dset_bad = h5f.create_dataset('/data_bad', data=np.random.random((1024, 1024, 4)))
#
#     class Model(BaseModel):
#         array: NDArray[Shape["* x, * y, 3 z"], Number]
#
#     mod = Model(array=NDArrayProxy(h5f_file=h5f_source, path='/data'))
#     subarray = mod.array[0:5, 0:5, :]
#     assert isinstance(subarray, np.ndarray)
#     assert isinstance(subarray.sum(), float)
#     assert mod.array.name == '/data'
#
#     with pytest.raises(NotImplementedError):
#         mod.array[0] = 5
#
#     with pytest.raises(ValidationError):
#         mod = Model(array=NDArrayProxy(h5f_file=h5f_source, path='/data_bad'))
