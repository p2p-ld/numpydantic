import pdb
import json

from pydantic import BaseModel


def test_to_json(hdf5_array, array_model):
    """
    Test serialization of HDF5 arrays to JSON
    Args:
        hdf5_array:

    Returns:

    """
    array = hdf5_array((10, 10), int)
    model = array_model((10, 10), int)

    instance = model(array=array)  # type: BaseModel

    json_str = instance.model_dump_json()
    json_dict = json.loads(json_str)["array"]

    assert json_dict["file"] == str(array.file)
    assert json_dict["path"] == str(array.path)
    assert json_dict["attrs"] == {}
    assert json_dict["array"] == instance.array[:].tolist()
