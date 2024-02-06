"""
Lol i know all these shoudl be made into parameterized test cases and i will do that after
i get them to work. don't @ me bro
"""

import pdb

import pytest

from ..fixtures import patch_slotarray, slotarray_schemaview, DATA_DIR

from numpydantic.linkml.slotarray import SlotNDArray
from numpydantic.linkml.pydanticgen import PydanticGenerator

# --------------------------------------------------
# Only dimensions
# --------------------------------------------------


def test_exact_dimensions(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("ExactDimension")
    array = cls.attributes["temp"]
    annotation = SlotNDArray.make(array)
    # hardcoding for now...
    assert annotation == 'NDArray[Shape["*, *, *"], Float]'


def test_min_dimensions(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("MinDimensions")
    array = cls.attributes["temp"]
    annotation = SlotNDArray.make(array)
    assert annotation == 'NDArray[Shape["*, *, *, ..."], Float]'


def test_max_dimensions(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("MaxDimensions")
    array = cls.attributes["temp"]
    annotation = SlotNDArray.make(array)
    assert (
        annotation
        == """Union[
        NDArray[Shape["*"], Float],
        NDArray[Shape["*, *"], Float],
        NDArray[Shape["*, *, *"], Float]
    ]"""
    )


def test_range_dimensions(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("RangeDimensions")
    array = cls.attributes["temp"]
    annotation = SlotNDArray.make(array)
    assert (
        annotation
        == """Union[
        NDArray[Shape["*, *"], Float],
        NDArray[Shape["*, *, *"], Float],
        NDArray[Shape["*, *, *, *"], Float],
        NDArray[Shape["*, *, *, *, *"], Float]
    ]"""
    )


def test_exact_cardinality(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("ExactCardinality")
    array = cls.attributes["temp"]
    annotation = SlotNDArray.make(array)
    assert annotation == """NDArray[Shape["3 x"], Float]"""


def test_min_cardinality(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("MinCardinality")
    array = cls.attributes["temp"]
    with pytest.raises(ValueError):
        # no way to specify ranges in nptyping ranges yet, this would go to infinity
        SlotNDArray.make(array)


def test_max_cardinality(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("MaxCardinality")
    array = cls.attributes["temp"]
    annotation = SlotNDArray.make(array)
    assert (
        annotation
        == """Union[
        NDArray[Shape["1 x"], Float],
        NDArray[Shape["2 x"], Float],
        NDArray[Shape["3 x"], Float]
    ]"""
    )


def test_range_cardinality(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("RangeCardinality")
    array = cls.attributes["temp"]
    annotation = SlotNDArray.make(array)

    assert (
        annotation
        == """Union[
        NDArray[Shape["2 x"], Float],
        NDArray[Shape["3 x"], Float],
        NDArray[Shape["4 x"], Float]
    ]"""
    )


def test_exclusive_axes(slotarray_schemaview):
    cls = slotarray_schemaview.get_class("ExclusiveAxes")
    array = cls.attributes["temp"]
    annotation = SlotNDArray.make(array)
    assert (
        annotation
        == """Union[
        NDArray[Shape["* x, * y, 3 rgb"], Float],
        NDArray[Shape["* x, * y, 4 rgba"], Float]
    ]"""
    )


# def test_generate_schema(patch_slotarray):
#     schema = DATA_DIR / "slotarray.yaml"
#     generator = PydanticGenerator(schema)
#     serialized = generator.serialize()
#     pdb.set_trace()


# def test_optional_axes(slotarray_schemaview):
#     cls = slotarray_schemaview.get_class("OptionalAxes")
#     array = cls.attributes["temp"]
#     annotation = SlotNDArray.make(array)
#     # assert annotation == ""
#
#
# def test_mixed_named_unnamed_dimensions(slotarray_schemaview):
#     cls = slotarray_schemaview.get_class("MixedNamedUnnamed")
#     array = cls.attributes["temp"]
#     annotation = SlotNDArray.make(array)
#     # assert annotation == ""
