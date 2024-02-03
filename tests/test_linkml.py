"""
Test custom features of the pydantic generator

Note that since this is largely a subclass, we don't test all of the functionality of the generator
because it's tested in the base linkml package.
"""
import re
import sys
import typing

import numpy as np
import pytest
from pydantic import BaseModel


def test_arraylike(imported_schema):
    """
    Arraylike classes are converted to slots that specify nptyping arrays

    array: Optional[Union[
        NDArray[Shape["* x, * y"], Number],
        NDArray[Shape["* x, * y, 3 z"], Number],
        NDArray[Shape["* x, * y, 3 z, 4 a"], Number]
    ]] = Field(None)
    """
    # check that we have gotten an NDArray annotation and its shape is correct
    array = imported_schema["core"].MainTopLevel.model_fields["array"].annotation
    args = typing.get_args(array)
    for i, shape in enumerate(("* x, * y", "* x, * y, 3 z", "* x, * y, 3 z, 4 a")):
        assert isinstance(args[i], NDArrayMeta)
        assert args[i].__args__[0].__args__
        assert args[i].__args__[1] == np.number

    # we shouldn't have an actual class for the array
    assert not hasattr(imported_schema["core"], "MainTopLevel__Array")
    assert not hasattr(imported_schema["core"], "MainTopLevelArray")


def test_inject_fields(imported_schema):
    """
    Our root model should have the special fields we injected
    """
    base = imported_schema["core"].ConfiguredBaseModel
    assert "hdf5_path" in base.model_fields
    assert "object_id" in base.model_fields


def test_linkml_meta(imported_schema):
    """
    We should be able to store some linkml metadata with our classes
    """
    meta = imported_schema["core"].LinkML_Meta
    assert "tree_root" in meta.model_fields
    assert imported_schema["core"].MainTopLevel.linkml_meta.default.tree_root == True
    assert imported_schema["core"].OtherClass.linkml_meta.default.tree_root == False


def test_skip(linkml_schema):
    """
    We can skip slots and classes
    """
    modules = generate_and_import(
        linkml_schema,
        split=False,
        generator_kwargs={
            "SKIP_SLOTS": ("SkippableSlot",),
            "SKIP_CLASSES": ("Skippable", "skippable"),
        },
    )
    assert not hasattr(modules["core"], "Skippable")
    assert "SkippableSlot" not in modules["core"].MainTopLevel.model_fields


def test_inline_with_identifier(imported_schema):
    """
    By default, if a class has an identifier attribute, it is inlined
    as a string rather than its class. We overrode that to be able to make dictionaries of collections
    """
    main = imported_schema["core"].MainTopLevel
    inline = main.model_fields["inline_dict"].annotation
    assert typing.get_origin(typing.get_args(inline)[0]) == dict
    # god i hate pythons typing interface
    otherclass, stillanother = typing.get_args(
        typing.get_args(typing.get_args(inline)[0])[1]
    )
    assert otherclass is imported_schema["core"].OtherClass
    assert stillanother is imported_schema["core"].StillAnotherClass


def test_namespace(imported_schema):
    """
    Namespace schema import all classes from the other schema
    Returns:

    """
    ns = imported_schema["namespace"]

    for classname, modname in (
        ("MainThing", "test_schema.imported"),
        ("Arraylike", "test_schema.imported"),
        ("MainTopLevel", "test_schema.core"),
        ("Skippable", "test_schema.core"),
        ("OtherClass", "test_schema.core"),
        ("StillAnotherClass", "test_schema.core"),
    ):
        assert hasattr(ns, classname)
        if imported_schema["split"]:
            assert getattr(ns, classname).__module__ == modname


def test_get_set_item(imported_schema):
    """We can get and set without explicitly addressing array"""
    cls = imported_schema["core"].MainTopLevel(array=np.array([[1, 2, 3], [4, 5, 6]]))
    cls[0] = 50
    assert (cls[0] == 50).all()
    assert (cls.array[0] == 50).all()

    cls[1, 1] = 100
    assert cls[1, 1] == 100
    assert cls.array[1, 1] == 100
