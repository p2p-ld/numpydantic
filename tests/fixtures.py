import shutil
from pathlib import Path

import pytest
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition


@pytest.fixture(scope="session")
def tmp_output_dir() -> Path:
    path = Path(__file__).parent.resolve() / "__tmp__"
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir()

    return path


@pytest.fixture(scope="function")
def tmp_output_dir_func(tmp_output_dir) -> Path:
    """
    tmp output dir that gets cleared between every function
    cleans at the start rather than at cleanup in case the output is to be inspected
    """
    subpath = tmp_output_dir / "__tmpfunc__"
    if subpath.exists():
        shutil.rmtree(str(subpath))
    subpath.mkdir()
    return subpath


@pytest.fixture(scope="module")
def tmp_output_dir_mod(tmp_output_dir) -> Path:
    """
    tmp output dir that gets cleared between every function
    cleans at the start rather than at cleanup in case the output is to be inspected
    """
    subpath = tmp_output_dir / "__tmpmod__"
    if subpath.exists():
        shutil.rmtree(str(subpath))
    subpath.mkdir()
    return subpath


@pytest.fixture()
def nwb_linkml_array() -> tuple[ClassDefinition, str]:
    classdef = ClassDefinition(
        name="NWB_Linkml Array",
        description="Main class's array",
        is_a="Arraylike",
        attributes=[
            SlotDefinition(name="x", range="numeric", required=True),
            SlotDefinition(name="y", range="numeric", required=True),
            SlotDefinition(
                name="z",
                range="numeric",
                required=False,
                maximum_cardinality=3,
                minimum_cardinality=3,
            ),
            SlotDefinition(
                name="a",
                range="numeric",
                required=False,
                minimum_cardinality=4,
                maximum_cardinality=4,
            ),
        ],
    )
    generated = """Union[
        NDArray[Shape["* x, * y"], Number],
        NDArray[Shape["* x, * y, 3 z"], Number],
        NDArray[Shape["* x, * y, 3 z, 4 a"], Number]
    ]"""
    return classdef, generated
