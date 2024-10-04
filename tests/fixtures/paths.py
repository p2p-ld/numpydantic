import shutil
from _warnings import warn
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tmp_output_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).parents[1].resolve() / "__tmp__"
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir()

    yield path

    if not request.config.getvalue("--with-output"):
        try:
            shutil.rmtree(str(path))
        except PermissionError as e:
            # sporadic error on windows machines...
            warn(
                "Temporary directory could not be removed due to a permissions error: "
                f"\n{str(e)}"
            )


@pytest.fixture(scope="function")
def tmp_output_dir_func(tmp_output_dir, request: pytest.FixtureRequest) -> Path:
    """
    tmp output dir that gets cleared between every function
    cleans at the start rather than at cleanup in case the output is to be inspected
    """
    subpath = tmp_output_dir / f"__tmpfunc_{request.node.name}__"
    if subpath.exists():
        shutil.rmtree(str(subpath))
    subpath.mkdir()
    return subpath


@pytest.fixture(scope="module")
def tmp_output_dir_mod(tmp_output_dir, request: pytest.FixtureRequest) -> Path:
    """
    tmp output dir that gets cleared between every function
    cleans at the start rather than at cleanup in case the output is to be inspected
    """
    subpath = tmp_output_dir / f"__tmpmod_{request.module}__"
    if subpath.exists():
        shutil.rmtree(str(subpath))
    subpath.mkdir()
    return subpath
