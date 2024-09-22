"""
Serialization helpers for :func:`pydantic.BaseModel.model_dump`
and :func:`pydantic.BaseModel.model_dump_json` .
"""

from pathlib import Path
from typing import Any, Callable, TypeVar, Union

from pydantic_core.core_schema import SerializationInfo

from numpydantic.interface import Interface, JsonDict

T = TypeVar("T")
U = TypeVar("U")


def jsonize_array(value: Any, info: SerializationInfo) -> Union[list, dict]:
    """Use an interface class to render an array as JSON"""
    interface_cls = Interface.match_output(value)
    array = interface_cls.to_json(value, info)
    if isinstance(array, JsonDict):
        array = array.model_dump(exclude_none=True)

    if info.context:
        if info.context.get("mark_interface", False):
            array = interface_cls.mark_json(array)
        if info.context.get("absolute_paths", False):
            array = _absolutize_paths(array)
        else:
            relative_to = info.context.get("relative_to", ".")
            array = _relativize_paths(array, relative_to)

    return array


def _relativize_paths(value: dict, relative_to: str = ".") -> dict:
    """
    Make paths relative to either the current directory or the provided
    ``relative_to`` directory, if provided in the context
    """
    relative_to = Path(relative_to).resolve()

    def _r_path(v: Any) -> Any:
        try:
            path = Path(v)
            if not path.exists():
                return v
            return str(relative_path(path, relative_to))
        except (TypeError, ValueError):
            return v

    return _walk_and_apply(value, _r_path)


def _absolutize_paths(value: dict) -> dict:
    def _a_path(v: Any) -> Any:
        try:
            path = Path(v)
            if not path.exists():
                return v
            return str(path.resolve())
        except (TypeError, ValueError):
            return v

    return _walk_and_apply(value, _a_path)


def _walk_and_apply(value: T, f: Callable[[U], U]) -> T:
    """
    Walk an object, applying a function
    """
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                _walk_and_apply(v, f)
            elif isinstance(v, list):
                value[k] = [_walk_and_apply(sub_v, f) for sub_v in v]
            else:
                value[k] = f(v)
    elif isinstance(value, list):
        value = [_walk_and_apply(v, f) for v in value]
    else:
        value = f(value)
    return value


def relative_path(target: Path, origin: Path) -> Path:
    """
    return path of target relative to origin, even if they're
    not in the same subpath

    References:
        - https://stackoverflow.com/a/71874881
    """
    try:
        return Path(target).resolve().relative_to(Path(origin).resolve())
    except ValueError:  # target does not start with origin
        # recursion with origin (eventually origin is root so try will succeed)
        return Path("..").joinpath(relative_path(target, Path(origin).parent))
