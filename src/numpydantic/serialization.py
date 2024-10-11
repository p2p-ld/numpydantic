"""
Serialization helpers for :func:`pydantic.BaseModel.model_dump`
and :func:`pydantic.BaseModel.model_dump_json` .
"""

from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar, Union

from pydantic_core.core_schema import SerializationInfo

from numpydantic.interface import Interface, JsonDict

T = TypeVar("T")
U = TypeVar("U")


def jsonize_array(value: Any, info: SerializationInfo) -> Union[list, dict]:
    """Use an interface class to render an array as JSON"""
    # perf: keys to skip in generation - anything named "value" is array data.
    skip = ["value"]

    interface_cls = Interface.match_output(value)
    array = interface_cls.to_json(value, info)
    if isinstance(array, JsonDict):
        array = array.model_dump(exclude_none=True)

    if info.context:
        if info.context.get("mark_interface", False):
            array = interface_cls.mark_json(array)

        if isinstance(array, list):
            return array

        # ---- Perf Barrier ------------------------------------------------------
        # put context args intended to **wrap** the array above
        # put context args intended to **modify** the array below
        #
        # above, we assume that a list is **data** not to be modified.
        # below, we must mark whenever the data is in the line of fire
        # to avoid an expensive iteration.

        if info.context.get("absolute_paths", False):
            array = _absolutize_paths(array, skip)
        else:
            relative_to = info.context.get("relative_to", ".")
            array = _relativize_paths(array, relative_to, skip)
    else:
        if isinstance(array, list):
            return array

        # ---- Perf Barrier ------------------------------------------------------
        # same as above, ensure any keys that contain array values are skipped right now

        array = _relativize_paths(array, ".", skip)

    return array


def _relativize_paths(
    value: dict, relative_to: str = ".", skip: Iterable = tuple()
) -> dict:
    """
    Make paths relative to either the current directory or the provided
    ``relative_to`` directory, if provided in the context
    """
    relative_to = Path(relative_to).resolve()

    def _r_path(v: Any) -> Any:
        if not isinstance(v, (str, Path)):
            return v
        try:
            path = Path(v)
            resolved = path.resolve()
            # skip things that are pathlike but either don't exist
            # or that are at the filesystem root (eg like /data)
            if (
                not path.exists()
                or (resolved.is_dir() and str(resolved.parent) == resolved.anchor)
                or relative_to.anchor != resolved.anchor
            ):
                return v
            return str(relative_path(path, relative_to))
        except (TypeError, ValueError, OSError):
            return v

    return _walk_and_apply(value, _r_path, skip)


def _absolutize_paths(value: dict, skip: Iterable = tuple()) -> dict:
    def _a_path(v: Any) -> Any:
        if not isinstance(v, (str, Path)):
            return v
        try:
            path = Path(v)
            if not path.exists():
                return v
            return str(path.resolve())
        except (TypeError, ValueError, OSError):
            return v

    return _walk_and_apply(value, _a_path, skip)


def _walk_and_apply(value: T, f: Callable[[U, bool], U], skip: Iterable = tuple()) -> T:
    """
    Walk an object, applying a function
    """
    if isinstance(value, dict):
        for k, v in value.items():
            if k in skip:
                continue
            if isinstance(v, dict):
                _walk_and_apply(v, f, skip)
            elif isinstance(v, list):
                value[k] = [_walk_and_apply(sub_v, f, skip) for sub_v in v]
            else:
                value[k] = f(v)
    elif isinstance(value, list):
        value = [_walk_and_apply(v, f, skip) for v in value]
    else:
        value = f(value)
    return value


def relative_path(self: Path, other: Path, walk_up: bool = True) -> Path:
    """
    "Backport" of :meth:`pathlib.Path.relative_to` with ``walk_up=True``
    that's not available pre 3.12.

    Return the relative path to another path identified by the passed
    arguments.  If the operation is not possible (because this is not
    related to the other path), raise ValueError.

    The *walk_up* parameter controls whether `..` may be used to resolve
    the path.

    References:
        https://github.com/python/cpython/blob/8a2baedc4bcb606da937e4e066b4b3a18961cace/Lib/pathlib/_abc.py#L244-L270
    """
    # pdb.set_trace()
    if not isinstance(other, Path):  # pragma: no cover - ripped from cpython
        other = Path(other)
    self_parts = self.parts
    other_parts = other.parts
    anchor0, parts0 = self_parts[0], list(reversed(self_parts[1:]))
    anchor1, parts1 = other_parts[0], list(reversed(other_parts[1:]))
    if anchor0 != anchor1:
        raise ValueError(f"{self!r} and {other!r} have different anchors")
    while parts0 and parts1 and parts0[-1] == parts1[-1]:
        parts0.pop()
        parts1.pop()
    for part in parts1:  # pragma: no cover - not testing, ripped off from cpython
        if not part or part == ".":
            pass
        elif not walk_up:
            raise ValueError(f"{self!r} is not in the subpath of {other!r}")
        elif part == "..":
            raise ValueError(f"'..' segment in {other!r} cannot be walked")
        else:
            parts0.append("..")
    return Path(*reversed(parts0))
