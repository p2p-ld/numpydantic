import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce
from itertools import product
from operator import ior
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
)

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from numpydantic import NDArray, Shape
from numpydantic.dtype import Float
from numpydantic.interface import Interface, InterfaceTyping
from numpydantic.types import DtypeType, NDArrayType

if TYPE_CHECKING:
    from _pytest.mark.structures import MarkDecorator


class InterfaceCase(ABC):
    """
    An interface test helper that allows a given interface to generate and validate
    arrays in one of its formats.

    Each instance of "interface test case" should be considered one of the
    potentially multiple realizations of a given interface.
    If an interface has multiple formats (eg. zarr's different `store` s),
    then it should have several test helpers.
    """

    @property
    @abstractmethod
    def interface(self) -> Interface:
        """The interface that this helper is for"""

    @classmethod
    def array_from_case(
        cls, case: "ValidationCase", path: Path | None = None
    ) -> NDArrayType | None:
        """
        Generate an array from the given validation case.

        Returns ``None`` if an array can't be generated for a specific case.
        """
        return cls.make_array(shape=case.shape, dtype=case.dtype, path=path)

    @classmethod
    @abstractmethod
    def make_array(
        cls,
        shape: tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Path | None = None,
        array: NDArrayType | None = None,
    ) -> NDArrayType | None:
        """
        Make an array from a shape and dtype, and a path if needed

        Args:
            shape: shape of the array
            dtype: dtype of the array
            path: Path, if needed to generate on disk
            array: Rather than passing shape and dtype, pass a literal arraylike thing
        """

    @classmethod
    def validate_case(cls, case: "ValidationCase", path: Path) -> bool:
        """
        Validate a generated array against the annotation in the validation case.

        Kept in the InterfaceCase in case an interface has specific
        needs aside from just validating against a model, but typically left as is.

        If an array can't be generated for a given case, returns `None`
        so that the calling function can know to skip rather than fail the case.

        Raises exceptions if validation fails (or succeeds when it shouldn't)

        Args:
            case (ValidationCase): The validation case to validate.
            path (Path): Path to generate arrays into, if any.

        Returns:
            ``True`` if array is valid and was supposed to be,
                or invalid and wasn't supposed to be
        """
        import pytest

        array = cls.array_from_case(case, path)
        if array is None:
            pytest.skip()
        if case.passes:
            case.model(array=array)
            return True
        else:
            with pytest.raises(ValidationError):
                case.model(array=array)
            return True

    @classmethod
    def skip(cls, shape: tuple[int, ...], dtype: DtypeType) -> bool:
        """
        Whether a given interface should be skipped for the case
        """
        # Assume an interface case is valid for all other cases
        return False


_a_shape_type = tuple[int | Literal["*"] | Literal["..."], ...]


class ValidationCase(BaseModel):
    """
    Test case for validating an array.

    Contains both the validating model and the parameterization for an array to
    test in a given interface
    """

    id: str | None = None
    """
    String identifying the validation case
    """
    annotation_shape: None | tuple[int | str, ...] | tuple[tuple[int | str, ...], ...] = (
        10,
        10,
        "*",
        "*",
    )
    """
    Shape to use in computed annotation used to validate against
    """
    annotation_dtype: DtypeType | Sequence[DtypeType] = Float
    """
    Dtype to use in computed annotation used to validate against
    """
    shape: tuple[int, ...] = (10, 10, 2, 2)
    """Shape of the array to validate"""
    dtype: type | np.dtype = float
    """Dtype of the array to validate"""
    passes: bool = False
    """Whether the validation should pass or not"""
    interface: type[InterfaceCase] | None = None
    """The interface test case to generate and validate the array with"""
    path: Path | None = None
    """The path to generate arrays into, if any."""
    marks: set[str] = Field(default_factory=set)
    """pytest marks to set for this test case"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field()
    def annotation(self) -> NDArray:
        """
        Annotation used in the model we validate against
        """
        # make a union type if we need to
        shape_union = (
            False
            if self.annotation_shape is None
            else all(
                isinstance(s, Sequence) and not isinstance(s, str)
                for s in self.annotation_shape
            )
        )
        dtype_union = isinstance(self.annotation_dtype, Sequence) and all(
            isinstance(s, Sequence) for s in self.annotation_dtype
        )
        if shape_union or dtype_union:
            shape_iter = (
                self.annotation_shape if shape_union else [self.annotation_shape]
            )
            dtype_iter = (
                self.annotation_dtype if dtype_union else [self.annotation_dtype]
            )
            annotations: list[type] = []
            for shape, dtype in product(shape_iter, dtype_iter):
                if shape is None:
                    annotations.append(NDArray[Any, dtype])
                else:
                    shape_str = ", ".join([str(i) for i in shape])
                    annotations.append(NDArray[Shape[shape_str], dtype])
            return Union[tuple(annotations)]  # noqa: UP007

        else:
            if self.annotation_shape is None:
                return NDArray[Any, self.annotation_dtype]
            else:
                shape_str = ", ".join([str(i) for i in self.annotation_shape])
                return NDArray[Shape[shape_str], self.annotation_dtype]

    @computed_field()
    def model(self) -> type[BaseModel]:
        """A model with a field ``array`` with the given annotation"""
        annotation = self.annotation

        class Model(BaseModel):
            array: annotation

        return Model

    @property
    def pytest_marks(self) -> list["MarkDecorator"]:
        """
        Instantiated pytest marks from :attr:`.ValidationCase.marks`
        plus the interface name.
        """
        import pytest

        marks = self.marks.copy()
        if self.interface is not None:
            marks.add(self.interface.interface.name)
        return [getattr(pytest.mark, m) for m in marks]

    def validate_case(self, path: Path | None = None) -> bool:
        """
        Whether the generated array correctly validated against the annotation,
        given the interface

        Args:
            path (:class:`pathlib.Path`): Directory to generate array into, if on disk.

        Raises:
            ValueError: if an ``interface`` is missing
        """
        if self.interface is None:  # pragma: no cover
            raise ValueError("Missing an interface")
        if path is None:
            if self.path:
                path = self.path
            else:  # pragma: no cover
                raise ValueError("Missing a path to generate arrays into")

        return self.interface.validate_case(self, path)

    def array(self, path: Path) -> NDArrayType:
        """Generate an array for the validation case if we have an interface to do so"""
        if self.interface is None:  # pragma: no cover
            raise ValueError("Missing an interface")
        if path is None:  # pragma: no cover
            if self.path:
                path = self.path
            else:
                raise ValueError("Missing a path to generate arrays into")

        return self.interface.array_from_case(self, path)

    def merge(
        self, other: Union["ValidationCase", Sequence["ValidationCase"]]
    ) -> "ValidationCase":
        """
        Merge two validation cases

        Dump both, excluding any unset fields, and merge, preferring `other`.

        ``valid`` is ``True`` if and only if it is ``True`` in both.
        """
        if isinstance(other, Sequence):
            return merge_cases(self, *other)
        else:
            return merge_cases(self, other)

    def skip(self) -> bool:
        """
        Whether this case should be skipped
        (eg. due to the interface case being incompatible
        with the requested dtype or shape)
        """
        return bool(
            self.interface is not None and self.interface.skip(self.shape, self.dtype)
        )

    def emit_mypy_source(self) -> str | None:
        """
        Full ``.py`` source for one mypy test case,
        assigning the dtype/shape annotation to a variable annotated with the
        annotation_dtype/annotation_shape.

        Returns ``None`` when the interface has no typing class,
        and thus has not opted-in to static typing.
        """
        if not self.interface.interface.typing:
            return None
        typer: InterfaceTyping = self.interface.interface.typing

        imports = typer.emit_imports() + [
            "from typing import Any",
            "from numpydantic import NDArray, Shape",
            "from numpydantic.dtype import Integer, Float",
            "import numpy",
        ]
        imports = [i for i in imports if i]

        rhs_dtype = _render_dtype_annotation(self.dtype)
        constructor = typer.emit_constructor_source(self.shape, rhs_dtype)
        rhs_shape = ",".join(str(s) for s in self.shape)
        rhs_annotation = f'NDArray[Shape["{rhs_shape}"], {rhs_dtype}]'

        if self.annotation_dtype == Float:
            # FIXME: tuple types were a bad idea.
            # handle the default case, we don't want to expand the tuple,
            # but type check as it would actually be declared.
            lhs_dtype = "Float"
        else:
            lhs_dtype = _render_dtype_annotation(self.annotation_dtype)
        lhs_shape = ",".join(str(s) for s in self.annotation_shape)
        lhs_annotation = f'NDArray[Shape["{lhs_shape}"], {lhs_dtype}]'
        return _MYPY_TEMPLATE.format(
            imports="\n".join(imports),
            constructor=constructor,
            rhs_annotation=rhs_annotation,
            lhs_annotation=lhs_annotation,
        )


_MYPY_TEMPLATE = """
{imports}
import sys
if sys.version_info < (3, 11):
    from typing_extensions import reveal_type
else:
    from typing import reveal_type

def make() -> {rhs_annotation}:
    array = {constructor}
    reveal_type(array)
    return array
    
reveal_type(make)

x: {lhs_annotation} = make()
reveal_type(x)
"""


def _render_dtype_annotation(a_dtype: type | str) -> str:
    """
    Render a dtype annotation as a source fragment.
    """
    if a_dtype is Any:
        return "Any"

    elif isinstance(a_dtype, str):
        return a_dtype

    elif isinstance(a_dtype, types.UnionType):
        parts = []
        for d in typing.get_args(a_dtype):
            rendered = _render_dtype_annotation(d)
            parts.append(rendered)
        return " | ".join(parts)

    elif isinstance(a_dtype, Sequence) and not isinstance(a_dtype, str):
        parts = [_render_dtype_annotation(d) for d in a_dtype]
        return " | ".join(parts) if len(parts) > 1 else parts[0]

    else:
        # finally, if we've got a type, render it as module.name
        if a_dtype.__module__ == "builtins":
            return a_dtype.__name__
        return f"{a_dtype.__module__}.{a_dtype.__name__}"


def merge_cases(*args: ValidationCase) -> ValidationCase:
    """
    Merge multiple validation cases
    """
    if len(args) == 1:  # pragma: no cover
        return args[0]

    dumped = [
        m.model_dump(
            exclude_unset=True, exclude={"model", "annotation", "pytest_marks"}
        )
        for m in args
    ]

    # self_dump = self.model_dump(exclude_unset=True)
    # other_dump = other.model_dump(exclude_unset=True)

    # dumps might not have set `passes`, use only the ones that have
    passes = [v.get("passes") for v in dumped if "passes" in v]
    passes = all(passes)

    # combine ids if present
    ids = "-".join([str(v.get("id")) for v in dumped if "id" in v])

    # merge dicts
    merged = reduce(ior, dumped, {})
    merged["passes"] = passes
    merged["id"] = ids
    merged["marks"] = set().union(*[v.get("marks", set()) for v in dumped])
    return ValidationCase.model_construct(**merged)


def merged_product(
    *args: Sequence[ValidationCase], conditions: dict = None
) -> list[ValidationCase]:
    """
    Generator for the product of the iterators of validation cases,
    merging each tuple, and respecting if they should be :meth:`.ValidationCase.skip`
    or not.

    Examples:

        .. code-block:: python

            shape_cases = [
                ValidationCase(shape=(10, 10, 10), passes=True, id="valid shape"),
                ValidationCase(shape=(10, 10), passes=False, id="missing dimension"),
            ]
            dtype_cases = [
                ValidationCase(dtype=float, passes=True, id="float"),
                ValidationCase(dtype=int, passes=False, id="int"),
            ]

            iterator = merged_product(shape_cases, dtype_cases))
            next(iterator)
            # ValidationCase(
            #     shape=(10, 10, 10),
            #     dtype=float,
            #     passes=True,
            #     id="valid shape-float"
            # )
            next(iterator)
            # ValidationCase(
            #     shape=(10, 10, 10),
            #     dtype=int,
            #     passes=False,
            #     id="valid shape-int"
            # )


    """
    iterator = product(*args)
    cases = []
    for case_tuple in iterator:
        case = merge_cases(*case_tuple)
        if case.skip():
            continue
        if conditions:
            matching = all([getattr(case, k, None) == v for k, v in conditions.items()])
            if not matching:
                continue
        cases.append(case)
    return cases
