"""
Isolated generator for array classes
"""

import warnings
from abc import ABC, abstractmethod

from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition

from numpydantic.maps import flat_to_nptyping


class ArrayFormat(ABC):
    """
    Metaclass for different LinkML array source formats
    """

    @classmethod
    def is_array(cls, cls_: ClassDefinition) -> bool:
        """Check whether a given class matches one of our subclasses definitions"""
        return any([subcls.check(cls_) for subcls in cls.__subclasses__()])

    @classmethod
    def get(cls, cls_: ClassDefinition) -> type["ArrayFormat"]:
        """Get matching ArrayFormat subclass"""
        for subcls in cls.__subclasses__():
            if subcls.check(cls_):
                return subcls

    @classmethod
    @abstractmethod
    def check(cls, cls_: ClassDefinition) -> bool:
        """Method for array format subclasses to check if they match a given source class"""

    @classmethod
    @abstractmethod
    def make(cls, cls_: ClassDefinition) -> str:
        """
        Make an annotation string from a given array format source class
        """


class LinkMLNDArray(ArrayFormat):
    """
    Tentative linkml-arrays style NDArray
    """

    @classmethod
    def check(cls, cls_: ClassDefinition) -> bool:
        """Check if linkml:NDArray in implements"""
        return "linkml:NDArray" in cls_.implements

    @classmethod
    def make(cls, cls_: ClassDefinition) -> str:
        """Make NDArray"""
        raise NotImplementedError("Havent implemented NDArrays yet!")


class LinkMLDataArray(ArrayFormat):
    """
    Tentative linkml-arrays style annotated array with indices
    """

    @classmethod
    def check(cls, cls_: ClassDefinition) -> bool:
        """Check if linkml:DataArray in implements"""
        return "linkml:DataArray" in cls_.implements

    @classmethod
    def make(cls, cls_: ClassDefinition) -> str:
        """Make DataArray"""
        raise NotImplementedError("Havent generated DataArray types yet!")


class NWBLinkMLArraylike(ArrayFormat):
    """
    Ye Olde nwb-linkml Arraylike class

    Examples:

        TimeSeries:
          is_a: Arraylike
          attributes:
            num_times:
              name: num_times
              range: AnyType
              required: true
            num_DIM2:
              name: num_DIM2
              range: AnyType
              required: false
            num_DIM3:
              name: num_DIM3
              range: AnyType
              required: false
            num_DIM4:
              name: num_DIM4
              range: AnyType
              required: false
    """

    @classmethod
    def check(cls, cls_: ClassDefinition) -> bool:
        """Check if class is Arraylike"""
        return cls_.is_a == "Arraylike"

    @classmethod
    def make(cls, cls_: ClassDefinition) -> str:
        """Make Arraylike annotation"""
        return cls._array_annotation(cls_)

    @classmethod
    def _array_annotation(cls, cls_: ClassDefinition) -> str:
        """
        Make an annotation for an NDArray :)

        Args:
            cls_:

        Returns:

        """
        # if none of the dimensions are optional, we just have one possible array shape
        if all([s.required for s in cls_.attributes.values()]):  # pragma: no cover
            return cls._make_npytyping_range(cls_.attributes)
        # otherwise we need to make permutations
        # but not all permutations, because we typically just want to be able to exlude the last possible dimensions
        # the array classes should always be well-defined where the optional dimensions are at the end, so
        requireds = {k: v for k, v in cls_.attributes.items() if v.required}
        optionals = [(k, v) for k, v in cls_.attributes.items() if not v.required]

        annotations = []
        if len(requireds) > 0:
            # first the base case
            annotations.append(cls._make_npytyping_range(requireds))
        # then add back each optional dimension
        for i in range(len(optionals)):
            attrs = {**requireds, **{k: v for k, v in optionals[0 : i + 1]}}
            annotations.append(cls._make_npytyping_range(attrs))

        # now combine with a union:
        union = "Union[\n" + " " * 8
        union += (",\n" + " " * 8).join(annotations)
        union += "\n" + " " * 4 + "]"
        return union

    @classmethod
    def _make_npytyping_range(cls, attrs: dict[str, SlotDefinition]) -> str:
        # slot always starts with...
        prefix = "NDArray["

        # and then we specify the shape:
        shape_prefix = 'Shape["'

        # using the cardinality from the attributes
        dim_pieces = []
        for attr in attrs.values():
            shape_part = (
                str(attr.maximum_cardinality) if attr.maximum_cardinality else "*"
            )

            # do this with the most heinous chain of string replacements rather than regex
            # because i am still figuring out what needs to be subbed lol
            name_part = (
                attr.name.replace(",", "_")
                .replace(" ", "_")
                .replace("__", "_")
                .replace("|", "_")
                .replace("-", "_")
                .replace("+", "plus")
            )

            dim_pieces.append(" ".join([shape_part, name_part]))

        dimension = ", ".join(dim_pieces)

        shape_suffix = '"], '

        # all dimensions should be the same dtype
        try:
            dtype = flat_to_nptyping[list(attrs.values())[0].range]
        except KeyError as e:  # pragma: no cover
            warnings.warn(str(e), stacklevel=2)
            range = list(attrs.values())[0].range
            return f"List[{range}] | {range}"
        suffix = "]"

        slot = "".join([prefix, shape_prefix, dimension, shape_suffix, dtype, suffix])
        return slot
