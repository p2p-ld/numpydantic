"""
Experimental Slot-only NDArray specification

References:
    - https://github.com/linkml/linkml-arrays/issues/7#issuecomment-1925999203
    - https://github.com/linkml/linkml-arrays/issues/7#issuecomment-1926195529

"""
import itertools
from dataclasses import dataclass, field, make_dataclass
from typing import Optional

import jsonasobj2
from linkml_runtime.linkml_model import SlotDefinition as SlotDefinition_Base

from numpydantic.linkml import ArrayFormat
from numpydantic.maps import flat_to_nptyping


@dataclass
class RangeAttr:
    min: Optional[int] = None
    max: Optional[int] = None


@dataclass
class DimensionsAttr(RangeAttr):
    """Specification for number of axes in array"""

    pass


@dataclass
class CardinalityAttr(RangeAttr):
    """Specification for size of a given axis"""

    pass


@dataclass
class Axis:
    """Specification of individual axis within an NDArray"""

    rank: Optional[int] = None
    alias: Optional[str] = None
    cardinality: Optional[int | CardinalityAttr] = None
    required: bool = True


@dataclass
class ArraySlot:
    """Specification of an NDArray as a slot"""

    axes: Optional[dict[str, Axis]] = None
    dimensions: Optional[int | DimensionsAttr] = None


def patch_linkml() -> type["SlotDefinition_Base"]:
    """
    Monkeypatch the LinkML runtime models to add the new properties to define a slotarray :)
    """
    import linkml_runtime
    from linkml_runtime.linkml_model.meta import SlotDefinition
    from linkml_runtime.utils import schemaview

    # make the new SlotDefinition...
    newslot = make_dataclass(
        "SlotDefinition",
        [("array", Optional[ArraySlot], field(default=None))],
        bases=(SlotDefinition,),
    )
    linkml_runtime.linkml_model.meta.SlotDefinition = newslot
    linkml_runtime.linkml_model.SlotDefinition = newslot
    schemaview.SlotDefinition = newslot
    return newslot


# janky local patching until we get metamodel amended
SlotDefinition = patch_linkml()


class SlotArrayTemplate:
    def __init__(self, slot: SlotDefinition):
        self.slot = slot

    def make(self) -> str:
        array: ArraySlot = self.slot.array

        shape = self._handle_shape(array)
        dtype = self._handle_dtype(self.slot)
        if isinstance(shape, list):
            shape = [f'NDArray[Shape["{a_shape}"], {dtype}]' for a_shape in shape]
            union = "Union[\n" + " " * 8
            union += (",\n" + " " * 8).join(shape)
            union += "\n" + " " * 4 + "]"
            template = union
        else:
            template = f'NDArray[Shape["{shape}"], {dtype}]'
        return template

    def _handle_shape(self, array: ArraySlot) -> str | list[str]:
        # If we have no axes or dimensions, any shape array
        if (
            getattr(array, "axes", None) is None
            and getattr(array, "dimensions", None) is None
        ):
            return "*, ..."
        elif getattr(array, "axes", None) is None:
            return self._shape_from_dimension(array)
        elif getattr(array, "dimensions", None) is None:
            return self._shape_from_axes(array)
        else:
            raise NotImplementedError("Joint axis and dimension arrays")

    def _shape_from_dimension(self, array: ArraySlot):
        # just dimensions
        if isinstance(array.dimensions, int):
            # exact number of unnamed dimensions
            return ", ".join(["*"] * array.dimensions)
        elif (
            getattr(array.dimensions, "min", None) is None
            and getattr(array.dimensions, "max", None) is None
        ):
            return "*, ..."
        elif (
            getattr(array.dimensions, "min", None) is not None
            and getattr(array.dimensions, "max", None) is None
        ):
            return ", ".join(["*"] * array.dimensions.min) + ", ..."
        else:
            min = (
                1
                if getattr(array.dimensions, "min", None) is None
                else array.dimensions.min
            )
            return [", ".join(["*"] * i) for i in range(min, array.dimensions.max + 1)]

    def _shape_from_axes(self, array: ArraySlot):
        # just axes!
        # FIXME: what in the hell is the matter with jsonasobj2 lmao
        axes = {
            name: Axis(**axis) for name, axis in jsonasobj2.as_dict(array.axes).items()
        }
        indices = [
            ax.rank for ax in axes.values() if getattr(ax, "rank", None) is not None
        ]
        # TODO: Sort by rank first

        if len(indices) == len(axes) and len(set(indices)) < len(axes):
            # mutually exclusive axes
            # FIXME: it's getting to be the braindead hours, this is a total nightmare of a way to check dupe idx
            # for the love of god why. do it better lol. get some food and go to bed. very last thing
            ax_defs = {}
            for name, ax in axes.items():
                if ax.rank not in ax_defs:
                    ax_defs[ax.rank] = []
                ax_defs[ax.rank].append(self._make_axis(name, ax))
            ax_defs = list(ax_defs.values())
        elif len(indices) < len(axes):
            raise NotImplementedError("Optional Axes")
        else:
            # independently defined axes, these are our friends :)
            ax_defs = [self._make_axis(name, ax) for name, ax in axes.items()]

        if all([isinstance(ax_def, str) for ax_def in ax_defs]):
            # no expansion needed
            return ", ".join(ax_defs)
        # expand such that for each axis, we get all possible combinations
        ax_defs = [
            list(ax_def) if not isinstance(ax_def, list) else ax_def
            for ax_def in ax_defs
        ]
        return [", ".join(inner_def) for inner_def in itertools.product(*ax_defs)]

    def _make_axis(self, name: str, ax: Axis) -> str | list[str]:
        if getattr(ax, "alias", None):
            name = ax.alias
        name = name.lower()

        if getattr(ax, "rank", None) is not None and not ax.required:
            # FIXME: Do this in class validation
            raise ValueError("Optional axes cannot be given explicit ranks")

        if getattr(ax, "cardinality", None) is None:
            return f"* {name}"
        elif isinstance(ax.cardinality, int):
            return f"{ax.cardinality} {name}"
        else:
            # FIXME: i remember why i hate dataclasses and love pydantic now lol - dataclass creation is not recursive
            min = ax.cardinality.get("min", 1)
            max = ax.cardinality.get("max", None)
            if max is None:
                raise ValueError("Cannot set minimum cardinality without maximum")
            return [f"{i} {name}" for i in range(min, max + 1)]

    def _handle_dtype(self, slot: SlotDefinition) -> str:
        # TODO: handle any_of, multiple typed arrays
        return flat_to_nptyping[slot.range]


class SlotNDArray(ArrayFormat):
    """
    Prospective array format specified as a slot/attribute alone

    Examples:
        Any shape array

        .. code-block:: yaml

            classes:
              TemperatureDataset:
                attributes:
                  temperatures_in_K:
                    range: float
                    multivalued: true
                    required: true
                    array:


        Exactly 3 unspecified axes/dimensions

        .. code-block:: yaml

            classes:
              TemperatureDataset:
                attributes:
                  temperatures_in_K:
                    range: float
                    multivalued: true
                    required: true
                    unit:
                      ucum_code: K
                    array:
                      dimensions: 3

        Exactly 3 named dimensions

        .. code-block:: yaml

            array:
              axes:
                x:
                  rank: 0
                  alias: latitude_in_deg
                y:
                  rank: 1
                  alias: longitude_in_deg
                t:
                  rank: 2
                  alias: time_in_d

        At least three, at most 5, and between 3 and 5 anonymous dimensions

        .. code-block:: yaml

            array:
              dimensions:
                min: 3

              # or
              dimensions:
                max: 5

              # or
              dimensions:
                min: 3
                max: 5

        One specified, named dimension, and any number of other dimensions

        .. code-block:: yaml

            array:
              dimensions:
                min: 1
                # optionally, to be explicit:
                max: null
              axes:
                x:
                  rank: 0
                  alias: latitude_in_deg

        Axes with specified cardinality

        .. code-block:: yaml

            array:
              axes:
                x:
                  rank: 0
                  cardinality: 3
                y:
                  rank: 1
                  cardinality:
                    min: 2
                z:
                  rank: 2
                  cardinality:
                    min: 3
                    max: 5

        Two required dimensions and two optional dimensions that will generate
        a union of the combinatoric product of the optional dimensions.
        Rank must be unspecified in optional dimensions

        .. code-block:: yaml

            array:
              axes:
                x:
                  rank: 0
                y:
                  rank: 1
                z:
                  cardinality: 3
                  required: false
                theta:
                  cardinality: 4
                  required: false


        Two required dimensions with a third dimension with two mutually exclusive forms
        as indicated by their equal rank

        .. code-block:: yaml

            array:
              axes:
                x:
                  rank: 0
                y:
                  rank: 1
                rgb:
                  rank: 2
                  cardinality: 3
                rgba:
                  rank: 2
                  cardinality: 4



    References:
        - https://github.com/linkml/linkml-arrays/issues/7#issuecomment-1925999203
        - https://github.com/linkml/linkml-arrays/issues/7#issuecomment-1926195529
    """

    DEFINITION_TYPE = "SLOT"

    @classmethod
    def check(cls, SlotDefinition) -> bool:
        if getattr(SlotDefinition, "array", None) is not None:
            return True
        return False

    @classmethod
    def make(cls, cls_: SlotDefinition) -> str:
        return SlotArrayTemplate(cls_).make()
