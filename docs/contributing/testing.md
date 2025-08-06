---
file_format: mystnb
mystnb:
    output_stderr: remove
    render_text_lexer: python
    render_markdown_format: myst
myst:
    enable_extensions: ["colon_fence"]
---
# Testing

```{code-cell}
---
tags: [hide-cell]
---

from pathlib import Path
from rich.console import Console
from rich.theme import Theme
from rich.style import Style
from rich.color import Color

theme = Theme({
    "repr.call": Style(color=Color.from_rgb(110,191,38), bold=True),
    "repr.attrib_name": Style(color="slate_blue1"),
    "repr.number": Style(color="deep_sky_blue1"),
    "repr.none": Style(color="bright_magenta", italic=True),
    "repr.attrib_name": Style(color="white"),
    "repr.tag_contents": Style(color="light_steel_blue"),
    "repr.str": Style(color="violet") 
})
console = Console(theme=theme)

```

```{note}
Also see the [`numpydantic.testing` API docs](../api/testing/index.md)
and the [Writing an Interface](../interfaces.md) guide
```

Numpydantic exposes a system for combinatoric testing across dtypes, shapes,
and interfaces in the {mod}`numpydantic.testing` module.

These helper classes and functions are included in the distributed package
so they can be used for downstream development of independent interfaces
(though we always welcome contributions!)

## Validation Cases

Each test case is parameterized by a {class}`.ValidationCase`. 

The case is intended to be able to be partially filled in so that multiple
validation cases can be merged together, but also used independently
by falling back on default values.

There are three major parts to a validation case:

- **Annotation specification:** {attr}`~.ValidationCase.annotation_dtype` and
  {attr}`~.ValidationCase.annotation_shape` specifies how the
  {class}`.NDArray` {attr}`.ValidationCase.annotation` that is used to test
  against is generated
- **Array specification:** {attr}`~.ValidationCase.dtype` and {attr}`~.ValidationCase.shape`
  specify that array that will be generated to test against the annotation
- **Interface specification:** An {class}`.InterfaceCase` that refers to
  an {class}`.Interface`, and provides array generation and other auxiliary logic.

Typically, one specifies a dtype along with an annotation dtype or
a shape along with an annotation shape (or implicitly against the defaults for either),
along with a value for `passes` that indicates if that combination is valid.

```{code-cell}
from numpydantic.testing import ValidationCase

dtype_case = ValidationCase(
    id="int_int", 
    dtype=int, 
    annotation_dtype=int, 
    passes=True
)
shape_case = ValidationCase(
    id="cool_shape", 
    shape=(1,2,3), 
    annotation_shape=(1,"*","2-4"), 
    passes=True
)

merged = dtype_case.merge(shape_case)
console.print(merged.model_dump(exclude={'annotation', 'model'}, exclude_unset=True))
```

When merging validation cases, the merged case only `passes` if all the
original cases do.

```{code-cell}
from numpydantic.testing import ValidationCase

dtype_case = ValidationCase(
    id="int_int", 
    dtype=int, 
    annotation_dtype=int, 
    passes=True
)
shape_case = ValidationCase(
    id="uncool_shape", 
    shape=(1,2,3), 
    annotation_shape=(9,8,7), 
    passes=False
)

merged = dtype_case.merge(shape_case)
console.print(merged.model_dump(exclude={'annotation', 'model'}, exclude_unset=True))
```

We provide a convenience function {func}`.merged_product` for creating a merged product of
multiple sets of test cases.

For example, you may want to create a set of dtype and shape cases and validate
against all combinations

```{code-cell}
from numpydantic.testing.helpers import merged_product

dtype_cases = [
    ValidationCase(dtype=int, annotation_dtype=int, passes=True),
    ValidationCase(dtype=int, annotation_dtype=float, passes=False)
]
shape_cases = [
    ValidationCase(shape=(1,2,3), annotation_shape=(1,2,3), passes=True),
    ValidationCase(shape=(4,5,6), annotation_shape=(1,2,3), passes=False)
]

iterator = merged_product(dtype_cases, shape_cases)

console.print([i.model_dump(exclude_unset=True, exclude={'model', 'annotation'}) for i in iterator])

```

You can pass constraints to the {func}`.merged_product` iterator to
filter cases that match some value, for example to get only the cases that pass:

```{code-cell}
iterator = merged_product(dtype_cases, shape_cases, conditions={"passes": True})
console.print([i.model_dump(exclude_unset=True, exclude={'model', 'annotation'}) for i in iterator])
```

## Interface Cases

Validation cases can be paired with interface cases that handle
generating arrays for the given interface from the specification in the
validation case.

Since some array interfaces like Zarr have multiple possible forms
of an array (in memory, on disk, in a zip file, etc.) an interface
may have multiple cases that are important to test against.

The {meth}`.InterfaceCase.make_array` method does what you'd expect it to,
creating an array, and returning the appropriate input type for the interface:

```{code-cell}
from numpydantic.testing.interfaces import NumpyCase, ZarrNestedCase

NumpyCase.make_array(shape=(1,2,3), dtype=float)
```

```{code-cell}
ZarrNestedCase.make_array(shape=(1,2,3), dtype=float, path=Path("__tmp__/zarr_dir"))
```

Interface cases also define when an interface should skip a given test
parameterization. For example, some array formats can't support arbitrary
object serialization, and the video class can only support 8-bit arrays
of a specific shape

```{code-cell}
from numpydantic.testing.interfaces import VideoCase

VideoCase.skip(shape=(1,1), dtype=float)
```

This, and the array generation methods are propagated up into 
a ValidationCase that contains them

```{code-cell}
case = ValidationCase(shape=(1,2,3), dtype=float, interface=VideoCase)
case.skip()
```

The {func}`.merged_product` iterator automatically excludes any
combinations of interfaces and test parameterizations that should be skipped.

## Making Fixtures

Pytest fixtures are a useful way to reuse validation case products.
To keep things tidy, you may want to use marks and ids when creating them
so that you can run tests against specific interfaces or conditions
with the `pytest -m mark` system.

```python
import pytest

@pytest.fixture(
    params=(
        pytest.param(
            p, 
            id=p.id, 
            marks=getattr(pytest.mark, p.interface.interface.name)
        )
        for p in iterator
    )
)
def my_cases(request):
    return request.param
```
