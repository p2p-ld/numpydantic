import inspect

from pydantic import BaseModel


def default_template(
    pydantic_ver: str = "2", extra_classes: list[type[BaseModel]] | None = None
) -> str:
    """Constructs a default template for pydantic classes based on the version of pydantic"""
    ### HEADER ###
    template = """
{#-

  Jinja2 Template for a pydantic classes
-#}
from __future__ import annotations
from datetime import datetime, date
from enum import Enum
from typing import Dict, Optional, Any, Union, ClassVar, Annotated, TypeVar, List, TYPE_CHECKING
from pydantic import BaseModel as BaseModel, Field"""
    if pydantic_ver == "2":
        template += """
from pydantic import ConfigDict, BeforeValidator
        """
    template += """
from nptyping import Shape, Float, Float32, Double, Float64, LongLong, Int64, Int, Int32, Int16, Short, Int8, UInt, UInt32, UInt16, UInt8, UInt64, Number, String, Unicode, Unicode, Unicode, String, Bool, Datetime64
from nwb_linkml.types import NDArray
import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
if TYPE_CHECKING:
    import numpy as np

{% for import_module, import_classes in imports.items() %}
from {{ import_module }} import (
    {{ import_classes | join(',\n    ') }}
)
{% endfor %}

metamodel_version = "{{metamodel_version}}"
version = "{{version if version else None}}"
"""
    ### BASE MODEL ###
    if pydantic_ver == "1":  # pragma: no cover
        template += """
List = BaseList

class WeakRefShimBaseModel(BaseModel):
   __slots__ = '__weakref__'

class ConfiguredBaseModel(WeakRefShimBaseModel,
                validate_assignment = False,
                validate_all = True,
                underscore_attrs_are_private = True,
                extra = {% if allow_extra %}'allow'{% else %}'forbid'{% endif %},
                arbitrary_types_allowed = True,
                use_enum_values = True):
"""
    else:
        template += """
class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment = True,
        validate_default = True,
        extra = {% if allow_extra %}'allow'{% else %}'forbid'{% endif %},
        arbitrary_types_allowed = True,
        use_enum_values = True
    )
"""
    ### Injected Fields
    template += """
{%- if injected_fields != None -%}
    {% for field in injected_fields %}
    {{ field }}
    {% endfor %}
{%- else -%}
    pass
{%- endif -%}
    """
    ### Getitem
    template += """

    def __getitem__(self, i: slice|int) -> 'np.ndarray':
        if hasattr(self, 'array'):
            return self.array[i]
        else:
            return super().__getitem__(i)

    def __setitem__(self, i: slice|int, value: Any):
        if hasattr(self, 'array'):
            self.array[i] = value
        else:
            super().__setitem__(i, value)
    """

    ### Extra classes
    if extra_classes is not None:
        template += """{{ '\n\n' }}"""
        for cls in extra_classes:
            template += inspect.getsource(cls) + "\n\n"
    ### ENUMS ###
    template += """
{% for e in enums.values() %}
class {{ e.name }}(str, Enum):
    {% if e.description -%}
    \"\"\"
    {{ e.description }}
    \"\"\"
    {%- endif %}
    {% for _, pv in e['values'].items() -%}
    {% if pv.description -%}
    # {{pv.description}}
    {%- endif %}
    {{pv.label}} = "{{pv.value}}"
    {% endfor %}
    {% if not e['values'] -%}
    dummy = "dummy"
    {% endif %}
{% endfor %}
"""
    ### CLASSES ###
    template += """
{%- for c in schema.classes.values() %}
class {{ c.name }}
    {%- if class_isa_plus_mixins[c.name] -%}
        ({{class_isa_plus_mixins[c.name]|join(', ')}})
    {%- else -%}
        (ConfiguredBaseModel)
    {%- endif -%}
                  :
    {% if c.description -%}
    \"\"\"
    {{ c.description }}
    \"\"\"
    {%- endif %}
    {% for attr in c.attributes.values() if c.attributes -%}
    {{attr.name}}:{{ ' ' }}{%- if attr.equals_string -%}
        Literal[{{ predefined_slot_values[c.name][attr.name] }}]
        {%- else -%}
        {{ attr.annotations['python_range'].value }}
        {%- endif -%}
        {%- if attr.annotations['fixed_field'] -%}
        {{ ' ' }}= {{ attr.annotations['fixed_field'].value }}
        {%- else -%}
        {{ ' ' }}= Field(
    {%- if predefined_slot_values[c.name][attr.name] is string -%}
        {{ predefined_slot_values[c.name][attr.name] }}
    {%- elif attr.required -%}
        ...
    {%- else -%}
        None
    {%- endif -%}
    {%- if attr.title != None %}, title="{{attr.title}}"{% endif -%}
    {%- if attr.description %}, description=\"\"\"{{attr.description}}\"\"\"{% endif -%}
    {%- if attr.minimum_value != None %}, ge={{attr.minimum_value}}{% endif -%}
    {%- if attr.maximum_value != None %}, le={{attr.maximum_value}}{% endif -%}
    )
    {%- endif %}
    {% else -%}
    None
    {% endfor %}
{% endfor %}
"""
    ### FWD REFS / REBUILD MODEL ###
    if pydantic_ver == "1":  # pragma: no cover
        template += """
# Update forward refs
# see https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
{% for c in schema.classes.values() -%}
{{ c.name }}.update_forward_refs()
{% endfor %}
"""
    else:
        template += """
# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
{% for c in schema.classes.values() -%}
{{ c.name }}.model_rebuild()
{% endfor %}    
"""
    return template
