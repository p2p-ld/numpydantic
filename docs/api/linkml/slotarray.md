# Slot Arrays

Will explain further in the morning :)

See: 
- [https://github.com/linkml/linkml-arrays/issues/7#issuecomment-1925999203](https://github.com/linkml/linkml-arrays/issues/7#issuecomment-1925999203)
- [https://github.com/linkml/linkml-arrays/issues/7#issuecomment-1926195529](https://github.com/linkml/linkml-arrays/issues/7#issuecomment-1926195529)

## Working Examples

`````{tab-set}
````{tab-item} YAML
```yaml
ExactDimension:
  description: exact anonymous dimensions
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
      array:
        dimensions: 3
```
````

````{tab-item} Pydantic
```python
class ExactDimension(ConfiguredBaseModel):
    """
    exact anonymous dimensions
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: NDArray[Shape[*, *, *], Float] = Field(...)
```
````
`````

`````{tab-set}
````{tab-item} YAML
```yaml
ExactNamedDimension:
  description: Exact named dimensions
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
      array:
        axes:
          x:
            rank: 0
            alias: latitude
          y:
            rank: 1
            alias: longitude
          t:
            rank: 2
            alias: time
```
````
````{tab-item} Pydantic
```python
class ExactNamedDimension(ConfiguredBaseModel):
    """
    Exact named dimensions
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: NDArray[Shape[* latitude, * longitude, * time], Float] = Field(...)
```    
````
`````

`````{tab-set}
````{tab-item} YAML
```yaml
MinDimensions:
  description: Minimum anonymous dimensions
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
      array:
        dimensions:
          min: 3
```
````
````{tab-item} Pydantic
```python
class MinDimensions(ConfiguredBaseModel):
    """
    Minimum anonymous dimensions
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: NDArray[Shape[*, *, *, ...], Float] = Field(...)
```  
````
`````

`````{tab-set}
````{tab-item} YAML
```yaml
MaxDimensions:
  description: Maximum anonymous dimensions
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
      array:
        dimensions:
          max: 3
```
````
````{tab-item} Pydantic
```python
class MaxDimensions(ConfiguredBaseModel):
    """
    Maximum anonymous dimensions
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: Union[
        NDArray[Shape["*"], Float],
        NDArray[Shape["*, *"], Float],
        NDArray[Shape["*, *, *"], Float]
    ] = Field(...)
```    
````
`````

`````{tab-set}
````{tab-item} YAML
```yaml
RangeDimensions:
  description: Range of anonymous dimensions
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
      array:
        dimensions:
          min: 2
          max: 5
```
````
````{tab-item} Pydantic
```python
class RangeDimensions(ConfiguredBaseModel):
    """
    Range of anonymous dimensions
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: Union[
        NDArray[Shape["*, *"], Float],
        NDArray[Shape["*, *, *"], Float],
        NDArray[Shape["*, *, *, *"], Float],
        NDArray[Shape["*, *, *, *, *"], Float]
    ] = Field(...)
```   
````
`````

`````{tab-set}
````{tab-item} YAML
```yaml
ExactCardinality:
  description: An axis with a specified cardinality
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
      array:
        axes:
          x:
            rank: 0
            cardinality: 3
```
````
````{tab-item} Pydantic
```python
class ExactCardinality(ConfiguredBaseModel):
    """
    An axis with a specified cardinality
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: NDArray[Shape["3 x"], Float] = Field(...)
```    
````
`````

`````{tab-set}
````{tab-item} YAML
```yaml
MaxCardinality:
  description: An axis with a maximum cardinality
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
      array:
        axes:
          x:
            rank: 0
            cardinality:
              max: 3
```
````
````{tab-item} Pydantic
```python
class MaxCardinality(ConfiguredBaseModel):
    """
    An axis with a maximum cardinality
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: Union[
        NDArray[Shape["1 x"], Float],
        NDArray[Shape["2 x"], Float],
        NDArray[Shape["3 x"], Float]
    ] = Field(...)
```    
````
`````

`````{tab-set}
````{tab-item} YAML
```yaml
RangeCardinality:
  description: An axis with a min and maximum cardinality
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
      array:
        axes:
          x:
            rank: 0
            cardinality:
              min: 2
              max: 4
```
````
````{tab-item} Pydantic
```python
class RangeCardinality(ConfiguredBaseModel):
    """
    An axis with a min and maximum cardinality
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: Union[
        NDArray[Shape["2 x"], Float],
        NDArray[Shape["3 x"], Float],
        NDArray[Shape["4 x"], Float]
    ] = Field(...)
```    
````
`````

`````{tab-set}
````{tab-item} YAML
```yaml
ExclusiveAxes:
  description: Two mutually exclusive definitions of an axis that define its different forms
  attributes:
    temp:
      range: float
      required: true
      unit:
        ucum_code: K
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
```
````
````{tab-item} Pydantic
```python
class ExclusiveAxes(ConfiguredBaseModel):
    """
    Two mutually exclusive definitions of an axis that define its different forms
    """
    linkml_meta: ClassVar[LinkML_Meta] = Field(LinkML_Meta(), frozen=True)
    temp: Union[
        NDArray[Shape["* x, * y, 3 rgb"], Float],
        NDArray[Shape["* x, * y, 4 rgba"], Float]
    ] = Field(...)
```
````
`````

## TODO

Any shape array

```yaml
classes:
  TemperatureDataset:
    attributes:
      temperatures_in_K:
        range: float
        multivalued: true
        required: true
        array:
```

One specified, named dimension, and any number of other dimensions

```yaml
array:
  dimensions:
    min: 1
    # optionally, to be explicit:
    max: null
  axes:
    x:
      rank: 0
      alias: latitude_in_deg
```

Two required dimensions and two optional dimensions that will generate
a union of the combinatoric product of the optional dimensions.
Rank must be unspecified in optional dimensions

```yaml
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
```

```{eval-rst}
.. automodule:: numpydantic.linkml.slotarray
    :members:
```