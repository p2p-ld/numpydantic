# Interfaces


```{mermaid}
flowchart LR
    classDef data fill:#2b8cee,color:#ffffff;
    classDef X fill:transparent,border:none,color:#ff0000;

    input

    subgraph Interface
    match
    end

    subgraph Numpy
    numpy_check["check"]
    end

    subgraph Dask
    direction TB
    
    dask_check["check"]

    subgraph Validation
    direction TB
    
    before_validation --> validate_dtype
    validate_dtype --> validate_shape
    validate_shape --> after_validation
    end
    
    dask_check --> Validation

    end

    subgraph Zarr
    zarr_check["check"]
    end

    subgraph Model
    output
    end

    zarr_x["X"]
    numpy_x["X"]

    input --> match
    match --> numpy_check
    match --> zarr_check
    match --> Dask
    zarr_check --> zarr_x
    numpy_check --> numpy_x

    Validation --> Model

    class input data
    class output data
    class zarr_x X
    class numpy_x X
``` 