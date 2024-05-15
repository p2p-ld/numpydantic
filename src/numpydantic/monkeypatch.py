"""
Functions to monkeypatch dependent packages - most notably nptyping
"""

# ruff: noqa: ANN001


def patch_npytyping_perf() -> None:
    """
    npytyping makes an expensive call to inspect.stack()
    that makes imports of pydantic models take ~200x longer than
    they should:

    References:
        - https://github.com/ramonhagenaars/nptyping/issues/110
    """
    import inspect
    from types import FrameType

    from nptyping import base_meta_classes, ndarray, recarray
    from nptyping.pandas_ import dataframe

    # make a new __module__ methods for the affected classes

    def new_module_ndarray(cls) -> str:  # pragma: no cover
        return cls._get_module(inspect.currentframe(), "nptyping.ndarray")

    def new_module_recarray(cls) -> str:  # pragma: no cover
        return cls._get_module(inspect.currentframe(), "nptyping.recarray")

    def new_module_dataframe(cls) -> str:  # pragma: no cover
        return cls._get_module(inspect.currentframe(), "nptyping.pandas_.dataframe")

    # and a new _get_module method for the parent class
    def new_get_module(cls, stack: FrameType, module: str) -> str:  # pragma: no cover
        return (
            "typing"
            if inspect.getframeinfo(stack.f_back).function == "formatannotation"
            else module
        )

    # now apply the patches
    ndarray.NDArrayMeta.__module__ = property(new_module_ndarray)
    recarray.RecArrayMeta.__module__ = property(new_module_recarray)
    dataframe.DataFrameMeta.__module__ = property(new_module_dataframe)
    base_meta_classes.SubscriptableMeta._get_module = new_get_module


def patch_nptyping_warnings() -> None:
    """
    nptyping shits out a bunch of numpy deprecation warnings from using
    olde aliases

    References:
        - https://github.com/ramonhagenaars/nptyping/issues/113
        - https://github.com/ramonhagenaars/nptyping/issues/102
    """
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping.*")


def apply_patches() -> None:
    """Apply all monkeypatches!"""
    patch_npytyping_perf()
    patch_nptyping_warnings()
