"""Patito, a data-modelling library built on top of polars and pydantic."""
from typing import TypeVar, Type

from polars import Expr, Series, col

from patito import exceptions
from patito.exceptions import DataFrameValidationError
from patito.polars import DataFrame, LazyFrame
from patito.pydantic import Field, Model

_CACHING_AVAILABLE = False
field = col("_")
__all__ = [
    "DataFrame",
    "DataFrameValidationError",
    "Expr",
    "Field",
    "LazyFrame",
    "Model",
    "Series",
    "_CACHING_AVAILABLE",
    "col",
    "exceptions",
    "field",
]

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


ModelType = TypeVar("ModelType", bound=Model)
PydanticModelType = TypeVar("PydanticModelType", bound=pydantic.BaseModel)


def from_pydantic(pydantic_model: Type[PydanticModelType]) -> Type[ModelType]:
    class PydanticModelMixedWithPatito(pydantic_model, Model):
        pass

    return PydanticModelMixedWithPatito
