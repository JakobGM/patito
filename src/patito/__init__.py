"""Patito, a data-modelling library built on top of polars and pydantic."""
from datetime import date, datetime
from typing import Dict, Any, Type, TypeVar, Union, Iterable, Optional, Literal

from polars import Expr, Series, col
import polars as pl

from patito import exceptions, sql
from patito.exceptions import ValidationError
from patito.polars import DataFrame, LazyFrame
from patito.pydantic import Field, Model
from patito import pydantic
from patito.validators import validate

_CACHING_AVAILABLE = False
_DUCKDB_AVAILABLE = False
field = col("_")
__all__ = [
    "DataFrame",
    "Expr",
    "Field",
    "LazyFrame",
    "Model",
    "Series",
    "ValidationError",
    "_CACHING_AVAILABLE",
    "_DUCKDB_AVAILABLE",
    "col",
    "exceptions",
    "field",
    "sql",
]

ModelType = TypeVar("ModelType", bound="Model")


try:
    from patito import duckdb

    _DUCKDB_AVAILABLE = True
    __all__ += ["duckdb"]
except ImportError:  # pragma: no cover
    pass

try:
    from patito.database import Database

    _CACHING_AVAILABLE = True
    __all__ += ["Database"]
except ImportError:
    pass


try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


class Foo:
    def __init__(self, model: pydantic.Model):
        self.pydantic_model = model

    @property
    def columns(self):
        return pydantic.get_model_columns(self.pydantic_model)

    @property
    def dtypes(self):
        return pydantic.get_dtypes(self.pydantic_model)

    @property
    def valid_dtypes(self):
        return pydantic.get_valid_dtypes(self.pydantic_model)

    @property
    def valid_sql_types(self):
        return pydantic.get_valid_sql_types(self.pydantic_model)

    @property
    def sql_types(self):
        return pydantic.get_sql_types(self.pydantic_model)

    @property
    def defaults(self):
        return pydantic.get_defaults(self.pydantic_model)

    @property
    def non_nullable_columns(self):
        return pydantic.get_non_nullable_columns(self.pydantic_model)

    @property
    def nullable_columns(self):
        return pydantic.get_nullable_columns(self.pydantic_model)

    @property
    def unique_columns(self):
        return pydantic.get_unique_columns(self.pydantic_model)

    def _schema_properties(self) -> Dict[str, Dict[str, Any]]:
        return pydantic.get_schema_properties(self.pydantic_model)

    def from_row(
        self,
        row: Union["pd.DataFrame", pl.DataFrame],
        validate: bool = True,
    ) -> ModelType:
        raise NotImplementedError

    def _from_polars(
        self,
        dataframe: pl.DataFrame,
        validate: bool = True,
    ) -> ModelType:
        raise NotImplementedError

    @classmethod
    def validate(cls, dataframe: Union["pd.DataFrame", pl.DataFrame]) -> None:
        validate(dataframe=dataframe, schema=cls)

    def example_values(self,
        field: str,
    ) -> Union[date, datetime, float, int, str, None]:
        raise NotImplementedError

    def example(
        self,
        **kwargs: Any,  # noqa: ANN401
    ) -> ModelType:
        raise NotImplementedError

    def pandas_examples(
        self,
        data: Union[dict, Iterable],
        columns: Optional[Iterable[str]] = None,
    ) -> "pd.DataFrame":
        raise NotImplementedError

    def examples(
        self,
        data: Optional[Union[dict, Iterable]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> "patito.polars.DataFrame":
        raise NotImplementedError

    def join(
        self,
        other: Type["Model"],
        how: Literal["inner", "left", "outer", "asof", "cross", "semi", "anti"],
    ) -> Type["Model"]:
        raise NotImplementedError

    def select(
        self, fields: Union[str, Iterable[str]]
    ) -> Type["Model"]:
        raise NotImplementedError

    def drop(self, name: Union[str, Iterable[str]]) -> Type["Model"]:
        raise NotImplementedError

    def prefix(self, prefix: str) -> Type["Model"]:
        raise NotImplementedError

    def suffix(self, suffix: str) -> Type["Model"]:
        raise NotImplementedError

    def rename(self, mapping: Dict[str, str]) -> Type["Model"]:
        raise NotImplementedError

    def with_fields(
        self,
        **field_definitions: Any,  # noqa: ANN401
    ) -> Type["Model"]:
        raise NotImplementedError

    def _derive_model(
        self,
        model_name: str,
        field_mapping: Dict[str, Any],
    ) -> Type["Model"]:
        raise NotImplementedError


def from_pydantic(pydantic_model: pydantic.Model) -> Foo:
    return Foo(pydantic_model)
