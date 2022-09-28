"""Patito, a data-modelling library built on top of polars and pydantic."""
from polars import Expr, Series, col

from patito import exceptions, sql
from patito.exceptions import ValidationError
from patito.polars import DataFrame, LazyFrame
from patito.pydantic import Field, Model

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
    "_DUCKDB_AVAILABLE",
    "col",
    "exceptions",
    "field",
    "sql",
]

try:
    from patito.duckdb import Database, Relation, RelationSource

    _DUCKDB_AVAILABLE = True
    __all__ += [
        "Database",
        "Relation",
        "RelationSource",
    ]
except ImportError:  # pragma: no cover
    pass


try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
