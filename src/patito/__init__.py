"""Patito, a data-modelling library built on top of polars and pydantic."""
from polars import Expr, Series, col

from patito import exceptions, sql
# from patito.exceptions import ValidationError
from patito.polars import DataFrame, LazyFrame
from patito.pydantic import Field, Model

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
    # "ValidationError",
    "_CACHING_AVAILABLE",
    "_DUCKDB_AVAILABLE",
    "col",
    "exceptions",
    "field",
    "sql",
]

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
