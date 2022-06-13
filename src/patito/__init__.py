"""Patito, a data-modelling library built on top of polars and pydantic."""
import pkg_resources
from pkg_resources import DistributionNotFound
from polars import Expr, Series, col

from patito import exceptions, sql
from patito.exceptions import ValidationError
from patito.polars import DataFrame
from patito.pydantic import Field, Model

_DUCKDB_AVAILABLE = False
__all__ = [
    "DataFrame",
    "Expr",
    "Field",
    "Model",
    "Series",
    "ValidationError",
    "_DUCKDB_AVAILABLE",
    "col",
    "exceptions",
    "sql",
]

try:
    pkg_resources.require(["duckdb>=0.3.2"])
    from patito.duckdb import Database, Relation, RelationSource

    _DUCKDB_AVAILABLE = True
    __all__ += [
        "Database",
        "Relation",
        "RelationSource",
    ]
except DistributionNotFound:  # pragma: no cover
    pass


try:
    from importlib.metadata import PackageNotFoundError, version  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
