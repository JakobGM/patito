"""Patito, a data-modelling library built on top of polars and pydantic."""
import pkg_resources
from pkg_resources import DistributionNotFound
from polars import Expr, Series, col

from patito import exceptions
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
    "col",
    "exceptions",
    "_DUCKDB_AVAILABLE",
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
