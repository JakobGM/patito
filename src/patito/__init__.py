import pkg_resources
from patito import exceptions
from patito.exceptions import ValidationError
from patito.polars import DataFrame
from patito.pydantic import Field, Model
from pkg_resources import DistributionNotFound, VersionConflict
from polars import Expr, Series, col


__all__ = [
    "DataFrame",
    "Expr",
    "Field",
    "Model",
    "Series",
    "ValidationError",
    "col",
    "exceptions",
]

try:
    pkg_resources.require(["duckdb>=0.3.2"])
    from patito.duckdb import Database, Relation, RelationSource

    __all__ += [
        "Database",
        "Relation",
        "RelationSource",
    ]
except DistributionNotFound:
    pass
