from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import polars as pl
from polars.datatypes import DataType, DataTypeClass
from pydantic import BaseModel, field_serializer

from patito._pydantic.dtypes import parse_composite_dtype


class ColumnInfo(BaseModel, arbitrary_types_allowed=True):
    """patito-side model for storing column metadata.

    Args:
        constraints (Union[polars.Expression, List[polars.Expression]): A single
            constraint or list of constraints, expressed as a polars expression objects.
            All rows must satisfy the given constraint. You can refer to the given column
            with ``pt.field``, which will automatically be replaced with
            ``polars.col(<field_name>)`` before evaluation.
        derived_from (Union[str, polars.Expr]): used to mark fields that are meant to be derived from other fields. Users can specify a polars expression that will be called to derive the column value when `pt.DataFrame.derive` is called.
        dtype (polars.datatype.DataType): The given dataframe column must have the given
            polars dtype, for instance ``polars.UInt64`` or ``pl.Float32``.
        unique (bool): All row values must be unique.

    """

    dtype: Optional[Union[DataTypeClass, DataType]] = None
    constraints: Optional[Union[pl.Expr, Sequence[pl.Expr]]] = None
    derived_from: Optional[Union[str, pl.Expr]] = None
    unique: Optional[bool] = None

    def __repr__(self) -> str:
        """Print only Field attributes whose values are not default (mainly None)."""
        not_default_field = {
            field: getattr(self, field)
            for field in self.model_fields
            if getattr(self, field) is not self.model_fields[field].default
        }

        string = ""
        for field, value in not_default_field.items():
            string += f"{field}={value}, "
        if string:
            # remove trailing comma and space
            string = string[:-2]
        return f"ColumnInfo({string})"

    @field_serializer("constraints", "derived_from")
    def serialize_exprs(self, exprs: str | pl.Expr | Sequence[pl.Expr] | None) -> Any:
        if exprs is None:
            return None
        elif isinstance(exprs, str):
            return exprs
        elif isinstance(exprs, pl.Expr):
            return self._serialize_expr(exprs)
        elif isinstance(exprs, Sequence):
            return [self._serialize_expr(c) for c in exprs]
        else:
            raise ValueError(f"Invalid type for exprs: {type(exprs)}")

    def _serialize_expr(self, expr: pl.Expr) -> Dict:
        if isinstance(expr, pl.Expr):
            return json.loads(
                expr.meta.serialize(None)
            )  # can we access the dictionary directly?
        else:
            raise ValueError(f"Invalid type for expr: {type(expr)}")

    @field_serializer("dtype")
    def serialize_dtype(self, dtype: DataTypeClass | DataType | None) -> Any:
        """Serialize a polars dtype.

        References:
            [1] https://stackoverflow.com/questions/76572310/how-to-serialize-deserialize-polars-datatypes
        """
        if dtype is None:
            return None
        elif isinstance(dtype, DataTypeClass) or isinstance(dtype, DataType):
            return parse_composite_dtype(dtype)
        else:
            raise ValueError(f"Invalid type for dtype: {type(dtype)}")


CI = TypeVar("CI", bound=Type[ColumnInfo])
