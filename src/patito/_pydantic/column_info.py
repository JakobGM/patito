from __future__ import annotations

import io
import json
from typing import Annotated, Optional, Union

import polars as pl
from polars.datatypes import *  # noqa: F403 # type: ignore
from polars.datatypes import DataType, DataTypeClass
from polars.exceptions import ComputeError
from pydantic import BaseModel, BeforeValidator, field_serializer


def dtype_deserializer(dtype: str | DataTypeClass | DataType | None):
    """Deserialize a dtype from json."""
    if isinstance(dtype, DataTypeClass) or isinstance(dtype, DataType):
        return dtype
    else:
        if dtype == "null" or dtype is None:
            return None
        else:
            return eval(dtype)


def expr_deserializer(
    expr: str | pl.Expr | list[pl.Expr] | None,
) -> pl.Expr | list[pl.Expr] | None:
    """Deserialize a polars expression or list thereof from json.

    This is applied both during deserialization and validation.
    """
    if expr is None:
        return None
    elif isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, list):
        return expr
    elif isinstance(expr, str):
        if expr == "null":
            return None
        # can be either a list of expr or expr
        elif expr[0] == "[":
            return [
                pl.Expr.deserialize(io.StringIO(e), format="json")
                for e in json.loads(expr)
            ]
        else:
            return pl.Expr.deserialize(io.StringIO(expr), format="json")
    else:
        raise ValueError(f"{expr} can not be deserialized.")


def expr_or_col_name_deserializer(expr: str | pl.Expr | None) -> pl.Expr | str | None:
    """Deserialize a polars expression or column name from json.

    This is applied both during deserialization and validation.
    """
    if expr is None:
        return None
    elif isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, list):
        return expr
    elif isinstance(expr, str):
        # Default behaviour
        if expr == "null":
            return None
        else:
            try:
                return pl.Expr.deserialize(io.StringIO(expr), format="json")
            except ComputeError:
                try:
                    # Column name is being deserialized
                    return json.loads(expr)
                except json.JSONDecodeError:
                    # Column name has been passed literally
                    # to ColumnInfo(derived_from="foo")
                    return expr
    else:
        raise ValueError(f"{expr} can not be deserialized.")


class ColumnInfo(BaseModel, arbitrary_types_allowed=True):
    """patito-side model for storing column metadata.

    Args:
        allow_missing (bool): Column may be missing.
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

    allow_missing: Optional[bool] = None
    dtype: Annotated[
        Optional[Union[DataTypeClass, DataType]],
        BeforeValidator(dtype_deserializer),
    ] = None
    constraints: Annotated[
        Optional[Union[pl.Expr, list[pl.Expr]]],
        BeforeValidator(expr_deserializer),
    ] = None
    derived_from: Annotated[
        Optional[Union[str, pl.Expr]],
        BeforeValidator(expr_or_col_name_deserializer),
    ] = None
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
    def expr_serializer(self, expr: None | pl.Expr | list[pl.Expr]):
        """Converts polars expr to json."""
        if expr is None:
            return "null"
        elif isinstance(expr, str):
            return json.dumps(expr)
        elif isinstance(expr, list):
            return json.dumps([e.meta.serialize(format="json") for e in expr])
        else:
            return expr.meta.serialize(format="json")

    @field_serializer("dtype")
    def dtype_serializer(self, dtype: DataTypeClass | DataType | None) -> str:
        """Converts polars dtype to json."""
        if dtype is None:
            return "null"
        else:
            return str(dtype)
