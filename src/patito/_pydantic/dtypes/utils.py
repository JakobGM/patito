from __future__ import annotations

import sys
from collections.abc import Sequence
from enum import Enum
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
)

import polars as pl
from polars.datatypes import DataType, DataTypeClass, convert
from polars.datatypes.group import (
    DATETIME_DTYPES,
    DURATION_DTYPES,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    DataTypeGroup,
)

PYTHON_TO_PYDANTIC_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}

BOOLEAN_DTYPES = DataTypeGroup([pl.Boolean])
STRING_DTYPES = DataTypeGroup([pl.String])
DATE_DTYPES = DataTypeGroup([pl.Date])
TIME_DTYPES = DataTypeGroup([pl.Time])

PT_BASE_SUPPORTED_DTYPES = DataTypeGroup(
    INTEGER_DTYPES
    | FLOAT_DTYPES
    | BOOLEAN_DTYPES
    | STRING_DTYPES
    | DATE_DTYPES
    | DATETIME_DTYPES
    | DURATION_DTYPES
    | TIME_DTYPES
)

if sys.version_info >= (3, 10):  # pragma: no cover
    from types import UnionType  # pyright: ignore

    UNION_TYPES = (Union, UnionType)
else:
    UNION_TYPES = (Union,)  # pragma: no cover


class PydanticBaseType(Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    NULL = "null"
    OBJECT = "object"


class PydanticStringFormat(Enum):
    DATE = "date"
    DATE_TIME = "date-time"
    DURATION = "duration"
    TIME = "time"


def is_optional(type_annotation: type[Any] | Any | None) -> bool:
    """Return True if the given type annotation is an Optional annotation.

    Args:
        type_annotation: The type annotation to be checked.

    Returns:
        True if the outermost type is Optional.

    """
    return (get_origin(type_annotation) in UNION_TYPES) and (
        type(None) in get_args(type_annotation)
    )


def unwrap_optional(type_annotation: type[Any] | Any) -> type:
    """Return the inner, wrapped type of an Optional.

    Is a no-op for non-Optional types.

    Args:
        type_annotation: The type annotation to be dewrapped.

    Returns:
        The input type, but with the outermost Optional removed.

    """
    return (
        next(  # pragma: no cover
            valid_type
            for valid_type in get_args(type_annotation)
            if valid_type is not type(None)  # noqa: E721
        )
        if is_optional(type_annotation)
        else type_annotation
    )


def parse_composite_dtype(dtype: DataTypeClass | DataType) -> str:
    """For serialization, converts polars dtype to string representation."""
    return str(dtype)


def dtype_from_string(v: str) -> DataTypeClass | DataType | None:
    """For deserialization."""
    # TODO test all dtypes
    return convert.dtype_short_repr_to_dtype(v)


def _pyd_type_to_valid_dtypes(
    pyd_type: PydanticBaseType, string_format: str | None, enum: list[str] | None
) -> DataTypeGroup:
    if enum is not None:
        _validate_enum_values(pyd_type, enum)
        return DataTypeGroup([pl.Enum(enum), pl.String], match_base_type=False)
    if pyd_type.value == "integer":
        return DataTypeGroup(INTEGER_DTYPES)
    elif pyd_type.value == "number":
        return (
            FLOAT_DTYPES
            if isinstance(FLOAT_DTYPES, DataTypeGroup)
            else DataTypeGroup(FLOAT_DTYPES)
        )
    elif pyd_type.value == "boolean":
        return BOOLEAN_DTYPES
    elif pyd_type.value == "string":
        _string_format = (
            PydanticStringFormat(string_format) if string_format is not None else None
        )
        return _pyd_string_format_to_valid_dtypes(_string_format)
    elif pyd_type.value == "null":
        return DataTypeGroup([pl.Null])
    else:
        return DataTypeGroup([])


def _pyd_type_to_default_dtype(
    pyd_type: PydanticBaseType, string_format: str | None, enum: list[str] | None
) -> DataTypeClass | DataType:
    if enum is not None:
        _validate_enum_values(pyd_type, enum)
        return pl.Enum(enum)
    elif pyd_type.value == "integer":
        return pl.Int64()
    elif pyd_type.value == "number":
        return pl.Float64()
    elif pyd_type.value == "boolean":
        return pl.Boolean()
    elif pyd_type.value == "string":
        _string_format = (
            PydanticStringFormat(string_format) if string_format is not None else None
        )
        return _pyd_string_format_to_default_dtype(_string_format)
    elif pyd_type.value == "null":
        return pl.Null()
    elif pyd_type.value == "object":
        raise ValueError("pydantic object types not currently supported by patito")
    else:
        raise NotImplementedError


def _pyd_string_format_to_valid_dtypes(
    string_format: PydanticStringFormat | None,
) -> DataTypeGroup:
    if string_format is None:
        return STRING_DTYPES
    elif string_format.value == "date":
        return DATE_DTYPES
    elif string_format.value == "date-time":
        return (
            DATETIME_DTYPES
            if isinstance(DATE_DTYPES, DataTypeGroup)
            else DataTypeGroup(DATE_DTYPES)
        )
    elif string_format.value == "duration":
        return (
            DURATION_DTYPES
            if isinstance(DURATION_DTYPES, DataTypeGroup)
            else DataTypeGroup(DURATION_DTYPES)
        )
    elif string_format.value == "time":
        return TIME_DTYPES
    else:
        raise NotImplementedError


def _pyd_string_format_to_default_dtype(
    string_format: PydanticStringFormat | None,
) -> DataTypeClass | DataType:
    if string_format is None:
        return pl.String()
    elif string_format.value == "date":
        return pl.Date()
    elif string_format.value == "date-time":
        return pl.Datetime()
    elif string_format.value == "duration":
        return pl.Duration()
    elif string_format.value == "time":
        return pl.Time()
    else:
        raise NotImplementedError


def _without_optional(schema: dict) -> dict:
    if "anyOf" in schema:
        for sub_props in schema["anyOf"]:
            if "type" in sub_props and sub_props["type"] == "null":
                schema["anyOf"].remove(sub_props)
    return schema


def _validate_enum_values(pyd_type: PydanticBaseType, enum: Sequence[Any]) -> None:
    enum_types = set(type(value) for value in enum)
    if len(enum_types) > 1:
        raise TypeError(
            f"All enumerated values of enums used to annotate Patito model fields must have the same type. Encountered types: {sorted(map(lambda t: t.__name__, enum_types))}."
        )
    if pyd_type.value != "string":
        raise TypeError(
            f"Enums used to annotate Patito model fields must be strings. Encountered type: {enum_types.pop().__name__}."
        )
