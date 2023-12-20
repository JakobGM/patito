from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional, get_args

import polars as pl
from polars.datatypes import DataType as PolarsDataType
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo


class Foo(Enum):
    A = 1
    B = 2


class SimpleExample(BaseModel):
    id: str
    name: str
    age: int = Field(json_schema_extra={"dtype": pl.Int64()})


class NearlyCompleteExample(BaseModel):
    int_with_dtype_value: int = Field(json_schema_extra={"dtype": pl.Int64()})
    int_value: int
    float_value: float
    str_value: str
    bool_value: bool
    list_value: list[int]
    list_value_nullable: list[int | None]
    literal_value: Literal["a", "b"]
    default_value: str = "my_default"
    optional_value: Optional[int]
    bounded_value: int = Field(ge=10, le=20)
    date_value: date
    datetime_value: datetime
    enum_value: Foo


class ColumnInfo(BaseModel, arbitrary_types_allowed=True):
    """A model containing info patito needs about a column."""

    name: str
    dtype: PolarsDataType
    required: bool
    nullable: bool
    unique: bool
    type_hint: type
    contraints: list[pl.Expr] | None = None


def is_single_type(type: type) -> bool:
    """Type hint is a single type.

    True for: int, str, float, bool, etc.
    False for: Optional[int], Union[int, str], Literal["a", "b"], etc.
    """
    return get_args(type) == ()


def is_literal(type_: type) -> bool:
    "Determine whether the type hint is a Literal type."
    try:
        return type_.__dict__["__origin__"] is Literal
    except KeyError:
        return False


def get_enum_inner_type(enum: type) -> type | None:
    "Get the type of the values of the enum if it exists, None otherwise."
    if issubclass(enum, Enum):
        enum_types = set(type(value) for value in enum)  # type: ignore
        if len(enum_types) > 1:
            raise TypeError(
                "All enumerated values of enums used to annotate "
                "Patito model fields must have the same type. "
                "Encountered types: "
                f"{sorted(map(lambda t: t.__name__, enum_types))}."
            )
        enum_type = enum_types.pop()
    else:
        enum_type = None
    return enum_type


# Working with the simple example
model_fields = SimpleExample.model_fields

for field_name, field_info in model_fields.items():
    print(field_name, field_info, "\t\t", get_args(field_info.annotation))


PYTHON_TO_POLARS_TYPES = {
    str: pl.Utf8,
    int: pl.Int64,
    float: pl.Float64,
    bool: pl.Boolean,
}


def get_polars_dtype(field_info: FieldInfo) -> PolarsDataType | None:
    if schema_extra := field_info.json_schema_extra:
        dtype = schema_extra.get("dtype")
    else:
        dtype = None
    return dtype


def get_is_unique(field_info: FieldInfo) -> bool:
    if schema_extra := field_info.json_schema_extra:
        is_unique = schema_extra.get("unique", False)
    else:
        is_unique = False
    return is_unique


def get_dtype(field_info: FieldInfo, type_hint_type: type) -> PolarsDataType:
    return get_polars_dtype(field_info) or PYTHON_TO_POLARS_TYPES[type_hint_type]


fields = {}
for field_name, field_info in model_fields.items():
    print(field_name)
    fields[field_name] = {}

    assert field_info.annotation is not None, (
        f"Encountered a case where `field_info.annotation` is None for field `{field_name}`.`"
        "Please report this with an example of your Model in an issue to the patito github repo."
    )
    if is_single_type(field_info.annotation):
        # e.g. regular type like int, float, str, bool, but also Enum
        if enum_type := get_enum_inner_type(field_info.annotation):
            type_hint_type = enum_type
        else:
            type_hint_type = field_info.annotation
        dtype = get_dtype(field_info, type_hint_type)
    else:
        # e.g. list[int], list[int | None], Literal["a", "b"] or nullable types like Optional[int] or date | None
        type_hint_type = int
        fields[field_name]["type_hint"] = type_hint_type

    PatitoReducedField(
        name=field_name,
        dtype=dtype,
        required=field_info.is_required(),
        nullable=field_info.allow_none,
        unique=False,
        type_hint=type_hint_type,
    )

print(get_required(model_fields))
