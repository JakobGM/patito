import itertools
import json
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    cast,
    get_args,
)

import polars as pl
from polars.datatypes import DataType, DataTypeClass, DataTypeGroup, convert
from polars.datatypes.constants import (
    DATETIME_DTYPES,
    DURATION_DTYPES,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
)
from polars.polars import (
    dtype_str_repr,  # TODO: this is a rust function, can we implement our own string parser for Time/Duration/Datetime?
)
from pydantic import TypeAdapter

from patito._pydantic.repr import display_as_type

PYTHON_TO_PYDANTIC_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}

BOOLEAN_DTYPES = DataTypeGroup([pl.Boolean])
STRING_DTYPES = DataTypeGroup([pl.Utf8])
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


def parse_composite_dtype(dtype: DataTypeClass | DataType) -> str:
    """for serialization, converts polars dtype to string representation
    """
    if dtype in pl.NESTED_DTYPES:
        if dtype == pl.Struct or isinstance(dtype, pl.Struct):
            raise NotImplementedError("Structs not yet supported by patito")
        if not isinstance(dtype, pl.List) or isinstance(dtype, pl.Array):
            raise NotImplementedError(
                f"Unsupported nested dtype: {dtype} of type {type(dtype)}"
            )
        if dtype.inner is None:
            return convert.DataTypeMappings.DTYPE_TO_FFINAME[dtype.base_type()]
        return f"{convert.DataTypeMappings.DTYPE_TO_FFINAME[dtype.base_type()]}[{parse_composite_dtype(dtype.inner)}]"
    elif dtype in pl.TEMPORAL_DTYPES:
        return dtype_str_repr(dtype)
    else:
        return convert.DataTypeMappings.DTYPE_TO_FFINAME[dtype]


def dtype_to_json(dtype: pl.DataType) -> str:
    """Serialize a polars dtype to a JSON string representation."""
    return json.dumps(str(dtype))


def json_to_dtype(json_dtype_str: str) -> pl.DataType:
    """Deserialize a polars dtype from a JSON string representation."""
    dtype = str_to_dtype(json.loads(json_dtype_str))
    return dtype


def str_to_dtype(dtype_str: str) -> pl.DataType:
    """Return the corresponding polars dtype."""
    from polars.datatypes.classes import (  # noqa F401
        Array,
        Binary,
        Boolean,
        Categorical,
        Date,
        Datetime,
        Decimal,
        Duration,
        Enum,
        Float32,
        Float64,
        Int8,
        Int16,
        Int32,
        Int64,
        List,
        Null,
        Object,
        Struct,
        Time,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Unknown,
        Utf8,
    )

    from polars.datatypes import DataTypeClass

    dtype = eval(dtype_str)
    if isinstance(dtype, DataTypeClass):
        # Float32() has string representation Float32, so we need to call it
        dtype = dtype()
    return dtype


def validate_polars_dtype(
    annotation: type[Any] | None,
    dtype: DataType | DataTypeClass | None,
    column: Optional[str] = None,
):
    """
    Check that the polars dtype is valid for the given annotation. Raises ValueError if not.

    Args:
        annotation (type[Any] | None): python type annotation
        dtype (DataType | DataTypeClass | None): polars dtype
        column (Optional[str], optional): column name. Defaults to None.
    """
    if (
        dtype is None or annotation is None
    ):  # no potential conflict between type annotation and chosen polars type
        return
    valid_dtypes = valid_polars_dtypes_for_annotation(annotation)
    if dtype not in valid_dtypes:
        if column:
            column_msg = f" for column `{column}`"
        else:
            column_msg = ""
        raise ValueError(
            f"Invalid dtype {dtype}{column_msg}. Allowable polars dtypes for {display_as_type(annotation)} are: {', '.join([str(x) for x in valid_dtypes])}."
        )
    return


def validate_annotation(annotation: type[Any] | None, column: Optional[str] = None):
    """
    Check that the provided annotation has polars/patito support (we can resolve it to a default dtype). Raises ValueError if not.

    Args:
        annotation (type[Any] | None): python type annotation
        column (Optional[str], optional): column name. Defaults to None.
    """
    default_dtype = default_polars_dtype_for_annotation(annotation)
    if default_dtype is None:
        valid_polars_dtypes = valid_polars_dtypes_for_annotation(annotation)
        if column:
            column_msg = f" for column `{column}`"
        else:
            column_msg = ""
        if len(valid_polars_dtypes) == 0:
            raise ValueError(
                f"Annotation {display_as_type(annotation)}{column_msg} is not compatible with any polars dtypes."
            )
        else:
            raise ValueError(
                f"Unable to determine default dtype for annotation {display_as_type(annotation)}{column_msg}. Please provide a valid default polars dtype via the `dtype` argument to `Field`. Valid dtypes are: {', '.join([str(x) for x in valid_polars_dtypes])}."
            )
    return


def valid_polars_dtypes_for_annotation(
    annotation: type[Any] | None,
) -> FrozenSet[DataTypeClass | DataType]:
    """Returns a set of polars types that are valid for the given annotation. If the annotation is Any, returns all supported polars dtypes.

    Args:
        annotation (type[Any] | None): python type annotation

    Returns:
        FrozenSet[DataTypeClass | DataType]: set of polars dtypes
    """
    if annotation == Any:
        return PT_BASE_SUPPORTED_DTYPES
    schema = TypeAdapter(annotation).json_schema()
    return _valid_polars_dtypes_for_schema(schema)


def default_polars_dtype_for_annotation(
    annotation: type[Any] | None,
) -> DataTypeClass | DataType | None:
    """Returns the default polars dtype for the given annotation. If the annotation is Any, returns pl.Utf8. If no default dtype can be determined, returns None.

    Args:
        annotation (type[Any] | None): python type annotation

    Returns:
        DataTypeClass | DataType | None: polars dtype
    """
    if annotation == Any:
        return pl.Utf8
    schema = TypeAdapter(annotation).json_schema()
    return _default_polars_dtype_for_schema(schema)


def _valid_polars_dtypes_for_schema(
    schema: Dict,
) -> FrozenSet[DataTypeClass | DataType]:
    valid_type_sets = []
    if "anyOf" in schema:
        schema = _without_optional(schema)
        for sub_props in schema["anyOf"]:
            valid_type_sets.append(
                set(_pydantic_subschema_to_valid_polars_types(sub_props))
            )
    else:
        valid_type_sets.append(set(_pydantic_subschema_to_valid_polars_types(schema)))
    return (
        set.intersection(*valid_type_sets) if valid_type_sets else frozenset()
    )  # pyright: ignore


def _default_polars_dtype_for_schema(schema: Dict) -> DataTypeClass | DataType | None:
    if "anyOf" in schema:
        if len(schema["anyOf"]) == 2:  # look for optionals first
            schema = _without_optional(schema)
        if len(schema["anyOf"]) == 1:
            schema = schema["anyOf"][0]
        else:
            return None
    return _pydantic_subschema_to_default_dtype(schema)


def _without_optional(schema: Dict) -> Dict:
    if "anyOf" in schema:
        for sub_props in schema["anyOf"]:
            if "type" in sub_props and sub_props["type"] == "null":
                schema["anyOf"].remove(sub_props)
    return schema


def _pydantic_subschema_to_valid_polars_types(
    props: Dict,
) -> FrozenSet[DataTypeClass | DataType]:
    if "type" not in props:
        if "enum" in props:
            raise TypeError("Mixed type enums not supported by patito.")
        elif "const" in props:
            return valid_polars_dtypes_for_annotation(type(props["const"]))
        return frozenset()
    pyd_type = props.get("type")
    if pyd_type == "array":
        if "items" not in props:
            raise NotImplementedError(
                "Unexpected error processing pydantic schema. Please file an issue."
            )
        array_props = props["items"]
        item_dtypes = _valid_polars_dtypes_for_schema(array_props)
        # TODO support pl.Array?
        return DataTypeGroup([pl.List(dtype) for dtype in item_dtypes])
    return _pyd_type_to_valid_dtypes(
        PydanticBaseType(pyd_type), props.get("format"), props.get("enum")
    )


def _pydantic_subschema_to_default_dtype(
    props: Dict,
) -> DataTypeClass | DataType | None:
    if "type" not in props:
        if "enum" in props:
            raise TypeError("Mixed type enums not supported by patito.")
        elif "const" in props:
            return default_polars_dtype_for_annotation(type(props["const"]))
        return None
    pyd_type = props.get("type")
    if pyd_type == "array":
        if "items" not in props:
            raise NotImplementedError(
                "Unexpected error processing pydantic schema. Please file an issue."
            )
        array_props = props["items"]
        inner_default_type = _default_polars_dtype_for_schema(array_props)
        if inner_default_type is None:
            return None
        return pl.List(inner_default_type)
    return _pyd_type_to_default_dtype(
        PydanticBaseType(pyd_type), props.get("format"), props.get("enum")
    )


def _pyd_type_to_valid_dtypes(
    pyd_type: PydanticBaseType, string_format: Optional[str], enum: List[str] | None
) -> FrozenSet[DataTypeClass | DataType]:
    if enum is not None:
        _validate_enum_values(pyd_type, enum)
        return DataTypeGroup(
            [pl.Categorical, pl.Utf8]
        )  # TODO use pl.Enum in future polars versions
    if pyd_type.value == "integer":
        return DataTypeGroup(INTEGER_DTYPES | FLOAT_DTYPES)
    elif pyd_type.value == "number":
        return FLOAT_DTYPES
    elif pyd_type.value == "boolean":
        return BOOLEAN_DTYPES
    elif pyd_type.value == "string":
        _string_format = (
            PydanticStringFormat(string_format) if string_format is not None else None
        )
        return _pyd_string_format_to_valid_dtypes(_string_format)
    elif pyd_type.value == "null":
        return frozenset({pl.Null})
    else:
        return frozenset()


def _pyd_type_to_default_dtype(
    pyd_type: PydanticBaseType, string_format: Optional[str], enum: List[str] | None
) -> DataTypeClass | DataType:
    if enum is not None:
        _validate_enum_values(pyd_type, enum)
        return pl.Categorical
    elif pyd_type.value == "integer":
        return pl.Int64
    elif pyd_type.value == "number":
        return pl.Float64
    elif pyd_type.value == "boolean":
        return pl.Boolean
    elif pyd_type.value == "string":
        _string_format = (
            PydanticStringFormat(string_format) if string_format is not None else None
        )
        return _pyd_string_format_to_default_dtype(_string_format)
    elif pyd_type.value == "null":
        return pl.Null
    elif pyd_type.value == "object":
        raise ValueError("pydantic object types not currently supported by patito")
    else:
        raise NotImplementedError


def _pyd_string_format_to_valid_dtypes(
    string_format: PydanticStringFormat | None,
) -> FrozenSet[DataTypeClass | DataType]:
    if string_format is None:
        return STRING_DTYPES
    elif string_format.value == "date":
        return DATE_DTYPES
    elif string_format.value == "date-time":
        return DATETIME_DTYPES
    elif string_format.value == "duration":
        return DURATION_DTYPES
    elif string_format.value == "time":
        return TIME_DTYPES
    else:
        raise NotImplementedError


def _pyd_string_format_to_default_dtype(
    string_format: PydanticStringFormat | None,
) -> DataTypeClass | DataType:
    if string_format is None:
        return pl.Utf8
    elif string_format.value == "date":
        return pl.Date
    elif string_format.value == "date-time":
        return pl.Datetime
    elif string_format.value == "duration":
        return pl.Duration
    elif string_format.value == "time":
        return pl.Time
    else:
        raise NotImplementedError


def _validate_enum_values(pyd_type: PydanticBaseType, enum: Sequence):
    enum_types = set(type(value) for value in enum)
    if len(enum_types) > 1:
        raise TypeError(
            f"All enumerated values of enums used to annotate Patito model fields must have the same type. Encountered types: {sorted(map(lambda t: t.__name__, enum_types))}."
        )
    if pyd_type.value != "string":
        raise TypeError(
            f"Enums used to annotate Patito model fields must be strings. Encountered type: {enum_types.pop().__name__}."
        )
