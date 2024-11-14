from __future__ import annotations

from collections.abc import Mapping
from functools import cache, reduce
from operator import or_
from typing import TYPE_CHECKING, Any

import polars as pl
from polars.datatypes import DataType, DataTypeClass
from polars.datatypes.group import DataTypeGroup
from pydantic import TypeAdapter

from patito._pydantic.column_info import ColumnInfo
from patito._pydantic.dtypes.utils import (
    PT_BASE_SUPPORTED_DTYPES,
    PydanticBaseType,
    _pyd_type_to_default_dtype,
    _pyd_type_to_valid_dtypes,
    _without_optional,
)
from patito._pydantic.repr import display_as_type

if TYPE_CHECKING:
    from patito.pydantic import ModelType


@cache
def valid_dtypes_for_model(
    cls: type[ModelType],
) -> Mapping[str, frozenset[DataTypeClass]]:
    return {
        column: (
            DtypeResolver(cls.model_fields[column].annotation).valid_polars_dtypes()
            if cls.column_infos[column].dtype is None
            else DataTypeGroup([cls.dtypes[column]], match_base_type=False)
        )
        for column in cls.columns
    }


@cache
def default_dtypes_for_model(
    cls: type[ModelType],
) -> dict[str, DataType]:
    default_dtypes: dict[str, DataType] = {}
    for column in cls.columns:
        dtype = (
            cls.column_infos[column].dtype
            or DtypeResolver(cls.model_fields[column].annotation).default_polars_dtype()
        )
        if dtype is None:
            raise ValueError(f"Unable to find a default dtype for column `{column}`")

        default_dtypes[column] = dtype if isinstance(dtype, DataType) else dtype()
    return default_dtypes


def validate_polars_dtype(
    annotation: type[Any] | None,
    dtype: DataType | DataTypeClass | None,
    column: str | None = None,
) -> None:
    """Check that the polars dtype is valid for the given annotation. Raises ValueError if not.

    Args:
        annotation (type[Any] | None): python type annotation
        dtype (DataType | DataTypeClass | None): polars dtype
        column (Optional[str], optional): column name. Defaults to None.

    """
    if (
        dtype is None or annotation is None
    ):  # no potential conflict between type annotation and chosen polars type
        return
    valid_dtypes = DtypeResolver(annotation).valid_polars_dtypes()
    if dtype not in valid_dtypes:
        if column:
            column_msg = f" for column `{column}`"
        else:
            column_msg = ""
        raise ValueError(
            f"Invalid dtype {dtype}{column_msg}. Allowable polars dtypes for {display_as_type(annotation)} are: {', '.join([str(x) for x in valid_dtypes])}."
        )
    return


def validate_annotation(
    annotation: type[Any] | Any | None, column: str | None = None
) -> None:
    """Check that the provided annotation has polars/patito support (we can resolve it to a default dtype). Raises ValueError if not.

    Args:
        annotation (type[Any] | None): python type annotation
        column (Optional[str], optional): column name. Defaults to None.

    """
    default_dtype = DtypeResolver(annotation).default_polars_dtype()
    if default_dtype is None:
        valid_polars_dtypes = DtypeResolver(annotation).valid_polars_dtypes()
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


class DtypeResolver:
    def __init__(self, annotation: Any | None):
        self.annotation = annotation
        # mode='serialization' allows nested models with structs, see #86
        self.schema = TypeAdapter(annotation).json_schema(mode="serialization")
        self.defs = self.schema.get("$defs", {})

    def valid_polars_dtypes(self) -> DataTypeGroup:
        if self.annotation == Any:
            return PT_BASE_SUPPORTED_DTYPES
        return self._valid_polars_dtypes_for_schema(self.schema)

    def default_polars_dtype(self) -> DataType | None:
        if self.annotation == Any:
            return pl.String()
        return self._default_polars_dtype_for_schema(self.schema)

    def _valid_polars_dtypes_for_schema(
        self,
        schema: dict,
    ) -> DataTypeGroup:
        valid_type_sets = []
        if "anyOf" in schema:
            schema = _without_optional(schema)
            for sub_props in schema["anyOf"]:
                valid_type_sets.append(
                    self._pydantic_subschema_to_valid_polars_types(sub_props)
                )
        else:
            valid_type_sets.append(
                self._pydantic_subschema_to_valid_polars_types(schema)
            )
        return reduce(or_, valid_type_sets) if valid_type_sets else DataTypeGroup([])

    def _pydantic_subschema_to_valid_polars_types(
        self,
        props: dict,
    ) -> DataTypeGroup:
        if "type" not in props:
            if "enum" in props:
                raise TypeError("Mixed type enums not supported by patito.")
            elif "const" in props:
                return DtypeResolver(type(props["const"])).valid_polars_dtypes()
            elif "$ref" in props:
                return self._pydantic_subschema_to_valid_polars_types(
                    self.defs[props["$ref"].split("/")[-1]]
                )
            return DataTypeGroup([])

        pyd_type = props.get("type")
        if pyd_type == "array":
            if "items" not in props:
                return DataTypeGroup([])
            array_props = props["items"]
            item_dtypes = self._valid_polars_dtypes_for_schema(array_props)
            # TODO support pl.Array?
            return DataTypeGroup(
                [pl.List(dtype) for dtype in item_dtypes], match_base_type=False
            )

        elif pyd_type == "object":
            if "properties" not in props:
                return DataTypeGroup([])
            object_props = props["properties"]
            struct_fields: list[pl.Field] = []
            for name, sub_props in object_props.items():
                dtype = self._default_polars_dtype_for_schema(sub_props)
                assert dtype is not None
                struct_fields.append(pl.Field(name, dtype))
            return DataTypeGroup(
                [pl.Struct(struct_fields)],
                match_base_type=False,
            )  # for structs, return only the default dtype set to avoid combinatoric issues
        return _pyd_type_to_valid_dtypes(
            PydanticBaseType(pyd_type), props.get("format"), props.get("enum")
        )

    def _default_polars_dtype_for_schema(
        self, schema: dict[str, Any]
    ) -> DataType | None:
        if "anyOf" in schema:
            if len(schema["anyOf"]) == 2:  # look for optionals first
                schema = _without_optional(schema)
            if len(schema["anyOf"]) == 1:
                if "column_info" in schema:
                    schema["anyOf"][0]["column_info"] = schema[
                        "column_info"
                    ]  # push column info through optional
                schema = schema["anyOf"][0]
            else:
                return None
        return self._pydantic_subschema_to_default_dtype(schema)

    def _pydantic_subschema_to_default_dtype(
        self,
        props: dict[str, Any],
    ) -> DataType | None:
        if "column_info" in props:  # user has specified in patito model
            ci = ColumnInfo.model_validate_json(props["column_info"])
            if ci.dtype is not None:
                dtype = ci.dtype() if isinstance(ci.dtype, DataTypeClass) else ci.dtype
                return dtype

        if "type" not in props:
            if "enum" in props:
                raise TypeError("Mixed type enums not supported by patito.")
            elif "const" in props:
                return DtypeResolver(type(props["const"])).default_polars_dtype()
            elif "$ref" in props:
                return self._pydantic_subschema_to_default_dtype(
                    self.defs[props["$ref"].split("/")[-1]]
                )
            return None

        pyd_type = props.get("type")
        if pyd_type == "numeric":
            pyd_type = "number"

        elif pyd_type == "array":
            if "items" not in props:
                raise NotImplementedError(
                    "Unexpected error processing pydantic schema. Please file an issue."
                )
            array_props = props["items"]
            inner_default_type = self._default_polars_dtype_for_schema(array_props)
            if inner_default_type is None:
                return None
            return pl.List(inner_default_type)

        elif pyd_type == "object":  # these are structs
            if "properties" not in props:
                raise NotImplementedError(
                    "dictionaries not currently supported by patito"
                )
            object_props: dict[str, dict[str, str]] = props["properties"]
            struct_fields: list[pl.Field] = []

            for name, sub_props in object_props.items():
                dtype = self._default_polars_dtype_for_schema(sub_props)
                assert dtype is not None
                struct_fields.append(pl.Field(name, dtype))
            return pl.Struct(struct_fields)

        return _pyd_type_to_default_dtype(
            PydanticBaseType(pyd_type), props.get("format"), props.get("enum")
        )
