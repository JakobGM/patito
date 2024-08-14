from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, Any, get_args

from pydantic.fields import FieldInfo

from patito._pydantic.column_info import ColumnInfo
from patito._pydantic.dtypes import PYTHON_TO_PYDANTIC_TYPES

if TYPE_CHECKING:
    from patito.pydantic import ModelType


@cache
def schema_for_model(cls: type[ModelType]) -> dict[str, dict[str, Any]]:
    """Return schema properties where definition references have been resolved.

    Returns:
        Field information as a dictionary where the keys are field names and the
            values are dictionaries containing metadata information about the field
            itself.

    Raises:
        TypeError: if a field is annotated with an enum where the values are of
            different types.

    """
    schema = cls.model_json_schema(by_alias=False, ref_template="{model}")
    fields = {}
    # first resolve definitions for nested models TODO checks for one-way references, if models are self-referencing this falls apart with recursion depth error
    for f in cls.model_fields.values():
        annotation = f.annotation
        cls._update_dfn(annotation, schema)
        for a in get_args(annotation):
            cls._update_dfn(a, schema)
    for field_name, field_info in schema["properties"].items():
        fields[field_name] = _append_field_info_to_props(
            field_info=field_info,
            field_name=field_name,
            required=field_name in schema.get("required", set()),
            model_schema=schema,
        )
    schema["properties"] = fields
    return schema


@cache
def column_infos_for_model(cls: type[ModelType]) -> Mapping[str, ColumnInfo]:
    fields = cls.model_fields

    def get_column_info(field: FieldInfo) -> ColumnInfo:
        if field.json_schema_extra is None:
            return ColumnInfo()
        elif callable(field.json_schema_extra):
            raise NotImplementedError(
                "Callable json_schema_extra not supported by patito."
            )
        return ColumnInfo.model_validate_json(field.json_schema_extra["column_info"])

    return {k: get_column_info(v) for k, v in fields.items()}


def _append_field_info_to_props(
    field_info: dict[str, Any],
    field_name: str,
    model_schema: dict[str, Any],
    required: bool | None = None,
) -> dict[str, Any]:
    if "$ref" in field_info:  # TODO onto runtime append
        definition = model_schema["$defs"][field_info["$ref"]]
        if "enum" in definition and "type" not in definition:
            enum_types = set(type(value) for value in definition["enum"])
            if len(enum_types) > 1:
                raise TypeError(
                    "All enumerated values of enums used to annotate "
                    "Patito model fields must have the same type. "
                    "Encountered types: "
                    f"{sorted(map(lambda t: t.__name__, enum_types))}."
                )
            enum_type = enum_types.pop()
            definition["type"] = PYTHON_TO_PYDANTIC_TYPES[enum_type]
        field = definition
    else:
        field = field_info
    if "items" in field_info:
        field["items"] = _append_field_info_to_props(
            field_info=field_info["items"],
            field_name=field_name,
            model_schema=model_schema,
        )
    if required is not None:
        field["required"] = required
    if "const" in field_info and "type" not in field_info:
        field["type"] = PYTHON_TO_PYDANTIC_TYPES[type(field_info["const"])]
    return field
