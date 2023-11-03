"""Module for validating datastructures with respect to model specifications."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Type, Union, cast, get_args, get_origin

import polars as pl

from patito.exceptions import (
    ColumnDTypeError,
    ErrorWrapper,
    MissingColumnsError,
    MissingValuesError,
    RowValueError,
    SuperflousColumnsError,
    DataFrameValidationError,
)

if sys.version_info >= (3, 10):  # pragma: no cover
    from types import UnionType  # pyright: ignore

    UNION_TYPES = (Union, UnionType)
else:
    UNION_TYPES = (Union,)  # pragma: no cover

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    from patito import Model


VALID_POLARS_TYPES = {
    "enum": {pl.Categorical},
    "boolean": {pl.Boolean},
    "string": {pl.Utf8, pl.Datetime, pl.Date},
    "number": {pl.Float32, pl.Float64},
    "integer": {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    },
}


def _is_optional(type_annotation: Type) -> bool:
    """
    Return True if the given type annotation is an Optional annotation.

    Args:
        type_annotation: The type annotation to be checked.

    Returns:
        True if the outermost type is Optional.
    """
    return (get_origin(type_annotation) in UNION_TYPES) and (
        type(None) in get_args(type_annotation)
    )


def _dewrap_optional(type_annotation: Type) -> Type:
    """
    Return the inner, wrapped type of an Optional.

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
        if _is_optional(type_annotation)
        else type_annotation
    )


def _find_errors(  # noqa: C901
    dataframe: pl.DataFrame,
    schema: Type[Model],
) -> list[ErrorWrapper]:
    """
    Validate the given dataframe.

    Args:
        dataframe: Polars DataFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.

    Returns:
        A list of patito.exception.ErrorWrapper instances. The specific validation
        error can be retrieved from the "exc" attribute on each error wrapper instance.

        MissingColumnsError: If there are any missing columns.
        SuperflousColumnsError: If there are additional, non-specified columns.
        MissingValuesError: If there are nulls in a non-optional column.
        ColumnDTypeError: If any column has the wrong dtype.
        NotImplementedError: If validation has not been implement for the given
            type.
    """
    errors: list[ErrorWrapper] = []
    # Check if any columns are missing
    for missing_column in set(schema.columns) - set(dataframe.columns):
        errors.append(
            ErrorWrapper(
                MissingColumnsError("Missing column"),
                loc=missing_column,
            )
        )

    # Check if any additional columns are included
    for superflous_column in set(dataframe.columns) - set(schema.columns):
        errors.append(
            ErrorWrapper(
                SuperflousColumnsError("Superflous column"),
                loc=superflous_column,
            )
        )

    # Check if any non-optional columns have null values
    for column in schema.non_nullable_columns.intersection(dataframe.columns):
        num_missing_values = dataframe.get_column(name=column).null_count()
        if num_missing_values:
            errors.append(
                ErrorWrapper(
                    MissingValuesError(
                        f"{num_missing_values} missing "
                        f"{'value' if num_missing_values == 1 else 'values'}"
                    ),
                    loc=column,
                )
            )

    for column, dtype in schema.dtypes.items():
        if not isinstance(dtype, pl.List):
            continue

        annotation = schema.model_fields[column].annotation  # type: ignore[unreachable]

        # Retrieve the annotation of the list itself,
        # dewrapping any potential Optional[...]
        list_type = _dewrap_optional(annotation)

        # Check if the list items themselves should be considered nullable
        item_type = get_args(list_type)[0]
        if _is_optional(item_type):
            continue

        num_missing_values = (
            dataframe.lazy()
            .select(column)
            # Remove those rows that do not contain lists at all
            .filter(pl.col(column).is_not_null())
            # Convert lists of N items to N individual rows
            .explode(column)
            # Calculate how many nulls are present in lists
            .filter(pl.col(column).is_null())
            .collect()
            .height
        )
        if num_missing_values != 0:
            errors.append(
                ErrorWrapper(
                    MissingValuesError(
                        f"{num_missing_values} missing "
                        f"{'value' if num_missing_values == 1 else 'values'} "
                        f"in lists"
                    ),
                    loc=column,
                )
            )

    # Check if any column has a wrong dtype
    valid_dtypes = schema.valid_dtypes
    dataframe_datatypes = dict(zip(dataframe.columns, dataframe.dtypes))
    for column_name, column_properties in schema._schema_properties().items():
        if column_name not in dataframe.columns:
            continue

        polars_type = dataframe_datatypes[column_name]
        if polars_type not in valid_dtypes[column_name]:
            errors.append(
                ErrorWrapper(
                    ColumnDTypeError(
                        f"Polars dtype {polars_type} does not match model field type."
                    ),
                    loc=column_name,
                )
            )

        # Test for when only specific values are accepted
        if "enum" in column_properties:
            permissible_values = set(column_properties["enum"])
            if column_name in schema.nullable_columns:
                permissible_values.add(None)
            actual_values = set(dataframe[column_name].unique())
            impermissible_values = actual_values - permissible_values
            if impermissible_values:
                errors.append(
                    ErrorWrapper(
                        RowValueError(
                            f"Rows with invalid values: {impermissible_values}."
                        ),
                        loc=column_name,
                    )
                )

        if column_properties.get("unique", False):
            # Coalescing to 0 in the case of dataframe of height 0
            num_duplicated = dataframe[column_name].is_duplicated().sum() or 0
            if num_duplicated > 0:
                errors.append(
                    ErrorWrapper(
                        RowValueError(f"{num_duplicated} rows with duplicated values."),
                        loc=column_name,
                    )
                )

        # Check for bounded value fields
        col = pl.col(column_name)
        filters = {
            "maximum": lambda v: col <= v,
            "exclusiveMaximum": lambda v: col < v,
            "minimum": lambda v: col >= v,
            "exclusiveMinimum": lambda v: col > v,
            "multipleOf": lambda v: (col == 0) | ((col % v) == 0),
            "const": lambda v: col == v,
            "pattern": lambda v: col.str.contains(v),
            "minLength": lambda v: col.str.len_chars() >= v,
            "maxLength": lambda v: col.str.len_chars() <= v,
        }
        if "anyOf" in column_properties:
            checks = [
                check(x[key])
                for key, check in filters.items()
                for x in column_properties["anyOf"]
                if key in x
            ]
        else:
            checks = []
        checks += [
            check(column_properties[key])
            for key, check in filters.items()
            if key in column_properties
        ]
        if checks:
            n_invalid_rows = 0
            for check in checks:
                lazy_df = dataframe.lazy()
                lazy_df = lazy_df.filter(
                    ~check
                )  # get failing rows (nulls will evaluate to null on boolean check, we only want failures (false)))
                invalid_rows = lazy_df.collect()
                n_invalid_rows += invalid_rows.height
            if n_invalid_rows > 0:
                errors.append(
                    ErrorWrapper(
                        RowValueError(
                            f"{n_invalid_rows} row{'' if n_invalid_rows == 1 else 's'} "
                            "with out of bound values."
                        ),
                        loc=column_name,
                    )
                )

        if "constraints" in column_properties:
            custom_constraints = column_properties["constraints"]
            if isinstance(custom_constraints, pl.Expr):
                custom_constraints = [custom_constraints]
            constraints = pl.all_horizontal(
                [constraint.not_() for constraint in custom_constraints]
            )
            if "_" in constraints.meta.root_names():
                # An underscore is an alias for the current field
                illegal_rows = dataframe.with_columns(
                    pl.col(column_name).alias("_")
                ).filter(constraints)
            else:
                illegal_rows = dataframe.filter(constraints)
            if illegal_rows.height > 0:
                errors.append(
                    ErrorWrapper(
                        RowValueError(
                            f"{illegal_rows.height} "
                            f"row{'' if illegal_rows.height == 1 else 's'} "
                            "does not match custom constraints."
                        ),
                        loc=column_name,
                    )
                )

    return errors


def validate(
    dataframe: Union["pd.DataFrame", pl.DataFrame], schema: Type[Model]
) -> None:
    """
    Validate the given dataframe.

    Args:
        dataframe: Polars DataFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.

    Raises:
        ValidationError: If the given dataframe does not match the given schema.
    """
    if _PANDAS_AVAILABLE and isinstance(dataframe, pd.DataFrame):
        polars_dataframe = pl.from_pandas(dataframe)
    else:
        polars_dataframe = cast(pl.DataFrame, dataframe)

    errors = _find_errors(dataframe=polars_dataframe, schema=schema)
    if errors:
        raise DataFrameValidationError(errors=errors, model=schema)
