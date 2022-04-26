"""Module for validating datastructures with respect to model specifications."""
from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union, cast

import pandas as pd
import polars as pl

from patito.exceptions import (
    ColumnDTypeError,
    ErrorWrapper,
    MissingColumnsError,
    MissingValuesError,
    RowValueError,
    SuperflousColumnsError,
    ValidationError,
)

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


def _find_errors(  # noqa: C901
    dataframe: pl.DataFrame,
    schema: Type[Model],
) -> list[ErrorWrapper]:
    """
    Validate the given dataframe.

    Args:
        dataframe: Polars DataFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.

    Return:
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
    for missig_column in set(schema.columns) - set(dataframe.columns):
        errors.append(
            ErrorWrapper(
                MissingColumnsError("Missing column"),
                loc=missig_column,
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
        if num_missing_values := dataframe.get_column(name=column).null_count():
            errors.append(
                ErrorWrapper(
                    MissingValuesError(
                        f"{num_missing_values} missing "
                        f"{'value' if num_missing_values == 1 else 'values'}"
                    ),
                    loc=column,
                )
            )

    # Check if any column has a wrong dtype
    valid_dtypes = schema.valid_dtypes
    model_schema = schema.schema()
    dataframe_datatypes = dict(zip(dataframe.columns, dataframe.dtypes))
    for column_name, column_properties in model_schema["properties"].items():
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
            if num_duplicated := dataframe[column_name].is_duplicated().sum():
                errors.append(
                    ErrorWrapper(
                        RowValueError(f"{num_duplicated} rows with duplicated values."),
                        loc=column_name,
                    )
                )

        # Check for bounded value fields
        column = pl.col(column_name)
        filters = {
            "maximum": lambda v: column <= v,
            "exclusiveMaximum": lambda v: column < v,
            "minimum": lambda v: column >= v,
            "exclusiveMinimum": lambda v: column > v,
            "multipleOf": lambda v: (column == 0) | ((column % v) == 0),
            "const": lambda v: column == v,
            "pattern": lambda v: column.str.contains(v),
            "minLength": lambda v: column.str.lengths() >= v,
            "maxLength": lambda v: column.str.lengths() <= v,
        }
        checks = [
            check(column_properties[key])
            for key, check in filters.items()
            if key in column_properties
        ]
        if checks:
            lazy_df = dataframe.lazy()
            for check in checks:
                lazy_df = lazy_df.filter(check)
            valid_rows = lazy_df.collect()
            if invalid_rows := dataframe.height - valid_rows.height:
                errors.append(
                    ErrorWrapper(
                        RowValueError(
                            f"{invalid_rows} row{'' if invalid_rows == 1 else 's'} "
                            "with out of bound values."
                        ),
                        loc=column_name,
                    )
                )

        if "constraints" in column_properties:
            custom_constraints = column_properties["constraints"]
            if isinstance(custom_constraints, pl.Expr):
                custom_constraints = [custom_constraints]
            illegal_rows = dataframe.filter(
                pl.all([constraint.is_not() for constraint in custom_constraints])
            )
            if num_illegal_rows := illegal_rows.height:
                errors.append(
                    ErrorWrapper(
                        RowValueError(
                            f"{num_illegal_rows} "
                            f"row{'' if num_illegal_rows == 1 else 's'} "
                            "does not match custom constraints."
                        ),
                        loc=column_name,
                    )
                )

    return errors


def validate(dataframe: Union[pd.DataFrame, pl.DataFrame], schema: Type[Model]) -> None:
    """
    Validate the given dataframe.

    Args:
        dataframe: Polars DataFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.

    Raises:
        patito.exceptions.ValidationError: If the given dataframe does not match the
            given schema.
    """
    if isinstance(dataframe, pd.DataFrame):
        dataframe = cast(pl.DataFrame, pl.from_pandas(dataframe))
    errors = _find_errors(dataframe=dataframe, schema=schema)
    if errors:
        raise ValidationError(errors=errors, model=schema)
