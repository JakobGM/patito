"""Module for validating datastructures with respect to model specifications."""
from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union, cast

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
            "minLength": lambda v: col.str.lengths() >= v,
            "maxLength": lambda v: col.str.lengths() <= v,
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
            invalid_rows = dataframe.height - valid_rows.height
            if invalid_rows > 0:
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
            constraints = pl.all(
                [constraint.is_not() for constraint in custom_constraints]
            )
            if "_" in constraints.meta.root_names():
                # An underscore is an alias for the current field
                illegal_rows = dataframe.with_column(
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
        raise ValidationError(errors=errors, model=schema)
