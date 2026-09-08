"""Module for validating datastructures with respect to model specifications."""

from __future__ import annotations

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import polars as pl
from pydantic.aliases import AliasGenerator
from typing_extensions import get_args

from patito._pydantic.dtypes import is_optional
from patito._pydantic.dtypes.utils import unwrap_optional
from patito.exceptions import (
    ColumnDTypeError,
    DataFrameValidationError,
    ErrorWrapper,
    MissingColumnsError,
    MissingValuesError,
    RowValueError,
    SuperfluousColumnsError,
)

try:
    import pandas as pd  # type: ignore

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    from patito import Model


VALID_POLARS_TYPES = {
    "enum": {pl.Categorical},
    "boolean": {pl.Boolean},
    "string": {pl.String, pl.Datetime, pl.Date},
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


def _transform_df(dataframe: pl.DataFrame, schema: type[Model]) -> pl.DataFrame:
    """Transform any properties of the dataframe according to the model.

    Currently only supports using AliasGenerator to transform column names to match a model.

    Args:
        dataframe: Polars DataFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.

    """
    # Check if an alias generator is present in model_config
    if alias_gen := schema.model_config.get("alias_generator"):
        if isinstance(alias_gen, AliasGenerator):
            alias_func = alias_gen.validation_alias or alias_gen.alias
            assert alias_func is not None, (
                "An AliasGenerator must contain a transforming function"
            )
        else:  # alias_gen is a function
            alias_func = alias_gen

        new_cols: list[str] = [
            alias_func(field_name) for field_name in dataframe.columns
        ]  # type: ignore
        dataframe.columns = new_cols
    return dataframe


def _find_errors(  # noqa: C901
    dataframe: pl.DataFrame,
    schema: type[Model],
    columns: Sequence[str] | None = None,
    allow_missing_columns: bool = False,
    allow_superfluous_columns: bool = False,
) -> list[ErrorWrapper]:
    """Validate the given dataframe.

    Args:
        dataframe: Polars DataFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.
        columns: If specified, only validate the given columns. Missing columns will
            check if any specified columns are missing from the inputted dataframe,
            and superfluous columns will check if any columns not specified in the
            schema are present in the columns list.
        allow_missing_columns: If True, missing columns will not be considered an error.
        allow_superfluous_columns: If True, additional columns will not be considered an error.

    Returns:
        A list of patito.exception.ErrorWrapper instances. The specific validation
        error can be retrieved from the "exc" attribute on each error wrapper instance.

        MissingColumnsError: If there are any missing columns.
        SuperfluousColumnsError: If there are additional, non-specified columns.
        MissingValuesError: If there are nulls in a non-optional column.
        ColumnDTypeError: If any column has the wrong dtype.
        NotImplementedError: If validation has not been implement for the given
            type.

    """
    errors: list[ErrorWrapper] = []
    schema_subset = columns or schema.columns
    column_subset = columns or dataframe.columns
    if not allow_missing_columns:
        # Check if any columns are missing
        for missing_column in set(schema_subset) - set(dataframe.columns):
            col_info = schema.column_infos.get(missing_column)
            if col_info is not None and col_info.allow_missing:
                continue

            errors.append(
                ErrorWrapper(
                    MissingColumnsError("Missing column"),
                    loc=missing_column,
                )
            )

    if not (allow_superfluous_columns or schema.model_config.get("extra") == "allow"):
        # Check if any additional columns are included
        for superfluous_column in set(column_subset) - set(schema.columns):
            errors.append(
                ErrorWrapper(
                    SuperfluousColumnsError("Superfluous column"),
                    loc=superfluous_column,
                )
            )

    # Check if any non-optional columns have null values
    for column in schema.non_nullable_columns.intersection(column_subset):
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
        if column not in column_subset:
            continue
        if not isinstance(dtype, pl.List):
            continue

        annotation = schema.model_fields[column].annotation  # type: ignore[unreachable]

        # Retrieve the annotation of the list itself,
        # dewrapping any potential Optional[...]
        list_type = unwrap_optional(annotation)

        # Check if the list items themselves should be considered nullable
        item_type = get_args(list_type)[0]
        if is_optional(item_type):
            continue

        num_missing_values = (
            dataframe.lazy()
            .select(column)
            # Remove those rows that do not contain lists at all
            .filter(pl.col(column).is_not_null())
            # Remove empty lists
            .filter(pl.col(column).list.len() > 0)
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
        # We rename to _tmp here to avoid overwriting the dataframe during filters below
        # TODO! Really we should be passing *Series* around rather than the entire dataframe
        dataframe_tmp = dataframe
        column_info = schema.column_infos[column_name]
        if column_name not in dataframe_tmp.columns or column_name not in column_subset:
            continue

        polars_type = dataframe_datatypes[column_name]
        if polars_type not in [
            pl.Struct,
            pl.List(pl.Struct),
        ]:  # defer struct validation for recursive call to _find_errors later
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
        e = _find_enum_errors(
            df=dataframe_tmp,
            column_name=column_name,
            props=column_properties,
            schema=schema,
        )
        if e is not None:
            errors.append(e)

        if column_info.unique:
            # Coalescing to 0 in the case of dataframe of height 0
            num_duplicated = dataframe_tmp[column_name].is_duplicated().sum() or 0
            if num_duplicated > 0:
                errors.append(
                    ErrorWrapper(
                        RowValueError(f"{num_duplicated} rows with duplicated values."),
                        loc=column_name,
                    )
                )

        # Intercept struct columns, and process errors separately
        if schema.dtypes[column_name] == pl.Struct:
            nested_schema = schema.model_fields[column_name].annotation
            assert nested_schema is not None
            # Additional unpack required if structs column is optional
            if is_optional(nested_schema):
                nested_schema = unwrap_optional(nested_schema)

                # An optional struct means that we allow the struct entry to be
                # null. It is the inner model that is responsible for determining
                # whether its fields are optional or not. Since the struct is optional,
                # we need to filter out any null rows as the inner model may disallow
                # nulls on a particular field

                # NB As of Polars 1.1, struct_col.is_null() cannot return True
                # The following code has been added to accomodate this

                struct_fields = dataframe_tmp[column_name].struct.fields
                col_struct = pl.col(column_name).struct
                only_non_null_expr = ~pl.all_horizontal(
                    [col_struct.field(name).is_null() for name in struct_fields]
                )
                dataframe_tmp = dataframe_tmp.filter(only_non_null_expr)
                if dataframe_tmp.is_empty():
                    continue

            struct_errors = _find_errors(
                dataframe=dataframe_tmp.select(column_name).unnest(column_name),
                schema=nested_schema,
            )

            # Format nested errors
            for error in struct_errors:
                error._loc = f"{column_name}.{error._loc}"

            errors.extend(struct_errors)

            # No need to do any more checks
            continue

        # Intercept list of structs columns, and process errors separately
        elif schema.dtypes[column_name] == pl.List(pl.Struct):
            list_annotation = schema.model_fields[column_name].annotation
            assert list_annotation is not None

            # Handle Optional[list[pl.Struct]]
            if is_optional(list_annotation):
                list_annotation = unwrap_optional(list_annotation)

                dataframe_tmp = dataframe_tmp.filter(pl.col(column_name).is_not_null())
                if dataframe_tmp.is_empty():
                    continue

            # Unpack list schema
            nested_schema = list_annotation.__args__[0]

            dataframe_tmp = (
                dataframe_tmp.select(column_name)
                .filter(pl.col(column_name).list.len() > 0)
                .explode(column_name)
                .unnest(column_name)
            )

            # Handle list[Optional[pl.Struct]]
            if is_optional(nested_schema):
                nested_schema = unwrap_optional(nested_schema)

                dataframe_tmp = dataframe_tmp.filter(pl.all().is_not_null())
                if dataframe_tmp.is_empty():
                    continue

            list_struct_errors = _find_errors(
                dataframe=dataframe_tmp,
                schema=nested_schema,
            )

            # Format nested errors
            for error in list_struct_errors:
                error._loc = f"{column_name}.{error._loc}"

            errors.extend(list_struct_errors)

            # No need to do any more checks
            continue

        # Check for bounded value fields
        col = pl.col(column_name)
        filters = {
            "maximum": lambda v, col=col: col <= v,
            "exclusiveMaximum": lambda v, col=col: col < v,
            "minimum": lambda v, col=col: col >= v,
            "exclusiveMinimum": lambda v, col=col: col > v,
            "multipleOf": lambda v, col=col: (col == 0) | ((col % v) == 0),
            "const": lambda v, col=col: col == v,
            "pattern": lambda v, col=col: col.str.contains(v),
            "minLength": lambda v, col=col: col.str.len_chars() >= v,
            "maxLength": lambda v, col=col: col.str.len_chars() <= v,
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
                lazy_df = dataframe_tmp.lazy()
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

        if column_info.constraints is not None:
            custom_constraints = column_info.constraints
            if isinstance(custom_constraints, pl.Expr):
                custom_constraints = [custom_constraints]
            constraints = pl.any_horizontal(
                [constraint.not_() for constraint in custom_constraints]
            )
            if "_" in constraints.meta.root_names():
                # An underscore is an alias for the current field
                illegal_rows = dataframe_tmp.with_columns(
                    pl.col(column_name).alias("_")
                ).filter(constraints)
            else:
                illegal_rows = dataframe_tmp.filter(constraints)
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


def _find_enum_errors(
    df: pl.DataFrame, column_name: str, props: dict[str, Any], schema: type[Model]
) -> ErrorWrapper | None:
    if "enum" not in props:
        if "items" in props and "enum" in props["items"]:
            return _find_enum_errors(df, column_name, props["items"], schema)
        for item in props.get("anyOf", []):
            if "enum" in item:
                return _find_enum_errors(df, column_name, item, schema)
            if (
                "$ref" in item
            ):  # If the item is a reference to another definition pass it as the properties
                return _find_enum_errors(
                    df,
                    column_name,
                    schema.model_json_schema()["$defs"][item["$ref"]],
                    schema,
                )
        return None
    permissible_values = set(props["enum"])
    if column_name in schema.nullable_columns:
        permissible_values.add(None)
    if isinstance(df[column_name].dtype, pl.List):
        actual_values = set(df[column_name].explode().unique())
    else:
        actual_values = set(df[column_name].unique())
    impermissible_values = actual_values - permissible_values
    if impermissible_values:
        return ErrorWrapper(
            RowValueError(f"Rows with invalid values: {impermissible_values}."),
            loc=column_name,
        )
    return None


def validate(
    dataframe: pd.DataFrame | pl.DataFrame,
    schema: type[Model],
    columns: Sequence[str] | None = None,
    allow_missing_columns: bool = False,
    allow_superfluous_columns: bool = False,
    drop_superfluous_columns: bool = False,
) -> pl.DataFrame:
    """Validate the given dataframe.

    Args:
        dataframe: Polars DataFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.
        columns: Optional list of columns to validate. If not provided, all columns
            of the dataframe will be validated.
        allow_missing_columns: If True, missing columns will not be considered an error.
        allow_superfluous_columns: If True, additional columns will not be considered an error.
        drop_superfluous_columns: If True, drop any columns not specified in the schema before validation.

    Raises:
        DataFrameValidationError: If the given dataframe does not match the given schema.

    """
    if drop_superfluous_columns and columns:
        raise ValueError(
            "Cannot specify both 'columns' and 'drop_superfluous_columns'."
        )

    if _PANDAS_AVAILABLE and isinstance(dataframe, pd.DataFrame):
        polars_dataframe = pl.from_pandas(dataframe)
    else:
        polars_dataframe = cast(pl.DataFrame, dataframe).clone()

    polars_dataframe = _transform_df(polars_dataframe, schema)

    if drop_superfluous_columns:
        # NOTE: dropping rather than selecting to get the correct error messages
        to_drop = set(dataframe.columns) - set(schema.columns)
        polars_dataframe = polars_dataframe.drop(to_drop)

    errors = _find_errors(
        dataframe=polars_dataframe,
        schema=schema,
        columns=columns,
        allow_missing_columns=allow_missing_columns,
        allow_superfluous_columns=allow_superfluous_columns,
    )
    if errors:
        raise DataFrameValidationError(errors=errors, model=schema)

    return polars_dataframe
