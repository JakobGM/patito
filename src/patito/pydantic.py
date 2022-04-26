from __future__ import annotations

import itertools
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any, Optional, Type, TypeVar, Union

import pandas as pd
import polars as pl
from pydantic import BaseConfig, BaseModel, Field  # noqa: F401
from pydantic.main import ModelMetaclass as PydanticModelMetaclass

from patito.polars import DataFrame
from patito.validators import validate

# The generic type of a single row in given Relation.
# Should be a typed subclass of Model.
ModelType = TypeVar("ModelType", bound="Model")

# A mapping from pydantic types to the equivalent type used in DuckDB
PYDANTIC_TO_DUCKDB_TYPES = {
    "integer": "BIGINT",
    "string": "VARCHAR",
    "number": "DOUBLE",
    "boolean": "BOOLEAN",
}

# A mapping from pydantic types to equivalent dtypes used in polars
PYDANTIC_TO_POLARS_TYPES = {
    "integer": pl.Int64,
    "string": pl.Utf8,
    "number": pl.Float64,
    "boolean": pl.Boolean,
}


class ModelMetaclass(PydanticModelMetaclass):
    def __init__(cls, name: str, bases: tuple, clsdict: dict) -> None:
        """Construct new patito.Model class."""
        super().__init__(name, bases, clsdict)  # type: ignore
        # Add a custom subclass of patito.DataFrame to the model class,
        # where .set_model() has been implicitly set.
        cls.DataFrame = DataFrame._construct_dataframe_model_class(
            model=cls,  # type: ignore
        )


class Model(BaseModel, metaclass=ModelMetaclass):
    """Custom pydantic class for representing table schema and constructing rows."""

    @classmethod
    @property
    def DataFrame(cls: Type[ModelType]) -> Type[DataFrame[ModelType]]:  # type: ignore
        """Return DataFrame class where DataFrame.set_model() is set to self."""

    @classmethod
    def from_row(
        cls: Type[ModelType],
        row: Union[pd.DataFrame, pl.DataFrame],
        validate: bool = True,
    ) -> ModelType:
        if isinstance(row, pd.DataFrame):
            dataframe = pl.DataFrame._from_pandas(row)
        elif isinstance(row, pd.Series):
            return cls(**dict(row.iteritems()))
        elif isinstance(row, pl.DataFrame):
            dataframe = row
        else:
            raise TypeError(f"{cls.__name__}.from_row not implemented for {type(row)}.")
        return cls.from_polars(dataframe=dataframe, validate=validate)

    @classmethod
    def from_polars(
        cls: Type[ModelType],
        dataframe: pl.DataFrame,
        validate: bool = True,
    ) -> ModelType:
        """Construct model from a single polars row."""
        if not isinstance(dataframe, pl.DataFrame):
            raise TypeError(
                f"{cls.__name__}.from_polars() must be invoked with polars.DataFrame, "
                f"not {type(dataframe)}!"
            )
        elif len(dataframe) != 1:
            raise ValueError(
                f"{cls.__name__}.from_polars() can only be invoked with exactly "
                f"1 row, while {len(dataframe)} rows were provided."
            )

        # We have been provided with a single polars.DataFrame row
        # Convert to the equivalent keyword invocation of the pydantic model
        if validate:
            return cls(**dataframe.to_dicts()[0])
        else:
            return cls.construct(**dataframe.to_dicts()[0])

    @classmethod
    def validate(cls, dataframe: Union[pd.DataFrame, pl.DataFrame]) -> None:
        """
        Validate the given dataframe.

        Args:
            dataframe: Polars DataFrame to be validated.

        Raises:
            patito.exceptions.ValidationError: If the given dataframe does not match the
                given schema.
        """
        validate(dataframe=dataframe, schema=cls)

    @classmethod
    @property
    def columns(cls) -> tuple[str, ...]:
        """Return tuple containing column names of row."""
        return tuple(cls.schema()["properties"].keys())

    @classmethod
    @property
    def defaults(cls) -> dict[str, Any]:
        """Return dictionary containing fields with their respective default values."""
        return {
            field_name: props["default"]
            for field_name, props in cls.schema()["properties"].items()
            if "default" in props
        }

    @classmethod
    def example_value(cls, field: str) -> Any:  # noqa: C901
        """Return an example value for the given field name defined on the model."""
        schema = cls.schema()
        field_data = schema["properties"]
        non_nullable = schema["required"]
        properties = field_data[field]
        field_type = properties["type"]
        if "const" in properties:
            # The default value is the only valid value, provided as const
            return properties["const"]

        elif "default" in properties:
            # A default value has been specified in the model field definiton
            return properties["default"]

        elif "enum" in properties:
            return properties["enum"][0]

        elif field not in non_nullable:
            return None

        elif field_type in {"integer", "number"}:
            # For integer and float types we must check if there are imposed bounds
            lower = properties.get("minimum") or properties.get("exclusiveMinimum")
            upper = properties.get("maximum") or properties.get("exclusiveMaximum")

            # If the dtype is an unsigned integer type, we must return a positive value
            if dtype := properties.get("dtype", False):
                if dtype in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                    lower = 0 if lower is None else max(lower, 0)

            # First we check the simple case, no upper or lower bound
            if lower is None and upper is None:
                if field_type == "number":
                    return -0.5
                else:
                    return -1

            # If we have a lower and upper bound, we return something in the middle
            elif lower is not None and upper is not None:
                if field_type == "number":
                    return (lower + upper) / 2
                else:
                    return (lower + upper) // 2

            # What remains is a single-sided bound, which we will return a value on the
            # "right side" of.
            number = float if field_type == "number" else int
            if lower is not None:
                return number(lower + 1)
            else:
                return number(upper - 1)

        elif field_type == "string":
            if "pattern" in properties:
                raise NotImplementedError(
                    "Example data generation has not been implemented for regex "
                    "patterns. You must valid data for such columns explicitly!"
                )
            elif "format" in properties and properties["format"] == "date":
                return "1970-01-01"
            elif "minLength" in properties:
                return "a" * properties["minLength"]
            elif "maxLength" in properties:
                return "a" * min(properties["maxLength"], 1)
            else:
                return "dummy_string"

        elif field_type == "boolean":
            return False

        else:
            raise NotImplementedError

    @classmethod
    def dummy(cls: Type[ModelType], **kwargs) -> ModelType:
        """
        Produce model with dummy data for all unspecified fields.

        The type annotation of unspecified field is used to fill in type-correct
        dummy data, e.g. -1 for int, "dummy_string" for str, and so on...
        The first item of Literal annotatations are used for dummy values.
        """
        # Non-iterable values besides strings must be repeated
        if wrong_columns := set(kwargs.keys()) - set(cls.columns):
            raise TypeError(f"{cls.__name__} does not contain fields {wrong_columns}!")

        schema = cls.schema()
        properties = schema["properties"]
        new_kwargs = {}
        for field_name in properties.keys():
            if field_name in kwargs:
                # The value has been explicitly specified
                new_kwargs[field_name] = kwargs[field_name]
            else:
                new_kwargs[field_name] = cls.example_value(field=field_name)
        return cls(**new_kwargs)

    @classmethod
    def example_pandas(
        cls: Type[ModelType],
        data: Union[dict, Iterable],
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate dataframe with dummy data for all unspecified columns.

        Offers the same API as the pandas.DataFrame constructor.
        Non-iterable values, besides strings, are repeated until they become as long as
        the iterable arguments.

        Args:
            data (Union[dict, Iterable]): Data to populate the dummy dataframe with. If
            not a dict, column names must also be provided.
            columns (Optional[Iterable[str]], optional): Ignored if data is a dict. If
            data is an iterable, it will be used as the column names in the resulting
            dataframe. Defaults to None.
            polars (bool, optional): If True, returns a polars DataFrame, else returns a
            pandas DataFrame. Equivalent to running .dummy_df(). Defaults to False.
        """
        if not isinstance(data, dict):
            if columns is None:
                raise TypeError(
                    f"{cls.__name__}.dummy_df() must be provided with column names!"
                )
            kwargs = dict(zip(columns, zip(*data)))
        else:
            kwargs = data

        kwargs = {
            key: (
                value
                if isinstance(value, Iterable) and not isinstance(value, str)
                else itertools.cycle([value])
            )
            for key, value in kwargs.items()
        }
        dummies = []
        for values in zip(*kwargs.values()):
            dummies.append(cls.dummy(**dict(zip(kwargs.keys(), values))))
        return pd.DataFrame([dummy.dict() for dummy in dummies])

    @classmethod
    def example(
        cls: Type[ModelType],
        data: Optional[Union[dict, Iterable]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pl.DataFrame:
        """
        Generate polars dataframe with dummy data for all unspecified columns.

        data: Data to populate the dummy dataframe with. If given as an iterable of
            values then column names must also be provided. If not provided at all,
            an empty dataframe with the correct column dtypes will be generated instead.
        columns: Ignored if data is a dict. If data is an iterable, it will be used as
            the column names in the resulting dataframe. Defaults to None.
        """
        if data is None:
            # We should create an empty dataframe, but with the correct dtypes
            kwargs = {}
        elif not isinstance(data, dict):
            if columns is None:
                raise TypeError(
                    f"{cls.__name__}.dummy_df() must be provided with column names!"
                )
            kwargs = dict(zip(columns, zip(*data)))
        else:
            kwargs = data

        if wrong_columns := set(kwargs.keys()) - set(cls.columns):
            raise TypeError(f"{cls.__name__} does not contain fields {wrong_columns}!")

        series = []
        unique_series = []
        for column_name, dtype in cls.dtypes.items():
            if column_name not in kwargs:
                if column_name in cls.unique_columns:
                    unique_series.append(
                        pl.first().cumcount().cast(dtype).alias(column_name)
                    )
                else:
                    example_value = cls.example_value(field=column_name)
                    series.append(pl.lit(example_value, dtype=dtype).alias(column_name))
                continue

            value = kwargs.get(column_name)
            if isinstance(value, Iterable) and not isinstance(value, str):
                # We make sure that at least one series is inserted first in the list,
                # otherwise polars will not be able to handle the shape mismatch between
                # series and literate values.
                series.insert(0, pl.Series(name=column_name, values=value, dtype=dtype))
            else:
                series.append(pl.lit(value, dtype=dtype).alias(column_name))

        return pl.DataFrame().with_columns(series).with_columns(unique_series)

    @classmethod
    @property
    def non_nullable_columns(cls: Type[ModelType]) -> set[str]:
        """Return names of those columns that are non-nullable in the schema."""
        return set(cls.schema()["required"])

    @classmethod
    @property
    def nullable_columns(cls: Type[ModelType]) -> set[str]:
        """Return names of those columns that are nullable in the schema."""
        return set(cls.columns) - cls.non_nullable_columns

    @classmethod
    @property
    def unique_columns(cls: Type[ModelType]) -> set[str]:
        """Return columns with uniqueness constraint."""
        props = cls.schema()["properties"]
        return {column for column in cls.columns if props[column].get("unique", False)}

    @classmethod
    @property
    def sql_types(cls: Type[ModelType]) -> dict[str, str]:
        """Return SQL types as a column name -> sql type dict mapping."""
        schema = cls.schema()
        props = schema["properties"]
        return {
            column: PYDANTIC_TO_DUCKDB_TYPES[props[column]["type"]]
            for column in cls.columns
        }

    @contextmanager
    def as_unfrozen(self):
        """Yield the model as a temporarily mutable pydantic model."""
        self.__config__.frozen = False
        try:
            yield self
        finally:
            self.__config__.frozen = True

    @classmethod
    @property
    def valid_dtypes(  # noqa: C901
        cls: Type[ModelType],
    ) -> dict[str, tuple[Type[pl.DataType], ...]]:
        """
        Return valid polars dtypes as a column name -> dtypes mapping.

        The first item of each tuple is the default dtype chosen by Patito.
        """
        schema = cls.schema()
        properties = schema["properties"]

        valid_dtypes = {}
        for column, props in properties.items():
            if "dtype" in props:
                valid_dtypes[column] = (props["dtype"],)
            elif "enum" in props:
                if props["type"] != "string":
                    raise NotImplementedError
                valid_dtypes[column] = (pl.Categorical, pl.Utf8)
            elif props["type"] == "integer":
                valid_dtypes[column] = (
                    pl.Int64,
                    pl.Int32,
                    pl.Int16,
                    pl.Int8,
                    pl.UInt64,
                    pl.UInt32,
                    pl.UInt16,
                    pl.UInt8,
                )
            elif props["type"] == "number":
                if props.get("format") == "time-delta":
                    valid_dtypes[column] = (
                        pl.Duration,
                    )  # pyright: reportPrivateImportUsage=false
                else:
                    valid_dtypes[column] = (pl.Float64, pl.Float32)
            elif props["type"] == "boolean":
                valid_dtypes[column] = (pl.Boolean,)
            elif props["type"] == "string":
                string_format = props.get("format")
                if string_format is None:
                    valid_dtypes[column] = (pl.Utf8,)
                elif string_format == "date":
                    valid_dtypes[column] = (pl.Date,)
                elif string_format == "date-time":
                    valid_dtypes[column] = (pl.Datetime,)
            elif props["type"] == "null":
                valid_dtypes[column] = (pl.Null,)
            else:
                raise NotImplementedError

        return valid_dtypes

    @classmethod
    @property
    def dtypes(cls: Type[ModelType]) -> dict[str, Type[pl.DataType]]:
        """
        Return polars dtypes as a column name -> polars type dict mapping.

        Unless Field(dtype=...) is specified, the highest signed column dtype
        is chosen for integer and float columns.
        """
        return {
            column: valid_dtypes[0] for column, valid_dtypes in cls.valid_dtypes.items()
        }

    class Config(BaseConfig):
        """Configuration for Pydantic BaseModel behaviour."""

        # Make fields immutable and model hashable
        frozen = True
