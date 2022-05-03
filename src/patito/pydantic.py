"""Logic related to wrapping logic around the pydantic library."""
from __future__ import annotations

import itertools
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any, ClassVar, Dict, List, Optional, Set, Type, TypeVar, Union

import polars as pl
from pydantic import BaseConfig, BaseModel, Field  # noqa: F401
from pydantic.main import ModelMetaclass as PydanticModelMetaclass

from patito.polars import DataFrame
from patito.validators import validate

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

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
    """
    Metclass used by patito.Model.

    Responsible for setting any relevant model-dependent class properties.
    """

    def __init__(cls, name: str, bases: tuple, clsdict: dict) -> None:
        """
        Construct new patito model.

        Args:
            name: Name of model class.
            bases: Tuple of superclasses.
            clsdict: Dictionary containing class properties.
        """
        super().__init__(name, bases, clsdict)  # type: ignore
        # Add a custom subclass of patito.DataFrame to the model class,
        # where .set_model() has been implicitly set.
        cls.DataFrame = DataFrame._construct_dataframe_model_class(
            model=cls,  # type: ignore
        )

    # --- Class properties ---
    # These properties will only be available on Model *classes*, not instantiated
    # objects This is backwards compatible to python versions before python 3.9,
    # unlike a combination of @classmethod and @property.
    @property
    def columns(cls: Type[ModelType]) -> List[str]:  # type: ignore
        """
        Return the name of the specified column fields the DataFrame.

        Returns:
            List of column names.
        """
        return list(cls.schema()["properties"].keys())

    @property
    def dtypes(  # type: ignore
        cls: Type[ModelType],
    ) -> dict[str, Type[pl.DataType]]:
        """
        Return the dtypes of the dataframe.

        Unless Field(dtype=...) is specified, the highest signed column dtype
        is chosen for integer and float columns.

        Returns:
            A dictionary mapping string column names to polars dtype classes.
        """
        return {
            column: valid_dtypes[0] for column, valid_dtypes in cls.valid_dtypes.items()
        }

    @property
    def valid_dtypes(  # type: ignore  # noqa: C901
        cls: Type[ModelType],
    ) -> dict[str, List[Type[pl.DataType]]]:
        """
        Return valid polars dtypes as a column name -> dtypes mapping.

        The first item of each list is the default dtype chosen by Patito.

        Returns:
            A dictionary mapping each column string name to a list of valid
            dtypes.

        Raises:
            NotImplementedError: If one or more model fields are annotated with types
                not compatible with polars.
        """
        schema = cls.schema()
        properties = schema["properties"]

        valid_dtypes = {}
        for column, props in properties.items():
            if "dtype" in props:
                valid_dtypes[column] = [
                    props["dtype"],
                ]
            elif "enum" in props:
                if props["type"] != "string":  # pragma: no cover
                    raise NotImplementedError
                valid_dtypes[column] = [pl.Categorical, pl.Utf8]
            elif "type" not in props:
                raise NotImplementedError(
                    f"No valid dtype mapping found for column '{column}'."
                )
            elif props["type"] == "integer":
                valid_dtypes[column] = [
                    pl.Int64,
                    pl.Int32,
                    pl.Int16,
                    pl.Int8,
                    pl.UInt64,
                    pl.UInt32,
                    pl.UInt16,
                    pl.UInt8,
                ]
            elif props["type"] == "number":
                if props.get("format") == "time-delta":
                    valid_dtypes[column] = [
                        pl.Duration,
                    ]  # pyright: reportPrivateImportUsage=false
                else:
                    valid_dtypes[column] = [pl.Float64, pl.Float32]
            elif props["type"] == "boolean":
                valid_dtypes[column] = [
                    pl.Boolean,
                ]
            elif props["type"] == "string":
                string_format = props.get("format")
                if string_format is None:
                    valid_dtypes[column] = [
                        pl.Utf8,
                    ]
                elif string_format == "date":
                    valid_dtypes[column] = [
                        pl.Date,
                    ]
                # TODO: Find out why this branch is not being hit
                elif string_format == "date-time":  # pragma: no cover
                    valid_dtypes[column] = [
                        pl.Datetime,
                    ]
            elif props["type"] == "null":
                valid_dtypes[column] = [
                    pl.Null,
                ]
            else:  # pragma: no cover
                raise NotImplementedError(
                    f"No valid dtype mapping found for column '{column}'"
                )

        return valid_dtypes

    @property
    def defaults(  # type: ignore
        cls: Type[ModelType],
    ) -> dict[str, Any]:
        """
        Return default field values specified on the model.

        Returns:
            Dictionary containing fields with their respective default values.
        """
        return {
            field_name: props["default"]
            for field_name, props in cls.schema()["properties"].items()
            if "default" in props
        }

    @property
    def non_nullable_columns(  # type: ignore
        cls: Type[ModelType],  # pyright: reportGeneralTypeIssues=false
    ) -> set[str]:
        """
        Return names of those columns that are non-nullable in the schema.

        Returns:
            Set of column name strings.
        """
        return set(cls.schema()["required"])

    @property
    def nullable_columns(  # type: ignore
        cls: Type[ModelType],  # pyright: reportGeneralTypeIssues=false
    ) -> set[str]:
        """
        Return names of those columns that are nullable in the schema.

        Returns:
            Set of column name strings.
        """
        return set(cls.columns) - cls.non_nullable_columns

    @property
    def unique_columns(  # type: ignore
        cls: Type[ModelType],
    ) -> set[str]:
        """
        Return columns with uniqueness constraint.

        Returns:
            Set of column name strings.
        """
        props = cls.schema()["properties"]
        return {column for column in cls.columns if props[column].get("unique", False)}

    @property
    def sql_types(  # type: ignore
        cls: Type[ModelType],
    ) -> dict[str, str]:
        """
        Return SQL types as a column name -> sql type dict mapping.

        Returns:
            Dictionary with column name keys and SQL type identifier strings.
        """
        schema = cls.schema()
        props = schema["properties"]
        return {
            column: PYDANTIC_TO_DUCKDB_TYPES[props[column]["type"]]
            for column in cls.columns
        }


class Model(BaseModel, metaclass=ModelMetaclass):
    """Custom pydantic class for representing table schema and constructing rows."""

    # -- Class properties set by model metaclass --
    # This weird combination of a MetaClass + type annotation
    # in order to make the following work simultaneously:
    #     1. Make these dynamically constructed properties of the class.
    #     2. Have the correct type information for type checkers.
    #     3. Allow sphinx-autodoc to construct correct documentation.
    #     4. Be compatible with python 3.7.
    # Once we drop support for python 3.7, we can replace all of this with just a simple
    # combination of @property and @classmethod.
    columns: ClassVar[List[str]]

    unique_columns: ClassVar[Set[str]]
    non_nullable_columns: ClassVar[Set[str]]
    nullable_columns: ClassVar[Set[str]]

    dtypes: ClassVar[Dict[str, Type[pl.DataType]]]
    sql_types: ClassVar[Dict[str, str]]
    valid_dtypes: ClassVar[Dict[str, List[Type[pl.DataType]]]]

    defaults: ClassVar[Dict[str, Any]]

    @classmethod
    @property
    def DataFrame(cls: Type[ModelType]) -> Type[DataFrame[ModelType]]:  # type: ignore
        """Return DataFrame class where DataFrame.set_model() is set to self."""

    @classmethod
    def from_row(
        cls: Type[ModelType],
        row: Union["pd.DataFrame", pl.DataFrame],
        validate: bool = True,
    ) -> ModelType:
        """
        Represent a single data frame row as a patito model.

        Args:
            row: A dataframe, either polars and pandas, consisting of a single row.
            validate: If False, skip pydantic validation of the given row data.

        Returns:
            A patito model representing the given row data.

        Raises:
            TypeError: If the given type is neither a pandas or polars DataFrame.
        """
        if isinstance(row, pl.DataFrame):
            dataframe = row
        elif _PANDAS_AVAILABLE and isinstance(row, pd.DataFrame):
            dataframe = pl.DataFrame._from_pandas(row)
        elif _PANDAS_AVAILABLE and isinstance(row, pd.Series):
            return cls(**dict(row.iteritems()))
        else:
            raise TypeError(f"{cls.__name__}.from_row not implemented for {type(row)}.")
        return cls.from_polars(dataframe=dataframe, validate=validate)

    @classmethod
    def from_polars(
        cls: Type[ModelType],
        dataframe: pl.DataFrame,
        validate: bool = True,
    ) -> ModelType:
        """
        Construct model from a single polars row.

        Args:
            dataframe: A polars dataframe consisting of one single row.
            validate: If True, run the pydantic validators. If False, pydantic will
                not cast any types in the resulting object.

        Returns:
            A pydantic model object representing the given polars row.

        Raises:
            TypeError: If the provided `dataframe` argument is not of type
                polars.DataFrame.
            ValueError: If the given `dataframe` argument does not consist of exactly
                one row.
        """
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
    def validate(  # type: ignore
        cls,
        dataframe: Union["pd.DataFrame", pl.DataFrame],
    ) -> None:
        """
        Validate the given dataframe.

        Args:
            dataframe: Polars DataFrame to be validated.

        Raises:
            patito.exceptions.ValidationError:  # noqa: DAR402
                If the given dataframe does not match the given schema.
        """
        validate(dataframe=dataframe, schema=cls)

    @classmethod
    def example_value(  # noqa: C901
        cls,
        field: str,
    ) -> Union[date, datetime, float, int, str, None]:
        """
        Return an example value for the given field name defined on the model.

        Args:
            field: Field name identifier.

        Returns:
            A single value which is consistent with the given field definition.

        Raises:
            NotImplementedError: If the given field has no example generator.
        """
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
            if "dtype" in properties:
                dtype = properties["dtype"]
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
                return date(year=1970, month=1, day=1)
            elif "format" in properties and properties["format"] == "date-time":
                return datetime(year=1970, month=1, day=1)
            elif "minLength" in properties:
                return "a" * properties["minLength"]
            elif "maxLength" in properties:
                return "a" * min(properties["maxLength"], 1)
            else:
                return "dummy_string"

        elif field_type == "boolean":
            return False

        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def example(
        cls: Type[ModelType],
        **kwargs: Any,  # noqa: ANN401
    ) -> ModelType:
        """
        Produce model with dummy data for all unspecified fields.

        The type annotation of unspecified field is used to fill in type-correct
        dummy data, e.g. -1 for int, "dummy_string" for str, and so on...
        The first item of Literal annotatations are used for dummy values.

        Args:
            **kwargs: Provide explicit values for any fields which should not be filled
                with dummy data.

        Returns:
            A pydantic model object filled with dummy data for all unspecified model
                fields.

        Raises:
            TypeError: If one or more of the provided keyword arguments do not match any
                fields on the model.
        """
        # Non-iterable values besides strings must be repeated
        wrong_columns = set(kwargs.keys()) - set(cls.columns)
        if wrong_columns:
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
    def pandas_examples(
        cls: Type[ModelType],
        data: Union[dict, Iterable],
        columns: Optional[Iterable[str]] = None,
    ) -> "pd.DataFrame":
        """
        Generate dataframe with dummy data for all unspecified columns.

        Offers the same API as the pandas.DataFrame constructor.
        Non-iterable values, besides strings, are repeated until they become as long as
        the iterable arguments.

        Args:
            data: Data to populate the dummy dataframe with. If
                not a dict, column names must also be provided.
            columns: Ignored if data is a dict. If
                data is an iterable, it will be used as the column names in the
                resulting dataframe. Defaults to None.

        Returns:
            A pandas DataFrame filled with dummy example data.

        Raises:
            ImportError: If pandas has not been installed. You should install
                patito[pandas] in order to integrate patito with pandas.
            TypeError: If column names have not been specified in the input data.
        """
        if not _PANDAS_AVAILABLE:
            # Re-trigger the import error, but this time don't catch it
            raise ImportError("No module named 'pandas'")

        if not isinstance(data, dict):
            if columns is None:
                raise TypeError(
                    f"{cls.__name__}.pandas_examples() must "
                    "be provided with column names!"
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
            dummies.append(cls.example(**dict(zip(kwargs.keys(), values))))
        return pd.DataFrame([dummy.dict() for dummy in dummies])

    @classmethod
    def examples(
        cls: Type[ModelType],
        data: Optional[Union[dict, Iterable]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pl.DataFrame:
        """
        Generate polars dataframe with dummy data for all unspecified columns.

        This constructor accepts the same data format as polars.DataFrame.

        Args:
            data: Data to populate the dummy dataframe with. If given as an iterable of
                values then column names must also be provided. If not provided at all,
                an empty dataframe with the correct column dtypes will be generated
                instead.
            columns: Ignored if data is a dict. If data is an iterable, it will be used
                as the column names in the resulting dataframe. Defaults to None.

        Returns:
            A polars dataframe where all unspecified columns have been filled with dummy
                data which should pass model validation.

        Raises:
            TypeError: If one or more of the model fields are not mappable to polars
                column dtype equivalents.
        """
        if data is None:
            # We should create an empty dataframe, but with the correct dtypes
            kwargs = {}
        elif not isinstance(data, dict):
            if columns is None:
                raise TypeError(
                    f"{cls.__name__}.examples() must be provided with column names!"
                )
            kwargs = dict(zip(columns, zip(*data)))
        else:
            kwargs = data

        wrong_columns = set(kwargs.keys()) - set(cls.columns)
        if wrong_columns:
            raise TypeError(f"{cls.__name__} does not contain fields {wrong_columns}!")

        series: List[Union[pl.Series, pl.Expr]] = []
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

        return (
            pl.DataFrame()
            .with_columns(series)  # type: ignore
            .with_columns(unique_series)
        )

    class Config(BaseConfig):
        """Configuration for Pydantic BaseModel behaviour."""

        # Make fields immutable and model hashable
        frozen = True
