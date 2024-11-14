"""Logic related to wrapping logic around the pydantic library."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from inspect import getfullargspec
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Optional,
    TypeVar,
    cast,
    get_args,
)
from zoneinfo import ZoneInfo

import polars as pl
from polars.datatypes import DataType, DataTypeClass
from pydantic import (  # noqa: F401
    BaseModel,
    create_model,
    fields,
)
from pydantic._internal._model_construction import (
    ModelMetaclass as PydanticModelMetaclass,
)

from patito._pydantic.column_info import ColumnInfo
from patito._pydantic.dtypes import (
    default_dtypes_for_model,
    is_optional,
    valid_dtypes_for_model,
    validate_annotation,
    validate_polars_dtype,
)
from patito._pydantic.schema import column_infos_for_model, schema_for_model
from patito.polars import DataFrame, LazyFrame, ModelGenerator
from patito.validators import validate

try:
    import pandas as pd  # type: ignore

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    import patito.polars

# The generic type of a single row in given Relation.
# Should be a typed subclass of Model.
ModelType = TypeVar("ModelType", bound="Model")


class ModelMetaclass(PydanticModelMetaclass):
    """Metaclass used by patito.Model.

    Responsible for setting any relevant model-dependent class properties.
    """

    if TYPE_CHECKING:
        model_fields: ClassVar[dict[str, fields.FieldInfo]]

    def __init__(cls, name: str, bases: tuple, clsdict: dict, **kwargs) -> None:
        """Construct new patito model.

        Args:
            name: Name of model class.
            bases: Tuple of superclasses.
            clsdict: Dictionary containing class properties.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(name, bases, clsdict, **kwargs)
        NewDataFrame = type(
            f"{cls.__name__}DataFrame",
            (DataFrame,),
            {"model": cls},
        )
        cls.DataFrame: type[DataFrame[cls]] = NewDataFrame  # type: ignore

        NewLazyFrame = type(
            f"{cls.__name__}LazyFrame",
            (LazyFrame,),
            {"model": cls},
        )
        cls.LazyFrame: type[LazyFrame[cls]] = NewLazyFrame  # type: ignore

    def __hash__(self) -> int:
        """Return hash of the model class."""
        return super().__hash__()

    @property
    def column_infos(cls: type[Model]) -> Mapping[str, ColumnInfo]:
        """Return column information for the model."""
        return column_infos_for_model(cls)

    @property
    def model_schema(cls: type[Model]) -> Mapping[str, Mapping[str, Any]]:
        """Return schema properties where definition references have been resolved.

        Returns:
            Field information as a dictionary where the keys are field names and the
                values are dictionaries containing metadata information about the field
                itself.

        Raises:
            TypeError: if a field is annotated with an enum where the values are of
                different types.

        """
        return schema_for_model(cls)

    @property
    def columns(cls: type[Model]) -> list[str]:
        """Return the name of the dataframe columns specified by the fields of the model.

        Returns:
            List of column names.

        Example:
            >>> import patito as pt
            >>> class Product(pt.Model):
            ...     name: str
            ...     price: int
            ...
            >>> Product.columns
            ['name', 'price']

        """
        return list(cls.model_fields.keys())

    @property
    def dtypes(cls: type[Model]) -> dict[str, DataTypeClass | DataType]:
        """Return the polars dtypes of the dataframe.

        Unless Field(dtype=...) is specified, the highest signed column dtype
        is chosen for integer and float columns.

        Returns:
            A dictionary mapping string column names to polars dtype classes.

        Example:
            >>> import patito as pt
            >>> class Product(pt.Model):
            ...     name: str
            ...     ideal_temperature: int
            ...     price: float
            ...
            >>> Product.dtypes
            {'name': String, 'ideal_temperature': Int64, 'price': Float64}

        """
        return default_dtypes_for_model(cls)

    @property
    def valid_dtypes(
        cls: type[Model],
    ) -> Mapping[str, frozenset[DataTypeClass | DataType]]:
        """Return a list of polars dtypes which Patito considers valid for each field.

        The first item of each list is the default dtype chosen by Patito.

        Returns:
            A dictionary mapping each column string name to a list of valid dtypes.

        Raises:
            NotImplementedError: If one or more model fields are annotated with types
                not compatible with polars.

        """
        return valid_dtypes_for_model(cls)

    @property
    def defaults(cls: type[Model]) -> dict[str, Any]:
        """Return default field values specified on the model.

        Returns:
            Dictionary containing fields with their respective default values.

        Example:
            >>> from typing_extensions import Literal
            >>> import patito as pt
            >>> class Product(pt.Model):
            ...     name: str
            ...     price: int = 0
            ...     temperature_zone: Literal["dry", "cold", "frozen"] = "dry"
            ...
            >>> Product.defaults
            {'price': 0, 'temperature_zone': 'dry'}

        """
        return {
            field_name: props["default"]
            for field_name, props in cls._schema_properties().items()
            if "default" in props
        }

    @property
    def non_nullable_columns(cls: type[Model]) -> set[str]:
        """Return names of those columns that are non-nullable in the schema.

        Returns:
            Set of column name strings.

        Example:
            >>> from typing import Optional
            >>> import patito as pt
            >>> class MyModel(pt.Model):
            ...     nullable_field: Optional[int]
            ...     another_nullable_field: Optional[int] = None
            ...     non_nullable_field: int
            ...     another_non_nullable_field: str
            ...
            >>> sorted(MyModel.non_nullable_columns)
            ['another_non_nullable_field', 'non_nullable_field']

        """
        return set(
            k
            for k in cls.columns
            if not (
                is_optional(cls.model_fields[k].annotation)
                or cls.model_fields[k].annotation is type(None)
            )
        )

    @property
    def nullable_columns(cls: type[Model]) -> set[str]:
        """Return names of those columns that are nullable in the schema.

        Returns:
            Set of column name strings.

        Example:
            >>> from typing import Optional
            >>> import patito as pt
            >>> class MyModel(pt.Model):
            ...     nullable_field: Optional[int]
            ...     another_nullable_field: Optional[int] = None
            ...     non_nullable_field: int
            ...     another_non_nullable_field: str
            ...
            >>> sorted(MyModel.nullable_columns)
            ['another_nullable_field', 'nullable_field']

        """
        return set(cls.columns) - cls.non_nullable_columns

    @property
    def unique_columns(cls: type[Model]) -> set[str]:
        """Return columns with uniqueness constraint.

        Returns:
            Set of column name strings.

        Example:
            >>> from typing import Optional
            >>> import patito as pt

            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     barcode: Optional[str] = pt.Field(unique=True)
            ...     name: str
            ...
            >>> sorted(Product.unique_columns)
            ['barcode', 'product_id']

        """
        infos = cls.column_infos
        return {column for column in cls.columns if infos[column].unique}

    @property
    def derived_columns(cls: type[Model]) -> set[str]:
        """Return set of columns which are derived from other columns."""
        infos = cls.column_infos
        return {
            column for column in cls.columns if infos[column].derived_from is not None
        }


class Model(BaseModel, metaclass=ModelMetaclass):
    """Custom pydantic class for representing table schema and constructing rows."""

    @classmethod
    def validate_schema(cls: type[ModelType]):
        """Users should run this after defining or edit a model. We withhold the checks at model definition time to avoid expensive queries of the model schema."""
        for column in cls.columns:
            col_info = cls.column_infos[column]
            field_info = cls.model_fields[column]
            if col_info.dtype:
                validate_polars_dtype(
                    annotation=field_info.annotation, dtype=col_info.dtype
                )
            else:
                validate_annotation(field_info.annotation)

    @classmethod
    def from_row(
        cls: type[ModelType],
        row: pd.DataFrame | pl.DataFrame,
        validate: bool = True,
    ) -> ModelType:
        """Represent a single data frame row as a Patito model.

        Args:
            row: A dataframe, either polars and pandas, consisting of a single row.
            validate: If ``False``, skip pydantic validation of the given row data.

        Returns:
            Model: A patito model representing the given row data.

        Raises:
            TypeError: If the given type is neither a pandas or polars DataFrame.

        Example:
            >>> import patito as pt
            >>> import polars as pl

            >>> class Product(pt.Model):
            ...     product_id: int
            ...     name: str
            ...     price: float
            ...

            >>> df = pl.DataFrame(
            ...     [["1", "product name", "1.22"]],
            ...     schema=["product_id", "name", "price"],
            ...     orient="row",
            ... )
            >>> Product.from_row(df)
            Product(product_id=1, name='product name', price=1.22)
            >>> Product.from_row(df, validate=False)
            Product(product_id='1', name='product name', price='1.22')

        """
        if isinstance(row, pl.DataFrame):
            dataframe = row
        elif _PANDAS_AVAILABLE and isinstance(row, pd.DataFrame):
            dataframe = pl.DataFrame._from_pandas(row)
        elif _PANDAS_AVAILABLE and isinstance(row, pd.Series):
            return cls(**dict(row.items()))  # type: ignore[unreachable]
        else:
            raise TypeError(f"{cls.__name__}.from_row not implemented for {type(row)}.")
        return cls._from_polars(dataframe=dataframe, validate=validate)

    @classmethod
    def _from_polars(
        cls: type[ModelType],
        dataframe: pl.DataFrame,
        validate: bool = True,
    ) -> ModelType:
        """Construct model from a single polars row.

        Args:
            dataframe: A polars dataframe consisting of one single row.
            validate: If ``True``, run the pydantic validators. If ``False``, pydantic
                will not cast any types in the resulting object.

        Returns:
            Model: A pydantic model object representing the given polars row.

        Raises:
            TypeError: If the provided ``dataframe`` argument is not of type
                ``polars.DataFrame``.
            ValueError: If the given ``dataframe`` argument does not consist of exactly
                one row.

        Example:
            >>> import patito as pt
            >>> import polars as pl

            >>> class Product(pt.Model):
            ...     product_id: int
            ...     name: str
            ...     price: float
            ...

            >>> df = pl.DataFrame(
            ...     [["1", "product name", "1.22"]],
            ...     schema=["product_id", "name", "price"],
            ...     orient="row",
            ... )
            >>> Product._from_polars(df)
            Product(product_id=1, name='product name', price=1.22)
            >>> Product._from_polars(df, validate=False)
            Product(product_id='1', name='product name', price='1.22')

        """
        if not isinstance(dataframe, pl.DataFrame):
            raise TypeError(
                f"{cls.__name__}._from_polars() must be invoked with polars.DataFrame, "
                f"not {type(dataframe)}!"
            )
        elif len(dataframe) != 1:
            raise ValueError(
                f"{cls.__name__}._from_polars() can only be invoked with exactly "
                f"1 row, while {len(dataframe)} rows were provided."
            )

        # We have been provided with a single polars.DataFrame row
        # Convert to the equivalent keyword invocation of the pydantic model
        if validate:
            return cls(**dataframe.to_dicts()[0])
        else:
            return cls.model_construct(**dataframe.to_dicts()[0])

    @classmethod
    def validate(
        cls: type[ModelType],
        dataframe: pd.DataFrame | pl.DataFrame,
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> DataFrame[ModelType]:
        """Validate the schema and content of the given dataframe.

        Args:
            dataframe: Polars DataFrame to be validated.
            columns: Optional list of columns to validate. If not provided, all columns
                of the dataframe will be validated.
            allow_missing_columns: If True, missing columns will not be considered an error.
            allow_superfluous_columns: If True, additional columns will not be considered an error.
            drop_superfluous_columns: If True, columns not present in the model will be
                dropped from the resulting dataframe.

        Returns:
            DataFrame: A patito DataFrame containing the validated data.

        Raises:
            patito.exceptions.DataFrameValidationError: If the given dataframe does not match
                the given schema.

        Examples:
            >>> import patito as pt
            >>> import polars as pl

            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     temperature_zone: Literal["dry", "cold", "frozen"]
            ...     is_for_sale: bool
            ...

            >>> df = pl.DataFrame(
            ...     {
            ...         "product_id": [1, 1, 3],
            ...         "temperature_zone": ["dry", "dry", "oven"],
            ...     }
            ... )
            >>> try:
            ...     Product.validate(df)
            ... except pt.DataFrameValidationError as exc:
            ...     print(exc)
            ...
            3 validation errors for Product
            is_for_sale
              Missing column (type=type_error.missingcolumns)
            product_id
              2 rows with duplicated values. (type=value_error.rowvalue)
            temperature_zone
              Rows with invalid values: {'oven'}. (type=value_error.rowvalue)

        """
        validate(
            dataframe=dataframe,
            schema=cls,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )
        return cls.DataFrame(dataframe)

    @classmethod
    def iter_models(
        cls: type[ModelType], dataframe: pd.DataFrame | pl.DataFrame
    ) -> ModelGenerator[ModelType]:
        """Validate the dataframe and iterate over the rows, yielding Patito models.

        Args:
            dataframe: Polars or pandas DataFrame to be validated.

        Returns:
            ListableIterator: An iterator of patito models over the validated data.

        Raises:
            patito.exceptions.DataFrameValidationError: If the given dataframe does not match
                the given schema.

        """
        return cls.DataFrame(dataframe).iter_models()

    @classmethod
    def example_value(  # noqa: C901
        cls,
        field: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> date | datetime | time | timedelta | float | int | str | None | Mapping | list:
        """Return a valid example value for the given model field.

        Args:
            field: Field name identifier.
            properties: Pydantic v2-style properties dict

        Returns:
            A single value which is consistent with the given field definition.

        Raises:
            NotImplementedError: If the given field has no example generator.

        Example:
            >>> from typing import Literal
            >>> import patito as pt

            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     name: str
            ...     temperature_zone: Literal["dry", "cold", "frozen"]
            ...
            >>> Product.example_value("product_id")
            -1
            >>> Product.example_value("name")
            'dummy_string'
            >>> Product.example_value("temperature_zone")
            'dry'

        """
        if field is None and properties is None:
            raise ValueError(
                "Either 'field' or 'properties' must be provided as argument."
            )
        if field is not None and properties is not None:
            raise ValueError(
                "Only one of 'field' or 'properties' can be provided as argument."
            )
        if field:
            properties = cls._schema_properties()[field]
            info = cls.column_infos[field]
        else:
            info = ColumnInfo()
        properties = properties or {}

        if "type" in properties:
            field_type = properties["type"]
        elif "anyOf" in properties:
            allowable = [x["type"] for x in properties["anyOf"] if "type" in x]
            if "null" in allowable:
                field_type = "null"
            else:
                field_type = allowable[0]
        else:
            raise NotImplementedError(
                f"Field type for {properties['title']} not found."
            )

        if "const" in properties:
            # The default value is the only valid value, provided as const
            return properties["const"]

        elif "default" in properties:
            # A default value has been specified in the model field definition
            return properties["default"]

        elif not properties.get("required", True):
            return None

        elif field_type == "null":
            return None

        elif "enum" in properties:
            return properties["enum"][0]

        elif field_type in {"integer", "number"}:
            # For integer and float types we must check if there are imposed bounds

            minimum = properties.get("minimum")
            exclusive_minimum = properties.get("exclusiveMinimum")
            maximum = properties.get("maximum")
            exclusive_maximum = properties.get("exclusiveMaximum")

            lower = minimum if minimum is not None else exclusive_minimum
            upper = maximum if maximum is not None else exclusive_maximum

            # If the dtype is an unsigned integer type, we must return a positive value
            if info.dtype:
                dtype = info.dtype
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
                return number(cast(float, upper) - 1)

        elif field_type == "string":
            if "pattern" in properties:
                raise NotImplementedError(
                    "Example data generation has not been implemented for regex "
                    "patterns. You must valid data for such columns explicitly!"
                )
            elif "format" in properties and properties["format"] == "date":
                return date(year=1970, month=1, day=1)
            elif "format" in properties and properties["format"] == "date-time":
                if "column_info" in properties:
                    ci = ColumnInfo.model_validate_json(properties["column_info"])
                    dtype = ci.dtype
                    if getattr(dtype, "time_zone", None) is not None:
                        tzinfo = ZoneInfo(dtype.time_zone)  # type: ignore
                    else:
                        tzinfo = None
                    return datetime(year=1970, month=1, day=1, tzinfo=tzinfo)
                return datetime(year=1970, month=1, day=1)
            elif "format" in properties and properties["format"] == "time":
                return time(12, 30)
            elif "format" in properties and properties["format"] == "duration":
                return timedelta(1)
            elif "minLength" in properties:
                return "a" * properties["minLength"]
            elif "maxLength" in properties:
                return "a" * min(properties["maxLength"], 1)
            else:
                return "dummy_string"

        elif field_type == "boolean":
            return False

        elif field_type == "object":
            try:
                props_o = cls.model_schema["$defs"][properties["title"]]["properties"]
                return {f: cls.example_value(properties=props_o[f]) for f in props_o}
            except AttributeError as err:
                raise NotImplementedError(
                    "Nested example generation only supported for nested pt.Model classes."
                ) from err

        elif field_type == "array":
            return [cls.example_value(properties=properties["items"])]

        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def example(
        cls: type[ModelType],
        **kwargs: Any,  # noqa: ANN401
    ) -> ModelType:
        """Produce model instance with filled dummy data for all unspecified fields.

        The type annotation of unspecified field is used to fill in type-correct
        dummy data, e.g. ``-1`` for ``int``, ``"dummy_string"`` for ``str``, and so
        on...

        The first item of ``typing.Literal`` annotations are used for dummy values.

        Args:
            **kwargs: Provide explicit values for any fields which should `not` be
                filled with dummy data.

        Returns:
            Model: A pydantic model object filled with dummy data for all unspecified
            model fields.

        Raises:
            TypeError: If one or more of the provided keyword arguments do not match any
                fields on the model.

        Example:
            >>> from typing import Literal
            >>> import patito as pt

            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     name: str
            ...     temperature_zone: Literal["dry", "cold", "frozen"]
            ...
            >>> Product.example(product_id=1)
            Product(product_id=1, name='dummy_string', temperature_zone='dry')

        """
        # Non-iterable values besides strings must be repeated
        wrong_columns = set(kwargs.keys()) - set(cls.columns)
        if wrong_columns:
            raise TypeError(f"{cls.__name__} does not contain fields {wrong_columns}!")

        new_kwargs = {}
        for field_name in cls._schema_properties().keys():
            if field_name in kwargs:
                # The value has been explicitly specified
                new_kwargs[field_name] = kwargs[field_name]
            else:
                new_kwargs[field_name] = cls.example_value(field=field_name)
        return cls(**new_kwargs)

    @classmethod
    def pandas_examples(
        cls: type[ModelType],
        data: dict | Iterable,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Generate dataframe with dummy data for all unspecified columns.

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

        Example:
            >>> from typing import Literal
            >>> import patito as pt

            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     name: str
            ...     temperature_zone: Literal["dry", "cold", "frozen"]
            ...

            >>> Product.pandas_examples({"name": ["product A", "product B"]})
            product_id       name temperature_zone
            0          -1  product A              dry
            1          -1  product B              dry

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
        return pd.DataFrame([dummy.model_dump() for dummy in dummies])

    @classmethod
    def examples(
        cls: type[ModelType],
        data: dict | Iterable | None = None,
        columns: Iterable[str] | None = None,
    ) -> patito.polars.DataFrame:
        """Generate polars dataframe with dummy data for all unspecified columns.

        This constructor accepts the same data format as polars.DataFrame.

        Args:
            data: Data to populate the dummy dataframe with. If given as an iterable of
                values, then column names must also be provided. If not provided at all,
                a dataframe with a single row populated with dummy data is provided.
            columns: Ignored if ``data`` is provided as a dictionary. If data is
                provided as an ``iterable``, then ``columns`` will be used as the
                column names in the resulting dataframe. Defaults to None.

        Returns:
            A polars dataframe where all unspecified columns have been filled with dummy
            data which should pass model validation.

        Raises:
            TypeError: If one or more of the model fields are not mappable to polars
                column dtype equivalents.

        Example:
            >>> from typing import Literal
            >>> import patito as pt

            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     name: str
            ...     temperature_zone: Literal["dry", "cold", "frozen"]
            ...

            >>> Product.examples()
            shape: (1, 3)
            ┌──────────────┬──────────────────┬────────────┐
            │ name         ┆ temperature_zone ┆ product_id │
            │ ---          ┆ ---              ┆ ---        │
            │ str          ┆ enum             ┆ i64        │
            ╞══════════════╪══════════════════╪════════════╡
            │ dummy_string ┆ dry              ┆ 1          │
            └──────────────┴──────────────────┴────────────┘

            >>> Product.examples({"name": ["product A", "product B"]})
            shape: (2, 3)
            ┌───────────┬──────────────────┬────────────┐
            │ name      ┆ temperature_zone ┆ product_id │
            │ ---       ┆ ---              ┆ ---        │
            │ str       ┆ enum             ┆ i64        │
            ╞═══════════╪══════════════════╪════════════╡
            │ product A ┆ dry              ┆ 1          │
            │ product B ┆ dry              ┆ 2          │
            └───────────┴──────────────────┴────────────┘

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

        series: list[pl.Series | pl.Expr] = []
        unique_series = []
        for column_name, dtype in cls.dtypes.items():
            if column_name not in kwargs:
                if column_name in cls.unique_columns:
                    unique_series.append(
                        pl.first().cum_count().cast(dtype).alias(column_name)
                    )
                else:
                    example_value = cls.example_value(field=column_name)
                    series.append(
                        pl.Series(column_name, values=[example_value], dtype=dtype)
                    )
                continue

            value = kwargs.get(column_name)
            if isinstance(value, Iterable) and not isinstance(value, str):
                # We make sure that at least one series is inserted first in the list,
                # otherwise polars will not be able to handle the shape mismatch between
                # series and literate values.
                series.insert(0, pl.Series(name=column_name, values=value, dtype=dtype))
            else:
                series.append(pl.lit(value, dtype=dtype).alias(column_name))

        return cls.DataFrame().with_columns(series).with_columns(unique_series)

    @classmethod
    def join(
        cls: type[Model],
        other: type[Model],
        how: Literal["inner", "left", "outer", "asof", "cross", "semi", "anti"],
    ) -> type[Model]:
        """Dynamically create a new model compatible with an SQL Join operation.

        For instance, ``ModelA.join(ModelB, how="left")`` will create a model containing
        all the fields of ``ModelA`` and ``ModelB``, but where all fields of ``ModelB``
        has been made ``Optional``, i.e. nullable. This is consistent with the LEFT JOIN
        SQL operation making all the columns of the right table nullable.

        Args:
            other: Another patito Model class.
            how: The type of SQL Join operation.

        Returns:
            A new model type compatible with the resulting schema produced by the given
            join operation.

        Examples:
            >>> class A(Model):
            ...     a: int
            ...
            >>> class B(Model):
            ...     b: int
            ...

            >>> InnerJoinedModel = A.join(B, how="inner")
            >>> InnerJoinedModel.columns
            ['a', 'b']
            >>> InnerJoinedModel.nullable_columns
            set()

            >>> LeftJoinedModel = A.join(B, how="left")
            >>> LeftJoinedModel.nullable_columns
            {'b'}

            >>> OuterJoinedModel = A.join(B, how="outer")
            >>> sorted(OuterJoinedModel.nullable_columns)
            ['a', 'b']

            >>> A.join(B, how="anti") is A
            True

        """
        if how in {"semi", "anti"}:
            return cls

        kwargs: dict[str, Any] = {}
        for model, nullable_methods in (
            (cls, {"outer"}),
            (other, {"left", "outer", "asof"}),
        ):
            for field_name, field in model.model_fields.items():
                make_nullable = how in nullable_methods and type(None) not in get_args(
                    field.annotation
                )
                kwargs[field_name] = cls._derive_field(
                    field, make_nullable=make_nullable
                )

        return create_model(
            f"{cls.__name__}{how.capitalize()}Join{other.__name__}",
            **kwargs,
            __base__=Model,
        )

    @classmethod
    def select(cls: type[ModelType], fields: str | Iterable[str]) -> type[Model]:
        """Create a new model consisting of only a subset of the model fields.

        Args:
            fields: A single field name as a string or a collection of strings.

        Returns:
            A new model containing only the fields specified by ``fields``.

        Raises:
            ValueError: If one or more non-existent fields are selected.

        Example:
            >>> class MyModel(Model):
            ...     a: int
            ...     b: int
            ...     c: int
            ...

            >>> MyModel.select("a").columns
            ['a']

            >>> sorted(MyModel.select(["b", "c"]).columns)
            ['b', 'c']

        """
        if isinstance(fields, str):
            fields = [fields]

        fields = set(fields)
        non_existent_fields = fields - set(cls.columns)
        if non_existent_fields:
            raise ValueError(
                f"The following selected fields do not exist: {non_existent_fields}"
            )

        mapping = {field_name: field_name for field_name in fields}
        return cls._derive_model(
            model_name=f"Selected{cls.__name__}", field_mapping=mapping
        )

    @classmethod
    def drop(cls: type[ModelType], name: str | Iterable[str]) -> type[Model]:
        """Return a new model where one or more fields are excluded.

        Args:
            name: A single string field name, or a list of such field names,
                which will be dropped.

        Returns:
            New model class where the given fields have been removed.

        Examples:
            >>> class MyModel(Model):
            ...     a: int
            ...     b: int
            ...     c: int
            ...

            >>> MyModel.columns
            ['a', 'b', 'c']

            >>> MyModel.drop("c").columns
            ['a', 'b']

            >>> MyModel.drop(["b", "c"]).columns
            ['a']

        """
        dropped_columns = {name} if isinstance(name, str) else set(name)
        mapping = {
            field_name: field_name
            for field_name in cls.columns
            if field_name not in dropped_columns
        }
        return cls._derive_model(
            model_name=f"Dropped{cls.__name__}",
            field_mapping=mapping,
        )

    @classmethod
    def prefix(cls: type[ModelType], prefix: str) -> type[Model]:
        """Return a new model where all field names have been prefixed.

        Args:
            prefix: String prefix to add to all field names.

        Returns:
            New model class with all the same fields only prefixed with the given prefix.

        Example:
            >>> class MyModel(Model):
            ...     a: int
            ...     b: int
            ...

            >>> MyModel.prefix("x_").columns
            ['x_a', 'x_b']

        """
        mapping = {f"{prefix}{field_name}": field_name for field_name in cls.columns}
        return cls._derive_model(
            model_name="Prefixed{cls.__name__}",
            field_mapping=mapping,
        )

    @classmethod
    def suffix(cls: type[ModelType], suffix: str) -> type[Model]:
        """Return a new model where all field names have been suffixed.

        Args:
            suffix: String suffix to add to all field names.

        Returns:
            New model class with all the same fields only suffixed with the given
            suffix.

        Example:
            >>> class MyModel(Model):
            ...     a: int
            ...     b: int
            ...

            >>> MyModel.suffix("_x").columns
            ['a_x', 'b_x']

        """
        mapping = {f"{field_name}{suffix}": field_name for field_name in cls.columns}
        return cls._derive_model(
            model_name="Suffixed{cls.__name__}",
            field_mapping=mapping,
        )

    @classmethod
    def rename(cls: type[ModelType], mapping: dict[str, str]) -> type[Model]:
        """Return a new model class where the specified fields have been renamed.

        Args:
            mapping: A dictionary where the keys are the old field names
                and the values are the new names.

        Returns:
            A new model class where the given fields have been renamed.

        Raises:
            ValueError: If non-existent fields are renamed.

        Example:
            >>> class MyModel(Model):
            ...     a: int
            ...     b: int
            ...

            >>> MyModel.rename({"a": "A"}).columns
            ['b', 'A']

        """
        non_existent_fields = set(mapping.keys()) - set(cls.columns)
        if non_existent_fields:
            raise ValueError(
                f"The following fields do not exist for renaming: {non_existent_fields}"
            )
        field_mapping = {
            field_name: field_name
            for field_name in cls.columns
            if field_name not in mapping
        }
        field_mapping.update({value: key for key, value in mapping.items()})
        return cls._derive_model(
            model_name=f"Renamed{cls.__name__}",
            field_mapping=field_mapping,
        )

    @classmethod
    def with_fields(
        cls: type[ModelType],
        **field_definitions: Any,  # noqa: ANN401
    ) -> type[Model]:
        """Return a new model class where the given fields have been added.

        Args:
            **field_definitions: the keywords are of the form:
                ``field_name=(field_type, field_default)``.
                Specify ``...`` if no default value is provided.
                For instance, ``column_name=(int, ...)`` will create a new non-optional
                integer field named ``"column_name"``.

        Returns:
            A new model with all the original fields and the additional field
            definitions.

        Example:
            >>> class MyModel(Model):
            ...     a: int
            ...
            >>> class ExpandedModel(MyModel):
            ...     b: int
            ...
            >>> MyModel.with_fields(b=(int, ...)).columns == ExpandedModel.columns
            True

        """
        fields = {field_name: field_name for field_name in cls.columns}
        fields.update(field_definitions)
        return cls._derive_model(
            model_name=f"Expanded{cls.__name__}",
            field_mapping=fields,
        )

    @classmethod
    def _schema_properties(cls: type[ModelType]) -> Mapping[str, Any]:
        return cls.model_schema["properties"]

    @classmethod
    def _update_dfn(cls, annotation: Any, schema: dict[str, Any]) -> None:
        try:
            if issubclass(annotation, Model) and annotation.__name__ != cls.__name__:
                schema["$defs"][annotation.__name__] = annotation.model_schema
        except TypeError:
            pass

    @classmethod
    def _derive_model(
        cls: type[ModelType],
        model_name: str,
        field_mapping: dict[str, Any],
    ) -> type[Model]:
        """Derive a new model with new field definitions.

        Args:
            model_name: Name of new model class.
            field_mapping: A mapping where the keys represent field names and the values
                represent field definitions. String field definitions are used as
                pointers to the original fields by name. Otherwise, specify field
                definitions as (field_type, field_default) as accepted by
                pydantic.create_model.

        Returns:
            A new model class derived from the model type of self.

        """
        new_fields = {}
        for new_field_name, field_definition in field_mapping.items():
            if isinstance(field_definition, str):
                # A single string, interpreted as the name of a field on the existing
                # model.
                old_field = cls.model_fields[field_definition]
                new_fields[new_field_name] = cls._derive_field(old_field)
            else:
                # We have been given a (field_type, field_default) tuple defining the
                # new field directly.
                field_type = field_definition[0]
                if field_definition[1] is None and type(None) not in get_args(
                    field_type
                ):
                    field_type = Optional[field_type]
                new_fields[new_field_name] = (field_type, field_definition[1])
        return create_model(  # type: ignore
            model_name,
            __base__=Model,
            **new_fields,
        )

    @staticmethod
    def _derive_field(
        field: fields.FieldInfo,
        make_nullable: bool = False,
    ) -> tuple[type | None, fields.FieldInfo]:
        field_type = field.annotation
        default = field.default
        extra_attrs = {
            x: getattr(field, x)
            for x in field._attributes_set
            if x in field.__slots__ and x not in ["annotation", "default"]
        }
        if make_nullable:
            if field_type is None:
                raise TypeError(
                    "Cannot make field nullable if no type annotation is provided!"
                )
            else:
                # This originally non-nullable field has become nullable
                field_type = Optional[field_type]
        elif field.is_required() and default is None:
            # We need to replace Pydantic's None default value with ... in order
            # to make it clear that the field is still non-nullable and
            # required.
            default = ...
        field_new = fields.Field(default=default, **extra_attrs)
        field_new.metadata = field.metadata
        return field_type, field_new


FIELD_KWARGS = getfullargspec(fields.Field)


# Helper function for patito Field.


def Field(
    *args: Any, **kwargs: Any
) -> Any:  # annotate with Any to make the downstream type annotations happy
    """Annotate model field with additional type and validation information.

    This class is built on ``pydantic.Field`` and you can find the list of parameters
    in the `API reference <https://docs.pydantic.dev/latest/api/fields/>`_.
    Patito adds additional parameters which are used when validating dataframes,
    these are documented here along with the main parameters which can be used for
    validation. Pydantic's `usage documentation <https://docs.pydantic.dev/latest/concepts/fields/>`_
    can be read with the below examples.

    Args:
        allow_missing (bool): Column may be missing.
        column_info: (Type[ColumnInfo]): ColumnInfo object to pass args to.
        constraints (Union[polars.Expression, List[polars.Expression]): A single
            constraint or list of constraints, expressed as a polars expression objects.
            All rows must satisfy the given constraint. You can refer to the given column
            with ``pt.field``, which will automatically be replaced with
            ``polars.col(<field_name>)`` before evaluation.
        derived_from (Union[str, polars.Expr]): used to mark fields that are meant to be
            derived from other fields. Users can specify a polars expression that will
            be called to derive the column value when `pt.DataFrame.derive` is called.
        dtype (polars.datatype.DataType): The given dataframe column must have the given
            polars dtype, for instance ``polars.UInt64`` or ``pl.Float32``.
        unique (bool): All row values must be unique.
        gt: All values must be greater than ``gt``.
        ge: All values must be greater than or equal to ``ge``.
        lt: All values must be less than ``lt``.
        le: All values must be less than or equal to ``lt``.
        multiple_of: All values must be multiples of the given value.
        const (bool): If set to ``True`` `all` values must be equal to the provided
            default value, the first argument provided to the ``Field`` constructor.
        regex (str): UTF-8 string column must match regex pattern for all row values.
        min_length (int): Minimum length of all string values in a UTF-8 column.
        max_length (int): Maximum length of all string values in a UTF-8 column.
        args (Any): additional arguments to pass to pydantic's field.
        kwargs (Any): additional keyword arguments to pass to pydantic's field.

    Return:
        `FieldInfo <https://docs.pydantic.dev/latest/api/fields/#pydantic.fields.FieldInfo>`_:
            Object used to represent additional constraints put upon the given field.

    Examples:
        >>> import patito as pt
        >>> import polars as pl
        >>> class Product(pt.Model):
        ...     # Do not allow duplicates
        ...     product_id: int = pt.Field(unique=True)
        ...
        ...     # Price must be stored as unsigned 16-bit integers
        ...     price: int = pt.Field(dtype=pl.UInt16)
        ...
        ...     # The product name should be from 3 to 128 characters long
        ...     name: str = pt.Field(min_length=3, max_length=128)
        ...
        ...
        >>> Product.DataFrame(
        ...     {
        ...         "product_id": [1, 1],
        ...         "price": [400, 600],
        ...     }
        ... ).validate()
        Traceback (most recent call last):
        patito.exceptions.DataFrameValidationError: 3 validation errors for Product
        name
            Missing column (type=type_error.missingcolumns)
        product_id
            2 rows with duplicated values. (type=value_error.rowvalue)
        price
            Polars dtype Int64 does not match model field type. (type=type_error.columndtype)

    """
    ci = ColumnInfo(**kwargs)
    for field in ci.model_fields_set:
        kwargs.pop(field)
    if kwargs.pop("modern_kwargs_only", True):
        for kwarg in kwargs:
            if kwarg not in FIELD_KWARGS.kwonlyargs and kwarg not in FIELD_KWARGS.args:
                raise ValueError(
                    f"unexpected kwarg {kwarg}={kwargs[kwarg]}.  Add modern_kwargs_only=False to ignore"
                )
    ci_json = ci.model_dump_json()
    existing_json_schema_extra = kwargs.pop("json_schema_extra", {})
    merged_json_schema_extra = {**existing_json_schema_extra, "column_info": ci_json}

    return fields.Field(
        *args,
        json_schema_extra=merged_json_schema_extra,
        **kwargs,
    )
