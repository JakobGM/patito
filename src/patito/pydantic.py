"""Logic related to wrapping logic around the pydantic library."""
from __future__ import annotations

import itertools
from collections.abc import Iterable
from datetime import date, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    Literal,
    get_args,
    Sequence,
    Tuple,
    Callable,
    Mapping,
)

import polars as pl
from polars.datatypes import PolarsDataType
from pydantic import fields
from pydantic import ConfigDict, BaseModel, create_model  # noqa: F401
from pydantic._internal._model_construction import (
    ModelMetaclass as PydanticModelMetaclass,
)

from patito.polars import DataFrame, LazyFrame
from patito.validators import validate
from patito._pydantic.repr import display_as_type

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    import patito.polars
    from patito.duckdb import DuckDBSQLType
    from pydantic.typing import CallableGenerator

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

PYTHON_TO_PYDANTIC_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}

PL_INTEGER_DTYPES = [
    pl.Int64,
    pl.Int32,
    pl.Int16,
    pl.Int8,
    pl.UInt64,
    pl.UInt32,
    pl.UInt16,
    pl.UInt8,
]


class classproperty:
    """Equivalent to @property, but works on a class (doesn't require an instance).

    https://github.com/pola-rs/polars/blob/8d29d3cebec713363db4ad5d782c74047e24314d/py-polars/polars/datatypes/classes.py#L25C12-L25C12
    """

    def __init__(self, method: Callable[..., Any] | None = None) -> None:
        self.fget = method

    def __get__(self, instance: Any, cls: type | None = None) -> Any:
        return self.fget(cls)  # type: ignore[misc]

    def getter(self, method: Callable[..., Any]) -> Any:  # noqa: D102
        self.fget = method
        return self


def contains_object(dtype: PolarsDataType) -> bool:
    try:
        inner = dtype.inner  # pyright: ignore
        if inner == pl.Object():
            return True
        else:
            return contains_object(inner)
    except AttributeError:
        return False


class ModelMetaclass(
    PydanticModelMetaclass
):  # keep around for typing on Model construction
    ...


class Model(BaseModel, metaclass=ModelMetaclass):
    """Custom pydantic class for representing table schema and constructing rows."""

    if TYPE_CHECKING:
        model_fields: ClassVar[Dict[str, FieldInfo]]

    model_config = ConfigDict(
        ignored_types=(classproperty,),
    )

    @classproperty
    def columns(cls: Type[ModelType]) -> List[str]:  # type: ignore
        """
        Return the name of the dataframe columns specified by the fields of the model.

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
        return list(cls.model_json_schema()["properties"].keys())

    @classproperty
    def dtypes(  # type: ignore
        cls: Type[ModelType],  # pyright: ignore
    ) -> dict[str, PolarsDataType]:
        """
        Return the polars dtypes of the dataframe.

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
            {'name': Utf8, 'ideal_temperature': Int64, 'price': Float64}
        """
        return {
            column: valid_dtypes[0] for column, valid_dtypes in cls.valid_dtypes.items()
        }

    @classproperty
    def valid_dtypes(  # type: ignore
        cls: Type[ModelType],  # pyright: ignore
    ) -> dict[str, List[Union[PolarsDataType, pl.List]]]:
        """
        Return a list of polars dtypes which Patito considers valid for each field.

        The first item of each list is the default dtype chosen by Patito.

        Returns:
            A dictionary mapping each column string name to a list of valid dtypes.

        Raises:
            NotImplementedError: If one or more model fields are annotated with types
                not compatible with polars.

        Example:
            >>> from pprint import pprint
            >>> import patito as pt

            >>> class MyModel(pt.Model):
            ...     bool_column: bool
            ...     str_column: str
            ...     int_column: int
            ...     float_column: float
            ...
            >>> pprint(MyModel.valid_dtypes)
            {'bool_column': [Boolean],
             'float_column': [Float64, Float32],
             'int_column': [Int64, Int32, Int16, Int8, UInt64, UInt32, UInt16, UInt8],
             'str_column': [Utf8]}
        """
        valid_dtypes = {}
        for column, props in cls._schema_properties().items():
            column_dtypes: List[Union[PolarsDataType, pl.List]]
            column_dtypes = cls._valid_dtypes(column, props=props)  # pyright: ignore

            if column_dtypes is None:
                raise NotImplementedError(
                    f"No valid dtype mapping found for column '{column}'."
                )
            valid_dtypes[column] = column_dtypes

        return valid_dtypes

    @classmethod
    def _valid_dtypes(  # noqa: C901
        cls: Type[ModelType],  # pyright: ignore
        column: str,
        props: Dict,
    ) -> Optional[List[PolarsDataType]]:
        """
        Map schema property to list of valid polars data types.

        Args:
            props: Dictionary value retrieved from BaseModel._schema_properties().

        Returns:
            List of valid dtypes. None if no mapping exists.
        """
        if "dtype" in props:

            def dtype_invalid(props: Dict) -> Tuple[bool, List[PolarsDataType]]:
                if "type" in props:
                    valid_pl_types = cls._pydantic_type_to_valid_polars_types(
                        column, props
                    )
                    if props["dtype"] not in valid_pl_types:
                        return True, valid_pl_types or []
                elif "anyOf" in props:
                    for sub_props in props["anyOf"]:
                        if sub_props["type"] == "null":
                            continue
                        else:
                            valid_pl_types = cls._pydantic_type_to_valid_polars_types(
                                column, sub_props
                            )
                            if props["dtype"] not in valid_pl_types:
                                return True, valid_pl_types or []
                return False, []

            invalid, valid_pl_types = dtype_invalid(props)
            if invalid:
                raise ValueError(
                    f"Invalid dtype {props['dtype']} for column '{column}'. Allowable polars dtypes for {display_as_type(cls.model_fields[column].annotation)} are: {', '.join([str(x) for x in valid_pl_types])}."
                )
            return [
                props["dtype"],
            ]
        elif "enum" in props and props["type"] == "string":
            return [pl.Categorical, pl.Utf8]
        elif "type" not in props:
            if "anyOf" in props:
                res = [
                    cls._valid_dtypes(column, sub_props) for sub_props in props["anyOf"]
                ]
                res = [x for x in res if x is not None]
                return list(itertools.chain.from_iterable(res))
            elif "const" in props:
                return cls._valid_dtypes(
                    column, {"type": PYTHON_TO_PYDANTIC_TYPES.get(type(props["const"]))}
                )
            return None

        return cls._pydantic_type_to_valid_polars_types(column, props)

    @classmethod
    def _pydantic_type_to_valid_polars_types(
        cls,
        column: str,
        props: Dict,
    ) -> Optional[List[PolarsDataType]]:
        if props["type"] == "array":
            array_props = props["items"]
            item_dtypes = (
                cls._valid_dtypes(column, array_props) if array_props else None
            )
            if item_dtypes is None:
                raise NotImplementedError(
                    f"No valid dtype mapping found for column '{column}'."
                )
            return [pl.List(dtype) for dtype in item_dtypes]
        elif props["type"] == "integer":
            return PL_INTEGER_DTYPES
        elif props["type"] == "number":
            if props.get("format") == "time-delta":
                return [pl.Duration]
            else:
                return [pl.Float64, pl.Float32]
        elif props["type"] == "boolean":
            return [pl.Boolean]
        elif props["type"] == "string":
            string_format = props.get("format")
            if string_format is None:
                return [pl.Utf8]
            elif string_format == "date":
                return [pl.Date]
            # TODO: Find out why this branch is not being hit
            elif string_format == "date-time":  # pragma: no cover
                return [pl.Datetime]
            elif string_format == "duration":
                return [pl.Duration]
            elif string_format.startswith("uuid"):
                return [pl.Object]
            else:
                return None  # pragma: no cover
        elif props["type"] == "null":
            return [pl.Null]
        elif props["type"] == "object":
            return [pl.Object]
        else:  # pragma: no cover
            return None

    @classproperty
    def valid_sql_types(  # type: ignore  # noqa: C901
        cls: Type[ModelType],  # pyright: ignore
    ) -> dict[str, List["DuckDBSQLType"]]:
        """
        Return a list of DuckDB SQL types which Patito considers valid for each field.

        The first item of each list is the default dtype chosen by Patito.

        Returns:
            A dictionary mapping each column string name to a list of DuckDB SQL types
            represented as strings.

        Raises:
            NotImplementedError: If one or more model fields are annotated with types
                not compatible with DuckDB.

        Example:
            >>> import patito as pt
            >>> from pprint import pprint

            >>> class MyModel(pt.Model):
            ...     bool_column: bool
            ...     str_column: str
            ...     int_column: int
            ...     float_column: float
            ...
            >>> pprint(MyModel.valid_sql_types)
            {'bool_column': ['BOOLEAN', 'BOOL', 'LOGICAL'],
             'float_column': ['DOUBLE',
                                                'FLOAT8',
                                                'NUMERIC',
                                                'DECIMAL',
                                                'REAL',
                                                'FLOAT4',
                                                'FLOAT'],
              'int_column': ['INTEGER',
                                             'INT4',
                                             'INT',
                                             'SIGNED',
                                             'BIGINT',
                                             'INT8',
                                             'LONG',
                                             'HUGEINT',
                                             'SMALLINT',
                                             'INT2',
                                             'SHORT',
                                             'TINYINT',
                                             'INT1',
                                             'UBIGINT',
                                             'UINTEGER',
                                             'USMALLINT',
                                             'UTINYINT'],
              'str_column': ['VARCHAR', 'CHAR', 'BPCHAR', 'TEXT', 'STRING']}
        """
        valid_dtypes: Dict[str, List["DuckDBSQLType"]] = {}
        for column, props in cls._schema_properties().items():
            if "sql_type" in props:
                valid_dtypes[column] = [
                    props["sql_type"],
                ]
            elif "enum" in props and props["type"] == "string":
                from patito.duckdb import _enum_type_name

                # fmt: off
                valid_dtypes[column] = [  # pyright: ignore
                    _enum_type_name(field_properties=props),  # type: ignore
                    "VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING",
                ]
                # fmt: on
            elif "type" not in props:
                raise NotImplementedError(
                    f"No valid sql_type mapping found for column '{column}'."
                )
            elif props["type"] == "integer":
                # fmt: off
                valid_dtypes[column] = [
                    "INTEGER", "INT4", "INT", "SIGNED",
                    "BIGINT", "INT8", "LONG",
                    "HUGEINT",
                    "SMALLINT", "INT2", "SHORT",
                    "TINYINT", "INT1",
                    "UBIGINT",
                    "UINTEGER",
                    "USMALLINT",
                    "UTINYINT",
                ]
                # fmt: on
            elif props["type"] == "number":
                if props.get("format") == "time-delta":
                    valid_dtypes[column] = [
                        "INTERVAL",
                    ]
                else:
                    # fmt: off
                    valid_dtypes[column] = [
                        "DOUBLE", "FLOAT8", "NUMERIC", "DECIMAL",
                        "REAL", "FLOAT4", "FLOAT",
                    ]
                    # fmt: on
            elif props["type"] == "boolean":
                # fmt: off
                valid_dtypes[column] = [
                    "BOOLEAN", "BOOL", "LOGICAL",
                ]
                # fmt: on
            elif props["type"] == "string":
                string_format = props.get("format")
                if string_format is None:
                    # fmt: off
                    valid_dtypes[column] = [
                        "VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING",
                    ]
                    # fmt: on
                elif string_format == "date":
                    valid_dtypes[column] = ["DATE"]
                # TODO: Find out why this branch is not being hit
                elif string_format == "date-time":  # pragma: no cover
                    # fmt: off
                    valid_dtypes[column] = [
                        "TIMESTAMP", "DATETIME",
                        "TIMESTAMP WITH TIMEZONE", "TIMESTAMPTZ",
                    ]
                    # fmt: on
            elif props["type"] == "null":
                valid_dtypes[column] = [
                    "INTEGER",
                ]
            else:  # pragma: no cover
                raise NotImplementedError(
                    f"No valid sql_type mapping found for column '{column}'"
                )

        return valid_dtypes

    @classproperty
    def sql_types(  # type: ignore
        cls: Type[ModelType],  # pyright: ignore
    ) -> dict[str, str]:
        """
        Return compatible DuckDB SQL types for all model fields.

        Returns:
            Dictionary with column name keys and SQL type identifier strings.

        Example:
            >>> from typing import Literal
            >>> import patito as pt

            >>> class MyModel(pt.Model):
            ...     int_column: int
            ...     str_column: str
            ...     float_column: float
            ...     literal_column: Literal["a", "b", "c"]
            ...
            >>> MyModel.sql_types
            {'int_column': 'INTEGER',
             'str_column': 'VARCHAR',
             'float_column': 'DOUBLE',
             'literal_column': 'enum__4a496993dde04060df4e15a340651b45'}
        """
        return {
            column: valid_types[0]
            for column, valid_types in cls.valid_sql_types.items()
        }

    @classproperty
    def defaults(  # type: ignore
        cls: Type[ModelType],  # pyright: ignore
    ) -> dict[str, Any]:
        """
        Return default field values specified on the model.

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

    @classproperty
    def non_nullable_columns(  # type: ignore
        cls: Type[ModelType],  # pyright: ignore
    ) -> set[str]:
        """
        Return names of those columns that are non-nullable in the schema.

        Returns:
            Set of column name strings.

        Example:
            >>> from typing import Optional
            >>> import patito as pt
            >>> class MyModel(pt.Model):
            ...     nullable_field: Optional[int]
            ...     inferred_nullable_field: int = None
            ...     non_nullable_field: int
            ...     another_non_nullable_field: str
            ...
            >>> sorted(MyModel.non_nullable_columns)
            ['another_non_nullable_field', 'non_nullable_field']
        """
        return set(
            k
            for k in cls.columns
            if type(None) not in get_args(cls.model_fields[k].annotation)
        )

    @classproperty
    def nullable_columns(  # type: ignore
        cls: Type[ModelType],  # pyright: ignore
    ) -> set[str]:
        """
        Return names of those columns that are nullable in the schema.

        Returns:
            Set of column name strings.

        Example:
            >>> from typing import Optional
            >>> import patito as pt
            >>> class MyModel(pt.Model):
            ...     nullable_field: Optional[int]
            ...     inferred_nullable_field: int = None
            ...     non_nullable_field: int
            ...     another_non_nullable_field: str
            ...
            >>> sorted(MyModel.nullable_columns)
            ['inferred_nullable_field', 'nullable_field']
        """
        return set(cls.columns) - cls.non_nullable_columns

    @classproperty
    def unique_columns(  # type: ignore
        cls: Type[ModelType],  # pyright: ignore
    ) -> set[str]:
        """
        Return columns with uniqueness constraint.

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
        props = cls._schema_properties()
        return {column for column in cls.columns if props[column].get("unique", False)}

    @classproperty
    def derived_columns(
        cls: Type[ModelType],  # type: ignore[misc]
    ) -> set[str]:
        return {
            column
            for column, props in cls._schema_properties().items()
            if "derived_from" in props
        }

    @classproperty
    def DataFrame(
        cls: Type[ModelType],  # type: ignore[misc]
    ) -> Type[DataFrame[ModelType]]:  # pyright: ignore  # noqa
        """Return DataFrame class where DataFrame.set_model() is set to self."""
        return DataFrame._construct_dataframe_model_class(
            model=cls,  # type: ignore
        )

    @classproperty
    def LazyFrame(
        cls: Type[ModelType],  # type: ignore[misc]
    ) -> Type[LazyFrame[ModelType]]:  # pyright: ignore
        """Return LazyFrame class where LazyFrame.set_model() is set to self."""
        return LazyFrame._construct_lazyframe_model_class(
            model=cls,  # type: ignore
        )

    @classmethod
    def from_row(
        cls: Type[ModelType],  # type: ignore[misc]
        row: Union["pd.DataFrame", pl.DataFrame],
        validate: bool = True,
    ) -> ModelType:
        """
        Represent a single data frame row as a Patito model.

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
        elif _PANDAS_AVAILABLE and isinstance(row, pd.Series):  # type: ignore[unreachable]
            return cls(**dict(row.items()))  # type: ignore[unreachable]
        else:
            raise TypeError(f"{cls.__name__}.from_row not implemented for {type(row)}.")
        return cls._from_polars(dataframe=dataframe, validate=validate)

    @classmethod
    def _from_polars(
        cls: Type[ModelType],
        dataframe: pl.DataFrame,
        validate: bool = True,
    ) -> ModelType:
        """
        Construct model from a single polars row.

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
        cls,
        dataframe: Union["pd.DataFrame", pl.DataFrame],
    ) -> None:
        """
        Validate the schema and content of the given dataframe.

        Args:
            dataframe: Polars DataFrame to be validated.

        Raises:
            patito.exceptions.ValidationError: If the given dataframe does not match
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
            ... except pt.ValidationError as exc:
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
        validate(dataframe=dataframe, schema=cls)

    @classmethod
    def example_value(  # noqa: C901
        cls,
        field: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Union[date, datetime, float, int, str, None, Mapping, List]:
        """
        Return a valid example value for the given model field.

        Args:
            field: Field name identifier.

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

        elif not properties["required"]:
            return None

        elif field_type == "null":
            return None

        elif "enum" in properties:
            return properties["enum"][0]

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
                return datetime(year=1970, month=1, day=1)
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
                props_o = cls.schema()["$defs"][properties["title"]]["properties"]
                return {f: cls.example_value(properties=props_o[f]) for f in props_o}
            except AttributeError:
                raise NotImplementedError(
                    "Nested example generation only supported for nested pt.Model classes."
                )

        elif field_type == "array":
            return [cls.example_value(properties=properties["items"])]

        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def example(
        cls: Type[ModelType],
        **kwargs: Any,  # noqa: ANN401
    ) -> ModelType:
        """
        Produce model instance with filled dummy data for all unspecified fields.

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
        return pd.DataFrame([dummy.dict() for dummy in dummies])

    @classmethod
    def examples(
        cls: Type[ModelType],
        data: Optional[Union[dict, Iterable]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> "patito.polars.DataFrame":
        """
        Generate polars dataframe with dummy data for all unspecified columns.

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
            │ str          ┆ cat              ┆ i64        │
            ╞══════════════╪══════════════════╪════════════╡
            │ dummy_string ┆ dry              ┆ 0          │
            └──────────────┴──────────────────┴────────────┘

            >>> Product.examples({"name": ["product A", "product B"]})
            shape: (2, 3)
            ┌───────────┬──────────────────┬────────────┐
            │ name      ┆ temperature_zone ┆ product_id │
            │ ---       ┆ ---              ┆ ---        │
            │ str       ┆ cat              ┆ i64        │
            ╞═══════════╪══════════════════╪════════════╡
            │ product A ┆ dry              ┆ 0          │
            │ product B ┆ dry              ┆ 1          │
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

        series: List[Union[pl.Series, pl.Expr]] = []
        unique_series = []
        for column_name, dtype in cls.dtypes.items():
            if dtype == pl.Object or contains_object(dtype):
                raise NotImplementedError(
                    "Example data frame generation not supported for models containing object dtypes."
                )
            if column_name not in kwargs:
                if column_name in cls.unique_columns:
                    unique_series.append(
                        pl.first().cumcount().cast(dtype).alias(column_name)
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

        return DataFrame().with_columns(series).with_columns(unique_series)

    @classmethod
    def join(
        cls: Type["Model"],
        other: Type["Model"],
        how: Literal["inner", "left", "outer", "asof", "cross", "semi", "anti"],
    ) -> Type["Model"]:
        """
        Dynamically create a new model compatible with an SQL Join operation.

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

        kwargs: Dict[str, Any] = {}
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
    def select(
        cls: Type[ModelType], fields: Union[str, Iterable[str]]
    ) -> Type["Model"]:
        """
        Create a new model consisting of only a subset of the model fields.

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
    def drop(cls: Type[ModelType], name: Union[str, Iterable[str]]) -> Type["Model"]:
        """
        Return a new model where one or more fields are excluded.

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
    def prefix(cls: Type[ModelType], prefix: str) -> Type["Model"]:
        """
        Return a new model where all field names have been prefixed.

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
    def suffix(cls: Type[ModelType], suffix: str) -> Type["Model"]:
        """
        Return a new model where all field names have been suffixed.

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
    def rename(cls: Type[ModelType], mapping: Dict[str, str]) -> Type["Model"]:
        """
        Return a new model class where the specified fields have been renamed.

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
        cls: Type[ModelType],
        **field_definitions: Any,  # noqa: ANN401
    ) -> Type["Model"]:
        """
        Return a new model class where the given fields have been added.

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
    def schema(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return schema properties where definition references have been resolved.

        Returns:
            Field information as a dictionary where the keys are field names and the
                values are dictionaries containing metadata information about the field
                itself.

        Raises:
            TypeError: if a field is annotated with an enum where the values are of
                different types.
        """
        schema = cls.model_json_schema(ref_template="{model}")
        fields = {}
        for (
            f
        ) in (
            cls.model_fields.values()
        ):  # first resolve definitions for nested models TODO checks for one-way references, if models are self-referencing this falls apart with recursion depth error
            annotation = f.annotation
            cls._update_dfn(annotation, schema)
            for a in get_args(annotation):
                cls._update_dfn(a, schema)
        for field_name, field_info in schema["properties"].items():
            fields[field_name] = cls._append_field_info_to_props(
                field_info=field_info,
                field_name=field_name,
                required=field_name in schema.get("required", set()),
                model_schema=schema,
            )
        schema["properties"] = fields
        return schema

    @classmethod
    def _update_dfn(cls, annotation: Any, schema: Dict[str, Any]):
        try:
            if issubclass(annotation, Model) and annotation.__name__ != cls.__name__:
                schema["$defs"][annotation.__name__] = annotation.schema()
        except TypeError:
            pass

    @classmethod
    def _schema_properties(cls) -> Dict[str, Any]:
        schema = cls.schema()
        return cls.schema()["properties"]

    @classmethod
    def _append_field_info_to_props(
        cls: Type[ModelType],
        field_info: Dict[str, Any],
        field_name: str,
        model_schema: Dict[str, Any],
        required: Optional[bool] = None,
    ) -> Dict[str, Any]:
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
                # TODO: Support time-delta, date, and date-time.
                definition["type"] = PYTHON_TO_PYDANTIC_TYPES[enum_type]
            field = definition
        else:
            field = field_info
        if "items" in field_info:
            field["items"] = cls._append_field_info_to_props(
                field_info["items"],
                field_name,
                model_schema,
            )
        if required is not None:
            field["required"] = required
        if "const" in field_info and "type" not in field_info:
            field["type"] = PYTHON_TO_PYDANTIC_TYPES[type(field_info["const"])]
        for f in get_args(PT_INFO):
            v = getattr(cls.model_fields[field_name], f, None)
            if v is not None:
                field[f] = v
        return field

    @classmethod
    def _derive_model(
        cls: Type[ModelType],
        model_name: str,
        field_mapping: Dict[str, Any],
    ) -> Type["Model"]:
        """
        Derive a new model with new field definitions.

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
            __model_name=model_name,
            __base__=Model,
            **new_fields,
        )

    @staticmethod
    def _derive_field(
        field: FieldInfo, make_nullable: bool = False
    ) -> Tuple[Type, FieldInfo]:
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
        field_new = Field(default=default, **extra_attrs)
        field_new.metadata = field.metadata
        return field_type, field_new


PT_INFO = Literal["constraints", "derived_from", "dtype", "unique"]


class FieldInfo(fields.FieldInfo):
    __slots__ = getattr(fields.FieldInfo, "__slots__") + (
        "constraints",
        "derived_from",
        "dtype",
        "unique",
    )

    def __init__(
        self,
        constraints: Optional[Union[pl.Expr, Sequence[pl.Expr]]] = None,
        derived_from: Optional[Union[str, pl.Expr]] = None,
        dtype: Optional[PolarsDataType] = None,
        unique: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.constraints = constraints
        self.derived_from = derived_from
        self.dtype = dtype
        self.unique = unique
        self._attributes_set.update(
            **{
                k: getattr(self, k)
                for k in get_args(PT_INFO)
                if getattr(self, k) is not None
            }
        )


def Field(  # noqa: C901
    *args,
    constraints: Optional[Union[pl.Expr, Sequence[pl.Expr]]] = None,
    derived_from: Optional[Union[str, pl.Expr]] = None,
    dtype: Optional[PolarsDataType] = None,
    unique: bool = False,
    **kwargs,
) -> Any:
    meta_kwargs = {
        k: v for k, v in kwargs.items() if k in fields.FieldInfo.metadata_lookup
    }
    base_kwargs = {k: v for k, v in kwargs.items() if k not in meta_kwargs}
    finfo = fields.Field(*args, **base_kwargs)
    return FieldInfo(
        **finfo._attributes_set,
        **meta_kwargs,
        constraints=constraints,
        derived_from=derived_from,
        dtype=dtype,
        unique=unique,
    )


class FieldDoc:
    """
    Annotate model field with additional type and validation information.

    This class is built on ``pydantic.Field`` and you can find its full documentation
    `here <https://pydantic-docs.helpmanual.io/usage/schema/#field-customization>`_.
    Patito adds additional parameters which are used when validating dataframes,
    these are documented here.

    Args:
        constraints (Union[polars.Expression, List[polars.Expression]): A single
            constraint or list of constraints, expressed as a polars expression objects.
            All rows must satisfy the given constraint. You can refer to the given column
            with ``pt.field``, which will automatically be replaced with
            ``polars.col(<field_name>)`` before evaluation.
        derived_from (Union[str, polars.Expr]): used to mark fields that are meant to be derived from other fields. Users can specify a polars expression that will be called to derive the column value when `pt.DataFrame.derive` is called.
        dtype (polars.datatype.DataType): The given dataframe column must have the given
            polars dtype, for instance ``polars.UInt64`` or ``pl.Float32``.
        unique (bool): All row values must be unique.

    Return:
        FieldInfo: Object used to represent additional constraints put upon the given
        field.

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
        ...     # Represent colors in the form of upper cased hex colors
        ...     brand_color: str = pt.Field(regex=r"^\\#[0-9A-F]{6}$")
        ...
        >>> Product.DataFrame(
        ...     {
        ...         "product_id": [1, 1],
        ...         "price": [400, 600],
        ...         "brand_color": ["#ab00ff", "AB00FF"],
        ...     }
        ... ).validate()
        Traceback (most recent call last):
          ...
        patito.exceptions.ValidationError: 4 validation errors for Product
        name
          Missing column (type=type_error.missingcolumns)
        product_id
          2 rows with duplicated values. (type=value_error.rowvalue)
        price
          Polars dtype Int64 does not match model field type. \
          (type=type_error.columndtype)
        brand_color
          2 rows with out of bound values. (type=value_error.rowvalue)
    """


Field.__doc__ = FieldDoc.__doc__
