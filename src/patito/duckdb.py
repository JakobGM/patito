"""
Module which wraps around the duckdb module in an opiniated manner.
"""
from __future__ import annotations

import hashlib
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import polars as pl
import pyarrow as pa  # type: ignore[import]
from pydantic import create_model
from typing_extensions import Literal

from patito import sql
from patito.exceptions import MultipleRowsReturned, RowDoesNotExist
from patito.polars import DataFrame
from patito.pydantic import Model, ModelType

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    import duckdb


# Types which can be used to instantiate a DuckDB Relation object
RelationSource = Union[
    DataFrame,
    pl.DataFrame,
    "pd.DataFrame",
    Path,
    str,
    "duckdb.DuckDBPyRelation",
    "Relation",
]

# Used to refer to type(self) in Relation methods which preserve the type.
# Hard-coding Relation or Relation[ModelType] does not work for subclasses
# that return type(self) since that will refer to the parent class.
# See relevant SO answer: https://stackoverflow.com/a/63178532
RelationType = TypeVar("RelationType", bound="Relation")

# The SQL types supported by DuckDB
# See: https://duckdb.org/docs/sql/data_types/overview
# fmt: off
DuckDBSQLType = Literal[
    "BIGINT", "INT8", "LONG",
    "BLOB", "BYTEA", "BINARY", "VARBINARY",
    "BOOLEAN", "BOOL", "LOGICAL",
    "DATE",
    "DOUBLE", "FLOAT8", "NUMERIC", "DECIMAL",
    "HUGEINT",
    "INTEGER", "INT4", "INT", "SIGNED",
    "INTERVAL",
    "REAL", "FLOAT4", "FLOAT",
    "SMALLINT", "INT2", "SHORT",
    "TIME",
    "TIMESTAMP", "DATETIME",
    "TIMESTAMP WITH TIMEZONE", "TIMESTAMPTZ",
    "TINYINT", "INT1",
    "UBIGINT",
    "UINTEGER",
    "USMALLINT",
    "UTINYINT",
    "UUID",
    "VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING",
]
# fmt: on

# Used for backward-compatible patches
POLARS_VERSION: Optional[Tuple[int, int, int]]
try:
    POLARS_VERSION = cast(
        Tuple[int, int, int],
        tuple(map(int, pl.__version__.split("."))),
    )
except ValueError:  # pragma: no cover
    POLARS_VERSION = None


def create_pydantic_model(relation: "duckdb.DuckDBPyRelation") -> Type[Model]:
    """Create pydantic model deserialization of the given relation."""
    pydantic_annotations = {column: (Any, ...) for column in relation.columns}
    return create_model(  # type: ignore
        relation.alias,
        __base__=Model,
        **pydantic_annotations,
    )


def _enum_type_name(field_properties: dict) -> str:
    """
    Return enum DuckDB SQL type name based on enum values.

    The same enum values, regardless of ordering, will always be given the same name.
    """
    enum_values = ", ".join(repr(value) for value in sorted(field_properties["enum"]))
    value_hash = hashlib.md5(enum_values.encode("utf-8")).hexdigest()  # noqa: #S303
    return f"enum__{value_hash}"


def _is_missing_enum_type_exception(exception: BaseException) -> bool:
    """
    Return True if the given exception might be caused by missing enum type definitions.

    Args:
        exception: Exception raised by DuckDB.

    Returns:
        True if the exception might be caused by a missing SQL enum type definition.
    """
    description = str(exception)
    # DuckDB version <= 0.3.4
    old_exception = description.startswith("Not implemented Error: DataType")
    # DuckDB version >= 0.4.0
    new_exception = description.startswith("Catalog Error: Type with name enum_")
    return old_exception or new_exception


class Relation(Generic[ModelType]):
    # The database connection which the given relation belongs to
    database: Database

    # The underlying DuckDB relation object which this class wraps around
    _relation: duckdb.DuckDBPyRelation

    # Can be set by subclasses in order to specify the serialization class for rows.
    # Must accept column names as keyword arguments.
    model: Optional[Type[ModelType]] = None

    # The alias that can be used to refer to the relation in queries
    alias: str

    def __init__(  # noqa: C901
        self,
        derived_from: RelationSource,
        database: Optional[Database] = None,
        model: Optional[Type[ModelType]] = None,
    ) -> None:
        """
        Create a new relation object containing data to be queried with DuckDB.

        Args:
            derived_from: Data to be represented as a DuckDB relation object.
                Can be one of the following types:

                - A pandas or polars DataFrame.
                - An SQL query represented as a string.
                - A ``Path`` object pointing to a CSV or a parquet file.
                  The path must point to an existing file with either a ``.csv``
                  or ``.parquet`` file extension.
                - A native DuckDB relation object (``duckdb.DuckDBPyRelation``).
                - A ``patito.Relation`` object.

            database: Which database to load the relation into. If not provided,
                the default DuckDB database will be used.

            model: Sub-class of ``patito.Model`` which specifies how to deserialize rows
                when fetched with methods such as :ref:`Relation.get()<Relation.get>`
                and ``__iter__()``.

                Will also be used to create a strict table schema if
                :ref:`Relation.create_table()<Relation.create_table>`.
                schema should be constructed.

                If not provided, a dynamic model fitting the relation schema will be created
                when required.

                Can also be set later dynamically by invoking
                :ref:`Relation.set_model()<Relation.set_model>`.

        Raises:
            ValueError: If any one of the following cases are encountered:

                - If a provided ``Path`` object does not have a ``.csv`` or
                  ``.parquet`` file extension.
                - If a database and relation object is provided, but the relation object
                  does not belong to the database.

            TypeError: If the type of ``derived_from`` is not supported.

        Examples:
            Instantiated from a dataframe:

            >>> import patito as pt
            >>> df = pt.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> pt.Relation(df).filter("a > 2").to_df()
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 3   ┆ 6   │
            └─────┴─────┘

            Instantiated from an SQL query:

            >>> pt.Relation("select 1 as a, 2 as b").to_df()
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            └─────┴─────┘
        """
        import duckdb

        if isinstance(derived_from, Relation):
            if (
                database is not None
                and derived_from.database.connection is not database.connection
            ):
                raise ValueError(
                    "Relations can't be casted between database connections."
                )
            self.database = derived_from.database
            self._relation = derived_from._relation
            self.model = derived_from.model
            return

        if database is None:
            self.database = Database.default()
        else:
            self.database = database

        if isinstance(derived_from, duckdb.DuckDBPyRelation):
            relation = derived_from
        elif isinstance(derived_from, str):
            relation = self.database.connection.from_query(derived_from)
        elif _PANDAS_AVAILABLE and isinstance(derived_from, pd.DataFrame):
            # We must replace pd.NA with np.nan in order for it to be considered
            # as null by DuckDB. Otherwise it will casted to the string <NA>
            # or even segfault.
            derived_from = derived_from.fillna(np.nan)
            relation = self.database.connection.from_df(derived_from)
        elif isinstance(derived_from, pl.DataFrame):
            relation = self.database.connection.from_arrow(derived_from.to_arrow())
        elif isinstance(derived_from, Path):
            if derived_from.suffix.lower() == ".parquet":
                relation = self.database.connection.from_parquet(str(derived_from))
            elif derived_from.suffix.lower() == ".csv":
                relation = self.database.connection.from_csv_auto(str(derived_from))
            else:
                raise ValueError(
                    f"Unsupported file suffix {derived_from.suffix!r} for data import!"
                )
        else:
            raise TypeError  # pragma: no cover

        self._relation = relation
        if model is not None:
            self.model = model  # pyright: ignore

    def aggregate(
        self,
        *aggregations: str,
        group_by: Union[str, Iterable[str]],
        **named_aggregations: str,
    ) -> Relation:
        """
        Return relation formed by ``GROUP BY`` SQL aggregation(s).

        Args:
            aggregations: Zero or more aggregation expressions such as
                "sum(column_name)" and "count(distinct column_name)".
            named_aggregations: Zero or more aggregated expressions where the keyword is
                used to name the given aggregation. For example,
                ``my_column="sum(column_name)"`` is inserted as
                ``"sum(column_name) as my_column"`` in the executed SQL query.
            group_by: A single column name or iterable collection of column names to
                group by.

        Examples:
            >>> import patito as pt
            >>> df = pt.DataFrame({"a": [1, 2, 3], "b": ["X", "Y", "X"]})
            >>> relation = pt.Relation(df)
            >>> relation.aggregate(
            ...     "b",
            ...     "sum(a)",
            ...     "greatest(b)",
            ...     max_a="max(a)",
            ...     group_by="b",
            ... ).to_df()
            shape: (2, 4)
            ┌─────┬────────┬─────────────┬───────┐
            │ b   ┆ sum(a) ┆ greatest(b) ┆ max_a │
            │ --- ┆ ---    ┆ ---         ┆ ---   │
            │ str ┆ f64    ┆ str         ┆ i64   │
            ╞═════╪════════╪═════════════╪═══════╡
            │ X   ┆ 4.0    ┆ X           ┆ 3     │
            ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
            │ Y   ┆ 2.0    ┆ Y           ┆ 2     │
            └─────┴────────┴─────────────┴───────┘
        """
        expression = ", ".join(
            aggregations
            + tuple(
                f"{expression} as {column_name}"
                for column_name, expression in named_aggregations.items()
            )
        )
        relation = self._relation.aggregate(
            aggr_expr=expression,
            group_expr=group_by if isinstance(group_by, str) else ", ".join(group_by),
        )
        return self._wrap(relation=relation, schema_change=True)

    def add_suffix(
        self,
        suffix: str,
        include: Optional[Collection[str]] = None,
        exclude: Optional[Collection[str]] = None,
    ) -> Relation:
        """
        Add a suffix to all the columns of the relation.

        Args:
            suffix: A string to append to add to all columns names.
            include: If provided, only the given columns will be renamed.
            exclude: If provided, the given columns will `not` be renamed.

        Raises:
            TypeError: If both include `and` exclude are provided at the same time.

        Examples:
            >>> import patito as pt
            >>> relation = pt.Relation("select 1 as column_1, 2 as column_2")
            >>> relation.add_suffix("_renamed").to_df()
            shape: (1, 2)
            ┌──────────────────┬──────────────────┐
            │ column_1_renamed ┆ column_2_renamed │
            │ ---              ┆ ---              │
            │ i64              ┆ i64              │
            ╞══════════════════╪══════════════════╡
            │ 1                ┆ 2                │
            └──────────────────┴──────────────────┘

            >>> relation.add_suffix("_renamed", include=["column_1"]).to_df()
            shape: (1, 2)
            ┌──────────────────┬──────────┐
            │ column_1_renamed ┆ column_2 │
            │ ---              ┆ ---      │
            │ i64              ┆ i64      │
            ╞══════════════════╪══════════╡
            │ 1                ┆ 2        │
            └──────────────────┴──────────┘

            >>> relation.add_suffix("_renamed", exclude=["column_1"]).to_df()
            shape: (1, 2)
            ┌──────────┬──────────────────┐
            │ column_1 ┆ column_2_renamed │
            │ ---      ┆ ---              │
            │ i64      ┆ i64              │
            ╞══════════╪══════════════════╡
            │ 1        ┆ 2                │
            └──────────┴──────────────────┘
        """
        if include is not None and exclude is not None:
            raise TypeError("Both include and exclude provided at the same time!")
        elif include is not None:
            included = lambda column: column in include
        elif exclude is not None:
            included = lambda column: column not in exclude
        else:
            included = lambda _: True  # noqa: E731

        return self.select(
            ", ".join(
                f"{column} as {column}{suffix}" if included(column) else column
                for column in self.columns
            )
        )

    def add_prefix(
        self,
        prefix: str,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Relation:
        """
        Add a prefix to all the columns of the relation.

        Args:
            prefix: A string to prepend to add to all the columns names.
            include: If provided, only the given columns will be renamed.
            exclude: If provided, the given columns will `not` be renamed.

        Raises:
            TypeError: If both include `and` exclude are provided at the same time.

        Examples:
            >>> import patito as pt
            >>> relation = pt.Relation("select 1 as column_1, 2 as column_2")
            >>> relation.add_prefix("renamed_").to_df()
            shape: (1, 2)
            ┌──────────────────┬──────────────────┐
            │ renamed_column_1 ┆ renamed_column_2 │
            │ ---              ┆ ---              │
            │ i64              ┆ i64              │
            ╞══════════════════╪══════════════════╡
            │ 1                ┆ 2                │
            └──────────────────┴──────────────────┘

            >>> relation.add_prefix("renamed_", include=["column_1"]).to_df()
            shape: (1, 2)
            ┌──────────────────┬──────────┐
            │ renamed_column_1 ┆ column_2 │
            │ ---              ┆ ---      │
            │ i64              ┆ i64      │
            ╞══════════════════╪══════════╡
            │ 1                ┆ 2        │
            └──────────────────┴──────────┘

            >>> relation.add_prefix("renamed_", exclude=["column_1"]).to_df()
            shape: (1, 2)
            ┌──────────┬──────────────────┐
            │ column_1 ┆ renamed_column_2 │
            │ ---      ┆ ---              │
            │ i64      ┆ i64              │
            ╞══════════╪══════════════════╡
            │ 1        ┆ 2                │
            └──────────┴──────────────────┘
        """
        if include is not None and exclude is not None:
            raise TypeError("Both include and exclude provided at the same time!")
        elif include is not None:
            included = lambda column: column in include
        elif exclude is not None:
            included = lambda column: column not in exclude
        else:
            included = lambda _: True

        return self.select(
            ", ".join(
                f"{column} as {prefix}{column}" if included(column) else column
                for column in self.columns
            )
        )

    def all(self, *filters: str, **equalities: Union[int, float, str]) -> bool:
        """
        Return ``True`` if the given predicate(s) are true for all rows in the relation.

        See :ref:`Relation.filter()<Relation.filter>` for additional information
        regarding the parameters.

        Args:
            filters: SQL predicates to satisfy.
            equalities: SQL equality predicates to satisfy.

        Examples:
            >>> import patito as pt
            >>> df = pt.DataFrame(
            ...     {
            ...         "even_number": [2, 4, 6],
            ...         "odd_number": [1, 3, 5],
            ...         "zero": [0, 0, 0],
            ...     }
            ... )
            >>> relation = pt.Relation(df)
            >>> relation.all(zero=0)
            True
            >>> relation.all(
            ...     "even_number % 2 = 0",
            ...     "odd_number % 2 = 1",
            ...     zero=0,
            ... )
            True
            >>> relation.all(zero=1)
            False
            >>> relation.all("odd_number % 2 = 0")
            False
        """
        return self.filter(*filters, **equalities).count() == self.count()

    def case(
        self,
        *,
        from_column: str,
        to_column: str,
        mapping: Dict[sql.SQLLiteral, sql.SQLLiteral],
        default: sql.SQLLiteral,
    ) -> Relation:
        """
        Map values of one column over to a new column.

        Args:
            from_column: Name of column defining the domain of the mapping.
            to_column: Name of column to insert the mapped values into.
            mapping: Dictionary defining the mapping. The dictionary keys represent the
                input values, while the dictionary values represent the output values.
                Items are inserted into the SQL case statement by their repr() string
                value.
            default: Default output value for inputs which have no provided mapping.

        Examples:
            The following case statement...

            >>> import patito as pt
            >>> db = pt.Database()
            >>> relation = db.to_relation("select 1 as a union select 2 as a")
            >>> relation.case(
            ...     from_column="a",
            ...     to_column="b",
            ...     mapping={1: "one", 2: "two"},
            ...     default="three",
            ... ).order(by="a").to_df()
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ str │
            ╞═════╪═════╡
            │ 1   ┆ one │
            ├╌╌╌╌╌┼╌╌╌╌╌┤
            │ 2   ┆ two │
            └─────┴─────┘

            ... is equivalent with:

            >>> case_statement = pt.sql.Case(
            ...     on_column="a",
            ...     mapping={1: "one", 2: "two"},
            ...     default="three",
            ...     as_column="b",
            ... )
            >>> relation.select(f"*, {case_statement}").order(by="a").to_df()
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ str │
            ╞═════╪═════╡
            │ 1   ┆ one │
            ├╌╌╌╌╌┼╌╌╌╌╌┤
            │ 2   ┆ two │
            └─────┴─────┘
        """

        case_statement = sql.Case(
            on_column=from_column,
            mapping=mapping,
            default=default,
            as_column=to_column,
        )
        new_relation = self._relation.project(f"*, {case_statement}")
        return self._wrap(relation=new_relation, schema_change=True)

    def cast(
        self: RelationType,
        model: Optional[ModelType] = None,
        strict: bool = False,
        include: Optional[Collection[str]] = None,
        exclude: Optional[Collection[str]] = None,
    ) -> RelationType:
        """
        Cast the columns of the relation to types compatible with the associated model.

        The associated model must either be set by invoking
        :ref:`Relation.set_model() <Relation.set_model>` or provided with the ``model``
        parameter.

        Any columns of the relation that are not part of the given model schema will be
        left as-is.

        Args:
            model: If :ref:`Relation.set_model() <Relation.set_model>` has not been
                invoked or is intended to be overwritten.
            strict: If set to ``False``, columns which are technically compliant with
                the specified field type, will not be casted. For example, a column
                annotated with ``int`` is technically compliant with ``SMALLINT``, even
                if ``INTEGER`` is the default SQL type associated with ``int``-annotated
                fields. If ``strict`` is set to ``True``, the resulting dtypes will
                be forced to the default dtype associated with each python type.
            include: If provided, only the given columns will be casted.
            exclude: If provided, the given columns will `not` be casted.

        Returns:
            New relation where the columns have been casted according to the model
            schema.

        Examples:
            >>> import patito as pt
            >>> class Schema(pt.Model):
            ...     float_column: float
            ...
            >>> relation = pt.Relation("select 1 as float_column")
            >>> relation.types["float_column"]
            'INTEGER'
            >>> relation.cast(model=Schema).types["float_column"]
            'DOUBLE'

            >>> relation = pt.Relation("select 1::FLOAT as float_column")
            >>> relation.cast(model=Schema).types["float_column"]
            'FLOAT'
            >>> relation.cast(model=Schema, strict=True).types["float_column"]
            'DOUBLE'

            >>> class Schema(pt.Model):
            ...     column_1: float
            ...     column_2: float
            ...
            >>> relation = pt.Relation("select 1 as column_1, 2 as column_2").set_model(
            ...     Schema
            ... )
            >>> relation.types
            {'column_1': 'INTEGER', 'column_2': 'INTEGER'}
            >>> relation.cast(include=["column_1"]).types
            {'column_1': 'DOUBLE', 'column_2': 'INTEGER'}
            >>> relation.cast(exclude=["column_1"]).types
            {'column_1': 'INTEGER', 'column_2': 'DOUBLE'}
        """
        if model is not None:
            relation = self.set_model(model)
            schema = model
        elif self.model is not None:
            relation = self
            schema = cast(ModelType, self.model)
        else:
            class_name = self.__class__.__name__
            raise TypeError(
                f"{class_name}.cast() invoked without "
                f"{class_name}.model having been set! "
                f"You should invoke {class_name}.set_model() first "
                "or explicitly provide a model to .cast()."
            )

        if include is not None and exclude is not None:
            raise ValueError(
                f"Both include and exclude provided to {self.__class__.__name__}.cast()!"
            )
        elif include is not None:
            include = set(include)
        elif exclude is not None:
            include = set(relation.columns) - set(exclude)
        else:
            include = set(relation.columns)

        new_columns = []
        for column, current_type in relation.types.items():
            if column not in schema.columns:
                new_columns.append(column)
            elif column in include and (
                strict or current_type not in schema.valid_sql_types[column]
            ):
                new_type = schema.sql_types[column]
                new_columns.append(f"{column}::{new_type} as {column}")
            else:
                new_columns.append(column)
        return cast(RelationType, self.select(*new_columns))

    def coalesce(
        self: RelationType,
        **column_expressions: Union[str, int, float],
    ) -> RelationType:
        """
        Replace null-values in given columns with respective values.

        For example, ``coalesce(column_name=value)`` is compiled to:
        ``f"coalesce({column_name}, {repr(value)}) as column_name"`` in the resulting
        SQL.

        Args:
            column_expressions: Keywords indicate which columns to coalesce, while the
                string representation of the respective arguments are used as the
                null-replacement.

        Return:
            Relation: Relation where values have been filled in for nulls in the given
            columns.

        Examples:
            >>> import patito as pt
            >>> df = pt.DataFrame(
            ...     {
            ...         "a": [1, None, 3],
            ...         "b": ["four", "five", None],
            ...         "c": [None, 8.0, 9.0],
            ...     }
            ... )
            >>> relation = pt.Relation(df)
            >>> relation.coalesce(a=2, b="six").to_df()
            shape: (3, 3)
            ┌─────┬──────┬──────┐
            │ a   ┆ b    ┆ c    │
            │ --- ┆ ---  ┆ ---  │
            │ i64 ┆ str  ┆ f64  │
            ╞═════╪══════╪══════╡
            │ 1   ┆ four ┆ null │
            ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │ 2   ┆ five ┆ 8.0  │
            ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │ 3   ┆ six  ┆ 9.0  │
            └─────┴──────┴──────┘
        """
        projections = []
        for column in self.columns:
            if column in column_expressions:
                expression = column_expressions[column]
                projections.append(f"coalesce({column}, {expression!r}) as {column}")
            else:
                projections.append(column)
        return cast(RelationType, self.select(*projections))

    @property
    def columns(self) -> List[str]:
        """
        Return the columns of the relation as a list of strings.

        Examples:
            >>> import patito as pt
            >>> pt.Relation("select 1 as a, 2 as b").columns
            ['a', 'b']
        """
        # Under certain specific circumstances columns are suffixed with
        # :1, which need to be removed from the column name.
        return [column.partition(":")[0] for column in self._relation.columns]

    def count(self) -> int:
        """
        Return the number of rows in the given relation.

        Returns:
            Number of rows in the relation as an integer.

        Examples:
            >>> import patito as pt
            >>> relation = pt.Relation("select 1 as a")
            >>> relation.count()
            1
            >>> (relation + relation).count()
            2

            The :ref:`Relation.__len__()<Relation.__len__>` method invokes
            ``Relation.count()`` under the hood, and is equivalent:

            >>> len(relation)
            1
            >>> len(relation + relation)
            2
        """
        return cast(Tuple[int], self._relation.aggregate("count(*)").fetchone())[0]

    def create_table(self: RelationType, name: str) -> RelationType:
        """
        Create new database table based on relation.

        If ``self.model`` is set with :ref:`Relation.set_model()<Relation.set_model>`,
        then the model is used to infer the table schema.
        Otherwise, a permissive table schema is created based on the relation data.

        Returns:
            Relation: A relation pointing to the newly created table.

        Examples:
            >>> from typing import Literal
            >>> import patito as pt

            >>> df = pt.DataFrame({"enum_column": ["A", "A", "B"]})
            >>> relation = pt.Relation(df)
            >>> relation.create_table("permissive_table").types
            {'enum_column': 'VARCHAR'}

            >>> class TableSchema(pt.Model):
            ...     enum_column: Literal["A", "B", "C"]
            ...
            >>> relation.set_model(TableSchema).create_table("strict_table").types
            {'enum_column': 'enum__7ba49365cc1b0fd57e61088b3bc9aa25'}
        """
        if self.model is not None:
            self.database.create_table(name=name, model=self.model)
            self.insert_into(table=name)
        else:
            self._relation.create(table_name=name)
        return cast(RelationType, self.database.table(name))

    def create_view(
        self: RelationType,
        name: str,
        replace: bool = False,
    ) -> RelationType:
        """
        Create new database view based on relation.

        Returns:
            Relation: A relation pointing to the newly created view.

        Examples:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> df = pt.DataFrame({"column": ["A", "A", "B"]})
            >>> relation = db.to_relation(df)
            >>> relation.create_view("my_view")
            >>> db.query("select * from my_view").to_df()
            shape: (3, 1)
            ┌────────┐
            │ column │
            │ ---    │
            │ str    │
            ╞════════╡
            │ A      │
            ├╌╌╌╌╌╌╌╌┤
            │ A      │
            ├╌╌╌╌╌╌╌╌┤
            │ B      │
            └────────┘
        """
        self._relation.create_view(view_name=name, replace=replace)
        return cast(RelationType, self.database.view(name))

    def drop(self, *columns: str) -> Relation:
        """
        Remove specified column(s) from relation.

        Args:
            columns (str): Any number of string column names to be dropped.

        Examples:
            >>> import patito as pt
            >>> relation = pt.Relation("select 1 as a, 2 as b, 3 as c")
            >>> relation.columns
            ['a', 'b', 'c']
            >>> relation.drop("c").columns
            ['a', 'b']
            >>> relation.drop("b", "c").columns
            ['a']
        """
        new_columns = self.columns.copy()
        for column in columns:
            new_columns.remove(column)
        return self[new_columns]

    def distinct(self: RelationType) -> RelationType:
        """
        Drop all duplicate rows of the relation.

        Example:
            >>> import patito as pt
            >>> df = pt.DataFrame(
            ...     [[1, 2, 3], [1, 2, 3], [3, 2, 1]],
            ...     columns=["a", "b", "c"],
            ...     orient="row",
            ... )
            >>> relation = pt.Relation(df)
            >>> relation.to_df()
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 2   ┆ 3   │
            ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
            │ 1   ┆ 2   ┆ 3   │
            ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
            │ 3   ┆ 2   ┆ 1   │
            └─────┴─────┴─────┘
            >>> relation.distinct().to_df()
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 2   ┆ 3   │
            ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
            │ 3   ┆ 2   ┆ 1   │
            └─────┴─────┴─────┘
        """
        return self._wrap(self._relation.distinct(), schema_change=False)

    def except_(self: RelationType, other: RelationSource) -> RelationType:
        """
        Remove all rows that can be found in the other other relation.

        Args:
            other: Another relation or something that can be casted to a relation.

        Returns:
            New relation without the rows that can be found in the other relation.

        Example:
            >>> import patito as pt
            >>> relation_123 = pt.Relation("select 1 union select 2 union select 3")
            >>> relation_123.order(by="1").to_df()
            shape: (3, 1)
            ┌─────┐
            │ 1   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            ├╌╌╌╌╌┤
            │ 2   │
            ├╌╌╌╌╌┤
            │ 3   │
            └─────┘
            >>> relation_2 = pt.Relation("select 2")
            >>> relation_2.to_df()
            shape: (1, 1)
            ┌─────┐
            │ 2   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 2   │
            └─────┘
            >>> relation_123.except_(relation_2).order(by="1").to_df()
            shape: (2, 1)
            ┌─────┐
            │ 1   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            ├╌╌╌╌╌┤
            │ 3   │
            └─────┘
        """
        return self._wrap(
            self._relation.except_(self.database.to_relation(other)._relation),
            schema_change=False,
        )

    def execute(self) -> duckdb.DuckDBPyResult:
        """
        Execute built relation query and return result object.

        Returns:
            A native ``duckdb.DuckDBPyResult`` object representing the executed query.

        Examples:
            >>> import patito as pt
            >>> relation = pt.Relation(
            ...     "select 1 as a, 2 as b union select 3 as a, 4 as b"
            ... )
            >>> result = relation.aggregate("sum(a)", group_by="").execute()
            >>> result.description()
            [('sum(a)', 'NUMBER', None, None, None, None, None)]
            >>> result.fetchall()
            [(4,)]
        """
        # A star-select is here performed in order to work around certain DuckDB bugs
        return self._relation.project("*").execute()

    def get(self, *filters: str, **equalities: Union[str, int, float]) -> ModelType:
        """
        Fetch the single row that matches the given filter(s).

        If you expect a relation to already return one row, you can use get() without
        any arguments to return that row.

        Raises:
            RuntimeError: RuntimeError is thrown if not exactly one single row matches
                the given filter.

        Args:
            filters (str): A conjunction of SQL where clauses.
            equalities (Any): A conjunction of SQL equality clauses. The keyword name
                is the column and the parameter is the value of the equality.

        Returns:
            Model: A Patito model representing the given row.

        Examples:
            >>> import patito as pt
            >>> import polars as pl
            >>> df = pt.DataFrame({"product_id": [1, 2, 3], "price": [10, 10, 20]})
            >>> relation = pt.Relation(df).set_alias("my_relation")

            The ``.get()`` method will by default return a dynamically constructed
            Patito model if no model has been associated with the given relation:

            >>> relation.get(product_id=1)
            my_relation(product_id=1, price=10)

            If a Patito model has been associated with the relation, by the use of
            :ref:`Relation.set_model()<Relation.set_model>`, then the given model will
            be used to represent the return type:

            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     price: float
            ...
            >>> relation.set_model(Product).get(product_id=1)
            Product(product_id=1, price=10.0)

            You can invoke ``.get()`` without any arguments on relations containing
            exactly one row:

            >>> relation.filter(product_id=1).get()
            my_relation(product_id=1, price=10)

            If the given predicate matches multiple rows a ``MultipleRowsReturned``
            exception will be raised:

            >>> try:
            ...     relation.get(price=10)
            ... except pt.exceptions.MultipleRowsReturned as e:
            ...     print(e)
            ...
            Relation.get(price=10) returned 2 rows!

            If the given predicate matches zero rows a ``RowDoesNotExist`` exception
            will be raised:

            >>> try:
            ...     relation.get(price=0)
            ... except pt.exceptions.RowDoesNotExist as e:
            ...     print(e)
            ...
            Relation.get(price=0) returned 0 rows!
        """
        if filters or equalities:
            relation = self.filter(*filters, **equalities)
        else:
            relation = self
        result = relation.execute()
        row = cast(tuple, result.fetchone())
        if row is None or result.fetchone() is not None:
            args = [repr(f) for f in filters]
            args.extend(f"{key}={value!r}" for key, value in equalities.items())
            args_string = ",".join(args)

            num_rows = relation.count()
            if num_rows == 0:
                raise RowDoesNotExist(f"Relation.get({args_string}) returned 0 rows!")
            else:
                raise MultipleRowsReturned(
                    f"Relation.get({args_string}) returned {num_rows} rows!"
                )
        return self._to_model(row=row)

    def _to_model(self, row: tuple) -> ModelType:
        """
        Cast row tuple to proper return type.

        If self.model is set, either by a class variable of a subclass or by the
        invocation of Relation.set_model(), that type is used to construct the return
        value. Otherwise, a pydantic model is dynamically created based on the column
        schema of the relation.
        """
        kwargs = {column: value for column, value in zip(self.columns, row)}
        if self.model:
            return self.model(**kwargs)
        else:
            RowModel = create_pydantic_model(relation=self._relation)
            return cast(
                ModelType,
                RowModel(**kwargs),
            )

    def filter(
        self: RelationType,
        *filters: str,
        **equalities: Union[str, int, float],
    ) -> RelationType:
        """
        Return subset of rows of relation that satisfy the given predicates.

        The method returns self if no filters are provided.

        Args:
            filters: A conjunction of SQL ``WHERE`` clauses.
            equalities: A conjunction of SQL equality clauses. The keyword name
                is the column and the parameter is the value of the equality.

        Returns:
            Relation: A new relation where all rows satisfy the given criteria.

        Examples:
            >>> import patito as pt
            >>> df = pt.DataFrame(
            ...     {
            ...         "number": [1, 2, 3, 4],
            ...         "string": ["A", "A", "B", "B"],
            ...     }
            ... )
            >>> relation = pt.Relation(df)
            >>> relation.filter("number % 2 = 0").to_df()
            shape: (2, 2)
            ┌────────┬────────┐
            │ number ┆ string │
            │ ---    ┆ ---    │
            │ i64    ┆ str    │
            ╞════════╪════════╡
            │ 2      ┆ A      │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 4      ┆ B      │
            └────────┴────────┘

            >>> relation.filter(number=1, string="A").to_df()
            shape: (1, 2)
            ┌────────┬────────┐
            │ number ┆ string │
            │ ---    ┆ ---    │
            │ i64    ┆ str    │
            ╞════════╪════════╡
            │ 1      ┆ A      │
            └────────┴────────┘
        """
        if not filters and not equalities:
            return self

        clauses: List[str] = []
        if filters:
            clauses.extend(filters)
        if equalities:
            clauses.extend(f"{key}={value!r}" for key, value in equalities.items())
        filter_string = " and ".join(f"({clause})" for clause in clauses)
        return self._wrap(self._relation.filter(filter_string), schema_change=False)

    def join(
        self: RelationType,
        other: RelationSource,
        *,
        on: str,
        how: Literal["inner", "left"] = "inner",
    ) -> RelationType:
        """
        Join relation with other relation source based on condition.

        See :ref:`Relation.inner_join() <Relation.inner_join>` and
        :ref:`Relation.left_join() <Relation.left_join>` for alternative method
        shortcuts instead of using ``how``.

        Args:
            other: A source which can be casted to a ``Relation`` object, and be used
                as the right table in the join.
            on: Join condition following the ``INNER JOIN ... ON`` in the SQL query.
            how: Either ``"left"`` or ``"inner"`` for what type of SQL join operation to
                perform.

        Returns:
            Relation: New relation based on the joined relations.

        Example:
            >>> import patito as pt
            >>> products_df = pt.DataFrame(
            ...     {
            ...         "product_name": ["apple", "banana", "oranges"],
            ...         "supplier_id": [2, 1, 3],
            ...     }
            ... )
            >>> products = pt.Relation(products_df)
            >>> supplier_df = pt.DataFrame(
            ...     {
            ...         "id": [1, 2],
            ...         "supplier_name": ["Banana Republic", "Applies Inc."],
            ...     }
            ... )
            >>> suppliers = pt.Relation(supplier_df)
            >>> products.set_alias("p").join(
            ...     suppliers.set_alias("s"),
            ...     on="p.supplier_id = s.id",
            ...     how="inner",
            ... ).to_df()
            shape: (2, 4)
            ┌──────────────┬─────────────┬─────┬─────────────────┐
            │ product_name ┆ supplier_id ┆ id  ┆ supplier_name   │
            │ ---          ┆ ---         ┆ --- ┆ ---             │
            │ str          ┆ i64         ┆ i64 ┆ str             │
            ╞══════════════╪═════════════╪═════╪═════════════════╡
            │ apple        ┆ 2           ┆ 2   ┆ Applies Inc.    │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ banana       ┆ 1           ┆ 1   ┆ Banana Republic │
            └──────────────┴─────────────┴─────┴─────────────────┘

            >>> products.set_alias("p").join(
            ...     suppliers.set_alias("s"),
            ...     on="p.supplier_id = s.id",
            ...     how="left",
            ... ).to_df()
            shape: (3, 4)
            ┌──────────────┬─────────────┬──────┬─────────────────┐
            │ product_name ┆ supplier_id ┆ id   ┆ supplier_name   │
            │ ---          ┆ ---         ┆ ---  ┆ ---             │
            │ str          ┆ i64         ┆ i64  ┆ str             │
            ╞══════════════╪═════════════╪══════╪═════════════════╡
            │ apple        ┆ 2           ┆ 2    ┆ Applies Inc.    │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ banana       ┆ 1           ┆ 1    ┆ Banana Republic │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ oranges      ┆ 3           ┆ null ┆ null            │
            └──────────────┴─────────────┴──────┴─────────────────┘
        """
        return self._wrap(
            self._relation.join(
                self.database.to_relation(other)._relation, condition=on, how=how
            ),
            schema_change=True,
        )

    def inner_join(self: RelationType, other: RelationSource, on: str) -> RelationType:
        """
        Inner join relation with other relation source based on condition.

        Args:
            other: A source which can be casted to a ``Relation`` object, and be used
                as the right table in the join.
            on: Join condition following the ``INNER JOIN ... ON`` in the SQL query.

        Returns:
            Relation: New relation based on the joined relations.

        Example:
            >>> import patito as pt
            >>> products_df = pt.DataFrame(
            ...     {
            ...         "product_name": ["apple", "banana", "oranges"],
            ...         "supplier_id": [2, 1, 3],
            ...     }
            ... )
            >>> products = pt.Relation(products_df)
            >>> supplier_df = pt.DataFrame(
            ...     {
            ...         "id": [1, 2],
            ...         "supplier_name": ["Banana Republic", "Applies Inc."],
            ...     }
            ... )
            >>> suppliers = pt.Relation(supplier_df)
            >>> products.set_alias("p").inner_join(
            ...     suppliers.set_alias("s"),
            ...     on="p.supplier_id = s.id",
            ... ).to_df()
            shape: (2, 4)
            ┌──────────────┬─────────────┬─────┬─────────────────┐
            │ product_name ┆ supplier_id ┆ id  ┆ supplier_name   │
            │ ---          ┆ ---         ┆ --- ┆ ---             │
            │ str          ┆ i64         ┆ i64 ┆ str             │
            ╞══════════════╪═════════════╪═════╪═════════════════╡
            │ apple        ┆ 2           ┆ 2   ┆ Applies Inc.    │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ banana       ┆ 1           ┆ 1   ┆ Banana Republic │
            └──────────────┴─────────────┴─────┴─────────────────┘
        """
        return self._wrap(
            self._relation.join(
                other_rel=self.database.to_relation(other)._relation,
                condition=on,
                how="inner",
            ),
            schema_change=True,
        )

    def left_join(self: RelationType, other: RelationSource, on: str) -> RelationType:
        """
        Left join relation with other relation source based on condition.

        Args:
            other: A source which can be casted to a Relation object, and be used as
                the right table in the join.
            on: Join condition following the ``LEFT JOIN ... ON`` in the SQL query.

        Returns:
            Relation: New relation based on the joined tables.

        Example:
            >>> import patito as pt
            >>> products_df = pt.DataFrame(
            ...     {
            ...         "product_name": ["apple", "banana", "oranges"],
            ...         "supplier_id": [2, 1, 3],
            ...     }
            ... )
            >>> products = pt.Relation(products_df)
            >>> supplier_df = pt.DataFrame(
            ...     {
            ...         "id": [1, 2],
            ...         "supplier_name": ["Banana Republic", "Applies Inc."],
            ...     }
            ... )
            >>> suppliers = pt.Relation(supplier_df)
            >>> products.set_alias("p").left_join(
            ...     suppliers.set_alias("s"),
            ...     on="p.supplier_id = s.id",
            ... ).to_df()
            shape: (3, 4)
            ┌──────────────┬─────────────┬──────┬─────────────────┐
            │ product_name ┆ supplier_id ┆ id   ┆ supplier_name   │
            │ ---          ┆ ---         ┆ ---  ┆ ---             │
            │ str          ┆ i64         ┆ i64  ┆ str             │
            ╞══════════════╪═════════════╪══════╪═════════════════╡
            │ apple        ┆ 2           ┆ 2    ┆ Applies Inc.    │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ banana       ┆ 1           ┆ 1    ┆ Banana Republic │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ oranges      ┆ 3           ┆ null ┆ null            │
            └──────────────┴─────────────┴──────┴─────────────────┘
        """
        return self._wrap(
            self._relation.join(
                other_rel=self.database.to_relation(other)._relation,
                condition=on,
                how="left",
            ),
            schema_change=True,
        )

    def limit(self: RelationType, n: int, *, offset: int = 0) -> RelationType:
        """
        Remove all but the first n rows.

        Args:
            n: The number of rows to keep.
            offset: Disregard the first ``offset`` rows before starting to count which
                rows to keep.

        Returns:
            New relation with only n rows.

        Example:
            >>> import patito as pt
            >>> relation = (
            ...     pt.Relation("select 1 as column")
            ...     + pt.Relation("select 2 as column")
            ...     + pt.Relation("select 3 as column")
            ...     + pt.Relation("select 4 as column")
            ... )
            >>> relation.limit(2).to_df()
            shape: (2, 1)
            ┌────────┐
            │ column │
            │ ---    │
            │ i64    │
            ╞════════╡
            │ 1      │
            ├╌╌╌╌╌╌╌╌┤
            │ 2      │
            └────────┘
            >>> relation.limit(2, offset=2).to_df()
            shape: (2, 1)
            ┌────────┐
            │ column │
            │ ---    │
            │ i64    │
            ╞════════╡
            │ 3      │
            ├╌╌╌╌╌╌╌╌┤
            │ 4      │
            └────────┘
        """
        return self._wrap(self._relation.limit(n=n, offset=offset), schema_change=False)

    def order(self: RelationType, by: Union[str, Iterable[str]]) -> RelationType:
        """
        Change the order of the rows of the relation.

        Args:
            by: An ``ORDER BY`` SQL expression such as ``"age DESC"`` or
                ``("age DESC", "name ASC")``.

        Returns:
            New relation where the rows have been ordered according to ``by``.

        Example:
            >>> import patito as pt
            >>> df = pt.DataFrame(
            ...     {
            ...         "name": ["Alice", "Bob", "Charles", "Diana"],
            ...         "age": [20, 20, 30, 35],
            ...     }
            ... )
            >>> df
            shape: (4, 2)
            ┌─────────┬─────┐
            │ name    ┆ age │
            │ ---     ┆ --- │
            │ str     ┆ i64 │
            ╞═════════╪═════╡
            │ Alice   ┆ 20  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Bob     ┆ 20  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Charles ┆ 30  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Diana   ┆ 35  │
            └─────────┴─────┘
            >>> relation = pt.Relation(df)
            >>> relation.order(by="age desc").to_df()
            shape: (4, 2)
            ┌─────────┬─────┐
            │ name    ┆ age │
            │ ---     ┆ --- │
            │ str     ┆ i64 │
            ╞═════════╪═════╡
            │ Diana   ┆ 35  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Charles ┆ 30  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Alice   ┆ 20  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Bob     ┆ 20  │
            └─────────┴─────┘
            >>> relation.order(by=["age desc", "name desc"]).to_df()
            shape: (4, 2)
            ┌─────────┬─────┐
            │ name    ┆ age │
            │ ---     ┆ --- │
            │ str     ┆ i64 │
            ╞═════════╪═════╡
            │ Diana   ┆ 35  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Charles ┆ 30  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Bob     ┆ 20  │
            ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
            │ Alice   ┆ 20  │
            └─────────┴─────┘
        """
        order_expr = by if isinstance(by, str) else ", ".join(by)
        return self._wrap(
            self._relation.order(order_expr=order_expr),
            schema_change=False,
        )

    def insert_into(
        self: RelationType,
        table: str,
    ) -> RelationType:
        """
        Insert all rows of the relation into a given table.

        The relation must contain all the columns present in the target table.
        Extra columns are ignored and the column order is automatically matched
        with the target table.

        Args:
            table: Name of table for which to insert values into.

        Returns:
            Relation: The original relation, i.e. ``self``.

        Examples:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> db.to_relation("select 1 as a").create_table("my_table")
            >>> db.table("my_table").to_df()
            shape: (1, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            └─────┘
            >>> db.to_relation("select 2 as a").insert_into("my_table")
            >>> db.table("my_table").to_df()
            shape: (2, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            ├╌╌╌╌╌┤
            │ 2   │
            └─────┘
        """
        table_relation = self.database.table(table)
        missing_columns = set(table_relation.columns) - set(self.columns)
        if missing_columns:
            raise TypeError(
                f"Relation is missing column(s) {missing_columns} "
                f"in order to be inserted into table '{table}'!",
            )

        reordered_relation = self[table_relation.columns]
        reordered_relation._relation.insert_into(table_name=table)
        return self

    def intersect(self: RelationType, other: RelationSource) -> RelationType:
        """
        Return a new relation containing the rows that are present in both relations.

        This is a set operation which will remove duplicate rows as well.

        Args:
            other: Another relation with the same column names.

        Returns:
            Relation[Model]: A new relation with only those rows that are present in
            both relations.

        Example:
            >>> import patito as pt
            >>> df1 = pt.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
            >>> df2 = pt.DataFrame({"a": [1, 1, 3], "b": [1, 1, 3]})
            >>> pt.Relation(df1).intersect(pt.Relation(df2)).to_df()
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 1   │
            └─────┴─────┘
        """
        other = self.database.to_relation(other)
        return self._wrap(
            self._relation.intersect(other._relation),
            schema_change=False,
        )

    def select(
        self,
        *projections: Union[str, int, float],
        **named_projections: Union[str, int, float],
    ) -> Relation:
        """
        Return relation based on one or more SQL ``SELECT`` projections.

        Keyword arguments are converted into ``{arg} as {keyword}`` in the executed SQL
        query.

        Args:
            *projections: One or more strings representing SQL statements to be
                selected. For example ``"2"`` or ``"another_column"``.
            **named_projections: One ore more keyword arguments where the keyword
                specifies the name of the new column and the value is an SQL statement
                defining the content of the new column. For example
                ``new_column="2 * another_column"``.

        Examples:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> relation = db.to_relation(pt.DataFrame({"original_column": [1, 2, 3]}))
            >>> relation.select("*").to_df()
            shape: (3, 1)
            ┌─────────────────┐
            │ original_column │
            │ ---             │
            │ i64             │
            ╞═════════════════╡
            │ 1               │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ 2               │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ 3               │
            └─────────────────┘
            >>> relation.select("*", multiplied_column="2 * original_column").to_df()
            shape: (3, 2)
            ┌─────────────────┬───────────────────┐
            │ original_column ┆ multiplied_column │
            │ ---             ┆ ---               │
            │ i64             ┆ i64               │
            ╞═════════════════╪═══════════════════╡
            │ 1               ┆ 2                 │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ 2               ┆ 4                 │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ 3               ┆ 6                 │
            └─────────────────┴───────────────────┘
        """
        # We expand '*' to an explicit list of columns in order to support redefining
        # columns within the star expressed columns.
        expanded_projections: list = list(projections)
        try:
            star_index = projections.index("*")
            if named_projections:
                # Allow explicitly named projections to overwrite star-selected columns
                expanded_projections[star_index : star_index + 1] = [
                    column for column in self.columns if column not in named_projections
                ]
            else:
                expanded_projections[star_index : star_index + 1] = self.columns
        except ValueError:
            pass

        projection = ", ".join(
            expanded_projections
            + list(  # pyright: ignore
                f"{expression} as {column_name}"
                for column_name, expression in named_projections.items()
            )
        )
        try:
            relation = self._relation.project(projection)
        except RuntimeError as exc:  # pragma: no cover
            # We might get a RunTime error if the enum type has not
            # been created yet. If so, we create all enum types for
            # this model.
            if self.model is not None and _is_missing_enum_type_exception(exc):
                self.database.create_enum_types(model=self.model)
                relation = self._relation.project(projection)
            else:
                raise exc
        return self._wrap(relation=relation, schema_change=True)

    def rename(self, **columns: str) -> Relation:
        """
        Rename columns as specified.

        Args:
            **columns: A set of keyword arguments where the keyword is the old column
                name and the value is the new column name.

        Raises:
            ValueError: If any of the given keywords do not exist as columns in the
                relation.

        Examples:
            >>> import patito as pt
            >>> relation = pt.Relation("select 1 as a, 2 as b")
            >>> relation.rename(b="c").to_df().select(["a", "c"])
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ c   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            └─────┴─────┘
        """
        existing_columns = set(self.columns)
        missing = set(columns.keys()) - set(existing_columns)
        if missing:
            raise ValueError(
                f"Column '{missing.pop()}' can not be renamed as it does not exist. "
                f"The columns of the relation are: {', '.join(existing_columns)}."
            )
        # If we rename a column to overwrite another existing one, the column should
        # be overwritten.
        existing_columns = set(existing_columns) - set(columns.values())
        relation = self._relation.project(
            ", ".join(
                f"{column} as {columns.get(column, column)}"
                for column in existing_columns
            )
        )
        return self._wrap(relation=relation, schema_change=True)

    def set_alias(self: RelationType, name: str) -> RelationType:
        """
        Set SQL alias for the given relation to be used in further queries.

        Args:
            name: The new alias for the given relation.

        Returns:
            Relation: A new relation containing the same query but addressable with the
            new alias.

        Example:
            >>> import patito as pt
            >>> relation_1 = pt.Relation("select 1 as a, 2 as b")
            >>> relation_2 = pt.Relation("select 1 as a, 3 as c")
            >>> relation_1.set_alias("x").inner_join(
            ...     relation_2.set_alias("y"),
            ...     on="x.a = y.a",
            ... ).select("x.a", "y.a", "b", "c").to_df()
            shape: (1, 4)
            ┌─────┬─────┬─────┬─────┐
            │ a   ┆ a:1 ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╪═════╡
            │ 1   ┆ 1   ┆ 2   ┆ 3   │
            └─────┴─────┴─────┴─────┘
        """
        return self._wrap(
            self._relation.set_alias(name),
            schema_change=False,
        )

    def set_model(self, model):  # type: ignore[no-untyped-def] # noqa: ANN
        """
        Associate a give Patito model with the relation.

        The returned relation has an associated ``.model`` attribute which can in turn
        be used by several methods such as :ref:`Relation.get()<Relation.get>`,
        :ref:`Relation.create_table()<Relation.create_table>`, and
        :ref:`Relation.__iter__<Relation.__iter__>`.

        Args:
            model: A Patito Model class specifying the intended schema of the relation.

        Returns:
            Relation[model]: A new relation with the associated model.

        Example:
            >>> from typing import Literal
            >>> import patito as pt
            >>> class MySchema(pt.Model):
            ...     float_column: float
            ...     enum_column: Literal["A", "B", "C"]
            ...
            >>> relation = pt.Relation("select 1 as float_column, 'A' as enum_column")
            >>> relation.get()
            query_relation(float_column=1, enum_column='A')
            >>> relation.set_model(MySchema).get()
            MySchema(float_column=1.0, enum_column='A')
            >>> relation.create_table("unmodeled_table").types
            {'float_column': 'INTEGER', 'enum_column': 'VARCHAR'}
            >>> relation.set_model(MySchema).create_table("modeled_table").types
            {'float_column': 'DOUBLE',
             'enum_column': 'enum__7ba49365cc1b0fd57e61088b3bc9aa25'}
        """
        # We are not able to annotate the generic instance of type(self)[type(model)]
        # due to the lack of higher-kinded generics in python as of this writing.
        # See: https://github.com/python/typing/issues/548
        # This cast() will be wrong for sub-classes of Relation...
        return cast(
            Relation[model],
            type(self)(
                derived_from=self._relation,
                database=self.database,
                model=model,
            ),
        )

    @property
    def types(self) -> Dict[str, DuckDBSQLType]:
        """
        Return the SQL types of all the columns of the given relation.

        Returns:
            dict[str, str]: A dictionary where the keys are the column names and the
            values are SQL types as strings.

        Examples:
            >>> import patito as pt
            >>> pt.Relation("select 1 as a, 'my_value' as b").types
            {'a': 'INTEGER', 'b': 'VARCHAR'}
        """
        return dict(zip(self.columns, self._relation.types))

    def to_pandas(self) -> "pd.DataFrame":
        """
        Return a pandas DataFrame representation of relation object.

        Returns: A ``pandas.DataFrame`` object containing all the data of the relation.

        Example:
            >>> import patito as pt
            >>> pt.Relation("select 1 as column union select 2 as column").order(
            ...     by="1"
            ... ).to_pandas()
                  column
               0       1
               1       2
        """
        return self._relation.to_df()

    def to_df(self) -> DataFrame:
        """
        Return a polars DataFrame representation of relation object.

        Returns: A ``patito.DataFrame`` object which inherits from ``polars.DataFrame``.

        Example:
            >>> import patito as pt
            >>> pt.Relation("select 1 as column union select 2 as column").order(
            ...     by="1"
            ... ).to_df()
            shape: (2, 1)
            ┌────────┐
            │ column │
            │ ---    │
            │ i64    │
            ╞════════╡
            │ 1      │
            ├╌╌╌╌╌╌╌╌┤
            │ 2      │
            └────────┘
        """
        # Here we do a star-select to work around certain weird issues with DuckDB
        self._relation = self._relation.project("*")
        arrow_table = cast(pa.lib.Table, self._relation.to_arrow_table())
        try:
            # We cast `INTEGER`-typed columns to `pl.Int64` when converting to Polars
            # because polars is much more eager to store integer Series as 64-bit
            # integers. Otherwise there must be done a lot of manual casting whenever
            # you cross the boundary between DuckDB and polars.
            return DataFrame._from_arrow(arrow_table).with_column(
                pl.col(pl.Int32).cast(pl.Int64)
            )
        except pa.ArrowInvalid:  # pragma: no cover
            # Empty relations with enum columns can sometimes produce errors.
            # As a last-ditch effort, we convert such columns to VARCHAR.
            casted_columns = [
                f"{field.name}::VARCHAR as {field.name}"
                if isinstance(field.type, pa.DictionaryType)
                else field.name
                for field in arrow_table.schema
            ]
            non_enum_relation = self._relation.project(", ".join(casted_columns))
            arrow_table = non_enum_relation.to_arrow_table()
            return DataFrame._from_arrow(arrow_table).with_column(
                pl.col(pl.Int32).cast(pl.Int64)
            )

    def to_series(self) -> pl.Series:
        """
        Convert the given relation to a polars Series.

        Raises:
            TypeError: If the given relation does not contain exactly one column.

        Returns: A ``polars.Series`` object containing the data of the relation.

        Example:
            >>> import patito as pt
            >>> relation = pt.Relation("select 1 as a union select 2 as a")
            >>> relation.order(by="a").to_series()
            shape: (2,)
            Series: 'a' [i32]
            [
                        1
                        2
            ]
        """
        if len(self._relation.columns) != 1:
            raise TypeError(
                f"{self.__class__.__name__}.to_series() was invoked on a relation with "
                f"{len(self._relation.columns)} columns, while exactly 1 is required!"
            )
        dataframe: DataFrame = DataFrame._from_arrow(self._relation.to_arrow_table())
        return dataframe.to_series(index=0).alias(name=self.columns[0])

    def union(self: RelationType, other: RelationSource) -> RelationType:
        """
        Produce a new relation that contains the rows of both relations.

        The ``+`` operator can also be used to union two relations.

        The two relations must have the same column names, but not necessarily in the
        same order as reordering of columns is automatically performed, unlike regular
        SQL.

        Duplicates are `not` dropped.

        Args:
            other: A ``patito.Relation`` object or something that can be `casted` to
                ``patito.Relation``. See :ref:`Relation<Relation.__init__>`.

        Returns:
            New relation containing the rows of both ``self`` and ``other``.

        Raises:
            TypeError: If the two relations do not contain the same columns.

        Examples:
            >>> import patito as pt
            >>> relation_1 = pt.Relation("select 1 as a")
            >>> relation_2 = pt.Relation("select 2 as a")
            >>> relation_1.union(relation_2).to_df()
            shape: (2, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            ├╌╌╌╌╌┤
            │ 2   │
            └─────┘

            >>> (relation_1 + relation_2).to_df()
            shape: (2, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            ├╌╌╌╌╌┤
            │ 2   │
            └─────┘
        """
        other_relation = self.database.to_relation(other)
        if set(self.columns) != set(other_relation.columns):
            msg = "Union between relations with different column names is not allowed."
            additional_left = set(self.columns) - set(other_relation.columns)
            additional_right = set(other_relation.columns) - set(self.columns)
            if additional_left:
                msg += f" Additional columns in left relation: {additional_left}."
            if additional_right:
                msg += f" Additional columns in right relation: {additional_right}."
            raise TypeError(msg)
        if other_relation.columns != self.columns:
            reordered_relation = other_relation[self.columns]
        else:
            reordered_relation = other_relation
        unioned_relation = self._relation.union(reordered_relation._relation)
        return self._wrap(relation=unioned_relation, schema_change=False)

    def with_columns(
        self,
        **named_projections: Union[str, int, float],
    ) -> Relation:
        """
        Return relations with additional columns.

        If the provided columns expressions already exists as a column on the relation,
        the given column is overwritten.

        Args:
            named_projections: A set of column expressions, where the keyword is used
                as the column name, while the right-hand argument is a valid SQL
                expression.

        Returns:
            Relation with the given columns appended, or possibly overwritten.

        Examples:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> relation = db.to_relation("select 1 as a, 2 as b")
            >>> relation.with_columns(c="a + b").to_df()
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 2   ┆ 3   │
            └─────┴─────┴─────┘
        """
        return self.select("*", **named_projections)

    def with_missing_defaultable_columns(
        self: RelationType,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> RelationType:
        """
        Add missing defaultable columns filled with the default values of correct type.

        Make sure to invoke :ref:`Relation.set_model()<Relation.set_model>` with the
        correct model schema before executing
        ``Relation.with_missing_default_columns()``.

        Args:
            include: If provided, only fill in default values for missing columns part
                of this collection of column names.
            exclude: If provided, do `not` fill in default values for missing columns
                part of this collection of column names.

        Returns:
            Relation: New relation where missing columns with default values according
            to the schema have been filled in.

        Example:
            >>> import patito as pt
            >>> class MyModel(pt.Model):
            ...     non_default_column: int
            ...     another_non_default_column: int
            ...     default_column: int = 42
            ...     another_default_column: int = 42
            ...
            >>> relation = pt.Relation(
            ...     "select 1 as non_default_column, 2 as default_column"
            ... )
            >>> relation.to_df()
            shape: (1, 2)
            ┌────────────────────┬────────────────┐
            │ non_default_column ┆ default_column │
            │ ---                ┆ ---            │
            │ i64                ┆ i64            │
            ╞════════════════════╪════════════════╡
            │ 1                  ┆ 2              │
            └────────────────────┴────────────────┘
            >>> relation.set_model(MyModel).with_missing_defaultable_columns().to_df()
            shape: (1, 3)
            ┌────────────────────┬────────────────┬────────────────────────┐
            │ non_default_column ┆ default_column ┆ another_default_column │
            │ ---                ┆ ---            ┆ ---                    │
            │ i64                ┆ i64            ┆ i64                    │
            ╞════════════════════╪════════════════╪════════════════════════╡
            │ 1                  ┆ 2              ┆ 42                     │
            └────────────────────┴────────────────┴────────────────────────┘
        """
        if self.model is None:
            class_name = self.__class__.__name__
            raise TypeError(
                f"{class_name}.with_missing_default_columns() invoked without "
                f"{class_name}.model having been set! "
                f"You should invoke {class_name}.set_model() first!"
            )
        elif include is not None and exclude is not None:
            raise TypeError("Both include and exclude provided at the same time!")

        missing_columns = set(self.model.columns) - set(self.columns)
        defaultable_columns = self.model.defaults.keys()
        missing_defaultable_columns = missing_columns & defaultable_columns

        if exclude is not None:
            missing_defaultable_columns -= set(exclude)
        elif include is not None:
            missing_defaultable_columns = missing_defaultable_columns & set(include)

        projection = "*"
        for column_name in missing_defaultable_columns:
            sql_type = self.model.sql_types[column_name]
            default_value = self.model.defaults[column_name]
            projection += f", {default_value!r}::{sql_type} as {column_name}"

        try:
            relation = self._relation.project(projection)
        except Exception as exc:  # pragma: no cover
            # We might get a RunTime error if the enum type has not
            # been created yet. If so, we create all enum types for
            # this model.
            if _is_missing_enum_type_exception(exc):
                self.database.create_enum_types(model=self.model)
                relation = self._relation.project(projection)
            else:
                raise exc
        return self._wrap(relation=relation, schema_change=False)

    def with_missing_nullable_columns(
        self: RelationType,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> RelationType:
        """
        Add missing nullable columns filled with correctly typed nulls.

        Make sure to invoke :ref:`Relation.set_model()<Relation.set_model>` with the
        correct model schema before executing
        ``Relation.with_missing_nullable_columns()``.

        Args:
            include: If provided, only fill in null values for missing columns part of
                this collection of column names.
            exclude: If provided, do `not` fill in null values for missing columns
                part of this collection of column names.

        Returns:
            Relation: New relation where missing nullable columns have been filled in
            with null values.

        Example:
            >>> from typing import Optional
            >>> import patito as pt
            >>> class MyModel(pt.Model):
            ...     non_nullable_column: int
            ...     nullable_column: Optional[int]
            ...     another_nullable_column: Optional[int]
            ...
            >>> relation = pt.Relation("select 1 as nullable_column")
            >>> relation.to_df()
            shape: (1, 1)
            ┌─────────────────┐
            │ nullable_column │
            │ ---             │
            │ i64             │
            ╞═════════════════╡
            │ 1               │
            └─────────────────┘
            >>> relation.set_model(MyModel).with_missing_nullable_columns().to_df()
            shape: (1, 2)
            ┌─────────────────┬─────────────────────────┐
            │ nullable_column ┆ another_nullable_column │
            │ ---             ┆ ---                     │
            │ i64             ┆ i64                     │
            ╞═════════════════╪═════════════════════════╡
            │ 1               ┆ null                    │
            └─────────────────┴─────────────────────────┘
        """
        if self.model is None:
            class_name = self.__class__.__name__
            raise TypeError(
                f"{class_name}.with_missing_nullable_columns() invoked without "
                f"{class_name}.model having been set! "
                f"You should invoke {class_name}.set_model() first!"
            )
        elif include is not None and exclude is not None:
            raise TypeError("Both include and exclude provided at the same time!")

        missing_columns = set(self.model.columns) - set(self.columns)
        missing_nullable_columns = self.model.nullable_columns & missing_columns

        if exclude is not None:
            missing_nullable_columns -= set(exclude)
        elif include is not None:
            missing_nullable_columns = missing_nullable_columns & set(include)

        projection = "*"
        for missing_nullable_column in missing_nullable_columns:
            sql_type = self.model.sql_types[missing_nullable_column]
            projection += f", null::{sql_type} as {missing_nullable_column}"

        try:
            relation = self._relation.project(projection)
        except Exception as exc:  # pragma: no cover
            # We might get a RunTime error if the enum type has not
            # been created yet. If so, we create all enum types for
            # this model.
            if _is_missing_enum_type_exception(exc):
                self.database.create_enum_types(model=self.model)
                relation = self._relation.project(projection)
            else:
                raise exc
        return self._wrap(relation=relation, schema_change=False)

    def __add__(self: RelationType, other: RelationSource) -> RelationType:
        """
        Execute ``self.union(other)``.

        See :ref:`Relation.union()<Relation.union>` for full documentation.
        """
        return self.union(other)

    def __eq__(self, other: object) -> bool:
        """Check if Relation is equal to a Relation-able data source."""
        other_relation = self.database.to_relation(other)  # type: ignore
        # Check if the number of rows are equal, and then check if each row is equal.
        # Use zip(self, other_relation, strict=True) when we upgrade to Python 3.10.
        return self.count() == other_relation.count() and all(
            row == other_row for row, other_row in zip(self, other_relation)
        )

    def __getitem__(self, key: Union[str, Iterable[str]]) -> Relation:
        """
        Return Relation with selected columns.

        Uses :ref:`Relation.select()<Relation.select>` under-the-hood in order to
        perform the selection. Can technically be used to rename columns,
        define derived columns, and so on, but prefer the use of Relation.select() for
        such use cases.

        Args:
            key: Columns to select, either a single column represented as a string, or
                an iterable of strings.

        Returns:
            New relation only containing the column subset specified.

        Example:
            >>> import patito as pt
            >>> relation = pt.Relation("select 1 as a, 2 as b, 3 as c")
            >>> relation.to_df()
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 2   ┆ 3   │
            └─────┴─────┴─────┘
            >>> relation[["a", "b"]].to_df()
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            └─────┴─────┘
            >>> relation["a"].to_df()
            shape: (1, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            └─────┘
        """
        projection = key if isinstance(key, str) else ", ".join(key)
        return self._wrap(
            relation=self._relation.project(projection),
            schema_change=True,
        )

    def __iter__(self) -> Iterator[ModelType]:
        """
        Iterate over rows in relation.

        If :ref:`Relation.set_model()<Relation.set_model>` has been invoked first, the
        given model will be used to deserialize each row. Otherwise a Patito model
        is dynamically constructed which fits the schema of the relation.

        Returns:
            Iterator[Model]: An iterator of patito Model objects representing each row.

        Example:
            >>> from typing import Literal
            >>> import patito as pt
            >>> df = pt.DataFrame({"float_column": [1, 2], "enum_column": ["A", "B"]})
            >>> relation = pt.Relation(df).set_alias("my_relation")
            >>> for row in relation:
            ...     print(row)
            ...
            float_column=1 enum_column='A'
            float_column=2 enum_column='B'
            >>> list(relation)
            [my_relation(float_column=1, enum_column='A'),
             my_relation(float_column=2, enum_column='B')]

            >>> class MySchema(pt.Model):
            ...     float_column: float
            ...     enum_column: Literal["A", "B", "C"]
            ...
            >>> relation = relation.set_model(MySchema)
            >>> for row in relation:
            ...     print(row)
            ...
            float_column=1.0 enum_column='A'
            float_column=2.0 enum_column='B'
            >>> list(relation)
            [MySchema(float_column=1.0, enum_column='A'),
             MySchema(float_column=2.0, enum_column='B')]
        """
        result = self._relation.execute()
        while True:
            row_tuple = result.fetchone()
            if not row_tuple:
                return
            else:
                yield self._to_model(row_tuple)

    def __len__(self) -> int:
        """
        Return the number of rows in the relation.

        See :ref:`Relation.count()<Relation.count>` for full documentation.
        """
        return self.count()

    def __str__(self) -> str:
        """
        Return string representation of Relation object.

        Includes an expression tree, the result columns, and a result preview.

        Example:
            >>> import patito as pt
            >>> products = pt.Relation(
            ...     pt.DataFrame(
            ...         {
            ...             "product_name": ["apple", "red_apple", "banana", "oranges"],
            ...             "supplier_id": [2, 2, 1, 3],
            ...         }
            ...     )
            ... ).set_alias("products")
            >>> print(str(products))  # xdoctest: +SKIP
            ---------------------
            --- Relation Tree ---
            ---------------------
            arrow_scan(94609350519648, 140317161740928, 140317161731168, 1000000)\

            ---------------------
            -- Result Columns  --
            ---------------------
            - product_name (VARCHAR)
            - supplier_id (BIGINT)\

            ---------------------
            -- Result Preview  --
            ---------------------
            product_name    supplier_id
            VARCHAR BIGINT
            [ Rows: 4]
            apple   2
            red_apple       2
            banana  1
            oranges 3

            >>> suppliers = pt.Relation(
            ...     pt.DataFrame(
            ...         {
            ...             "id": [1, 2],
            ...             "supplier_name": ["Banana Republic", "Applies Inc."],
            ...         }
            ...     )
            ... ).set_alias("suppliers")
            >>> relation = (
            ...     products.set_alias("p")
            ...     .inner_join(
            ...         suppliers.set_alias("s"),
            ...         on="p.supplier_id = s.id",
            ...     )
            ...     .aggregate(
            ...         "supplier_name",
            ...         num_products="count(product_name)",
            ...         group_by=["supplier_id", "supplier_name"],
            ...     )
            ... )
            >>> print(str(relation))  # xdoctest: +SKIP
            ---------------------
            --- Relation Tree ---
            ---------------------
            Aggregate [supplier_name, count(product_name)]
              Join INNER p.supplier_id = s.id
                arrow_scan(94609350519648, 140317161740928, 140317161731168, 1000000)
                arrow_scan(94609436221024, 140317161740928, 140317161731168, 1000000)\

            ---------------------
            -- Result Columns  --
            ---------------------
            - supplier_name (VARCHAR)
            - num_products (BIGINT)\

            ---------------------
            -- Result Preview  --
            ---------------------
            supplier_name   num_products
            VARCHAR BIGINT
            [ Rows: 2]
            Applies Inc.    2
            Banana Republic 1

        """
        return str(self._relation)

    def _wrap(
        self: RelationType,
        relation: "duckdb.DuckDBPyRelation",
        schema_change: bool = False,
    ) -> RelationType:
        """
        Wrap DuckDB Relation object in same Relation wrapper class as self.

        This will preserve the type of the relation, even for subclasses Relation.
        It should therefore only be used for relations which can be considered schema-
        compatible with the original relation. Otherwise set schema_change to True
        in order to create a Relation base object instead.
        """
        return type(self)(
            derived_from=relation,
            database=self.database,
            model=self.model if not schema_change else None,
        )


class Database:
    # Types created in order to represent enum strings
    enum_types: Set[str]

    def __init__(
        self,
        path: Optional[Path] = None,
        read_only: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Instantiate a new DuckDB database, either persisted to disk or in-memory.

        Args:
            path: Optional path to store all the data to. If ``None`` the data is
                persisted in-memory only.
            read_only: If the database connection should be a read-only connection.
            **kwargs: Additional keywords forwarded to ``duckdb.connect()``.

        Examples:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> db.to_relation("select 1 as a, 2 as b").create_table("my_table")
            >>> db.query("select * from my_table").to_df()
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            └─────┴─────┘
        """
        import duckdb

        self.path = path
        self.connection = duckdb.connect(
            database=str(path) if path else ":memory:",
            read_only=read_only,
            **kwargs,
        )
        self.enum_types: Set[str] = set()

    @classmethod
    def default(cls) -> Database:
        """
        Return the default DuckDB database.

        Returns:
            A patito :ref:`Database<Database>` object wrapping around the given
            connection.

        Example:
            >>> import patito as pt
            >>> db = pt.Database.default()
            >>> db.query("select 1 as a, 2 as b").to_df()
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            └─────┴─────┘
        """
        import duckdb

        return cls.from_connection(duckdb.default_connection)

    @classmethod
    def from_connection(cls, connection: "duckdb.DuckDBPyConnection") -> Database:
        """
        Create database from native DuckDB connection object.

        Args:
            connection: A native DuckDB connection object created with
                ``duckdb.connect()``.

        Returns:
            A :ref:`Database<Database>` object wrapping around the given connection.

        Example:
            >>> import duckdb
            >>> import patito as pt
            >>> connection = duckdb.connect()
            >>> database = pt.Database.from_connection(connection)
        """
        obj = cls.__new__(cls)
        obj.connection = connection
        obj.enum_types = set()
        return obj

    def to_relation(
        self,
        derived_from: RelationSource,
    ) -> Relation:
        """
        Create a new relation object based on data source.

        The given data will be represented as a relation associated with the database.
        ``Database(x).to_relation(y)`` is equivalent to
        ``Relation(y, database=Database(x))``.

        Args:
            derived_from (RelationSource): One of either a polars or pandas
                ``DataFrame``, a ``pathlib.Path`` to a parquet or CSV file, a SQL query
                string, or an existing relation.

        Example:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> db.to_relation("select 1 as a, 2 as b").to_df()
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 2   │
            └─────┴─────┘
            >>> db.to_relation(pt.DataFrame({"c": [3, 4], "d": ["5", "6"]})).to_df()
            shape: (2, 2)
            ┌─────┬─────┐
            │ c   ┆ d   │
            │ --- ┆ --- │
            │ i64 ┆ str │
            ╞═════╪═════╡
            │ 3   ┆ 5   │
            ├╌╌╌╌╌┼╌╌╌╌╌┤
            │ 4   ┆ 6   │
            └─────┴─────┘
        """
        return Relation(
            derived_from=derived_from,
            database=self,
        )

    def execute(
        self,
        query: str,
        *parameters: Collection[Union[str, int, float, bool]],
    ) -> None:
        """
        Execute SQL query in DuckDB database.

        Args:
            query: A SQL statement to execute. Does `not` have to be terminated with
                a semicolon (``;``).
            parameters: One or more sets of parameters to insert into prepared
                statements. The values are replaced in place of the question marks
                (``?``) in the prepared query.

        Example:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> db.execute("create table my_table (x bigint);")
            >>> db.execute("insert into my_table values (1), (2), (3)")
            >>> db.table("my_table").to_df()
            shape: (3, 1)
            ┌─────┐
            │ x   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            ├╌╌╌╌╌┤
            │ 2   │
            ├╌╌╌╌╌┤
            │ 3   │
            └─────┘

            Parameters can be specified when executing prepared queries.

            >>> db.execute("delete from my_table where x = ?", (2,))
            >>> db.table("my_table").to_df()
            shape: (2, 1)
            ┌─────┐
            │ x   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            ├╌╌╌╌╌┤
            │ 3   │
            └─────┘

            Multiple parameter sets can be specified when executing multiple prepared
            queries.

            >>> db.execute(
            ...     "delete from my_table where x = ?",
            ...     (1,),
            ...     (3,),
            ... )
            >>> db.table("my_table").to_df()
            shape: (0, 1)
            ┌─────┐
            │ x   │
            │ --- │
            │ i64 │
            ╞═════╡
            └─────┘
        """
        duckdb_parameters: Union[
            Collection[Union[str, int, float, bool]],
            Collection[Collection[Union[str, int, float, bool]]],
            None,
        ]
        if parameters is None or len(parameters) == 0:
            duckdb_parameters = []
            multiple_parameter_sets = False
        elif len(parameters) == 1:
            duckdb_parameters = parameters[0]
            multiple_parameter_sets = False
        else:
            duckdb_parameters = parameters
            multiple_parameter_sets = True

        self.connection.execute(
            query=query,
            parameters=duckdb_parameters,
            multiple_parameter_sets=multiple_parameter_sets,
        )

    def query(self, query: str, alias: str = "query_relation") -> Relation:
        """
        Execute arbitrary SQL select query and return the relation.

        Args:
            query: Arbitrary SQL select query.
            alias: The alias to assign to the resulting relation, to be used in further
                queries.

        Returns: A relation representing the data produced by the given query.

        Example:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> relation = db.query("select 1 as a, 2 as b, 3 as c")
            >>> relation.to_df()
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 2   ┆ 3   │
            └─────┴─────┴─────┘

            >>> relation = db.query("select 1 as a, 2 as b, 3 as c", alias="my_alias")
            >>> relation.select("my_alias.a").to_df()
            shape: (1, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            └─────┘
        """
        return Relation(
            self.connection.query(query=query, alias=alias),
            database=self,
        )

    def empty_relation(self, schema: Type[ModelType]) -> Relation[ModelType]:
        """
        Create relation with zero rows, but correct schema that matches the given model.

        Args:
            schema: A patito model which specifies the column names and types of the
                given relation.

        Example:
            >>> import patito as pt
            >>> class Schema(pt.Model):
            ...     string_column: str
            ...     bool_column: bool
            ...
            >>> db = pt.Database()
            >>> empty_relation = db.empty_relation(Schema)
            >>> empty_relation.to_df()
            shape: (0, 2)
            ┌───────────────┬─────────────┐
            │ string_column ┆ bool_column │
            │ ---           ┆ ---         │
            │ str           ┆ bool        │
            ╞═══════════════╪═════════════╡
            └───────────────┴─────────────┘
            >>> non_empty_relation = db.query(
            ...     "select 'dummy' as string_column, true as bool_column"
            ... )
            >>> non_empty_relation.union(empty_relation).to_df()
            shape: (1, 2)
            ┌───────────────┬─────────────┐
            │ string_column ┆ bool_column │
            │ ---           ┆ ---         │
            │ str           ┆ bool        │
            ╞═══════════════╪═════════════╡
            │ dummy         ┆ true        │
            └───────────────┴─────────────┘
        """
        return self.to_relation(schema.examples()).limit(0)

    def table(self, name: str) -> Relation:
        """
        Return relation representing all the data in the given table.

        Args:
            name: The name of the table.

        Example:
            >>> import patito as pt
            >>> df = pt.DataFrame({"a": [1, 2], "b": [3, 4]})
            >>> db = pt.Database()
            >>> relation = db.to_relation(df)
            >>> relation.create_table(name="my_table")
            >>> db.table("my_table").to_df()
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            ├╌╌╌╌╌┼╌╌╌╌╌┤
            │ 2   ┆ 4   │
            └─────┴─────┘
        """
        return Relation(
            self.connection.table(name),
            database=self.from_connection(self.connection),
        )

    def view(self, name: str) -> Relation:
        """
        Return relation representing all the data in the given view.

        Args:
            name: The name of the view.

        Example:
            >>> import patito as pt
            >>> df = pt.DataFrame({"a": [1, 2], "b": [3, 4]})
            >>> db = pt.Database()
            >>> relation = db.to_relation(df)
            >>> relation.create_view(name="my_view")
            >>> db.view("my_view").to_df()
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            ├╌╌╌╌╌┼╌╌╌╌╌┤
            │ 2   ┆ 4   │
            └─────┴─────┘
        """
        return Relation(
            self.connection.view(name),
            database=self.from_connection(self.connection),
        )

    def create_table(
        self,
        name: str,
        model: Type[ModelType],
    ) -> Relation[ModelType]:
        """
        Create table with schema matching the provided Patito model.

        See :ref:`Relation.insert_into()<Relation.insert_into>` for how to insert data
        into the table after creation.
        The :ref:`Relation.create_table()<Relation.create_table>` method can also be
        used to create a table from a given relation `and` insert the data at the same
        time.

        Args:
            name: Name of new database table.
            model (Type[Model]): Patito model indicating names and types of table
                columns.
        Returns:
            Relation[ModelType]: Relation pointing to the new table.

        Example:
            >>> from typing import Optional
            >>> import patito as pt
            >>> class MyModel(pt.Model):
            ...     str_column: str
            ...     nullable_string_column: Optional[str]
            ...
            >>> db = pt.Database()
            >>> db.create_table(name="my_table", model=MyModel)
            >>> db.table("my_table").types
            {'str_column': 'VARCHAR', 'nullable_string_column': 'VARCHAR'}
        """
        self.create_enum_types(model=model)
        schema = model.schema()
        non_nullable = schema.get("required", [])
        columns = []
        for column_name, sql_type in model.sql_types.items():
            column = f"{column_name} {sql_type}"
            if column_name in non_nullable:
                column += " not null"
            columns.append(column)
        self.connection.execute(f"create table {name} ({','.join(columns)})")
        # TODO: Fix typing
        return self.table(name).set_model(model)  # pyright: ignore

    def create_enum_types(self, model: Type[ModelType]) -> None:
        """
        Define SQL enum types in DuckDB database.

        Args:
            model: Model for which all Literal-annotated or enum-annotated string fields
                will get respective DuckDB enum types.

        Example:
            >>> import patito as pt
            >>> class EnumModel(pt.Model):
            ...     enum_column: Literal["A", "B", "C"]
            ...
            >>> db = pt.Database()
            >>> db.create_enum_types(EnumModel)
            >>> db.enum_types
            {'enum__7ba49365cc1b0fd57e61088b3bc9aa25'}
        """
        import duckdb

        for props in model._schema_properties().values():
            if "enum" not in props or props["type"] != "string":
                # DuckDB enums only support string values
                continue

            enum_type_name = _enum_type_name(field_properties=props)
            if enum_type_name in self.enum_types:
                # This enum type has already been created
                continue

            enum_values = ", ".join(repr(value) for value in sorted(props["enum"]))
            try:
                self.connection.execute(
                    f"create type {enum_type_name} as enum ({enum_values})"
                )
            except duckdb.CatalogException as e:
                if "already exists" not in str(e):
                    raise e  # pragma: no cover
            self.enum_types.add(enum_type_name)

    def create_view(
        self,
        name: str,
        data: RelationSource,
    ) -> Relation:
        """Create a view based on the given data source."""
        return self.to_relation(derived_from=data).create_view(name)

    def __contains__(self, table: str) -> bool:
        """
        Return ``True`` if the database contains a table with the given name.

        Args:
            table: The name of the table to be checked for.

        Examples:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> "my_table" in db
            False
            >>> db.to_relation("select 1 as a, 2 as b").create_table(name="my_table")
            >>> "my_table" in db
            True
        """
        try:
            self.connection.table(table_name=table)
            return True
        except Exception:
            return False
