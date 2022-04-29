"""
Module which wraps around the duckdb module in an opiniated manner.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import polars as pl
from pydantic import create_model
from typing_extensions import Literal

from patito.pydantic import PYDANTIC_TO_DUCKDB_TYPES, Model, ModelType

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    import duckdb


# Types which can be used to instantiate a DuckDB Relation object
RelationSource = Union[
    "pd.DataFrame", pl.DataFrame, Path, str, "duckdb.DuckDBPyRelation", "Relation"
]

# Used to refer to type(self) in Relation methods which preserve the type.
# Hard-coding Relation or Relation[ModelType] does not work for subclasses
# that return type(self) since that will refer to the parent class.
# See relevant SO answer: https://stackoverflow.com/a/63178532
RelationType = TypeVar("RelationType", bound="Relation")

# The SQL types supported by DuckDB
# See: https://duckdb.org/docs/sql/data_types/overview
DuckDBSQLType = Literal[
    "BIGINT",
    "BOOLEAN",
    "BLOB",
    "DATE",
    "DOUBLE",
    "DECIMAL",
    "HUGEINT",
    "INTEGER",
    "REAL",
    "SMALLINT",
    "TIME",
    "TIMESTAMP",
    "TINYINT",
    "UBIGINT",
    "UINTEGER",
    "USMALLINT",
    "UTINYINT",
    "UUID",
    "VARCHAR",
]


def create_pydantic_model(relation: "duckdb.DuckDBPyRelation") -> Type[Model]:
    """Create pydantic model deserialization of the given relation."""
    pydantic_annotations = {column: (Any, ...) for column in relation.columns}
    return create_model(
        relation.alias,
        __base__=Model,
        **pydantic_annotations,
    )


class Relation(Generic[ModelType]):
    # Can be set by subclasses in order to specify the serialization class for rows.
    # Must accept column names as keyword arguments.
    model: Optional[Type[ModelType]] = None

    # The alias that can be used to refer to the relation in queries
    alias: str
    # The columns of the relation
    columns: List[str]
    # The SQL types of the relation
    types: List[DuckDBSQLType]

    def __init__(
        self,
        relation: "duckdb.DuckDBPyRelation",
        database: Database,
        model: Optional[Type[ModelType]] = None,
    ) -> None:
        """
        Wrap around the given DuckDB Relation object associated with the given database.

        Args:
            relation (duckdb.DuckDBPyRelation): Relation to be wrapped.
            database (Database): Associated database which will be queried/mutated if
            relation is executed.
            model (Optional[Type[ModelType]]): Sub-class of patito.Model which
            specifies how to deserialize rows when fetched with methods such as .get()
            and __iter__() and how table schema should be constructed.
        """
        import duckdb

        if not isinstance(relation, duckdb.DuckDBPyRelation):
            raise TypeError(f"Invalid relation type {type(relation)}!")
        self.database = database
        self._relation = relation
        if model is not None:
            self.model = model

    def aggregate(
        self,
        *aggregations: str,
        group_by: Union[str, Iterable[str]],
        **named_aggregations: str,
    ) -> Relation:
        """
        Return relation formed by group by SQL aggregation(s).

        Args:
            aggregations: Zero or more aggregation expressions such as
                "sum(column_name)" and "count(distinct column_name)".
            group_by: A single column name or iterable collection of column names to
                group by.
            named_aggregations: Zero or more aggregated expressions where the keyword is
                used to name the given aggregation. For example,
                my_column="sum(column_name)" is inserted as
                "sum(column_name) as my_column" in the executed SQL query.
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

    def add_suffix(self, suffix: str) -> Relation:
        """Add suffix to all columns of relation."""
        return self.project(
            ", ".join(f"{column} as {column}{suffix}" for column in self.columns)
        )

    def add_prefix(self, prefix: str) -> Relation:
        """Add prefix to all columns of relation."""
        return self.project(
            ", ".join(f"{column} as {prefix}{column}" for column in self.columns)
        )

    def all(self, *filters: str, **equalities: Union[int, float, str]) -> bool:
        """
        Return True if the given conditions are true for all rows in the relation.

        See Relation.filter() for additional information regarding the parameters.
        """
        return self.filter(*filters, **equalities).count() == self.count()

    def case(
        self,
        *,
        from_column: str,
        to_column: str,
        mapping: Mapping[Any, Any],
        default: Union[str, int, float],
    ) -> Relation:
        """
        Map values of one column over to a new column.

        Args:
            from_column: Name of column definining the domain of the mapping.
            to_column: Name of column to insert the mapped values into.
            mapping: Dictionary defining the mapping. The dictionary keys represent the
                input values, while the dictionary values represent the output values.
                Items are inserted into the SQL case statement by their repr() string
                value.
            default: Default output value for inputs which have no provided mapping.
        """
        case_column_definition = f"case {from_column} " + (
            " ".join(f"when {key!r} then {value!r}" for key, value in mapping.items())
            + f" else {default} end as {to_column}"
        )
        new_relation = self._relation.project("*, " + case_column_definition)
        return self._wrap(relation=new_relation, schema_change=True)

    def coalesce(
        self: RelationType,
        **column_expressions: Union[str, int, float],
    ) -> RelationType:
        """
        Replace null-values in given columns with respective values.

        For example, coalesce(column_name=value) is compiled to:
        f"coalesce({column_name}, {repr(value)}) as column_name" in the resulting SQL.

        Args:
            column_expressions: Keywords indicate which columns to coalesce, while the
                string representation of the respective arguments are used as the
                null-replacement.
        """
        projections = []
        for column in self.columns:
            if column in column_expressions:
                expression = column_expressions[column]
                projections.append(f"coalesce({column}, {expression!r}) as {column}")
            else:
                projections.append(column)
        return self.project(*projections)

    def count(self) -> int:
        """Return the number of rows in the given relation."""
        return cast(Tuple[int], self._relation.aggregate("count(*)").fetchone())[0]

    def create_table(self: RelationType, name: str) -> RelationType:
        """
        Create new database table based on relation.

        If self.model is set, then the model is used to infer the table schema.
        Otherwise, a permissive table schema is created based on the relation data.
        """
        if self.model is not None:
            self.database.create_table(name=name, model=self.model)
            self.insert_into(table_name=name)
        else:
            self._relation.create(table_name=name)
        return self.database.table(name)

    def drop(self, *columns: str) -> Relation:
        """
        Drop specified column(s).

        Args:
            columns (str): Any number of string column names to be dropped.
        """
        new_columns = self.columns.copy()
        for column in columns:
            new_columns.remove(column)
        return self[new_columns]

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
            row (Row): A pydantic-derived base model representing the given row.
        """
        if filters or equalities:
            relation = self.filter(*filters, **equalities)
        else:
            relation = self
        result = relation.execute()
        row = result.fetchone()
        if row is None or result.fetchone() is not None:
            args = ", ".join(repr(f) for f in filters)
            kwargs = ", ".join(f"{key}={value!r}" for key, value in equalities.items())
            raise RuntimeError(
                f"Relation.get({args}, {kwargs}) returned {relation.count()} rows!"
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
        kwargs = {column: value for column, value in zip(self._relation.columns, row)}
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
        Filter rows of relation.

        Args:
            filters (str): A conjunction of SQL where clauses.
            equalities (Any): A conjunction of SQL equality clauses. The keyword name
            is the column and the parameter is the value of the equality.

        Returns:
            rel (Relation): A new relation where all rows satisfy the given criteria.
        """
        clauses = []
        if filters:
            clauses.extend(filters)
        if equalities:
            clauses.extend(f"{key}={value!r}" for key, value in equalities.items())
        filter_string = " and ".join(clauses)
        return self._wrap(self._relation.filter(filter_string), schema_change=False)

    def inner_join(self: RelationType, other: RelationSource, on: str) -> RelationType:
        """
        Inner join relation with other relation source based on condition.

        Args:
            other (RelationSource): A source which can be casted to a Relation object,
            and be used as the right table in the join.
            on (str): Join condition following the "INNER JOIN ... ON" in the SQL query.

        Returns:
            relation (Relation): New relation based on the joined tables.
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
            other (RelationSource): A source which can be casted to a Relation object,
            and be used as the right table in the join.
            on (str): Join condition following the "LEFT JOIN ... ON" in the SQL query.

        Returns:
            relation (Relation): New relation based on the joined tables.
        """
        return self._wrap(
            self._relation.join(
                other_rel=self.database.to_relation(other)._relation,
                condition=on,
                how="left",
            ),
            schema_change=True,
        )

    def insert_into(
        self: RelationType,
        table_name: str,
    ) -> RelationType:
        """
        Insert relation values into table.

        The relation must contain all the columns present in the target table.
        Extra columns are ignored and the column order is automatically matched
        with the target table.

        Args:
            table_name (str): Name of table for which to insert values into.

        Returns:
            relation: The original relation (self).
        """
        table = self.database.table(table_name)
        missing_columns = set(table.columns) - set(self.columns)
        if missing_columns:
            raise TypeError(
                f"Relation is missing column(s) {missing_columns} "
                f"in order to be inserted into table '{table_name}'!",
            )

        reordered_relation = self[table.columns]
        reordered_relation._relation.insert_into(table_name=table_name)
        return self

    def project(self, *projections: str, **named_projections: str) -> Relation:
        """
        Return relation based on one or more select projections.

        Keyword arguments are converted into "{arg} as {keyword}" SQL statements.
        For example relation.project("null as column_name") or
        relation.project("a", "b", "c is null as is_c_null", d="coalesce(a, b)").
        """
        # We expand '*' to an explicit list of columns in order to support redefining
        # columns within the star expressed columns.
        expanded_projections: list = list(projections)
        try:
            star_index = projections.index("*")
            expanded_projections[star_index : star_index + 1] = self.columns
        except ValueError:
            pass

        projection = ", ".join(
            expanded_projections
            + list(
                f"{expression} as {column_name}"
                for column_name, expression in named_projections.items()
            )
        )
        relation = self._relation.project(projection)
        return self._wrap(relation=relation, schema_change=True)

    def rename(self, **columns: str) -> Relation:
        """
        Rename columns from left-hand-side value to right-hand-side value.

        For instance, relation.rename(a="b") will rename column "a" to "b".
        """
        existing_columns = self._relation.columns
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

    def set_model(self, model):  # noqa: ANN
        """
        Specify column schema and the constructor method for rows in the relation.

        Used by methods which return individual rows, such as Relation.get()
        and Relation.__iter__().
        """
        # We are not able to annotate the generic instance of type(self)[type(model)]
        # due to the lack of higher-kinded generics in python as of this writing.
        # See: https://github.com/python/typing/issues/548
        # This cast() will be wrong for sub-classes of Relation...
        return cast(
            Relation[model],
            type(self)(relation=self._relation, database=self.database, model=model),
        )

    @property
    def sql_types(self) -> dict[str, str]:
        """Return column name -> DuckDB SQL type dictionary mapping."""
        return dict(zip(self._relation.columns, self._relation.types))

    def to_pandas(self) -> "pd.DataFrame":
        """Return a pandas DataFrame representation of relation object."""
        return cast("pd.DataFrame", self._relation.to_df())

    def to_df(self) -> pl.DataFrame:
        """Return a polars DataFrame representation of relation object."""
        return cast(pl.DataFrame, pl.from_arrow(self._relation.to_arrow_table()))

    def to_series(self) -> pl.Series:
        if len(self._relation.columns) != 1:
            raise TypeError(
                f"{self.__class__.__name__}.to_series() was invoked on a relation with "
                f"{len(self._relation.columns)} columns, while exactly 1 is required!"
            )
        dataframe = cast(pl.DataFrame, pl.from_arrow(self._relation.to_arrow_table()))
        return dataframe.to_series(index=0).alias(name=self._relation.columns[0])

    def union(self: RelationType, other: RelationSource) -> RelationType:
        """
        Produce new relation from union of two relation (sources).

        The two relations must have the same column names, but not necessarily in the
        same order as reordering of columns is automatically performed, unlike regular
        SQL.
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
        reordered_relation = other_relation[self._relation.columns]
        unioned_relation = self._relation.union(reordered_relation._relation)
        return self._wrap(relation=unioned_relation, schema_change=False)

    def with_missing_defaultable_columns(self: RelationType) -> RelationType:
        """
        Add missing defaultable columns filled with the default values of correct type.

        Make sure to invoke Relation.set_model() with the correct model schema before
        executing with_missing_default_columns().
        """
        if self.model is None:
            class_name = self.__class__.__name__
            raise TypeError(
                f"{class_name}.with_missing_default_columns() invoked without "
                f"{class_name}.model having been set! "
                "You should invoke {class_name}.set_model() first!"
            )

        missing_columns = set(self.model.columns) - set(self._relation.columns)
        defaultable_columns = self.model.defaults.keys()
        missing_defaultable_columns = missing_columns & defaultable_columns

        projection = "*"
        for column_name in missing_defaultable_columns:
            sql_type = self.model.sql_types[column_name]
            default_value = self.model.defaults[column_name]
            projection += f", {default_value!r}::{sql_type} as {column_name}"

        relation = self._relation.project(projection)
        return self._wrap(relation=relation, schema_change=False)

    def with_missing_nullable_columns(self: RelationType) -> RelationType:
        """
        Add missing nullable columns filled with correctly typed nulls.

        Make sure to invoke Relation.set_model() with the correct schema before
        executing with_missing_default_columns().
        """
        if self.model is None:
            class_name = self.__class__.__name__
            raise TypeError(
                f"{class_name}.with_missing_nullable_columns() invoked without "
                f"{class_name}.model having been set! "
                "You should invoke {class_name}.set_model() first!"
            )
        missing_columns = set(self.model.columns) - set(self.columns)
        missing_nullable_columns = self.model.nullable_columns & missing_columns

        projection = "*"
        for missing_nullable_column in missing_nullable_columns:
            sql_type = self.model.sql_types[missing_nullable_column]
            projection += f", null::{sql_type} as {missing_nullable_column}"

        relation = self._relation.project(projection)
        return self._wrap(relation=relation, schema_change=False)

    def __add__(self: RelationType, other: RelationSource) -> RelationType:
        """Invoke self.union(other)."""
        return self.union(other)

    def __eq__(self, other: RelationSource) -> bool:
        """Check if Relation is equal to a Relation-able data source."""
        other_relation = self.database.to_relation(other)
        # Check if the number of rows are equal, and then check if each row is equal.
        # Use zip(self, other_relation, strict=True) when we upgrade to Python 3.10.
        return self.count() == other_relation.count() and all(
            row == other_row for row, other_row in zip(self, other_relation)
        )

    def __getattr__(self, name: str) -> Any:  # noqa: ANN
        """
        Resolve object attribute access.

        This magic method is called whenever a non-existing attribute of self is
        accessed. We delegate all attributes to the underlying DuckDB relation
        but wrap methods which return new relations.
        """
        # We do not want end-users to use these DuckDB methods, but rather:
        # create_table, to_df, and inner_join.
        if name in {"create", "df", "join"}:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

        schema_preserving = {
            "distinct",
            "except_",
            "intersect",
            "limit",
            "order",
            "set_alias",
        }
        schema_changing = {
            "join",
            "map",
        }
        relation_returning_methods = schema_preserving | schema_changing

        if name in relation_returning_methods:
            return lambda *args, **kwargs: self._wrap(
                relation=getattr(self._relation, name)(*args, **kwargs),
                schema_change=name in schema_changing,
            )

        try:
            return getattr(self._relation, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __getitem__(self, key: Union[str, Iterable[str]]) -> Relation:
        """
        Return Relation with selected columns.

        Uses Relation.project() under-the-hood in order to perform the selection.
        Can technically be used to rename columns, define derived columns, and
        so on, but prefer the use of Relation.project() for such use cases.

        Args:
            key (Union[str, Iterable[str]): Columns to select, either a single
            column represented as a string, or an iterable of strings.

        Returns:
            relation (Relation): The return type is not exectly Relation[ModelType],
            but rather a subset of the fields present in ModelType.
        """
        projection = key if isinstance(key, str) else ", ".join(key)
        return self._wrap(
            relation=self._relation.project(projection),
            schema_change=True,
        )

    def __iter__(self) -> Iterator[ModelType]:
        """Iterate over rows in relation as column-named tuples."""
        result = self._relation.execute()
        while True:
            row_tuple = result.fetchone()
            if not row_tuple:
                return
            else:
                yield self._to_model(row_tuple)

    def __len__(self) -> int:
        """Return number of rows in relation."""
        return self.count()

    def __str__(self) -> str:
        """
        Return string representation of Relation object.

        Includes an expression tree, the result columns, and a result preview.
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
            relation=relation,
            database=self.database,
            model=self.model if not schema_change else None,
        )


class Database:
    # Method for executing SQL queries which do not return relations
    execute: Callable[[str], None]

    # Method for executing arbitrary select queries which return relations
    query: Callable[[str], Relation]

    # Method for extracting a relation referring to a table
    table: Callable[[str], Relation]

    def __init__(self, path: Optional[Path] = None) -> None:
        """
        Instantiate a new in-memory DuckDB database.

        Args:
            path: Optional path to store all the data to. If None, the data is persisted
                in-memory only.
        """
        import duckdb

        if path:
            self.connection = duckdb.connect(database=str(path))
            self.path = path
        else:
            self.connection = duckdb.connect(database=":memory:")

    def to_relation(
        self,
        derived_from: RelationSource,
    ) -> Relation:
        """
        Create a new relation object based on data source.

        Args:
            derived_from (RelationSource): One of either a pandas DataFrame,
            a pathlib.Path to a parquet or CSV file, a SQL query string,
            or an existing relation.
        """
        import duckdb

        if isinstance(derived_from, Relation):
            if derived_from.database is not self:
                raise ValueError(
                    "Relations can't be casted between database connections."
                )
            return derived_from
        elif isinstance(derived_from, duckdb.DuckDBPyRelation):
            relation = derived_from
        elif isinstance(derived_from, str):
            relation = self.connection.from_query(derived_from)
        elif _PANDAS_AVAILABLE and isinstance(derived_from, pd.DataFrame):
            # We must replace pd.NA with np.nan in order for it to be considered
            # as null by DuckDB. Otherwise it will casted to the string <NA>
            # or even segfault.
            derived_from = cast(pd.DataFrame, derived_from).fillna(np.nan)
            relation = self.connection.from_df(derived_from)
        elif isinstance(derived_from, pl.DataFrame):
            relation = self.connection.from_arrow(derived_from.to_arrow())
        elif isinstance(derived_from, Path):
            if derived_from.suffix == ".parquet":
                relation = self.connection.from_parquet(str(derived_from))
            elif derived_from.suffix == ".csv":
                relation = self.connection.from_csv_auto(str(derived_from))
            else:
                raise ValueError(
                    f"Unsupported file suffix {derived_from.suffix} for data import!"
                )
        else:
            raise TypeError
        return Relation(relation, database=self)

    def empty_relation(self, schema: Type[ModelType]) -> Relation[ModelType]:
        """
        Create relation with zero rows, but correct schema that matches the model.

        Args:
            schema: A patito model which specifies the column names and types of the
                given relation.
        """
        return self.to_relation(schema.examples()).limit(0)

    def create_table(
        self,
        name: str,
        model: Type[ModelType],
    ) -> Relation[ModelType]:
        """
        Create table with schema matching the provided model.

        Args:
            name (str): Name of new table.
            model (Type[Model]): Pydantic-derived model indicating names and types of
            table columns.
        Returns:
            relation (Relation[ModelType]): Relation pointing to the new table.
        """
        schema = model.schema()
        non_nullable = schema["required"]
        columns = []
        for column_name, props in model.schema()["properties"].items():
            if (
                "enum" in props
                and props["type"] == "string"
                and column_name in non_nullable
            ):
                enum_values = ", ".join(repr(value) for value in props["enum"])
                enum_type_name = f"{schema['title']}__{column_name}".lower()
                self.connection.execute(
                    f"create type {enum_type_name} as enum ({enum_values})"
                )
                column = f"{column_name} {enum_type_name}"
            else:
                sql_type = PYDANTIC_TO_DUCKDB_TYPES[props["type"]]
                column = f"{column_name} {sql_type}"
                if column_name in non_nullable:
                    column += " not null"
            columns.append(column)
        self.connection.execute(f"create table {name} ({','.join(columns)})")
        # TODO: Fix typing
        return self.table(name).set_model(model)  # type: ignore

    def create_view(
        self,
        name: str,
        data: RelationSource,
    ) -> Relation:
        """Create a view based on the given data source."""
        return self.to_relation(derived_from=data).create_view(name)

    def __getattr__(
        self, name: str
    ) -> Union[duckdb.DuckDBPyRelation, Relation, Callable[..., Relation]]:
        """
        Resolve object attribute access.

        This magic method is called whenever a non-existing attribute of self is
        accessed. We delegate all attributes to the underlying DuckDB connection,
        but wrap methods which return relations in the Relations wrapper.
        """
        relation_methods = {
            "from_arrow_table",
            "from_csv_auto",
            "from_df",
            "from_parquet",
            "from_query",
            "query",
            "table",
            "table_function",
            "values",
            "view",
        }
        if name in relation_methods:
            return lambda *args, **kwargs: Relation(
                getattr(self.connection, name)(*args, **kwargs), database=self
            )
        else:
            return getattr(self.connection, name)

    def __contains__(self, table: str) -> bool:
        """Return True if the database contains a table with the given name."""
        try:
            self.connection.table(table_name=table)
            return True
        except RuntimeError:
            return False
