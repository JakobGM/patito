"""Module containing SQL generation utilities."""
from typing import Dict, Optional, Union

from typing_extensions import TypeAlias

SQLLiteral: TypeAlias = Union[str, float, int, None]


def sql_repr(value: SQLLiteral) -> str:
    """
    Convert python value to equivalent SQL literal value representation.

    Args:
        value: Python object which is convertible to an equivalent SQL value type.

    Returns:
        A SQL literal representation of the given value as a string.
    """
    return "null" if value is None else repr(value)


class Case:
    """Class representing an SQL case statement."""

    def __init__(
        self,
        on_column: str,
        mapping: Dict[SQLLiteral, SQLLiteral],
        default: SQLLiteral,
        as_column: Optional[str] = None,
    ) -> None:
        """
        Map values of one column over to a new column.

        Args:
            on_column: Name of column defining the domain of the mapping.
            mapping: Dictionary defining the mapping. The dictionary keys represent the
                input values, while the dictionary values represent the output values.
                Items are inserted into the SQL case statement by their repr() string
                value. None is converted to SQL NULL.
            default: Default output value for inputs which have no provided mapping.
                If set to None, SQL NULL will be inserted as the default value.
            as_column: Name of column to insert the mapped values into. If not provided
                the SQL string expression will not end with "AS <as_column>".

        Examples:
            >>> import patito as pt
            >>> db = pt.Database()
            >>> relation = db.to_relation("select 1 as a union select 2 as a")
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
        self.on_column = on_column
        self.as_column = as_column
        self.mapping = {
            sql_repr(key): sql_repr(value) for key, value in mapping.items()
        }
        self.default_value = sql_repr(default)
        self.sql_string = f"case {self.on_column} " + (
            " ".join(f"when {key} then {value}" for key, value in self.mapping.items())
            + f" else {self.default_value} end"
        )
        if self.as_column:
            self.sql_string += f" as {as_column}"

    def __str__(self) -> str:
        """
        Return string representation of SQL case statement.

        Returns:
            String representing the case expression which can be directly inserted into
            an SQL query.
        """
        return self.sql_string
