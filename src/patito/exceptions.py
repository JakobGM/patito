"""Module containing all custom exceptions raised by patito."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import pydantic.v1 as pydantic

if TYPE_CHECKING:
    from patito.pydantic import Model


class ValidationError(ValueError):
    """Exception raised when dataframe does not match schema."""

    def __init__(self, errors: Sequence["ErrorWrapper"], model: Model) -> None:
        self.raw_errors = errors
        self.model = model

    def errors(self) -> Sequence[pydantic.ValidationError]:
        """Return a list of pydantic validation errors."""
        return [e.as_dict() for e in self.raw_errors]


class ErrorWrapper:
    """Wrapper for specific column validation error."""

    def __init__(self, exc: Exception, loc: str) -> None:
        self.exc = exc
        self._loc = loc

    def as_dict(self) -> dict:
        return {
            "loc": (self._loc,),
            "msg": str(self.exc),
            "type": self.exc._type,
        }


class WrongColumnsError(TypeError):
    """Validation exception for column name mismatches."""

    _type = "type_error.wrongcolumns"


class MissingColumnsError(WrongColumnsError):
    """Exception for when a dataframe is missing one or more columns."""

    _type = "type_error.missingcolumns"


class SuperflousColumnsError(WrongColumnsError):
    """Exception for when a dataframe has one ore more non-specified columns."""

    _type = "type_error.superflouscolumns"


class MissingValuesError(ValueError):
    """Exception for when a dataframe has non-nullable columns with nulls."""

    _type = "value_error.missingvalues"


class ColumnDTypeError(TypeError):
    """Exception for when a dataframe has one or more columns with wrong dtypes."""

    _type = "type_error.columndtype"


class RowValueError(ValueError):
    """Exception for when a dataframe has a row with a impermissible value."""

    _type = "value_error.rowvalue"


class RowDoesNotExist(RuntimeError):
    """Exception for when a single row was expected, but none were returned."""

    _type = "value_error.rowdoesnotexist"


class MultipleRowsReturned(RuntimeError):
    """Exception for when a single row was expected, but several were returned."""

    _type = "value_error.multiplerowsreturned"
