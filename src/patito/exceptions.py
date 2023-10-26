"""Module containing all custom exceptions raised by patito."""

from typing import Any, Callable, Generator, Iterable, Optional, Sequence, Tuple, Union
from pydantic import ValidationError as ValidationError

Loc = Tuple[Union[int, str], ...]
ReprArgs = Sequence[Tuple[Optional[str], Any]]
RichReprResult = Iterable[Union[Any, Tuple[Any], Tuple[str, Any], Tuple[str, Any, Any]]]
class Representation:
    """
    Mixin to provide __str__, __repr__, and __pretty__ methods. See #884 for more details.

    __pretty__ is used by [devtools](https://python-devtools.helpmanual.io/) to provide human readable representations
    of objects.
    """

    __slots__: Tuple[str, ...] = tuple()

    def __repr_args__(self) -> 'ReprArgs':
        """
        Returns the attributes to show in __str__, __repr__, and __pretty__ this is generally overridden.

        Can either return:
        * name - value pairs, e.g.: `[('foo_name', 'foo'), ('bar_name', ['b', 'a', 'r'])]`
        * or, just values, e.g.: `[(None, 'foo'), (None, ['b', 'a', 'r'])]`
        """
        attrs = ((s, getattr(self, s)) for s in self.__slots__)
        return [(a, v) for a, v in attrs if v is not None]

    def __repr_name__(self) -> str:
        """
        Name of the instance's class, used in __repr__.
        """
        return self.__class__.__name__

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(repr(v) if a is None else f'{a}={v!r}' for a, v in self.__repr_args__())

    def __pretty__(self, fmt: Callable[[Any], Any], **kwargs: Any) -> Generator[Any, None, None]:
        """
        Used by devtools (https://python-devtools.helpmanual.io/) to provide a human readable representations of objects
        """
        yield self.__repr_name__() + '('
        yield 1
        for name, value in self.__repr_args__():
            if name is not None:
                yield name + '='
            yield fmt(value)
            yield ','
            yield 0
        yield -1
        yield ')'

    def __str__(self) -> str:
        return self.__repr_str__(' ')

    def __repr__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'

    def __rich_repr__(self) -> 'RichReprResult':
        """Get fields for Rich library"""
        for name, field_repr in self.__repr_args__():
            if name is None:
                yield field_repr
            else:
                yield name, field_repr

class ErrorWrapper(Representation):
    """Wrapper for specific column validation error."""
    __slots__ = 'exc', '_loc'

    def __init__(self, exc: Exception, loc: Union[str, 'Loc']) -> None:
        self.exc = exc
        self._loc = loc

    def loc_tuple(self) -> 'Loc':
        if isinstance(self._loc, tuple):
            return self._loc
        else:
            return (self._loc,)

    def __repr_args__(self) -> 'ReprArgs':
        return [('exc', self.exc), ('loc', self.loc_tuple())]
    

class WrongColumnsError(TypeError):
    """Validation exception for column name mismatches."""


class MissingColumnsError(WrongColumnsError):
    """Exception for when a dataframe is missing one or more columns."""


class SuperflousColumnsError(WrongColumnsError):
    """Exception for when a dataframe has one ore more non-specified columns."""


class MissingValuesError(ValueError):
    """Exception for when a dataframe has non-nullable columns with nulls."""


class ColumnDTypeError(TypeError):
    """Exception for when a dataframe has one or more columns with wrong dtypes."""


class RowValueError(ValueError):
    """Exception for when a dataframe has a row with a impermissible value."""


class RowDoesNotExist(RuntimeError):
    """Exception for when a single row was expected, but none were returned."""


class MultipleRowsReturned(RuntimeError):
    """Exception for when a single row was expected, but several were returned."""
