import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    Callable,
    TypedDict,
    Iterable,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    Loc = Tuple[Union[int, str], ...]

    class _ErrorDictRequired(TypedDict):
        loc: Loc
        msg: str
        type: str

    class ErrorDict(_ErrorDictRequired, total=False):
        ctx: Dict[str, Any]

    Loc = Tuple[Union[int, str], ...]
    ReprArgs = Sequence[Tuple[Optional[str], Any]]
    RichReprResult = Iterable[
        Union[Any, Tuple[Any], Tuple[str, Any], Tuple[str, Any, Any]]
    ]


__all__ = "ErrorWrapper", "ValidationError"


class Representation:
    """
    Mixin to provide __str__, __repr__, and __pretty__ methods. See #884 for more details.

    __pretty__ is used by [devtools](https://python-devtools.helpmanual.io/) to provide human readable representations
    of objects.
    """

    __slots__: Tuple[str, ...] = tuple()

    def __repr_args__(self) -> "ReprArgs":
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
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}" for a, v in self.__repr_args__()
        )

    def __pretty__(
        self, fmt: Callable[[Any], Any], **kwargs: Any
    ) -> Generator[Any, None, None]:
        """
        Used by devtools (https://python-devtools.helpmanual.io/) to provide a human readable representations of objects
        """
        yield self.__repr_name__() + "("
        yield 1
        for name, value in self.__repr_args__():
            if name is not None:
                yield name + "="
            yield fmt(value)
            yield ","
            yield 0
        yield -1
        yield ")"

    def __str__(self) -> str:
        return self.__repr_str__(" ")

    def __repr__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'

    def __rich_repr__(self) -> "RichReprResult":
        """Get fields for Rich library"""
        for name, field_repr in self.__repr_args__():
            if name is None:
                yield field_repr
            else:
                yield name, field_repr


class ErrorWrapper(Representation):
    __slots__ = "exc", "_loc"

    def __init__(self, exc: Exception, loc: Union[str, "Loc"]) -> None:
        self.exc = exc
        self._loc = loc

    def loc_tuple(self) -> "Loc":
        if isinstance(self._loc, tuple):
            return self._loc
        else:
            return (self._loc,)

    def __repr_args__(self) -> "ReprArgs":
        return [("exc", self.exc), ("loc", self.loc_tuple())]


# ErrorList is something like Union[List[Union[List[ErrorWrapper], ErrorWrapper]], ErrorWrapper]
# but recursive, therefore just use:
ErrorList = Union[Sequence[Any], ErrorWrapper]


class DataFrameValidationError(Representation, ValueError):
    __slots__ = "raw_errors", "model", "_error_cache"

    def __init__(self, errors: Sequence[ErrorList], model: Type["BaseModel"]) -> None:
        self.raw_errors = errors
        self.model = model
        self._error_cache: Optional[List["ErrorDict"]] = None

    def errors(self) -> List["ErrorDict"]:
        if self._error_cache is None:
            self._error_cache = list(flatten_errors(self.raw_errors))
        return self._error_cache

    def __str__(self) -> str:
        errors = self.errors()
        no_errors = len(errors)
        return (
            f'{no_errors} validation error{"" if no_errors == 1 else "s"} for {self.model.__name__}\n'
            f"{display_errors(errors)}"
        )

    def __repr_args__(self) -> "ReprArgs":
        return [("model", self.model.__name__), ("errors", self.errors())]


def display_errors(errors: List["ErrorDict"]) -> str:
    return "\n".join(
        f'{_display_error_loc(e)}\n  {e["msg"]} ({_display_error_type_and_ctx(e)})'
        for e in errors
    )


def _display_error_loc(error: "ErrorDict") -> str:
    return " -> ".join(str(e) for e in error["loc"])


def _display_error_type_and_ctx(error: "ErrorDict") -> str:
    t = "type=" + error["type"]
    ctx = error.get("ctx")
    if ctx:
        return t + "".join(f"; {k}={v}" for k, v in ctx.items())
    else:
        return t


def flatten_errors(
    errors: Sequence[Any], loc: Optional["Loc"] = None
) -> Generator["ErrorDict", None, None]:
    for error in errors:
        if isinstance(error, ErrorWrapper):
            if loc:
                error_loc = loc + error.loc_tuple()
            else:
                error_loc = error.loc_tuple()

            if isinstance(error.exc, DataFrameValidationError):
                yield from flatten_errors(error.exc.raw_errors, error_loc)
            else:
                yield error_dict(error.exc, error_loc)
        elif isinstance(error, list):
            yield from flatten_errors(error, loc=loc)
        else:
            raise RuntimeError(f"Unknown error object: {error}")


def error_dict(exc: Exception, loc: "Loc") -> "ErrorDict":
    type_ = get_exc_type(exc.__class__)
    msg_template = getattr(exc, "msg_template", None)
    ctx = exc.__dict__
    if msg_template:
        msg = msg_template.format(**ctx)
    else:
        msg = str(exc)

    d: "ErrorDict" = {"loc": loc, "msg": msg, "type": type_}

    if ctx:
        d["ctx"] = ctx

    return d


_EXC_TYPE_CACHE: Dict[Type[Exception], str] = {}


def get_exc_type(cls: Type[Exception]) -> str:
    # slightly more efficient than using lru_cache since we don't need to worry about the cache filling up
    try:
        return _EXC_TYPE_CACHE[cls]
    except KeyError:
        r = _get_exc_type(cls)
        _EXC_TYPE_CACHE[cls] = r
        return r


def _get_exc_type(cls: Type[Exception]) -> str:
    if issubclass(cls, AssertionError):
        return "assertion_error"

    base_name = "type_error" if issubclass(cls, TypeError) else "value_error"
    if cls in (TypeError, ValueError):
        # just TypeError or ValueError, no extra code
        return base_name

    # if it's not a TypeError or ValueError, we just take the lowercase of the exception name
    # no chaining or snake case logic, use "code" for more complex error types.
    code = getattr(cls, "code", None) or cls.__name__.replace("Error", "").lower()
    return base_name + "." + code


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
