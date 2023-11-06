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

from patito._pydantic.repr import Representation

if TYPE_CHECKING:
    from pydantic import BaseModel

    Loc = Tuple[Union[int, str], ...]

    class _ErrorDictRequired(TypedDict):
        loc: Loc
        msg: str
        type: str

    class ErrorDict(_ErrorDictRequired, total=False):
        ctx: Dict[str, Any]

    from patito._pydantic.repr import ReprArgs


__all__ = "ErrorWrapper", "DataFrameValidationError"


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
