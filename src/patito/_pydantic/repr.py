import sys
import types
import typing
from collections.abc import Generator, Iterable, Sequence
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
)
from typing import GenericAlias as TypingGenericAlias  # type: ignore

if typing.TYPE_CHECKING:
    Loc = tuple[Union[int, str], ...]
    ReprArgs = Sequence[tuple[Optional[str], Any]]
    RichReprResult = Iterable[
        Union[Any, tuple[Any], tuple[str, Any], tuple[str, Any, Any]]
    ]

try:
    from typing import _TypingBase  # type: ignore[attr-defined]
except ImportError:
    from typing import _Final as _TypingBase  # type: ignore[attr-defined]

typing_base = _TypingBase


if sys.version_info < (3, 10):

    def origin_is_union(tp: Optional[type[Any]]) -> bool:
        return tp is typing.Union

    WithArgsTypes = (TypingGenericAlias,)

else:

    def origin_is_union(tp: type[Any] | None) -> bool:
        return tp is typing.Union or tp is types.UnionType

    WithArgsTypes = typing._GenericAlias, types.GenericAlias, types.UnionType  # type: ignore[attr-defined]


class Representation:
    """Mixin to provide __str__, __repr__, and __pretty__ methods. See #884 for more details.

    __pretty__ is used by [devtools](https://python-devtools.helpmanual.io/) to provide human readable representations
    of objects.
    """

    __slots__: tuple[str, ...] = tuple()

    def __repr_args__(self) -> "ReprArgs":
        """Returns the attributes to show in __str__, __repr__, and __pretty__ this is generally overridden.

        Can either return:
        * name - value pairs, e.g.: `[('foo_name', 'foo'), ('bar_name', ['b', 'a', 'r'])]`
        * or, just values, e.g.: `[(None, 'foo'), (None, ['b', 'a', 'r'])]`
        """
        attrs = ((s, getattr(self, s)) for s in self.__slots__)
        return [(a, v) for a, v in attrs if v is not None]

    def __repr_name__(self) -> str:
        """Name of the instance's class, used in __repr__."""
        return self.__class__.__name__

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}" for a, v in self.__repr_args__()
        )

    def __pretty__(
        self, fmt: Callable[[Any], Any], **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Used by devtools (https://python-devtools.helpmanual.io/) to provide a human readable representations of objects."""
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
        return f"{self.__repr_name__()}({self.__repr_str__(', ')})"

    def __rich_repr__(self) -> "RichReprResult":
        """Get fields for Rich library."""
        for name, field_repr in self.__repr_args__():
            if name is None:
                yield field_repr
            else:
                yield name, field_repr


def display_as_type(obj: Any) -> str:
    """Pretty representation of a type, should be as close as possible to the original type definition string.

    Takes some logic from `typing._type_repr`.
    """
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    elif obj is ...:
        return "..."
    elif isinstance(obj, Representation):
        return repr(obj)

    if not isinstance(obj, (typing_base, WithArgsTypes, type)):
        obj = obj.__class__

    if origin_is_union(get_origin(obj)):
        args = ", ".join(map(display_as_type, get_args(obj)))
        return f"Union[{args}]"
    elif isinstance(obj, WithArgsTypes):
        if get_origin(obj) == Literal:
            args = ", ".join(map(repr, get_args(obj)))
        else:
            args = ", ".join(map(display_as_type, get_args(obj)))
        return f"{obj.__qualname__}[{args}]"
    elif isinstance(obj, type):
        return obj.__qualname__
    else:
        return repr(obj).replace("typing.", "").replace("typing_extensions.", "")
