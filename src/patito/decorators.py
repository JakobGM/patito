import inspect
import typing
from functools import wraps
from typing import Any, Callable, TypeVar

import patito as pt


T = TypeVar("T")

def validate_hints(wrapped: Callable[..., T]) -> Callable[..., T]:
    """Validate function arguments and return on pt.Model type hints
    
    :param wrapped: the function to decorate
    """
    def _validate_or_skip(validator: Any, target: Any) -> None:
        if not isinstance(validator, pt.pydantic.ModelMetaclass):
            return

        validator.validate(target)

    @wraps(wrapped)
    def wrapper(*args, **kwargs) -> T:
        type_hints = typing.get_type_hints(wrapped)
        signature = inspect.signature(wrapped)

        for arg_label, arg in zip(signature.parameters.keys(), args):
            if arg_label in type_hints:
                _validate_or_skip(type_hints[arg_label], arg)

        result = wrapped(*args, **kwargs)

        if "return" in type_hints:
            _validate_or_skip(type_hints["return"], result)

        return result

    return wrapper
