"""Minimal NVTX-compatible no-op API for environments without NVTX bindings."""

from contextlib import ContextDecorator
from typing import Any, Callable, TypeVar, Union


Function = TypeVar("Function", bound=Callable[..., Any])


class _Annotate(ContextDecorator):
    def __enter__(self) -> "_Annotate":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def annotate(
    message: Union[str, Function] = "",
    **_: Any,
) -> Union[_Annotate, Function]:
    if callable(message):
        return message
    return _Annotate()


def range_push(*_: Any, **__: Any) -> int:
    return 0


def range_pop(*_: Any, **__: Any) -> None:
    return None


def mark(*_: Any, **__: Any) -> None:
    return None
