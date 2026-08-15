"""Minimal local type stubs for the joblib parallel bindings.

Only the surface pyapprox touches is declared; everything else falls back
to Any via module-level __getattr__. joblib ships no py.typed marker, so
without these stubs ``Parallel(...)(...)`` returns Any into a declared
List[T].

``delayed`` is modelled as preserving its function's return type through
to ``Parallel.__call__``, which is what makes the list element type
recoverable. joblib really returns a _DelayedFunction wrapper whose call
produces a (func, args, kwargs) tuple; that indirection is deliberately
not modelled, since nothing in pyapprox inspects it and doing so would
lose the type link the call site needs.
"""

from typing import Any, Callable, Iterable, List, TypeVar

_T = TypeVar("_T")

def __getattr__(name: str) -> Any: ...

class Parallel:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, iterable: Iterable[_T]) -> List[_T]: ...

def delayed(function: Callable[..., _T]) -> Callable[..., _T]: ...
