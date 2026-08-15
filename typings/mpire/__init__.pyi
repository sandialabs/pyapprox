"""Minimal local type stubs for the mpire parallel-map bindings.

Only the surface pyapprox touches is declared; everything else falls back
to Any via module-level __getattr__. mpire ships no py.typed marker, so
without these stubs ``WorkerPool`` is an implicit-reexport error and
``map`` returns Any into a declared List[T].

``map`` is typed as generic in its element type rather than mirroring
mpire's full overload set: pyapprox calls it two ways -- with 1-tuples
that mpire unpacks, and with argument tuples it unpacks like starmap --
and both want the same "returns a list of whatever func returns"
guarantee. The looser first parameter keeps both call sites honest at
the return type, which is the part pyapprox depends on.
"""

from types import TracebackType
from typing import Any, Callable, Iterable, List, Optional, Type, TypeVar

_T = TypeVar("_T")

def __getattr__(name: str) -> Any: ...

class WorkerPool:
    def __init__(
        self,
        n_jobs: Optional[int] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def __enter__(self) -> "WorkerPool": ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...
    def map(
        self,
        func: Callable[..., _T],
        iterable_of_args: Iterable[Any],
        *args: Any,
        **kwargs: Any,
    ) -> List[_T]: ...
