"""Typed wrappers for numba decorators.

numba ships no type stubs, so ``@njit`` erases the decorated function's
signature under strict mypy (untyped-decorator) and every call returns
``Any``. These wrappers declare that ``njit`` preserves the wrapped
function's signature, which matches runtime behavior for the plain and
keyword-argument forms used in this codebase.

This module imports numba at module level, so it must only be imported
from ``*_numba.py`` modules that are themselves lazily imported by
dispatch code (see the lazy-import rule in CLAUDE.md).
"""

from typing import Any, Callable, TypeVar, overload

from pyapprox.util.openmp import keep_numba_off_openmp
from pyapprox.util.optional_deps import missing_dependency_message

# Every *_numba.py kernel imports numba through this module, so this one
# guard gives all of them the install message.
try:
    from numba import njit as _numba_njit
    from numba import prange as _numba_prange
except ImportError as err:
    raise ImportError(
        missing_dependency_message("numba", "The Numba kernels (*_numba.py)", "numba")
    ) from err

F = TypeVar("F", bound=Callable[..., Any])

# Every numba kernel in the library is defined through this module, so this
# runs before any of them launches threads (numba fixes its threading layer
# at the first parallel launch).
keep_numba_off_openmp()


@overload
def njit(func: F, /) -> F: ...


@overload
def njit(*args: Any, **kwargs: Any) -> Callable[[F], F]: ...


def njit(*args: Any, **kwargs: Any) -> Any:
    """Signature-preserving ``numba.njit`` (bare or parametrized form)."""
    return _numba_njit(*args, **kwargs)


# prange must be numba's own object, only re-exported with a type: the
# nopython compiler resolves it by value inside jitted functions, so a
# Python wrapper function would break compilation at runtime.
prange: Callable[..., range] = _numba_prange
