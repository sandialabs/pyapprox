"""Checks for tests that run under several precisions (``config_bkd``).

:func:`assert_close` compares within a multiple of the configured
precision's eps, so one assertion holds in float64 and float32.
:func:`no_dtype_leak` fails a torch computation that produces a float
wider than the configured precision anywhere inside, which an assertion
on the result misses when the result is cast back at the end.

Tests opt in as they need these; nothing is converted to them.
"""

from contextlib import contextmanager, nullcontext
from typing import Any, Callable, ContextManager, Iterator, List

from pyapprox.util.backends.protocols import Array, Backend


def assert_close(
    bkd: Backend[Array], actual: Array, desired: Array, factor: float = 10.0
) -> None:
    """``actual`` equals ``desired`` in shape and dtype, and in value within
    ``factor`` times the eps of ``bkd``'s configured precision (``rtol``
    and ``atol`` alike); integer and boolean arrays exactly.

    The dtype check stays strict, so a result in a wider dtype than
    configured fails here rather than passing at a loose tolerance.
    """
    exact = bkd.isdtype(desired, "integral") or bkd.isdtype(desired, "bool")
    tol = 0.0 if exact else factor * bkd.eps()
    bkd.assert_allclose(actual, desired, rtol=tol, atol=tol)


def no_dtype_leak(bkd: Backend[Array]) -> ContextManager[None]:
    """A context in which every torch operation's floating output must be
    no wider than ``bkd``'s precision: float32 or complex64 under single
    precision. On leaving it, raises ``AssertionError`` naming each
    operation that produced a wider float.

    Torch only: it records operations through torch's dispatcher, which
    NumPy has no counterpart of, so for any other backend the context
    checks nothing. It uses the private ``torch.utils._python_dispatch``,
    so it may need updating on a torch upgrade.
    """
    try:
        from pyapprox.util.backends.torch import TorchBkd
    except ImportError:
        return nullcontext()
    if not isinstance(bkd, TorchBkd):
        return nullcontext()
    return _torch_no_dtype_leak(bkd)


@contextmanager
def _torch_no_dtype_leak(bkd: Backend[Array]) -> Iterator[None]:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    limit = torch.finfo(bkd.real_dtype()).bits
    leaks: List[str] = []

    class _Recorder(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: Any = (),
            kwargs: Any = None,
        ) -> Any:
            out = func(*args, **(kwargs or {}))
            for leaf in tree_leaves(out):
                if not isinstance(leaf, torch.Tensor):
                    continue
                dtype = leaf.dtype
                if dtype.is_complex:
                    dtype = bkd.real_dtype_of(dtype)
                if dtype.is_floating_point and torch.finfo(dtype).bits > limit:
                    leaks.append(f"{func} -> {leaf.dtype}")
            return out

    # TorchDispatchMode's constructor is untyped in torch's stubs.
    recorder: Callable[[], ContextManager[None]] = _Recorder
    with recorder():
        yield
    if leaks:
        raise AssertionError(
            f"{len(leaks)} operations produced a float wider than "
            f"{bkd.real_dtype()}: " + "; ".join(sorted(set(leaks)))
        )
