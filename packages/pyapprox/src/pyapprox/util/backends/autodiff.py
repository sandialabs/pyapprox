"""Backends able to synthesize derivatives via automatic differentiation.

Semantic contract: a backend opts into autograd dispatch by exposing
methods named ``jacobian`` and ``hvp`` with the signatures below — the
names are a contract, not a coincidence. A backend without automatic
differentiation must NOT define methods with these names, because
``isinstance(bkd, AutodiffBackend)`` (``runtime_checkable``) keys purely
on their presence. ``TorchBkd`` conforms structurally with no changes; a
future JAX backend participates by defining the same two methods.

Typing note: ``Array`` stays invariant here because it appears in both
argument and return positions. A future protocol whose methods are
genuinely return-only must declare a COVARIANT type variable instead —
never add a dummy argument to silence mypy's variance check.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class AutodiffBackend(Backend[Array], Protocol):
    """A Backend that can differentiate array-valued callables."""

    def jacobian(
        self, fun: Callable[[Array], Array], params: Array
    ) -> Array:
        """Jacobian of ``fun`` at ``params``.

        For ``fun`` mapping shape ``(n,)`` to ``(m,)`` the result has
        shape ``(m, n)``.
        """
        ...

    def hvp(
        self,
        fun: Callable[[Array], Array],
        params: Array,
        vec: Array,
    ) -> Array:
        """Hessian-vector product of scalar-valued ``fun`` at ``params``.

        For ``fun`` mapping shape ``(n,)`` to a scalar the result has
        shape ``(n,)``. Requires a twice-differentiable computation
        graph.
        """
        ...
