"""Automatic differentiation as a Derivatives-bundle source.

Composition replaces mutation: instead of monkey-patching
``loss.jacobian = ...`` onto an instance, wrap it
(:class:`WithAutogradJacobian`) or build the bundle explicitly
(:func:`autograd_derivatives`). Policy: framework-owned classes use the
automatic fallback (analytic -> autograd if
``isinstance(bkd, AutodiffBackend)`` -> empty); user classes opt in with
one line in ``__init__``.

This module never imports torch: it reaches autograd purely through the
:class:`~pyapprox.util.backends.autodiff.AutodiffBackend` protocol, so a
future JAX backend participates with zero new code here. All bundle
fields built here are module-level callable objects (not closures) so
they stay picklable, e.g. for multiprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class _FlatFunction(Generic[Array]):
    """(nvars,) -> (nqoi,) view of a FunctionProtocol-shaped callable."""

    fun: Callable[[Array], Array]

    def __call__(self, flat_sample: Array) -> Array:
        return self.fun(flat_sample[:, None])[:, 0]


@dataclass(frozen=True)
class _ScalarFunction(Generic[Array]):
    """(nvars,) -> scalar view of a FunctionProtocol-shaped callable."""

    fun: Callable[[Array], Array]

    def __call__(self, flat_sample: Array) -> Array:
        return self.fun(flat_sample[:, None])[0, 0]


@dataclass(frozen=True)
class _AutogradJacobian(Generic[Array]):
    """Bundle jacobian field computed via backend autodiff."""

    fun: Callable[[Array], Array]
    bkd: AutodiffBackend[Array]

    def __call__(self, sample: Array) -> Array:
        # (nvars,) -> (nqoi,) so bkd.jacobian returns (nqoi, nvars)
        return self.bkd.jacobian(_FlatFunction(self.fun), sample[:, 0])


@dataclass(frozen=True)
class _AutogradHVP(Generic[Array]):
    """Bundle hvp field computed via backend autodiff (nqoi == 1)."""

    fun: Callable[[Array], Array]
    bkd: AutodiffBackend[Array]

    def __call__(self, sample: Array, vec: Array) -> Array:
        return self.bkd.hvp(
            _ScalarFunction(self.fun), sample[:, 0], vec[:, 0]
        )[:, None]


def autograd_derivatives(
    fun: Callable[[Array], Array],
    bkd: AutodiffBackend[Array],
    *,
    fill_hvp: bool = False,
) -> Derivatives[Array]:
    """Build a bundle whose fields differentiate ``fun`` via autograd.

    Parameters
    ----------
    fun : Callable[[Array], Array]
        Evaluation map with FunctionProtocol shapes: ``(nvars, nsamples)``
        -> ``(nqoi, nsamples)``. Must be built from backend operations so
        the computation graph is preserved.
    bkd : AutodiffBackend[Array]
        Backend providing ``jacobian`` (and ``hvp`` when
        ``fill_hvp=True``).
    fill_hvp : bool
        Opt-in because second order requires a TWICE-differentiable
        computation graph, which not all first-order-differentiable code
        provides: notably ``torch.cdist`` does not support second-order
        autograd, so any evaluation path through it (e.g. distance-based
        kernels) must leave ``fill_hvp`` False. Only valid for
        ``nqoi == 1``.

    Returns
    -------
    Derivatives[Array]
        Bundle with ``jacobian`` (and optionally ``hvp``) populated.
    """
    if not isinstance(bkd, AutodiffBackend):
        raise TypeError(
            "bkd must satisfy AutodiffBackend (methods named 'jacobian' "
            f"and 'hvp' are required), got {type(bkd).__name__}"
        )
    if not fill_hvp:
        return Derivatives.first_order(jacobian=_AutogradJacobian(fun, bkd))
    return Derivatives.second_order(
        _AutogradJacobian(fun, bkd), _AutogradHVP(fun, bkd)
    )


class WithAutogradJacobian(Generic[Array]):
    """Wrapper delegating evaluation; bundle is the inner bundle with the
    jacobian field filled from backend autodiff.

    Replaces instance monkey-patching in the GP fitters. The wrapper is
    itself an ``ObjectiveProtocol``; the inner object's other capabilities
    pass through unchanged.
    """

    def __init__(
        self, inner: ObjectiveProtocol[Array], bkd: AutodiffBackend[Array]
    ) -> None:
        if not isinstance(inner, ObjectiveProtocol):
            raise TypeError(
                "inner must satisfy ObjectiveProtocol, got "
                f"{type(inner).__name__}"
            )
        if not isinstance(bkd, AutodiffBackend):
            raise TypeError(
                "bkd must satisfy AutodiffBackend (methods named "
                "'jacobian' and 'hvp' are required), got "
                f"{type(bkd).__name__}"
            )
        self._inner = inner
        self._grad_bkd = bkd
        self._derivs: Derivatives[Array] = inner.derivatives().with_(
            jacobian=_AutogradJacobian(inner, bkd)
        )

    def bkd(self) -> Backend[Array]:
        return self._inner.bkd()

    def nvars(self) -> int:
        return self._inner.nvars()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def __call__(self, samples: Array) -> Array:
        return self._inner(samples)

    def derivatives(self) -> Derivatives[Array]:
        return self._derivs
