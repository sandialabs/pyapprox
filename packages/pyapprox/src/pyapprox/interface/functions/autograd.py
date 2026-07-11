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
future JAX backend participates with zero new code here.
"""

from __future__ import annotations

from typing import Callable, Generic

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend


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

    def _flat_fun(flat_sample: Array) -> Array:
        # (nvars,) -> (nqoi,) so bkd.jacobian returns (nqoi, nvars)
        return fun(flat_sample[:, None])[:, 0]

    def _jacobian(sample: Array) -> Array:
        return bkd.jacobian(_flat_fun, sample[:, 0])

    if not fill_hvp:
        return Derivatives.first_order(jacobian=_jacobian)

    def _scalar_fun(flat_sample: Array) -> Array:
        return fun(flat_sample[:, None])[0, 0]

    def _hvp(sample: Array, vec: Array) -> Array:
        return bkd.hvp(_scalar_fun, sample[:, 0], vec[:, 0])[:, None]

    return Derivatives.second_order(_jacobian, _hvp)


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
            jacobian=self._jacobian
        )

    def bkd(self) -> Backend[Array]:
        return self._inner.bkd()

    def nvars(self) -> int:
        return self._inner.nvars()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def __call__(self, samples: Array) -> Array:
        return self._inner(samples)

    def _jacobian(self, sample: Array) -> Array:
        def _flat_fun(flat_sample: Array) -> Array:
            return self._inner(flat_sample[:, None])[:, 0]

        return self._grad_bkd.jacobian(_flat_fun, sample[:, 0])

    def derivatives(self) -> Derivatives[Array]:
        return self._derivs
