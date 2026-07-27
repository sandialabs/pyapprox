"""
Tikhonov augmentation of transient functionals.

Wraps any transient functional Q(y, p) as
:math:`Q(y, p) + (\\alpha/2)\\, p^\\top W_q\\, p` (:math:`W_q = I` by
default), adding a direct parameter dependence.
"""

from typing import Generic, Optional

from pyapprox.ode.functionals.protocols import (
    TimeQuadratureAwareFunctionalProtocol,
    TransientFunctionalWithJacobianAndHVPProtocol,
    TransientFunctionalWithJacobianProtocol,
)
from pyapprox.ode.time_quadrature import TrajectoryQuadratureProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class TikhonovAugmentedFunctional(Generic[Array]):
    """
    Augment a transient functional with a quadratic parameter cost.

    Computes :math:`Q(y, p) + (\\alpha/2)\\, p^\\top W_q\\, p` where Q is
    any transient functional. State derivatives delegate to the inner
    functional; the parameter derivatives gain :math:`\\alpha W_q p`
    (Jacobian) and :math:`\\alpha W_q v` (HVP). Second-order methods are
    bound only when the inner functional provides them, so the wrapper
    sits at the same protocol tier as what it wraps.

    Parameters
    ----------
    inner : TransientFunctionalWithJacobianProtocol
        The functional to augment. Must have ``nqoi() == 1``.
    alpha : float
        Cost coefficient (exchange rate between Q and parameter effort).
    bkd : Backend
        Backend for array operations.
    weight : Array, optional
        Symmetric positive definite weight :math:`W_q`. Shape:
        ``(nparams, nparams)``. Identity when omitted.
    """

    def __init__(
        self,
        inner: TransientFunctionalWithJacobianProtocol[Array],
        alpha: float,
        bkd: Backend[Array],
        weight: Optional[Array] = None,
    ):
        validate_backend(bkd)
        if not isinstance(inner, TransientFunctionalWithJacobianProtocol):
            raise TypeError(
                "inner must satisfy "
                "TransientFunctionalWithJacobianProtocol, got "
                f"{type(inner).__name__}"
            )
        if inner.nqoi() != 1:
            raise ValueError(
                f"inner must have nqoi() == 1, got {inner.nqoi()}: adding "
                "a scalar parameter cost to a multi-QoI functional is "
                "ambiguous"
            )
        if weight is not None:
            nparams = inner.nparams()
            if weight.ndim != 2 or weight.shape != (nparams, nparams):
                raise ValueError(
                    f"weight must have shape ({nparams}, {nparams}), got "
                    f"{weight.shape}"
                )
            asymmetry = bkd.max(bkd.abs(weight - weight.T))
            scale = bkd.max(bkd.abs(weight))
            if bkd.to_float(asymmetry) > 1e-12 * max(
                bkd.to_float(scale), 1.0
            ):
                raise ValueError("weight must be symmetric")
        self._inner = inner
        self._alpha = alpha
        self._weight = weight
        self._bkd = bkd
        # Bind second-order methods only when the inner functional has
        # them, so isinstance checks report the wrapped tier; likewise
        # forward time-quadrature injection to a quadrature-aware inner.
        # The narrowed references keep the tier information mypy-visible.
        self._hvp_inner: Optional[
            TransientFunctionalWithJacobianAndHVPProtocol[Array]
        ] = None
        if isinstance(inner, TransientFunctionalWithJacobianAndHVPProtocol):
            self._hvp_inner = inner
            self.state_state_hvp = self._state_state_hvp
            self.state_param_hvp = self._state_param_hvp
            self.param_state_hvp = self._param_state_hvp
            self.param_param_hvp = self._param_param_hvp
        self._quadrature_inner: Optional[
            TimeQuadratureAwareFunctionalProtocol[Array]
        ] = None
        if isinstance(inner, TimeQuadratureAwareFunctionalProtocol):
            self._quadrature_inner = inner
            self.set_time_quadrature = self._set_time_quadrature

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        return self._inner.nqoi()

    def nstates(self) -> int:
        """Return the number of state variables."""
        return self._inner.nstates()

    def nparams(self) -> int:
        """Return the total number of parameters."""
        return self._inner.nparams()

    def nunique_params(self) -> int:
        """Return the number of parameters unique to the functional."""
        return self._inner.nunique_params()

    def inner(self) -> TransientFunctionalWithJacobianProtocol[Array]:
        """Return the wrapped functional."""
        return self._inner

    def alpha(self) -> float:
        """Return the cost coefficient."""
        return self._alpha

    def _require_hvp_inner(
        self,
    ) -> TransientFunctionalWithJacobianAndHVPProtocol[Array]:
        if self._hvp_inner is None:
            raise RuntimeError(
                "HVP method called but the inner functional is not "
                "HVP-tier; these methods are only bound when it is"
            )
        return self._hvp_inner

    def _set_time_quadrature(
        self, quadrature: TrajectoryQuadratureProtocol[Array]
    ) -> None:
        """Forward the scheme-implied quadrature to the inner functional."""
        if self._quadrature_inner is None:
            raise RuntimeError(
                "set_time_quadrature called but the inner functional is "
                "not quadrature-aware; the method is only bound when it is"
            )
        self._quadrature_inner.set_time_quadrature(quadrature)

    def _weighted(self, vec: Array) -> Array:
        """Return :math:`W_q v` (identity weight when W_q was omitted)."""
        if self._weight is None:
            return vec
        return self._bkd.dot(self._weight, vec)

    def __call__(self, sol: Array, param: Array) -> Array:
        """
        Evaluate :math:`Q(y, p) + (\\alpha/2) p^\\top W_q p`.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Augmented QoI. Shape: (1, 1)
        """
        cost = 0.5 * self._alpha * self._bkd.sum(
            param * self._weighted(param)
        )
        return self._inner(sol, param) + self._bkd.reshape(cost, (1, 1))

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """Compute dQ/dy; the parameter cost contributes nothing."""
        return self._inner.state_jacobian(sol, param)

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dp, adding :math:`\\alpha (W_q p)^\\top`.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (1, nparams)
        """
        return self._inner.param_jacobian(
            sol, param
        ) + self._alpha * self._weighted(param).T

    def _state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """Compute (d^2Q/dy^2)·w; the parameter cost contributes nothing."""
        return self._require_hvp_inner().state_state_hvp(
            sol, param, time_idx, wvec
        )

    def _state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """Compute (d^2Q/dy dp)·v; the parameter cost contributes nothing."""
        return self._require_hvp_inner().state_param_hvp(
            sol, param, time_idx, vvec
        )

    def _param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """Compute (d^2Q/dp dy)·w; the parameter cost contributes nothing."""
        return self._require_hvp_inner().param_state_hvp(
            sol, param, time_idx, wvec
        )

    def _param_param_hvp(
        self, sol: Array, param: Array, vvec: Array
    ) -> Array:
        """Compute (d^2Q/dp^2)·v, adding :math:`\\alpha W_q v`."""
        return self._require_hvp_inner().param_param_hvp(
            sol, param, vvec
        ) + self._alpha * self._weighted(vvec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"inner={self._inner!r}, "
            f"alpha={self._alpha}, "
            f"weighted={self._weight is not None})"
        )
