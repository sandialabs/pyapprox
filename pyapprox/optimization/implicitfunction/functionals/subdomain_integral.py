"""Subdomain integral functional for collocation-based PDE solutions.

Computes Q(u) = integral_a^b c(x)*u(x) dx (linear) or
Q(u) = integral_a^b g(u(x), x) dx (nonlinear) using quadrature weights
projected to collocation nodes.
"""

from typing import Callable, Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)


class SubdomainIntegralFunctional(Generic[Array]):
    """Integrate a function of the PDE state over a subdomain.

    Supports two modes:

    **Linear mode** (``coefficient`` provided): Computes
    ``Q(u) = integral_a^b c(x) * u(x) dx`` where ``c`` is a fixed
    coefficient array. The state Jacobian is constant.

    **Nonlinear mode** (``integrand`` provided): Computes
    ``Q(u) = integral_a^b g(u(x)) dx`` where ``g`` is a user-supplied
    callable returning both function values and their derivatives.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    nparams : int
        Number of parameters in the forward model.
    bkd : Backend
        Computational backend.
    a_sub : float, optional
        Left endpoint of integration domain (physical coordinates).
        If None, integrates over the full domain.
    b_sub : float, optional
        Right endpoint of integration domain (physical coordinates).
        If None, integrates over the full domain.
    coefficient : Array, optional
        Coefficient c(x) at collocation nodes, shape ``(npts,)``.
        Mutually exclusive with ``integrand``.
    integrand : callable, optional
        Nonlinear integrand ``g(state_1d, bkd) -> (values, dvalues_dstate)``
        where both arrays have shape ``(npts,)``.
        Mutually exclusive with ``coefficient``.
    """

    def __init__(
        self,
        basis: ChebyshevBasis1D[Array],
        nparams: int,
        bkd: Backend[Array],
        a_sub: Optional[float] = None,
        b_sub: Optional[float] = None,
        coefficient: Optional[Array] = None,
        integrand: Optional[
            Callable[[Array, Backend[Array]], Tuple[Array, Array]]
        ] = None,
    ) -> None:
        if coefficient is None and integrand is None:
            raise ValueError(
                "Exactly one of coefficient or integrand must be provided"
            )
        if coefficient is not None and integrand is not None:
            raise ValueError(
                "Exactly one of coefficient or integrand must be provided"
            )

        self._bkd = bkd
        self._nparams = nparams
        self._coefficient = coefficient
        self._integrand = integrand

        # Compute quadrature weights
        quad = CollocationQuadrature1D(basis, bkd)
        if a_sub is None and b_sub is None:
            self._weights = quad.full_domain_weights()
        else:
            if a_sub is None or b_sub is None:
                raise ValueError(
                    "Both a_sub and b_sub must be provided, or both None"
                )
            self._weights = quad.weights(a_sub, b_sub)

        self._nstates = self._weights.shape[0]

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return self._nstates

    def nparams(self) -> int:
        return self._nparams

    def nunique_params(self) -> int:
        return 0

    def __call__(self, state: Array, param: Array) -> Array:
        """Evaluate the integral Q(u).

        Parameters
        ----------
        state : Array, shape (nstates, 1)
            PDE state vector.
        param : Array, shape (nparams, 1)
            Parameters (unused).

        Returns
        -------
        Array, shape (1, 1)
            Integral value.
        """
        bkd = self._bkd
        u = state[:, 0]  # shape (npts,)
        if self._coefficient is not None:
            val = bkd.sum(self._weights * self._coefficient * u)
        else:
            assert self._integrand is not None
            values, _ = self._integrand(u, bkd)
            val = bkd.sum(self._weights * values)
        return bkd.reshape(val, (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/du.

        For linear mode, this is constant. For nonlinear mode, depends
        on the current state.

        Returns
        -------
        Array, shape (1, nstates)
        """
        bkd = self._bkd
        if self._coefficient is not None:
            return (self._weights * self._coefficient)[None, :]
        else:
            assert self._integrand is not None
            u = state[:, 0]
            _, dvalues = self._integrand(u, bkd)
            return (self._weights * dvalues)[None, :]

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Return dQ/dp = 0 (no parameter dependence).

        Returns
        -------
        Array, shape (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))

    def __repr__(self) -> str:
        mode = "linear" if self._coefficient is not None else "nonlinear"
        return (
            f"{self.__class__.__name__}("
            f"mode={mode}, "
            f"nstates={self._nstates}, "
            f"nparams={self._nparams}, "
            f"bkd={type(self._bkd).__name__})"
        )
