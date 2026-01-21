"""Manufactured solutions for linear elasticity equations.

Provides manufactured solution class for verifying 2D linear elasticity
physics implementations.

Linear Elasticity equations:
    -div(σ) + f = 0

where:
    σ = λ*tr(ε)*I + 2μ*ε  (stress tensor)
    ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)  (strain tensor)
    u = (u, v) is the displacement field
"""

from typing import Generic, List

import sympy as sp

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    VectorSolutionMixin,
)


class ManufacturedLinearElasticityEquations(
    VectorSolutionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for 2D linear elasticity equations.

    Solves: -div(σ) + f = 0

    where:
        σ = λ*tr(ε)*I + 2μ*ε  (stress tensor)
        ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)  (strain tensor)

    The forcing f is computed from the manufactured solution to satisfy the PDE.

    Parameters
    ----------
    sol_strs : List[str]
        String representations of the exact solution components [u, v].
        May contain 'x', 'y' for spatial coordinates and 'T' for time.
    nvars : int
        Number of spatial dimensions (must be 2).
    lambda_str : str
        String representation of Lamé's first parameter λ.
    mu_str : str
        String representation of shear modulus μ.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 2D manufactured solution: u = (1-x**2)*(1-y**2), v = x*y*(1-x)*(1-y)
    >>> man_sol = ManufacturedLinearElasticityEquations(
    ...     sol_strs=["(1-x**2)*(1-y**2)", "x*y*(1-x)*(1-y)"],
    ...     nvars=2,
    ...     lambda_str="1.0",
    ...     mu_str="1.0",
    ...     bkd=bkd,
    ... )
    >>> # Get forcing function
    >>> x = bkd.linspace(-1, 1, 10)
    >>> y = bkd.linspace(-1, 1, 10)
    >>> xx, yy = bkd.meshgrid(x, y, indexing='xy')
    >>> nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)
    >>> forcing = man_sol.functions["forcing"](nodes)
    """

    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        lambda_str: str,
        mu_str: str,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        if nvars != 2:
            raise ValueError("Linear elasticity requires nvars=2")
        if len(sol_strs) != 2:
            raise ValueError("Linear elasticity requires 2 solution components [u, v]")

        self._lambda_str = lambda_str
        self._mu_str = mu_str
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build sympy expressions for linear elasticity equation."""
        cartesian_symbs = self.cartesian_symbols()
        x, y = cartesian_symbs[0], cartesian_symbs[1]

        # Lamé parameters
        lambda_expr = sp.sympify(self._lambda_str)
        mu_expr = sp.sympify(self._mu_str)

        self._set_expression("lambda", lambda_expr, self._lambda_str)
        self._set_expression("mu", mu_expr, self._mu_str)

        # Displacement field
        disp_expr = self._expressions["solution"]
        u_expr = disp_expr[0]
        v_expr = disp_expr[1]

        # Strain tensor components: ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)
        exx = u_expr.diff(x)
        exy = sp.Rational(1, 2) * (u_expr.diff(y) + v_expr.diff(x))
        eyy = v_expr.diff(y)

        # Trace of strain
        trace_e = exx + eyy

        # Stress tensor: σ = λ*tr(ε)*I + 2μ*ε
        two_mu = 2 * mu_expr
        sigma_xx = lambda_expr * trace_e + two_mu * exx
        sigma_xy = two_mu * exy
        sigma_yy = lambda_expr * trace_e + two_mu * eyy

        # Store stress tensor (flux)
        tau = [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]]
        self._set_expression("flux", tau, self._sol_strs[0])

        # Compute divergence of stress tensor
        # div(σ)_i = ∂σ_i1/∂x + ∂σ_i2/∂y
        div_sigma_x = sigma_xx.diff(x) + sigma_xy.diff(y)
        div_sigma_y = sigma_xy.diff(x) + sigma_yy.diff(y)

        # Forcing: f = -div(σ) so that div(σ) + f = 0
        # The physics residual is div(σ) + f, so for residual=0 we need f = -div(σ)
        forc_exprs = [-div_sigma_x, -div_sigma_y]

        # Add forcing contribution to existing forcing (initialized to zeros)
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]
