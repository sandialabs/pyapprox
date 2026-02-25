"""Manufactured solutions for Shallow Ice Approximation (SIA).

Shallow Ice Approximation equation:
    dH/dt - div(D * grad(s)) = f

where:
    H = ice thickness (solution variable)
    s = H + b = surface elevation
    b = bed elevation
    D = nonlinear diffusion coefficient:
        D = γ * H^(n+2) * |grad(s)|^(n-1) + (ρg/C) * H²

    γ = 2A(ρg)^n / (n+2)
    n = Glen's flow law exponent (typically 3)
    A = rate factor
    ρ = ice density
    g = gravitational acceleration
    C = friction coefficient
"""

from typing import Generic

import sympy as sp

from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    ScalarSolutionMixin,
)
from pyapprox.util.backends.protocols import Array, Backend


class ManufacturedShallowIce(
    ScalarSolutionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for Shallow Ice Approximation.

    Solves: dH/dt - div(D * grad(s)) = f

    where D is a nonlinear diffusion coefficient depending on ice thickness
    and surface gradient.

    Parameters
    ----------
    sol_str : str
        String representation of ice thickness H.
        May contain 'x', 'y' for spatial coordinates and 'T' for time.
    nvars : int
        Number of spatial dimensions (1 or 2).
    bed_str : str
        String representation of bed elevation b.
    friction_str : str
        String representation of friction coefficient C.
    A : float
        Rate factor in Glen's flow law.
    rho : float
        Ice density (kg/m³).
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 1D manufactured solution: H = (1-x**2)
    >>> man_sol = ManufacturedShallowIce(
    ...     sol_str="1 + 0.5*(1 - x**2)",
    ...     nvars=1,
    ...     bed_str="0.1*x",
    ...     friction_str="1e6",
    ...     A=1e-16,
    ...     rho=917.0,
    ...     bkd=bkd,
    ... )
    >>> # Get forcing function
    >>> x = bkd.linspace(-1, 1, 10)
    >>> nodes = x.reshape(1, -1)
    >>> forcing = man_sol.functions["forcing"](nodes)
    """

    def __init__(
        self,
        sol_str: str,
        nvars: int,
        bed_str: str,
        friction_str: str,
        A: float,
        rho: float,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        self._bed_str = bed_str
        self._friction_str = friction_str
        self._A = A
        self._rho = rho
        self._n = 3  # Glen's flow law exponent
        self._g = 9.81  # Gravitational acceleration
        # Compute gamma: γ = 2A(ρg)^n / (n+2)
        self._gamma = 2 * self._A * (self._rho * self._g) ** self._n / (self._n + 2)
        super().__init__(sol_str, nvars, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build sympy expressions for shallow ice equation."""
        cartesian_symbs = self.cartesian_symbols()

        # Parse expressions
        bed_expr = sp.sympify(self._bed_str)
        friction_expr = sp.sympify(self._friction_str)
        sol_expr = self._expressions["solution"]  # Ice thickness H

        # Surface elevation: s = H + b
        surface_expr = bed_expr + sol_expr

        # Surface gradient components
        surface_grad_exprs = [surface_expr.diff(s, 1) for s in cartesian_symbs]

        # |grad(s)|² = sum of squared gradient components
        grad_s_squared = sum(gs**2 for gs in surface_grad_exprs)

        # Nonlinear diffusion coefficient:
        # D = γ * H^(n+2) * |grad(s)|^(n-1) + (ρg/C) * H²
        deformation_diffusion = (
            self._gamma
            * sol_expr ** (self._n + 2)
            * grad_s_squared ** ((self._n - 1) / 2)
        )
        sliding_diffusion = self._rho * self._g / friction_expr * sol_expr**2
        diffusion = deformation_diffusion + sliding_diffusion

        # Flux: F = D * grad(s) (note: positive diffusion unlike standard -D*grad)
        flux_exprs = [diffusion * gs for gs in surface_grad_exprs]

        # Forcing contribution: -div(flux) = -div(D * grad(s))
        # Since PDE is dH/dt - div(D*grad(s)) = f
        # Residual is -div(flux) + f, so f = div(flux) for zero residual
        forc_expr = -sum(
            flux_expr.diff(symb, 1)
            for symb, flux_expr in zip(cartesian_symbs, flux_exprs)
        )

        # Store expressions
        self._set_expression("bed", bed_expr, self._bed_str)
        self._set_expression("friction", friction_expr, self._friction_str)
        self._set_expression("surface", surface_expr, self._sol_str)
        self._set_expression("diffusion", diffusion, self._sol_str)
        self._set_expression("flux", flux_exprs, self._sol_str)
        self._expressions["forcing"] += forc_expr
