"""Manufactured solutions for Burgers equation.

1D Burgers equation (conservative form):
    du/dt + d/dx(u²/2) = ν * d²u/dx²

or in flux form:
    du/dt + d/dx(F) = 0
    where F = u²/2 - ν * du/dx
"""

from typing import Generic

import sympy as sp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    ScalarSolutionMixin,
)


class ManufacturedBurgers1D(
    ScalarSolutionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for 1D Burgers equation.

    Solves: du/dt + d/dx(u²/2 - ν*du/dx) = f

    The flux is: F = u²/2 - ν*du/dx

    For transient problems, the temporal derivative is added to the forcing.

    Parameters
    ----------
    sol_str : str
        String representation of the exact solution.
        May contain 'x' for spatial coordinate and 'T' for time.
    visc_str : str
        String representation of kinematic viscosity ν.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create manufactured solution: u = sin(pi*x)
    >>> man_sol = ManufacturedBurgers1D(
    ...     sol_str="sin(pi*x)",
    ...     visc_str="0.1",
    ...     bkd=bkd,
    ... )
    >>> # Get forcing function
    >>> x = bkd.linspace(0, 1, 10)
    >>> nodes = x.reshape(1, -1)  # Shape (1, npts) for 1D
    >>> forcing = man_sol.functions["forcing"](nodes)
    """

    def __init__(
        self,
        sol_str: str,
        visc_str: str,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        self._visc_str = visc_str
        # 1D problem
        super().__init__(sol_str, 1, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build sympy expressions for Burgers equation."""
        cartesian_symbs = self.cartesian_symbols()
        x = cartesian_symbs[0]

        visc_expr = sp.sympify(self._visc_str)
        sol_expr = self._expressions["solution"]

        # Flux: F = u²/2 - ν*du/dx
        flux_expr = sol_expr**2 / 2 - visc_expr * sol_expr.diff(x)
        flux_exprs = [flux_expr]

        # Forcing = dF/dx (for residual = dF/dx + f = 0, so f = -dF/dx)
        # But since we compute du/dt = -dF/dx + f, we need f = dF/dx
        forc_expr = flux_expr.diff(x)

        self._set_expression("viscosity", visc_expr, self._visc_str)
        self._set_expression("flux", flux_exprs, self._sol_str)
        # Store diffusive flux separately (just -ν*du/dx) for Galerkin
        # natural BCs, which only involve the diffusive part.
        diff_flux_expr = -visc_expr * sol_expr.diff(x)
        self._set_expression("diffusive_flux", [diff_flux_expr], self._sol_str)
        self._expressions["forcing"] += forc_expr
