"""Manufactured solutions for Helmholtz equation.

Helmholtz equation: -div(grad(u)) + k²*u = f
or equivalently: -Δu + k²*u = f

where k is the wavenumber.
"""

from typing import Generic

import sympy as sp

from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    ScalarSolutionMixin,
)
from pyapprox.pde.collocation.manufactured_solutions.mixins import (
    DiffusionMixin,
    ReactionMixin,
)
from pyapprox.util.backends.protocols import Array, Backend


class ManufacturedHelmholtz(
    ScalarSolutionMixin,
    DiffusionMixin,
    ReactionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for Helmholtz equation.

    Solves: -Δu + k²*u = f

    where k² is the squared wavenumber.

    The forcing f is computed from the manufactured solution to satisfy the PDE.

    Parameters
    ----------
    sol_str : str
        String representation of the exact solution.
        May contain 'x', 'y', 'z' for spatial coordinates and 'T' for time.
    nvars : int
        Number of spatial dimensions.
    sqwavenum_str : str
        String representation of squared wavenumber k².
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 2D manufactured solution: u = sin(pi*x)*sin(pi*y)
    >>> man_sol = ManufacturedHelmholtz(
    ...     sol_str="sin(pi*x)*sin(pi*y)",
    ...     nvars=2,
    ...     sqwavenum_str="1.0",
    ...     bkd=bkd,
    ... )
    >>> # Get forcing function
    >>> x = bkd.linspace(0, 1, 10)
    >>> y = bkd.linspace(0, 1, 10)
    >>> xx, yy = bkd.meshgrid(x, y, indexing='xy')
    >>> nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)
    >>> forcing = man_sol.functions["forcing"](nodes)
    """

    def __init__(
        self,
        sol_str: str,
        nvars: int,
        sqwavenum_str: str,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        # Helmholtz is: -Δu + k²*u = f
        # Using diffusion with D=1: -div(grad(u)) and reaction R(u) = k²*u
        self._diff_str = "1"
        self._sqwavenum_str = sqwavenum_str
        # Reaction term is k²*u
        self._react_str = f"u*({sqwavenum_str})"
        super().__init__(sol_str, nvars, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build sympy expressions for Helmholtz equation."""
        # Add diffusion contribution: div(grad(u))
        self.sympy_diffusion_expressions()
        # Add reaction contribution: k²*u
        self.sympy_reaction_expressions()
        # Store squared wavenumber
        self._set_expression(
            "sqwavenum",
            sp.sympify(self._sqwavenum_str),
            self._sqwavenum_str,
        )
