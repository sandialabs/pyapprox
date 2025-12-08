"""Manufactured solutions for two-species reaction-diffusion systems.

Two-species reaction-diffusion system:
    du0/dt = div(D0 * grad(u0)) - R0(u0, u1) + f0
    du1/dt = div(D1 * grad(u1)) - R1(u0, u1) + f1

where:
    R0(u0, u1) = c0*u0^p0 - u1  (production of u0, consumption by u1)
    R1(u0, u1) = c1*u1^p1 + u0  (production of u1, coupling to u0)
"""

from typing import Generic, List, Tuple, Any, Dict, Callable

import sympy as sp

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    VectorSolutionMixin,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.mixins import (
    DiffusionMixin,
    ReactionMixin,
)


class ManufacturedTwoSpeciesReactionDiffusion(
    VectorSolutionMixin,
    DiffusionMixin,
    ReactionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for two-species reaction-diffusion system.

    Reaction Vector: R(u0, u1) = [c0*u0^p0 - u1, c1*u1^p1 + u0]
    for coefficients c0, c1 and powers p0 and p1.

    Parameters
    ----------
    sol_strs : List[str]
        String representations of species concentrations [u0, u1].
    nvars : int
        Number of spatial dimensions.
    diff_strs : List[str]
        String representations of diffusion coefficients [D0, D1].
    react_strs : List[str]
        String representations of reaction terms [c0*u^p0, c1*u^p1].
        Note: The 'u' in react_strs[i] refers to u_i.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 2D two-species system
    >>> man_sol = ManufacturedTwoSpeciesReactionDiffusion(
    ...     sol_strs=["sin(pi*x)*sin(pi*y)", "cos(pi*x)*cos(pi*y)"],
    ...     nvars=2,
    ...     diff_strs=["1.0", "0.5"],
    ...     react_strs=["u**2", "u"],
    ...     bkd=bkd,
    ... )
    """

    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        diff_strs: List[str],
        react_strs: List[str],
        bkd: Backend[Array],
        oned: bool = False,
    ):
        if len(sol_strs) != 2:
            raise ValueError("TwoSpeciesReactionDiffusion requires 2 species")
        if len(diff_strs) != 2:
            raise ValueError("Expected 2 diffusion coefficients")
        if len(react_strs) != 2:
            raise ValueError("Expected 2 reaction terms")

        self._diff_strs = diff_strs
        self._react_strs = react_strs
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_diffusion_expressions(self) -> None:
        """Build diffusion expressions for both species."""
        # Diffusion for species 0
        diff_expr0, flux_exprs0, forc_expr0 = self._sympy_diffusion_expressions(
            self._diff_strs[0], self._expressions["solution"][0]
        )
        # Diffusion for species 1
        diff_expr1, flux_exprs1, forc_expr1 = self._sympy_diffusion_expressions(
            self._diff_strs[1], self._expressions["solution"][1]
        )

        forc_exprs = [forc_expr0, forc_expr1]
        flux_exprs = [flux_exprs0, flux_exprs1]
        diff_exprs = [diff_expr0, diff_expr1]

        self._set_expression("diffusion", diff_exprs, self._diff_strs[0])
        self._set_expression("flux", flux_exprs, self._sol_strs[0])
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]

    def sympy_reaction_expressions(self) -> None:
        """Build reaction expressions for the coupled system.

        Reaction terms:
        R0 = c0*u0^p0 - u1  (reaction of species 0 minus consumption by species 1)
        R1 = c1*u1^p1 + u0  (reaction of species 1 plus production from species 0)
        """
        # Parse base reaction terms (substituting 'u' with actual solution)
        react_str0, react_expr0 = self._sympy_reaction_expressions(
            self._react_strs[0], self._sol_strs[0]
        )
        react_str1, react_expr1 = self._sympy_reaction_expressions(
            self._react_strs[1], self._sol_strs[1]
        )

        # Full reaction terms with cross-species coupling
        # R0 = c0*u0^p0 - u1, R1 = c1*u1^p1 + u0
        react_exprs = [
            react_expr0 - self._expressions["solution"][1],
            react_expr1 + self._expressions["solution"][0],
        ]

        self._set_expression("reaction", react_exprs, self._react_strs[0])

        # Subtract reaction from forcing (standard ADR convention)
        self._expressions["forcing"] = [
            f - g for f, g in zip(self._expressions["forcing"], react_exprs)
        ]

    def sympy_expressions(self) -> None:
        """Build all sympy expressions for the system."""
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()
