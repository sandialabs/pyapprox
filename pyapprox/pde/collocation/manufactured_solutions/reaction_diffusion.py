"""Manufactured solutions for two-species reaction-diffusion systems.

Two-species reaction-diffusion system:
    du0/dt = div(D0 * grad(u0)) + R0(u0, u1) + f0
    du1/dt = div(D1 * grad(u1)) + R1(u0, u1) + f1

where:
    R0(u0, u1), R1(u0, u1) = reaction terms (functions of both species)
    D0, D1 = diffusion coefficients
    f0, f1 = forcing/source terms

The reaction is provided via a SymbolicReactionProtocol object that can be
shared between the manufactured solution and the physics class.
"""

from typing import Generic, List, Tuple, Any, Dict, Callable, TYPE_CHECKING

import sympy as sp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    VectorSolutionMixin,
)
from pyapprox.pde.collocation.manufactured_solutions.mixins import (
    DiffusionMixin,
)

if TYPE_CHECKING:
    from pyapprox.pde.collocation.physics.reaction_diffusion import (
        SymbolicReactionProtocol,
    )


class ManufacturedTwoSpeciesReactionDiffusion(
    VectorSolutionMixin,
    DiffusionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for two-species reaction-diffusion system.

    The reaction term is provided via a SymbolicReactionProtocol object,
    allowing the same reaction definition to be used by both the manufactured
    solution (for forcing computation) and the physics class (for residual
    evaluation).

    Parameters
    ----------
    sol_strs : List[str]
        String representations of species concentrations [u0, u1].
    nvars : int
        Number of spatial dimensions.
    diff_strs : List[str]
        String representations of diffusion coefficients [D0, D1].
    reaction : SymbolicReactionProtocol
        Reaction object providing sympy_expressions() method.
        The same object should be passed to TwoSpeciesReactionDiffusionPhysics.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.collocation.physics import LinearReaction
    >>> bkd = NumpyBkd()
    >>> # Create reaction object (shared between manufactured and physics)
    >>> reaction = LinearReaction(a00=1.0, a01=-1.0, a10=1.0, a11=-0.5, bkd=bkd)
    >>> # Create manufactured solution
    >>> man_sol = ManufacturedTwoSpeciesReactionDiffusion(
    ...     sol_strs=["sin(pi*x)*sin(pi*y)", "cos(pi*x)*cos(pi*y)"],
    ...     nvars=2,
    ...     diff_strs=["1.0", "0.5"],
    ...     reaction=reaction,
    ...     bkd=bkd,
    ... )
    """

    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        diff_strs: List[str],
        reaction: "SymbolicReactionProtocol[Array]",
        bkd: Backend[Array],
        oned: bool = False,
    ):
        if len(sol_strs) != 2:
            raise ValueError("TwoSpeciesReactionDiffusion requires 2 species")
        if len(diff_strs) != 2:
            raise ValueError("Expected 2 diffusion coefficients")
        if not hasattr(reaction, 'sympy_expressions'):
            raise TypeError(
                "reaction must implement SymbolicReactionProtocol "
                "(must have sympy_expressions method)"
            )

        self._diff_strs = diff_strs
        self._reaction = reaction
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_diffusion_expressions(self) -> None:
        """Build diffusion expressions for both species."""
        # Diffusion for species 0
        diff_expr0, _, flux_exprs0, forc_expr0 = (
            self._sympy_diffusion_expressions(
                self._diff_strs[0], self._expressions["solution"][0]
            )
        )
        # Diffusion for species 1
        diff_expr1, _, flux_exprs1, forc_expr1 = (
            self._sympy_diffusion_expressions(
                self._diff_strs[1], self._expressions["solution"][1]
            )
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
        """Build reaction expressions using the SymbolicReactionProtocol.

        The reaction object provides sympy_expressions(u0_expr, u1_expr)
        which returns (R0_expr, R1_expr) evaluated at the manufactured
        solution expressions.
        """
        u0_expr = self._expressions["solution"][0]
        u1_expr = self._expressions["solution"][1]

        # Get symbolic reaction expressions from the reaction object
        R0_expr, R1_expr = self._reaction.sympy_expressions(u0_expr, u1_expr)
        react_exprs = [R0_expr, R1_expr]

        self._set_expression("reaction", react_exprs, "reaction(u0, u1)")

        # Physics computes: du/dt = diffusion + reaction + forcing
        # Manufactured solution: du/dt = 0 (steady state)
        # So: 0 = diffusion + reaction + forcing
        # => forcing = -diffusion - reaction
        # But diffusion forcing was already added as +div(D*grad(u)) which
        # equals -diffusion term, so we need forcing -= reaction
        self._expressions["forcing"] = [
            f - r for f, r in zip(self._expressions["forcing"], react_exprs)
        ]

    def sympy_expressions(self) -> None:
        """Build all sympy expressions for the system."""
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()
