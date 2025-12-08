"""Manufactured solutions for advection-diffusion-reaction equations.

Provides manufactured solution classes for verifying ADR physics implementations.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    ScalarSolutionMixin,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.mixins import (
    DiffusionMixin,
    ReactionMixin,
    AdvectionMixin,
)


class ManufacturedAdvectionDiffusionReaction(
    ScalarSolutionMixin,
    DiffusionMixin,
    ReactionMixin,
    AdvectionMixin,
    ManufacturedSolution,
    Generic[Array],
):
    """Manufactured solution for advection-diffusion-reaction equations.

    Solves: du/dt + v . grad(u) = div(D * grad(u)) + R(u) + f

    where f is computed from the manufactured solution to satisfy the PDE.

    Parameters
    ----------
    sol_str : str
        String representation of the exact solution.
        May contain 'x', 'y', 'z' for spatial coordinates and 'T' for time.
    nvars : int
        Number of spatial dimensions.
    diff_str : str
        String representation of diffusion coefficient D.
    react_str : str
        String representation of reaction term R(u).
        Use 'u' as placeholder for solution.
    vel_strs : List[str]
        String representations of velocity components [v_x, v_y, ...].
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.
    conservative : bool
        If True, store conservative flux (v * u) in addition to advection term.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 1D manufactured solution: u = sin(pi*x)
    >>> man_sol = ManufacturedAdvectionDiffusionReaction(
    ...     sol_str="sin(pi*x)",
    ...     nvars=1,
    ...     diff_str="1.0",
    ...     react_str="0",
    ...     vel_strs=["0"],
    ...     bkd=bkd,
    ...     oned=True,
    ... )
    >>> # Get forcing function
    >>> x = bkd.linspace(-1, 1, 10).reshape(1, -1)
    >>> forcing = man_sol.functions["forcing"](x)
    """

    def __init__(
        self,
        sol_str: str,
        nvars: int,
        diff_str: str,
        react_str: str,
        vel_strs: List[str],
        bkd: Backend[Array],
        oned: bool = False,
        conservative: bool = True,
    ):
        self._diff_str = diff_str
        self._react_str = react_str
        self._vel_strs = vel_strs
        self._conservative = conservative
        super().__init__(sol_str, nvars, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build all sympy expressions for ADR equation."""
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()
        self.sympy_advection_expressions()
