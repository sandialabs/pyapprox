"""Manufactured solutions for advection-diffusion-reaction equations.

Provides manufactured solution classes for verifying ADR physics implementations.
"""

from typing import Generic, List, Optional

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

    def neumann_values(
        self,
        pts: Array,
        normals: Array,
        time: Optional[float] = None,
        convention: str = "gradient",
    ) -> Array:
        """Compute Neumann boundary values at given points.

        Parameters
        ----------
        pts : Array
            Physical coordinates. Shape: (ndim, npts)
        normals : Array
            Outward unit normals. Shape: (npts, ndim)
        time : float, optional
            Time for transient problems.
        convention : str
            "gradient": returns grad(u) . n
            "flux": returns flux . n = (-D*grad(u) + v*u) . n

        Returns
        -------
        Array
            Neumann values at each point. Shape: (npts,)
        """
        bkd = self._bkd
        if convention == "gradient":
            if self.is_transient():
                vals = self.functions["gradient"](pts, time)
            else:
                vals = self.functions["gradient"](pts)
        elif convention == "flux":
            if self.is_transient():
                vals = self.functions["flux"](pts, time)
            else:
                vals = self.functions["flux"](pts)
        else:
            raise ValueError(
                f"convention must be 'gradient' or 'flux', got '{convention}'"
            )
        # vals shape: (npts, ndim), normals shape: (npts, ndim)
        return bkd.sum(vals * normals, axis=1)

    def robin_values(
        self,
        pts: Array,
        normals: Array,
        alpha: float,
        beta: float,
        time: Optional[float] = None,
        convention: str = "gradient",
    ) -> Array:
        """Compute Robin boundary values g = alpha*u + beta*N(u).

        Parameters
        ----------
        pts : Array
            Physical coordinates. Shape: (ndim, npts)
        normals : Array
            Outward unit normals. Shape: (npts, ndim)
        alpha : float
            Coefficient for u term.
        beta : float
            Coefficient for normal term.
        time : float, optional
            Time for transient problems.
        convention : str
            "gradient": N(u) = grad(u) . n
            "flux": N(u) = flux . n = (-D*grad(u) + v*u) . n

        Returns
        -------
        Array
            Robin values at each point. Shape: (npts,)
        """
        if self.is_transient():
            u_vals = self.functions["solution"](pts, time).flatten()
        else:
            u_vals = self.functions["solution"](pts).flatten()
        normal_term = self.neumann_values(pts, normals, time, convention)
        return alpha * u_vals + beta * normal_term
