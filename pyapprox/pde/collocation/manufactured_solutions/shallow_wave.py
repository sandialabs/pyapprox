"""Manufactured solutions for shallow water wave equations.

Shallow water equations (1D):
    dh/dt + d(uh)/dx = 0
    d(uh)/dt + d(u²h + 0.5*g*h²)/dx = -g*h*db/dx

Shallow water equations (2D):
    dh/dt + d(uh)/dx + d(vh)/dy = 0
    d(uh)/dt + d(u²h + 0.5*g*h²)/dx + d(uvh)/dy = -g*h*db/dx
    d(vh)/dt + d(uvh)/dx + d(v²h + 0.5*g*h²)/dy = -g*h*db/dy

where:
    h = water depth
    uh, vh = momentum components
    b = bed elevation
    g = gravitational acceleration
"""

from typing import Generic, List

import sympy as sp

from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    VectorSolutionMixin,
)
from pyapprox.util.backends.protocols import Array, Backend


class ManufacturedShallowWave(
    VectorSolutionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for shallow water wave equations.

    Solution components:
    - 1D: [h, uh] (depth, x-momentum)
    - 2D: [h, uh, vh] (depth, x-momentum, y-momentum)

    Parameters
    ----------
    nvars : int
        Number of spatial dimensions (1 or 2).
    depth_str : str
        String representation of water depth h.
    mom_strs : List[str]
        String representations of momentum components.
        1D: [uh], 2D: [uh, vh]
    bed_str : str
        String representation of bed elevation b.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 1D manufactured solution
    >>> man_sol = ManufacturedShallowWave(
    ...     nvars=1,
    ...     depth_str="1 + 0.1*sin(pi*x)",
    ...     mom_strs=["0.5*sin(pi*x)"],
    ...     bed_str="0.1*x",
    ...     bkd=bkd,
    ... )
    """

    def __init__(
        self,
        nvars: int,
        depth_str: str,
        mom_strs: List[str],
        bed_str: str,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        if nvars not in (1, 2):
            raise ValueError("ShallowWave requires nvars=1 or nvars=2")
        if len(mom_strs) != nvars:
            raise ValueError(
                f"Expected {nvars} momentum components, got {len(mom_strs)}"
            )

        self._depth_str = depth_str
        self._mom_strs = mom_strs
        self._bed_str = bed_str
        self._g = 9.81  # Gravitational acceleration

        # Solution order: [h, uh] for 1D, [h, uh, vh] for 2D
        sol_strs = [depth_str] + mom_strs
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build sympy expressions for shallow water equations."""
        cartesian_symbs = self.cartesian_symbols()
        bed_expr = sp.sympify(self._bed_str)

        if self._nvars == 1:
            self._build_1d_expressions(cartesian_symbs, bed_expr)
        else:
            self._build_2d_expressions(cartesian_symbs, bed_expr)

        self._set_expression("bed", bed_expr, self._bed_str)

    def _build_1d_expressions(
        self, cartesian_symbs: List[sp.Symbol], bed_expr: sp.Expr
    ) -> None:
        """Build expressions for 1D shallow water equations."""
        x = cartesian_symbs[0]
        h, uh = self._expressions["solution"]

        # Flux components
        # Continuity: flux = uh
        # Momentum: flux = uh²/h + 0.5*g*h²
        flux_h = [uh]
        flux_uh = [(uh**2) / h + sp.Rational(1, 2) * self._g * h**2]
        flux_exprs = [flux_h, flux_uh]

        # Forcing = d(flux)/dx (hyperbolic form: du/dt + dF/dx = S)
        # For manufactured solution, f = dF/dx + S where S is source
        forc_h = flux_h[0].diff(x)
        forc_uh = flux_uh[0].diff(x)

        # Add bed slope source term for momentum: +g*h*db/dx
        # (This is the term that appears on RHS of momentum equation)
        forc_uh += self._g * h * bed_expr.diff(x)

        forc_exprs = [forc_h, forc_uh]

        self._set_expression("flux", flux_exprs, self._depth_str)
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]

    def _build_2d_expressions(
        self, cartesian_symbs: List[sp.Symbol], bed_expr: sp.Expr
    ) -> None:
        """Build expressions for 2D shallow water equations."""
        _x, _y = cartesian_symbs[0], cartesian_symbs[1]
        h, uh, vh = self._expressions["solution"]

        # Cross momentum term
        uvh = uh * vh / h

        # Flux components [F_x, F_y] for each equation
        # Continuity: [uh, vh]
        # x-momentum: [uh²/h + 0.5*g*h², uvh]
        # y-momentum: [uvh, vh²/h + 0.5*g*h²]
        flux_exprs = [
            [uh, vh],  # continuity
            [(uh**2) / h + sp.Rational(1, 2) * self._g * h**2, uvh],  # x-mom
            [uvh, (vh**2) / h + sp.Rational(1, 2) * self._g * h**2],  # y-mom
        ]

        # Forcing = div(flux) = dF_x/dx + dF_y/dy
        forc_exprs = []
        for ii in range(self._nvars + 1):
            forc = sum(flux.diff(s) for flux, s in zip(flux_exprs[ii], cartesian_symbs))
            forc_exprs.append(forc.simplify())

        # Add bed slope source terms for momentum equations
        # x-momentum: +g*h*db/dx
        # y-momentum: +g*h*db/dy
        for ii in range(self._nvars):
            forc_exprs[ii + 1] += self._g * h * bed_expr.diff(cartesian_symbs[ii])

        self._set_expression("flux", flux_exprs, self._depth_str)
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]
