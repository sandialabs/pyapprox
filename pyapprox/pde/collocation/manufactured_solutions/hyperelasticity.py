"""Manufactured solutions for hyperelasticity equations.

Provides a sympy-based manufactured solution class that works with any
stress model implementing SymbolicStressModelProtocol. Computes forcing
f such that div(P(u*)) + f = 0 for a given exact displacement u*.

Supports 1D, 2D, and 3D.
"""

from typing import Generic, List

import sympy as sp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    VectorSolutionMixin,
)
from pyapprox.pde.collocation.physics.stress_models.protocols import (
    SymbolicStressModelProtocol,
)


class ManufacturedHyperelasticityEquations(
    VectorSolutionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for hyperelasticity equations.

    Solves: div(P) + f = 0

    where P is the first Piola-Kirchhoff stress computed by the provided
    symbolic stress model from the deformation gradient F = I + grad(u*).

    The forcing f is computed symbolically so that the residual vanishes
    at the manufactured displacement u*.

    Supports 1D (nvars=1), 2D (nvars=2), and 3D (nvars=3).

    Parameters
    ----------
    sol_strs : List[str]
        Exact displacement component expressions. Length must equal nvars.
        For 1D: ["u(x)"], for 2D: ["u(x,y)", "v(x,y)"], etc.
    nvars : int
        Number of spatial dimensions (1, 2, or 3).
    stress_model : SymbolicStressModelProtocol
        Stress model with sympy expression support.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.
    """

    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        stress_model: SymbolicStressModelProtocol,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        if nvars not in (1, 2, 3):
            raise ValueError(f"Unsupported dimension: {nvars}")
        if len(sol_strs) != nvars:
            raise ValueError(
                f"Need {nvars} solution components, got {len(sol_strs)}"
            )
        self._stress_model = stress_model
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build sympy expressions for hyperelasticity."""
        nvars = self.nvars()
        if nvars == 1:
            self._sympy_expressions_1d()
        elif nvars == 2:
            self._sympy_expressions_2d()
        else:
            self._sympy_expressions_3d()

    def _sympy_expressions_1d(self) -> None:
        """Build 1D expressions: dP/dx + f = 0."""
        x = self.cartesian_symbols()[0]
        u_expr = self._expressions["solution"][0]

        # Deformation gradient
        F_expr = 1 + u_expr.diff(x)

        # PK1 stress
        P_expr = self._stress_model.sympy_stress_1d(F_expr)

        # Store flux
        self._set_expression("flux", [[P_expr]], self._sol_strs[0])

        # Divergence: dP/dx
        div_P = P_expr.diff(x)

        # Forcing: f = -div(P) so that div(P) + f = 0
        self._expressions["forcing"] = [
            self._expressions["forcing"][0] - div_P
        ]

    def _sympy_expressions_2d(self) -> None:
        """Build 2D expressions: dP_iJ/dX_J + f_i = 0."""
        x, y = self.cartesian_symbols()
        u_expr = self._expressions["solution"][0]
        v_expr = self._expressions["solution"][1]

        # Deformation gradient F = I + grad(u)
        F11 = 1 + u_expr.diff(x)
        F12 = u_expr.diff(y)
        F21 = v_expr.diff(x)
        F22 = 1 + v_expr.diff(y)

        # PK1 stress
        P11, P12, P21, P22 = self._stress_model.sympy_stress_2d(
            F11, F12, F21, F22
        )

        # Store flux (stress tensor)
        tau = [[P11, P12], [P21, P22]]
        self._set_expression("flux", tau, self._sol_strs[0])

        # Divergence: div(P)_i = dP_i1/dx + dP_i2/dy
        div_P_x = P11.diff(x) + P12.diff(y)
        div_P_y = P21.diff(x) + P22.diff(y)

        # Forcing: f_i = -div(P)_i
        self._expressions["forcing"] = [
            self._expressions["forcing"][0] - div_P_x,
            self._expressions["forcing"][1] - div_P_y,
        ]

    def _sympy_expressions_3d(self) -> None:
        """Build 3D expressions: dP_iJ/dX_J + f_i = 0."""
        x, y, z = self.cartesian_symbols()
        u_expr = self._expressions["solution"][0]
        v_expr = self._expressions["solution"][1]
        w_expr = self._expressions["solution"][2]

        # Deformation gradient F = I + grad(u)
        F = (
            (1 + u_expr.diff(x), u_expr.diff(y), u_expr.diff(z)),
            (v_expr.diff(x), 1 + v_expr.diff(y), v_expr.diff(z)),
            (w_expr.diff(x), w_expr.diff(y), 1 + w_expr.diff(z)),
        )

        # PK1 stress
        P = self._stress_model.sympy_stress_3d(F)

        # Store flux
        tau = [list(row) for row in P]
        self._set_expression("flux", tau, self._sol_strs[0])

        # Divergence and forcing
        coords = [x, y, z]
        forc_exprs = []
        for i in range(3):
            div_P_i = sum(P[i][j].diff(coords[j]) for j in range(3))
            forc_exprs.append(
                self._expressions["forcing"][i] - div_P_i
            )
        self._expressions["forcing"] = forc_exprs
