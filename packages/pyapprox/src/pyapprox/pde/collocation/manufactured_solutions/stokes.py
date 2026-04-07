"""Manufactured solutions for Stokes and Navier-Stokes equations.

Stokes equations (2D):
    -Δu + dp/dx = f_u
    -Δv + dp/dy = f_v
    du/dx + dv/dy = 0  (incompressibility)

Navier-Stokes equations (2D, adds convective term):
    -Δu + u*du/dx + v*du/dy + dp/dx = f_u
    -Δv + u*dv/dx + v*dv/dy + dp/dy = f_v
    du/dx + dv/dy = 0  (incompressibility)

where:
    (u, v) = velocity field
    p = pressure
"""

import copy
from typing import Generic, List

from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    VectorSolutionMixin,
)
from pyapprox.util.backends.protocols import Array, Backend


class ManufacturedStokes(
    VectorSolutionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for Stokes/Navier-Stokes equations.

    Solution components: [u, p] for 1D, [u, v, p] for 2D, or
    [u, v, w, p] for 3D, where (u, v, w) is velocity and p is pressure.

    Parameters
    ----------
    sol_strs : List[str]
        String representations of solution components.
        1D: [u, p], 2D: [u, v, p], 3D: [u, v, w, p]
    nvars : int
        Number of spatial dimensions (1, 2, or 3).
    navier_stokes : bool
        If True, include convective (nonlinear) terms.
        If False, solve linear Stokes equations.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 2D Stokes manufactured solution
    >>> man_sol = ManufacturedStokes(
    ...     sol_strs=["sin(pi*x)*cos(pi*y)", "-cos(pi*x)*sin(pi*y)", "x*y"],
    ...     nvars=2,
    ...     navier_stokes=False,
    ...     bkd=bkd,
    ... )
    """

    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        navier_stokes: bool,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        if nvars not in (1, 2, 3):
            raise ValueError("Stokes requires nvars=1, 2, or 3")
        if len(sol_strs) != nvars + 1:
            raise ValueError(
                f"Expected {nvars + 1} solution components (velocity + pressure)"
            )

        self._navier_stokes = navier_stokes
        self._vel_strs = sol_strs[:nvars]
        self._pres_str = sol_strs[nvars]
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build sympy expressions for Stokes/Navier-Stokes equations."""
        cartesian_symbs = self.cartesian_symbols()
        exprs = self._expressions["solution"]
        vel_exprs = exprs[: self._nvars]
        pres_expr = exprs[self._nvars]

        # Build velocity forcing: f_i = -Δv_i + dp/dx_i + (v·∇)v_i (if NS)
        vel_forc_exprs = []
        for ii, (vel, s1) in enumerate(zip(vel_exprs, cartesian_symbs)):
            # Diffusion term: -Δv_i (negative Laplacian)
            forc = sum(-vel.diff(s2, 2) for s2 in cartesian_symbs)

            # Pressure gradient: +dp/dx_i
            forc += pres_expr.diff(s1)

            # Convective term for Navier-Stokes: (v·∇)v_i = sum_j v_j * dv_i/dx_j
            if self._navier_stokes:
                forc += sum(
                    u * vel.diff(s2) for u, s2 in zip(vel_exprs, cartesian_symbs)
                )

            vel_forc_exprs.append(forc)

        # Pressure forcing: continuity constraint div(v) = 0
        # f_p = du/dx + dv/dy (+ dw/dz for 3D)
        pres_forc_expr = sum(vel.diff(s) for vel, s in zip(vel_exprs, cartesian_symbs))

        # Store velocity and pressure gradients as flux
        vel_grad_exprs = [[v.diff(s) for s in cartesian_symbs] for v in vel_exprs]
        pres_grad_expr = [pres_expr.diff(s) for s in cartesian_symbs]
        flux_exprs = vel_grad_exprs + [pres_grad_expr]

        self._set_expression("flux", flux_exprs, self._sol_strs[0])
        self._set_expression("vel_forcing", vel_forc_exprs, self._sol_strs[0])
        self._set_expression("pres_forcing", pres_forc_expr, self._sol_strs[-1])

        # Full forcing: velocity forcing + pressure forcing
        forc_exprs = vel_forc_exprs + [pres_forc_expr]
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]

    def sympy_temporal_derivative_expression(self) -> None:
        """Add temporal derivatives to forcing for transient problems.

        Overrides the base VectorSolutionMixin to only add du/dT for
        velocity components (indices 0..nvars-1). The pressure component
        (continuity equation div(u)=0) has no temporal derivative.
        """
        if self.is_transient():
            self._set_expression(
                "forcing_without_time_deriv",
                copy.deepcopy(self._expressions["forcing"]),
                self._sol_strs[0],
            )
            # Only add temporal derivative for velocity, NOT pressure
            for ii in range(self._nvars):
                self._expressions["forcing"][ii] += self._expressions["solution"][
                    ii
                ].diff(self.time_symbol()[0])
