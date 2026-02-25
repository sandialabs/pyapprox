"""Manufactured solutions for Shallow Shelf Approximation (SSA) equations.

SSA momentum equations:
    -div(2*μ*H*ε) + C*u + ρg*H*ds/dx = f_u
    -div(2*μ*H*ε) + C*v + ρg*H*ds/dy = f_v

where:
    (u, v) = velocity field
    H = ice thickness
    s = H + b = surface elevation
    b = bed elevation
    C = friction coefficient
    ρ = ice density
    g = gravitational acceleration
    μ = effective viscosity (Glen's flow law)
    ε = strain rate tensor

Effective viscosity (Glen's flow law with n=3):
    μ = 0.5 * A^(-1/n) * (effective_strain_rate)^((1-n)/n)

Effective strain rate:
    ε_eff = sqrt(ε_xx² + ε_yy² + ε_xx*ε_yy + 0.25*(ε_xy + ε_yx)²)

Strain tensor:
    ε_xx = du/dx
    ε_yy = dv/dy
    ε_xy = ε_yx = 0.5*(du/dy + dv/dx)
"""

from typing import Generic, List
import copy

import sympy as sp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    VectorSolutionMixin,
)


class ManufacturedShallowShelfVelocityEquations(
    VectorSolutionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for Shallow Shelf Approximation (velocity only).

    Solution components: [u, v] (velocity field)

    Parameters
    ----------
    sol_strs : List[str]
        String representations of velocity components [u, v].
    nvars : int
        Number of spatial dimensions (must be 2).
    bed_str : str
        String representation of bed elevation b.
    depth_str : str
        String representation of ice thickness H.
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
    >>> man_sol = ManufacturedShallowShelfVelocityEquations(
    ...     sol_strs=["(1-x**2)*(1-y**2)", "(1-x**2)*(1-y**2)*0.5"],
    ...     nvars=2,
    ...     bed_str="0.0",
    ...     depth_str="1.0",
    ...     friction_str="1e4",
    ...     A=1e-16,
    ...     rho=917.0,
    ...     bkd=bkd,
    ... )
    """

    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        bed_str: str,
        depth_str: str,
        friction_str: str,
        A: float,
        rho: float,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        if nvars != 2:
            raise ValueError("ShallowShelfVelocityEquations requires nvars=2")
        if len(sol_strs) < 2:
            raise ValueError("Expected at least 2 velocity components [u, v]")

        self._bed_str = bed_str
        self._depth_str = depth_str
        self._friction_str = friction_str
        self._A = A
        self._rho = rho
        self._g = 9.81
        self._n = 3  # Glen's flow law exponent
        super().__init__(sol_strs, nvars, bkd, oned)

    def velocity_expressions(self) -> List[sp.Expr]:
        """Return velocity expressions."""
        return self._expressions["solution"]

    def sympy_expressions(self) -> None:
        """Build sympy expressions for SSA equations."""
        cartesian_symbs = self.cartesian_symbols()
        x, y = cartesian_symbs[0], cartesian_symbs[1]

        # Parse prescribed fields
        bed_expr = sp.sympify(self._bed_str)
        depth_expr = sp.sympify(self._depth_str)
        friction_expr = sp.sympify(self._friction_str)
        surface_expr = bed_expr + depth_expr

        # Velocity field
        u, v = self.velocity_expressions()

        # Velocity gradients
        ux = u.diff(x)
        uy = u.diff(y)
        vx = v.diff(x)
        vy = v.diff(y)

        # Strain tensor components
        exx = ux
        eyy = vy
        exy = sp.Rational(1, 2) * (uy + vx)

        # Effective strain rate (regularized to avoid singularity)
        effective_strain_rate = (
            exx**2 + eyy**2 + exx * eyy + exy**2
        ) ** sp.Rational(1, 2)

        # Effective viscosity (Glen's flow law)
        # μ = 0.5 * A^(-1/n) * ε_eff^((1-n)/n)
        # For n=3: μ = 0.5 * A^(-1/3) * ε_eff^(-2/3)
        mu_expr = (
            sp.Rational(1, 2)
            * self._A ** (-sp.Rational(1, self._n))
            * effective_strain_rate ** (sp.Rational(1, self._n) - 1)
        )

        # Stress tensor: τ = 2*μ*H*ε (symmetric)
        # For 2D plane stress: τ_ij = 2*μ*H*[ε_ij + (trace(ε))*δ_ij for diagonal]
        # Actually for SSA: τ = 2*μ*H*[[2*ε_xx + ε_yy, ε_xy], [ε_xy, ε_xx + 2*ε_yy]]
        strain_tensor = [
            [2.0 * exx + eyy, exy],
            [exy, exx + 2.0 * eyy],
        ]
        flux = [
            [2 * mu_expr * depth_expr * s for s in row]
            for row in strain_tensor
        ]

        # Forcing for momentum equations
        # -div(flux) + friction*velocity + driving_stress = f
        # driving_stress = ρ*g*H*grad(surface)
        surface_grad = [surface_expr.diff(s) for s in cartesian_symbs]

        forc_expr_u = (
            -(flux[0][0]).diff(x)
            - (flux[0][1]).diff(y)
            + friction_expr * u
            + self._rho * self._g * depth_expr * surface_grad[0]
        )
        forc_expr_v = (
            -(flux[1][0]).diff(x)
            - (flux[1][1]).diff(y)
            + friction_expr * v
            + self._rho * self._g * depth_expr * surface_grad[1]
        )
        forc_exprs = [forc_expr_u, forc_expr_v]

        # Store expressions
        self._set_expression("bed", bed_expr, self._bed_str)
        self._set_expression("depth", depth_expr, self._depth_str)
        self._set_expression("surface", surface_expr, self._depth_str)
        self._set_expression("friction", friction_expr, self._friction_str)
        self._set_expression("effective_strain_rate", effective_strain_rate, self._sol_strs[0])
        self._set_expression("flux", flux, self._sol_strs[0])

        # Add to forcing
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]


class ManufacturedShallowShelfVelocityAndDepthEquations(
    ManufacturedShallowShelfVelocityEquations
):
    """Manufactured solution for SSA with both velocity and depth evolution.

    Solution components: [H, u, v] (depth, x-velocity, y-velocity)

    Adds mass conservation equation:
        dH/dt + div(H*u) = f_H

    Parameters
    ----------
    vel_strs : List[str]
        String representations of velocity components [u, v].
    nvars : int
        Number of spatial dimensions (must be 2).
    bed_str : str
        String representation of bed elevation.
    depth_str : str
        String representation of ice thickness (also a solution component).
    friction_str : str
        String representation of friction coefficient.
    A : float
        Rate factor in Glen's flow law.
    rho : float
        Ice density.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.
    """

    def __init__(
        self,
        vel_strs: List[str],
        nvars: int,
        bed_str: str,
        depth_str: str,
        friction_str: str,
        A: float,
        rho: float,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        # Solution order: [H, u, v]
        sol_strs = [depth_str] + vel_strs
        super().__init__(
            sol_strs,
            nvars,
            bed_str,
            depth_str,
            friction_str,
            A,
            rho,
            bkd,
            oned,
        )

    def velocity_expressions(self) -> List[sp.Expr]:
        """Return velocity expressions (components 1 and 2)."""
        return self._expressions["solution"][1:]

    def sympy_expressions(self) -> None:
        """Build sympy expressions including mass conservation."""
        # Build velocity equations first
        super().sympy_expressions()

        cartesian_symbs = self.cartesian_symbols()
        depth = self._expressions["depth"]
        vel_exprs = self.velocity_expressions()

        # Mass conservation: dH/dt + div(H*vel) = f_H
        # For manufactured solution: f_H = div(H*vel)
        depth_forc = sum(
            (depth * vel_expr).diff(symb)
            for vel_expr, symb in zip(vel_exprs, cartesian_symbs)
        )

        # Store velocity forcing separately
        self._expressions["velocity_forcing"] = self._expressions["forcing"]

        # Full forcing: [depth_forcing, velocity_forcing...]
        self._expressions["forcing"] = [depth_forc] + self._expressions["forcing"]
        self._expressions["depth_forcing"] = depth_forc

        # Update transient flags
        self.transient["depth_forcing"] = self.is_transient()
        self.transient["velocity_forcing"] = self.is_transient()

        # Update flux to include depth flux
        velocity_flux = self._expressions["flux"]
        depth_flux = [[depth * vel for vel in vel_exprs]]
        flux = depth_flux + velocity_flux
        del self._expressions["flux"]
        self._set_expression("flux", flux, self._sol_strs[0])

    def sympy_temporal_derivative_expression(self) -> None:
        """Add temporal derivative for mass conservation (depth only).

        Only the depth equation is time-dependent (dH/dt + div(H*vel) = f_H).
        The velocity equations are elliptic (no du/dt or dv/dt).
        """
        if not self.is_transient():
            raise ValueError("VelocityAndDepth equations must be transient")

        # Store forcing without time derivative
        self._set_expression(
            "depth_forcing_without_time_deriv",
            copy.deepcopy(self._expressions["depth_forcing"]),
            self._sol_strs[0],
        )

        # Add dH/dt to depth forcing and to forcing[0]
        time_symb = self.time_symbol()[0]
        dH_dt = self._expressions["solution"][0].diff(time_symb)
        self._expressions["depth_forcing"] += dH_dt
        self._expressions["forcing"][0] += dH_dt
