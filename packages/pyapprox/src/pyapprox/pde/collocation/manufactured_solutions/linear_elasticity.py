"""Manufactured solutions for linear elasticity equations.

Provides manufactured solution class for verifying 1D, 2D, and 3D linear
elasticity physics implementations.

Linear Elasticity equations:
    -div(σ) + f = 0

where:
    σ = λ*tr(ε)*I + 2μ*ε  (stress tensor)
    ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)  (strain tensor)
    u = (u, v) is the displacement field
"""

from typing import Generic, List

import sympy as sp

from pyapprox.pde.collocation.manufactured_solutions.base import (
    ManufacturedSolution,
    VectorSolutionMixin,
)
from pyapprox.util.backends.protocols import Array, Backend


class ManufacturedLinearElasticityEquations(
    VectorSolutionMixin,
    ManufacturedSolution[Array],
    Generic[Array],
):
    """Manufactured solution for linear elasticity equations (1D/2D/3D).

    Solves: -div(σ) + f = 0

    where:
        σ = λ*tr(ε)*I + 2μ*ε  (stress tensor)
        ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)  (strain tensor)

    The forcing f is computed from the manufactured solution to satisfy the PDE.

    Parameters
    ----------
    sol_strs : List[str]
        String representations of the exact solution components, one per
        spatial dimension. May contain 'x', 'y', 'z' for spatial
        coordinates and 'T' for time.
    nvars : int
        Number of spatial dimensions (1, 2, or 3).
    lambda_str : str
        String representation of Lamé's first parameter λ.
    mu_str : str
        String representation of shear modulus μ.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from evaluation functions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 2D manufactured solution: u = (1-x**2)*(1-y**2), v = x*y*(1-x)*(1-y)
    >>> man_sol = ManufacturedLinearElasticityEquations(
    ...     sol_strs=["(1-x**2)*(1-y**2)", "x*y*(1-x)*(1-y)"],
    ...     nvars=2,
    ...     lambda_str="1.0",
    ...     mu_str="1.0",
    ...     bkd=bkd,
    ... )
    >>> # Get forcing function
    >>> x = bkd.linspace(-1, 1, 10)
    >>> y = bkd.linspace(-1, 1, 10)
    >>> xx, yy = bkd.meshgrid(x, y, indexing='xy')
    >>> nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)
    >>> forcing = man_sol.functions["forcing"](nodes)
    """

    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        lambda_str: str,
        mu_str: str,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        if nvars not in (1, 2, 3):
            raise ValueError(
                f"Linear elasticity requires nvars in (1, 2, 3), got {nvars}"
            )
        if len(sol_strs) != nvars:
            raise ValueError(
                f"Linear elasticity requires {nvars} solution components, "
                f"got {len(sol_strs)}"
            )

        self._lambda_str = lambda_str
        self._mu_str = mu_str
        self._nvars_el = nvars
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_expressions(self) -> None:
        """Build sympy expressions for linear elasticity equation."""
        if self._nvars_el == 1:
            self._sympy_expressions_1d()
        elif self._nvars_el == 2:
            self._sympy_expressions_2d()
        else:
            self._sympy_expressions_3d()

    def _sympy_expressions_1d(self) -> None:
        """Build sympy expressions for 1D linear elasticity.

        σ = (λ + 2μ) * du/dx,  -dσ/dx = f.
        """
        cartesian_symbs = self.cartesian_symbols()
        x = cartesian_symbs[0]

        lambda_expr = sp.sympify(self._lambda_str)
        mu_expr = sp.sympify(self._mu_str)

        self._set_expression("lambda", lambda_expr, self._lambda_str)
        self._set_expression("mu", mu_expr, self._mu_str)

        u_expr = self._expressions["solution"][0]
        E_eff = lambda_expr + 2 * mu_expr
        sigma_xx = E_eff * u_expr.diff(x)

        self._set_expression("flux", [[sigma_xx]], self._sol_strs[0])

        div_sigma = sigma_xx.diff(x)
        forc_exprs = [-div_sigma]

        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]

    def _sympy_expressions_2d(self) -> None:
        """Build sympy expressions for 2D linear elasticity."""
        cartesian_symbs = self.cartesian_symbols()
        x, y = cartesian_symbs[0], cartesian_symbs[1]

        # Lamé parameters
        lambda_expr = sp.sympify(self._lambda_str)
        mu_expr = sp.sympify(self._mu_str)

        self._set_expression("lambda", lambda_expr, self._lambda_str)
        self._set_expression("mu", mu_expr, self._mu_str)

        # Displacement field
        disp_expr = self._expressions["solution"]
        u_expr = disp_expr[0]
        v_expr = disp_expr[1]

        # Strain tensor components: ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)
        exx = u_expr.diff(x)
        exy = sp.Rational(1, 2) * (u_expr.diff(y) + v_expr.diff(x))
        eyy = v_expr.diff(y)

        # Trace of strain
        trace_e = exx + eyy

        # Stress tensor: σ = λ*tr(ε)*I + 2μ*ε
        two_mu = 2 * mu_expr
        sigma_xx = lambda_expr * trace_e + two_mu * exx
        sigma_xy = two_mu * exy
        sigma_yy = lambda_expr * trace_e + two_mu * eyy

        # Store stress tensor (flux)
        tau = [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]]
        self._set_expression("flux", tau, self._sol_strs[0])

        # Compute divergence of stress tensor
        # div(σ)_i = ∂σ_i1/∂x + ∂σ_i2/∂y
        div_sigma_x = sigma_xx.diff(x) + sigma_xy.diff(y)
        div_sigma_y = sigma_xy.diff(x) + sigma_yy.diff(y)

        # Forcing: f = -div(σ) so that div(σ) + f = 0
        # The physics residual is div(σ) + f, so for residual=0 we need f = -div(σ)
        forc_exprs = [-div_sigma_x, -div_sigma_y]

        # Add forcing contribution to existing forcing (initialized to zeros)
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]

    def _sympy_expressions_3d(self) -> None:
        """Build sympy expressions for 3D linear elasticity."""
        cartesian_symbs = self.cartesian_symbols()
        x, y, z = cartesian_symbs[0], cartesian_symbs[1], cartesian_symbs[2]

        # Lamé parameters
        lambda_expr = sp.sympify(self._lambda_str)
        mu_expr = sp.sympify(self._mu_str)

        self._set_expression("lambda", lambda_expr, self._lambda_str)
        self._set_expression("mu", mu_expr, self._mu_str)

        # Displacement field
        disp_expr = self._expressions["solution"]
        u_expr = disp_expr[0]
        v_expr = disp_expr[1]
        w_expr = disp_expr[2]

        # Strain tensor components: ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)
        exx = u_expr.diff(x)
        eyy = v_expr.diff(y)
        ezz = w_expr.diff(z)
        exy = sp.Rational(1, 2) * (u_expr.diff(y) + v_expr.diff(x))
        exz = sp.Rational(1, 2) * (u_expr.diff(z) + w_expr.diff(x))
        eyz = sp.Rational(1, 2) * (v_expr.diff(z) + w_expr.diff(y))

        # Trace of strain
        trace_e = exx + eyy + ezz

        # Stress tensor: σ = λ*tr(ε)*I + 2μ*ε
        two_mu = 2 * mu_expr
        sigma_xx = lambda_expr * trace_e + two_mu * exx
        sigma_yy = lambda_expr * trace_e + two_mu * eyy
        sigma_zz = lambda_expr * trace_e + two_mu * ezz
        sigma_xy = two_mu * exy
        sigma_xz = two_mu * exz
        sigma_yz = two_mu * eyz

        # Store stress tensor (flux)
        tau = [
            [sigma_xx, sigma_xy, sigma_xz],
            [sigma_xy, sigma_yy, sigma_yz],
            [sigma_xz, sigma_yz, sigma_zz],
        ]
        self._set_expression("flux", tau, self._sol_strs[0])

        # Compute divergence of stress tensor
        # div(σ)_i = ∂σ_i1/∂x + ∂σ_i2/∂y + ∂σ_i3/∂z
        div_sigma_x = sigma_xx.diff(x) + sigma_xy.diff(y) + sigma_xz.diff(z)
        div_sigma_y = sigma_xy.diff(x) + sigma_yy.diff(y) + sigma_yz.diff(z)
        div_sigma_z = sigma_xz.diff(x) + sigma_yz.diff(y) + sigma_zz.diff(z)

        # Forcing: f = -div(σ) so that div(σ) + f = 0
        forc_exprs = [-div_sigma_x, -div_sigma_y, -div_sigma_z]

        # Add forcing contribution to existing forcing (initialized to zeros)
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]

    def traction_values(self, pts: Array, normals: Array) -> Array:
        """Compute exact traction t = σ·n at given points.

        Parameters
        ----------
        pts : Array
            Physical coordinates. Shape: (nvars, npts)
        normals : Array
            Outward unit normals. Shape: (npts, nvars)

        Returns
        -------
        Array
            Traction components. Shape: (npts, nvars)
        """
        bkd = self._bkd
        nvars = self._nvars_el
        # flux shape: (nvars, npts, nvars) from list-of-lists expression;
        # sigma[i, pt, j] is σ_ij, so t_i = sum_j σ_ij * n_j
        sigma = self.functions["flux"](pts)
        tractions = []
        for i in range(nvars):
            t_i = sigma[i, :, 0] * normals[:, 0]
            for j in range(1, nvars):
                t_i = t_i + sigma[i, :, j] * normals[:, j]
            tractions.append(t_i[:, None])
        return bkd.hstack(tractions)

    def robin_values(
        self, pts: Array, normals: Array, alpha: float, beta: float
    ) -> Array:
        """Compute vector Robin BC values g = α*u + β*(σ·n).

        Parameters
        ----------
        pts : Array
            Physical coordinates. Shape: (nvars, npts)
        normals : Array
            Outward unit normals. Shape: (npts, nvars)
        alpha : float
            Coefficient for displacement term.
        beta : float
            Coefficient for traction term.

        Returns
        -------
        Array
            Robin values. Shape: (npts, nvars)
        """
        u = self.functions["solution"](pts)  # (npts, nvars)
        traction = self.traction_values(pts, normals)  # (npts, nvars)
        return alpha * u + beta * traction
