"""Linear elasticity physics for spectral collocation.

Implements the 2D linear elasticity equations:
    -div(σ) + f = 0

where:
    σ = λ*tr(ε)*I + 2μ*ε  (stress tensor)
    ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)  (strain tensor)
    u = (u, v) is the displacement field
    λ, μ are Lamé parameters
"""

from typing import Any, Callable, Generic, Optional, Tuple

from pyapprox.pde.collocation.physics.base import AbstractVectorPhysics
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class LinearElasticityPhysics(AbstractVectorPhysics[Array], Generic[Array]):
    """2D Linear elasticity physics.

    Implements the equilibrium equation:
        -div(σ) + f = 0

    where the stress tensor is:
        σ = λ*tr(ε)*I + 2μ*ε

    and the strain tensor is:
        ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)

    The residual is formulated as:
        residual = div(σ) + f

    So that residual = 0 at equilibrium.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    lamda : float or Array
        Lamé's first parameter λ. If Array, shape: (npts,).
    mu : float or Array
        Shear modulus μ. If Array, shape: (npts,).
    forcing : Callable[[float], Array] or Array, optional
        Forcing term. If callable, takes time and returns (2*npts,) array
        with [f_x, f_y] components stacked.
        If Array, shape: (2*npts,). Default: None (no forcing)
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        lamda: float,
        mu: float,
        forcing: Optional[Callable[[float], Array]] = None,
    ):
        if basis.ndim() != 2:
            raise ValueError("LinearElasticityPhysics requires 2D basis")

        super().__init__(basis, bkd, ncomponents=2)

        npts = basis.npts()

        # Store Lamé parameters
        if isinstance(lamda, (int, float)):
            self._lambda_array = bkd.full((npts,), float(lamda))
            self._lambda_value = float(lamda)
        else:
            self._lambda_array = lamda
            self._lambda_value = None

        if isinstance(mu, (int, float)):
            self._mu_array = bkd.full((npts,), float(mu))
            self._mu_value = float(mu)
        else:
            self._mu_array = mu
            self._mu_value = None

        # Store forcing function
        self._forcing_func = forcing

        # Precompute derivative matrices
        self._Dx = basis.derivative_matrix(1, 0)  # d/dx
        self._Dy = basis.derivative_matrix(1, 1)  # d/dy
        self._Dxx = basis.derivative_matrix(2, 0)  # d²/dx²
        self._Dyy = basis.derivative_matrix(2, 1)  # d²/dy²
        # Mixed derivatives: Dx@Dy != Dy@Dx on curvilinear domains
        self._DxDy = self._Dx @ self._Dy  # Dx applied after Dy
        self._DyDx = self._Dy @ self._Dx  # Dy applied after Dx

    # ------------------------------------------------------------------
    # Material property setters
    # ------------------------------------------------------------------

    def set_mu(self, mu_values: Any) -> None:
        """Set shear modulus (scalar or per-point array).

        Parameters
        ----------
        mu_values : float or Array
            Shear modulus values. Must be positive.
        """
        mu_arr = self._bkd.ravel(self._bkd.asarray(mu_values))
        min_val = self._bkd.to_float(self._bkd.min(mu_arr))
        if min_val <= 0.0:
            raise ValueError(f"mu must be positive; found min {min_val:.2e}")
        if isinstance(mu_values, (int, float)):
            npts = self.npts()
            self._mu_array = self._bkd.full((npts,), float(mu_values))
            self._mu_value = float(mu_values)
        else:
            self._mu_array = mu_values
            self._mu_value = None

    def set_lamda(self, lamda_values: Any) -> None:
        """Set Lame's first parameter (scalar or per-point array).

        Parameters
        ----------
        lamda_values : float or Array
            Lame's first parameter values. Must be non-negative.
        """
        lam_arr = self._bkd.ravel(self._bkd.asarray(lamda_values))
        min_val = self._bkd.to_float(self._bkd.min(lam_arr))
        if min_val < 0.0:
            raise ValueError(f"lamda must be non-negative; found min {min_val:.2e}")
        if isinstance(lamda_values, (int, float)):
            npts = self.npts()
            self._lambda_array = self._bkd.full((npts,), float(lamda_values))
            self._lambda_value = float(lamda_values)
        else:
            self._lambda_array = lamda_values
            self._lambda_value = None

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        nstates = self.nstates()
        if self._forcing_func is None:
            return self._bkd.zeros((nstates,))
        if callable(self._forcing_func):
            return self._forcing_func(time)
        return self._forcing_func

    def _extract_components(self, state: Array) -> Tuple[Array, Array]:
        """Extract u and v components from state vector.

        State is ordered as [u_0, u_1, ..., u_{n-1}, v_0, v_1, ..., v_{n-1}]

        Parameters
        ----------
        state : Array
            Full state vector. Shape: (2*npts,)

        Returns
        -------
        Tuple[Array, Array]
            u and v components, each shape (npts,)
        """
        npts = self.npts()
        u = state[:npts]
        v = state[npts:]
        return u, v

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        The residual is: div(σ) + forcing

        where σ = λ*tr(ε)*I + 2μ*ε

        Parameters
        ----------
        state : Array
            Solution state [u, v]. Shape: (2*npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual [res_u, res_v]. Shape: (2*npts,)
        """
        bkd = self._bkd
        npts = self.npts()

        u, v = self._extract_components(state)

        # Compute strain components
        ux = self._Dx @ u  # du/dx
        uy = self._Dy @ u  # du/dy
        vx = self._Dx @ v  # dv/dx
        vy = self._Dy @ v  # dv/dy

        exx = ux
        exy = 0.5 * (uy + vx)
        eyy = vy

        # Trace of strain
        trace_e = exx + eyy

        # Stress tensor components: σ = λ*tr(ε)*I + 2μ*ε
        two_mu = 2.0 * self._mu_array
        sigma_xx = self._lambda_array * trace_e + two_mu * exx
        sigma_xy = two_mu * exy
        sigma_yy = self._lambda_array * trace_e + two_mu * eyy

        # Compute divergence of stress
        # div(σ)_x = ∂σ_xx/∂x + ∂σ_xy/∂y
        # div(σ)_y = ∂σ_xy/∂x + ∂σ_yy/∂y
        div_sigma_x = self._Dx @ sigma_xx + self._Dy @ sigma_xy
        div_sigma_y = self._Dx @ sigma_xy + self._Dy @ sigma_yy

        # Get forcing
        forcing = self._get_forcing(time)
        fx = forcing[:npts]
        fy = forcing[npts:]

        # Residual: div(σ) + f
        res_u = div_sigma_x + fx
        res_v = div_sigma_y + fy

        return bkd.concatenate([res_u, res_v])

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/d(u,v).

        For linear elasticity with constant Lamé parameters, the Jacobian
        is constant (independent of state).

        The Jacobian has block structure:
            J = [[J_uu, J_uv],
                 [J_vu, J_vv]]

        where:
            J_uu = (λ + 2μ)*D²_x + μ*D²_y
            J_uv = (λ + μ)*D_x @ D_y
            J_vu = (λ + μ)*D_y @ D_x
            J_vv = μ*D²_x + (λ + 2μ)*D²_y

        Parameters
        ----------
        state : Array
            Solution state. Shape: (2*npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (2*npts, 2*npts)
        """
        bkd = self._bkd
        self.npts()

        # For constant Lamé parameters, use scalar values
        if self._lambda_value is not None and self._mu_value is not None:
            lam = self._lambda_value
            mu = self._mu_value

            # Block Jacobians derived from residual:
            #   res_u = Dx @ sigma_xx + Dy @ sigma_xy + fx
            #   res_v = Dx @ sigma_xy + Dy @ sigma_yy + fy
            #
            # J_uu: d(res_u)/du = (λ+2μ)*Dxx + μ*Dyy
            J_uu = (lam + 2.0 * mu) * self._Dxx + mu * self._Dyy

            # J_uv: d(res_u)/dv = λ*(Dx@Dy) + μ*(Dy@Dx)
            # Note: Dx@Dy != Dy@Dx on curvilinear domains
            J_uv = lam * self._DxDy + mu * self._DyDx

            # J_vu: d(res_v)/du = μ*(Dx@Dy) + λ*(Dy@Dx)
            J_vu = mu * self._DxDy + lam * self._DyDx

            # J_vv: d(res_v)/dv = μ*Dxx + (λ+2μ)*Dyy
            J_vv = mu * self._Dxx + (lam + 2.0 * mu) * self._Dyy
        else:
            # Variable Lamé parameters: use Dx @ diag(coeff) @ Dx form
            # which automatically handles the product rule
            # d/dx(coeff * du/dx) = coeff * d²u/dx² + dcoeff/dx * du/dx
            lam = self._lambda_array
            mu = self._mu_array
            Dx, Dy = self._Dx, self._Dy

            diag_lam_2mu = bkd.diag(lam + 2.0 * mu)
            diag_lam = bkd.diag(lam)
            diag_mu = bkd.diag(mu)

            J_uu = Dx @ diag_lam_2mu @ Dx + Dy @ diag_mu @ Dy
            J_uv = Dx @ diag_lam @ Dy + Dy @ diag_mu @ Dx
            J_vu = Dx @ diag_mu @ Dy + Dy @ diag_lam @ Dx
            J_vv = Dx @ diag_mu @ Dx + Dy @ diag_lam_2mu @ Dy

        # Assemble block Jacobian
        # [[J_uu, J_uv], [J_vu, J_vv]]
        top_row = bkd.concatenate([J_uu, J_uv], axis=1)
        bottom_row = bkd.concatenate([J_vu, J_vv], axis=1)
        jacobian = bkd.concatenate([top_row, bottom_row], axis=0)

        return jacobian

    # ------------------------------------------------------------------
    # Residual sensitivity to material parameters (2D)
    # ------------------------------------------------------------------

    def residual_mu_sensitivity(
        self, state: Array, time: float, delta_mu: Array
    ) -> Array:
        """Compute d(residual)/d(mu_field) * delta_mu.

        For linear elasticity sigma = lam*tr(eps)*I + 2*mu*eps,
        d(sigma)/d(mu) = 2*eps, so:
            d(res_u)/d(mu)*delta_mu = Dx@(delta_mu*2*exx) + Dy@(delta_mu*2*exy)
            d(res_v)/d(mu)*delta_mu = Dx@(delta_mu*2*exy) + Dy@(delta_mu*2*eyy)

        Parameters
        ----------
        state : Array
            Displacement state [u, v]. Shape: (2*npts,)
        time : float
            Current time.
        delta_mu : Array
            Perturbation in mu field. Shape: (npts,)

        Returns
        -------
        Array
            Residual sensitivity. Shape: (2*npts,)
        """
        bkd = self._bkd
        u, v = self._extract_components(state)

        ux = self._Dx @ u
        uy = self._Dy @ u
        vx = self._Dx @ v
        vy = self._Dy @ v

        exx = ux
        exy = 0.5 * (uy + vx)
        eyy = vy

        sens_u = self._Dx @ (delta_mu * 2.0 * exx) + self._Dy @ (delta_mu * 2.0 * exy)
        sens_v = self._Dx @ (delta_mu * 2.0 * exy) + self._Dy @ (delta_mu * 2.0 * eyy)
        return bkd.concatenate([sens_u, sens_v])

    def residual_lamda_sensitivity(
        self, state: Array, time: float, delta_lam: Array
    ) -> Array:
        """Compute d(residual)/d(lamda_field) * delta_lam.

        For linear elasticity sigma = lam*tr(eps)*I + 2*mu*eps,
        d(sigma)/d(lam) = tr(eps)*I, so:
            d(res_u)/d(lam)*delta_lam = Dx@(delta_lam * trace_e)
            d(res_v)/d(lam)*delta_lam = Dy@(delta_lam * trace_e)

        Parameters
        ----------
        state : Array
            Displacement state [u, v]. Shape: (2*npts,)
        time : float
            Current time.
        delta_lam : Array
            Perturbation in lambda field. Shape: (npts,)

        Returns
        -------
        Array
            Residual sensitivity. Shape: (2*npts,)
        """
        bkd = self._bkd
        u, v = self._extract_components(state)

        ux = self._Dx @ u
        vy = self._Dy @ v
        trace_e = ux + vy

        sens_u = self._Dx @ (delta_lam * trace_e)
        sens_v = self._Dy @ (delta_lam * trace_e)
        return bkd.concatenate([sens_u, sens_v])

    def compute_interface_flux(
        self, state: Array, boundary_indices: Array, normal: Array
    ) -> Array:
        """Compute traction at boundary for DtN domain decomposition.

        Computes t = σ · n at the specified boundary points, where:
        - σ is the stress tensor
        - n is the outward unit normal

        Parameters
        ----------
        state : Array
            Solution state [u, v]. Shape: (2*npts,)
        boundary_indices : Array
            Mesh indices at interface. Shape: (nboundary,)
        normal : Array
            Outward unit normal. Shape: (2,)

        Returns
        -------
        Array
            Traction [t_x, t_y] at boundary points.
            Shape: (2*nboundary,) with component-stacked ordering.
        """
        bkd = self._bkd
        boundary_indices.shape[0]

        u, v = self._extract_components(state)

        # Compute strain components at boundary
        ux = (self._Dx @ u)[boundary_indices]
        uy = (self._Dy @ u)[boundary_indices]
        vx = (self._Dx @ v)[boundary_indices]
        vy = (self._Dy @ v)[boundary_indices]

        exx = ux
        exy = 0.5 * (uy + vx)
        eyy = vy
        trace_e = exx + eyy

        # Stress tensor at boundary
        lam = self._lambda_array[boundary_indices]
        mu = self._mu_array[boundary_indices]
        sigma_xx = lam * trace_e + 2.0 * mu * exx
        sigma_xy = 2.0 * mu * exy
        sigma_yy = lam * trace_e + 2.0 * mu * eyy

        # Traction t = σ · n
        nx = bkd.to_float(normal[0])
        ny = bkd.to_float(normal[1])
        traction_x = sigma_xx * nx + sigma_xy * ny
        traction_y = sigma_xy * nx + sigma_yy * ny

        # Component-stacked ordering: [t_x_0, ..., t_x_n, t_y_0, ..., t_y_n]
        return bkd.concatenate([traction_x, traction_y])


def create_linear_elasticity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    lamda: float,
    mu: float,
    forcing: Optional[Callable[[float], Array]] = None,
) -> LinearElasticityPhysics[Array]:
    """Create linear elasticity physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis.
    bkd : Backend
        Computational backend.
    lamda : float
        Lamé's first parameter λ.
    mu : float
        Shear modulus μ.
    forcing : Callable or Array, optional
        Source term.

    Returns
    -------
    LinearElasticityPhysics
        Linear elasticity physics instance.
    """
    return LinearElasticityPhysics(basis, bkd, lamda, mu, forcing)
