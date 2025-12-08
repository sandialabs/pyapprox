"""Linear elasticity physics for spectral collocation.

Implements the 2D linear elasticity equations:
    -div(σ) + f = 0

where:
    σ = λ*tr(ε)*I + 2μ*ε  (stress tensor)
    ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)  (strain tensor)
    u = (u, v) is the displacement field
    λ, μ are Lamé parameters
"""

from typing import Generic, Optional, Callable, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.typing.pde.collocation.physics.base import AbstractVectorPhysics


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
        self._Dxy = self._Dx @ self._Dy  # d²/dxdy (mixed derivative)

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
        npts = self.npts()

        # For constant Lamé parameters, use scalar values
        if self._lambda_value is not None and self._mu_value is not None:
            lam = self._lambda_value
            mu = self._mu_value

            # Block Jacobians
            # J_uu: d(res_u)/du = (λ + 2μ)*D²_x + μ*D²_y
            J_uu = (lam + 2.0 * mu) * self._Dxx + mu * self._Dyy

            # J_uv: d(res_u)/dv = (λ + μ)*D_x @ D_y
            J_uv = (lam + mu) * self._Dxy

            # J_vu: d(res_v)/du = (λ + μ)*D_y @ D_x
            # Note: D_y @ D_x = D_x @ D_y for smooth functions (mixed partials commute)
            J_vu = (lam + mu) * self._Dxy

            # J_vv: d(res_v)/dv = μ*D²_x + (λ + 2μ)*D²_y
            J_vv = mu * self._Dxx + (lam + 2.0 * mu) * self._Dyy
        else:
            # Variable Lamé parameters - more complex Jacobian
            raise NotImplementedError(
                "Variable Lamé parameters not yet implemented"
            )

        # Assemble block Jacobian
        # [[J_uu, J_uv], [J_vu, J_vv]]
        top_row = bkd.concatenate([J_uu, J_uv], axis=1)
        bottom_row = bkd.concatenate([J_vu, J_vv], axis=1)
        jacobian = bkd.concatenate([top_row, bottom_row], axis=0)

        return jacobian


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
