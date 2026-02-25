"""Shallow Shelf Approximation (SSA) physics for spectral collocation.

Implements the Shallow Shelf Approximation for ice sheet dynamics.

The SSA velocity equations (2D):
    div(2*mu*H*strain_tensor) - C*velocity - H*rho*g*grad(s) = f

where:
    velocity = (u, v) = ice velocity
    H = ice thickness
    s = H + b = surface elevation
    b = bed elevation
    mu = effective viscosity = 0.5 * A^(-1/n) * strain_rate^((1-n)/n)
    strain_rate = effective strain rate
    C = friction coefficient
    rho = ice density
    g = gravitational acceleration
    n = Glen's flow law exponent (typically 3)

The depth equation (mass conservation):
    dH/dt + div(H*velocity) = f_H
"""

from typing import Callable, Optional, Tuple, Union

from pyapprox.pde.collocation.physics.base import (
    AbstractPhysics,
    AbstractVectorPhysics,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ShallowShelfVelocityPhysics(AbstractVectorPhysics[Array]):
    """Shallow Shelf Approximation velocity physics.

    Solves for ice velocity given ice thickness. This is a nonlinear
    elliptic system due to the strain-rate-dependent viscosity.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis.
    bkd : Backend
        Computational backend.
    depth : Array
        Ice thickness H. Shape: (npts,)
    bed : Array
        Bed elevation b. Shape: (npts,)
    friction : float or Array
        Friction coefficient C.
    A : float
        Rate factor in Glen's flow law.
    rho : float
        Ice density (kg/m^3).
    g : float
        Gravitational acceleration (default: 9.81).
    forcing : Callable[[float], Array], optional
        Velocity forcing term.
    eps : float
        Regularization for strain rate (default: 1e-12).
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        depth: Array,
        bed: Array,
        friction: Union[float, Array],
        A: float,
        rho: float,
        g: float = 9.81,
        forcing: Optional[Callable[[float], Array]] = None,
        eps: float = 1e-12,
    ):
        if basis.ndim() != 2:
            raise ValueError("ShallowShelfVelocityPhysics requires a 2D basis")

        super().__init__(basis, bkd, ncomponents=2)

        self._depth = depth
        self._bed = bed
        self._A = A
        self._rho = rho
        self._g = g
        self._n = 3  # Glen's flow law exponent
        self._eps = eps
        self._forcing_func = forcing

        npts = basis.npts()

        # Store friction
        if isinstance(friction, (int, float)):
            self._friction = bkd.full((npts,), float(friction))
        else:
            self._friction = friction

        # Precompute derivative matrices
        self._Dx = basis.derivative_matrix(1, 0)
        self._Dy = basis.derivative_matrix(1, 1)

        # Precompute surface gradient (driving stress)
        surface = depth + bed
        self._surf_grad_x = self._Dx @ surface
        self._surf_grad_y = self._Dy @ surface

        # Constant in viscosity: 0.5 * A^(-1/n)
        self._visc_const = 0.5 * self._A ** (-1.0 / self._n)

    def set_depth(self, depth: Array) -> None:
        """Update ice thickness.

        Parameters
        ----------
        depth : Array
            New ice thickness. Shape: (npts,)
        """
        self._depth = depth
        surface = depth + self._bed
        self._surf_grad_x = self._Dx @ surface
        self._surf_grad_y = self._Dy @ surface

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        npts = self.npts()
        if self._forcing_func is None:
            return self._bkd.zeros((2 * npts,))
        if callable(self._forcing_func):
            return self._forcing_func(time)
        return self._forcing_func

    def _split_state(self, state: Array) -> Tuple[Array, Array]:
        """Split state into velocity components."""
        npts = self.npts()
        u = state[:npts]
        v = state[npts:]
        return u, v

    def _combine_state(self, u: Array, v: Array) -> Array:
        """Combine velocity components into state."""
        return self._bkd.hstack([u, v])

    def _effective_strain_rate(
        self, ux: Array, uy: Array, vx: Array, vy: Array
    ) -> Array:
        """Compute effective strain rate.

        strain_rate = sqrt(ux^2 + vy^2 + ux*vy + 0.25*(uy+vx)^2 + eps)
        """
        return (ux**2 + vy**2 + ux * vy + 0.25 * (uy + vx) ** 2 + self._eps) ** 0.5

    def _compute_viscosity(self, ux: Array, uy: Array, vx: Array, vy: Array) -> Array:
        """Compute effective viscosity.

        mu = 0.5 * A^(-1/n) * strain_rate^((1-n)/n)
        """
        strain_rate = self._effective_strain_rate(ux, uy, vx, vy)
        return self._visc_const * strain_rate ** ((1.0 - self._n) / self._n)

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual.

        residual = div(2*mu*H*strain_tensor) - C*vel - H*rho*g*grad(s) + f
        """
        npts = self.npts()

        u, v = self._split_state(state)

        # Compute velocity gradients
        ux = self._Dx @ u
        uy = self._Dy @ u
        vx = self._Dx @ v
        vy = self._Dy @ v

        # Compute effective viscosity
        mu = self._compute_viscosity(ux, uy, vx, vy)

        # Membrane stress components (strain tensor entries)
        # tau_xx = 2*mu*H*(2*ux + vy)
        # tau_yy = 2*mu*H*(ux + 2*vy)
        # tau_xy = 2*mu*H*0.5*(uy + vx) = mu*H*(uy + vx)
        H = self._depth
        coef = 2.0 * mu * H

        tau_xx = coef * (2.0 * ux + vy)
        tau_yy = coef * (ux + 2.0 * vy)
        tau_xy = coef * 0.5 * (uy + vx)

        # Divergence of stress tensor
        # d/dx(tau_xx) + d/dy(tau_xy)
        # d/dx(tau_xy) + d/dy(tau_yy)
        div_tau_x = self._Dx @ tau_xx + self._Dy @ tau_xy
        div_tau_y = self._Dx @ tau_xy + self._Dy @ tau_yy

        # Friction
        friction_x = self._friction * u
        friction_y = self._friction * v

        # Driving stress: H * rho * g * grad(s)
        drive_coef = H * self._rho * self._g
        drive_x = drive_coef * self._surf_grad_x
        drive_y = drive_coef * self._surf_grad_y

        # Residual = div(tau) - friction - driving + forcing
        forcing = self._get_forcing(time)
        res_u = div_tau_x - friction_x - drive_x + forcing[:npts]
        res_v = div_tau_y - friction_y - drive_y + forcing[npts:]

        return self._combine_state(res_u, res_v)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute Jacobian via finite differences.

        The analytical Jacobian is complex due to the nonlinear viscosity,
        so we use finite differences for robustness.
        """
        bkd = self._bkd
        nstates = self.nstates()

        jacobian = bkd.zeros((nstates, nstates))

        eps = 1e-7
        for j in range(nstates):
            state_plus = bkd.copy(state)
            state_plus[j] = state_plus[j] + eps
            state_minus = bkd.copy(state)
            state_minus[j] = state_minus[j] - eps

            res_plus = self.residual(state_plus, time)
            res_minus = self.residual(state_minus, time)

            for i in range(nstates):
                jacobian[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        return jacobian


class ShallowShelfDepthPhysics(AbstractPhysics[Array]):
    """Shallow shelf depth evolution physics (mass conservation).

    dH/dt = -div(H * velocity) + f

    This is a first-order hyperbolic equation for depth given velocity.
    Note: This physics requires a 2D basis since shallow shelf equations
    are inherently 2D (ice sheet flow over a 2D domain).

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis.
    bkd : Backend
        Computational backend.
    forcing : Callable[[float], Array], optional
        Source/sink term (e.g., surface mass balance).
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        forcing: Optional[Callable[[float], Array]] = None,
    ):
        if basis.ndim() != 2:
            raise ValueError("ShallowShelfDepthPhysics requires a 2D basis")

        super().__init__(basis, bkd)
        self._forcing_func = forcing
        self._velocities: Optional[Array] = None

        # Precompute derivative matrices (2D)
        self._D1_matrices = [basis.derivative_matrix(1, dim) for dim in range(2)]

    def ncomponents(self) -> int:
        return 1

    def set_velocities(self, velocities: Array) -> None:
        """Set velocity field for advection.

        Parameters
        ----------
        velocities : Array
            Velocity components [u, v] or [u]. Shape: (ndim * npts,)
        """
        self._velocities = velocities

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        npts = self.npts()
        if self._forcing_func is None:
            return self._bkd.zeros((npts,))
        if callable(self._forcing_func):
            return self._forcing_func(time)
        return self._forcing_func

    def residual(self, state: Array, time: float) -> Array:
        """Compute residual -div(H*velocity) + f."""
        bkd = self._bkd
        npts = self.npts()
        ndim = self._basis.ndim()

        H = state  # Ice thickness

        if self._velocities is None:
            raise RuntimeError("Velocities must be set before computing residual")

        # Flux = H * velocity
        residual = bkd.zeros((npts,))
        for dim in range(ndim):
            vel_component = self._velocities[dim * npts : (dim + 1) * npts]
            flux = H * vel_component
            residual = residual - self._D1_matrices[dim] @ flux

        # Add forcing
        residual = residual + self._get_forcing(time)

        return residual

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute Jacobian d(residual)/dH."""
        bkd = self._bkd
        npts = self.npts()
        ndim = self._basis.ndim()

        # d/dH[-div(H*vel)] = -div(vel * I) = -sum_dim D_dim @ diag(vel_dim)
        jacobian = bkd.zeros((npts, npts))

        if self._velocities is None:
            raise RuntimeError("Velocities must be set before computing Jacobian")

        for dim in range(ndim):
            vel_component = self._velocities[dim * npts : (dim + 1) * npts]
            jacobian = jacobian - self._D1_matrices[dim] @ bkd.diag(vel_component)

        return jacobian


class ShallowShelfDepthVelocityPhysics(AbstractVectorPhysics[Array]):
    """Coupled shallow shelf depth and velocity physics.

    Combines the depth equation (mass conservation, transient) with the
    velocity equations (momentum balance, steady/elliptic) into a single
    coupled system suitable for implicit time integration.

    State vector: [H, u, v] with shape (3*npts,)
    - H: ice thickness (transient: dH/dt + div(H*vel) = f_H)
    - u, v: ice velocity (steady: div(stress) - friction - driving = f_vel)

    The mass matrix is block-diagonal [[I, 0], [0, 0]] reflecting that
    only depth has a time derivative. The velocity equations are algebraic
    constraints solved simultaneously via Newton iteration.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis.
    bkd : Backend
        Computational backend.
    bed : Array
        Bed elevation b. Shape: (npts,)
    friction : float or Array
        Friction coefficient C.
    A : float
        Rate factor in Glen's flow law.
    rho : float
        Ice density (kg/m^3).
    g : float
        Gravitational acceleration (default: 9.81).
    depth_forcing : Callable[[float], Array], optional
        Forcing for depth equation. Shape: (npts,)
    velocity_forcing : Callable[[float], Array], optional
        Forcing for velocity equations. Shape: (2*npts,)
    eps : float
        Regularization for strain rate (default: 1e-12).
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        bed: Array,
        friction: Union[float, Array],
        A: float,
        rho: float,
        g: float = 9.81,
        depth_forcing: Optional[Callable[[float], Array]] = None,
        velocity_forcing: Optional[Callable[[float], Array]] = None,
        eps: float = 1e-12,
    ):
        if basis.ndim() != 2:
            raise ValueError("ShallowShelfDepthVelocityPhysics requires a 2D basis")

        super().__init__(basis, bkd, ncomponents=3)

        npts = basis.npts()

        # Initialize depth to zero; will be set from state in residual
        initial_depth = bkd.zeros((npts,))

        self._depth_physics = ShallowShelfDepthPhysics(
            basis, bkd, forcing=depth_forcing
        )
        self._velocity_physics = ShallowShelfVelocityPhysics(
            basis,
            bkd,
            depth=initial_depth,
            bed=bed,
            friction=friction,
            A=A,
            rho=rho,
            g=g,
            forcing=velocity_forcing,
            eps=eps,
        )

        # Cache mass matrix
        self._cached_mass_matrix = bkd.zeros((3 * npts, 3 * npts))
        for i in range(npts):
            self._cached_mass_matrix[i, i] = 1.0

    def _split_state(self, state: Array) -> Tuple[Array, Array, Array]:
        """Split state [H, u, v] into components."""
        npts = self.npts()
        H = state[:npts]
        u = state[npts : 2 * npts]
        v = state[2 * npts :]
        return H, u, v

    def _update_cross_dependencies(self, H: Array, u: Array, v: Array) -> None:
        """Update cross-physics dependencies.

        Depth physics needs current velocity for flux computation.
        Velocity physics needs current depth for stress computation.
        """
        self.npts()
        self._depth_physics.set_velocities(self._bkd.hstack([u, v]))
        self._velocity_physics.set_depth(H)

    def residual(self, state: Array, time: float) -> Array:
        """Compute coupled residual [R_H, R_u, R_v].

        R_H = -div(H*vel) + f_H  (depth: transient)
        R_u, R_v = SSA momentum   (velocity: steady/algebraic)
        """
        H, u, v = self._split_state(state)
        self._update_cross_dependencies(H, u, v)

        depth_res = self._depth_physics.residual(H, time)
        vel_state = self._bkd.hstack([u, v])
        vel_res = self._velocity_physics.residual(vel_state, time)

        return self._bkd.hstack([depth_res, vel_res])

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute full coupled Jacobian via finite differences.

        The Jacobian is a 3x3 block matrix with cross-coupling terms
        (depth depends on velocity through div(H*vel), velocity depends
        on depth through surface gradient and stress).
        """
        bkd = self._bkd
        nstates = self.nstates()
        eps = 1e-7

        jacobian = bkd.zeros((nstates, nstates))

        for j in range(nstates):
            state_plus = bkd.copy(state)
            state_plus[j] = state_plus[j] + eps
            state_minus = bkd.copy(state)
            state_minus[j] = state_minus[j] - eps

            res_plus = self.residual(state_plus, time)
            res_minus = self.residual(state_minus, time)

            jacobian[:, j] = (res_plus - res_minus) / (2 * eps)

        return jacobian

    def mass_matrix(self) -> Array:
        """Return mass matrix [[I, 0], [0, 0]].

        Only depth has a time derivative. Velocity equations are
        algebraic (steady).
        """
        return self._cached_mass_matrix

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix: keep depth component, zero velocity components.

        For state [H, u, v], returns [H, 0, 0].
        """
        npts = self.npts()
        result = self._bkd.copy(vec)
        result[npts:] = 0.0
        return result


def create_shallow_shelf_velocity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    depth: Array,
    bed: Array,
    friction: Union[float, Array],
    A: float = 1e-16,
    rho: float = 917.0,
    g: float = 9.81,
    forcing: Optional[Callable[[float], Array]] = None,
    eps: float = 1e-12,
) -> ShallowShelfVelocityPhysics[Array]:
    """Create shallow shelf velocity physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis.
    bkd : Backend
        Computational backend.
    depth : Array
        Ice thickness.
    bed : Array
        Bed elevation.
    friction : float or Array
        Friction coefficient.
    A : float
        Rate factor (default: 1e-16).
    rho : float
        Ice density (default: 917.0).
    g : float
        Gravitational acceleration (default: 9.81).
    forcing : Callable, optional
        Velocity forcing.
    eps : float
        Strain rate regularization.

    Returns
    -------
    ShallowShelfVelocityPhysics
        SSA velocity physics.
    """
    return ShallowShelfVelocityPhysics(
        basis=basis,
        bkd=bkd,
        depth=depth,
        bed=bed,
        friction=friction,
        A=A,
        rho=rho,
        g=g,
        forcing=forcing,
        eps=eps,
    )


def create_shallow_shelf_depth(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    forcing: Optional[Callable[[float], Array]] = None,
) -> ShallowShelfDepthPhysics[Array]:
    """Create shallow shelf depth physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    forcing : Callable, optional
        Source/sink term.

    Returns
    -------
    ShallowShelfDepthPhysics
        Depth evolution physics.
    """
    return ShallowShelfDepthPhysics(
        basis=basis,
        bkd=bkd,
        forcing=forcing,
    )


def create_shallow_shelf_depth_velocity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    bed: Array,
    friction: Union[float, Array],
    A: float = 1e-16,
    rho: float = 917.0,
    g: float = 9.81,
    depth_forcing: Optional[Callable[[float], Array]] = None,
    velocity_forcing: Optional[Callable[[float], Array]] = None,
    eps: float = 1e-12,
) -> ShallowShelfDepthVelocityPhysics[Array]:
    """Create coupled shallow shelf depth-velocity physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis.
    bkd : Backend
        Computational backend.
    bed : Array
        Bed elevation.
    friction : float or Array
        Friction coefficient.
    A : float
        Rate factor (default: 1e-16).
    rho : float
        Ice density (default: 917.0).
    g : float
        Gravitational acceleration (default: 9.81).
    depth_forcing : Callable, optional
        Forcing for depth equation.
    velocity_forcing : Callable, optional
        Forcing for velocity equations.
    eps : float
        Strain rate regularization.

    Returns
    -------
    ShallowShelfDepthVelocityPhysics
        Coupled depth-velocity physics.
    """
    return ShallowShelfDepthVelocityPhysics(
        basis=basis,
        bkd=bkd,
        bed=bed,
        friction=friction,
        A=A,
        rho=rho,
        g=g,
        depth_forcing=depth_forcing,
        velocity_forcing=velocity_forcing,
        eps=eps,
    )
