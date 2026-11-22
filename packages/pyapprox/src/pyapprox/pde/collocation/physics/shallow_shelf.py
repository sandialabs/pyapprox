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

The momentum and mass terms are evaluated by two kernels that take every
field as an argument. The velocity physics, the depth physics and the
coupled physics each read their own fields and pass them in, so the
coupled physics shares the formulas without holding, or writing into,
the two single-field physics.
"""

from typing import Generic, List, Optional, Tuple

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.physics.base import (
    AbstractPhysics,
    AbstractVectorPhysics,
)
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalCoefficientProtocol,
    as_nodal_coefficient,
    declared_time_invariant,
    optional_nodal_coefficient,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class _ShelfMomentum(Generic[Array]):
    """The SSA momentum residual and its derivatives, every field given.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    Dx, Dy : Array
        First-derivative matrices. Shape: (npts, npts)
    A : float
        Rate factor in Glen's flow law.
    rho : float
        Ice density.
    g : float
        Gravitational acceleration.
    eps : float
        Regularization of the squared strain rate.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        Dx: Array,
        Dy: Array,
        A: float,
        rho: float,
        g: float,
        eps: float,
    ) -> None:
        self._bkd = bkd
        self._Dx = Dx
        self._Dy = Dy
        self._rho = rho
        self._g = g
        self._eps = eps
        self._n = 3  # Glen's flow law exponent
        # Constant in viscosity: 0.5 * A^(-1/n). The annotation is
        # required: float ** float is typed Any in typeshed.
        self._visc_const: float = 0.5 * A ** (-1.0 / self._n)

    def _gradients(self, velocity: Array) -> Tuple[Array, Array, Array, Array]:
        """(u_x, u_y, v_x, v_y) of the stacked velocity [u, v]."""
        npts = velocity.shape[0] // 2
        u, v = velocity[:npts], velocity[npts:]
        return self._Dx @ u, self._Dy @ u, self._Dx @ v, self._Dy @ v

    def _viscosity_derivatives(
        self, ux: Array, uy: Array, vx: Array, vy: Array
    ) -> Tuple[Array, Array]:
        """Compute the viscosity and its derivative w.r.t. the
        regularized squared strain rate.

        With :math:`Q + \\epsilon` the regularized squared effective
        strain rate, :math:`Q = u_x^2 + v_y^2 + u_x v_y + (u_y+v_x)^2/4`,
        and :math:`p = (1-n)/(2n)`:

        .. math::

            \\mu = c (Q+\\epsilon)^p, \\qquad
            \\mu_Q = d\\mu/dQ = p\\,\\mu/(Q+\\epsilon).

        Returns (mu, mu_q), each shape (npts,).
        """
        strain_sq = ux**2 + vy**2 + ux * vy + 0.25 * (uy + vx) ** 2 + self._eps
        power = (1.0 - self._n) / (2.0 * self._n)
        mu = self._visc_const * strain_sq**power
        mu_q = power * mu / strain_sq
        return mu, mu_q

    def stress(self, velocity: Array, depth: Array) -> Tuple[Array, Array, Array]:
        """The membrane stresses (tau_xx, tau_yy, tau_xy), tau = 2 mu H a
        with a_xx = 2 u_x + v_y, a_yy = u_x + 2 v_y, a_xy = (u_y + v_x)/2.
        Each shape: (npts,)"""
        ux, uy, vx, vy = self._gradients(velocity)
        mu, _ = self._viscosity_derivatives(ux, uy, vx, vy)
        coef = 2.0 * mu * depth
        tau_xx = coef * (2.0 * ux + vy)
        tau_yy = coef * (ux + 2.0 * vy)
        tau_xy = coef * 0.5 * (uy + vx)
        return tau_xx, tau_yy, tau_xy

    def stress_jacobian(
        self, velocity: Array, depth: Array
    ) -> Tuple[Array, Array, Array]:
        """The velocity Jacobians of (tau_xx, tau_yy, tau_xy).

        With :math:`c = 2\\mu H` and :math:`h = 2 H \\mu_Q`, each stress
        :math:`\\tau_{ab} = c\\,a_{ab}` varies as
        :math:`d\\tau_{ab} = \\mathrm{diag}(c)\\,da_{ab}
        + \\mathrm{diag}(h\\,a_{ab}) [G_u, G_v]`, where
        :math:`G_u = \\mathrm{diag}(a_{xx}) D_x + \\mathrm{diag}(a_{xy}) D_y`
        and :math:`G_v = \\mathrm{diag}(a_{xy}) D_x + \\mathrm{diag}(a_{yy})
        D_y` map a velocity change to :math:`dQ` (since
        :math:`\\partial Q/\\partial u_x = a_{xx}` etc.).

        Returns
        -------
        Tuple[Array, Array, Array]
            Each shape: (npts, 2*npts), columns [u, v].
        """
        bkd = self._bkd
        Dx, Dy = self._Dx, self._Dy
        ux, uy, vx, vy = self._gradients(velocity)
        a_xx = 2.0 * ux + vy
        a_yy = ux + 2.0 * vy
        a_xy = 0.5 * (uy + vx)
        mu, mu_q = self._viscosity_derivatives(ux, uy, vx, vy)
        coef = (2.0 * mu * depth)[:, None]
        h2 = 2.0 * depth * mu_q
        # dQ = G_u du + G_v dv; products with diagonals by row scaling
        g = bkd.concatenate(
            [
                a_xx[:, None] * Dx + a_xy[:, None] * Dy,
                a_xy[:, None] * Dx + a_yy[:, None] * Dy,
            ],
            axis=1,
        )

        def variation(a: Array, da_u: Array, da_v: Array) -> Array:
            frozen = bkd.concatenate([coef * da_u, coef * da_v], axis=1)
            return frozen + (h2 * a)[:, None] * g

        return (
            variation(a_xx, 2.0 * Dx, Dy),
            variation(a_yy, Dx, 2.0 * Dy),
            variation(a_xy, 0.5 * Dy, 0.5 * Dx),
        )

    def residual(
        self,
        velocity: Array,
        depth: Array,
        bed: Array,
        friction: Array,
        forcing: Optional[Array],
    ) -> Array:
        """div(2 mu H strain) - C vel - H rho g grad(H + b) + f.

        Shape: (2*npts,)
        """
        npts = depth.shape[0]
        u, v = velocity[:npts], velocity[npts:]
        tau_xx, tau_yy, tau_xy = self.stress(velocity, depth)
        div_tau_x = self._Dx @ tau_xx + self._Dy @ tau_xy
        div_tau_y = self._Dx @ tau_xy + self._Dy @ tau_yy

        # Driving stress: H * rho * g * grad(s)
        surface = depth + bed
        drive_coef = depth * self._rho * self._g
        res_u = div_tau_x - friction * u - drive_coef * (self._Dx @ surface)
        res_v = div_tau_y - friction * v - drive_coef * (self._Dy @ surface)
        if forcing is not None:
            res_u = res_u + forcing[:npts]
            res_v = res_v + forcing[npts:]
        return self._bkd.hstack([res_u, res_v])

    def jacobian(self, velocity: Array, depth: Array, friction: Array) -> Array:
        """Compute the exact velocity Jacobian of the momentum residual.

        The membrane part is the divergence of the stress Jacobians of
        :meth:`stress_jacobian`: rows u are :math:`D_x J_{xx} + D_y
        J_{xy}`, rows v :math:`D_x J_{xy} + D_y J_{yy}`. Driving stress
        and forcing are velocity independent; friction contributes
        diagonals. The result is nonsymmetric.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (2*npts, 2*npts)
        """
        bkd = self._bkd
        Dx, Dy = self._Dx, self._Dy
        j_xx, j_yy, j_xy = self.stress_jacobian(velocity, depth)
        zeros = bkd.zeros((friction.shape[0], friction.shape[0]))
        diag_friction = bkd.diag(friction)
        sliding = bkd.concatenate(
            [
                bkd.concatenate([diag_friction, zeros], axis=1),
                bkd.concatenate([zeros, diag_friction], axis=1),
            ],
            axis=0,
        )
        membrane = bkd.concatenate(
            [Dx @ j_xx + Dy @ j_xy, Dx @ j_xy + Dy @ j_yy], axis=0
        )
        return membrane - sliding

    def depth_jacobian(self, velocity: Array, depth: Array, bed: Array) -> Array:
        """Compute d(R_vel)/dH holding the velocities fixed.

        The stress coefficient :math:`2\\mu H` is linear in H (the
        viscosity depends only on velocity gradients), and the driving
        stress :math:`H \\rho g \\nabla(H+b)` contributes the product
        rule :math:`\\rho g (\\mathrm{diag}(\\nabla s)
        + \\mathrm{diag}(H) D)`.

        Returns
        -------
        Array
            Assembly. Shape: (2*npts, npts)
        """
        bkd = self._bkd
        Dx, Dy = self._Dx, self._Dy
        ux, uy, vx, vy = self._gradients(velocity)
        a_xx = 2.0 * ux + vy
        a_yy = ux + 2.0 * vy
        a_xy = 0.5 * (uy + vx)
        mu, _ = self._viscosity_derivatives(ux, uy, vx, vy)
        two_mu = 2.0 * mu
        rg = self._rho * self._g
        surface = depth + bed
        diag_depth = bkd.diag(depth)

        top = (
            Dx @ bkd.diag(two_mu * a_xx)
            + Dy @ bkd.diag(two_mu * a_xy)
            - rg * (bkd.diag(Dx @ surface) + diag_depth @ Dx)
        )
        bottom = (
            Dx @ bkd.diag(two_mu * a_xy)
            + Dy @ bkd.diag(two_mu * a_yy)
            - rg * (bkd.diag(Dy @ surface) + diag_depth @ Dy)
        )
        return bkd.concatenate([top, bottom], axis=0)


class _ShelfMass(Generic[Array]):
    """The mass-conservation residual -div(H vel) + f and its
    derivatives, every field given.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    D1_matrices : list of Array
        First-derivative matrices, one per dimension. Each shape:
        (npts, npts)
    """

    def __init__(self, bkd: Backend[Array], D1_matrices: List[Array]) -> None:
        self._bkd = bkd
        self._D1_matrices = D1_matrices

    def _component(self, velocity: Array, dim: int, npts: int) -> Array:
        return velocity[dim * npts : (dim + 1) * npts]

    def residual(
        self, depth: Array, velocity: Array, forcing: Optional[Array]
    ) -> Array:
        """-div(H vel) + f. Shape: (npts,)"""
        npts = depth.shape[0]
        residual = self._bkd.zeros((npts,))
        for dim, D in enumerate(self._D1_matrices):
            residual = residual - D @ (depth * self._component(velocity, dim, npts))
        if forcing is not None:
            residual = residual + forcing
        return residual

    def jacobian(self, velocity: Array) -> Array:
        """d(residual)/dH = -sum_d D_d diag(vel_d). Shape: (npts, npts)"""
        npts = velocity.shape[0] // len(self._D1_matrices)
        jacobian = self._bkd.zeros((npts, npts))
        for dim, D in enumerate(self._D1_matrices):
            jacobian = jacobian - D @ self._bkd.diag(
                self._component(velocity, dim, npts)
            )
        return jacobian

    def velocity_jacobian(self, depth: Array) -> Array:
        """d(residual)/d[vel]; block d is -D_d diag(H).

        Shape: (npts, ndim*npts)
        """
        blocks = [-(D @ self._bkd.diag(depth)) for D in self._D1_matrices]
        return self._bkd.concatenate(blocks, axis=1)


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
    depth : float, NodalCoefficientProtocol or callable
        Ice thickness H, in the forms ``as_nodal_coefficient`` takes
        (nodal values arrive as ``NodalField``).
    bed : float, NodalCoefficientProtocol or callable
        Bed elevation b, in the same forms.
    friction : float, NodalCoefficientProtocol or callable
        Friction coefficient C, in the same forms.
    A : float
        Rate factor in Glen's flow law.
    rho : float
        Ice density (kg/m^3).
    g : float
        Gravitational acceleration (default: 9.81).
    forcing : float, NodalCoefficientProtocol or callable, optional
        Velocity forcing, stacked [f_u, f_v], in the same forms. Shape of
        its values: (2*npts,). Absent means none.
    eps : float
        Regularization for strain rate (default: 1e-12).
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        depth: CoefficientInput[Array],
        bed: CoefficientInput[Array],
        friction: CoefficientInput[Array],
        A: float,
        rho: float,
        g: float = 9.81,
        forcing: Optional[CoefficientInput[Array]] = None,
        eps: float = 1e-12,
    ):
        if basis.ndim() != 2:
            raise ValueError("ShallowShelfVelocityPhysics requires a 2D basis")

        super().__init__(basis, bkd, ncomponents=2)

        npts = basis.npts()
        self._depth = as_nodal_coefficient(depth, npts, bkd)
        self._bed = as_nodal_coefficient(bed, npts, bkd)
        self._friction = as_nodal_coefficient(friction, npts, bkd)
        self._forcing = optional_nodal_coefficient(forcing, 2 * npts, bkd)
        self._momentum = _ShelfMomentum(
            bkd,
            basis.derivative_matrix(1, 0),
            basis.derivative_matrix(1, 1),
            A,
            rho,
            g,
            eps,
        )

    def depth_function(self) -> NodalCoefficientProtocol[Array]:
        """The ice thickness."""
        return self._depth

    def bed_function(self) -> NodalCoefficientProtocol[Array]:
        """The bed elevation."""
        return self._bed

    def friction_function(self) -> NodalCoefficientProtocol[Array]:
        """The friction coefficient."""
        return self._friction

    def forcing_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The stacked forcing; None without a forcing term."""
        return self._forcing

    def _forcing_values(self, time: float) -> Optional[Array]:
        return None if self._forcing is None else self._forcing.values(time)

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual.

        residual = div(2*mu*H*strain_tensor) - C*vel - H*rho*g*grad(s) + f,
        with H, b, C and f read at ``time``. Shape: (2*npts,)
        """
        return self._momentum.residual(
            state,
            self._depth.values(time),
            self._bed.values(time),
            self._friction.values(time),
            self._forcing_values(time),
        )

    def jacobian(self, state: Array, time: float) -> Array:
        """Exact state Jacobian of the momentum residual (see
        ``_ShelfMomentum.jacobian``). Shape: (2*npts, 2*npts)"""
        return self._momentum.jacobian(
            state, self._depth.values(time), self._friction.values(time)
        )

    def membrane_stress(self, state: Array, time: float) -> Tuple[Array, Array, Array]:
        """The membrane stresses (tau_xx, tau_yy, tau_xy) of the velocity
        ``state`` [u, v], at the depth read at ``time``. The membrane part
        of the residual is div(tau) row by row. Each shape: (npts,)"""
        return self._momentum.stress(state, self._depth.values(time))

    def membrane_stress_jacobian(
        self, state: Array, time: float
    ) -> Tuple[Array, Array, Array]:
        """The velocity Jacobians of :meth:`membrane_stress`. Each shape:
        (npts, 2*npts)"""
        return self._momentum.stress_jacobian(state, self._depth.values(time))

    def residual_depth_jacobian(self, state: Array, time: float) -> Array:
        """Compute d(R_vel)/dH holding the velocities fixed, at the depth
        and bed read at ``time``.

        Parameters
        ----------
        state : Array
            Velocity state [u, v]. Shape: (2*npts,)
        time : float
            Time at which the depth and bed are read.

        Returns
        -------
        Array
            Assembly. Shape: (2*npts, npts)
        """
        return self._momentum.depth_jacobian(
            state, self._depth.values(time), self._bed.values(time)
        )

    # -- the interior operator (CollocationInteriorOperatorProtocol)

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`f(u, v, t)`, no boundary condition applied.
        Shape: (2*npts,)"""
        return self.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial f/\partial (u, v)`. Shape: (2*npts, 2*npts)"""
        return self.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """No second derivatives are supplied."""
        return StateDerivatives.none()

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether the depth and friction are declared time-invariant."""
        return declared_time_invariant(self._depth, self._friction)

    def interior_is_time_invariant(self) -> bool:
        """Whether, besides the depth and friction, the bed and forcing are
        declared time-invariant."""
        return declared_time_invariant(
            self._depth, self._bed, self._friction, self._forcing
        )


class ShallowShelfStressFlux(Generic[Array]):
    r"""The flux of one velocity component of a shallow shelf physics, as
    a boundary condition prescribes it: minus that row of the membrane
    stress, :math:`q_u = -(\tau_{xx}, \tau_{xy})` or :math:`q_v =
    -(\tau_{xy}, \tau_{yy})`, so the membrane part of the residual is
    :math:`-\nabla\cdot q` and :math:`q\cdot n` is minus the membrane
    traction.

    Reads the physics' current depth at each call. Satisfies
    ``FluxProviderProtocol``.

    Parameters
    ----------
    physics : ShallowShelfVelocityPhysics
        Supplies the stress and its Jacobian.
    component : int
        0 for u, 1 for v.

    Raises
    ------
    ValueError
        If ``component`` is not 0 or 1.
    """

    def __init__(
        self, physics: ShallowShelfVelocityPhysics[Array], component: int
    ) -> None:
        if component not in (0, 1):
            raise ValueError(f"component must be 0 or 1, got {component}")
        self._physics, self._component = physics, component

    def compute_flux(self, state: Array, time: float) -> List[Array]:
        """The flux at all mesh points, (q_x, q_y). Each shape: (npts,)"""
        tau_xx, tau_yy, tau_xy = self._physics.membrane_stress(state, time)
        if self._component == 0:
            return [-tau_xx, -tau_xy]
        return [-tau_xy, -tau_yy]

    def compute_flux_jacobian(self, state: Array, time: float) -> List[Array]:
        """The velocity Jacobian of each flux direction. Each shape:
        (npts, 2*npts)"""
        j_xx, j_yy, j_xy = self._physics.membrane_stress_jacobian(state, time)
        if self._component == 0:
            return [-j_xx, -j_xy]
        return [-j_xy, -j_yy]

    def flux_is_time_dependent(self) -> bool:
        """Whether the depth is declared time-dependent."""
        return not declared_time_invariant(self._physics.depth_function())

    def flux_is_linear_in_state(self) -> bool:
        """False: the effective viscosity reads the strain rate."""
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(component={self._component})"


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
    velocities : float, NodalCoefficientProtocol or callable
        Advecting velocity, stacked [u, v], in the forms
        ``as_nodal_coefficient`` takes. Shape of its values: (2*npts,)
    forcing : float, NodalCoefficientProtocol or callable, optional
        Source/sink term (e.g., surface mass balance), in the same forms.
        Absent means none.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        velocities: CoefficientInput[Array],
        forcing: Optional[CoefficientInput[Array]] = None,
    ):
        if basis.ndim() != 2:
            raise ValueError("ShallowShelfDepthPhysics requires a 2D basis")

        super().__init__(basis, bkd)
        npts = basis.npts()
        self._velocities = as_nodal_coefficient(velocities, 2 * npts, bkd)
        self._forcing = optional_nodal_coefficient(forcing, npts, bkd)
        self._mass = _ShelfMass(
            bkd, [basis.derivative_matrix(1, dim) for dim in range(2)]
        )

    def ncomponents(self) -> int:
        return 1

    def velocity_function(self) -> NodalCoefficientProtocol[Array]:
        """The advecting velocity."""
        return self._velocities

    def forcing_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The forcing; None without a forcing term."""
        return self._forcing

    def residual(self, state: Array, time: float) -> Array:
        """Compute residual -div(H*velocity) + f, the velocity and forcing
        read at ``time``. Shape: (npts,)"""
        forcing = None if self._forcing is None else self._forcing.values(time)
        return self._mass.residual(state, self._velocities.values(time), forcing)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute Jacobian d(residual)/dH. Shape: (npts, npts)"""
        return self._mass.jacobian(self._velocities.values(time))

    def residual_velocity_jacobian(self, state: Array) -> Array:
        """Compute d(R_H)/d[vel] holding the depth fixed.

        With :math:`R_H = -\\sum_d D_d (H\\, \\mathrm{vel}_d) + f`,
        block d is :math:`-D_d\\,\\mathrm{diag}(H)`.

        Parameters
        ----------
        state : Array
            Ice thickness H. Shape: (npts,)

        Returns
        -------
        Array
            Assembly. Shape: (npts, 2*npts)
        """
        return self._mass.velocity_jacobian(state)

    # -- the interior operator (CollocationInteriorOperatorProtocol)

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`f(H, t)`, no boundary condition applied. Shape: (npts,)"""
        return self.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial f/\partial H`. Shape: (npts, npts)"""
        return self.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """The residual is linear in H."""
        return StateDerivatives.linear(self._bkd)

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether the velocity is declared time-invariant."""
        return declared_time_invariant(self._velocities)

    def interior_is_time_invariant(self) -> bool:
        """Whether, besides the velocity, the forcing is declared
        time-invariant."""
        return declared_time_invariant(self._velocities, self._forcing)


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
    bed : float, NodalCoefficientProtocol or callable
        Bed elevation b, in the forms ``as_nodal_coefficient`` takes.
    friction : float, NodalCoefficientProtocol or callable
        Friction coefficient C, in the same forms.
    A : float
        Rate factor in Glen's flow law.
    rho : float
        Ice density (kg/m^3).
    g : float
        Gravitational acceleration (default: 9.81).
    depth_forcing : float, NodalCoefficientProtocol or callable, optional
        Forcing for depth equation, in the same forms. Shape of its
        values: (npts,)
    velocity_forcing : float, NodalCoefficientProtocol or callable, optional
        Forcing for velocity equations, in the same forms. Shape of its
        values: (2*npts,)
    eps : float
        Regularization for strain rate (default: 1e-12).
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        bed: CoefficientInput[Array],
        friction: CoefficientInput[Array],
        A: float,
        rho: float,
        g: float = 9.81,
        depth_forcing: Optional[CoefficientInput[Array]] = None,
        velocity_forcing: Optional[CoefficientInput[Array]] = None,
        eps: float = 1e-12,
    ):
        if basis.ndim() != 2:
            raise ValueError("ShallowShelfDepthVelocityPhysics requires a 2D basis")

        super().__init__(basis, bkd, ncomponents=3)

        npts = basis.npts()
        self._bed = as_nodal_coefficient(bed, npts, bkd)
        self._friction = as_nodal_coefficient(friction, npts, bkd)
        self._depth_forcing = optional_nodal_coefficient(depth_forcing, npts, bkd)
        self._velocity_forcing = optional_nodal_coefficient(
            velocity_forcing, 2 * npts, bkd
        )
        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        self._momentum = _ShelfMomentum(bkd, Dx, Dy, A, rho, g, eps)
        self._mass = _ShelfMass(bkd, [Dx, Dy])

        # Cache mass matrix
        self._cached_mass_matrix = bkd.zeros((3 * npts, 3 * npts))
        for i in range(npts):
            self._cached_mass_matrix[i, i] = 1.0

    def bed_function(self) -> NodalCoefficientProtocol[Array]:
        """The bed elevation."""
        return self._bed

    def friction_function(self) -> NodalCoefficientProtocol[Array]:
        """The friction coefficient."""
        return self._friction

    def depth_forcing_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The depth forcing; None without one."""
        return self._depth_forcing

    def velocity_forcing_function(
        self,
    ) -> Optional[NodalCoefficientProtocol[Array]]:
        """The stacked velocity forcing; None without one."""
        return self._velocity_forcing

    def _split_state(self, state: Array) -> Tuple[Array, Array]:
        """Split state [H, u, v] into the depth and the stacked velocity."""
        npts = self.npts()
        return state[:npts], state[npts:]

    def residual(self, state: Array, time: float) -> Array:
        """Compute coupled residual [R_H, R_u, R_v].

        R_H = -div(H*vel) + f_H  (depth: transient)
        R_u, R_v = SSA momentum   (velocity: steady/algebraic)
        """
        H, velocity = self._split_state(state)
        depth_forcing = (
            None if self._depth_forcing is None else self._depth_forcing.values(time)
        )
        velocity_forcing = (
            None
            if self._velocity_forcing is None
            else self._velocity_forcing.values(time)
        )
        depth_res = self._mass.residual(H, velocity, depth_forcing)
        vel_res = self._momentum.residual(
            velocity,
            H,
            self._bed.values(time),
            self._friction.values(time),
            velocity_forcing,
        )
        return self._bkd.hstack([depth_res, vel_res])

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute the exact coupled Jacobian.

        The block structure (depth couples to velocity through
        div(H*vel), velocity to depth through the surface gradient and
        the 2*mu*H stress coefficient) is

        .. math::

            \\begin{bmatrix}
            \\partial R_H/\\partial H &
            \\partial R_H/\\partial \\mathrm{vel} \\\\
            \\partial R_{\\mathrm{vel}}/\\partial H &
            \\partial R_{\\mathrm{vel}}/\\partial \\mathrm{vel}
            \\end{bmatrix}

        with every block assembled analytically from the state's own
        depth and velocity.
        """
        bkd = self._bkd
        H, velocity = self._split_state(state)
        top = bkd.concatenate(
            [self._mass.jacobian(velocity), self._mass.velocity_jacobian(H)],
            axis=1,
        )
        bottom = bkd.concatenate(
            [
                self._momentum.depth_jacobian(velocity, H, self._bed.values(time)),
                self._momentum.jacobian(velocity, H, self._friction.values(time)),
            ],
            axis=1,
        )
        return bkd.concatenate([top, bottom], axis=0)

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

    def interior_mass_weights(self) -> Array:
        """One on the depth rows, zero on the velocity rows: only the depth
        evolves; the velocity equations are algebraic. Shape: (nstates,)"""
        bkd, npts = self._bkd, self.npts()
        return bkd.concatenate([bkd.ones((npts,)), bkd.zeros((2 * npts,))])

    # -- the interior operator (CollocationInteriorOperatorProtocol)

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`f(H, u, v, t)`, no boundary condition applied.
        Shape: (3*npts,)"""
        return self.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial f/\partial (H, u, v)`. Shape: (3*npts, 3*npts)"""
        return self.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """No second derivatives are supplied."""
        return StateDerivatives.none()

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether the bed and friction are declared time-invariant."""
        return declared_time_invariant(self._bed, self._friction)

    def interior_is_time_invariant(self) -> bool:
        """Whether, besides the bed and friction, both forcings are
        declared time-invariant."""
        return declared_time_invariant(
            self._bed, self._friction, self._depth_forcing, self._velocity_forcing
        )


def create_shallow_shelf_velocity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    depth: CoefficientInput[Array],
    bed: CoefficientInput[Array],
    friction: CoefficientInput[Array],
    A: float = 1e-16,
    rho: float = 917.0,
    g: float = 9.81,
    forcing: Optional[CoefficientInput[Array]] = None,
    eps: float = 1e-12,
) -> ShallowShelfVelocityPhysics[Array]:
    """Create shallow shelf velocity physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis.
    bkd : Backend
        Computational backend.
    depth : float, NodalCoefficientProtocol or callable
        Ice thickness.
    bed : float, NodalCoefficientProtocol or callable
        Bed elevation.
    friction : float, NodalCoefficientProtocol or callable
        Friction coefficient.
    A : float
        Rate factor (default: 1e-16).
    rho : float
        Ice density (default: 917.0).
    g : float
        Gravitational acceleration (default: 9.81).
    forcing : float, NodalCoefficientProtocol or callable, optional
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
    velocities: CoefficientInput[Array],
    forcing: Optional[CoefficientInput[Array]] = None,
) -> ShallowShelfDepthPhysics[Array]:
    """Create shallow shelf depth physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    velocities : float, NodalCoefficientProtocol or callable
        Advecting velocity, stacked [u, v].
    forcing : float, NodalCoefficientProtocol or callable, optional
        Source/sink term.

    Returns
    -------
    ShallowShelfDepthPhysics
        Depth evolution physics.
    """
    return ShallowShelfDepthPhysics(
        basis=basis,
        bkd=bkd,
        velocities=velocities,
        forcing=forcing,
    )


def create_shallow_shelf_depth_velocity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    bed: CoefficientInput[Array],
    friction: CoefficientInput[Array],
    A: float = 1e-16,
    rho: float = 917.0,
    g: float = 9.81,
    depth_forcing: Optional[CoefficientInput[Array]] = None,
    velocity_forcing: Optional[CoefficientInput[Array]] = None,
    eps: float = 1e-12,
) -> ShallowShelfDepthVelocityPhysics[Array]:
    """Create coupled shallow shelf depth-velocity physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D collocation basis.
    bkd : Backend
        Computational backend.
    bed : float, NodalCoefficientProtocol or callable
        Bed elevation.
    friction : float, NodalCoefficientProtocol or callable
        Friction coefficient.
    A : float
        Rate factor (default: 1e-16).
    rho : float
        Ice density (default: 917.0).
    g : float
        Gravitational acceleration (default: 9.81).
    depth_forcing : float, NodalCoefficientProtocol or callable, optional
        Forcing for depth equation.
    velocity_forcing : float, NodalCoefficientProtocol or callable, optional
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
