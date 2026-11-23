"""Second-order reduction fixtures shared by its storage tests: a
parameterized nonlinear spring, dampers, and the transient map of an
endpoint displacement."""

import math
from typing import Generic, Optional, TypeVar

import numpy as np

from pyapprox.implicitfunction.implicit_map import ArrayParameters, ImplicitMapFunction
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import RowPinningProtocol
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    ODEResidualDerivatives,
    backend_spaces,
    constant_jacobian,
    dense_jacobian,
    dense_operator,
    state_free_jacobian,
)
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.second_order.assembled_blocks import backend_dense_state_blocks
from pyapprox.ode.second_order.reduction import (
    InitialVelocityDerivatives,
    StateBlocksProtocol,
    VelocityForceDerivatives,
    second_order_system,
)
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import crank_nicolson
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
    TransientPoint,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    DenseFormat,
)
from pyapprox.util.linalg.operators import FormatProtocol, OperatorProperties
from pyapprox.util.rootfinding.nonlinear import Newton

N = 3
Dense = MassFormSystem[Array, Array, DenseFormat[Array]]
# The format of the storage's assembled operators.
G = TypeVar("G", bound=FormatProtocol)


def tridiagonal(bkd: Backend[Array], diagonal: float, off: float) -> Array:
    return bkd.asarray(
        np.diag(np.full(N, diagonal))
        + np.diag(np.full(N - 1, off), 1)
        + np.diag(np.full(N - 1, off), -1)
    )


class Spring(Generic[Array]):
    """The restoring force of the module docstring, with its bundle."""

    def __init__(self, bkd: Backend[Array], cubic: float = 0.4) -> None:
        self._bkd, self._cubic = bkd, cubic
        self._k0 = tridiagonal(bkd, 2.0, -1.0)
        self._k1 = tridiagonal(bkd, 1.0, 0.3)
        self._b = bkd.asarray([1.0, -0.5, 0.25])
        self._e = bkd.asarray([0.3, 0.1, -0.2])
        self._base = bkd.asarray([0.5, -0.2, 0.1])
        self._param = bkd.asarray([0.2, 0.7])
        self._version = 0
        self._space, self._params = backend_spaces(bkd, N, 2)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_param(self, param: Array) -> None:
        self._param = param
        self._version += 1

    def parameters_version(self) -> int:
        return self._version

    def stiffness(self) -> Array:
        return self._k0 + self._param[0] * self._k1

    def initial_displacement(self, param: Array) -> Array:
        return self._base + param[0] * self._e

    def __call__(self, state: Array, time: float) -> Array:
        bkd = self._bkd
        return (
            -bkd.dot(self.stiffness(), state)
            - self._cubic * state**3
            + self._param[1] * math.sin(time) * self._b
        )

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        bkd, space, params = self._bkd, self._space, self._params
        # Without the cubic term f is affine in the state, and declares so.
        affine = self._cubic == 0.0
        return ODEResidualDerivatives.second_order(
            space,
            params,
            state_free_jacobian(self._linear_jacobian)
            if affine
            else dense_jacobian(self._jacobian, space, bkd),
            dense_operator(self._param_matrix, params, space, bkd),
            dense_operator(self._initial, params, space, bkd),
            STATED_ZERO,
            STATED_ZERO
            if affine
            else dense_operator(self._state_state, space, space, bkd, symmetric=True),
            dense_operator(self._state_param, params, space, bkd),
            STATED_ZERO,
        )

    def _linear_jacobian(self, time: float) -> BackendDenseOperator[Array]:
        return BackendDenseOperator(
            -self.stiffness(), self._space.descriptor(), self._bkd
        )

    def _jacobian(self, state: Array, time: float) -> Array:
        return -self.stiffness() - 3.0 * self._cubic * self._bkd.diag(state**2)

    def _param_matrix(self, state: Array, time: float) -> Array:
        bkd = self._bkd
        return bkd.stack([-bkd.dot(self._k1, state), math.sin(time) * self._b], axis=1)

    def _initial(self) -> Array:
        return self._bkd.stack([self._e, self._bkd.zeros((N,))], axis=1)

    def _state_state(self, state: Array, time: float, adj: Array) -> Array:
        return -6.0 * self._cubic * self._bkd.diag(adj * state)

    def _state_param(self, state: Array, time: float, adj: Array) -> Array:
        bkd = self._bkd
        return bkd.stack([-bkd.dot(self._k1, adj), bkd.zeros((N,))], axis=1)


def inertia(bkd: Backend[Array]) -> BackendDenseOperator[Array]:
    space, _ = backend_spaces(bkd, N, 2)
    return BackendDenseOperator(
        bkd.diag(bkd.asarray([1.0, 2.0, 1.5])),
        space.descriptor(),
        bkd,
        properties=OperatorProperties(symmetric=True, spd=True, diagonal=True),
    )


def restoring_system(bkd: Backend[Array], spring: Spring[Array]) -> Dense[Array]:
    space, _ = backend_spaces(bkd, N, 2)
    return MassFormSystem(
        rhs=spring,
        mass=inertia(bkd),
        space=space,
        solver=BackendDenseLU(bkd),
        form=BackendColumnForm(bkd),
    )


class LinearDamper(Generic[Array]):
    r""":math:`g = -C v`, :math:`C` symmetric positive definite."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._c = tridiagonal(bkd, 0.5, 0.1)
        space, _ = backend_spaces(bkd, N, 2)
        descriptor = space.descriptor()
        zero = BackendDenseOperator(bkd.zeros((N, N)), descriptor, bkd)
        self._derivatives = VelocityForceDerivatives.parameter_free(
            constant_jacobian(zero),
            constant_jacobian(BackendDenseOperator(-self._c, descriptor, bkd)),
            affine=True,
        )

    def __call__(self, displacement: Array, velocity: Array, time: float) -> Array:
        return -self._bkd.dot(self._c, velocity)

    def derivatives(self) -> VelocityForceDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives


class CubicDamper(Generic[Array]):
    r""":math:`g = -\gamma u^2 v`."""

    def __init__(self, bkd: Backend[Array], declared: bool) -> None:
        self._bkd, self._gamma = bkd, 0.6
        space, _ = backend_spaces(bkd, N, 2)
        wrt_u = dense_jacobian(self._wrt_displacement, space, bkd)
        wrt_v = dense_jacobian(self._wrt_velocity, space, bkd)
        self._derivatives = (
            VelocityForceDerivatives.parameter_free(wrt_u, wrt_v, affine=False)
            if declared
            else VelocityForceDerivatives.state_only(wrt_u, wrt_v)
        )

    def __call__(self, displacement: Array, velocity: Array, time: float) -> Array:
        return -self._gamma * displacement**2 * velocity

    # Called with the whole state w = [u; v].

    def _wrt_displacement(self, state: Array, time: float) -> Array:
        u, v = state[:N], state[N:]
        return self._bkd.diag(-2.0 * self._gamma * u * v)

    def _wrt_velocity(self, state: Array, time: float) -> Array:
        return self._bkd.diag(-self._gamma * state[:N] ** 2)

    def derivatives(self) -> VelocityForceDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives


class Checked(Generic[Array]):
    """The reduced right-hand side with the backend and parameter setter
    the bundle checker reads."""

    def __init__(self, system: Dense[Array], spring: Spring[Array]) -> None:
        self._system, self._spring = system, spring

    def bkd(self) -> Backend[Array]:
        return self._spring.bkd()

    def set_param(self, param: Array) -> None:
        self._spring.set_param(param)

    def parameters_version(self) -> int:
        return self._spring.parameters_version()

    def __call__(self, state: Array, time: float) -> Array:
        return self._system.rhs(state, time)

    def derivatives(self) -> ODEResidualDerivatives[Array, FormatProtocol]:
        return self._system.rhs.derivatives()


def velocity_derivative(bkd: Backend[Array]) -> InitialVelocityDerivatives[Array]:
    space, params = backend_spaces(bkd, N, 2)

    def matrix() -> Array:
        return bkd.asarray([[0.0, 0.4], [0.2, 0.0], [-0.1, 0.3]])

    return InitialVelocityDerivatives(
        dense_operator(matrix, params, space, bkd), STATED_ZERO
    )


def reduced_system_on(
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
    bkd: Backend[Array],
    spring: Spring[Array],
    damper: Optional[str] = None,
    initial: Optional[InitialVelocityDerivatives[Array]] = None,
) -> MassFormSystem[Array, Array, G]:
    """The spring's reduced system stored in ``blocks``, with the named
    damper."""
    force = {
        None: None,
        "linear": LinearDamper(bkd),
        "cubic": CubicDamper(bkd, declared=True),
        "undeclared": CubicDamper(bkd, declared=False),
    }[damper]
    return second_order_system(
        restoring_system(bkd, spring),
        blocks,
        InitialVelocityDerivatives.independent() if initial is None else initial,
        force,
    )


def reduced_system(
    bkd: Backend[Array],
    spring: Spring[Array],
    damper: Optional[str] = None,
    initial: Optional[InitialVelocityDerivatives[Array]] = None,
) -> Dense[Array]:
    """The spring's reduced system in the assembled storage."""
    return reduced_system_on(
        backend_dense_state_blocks(bkd, N), bkd, spring, damper, initial
    )


def sample_state(bkd: Backend[Array]) -> Array:
    return bkd.asarray([0.4, -0.3, 0.2, 0.1, 0.5, -0.6])


class EndpointDiscretization(Generic[Array, G]):
    """The reduced system for the transient map, on any storage: no
    pinned rows."""

    def __init__(
        self, system: MassFormSystem[Array, Array, G], spring: Spring[Array]
    ) -> None:
        self._system, self._spring = system, spring

    def system(self) -> MassFormSystem[Array, Array, G]:
        return self._system

    def pinning(self) -> Optional[RowPinningProtocol[Array, G]]:
        return None

    def initial_state(self, state: Array, time: float) -> Array:
        return state

    def nparams(self) -> int:
        return 2

    def set_param(self, param: Array) -> None:
        self._spring.set_param(param[:, 0])


def endpoint_function(
    bkd: Backend[Array],
    damper: Optional[str],
    param: Array,
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    """:math:`u_1(T)`, its initial state :math:`[u_0(\\text{param}); v_0]`,
    on the storage ``blocks``."""
    spring = Spring(bkd)
    system = reduced_system_on(blocks, bkd, spring, damper)
    discretization = EndpointDiscretization(system, spring)
    w0 = bkd.concatenate(
        [spring.initial_displacement(param[:, 0]), bkd.asarray([0.1, 0.0, -0.2])]
    )
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(EndpointTerm(StateEntries(bkd, 2 * N, [1])),)
        )
    )
    implicit = TransientImplicitMap(
        discretization,
        discretization,
        RungeKutta(crank_nicolson(), newton),
        RecomputeAlways(system.solver),
        FixedStep(0.05, (0.0, 0.4)),
        bkd.reshape(w0, (-1, 1)),
        functional,
    )
    return ImplicitMapFunction(implicit, ArrayParameters(bkd))
