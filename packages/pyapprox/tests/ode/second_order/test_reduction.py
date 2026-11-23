r"""The first-order reduction of a plain matrix mass-spring system.

Three masses, :math:`M = \mathrm{diag}(1, 2, 1.5)`, a restoring force

.. math::

    f(u, t; p) = -(K_0 + p_0 K_1) u - c\, u^3 + p_1 \sin(t)\, b

with :math:`K_0, K_1` symmetric tridiagonal and :math:`u_0(p) = u_b + p_0
e`, so :math:`f` has every derivative and :math:`p` reaches the state, the
Jacobian and the initial state. Velocity forces: a linear damper
:math:`g = -C v` (affine, parameter-free), a nonlinear one
:math:`g = -\gamma u^2 v` (parameter-free, not affine) and the same with
its parameter dependence left undeclared. Storage: one dense :math:`2n`
array.

- the block mass is :math:`\mathrm{blockdiag}(I, M)` and the storage's
  solver solves with it; :math:`R(w)` is :math:`[v; f + g]`;
- the lifted bundle passes the shared bundle checker (finite differences
  and dot tests of every field) undamped, with each damper, and with
  :math:`\partial v_0/\partial p \ne 0`;
- the bundle has the weakest tier of its parts;
- the gradient and HVP of :math:`u_1(T)` through ``TransientImplicitMap``
  pass central differences (CN, 8 steps of 0.05);
- undamped and linear, the implicit midpoint rule conserves the energy
  :math:`\tfrac12 v^\top M v + \tfrac12 u^\top K u` to 1e-12 over 200
  steps;
- the reduction refuses a storage whose block space differs from the
  restoring system's.
"""

import math
from typing import Generic, Optional

import numpy as np
import pytest

from pyapprox.implicitfunction.implicit_map import ArrayParameters, ImplicitMapFunction
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.ode.integration import Integrator
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
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.second_order.backend_blocks import BackendDenseStateBlocks
from pyapprox.ode.second_order.rayleigh import RayleighDamping
from pyapprox.ode.second_order.reduction import (
    InitialVelocityDerivatives,
    VelocityForceDerivatives,
    second_order_system,
)
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import crank_nicolson, implicit_midpoint
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
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

_N = 3
_Dense = MassFormSystem[Array, Array, DenseFormat[Array]]


def _tridiagonal(bkd: Backend[Array], diagonal: float, off: float) -> Array:
    return bkd.asarray(
        np.diag(np.full(_N, diagonal))
        + np.diag(np.full(_N - 1, off), 1)
        + np.diag(np.full(_N - 1, off), -1)
    )


class _Spring(Generic[Array]):
    """The restoring force of the module docstring, with its bundle."""

    def __init__(self, bkd: Backend[Array], cubic: float = 0.4) -> None:
        self._bkd, self._cubic = bkd, cubic
        self._k0 = _tridiagonal(bkd, 2.0, -1.0)
        self._k1 = _tridiagonal(bkd, 1.0, 0.3)
        self._b = bkd.asarray([1.0, -0.5, 0.25])
        self._e = bkd.asarray([0.3, 0.1, -0.2])
        self._base = bkd.asarray([0.5, -0.2, 0.1])
        self._param = bkd.asarray([0.2, 0.7])
        self._version = 0
        self._space, self._params = backend_spaces(bkd, _N, 2)

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
        return self._bkd.stack([self._e, self._bkd.zeros((_N,))], axis=1)

    def _state_state(self, state: Array, time: float, adj: Array) -> Array:
        return -6.0 * self._cubic * self._bkd.diag(adj * state)

    def _state_param(self, state: Array, time: float, adj: Array) -> Array:
        bkd = self._bkd
        return bkd.stack([-bkd.dot(self._k1, adj), bkd.zeros((_N,))], axis=1)


def _mass(bkd: Backend[Array]) -> BackendDenseOperator[Array]:
    space, _ = backend_spaces(bkd, _N, 2)
    return BackendDenseOperator(
        bkd.diag(bkd.asarray([1.0, 2.0, 1.5])),
        space.descriptor(),
        bkd,
        properties=OperatorProperties(symmetric=True, spd=True, diagonal=True),
    )


def _restoring(bkd: Backend[Array], spring: _Spring[Array]) -> _Dense[Array]:
    space, _ = backend_spaces(bkd, _N, 2)
    return MassFormSystem(
        rhs=spring,
        mass=_mass(bkd),
        space=space,
        solver=BackendDenseLU(bkd),
        form=BackendColumnForm(bkd),
    )


class _LinearDamper(Generic[Array]):
    r""":math:`g = -C v`, :math:`C` symmetric positive definite."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._c = _tridiagonal(bkd, 0.5, 0.1)
        space, _ = backend_spaces(bkd, _N, 2)
        descriptor = space.descriptor()
        zero = BackendDenseOperator(bkd.zeros((_N, _N)), descriptor, bkd)
        self._derivatives = VelocityForceDerivatives.parameter_free(
            constant_jacobian(zero),
            constant_jacobian(BackendDenseOperator(-self._c, descriptor, bkd)),
            affine=True,
        )

    def __call__(self, displacement: Array, velocity: Array, time: float) -> Array:
        return -self._bkd.dot(self._c, velocity)

    def derivatives(self) -> VelocityForceDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives


class _CubicDamper(Generic[Array]):
    r""":math:`g = -\gamma u^2 v`."""

    def __init__(self, bkd: Backend[Array], declared: bool) -> None:
        self._bkd, self._gamma = bkd, 0.6
        space, _ = backend_spaces(bkd, _N, 2)
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
        u, v = state[:_N], state[_N:]
        return self._bkd.diag(-2.0 * self._gamma * u * v)

    def _wrt_velocity(self, state: Array, time: float) -> Array:
        return self._bkd.diag(-self._gamma * state[:_N] ** 2)

    def derivatives(self) -> VelocityForceDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives


class _Checked(Generic[Array]):
    """The reduced right-hand side with the backend and parameter setter
    the bundle checker reads."""

    def __init__(self, system: _Dense[Array], spring: _Spring[Array]) -> None:
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


def _velocity_derivative(bkd: Backend[Array]) -> InitialVelocityDerivatives[Array]:
    space, params = backend_spaces(bkd, _N, 2)

    def matrix() -> Array:
        return bkd.asarray([[0.0, 0.4], [0.2, 0.0], [-0.1, 0.3]])

    return InitialVelocityDerivatives(
        dense_operator(matrix, params, space, bkd), STATED_ZERO
    )


def _system(
    bkd: Backend[Array],
    spring: _Spring[Array],
    damper: Optional[str] = None,
    initial: Optional[InitialVelocityDerivatives[Array]] = None,
) -> _Dense[Array]:
    force = {
        None: None,
        "linear": _LinearDamper(bkd),
        "cubic": _CubicDamper(bkd, declared=True),
        "undeclared": _CubicDamper(bkd, declared=False),
    }[damper]
    return second_order_system(
        _restoring(bkd, spring),
        BackendDenseStateBlocks(bkd, _N),
        InitialVelocityDerivatives.independent() if initial is None else initial,
        force,
    )


def _state(bkd: Backend[Array]) -> Array:
    return bkd.asarray([0.4, -0.3, 0.2, 0.1, 0.5, -0.6])


class TestStructure:
    def test_block_mass_and_its_solve(self, bkd: Backend[Array]) -> None:
        system = _system(bkd, _Spring(bkd))
        mass = bkd.diag(bkd.asarray([1.0, 1.0, 1.0, 1.0, 2.0, 1.5]))
        bkd.assert_allclose(system.mass.format().matrix(), mass, rtol=0.0, atol=0.0)
        assert system.mass.properties().spd
        rhs = bkd.reshape(_state(bkd), (-1, 1))
        solution, _ = system.solver.setup(system.mass).solve(rhs)
        bkd.assert_allclose(solution, bkd.solve(mass, rhs), rtol=1e-14)

    @pytest.mark.parametrize("damper", [None, "linear", "cubic"])
    def test_value_is_velocity_and_force(
        self, bkd: Backend[Array], damper: Optional[str]
    ) -> None:
        spring = _Spring(bkd)
        system = _system(bkd, spring, damper)
        state = _state(bkd)
        u, v = state[:_N], state[_N:]
        force = spring(u, 0.3)
        if damper == "linear":
            force = force - bkd.dot(_tridiagonal(bkd, 0.5, 0.1), v)
        if damper == "cubic":
            force = force - 0.6 * u**2 * v
        bkd.assert_allclose(
            system.rhs(state, 0.3), bkd.concatenate([v, force]), rtol=1e-14
        )

    def test_refuses_a_storage_of_another_size(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            second_order_system(
                _restoring(bkd, _Spring(bkd)),
                BackendDenseStateBlocks(bkd, _N + 1),
                InitialVelocityDerivatives.independent(),
            )


class TestTiers:
    @pytest.mark.parametrize(
        "damper, tier",
        [
            (None, "second"),
            ("linear", "second"),
            ("cubic", "first"),
            ("undeclared", "state"),
        ],
    )
    def test_weakest_part_sets_the_tier(
        self, bkd: Backend[Array], damper: Optional[str], tier: str
    ) -> None:
        derivatives = _system(bkd, _Spring(bkd), damper).rhs.derivatives()
        assert derivatives.jacobian is not None
        assert (derivatives.param is not None) == (tier != "state")
        assert (derivatives.state_state is not None) == (tier == "second")

    def test_uncurved_initial_velocity_unknown_drops_to_first(
        self, bkd: Backend[Array]
    ) -> None:
        space, params = backend_spaces(bkd, _N, 2)
        initial = InitialVelocityDerivatives[Array](STATED_ZERO, None)
        derivatives = _system(bkd, _Spring(bkd), None, initial).rhs.derivatives()
        assert derivatives.param is not None
        assert derivatives.state_state is None


@pytest.mark.parametrize(
    "damper, moving",
    [(None, False), (None, True), ("linear", True), ("cubic", False)],
)
def test_lifted_bundle_passes_the_checker(
    bkd: Backend[Array], damper: Optional[str], moving: bool
) -> None:
    spring = _Spring(bkd)
    velocity = _velocity_derivative(bkd) if moving else None
    system = _system(bkd, spring, damper, velocity)
    v0 = bkd.asarray([0.1, 0.0, -0.2])

    def initial_state(param: Array) -> Array:
        if velocity is None:
            return bkd.concatenate([spring.initial_displacement(param), v0])
        matrix = bkd.asarray([[0.0, 0.4], [0.2, 0.0], [-0.1, 0.3]])
        return bkd.concatenate(
            [spring.initial_displacement(param), v0 + bkd.dot(matrix, param)]
        )

    report = ODEResidualBundleChecker(
        _Checked(system, spring), initial_state=initial_state
    ).check(
        _state(bkd),
        0.3,
        adj=bkd.asarray([0.3, -0.1, 0.2, 0.5, 0.4, -0.7]),
        param=bkd.asarray([0.2, 0.7]),
    )
    assert report.failures() == []


class _Discretization(Generic[Array]):
    """The reduced system for the transient map: no pinned rows."""

    def __init__(self, system: _Dense[Array], spring: _Spring[Array]) -> None:
        self._system, self._spring = system, spring

    def system(self) -> _Dense[Array]:
        return self._system

    def pinning(self) -> Optional[RowPinningProtocol[Array, DenseFormat[Array]]]:
        return None

    def initial_state(self, state: Array, time: float) -> Array:
        return state

    def nparams(self) -> int:
        return 2

    def set_param(self, param: Array) -> None:
        self._spring.set_param(param[:, 0])


def _endpoint_function(
    bkd: Backend[Array], damper: Optional[str], param: Array
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    """:math:`u_1(T)`, its initial state :math:`[u_0(\\text{param}); v_0]`."""
    spring = _Spring(bkd)
    system = _system(bkd, spring, damper)
    discretization = _Discretization(system, spring)
    w0 = bkd.concatenate(
        [spring.initial_displacement(param[:, 0]), bkd.asarray([0.1, 0.0, -0.2])]
    )
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(EndpointTerm(StateEntries(bkd, 2 * _N, [1])),)
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


@pytest.mark.parametrize("damper", [None, "linear"])
def test_gradient_and_hvp_through_the_map(
    numpy_bkd: Backend[Array], damper: Optional[str]
) -> None:
    bkd = numpy_bkd

    def value(samples: Array) -> Array:
        return bkd.hstack(
            [
                _endpoint_function(bkd, damper, samples[:, k : k + 1])(
                    samples[:, k : k + 1]
                )
                for k in range(samples.shape[1])
            ]
        )

    def derivative(
        sample: Array,
    ) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
        return _endpoint_function(bkd, damper, sample)

    def jacobian(sample: Array) -> Array:
        field = derivative(sample).derivatives().jacobian
        if field is None:
            raise AssertionError("the function has no Jacobian")
        return field(sample)

    def hvp(sample: Array, vec: Array) -> Array:
        field = derivative(sample).derivatives().hvp
        if field is None:
            raise AssertionError("the function has no HVP")
        return field(sample, vec)

    wrapper = FunctionWithJacobianAndHVPFromCallable(
        nvars=2, fun=value, jacobian=jacobian, hvp=hvp, bkd=bkd
    )
    checker = DerivativeChecker(wrapper)
    errors = checker.check_derivatives(
        bkd.asarray([[0.2], [0.7]]),
        direction=bkd.asarray([[0.6], [-0.8]]),
        relative=True,
        central=True,
    )
    assert len(errors) == 2
    for error in errors:
        assert bkd.to_float(bkd.min(error)) <= 1e-7
        assert bkd.to_float(checker.error_ratio(error)) <= 1e-6


def test_midpoint_conserves_the_energy_of_a_linear_undamped_system(
    bkd: Backend[Array],
) -> None:
    spring = _Spring(bkd, cubic=0.0)
    spring.set_param(bkd.asarray([0.2, 0.0]))
    system = _system(bkd, spring)
    newton: Newton[Array, Array] = Newton(atol=1e-14, rtol=0.0, maxiters=20)
    stepper = RungeKutta(implicit_midpoint(), newton).stepper(
        system, RecomputeAlways(system.solver)
    )
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    w0 = bkd.reshape(bkd.asarray([0.5, -0.2, 0.1, 0.1, 0.0, -0.2]), (-1, 1))
    Integrator(stepper).solve(w0, FixedStep(0.05, (0.0, 10.0)), store)
    mass = bkd.asarray([1.0, 2.0, 1.5])
    stiffness = spring.stiffness()

    def energy(w: Array) -> float:
        u, v = w[:_N, 0], w[_N:, 0]
        return bkd.to_float(
            0.5 * bkd.sum(mass * v**2) + 0.5 * bkd.sum(u * bkd.dot(stiffness, u))
        )

    start = energy(store.state(0))
    drift = max(abs(energy(store.state(k)) - start) for k in range(store.ntimes()))
    assert store.ntimes() == 201
    assert drift <= 1e-12 * start


class TestRayleighDamping:
    def _system(
        self, bkd: Backend[Array], spring: _Spring[Array], alpha: float, beta: float
    ) -> _Dense[Array]:
        restoring = _restoring(bkd, spring)
        return second_order_system(
            restoring,
            BackendDenseStateBlocks(bkd, _N),
            InitialVelocityDerivatives.independent(),
            RayleighDamping(restoring, alpha, beta),
        )

    def test_value_reads_the_current_stiffness(self, bkd: Backend[Array]) -> None:
        spring = _Spring(bkd, cubic=0.0)
        system = self._system(bkd, spring, 0.3, 0.1)
        spring.set_param(bkd.asarray([0.5, 0.0]))
        state = _state(bkd)
        u, v = state[:_N], state[_N:]
        damping = -bkd.dot(
            0.3 * bkd.diag(bkd.asarray([1.0, 2.0, 1.5])) + 0.1 * spring.stiffness(),
            v,
        )
        bkd.assert_allclose(
            system.rhs(state, 0.3)[_N:], spring(u, 0.3) + damping, rtol=1e-13
        )

    def test_refuses_stiffness_damping_of_a_nonlinear_force(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(ValueError, match="affine"):
            RayleighDamping(_restoring(bkd, _Spring(bkd, cubic=0.4)), 0.3, 0.1)

    @pytest.mark.parametrize("beta, second", [(0.0, True), (0.1, False)])
    def test_tier(self, bkd: Backend[Array], beta: float, second: bool) -> None:
        derivatives = self._system(
            bkd, _Spring(bkd, cubic=0.0), 0.3, beta
        ).rhs.derivatives()
        assert derivatives.jacobian is not None
        assert (derivatives.state_state is not None) == second
        assert (derivatives.param is not None) == second

    @pytest.mark.parametrize("beta", [0.0, 0.1])
    def test_bundle_passes_the_checker(self, bkd: Backend[Array], beta: float) -> None:
        spring = _Spring(bkd, cubic=0.0)
        system = self._system(bkd, spring, 0.3, beta)
        second = beta == 0.0
        report = ODEResidualBundleChecker(
            _Checked(system, spring),
            initial_state=lambda param: bkd.concatenate(
                [spring.initial_displacement(param), bkd.zeros((_N,))]
            ),
        ).check(
            _state(bkd),
            0.3,
            adj=bkd.asarray([0.3, -0.1, 0.2, 0.5, 0.4, -0.7]) if second else None,
            param=bkd.asarray([0.2, 0.7]) if second else None,
        )
        assert report.failures() == []

    def test_midpoint_energy_never_increases(self, bkd: Backend[Array]) -> None:
        spring = _Spring(bkd, cubic=0.0)
        spring.set_param(bkd.asarray([0.2, 0.0]))
        system = self._system(bkd, spring, 0.2, 0.05)
        newton: Newton[Array, Array] = Newton(atol=1e-14, rtol=0.0, maxiters=20)
        stepper = RungeKutta(implicit_midpoint(), newton).stepper(
            system, RecomputeAlways(system.solver)
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        w0 = bkd.reshape(bkd.asarray([0.5, -0.2, 0.1, 0.1, 0.0, -0.2]), (-1, 1))
        Integrator(stepper).solve(w0, FixedStep(0.05, (0.0, 5.0)), store)
        mass, stiffness = bkd.asarray([1.0, 2.0, 1.5]), spring.stiffness()
        energies = [
            bkd.to_float(
                0.5 * bkd.sum(mass * store.state(k)[_N:, 0] ** 2)
                + 0.5
                * bkd.sum(
                    store.state(k)[:_N, 0] * bkd.dot(stiffness, store.state(k)[:_N, 0])
                )
            )
            for k in range(store.ntimes())
        ]
        increases = [b - a for a, b in zip(energies, energies[1:])]
        assert max(increases) <= 1e-14 * energies[0]
        assert energies[-1] < 0.5 * energies[0]
