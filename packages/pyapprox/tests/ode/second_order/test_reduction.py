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

from typing import Optional

import pytest

from pyapprox.implicitfunction.implicit_map import ImplicitMapFunction
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.ode.integration import Integrator
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    backend_spaces,
)
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.ode.second_order.assembled_blocks import backend_dense_state_blocks
from pyapprox.ode.second_order.rayleigh import RayleighDamping
from pyapprox.ode.second_order.reduction import (
    InitialVelocityDerivatives,
    second_order_system,
)
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import implicit_midpoint
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.transient_implicit_map import (
    TransientPoint,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.second_order_cases import (
    Checked,
    Dense,
    N,
    Spring,
    endpoint_function,
    reduced_system,
    restoring_system,
    sample_state,
    tridiagonal,
    velocity_derivative,
)


class TestStructure:
    def test_block_mass_and_its_solve(self, bkd: Backend[Array]) -> None:
        system = reduced_system(bkd, Spring(bkd))
        mass = bkd.diag(bkd.asarray([1.0, 1.0, 1.0, 1.0, 2.0, 1.5]))
        bkd.assert_allclose(system.mass.format().matrix(), mass, rtol=0.0, atol=0.0)
        assert system.mass.properties().spd
        rhs = bkd.reshape(sample_state(bkd), (-1, 1))
        solution, _ = system.solver.setup(system.mass).solve(rhs)
        bkd.assert_allclose(solution, bkd.solve(mass, rhs), rtol=1e-14)

    @pytest.mark.parametrize("damper", [None, "linear", "cubic"])
    def test_value_is_velocity_and_force(
        self, bkd: Backend[Array], damper: Optional[str]
    ) -> None:
        spring = Spring(bkd)
        system = reduced_system(bkd, spring, damper)
        state = sample_state(bkd)
        u, v = state[:N], state[N:]
        force = spring(u, 0.3)
        if damper == "linear":
            force = force - bkd.dot(tridiagonal(bkd, 0.5, 0.1), v)
        if damper == "cubic":
            force = force - 0.6 * u**2 * v
        bkd.assert_allclose(
            system.rhs(state, 0.3), bkd.concatenate([v, force]), rtol=1e-14
        )

    def test_refuses_a_storage_of_another_size(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            second_order_system(
                restoring_system(bkd, Spring(bkd)),
                backend_dense_state_blocks(bkd, N + 1),
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
        derivatives = reduced_system(bkd, Spring(bkd), damper).rhs.derivatives()
        assert derivatives.jacobian is not None
        assert (derivatives.param is not None) == (tier != "state")
        assert (derivatives.state_state is not None) == (tier == "second")

    def test_uncurved_initial_velocity_unknown_drops_to_first(
        self, bkd: Backend[Array]
    ) -> None:
        space, params = backend_spaces(bkd, N, 2)
        initial = InitialVelocityDerivatives[Array](STATED_ZERO, None)
        derivatives = reduced_system(bkd, Spring(bkd), None, initial).rhs.derivatives()
        assert derivatives.param is not None
        assert derivatives.state_state is None


@pytest.mark.parametrize(
    "damper, moving",
    [(None, False), (None, True), ("linear", True), ("cubic", False)],
)
def test_lifted_bundle_passes_the_checker(
    bkd: Backend[Array], damper: Optional[str], moving: bool
) -> None:
    spring = Spring(bkd)
    velocity = velocity_derivative(bkd) if moving else None
    system = reduced_system(bkd, spring, damper, velocity)
    v0 = bkd.asarray([0.1, 0.0, -0.2])

    def initial_state(param: Array) -> Array:
        if velocity is None:
            return bkd.concatenate([spring.initial_displacement(param), v0])
        matrix = bkd.asarray([[0.0, 0.4], [0.2, 0.0], [-0.1, 0.3]])
        return bkd.concatenate(
            [spring.initial_displacement(param), v0 + bkd.dot(matrix, param)]
        )

    report = ODEResidualBundleChecker(
        Checked(system, spring), initial_state=initial_state
    ).check(
        sample_state(bkd),
        0.3,
        adj=bkd.asarray([0.3, -0.1, 0.2, 0.5, 0.4, -0.7]),
        param=bkd.asarray([0.2, 0.7]),
    )
    assert report.failures() == []


@pytest.mark.parametrize("damper", [None, "linear"])
def test_gradient_and_hvp_through_the_map(
    numpy_bkd: Backend[Array], damper: Optional[str]
) -> None:
    bkd = numpy_bkd
    blocks = backend_dense_state_blocks(bkd, N)

    def value(samples: Array) -> Array:
        return bkd.hstack(
            [
                endpoint_function(bkd, damper, samples[:, k : k + 1], blocks)(
                    samples[:, k : k + 1]
                )
                for k in range(samples.shape[1])
            ]
        )

    def derivative(
        sample: Array,
    ) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
        return endpoint_function(bkd, damper, sample, blocks)

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
    spring = Spring(bkd, cubic=0.0)
    spring.set_param(bkd.asarray([0.2, 0.0]))
    system = reduced_system(bkd, spring)
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
        u, v = w[:N, 0], w[N:, 0]
        return bkd.to_float(
            0.5 * bkd.sum(mass * v**2) + 0.5 * bkd.sum(u * bkd.dot(stiffness, u))
        )

    start = energy(store.state(0))
    drift = max(abs(energy(store.state(k)) - start) for k in range(store.ntimes()))
    assert store.ntimes() == 201
    assert drift <= 1e-12 * start


class TestRayleighDamping:
    def _system(
        self,
        bkd: Backend[Array],
        spring: Spring[Array],
        alpha: float,
        beta: float,
        reference: Optional[Array] = None,
    ) -> Dense[Array]:
        restoring = restoring_system(bkd, spring)
        return second_order_system(
            restoring,
            backend_dense_state_blocks(bkd, N),
            InitialVelocityDerivatives.independent(),
            RayleighDamping(restoring, alpha, beta, reference),
        )

    def test_value_reads_the_current_stiffness(self, bkd: Backend[Array]) -> None:
        spring = Spring(bkd, cubic=0.0)
        system = self._system(bkd, spring, 0.3, 0.1)
        spring.set_param(bkd.asarray([0.5, 0.0]))
        state = sample_state(bkd)
        u, v = state[:N], state[N:]
        damping = -bkd.dot(
            0.3 * bkd.diag(bkd.asarray([1.0, 2.0, 1.5])) + 0.1 * spring.stiffness(),
            v,
        )
        bkd.assert_allclose(
            system.rhs(state, 0.3)[N:], spring(u, 0.3) + damping, rtol=1e-13
        )

    def test_refuses_stiffness_damping_of_a_nonlinear_force(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(ValueError, match="reference_state"):
            RayleighDamping(restoring_system(bkd, Spring(bkd, cubic=0.4)), 0.3, 0.1)

    def test_nonlinear_force_damps_with_the_stiffness_at_the_reference(
        self, bkd: Backend[Array]
    ) -> None:
        """K = -f_u(u_r) for the cubic spring, frozen while u moves, and
        following the parameters set after construction."""
        spring = Spring(bkd, cubic=0.4)
        reference = bkd.asarray([0.3, -0.5, 0.2])
        system = self._system(bkd, spring, 0.3, 0.1, reference)
        spring.set_param(bkd.asarray([0.5, 0.0]))
        state = sample_state(bkd)
        u, v = state[:N], state[N:]
        stiffness = spring.stiffness() + 1.2 * bkd.diag(reference**2)
        damping = -bkd.dot(
            0.3 * bkd.diag(bkd.asarray([1.0, 2.0, 1.5])) + 0.1 * stiffness, v
        )
        bkd.assert_allclose(
            system.rhs(state, 0.3)[N:], spring(u, 0.3) + damping, rtol=1e-13
        )

    def test_reference_damping_keeps_the_force_tier_without_parameters(
        self, bkd: Backend[Array]
    ) -> None:
        """Linear in v and independent of u, the damping is affine; its
        parameter dependence through K(u_r; p) is not supplied."""
        spring = Spring(bkd, cubic=0.4)
        reference = bkd.asarray([0.3, -0.5, 0.2])
        derivatives = self._system(
            bkd, spring, 0.3, 0.1, reference
        ).rhs.derivatives()
        assert derivatives.jacobian is not None
        assert derivatives.param is None
        report = ODEResidualBundleChecker(
            Checked(self._system(bkd, spring, 0.3, 0.1, reference), spring)
        ).check(sample_state(bkd), 0.3)
        assert report.failures() == []

    @pytest.mark.parametrize("beta, second", [(0.0, True), (0.1, False)])
    def test_tier(self, bkd: Backend[Array], beta: float, second: bool) -> None:
        derivatives = self._system(
            bkd, Spring(bkd, cubic=0.0), 0.3, beta
        ).rhs.derivatives()
        assert derivatives.jacobian is not None
        assert (derivatives.state_state is not None) == second
        assert (derivatives.param is not None) == second

    @pytest.mark.parametrize("beta", [0.0, 0.1])
    def test_bundle_passes_the_checker(self, bkd: Backend[Array], beta: float) -> None:
        spring = Spring(bkd, cubic=0.0)
        system = self._system(bkd, spring, 0.3, beta)
        second = beta == 0.0
        report = ODEResidualBundleChecker(
            Checked(system, spring),
            initial_state=lambda param: bkd.concatenate(
                [spring.initial_displacement(param), bkd.zeros((N,))]
            ),
        ).check(
            sample_state(bkd),
            0.3,
            adj=bkd.asarray([0.3, -0.1, 0.2, 0.5, 0.4, -0.7]) if second else None,
            param=bkd.asarray([0.2, 0.7]) if second else None,
        )
        assert report.failures() == []

    def test_midpoint_energy_never_increases(self, bkd: Backend[Array]) -> None:
        spring = Spring(bkd, cubic=0.0)
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
                0.5 * bkd.sum(mass * store.state(k)[N:, 0] ** 2)
                + 0.5
                * bkd.sum(
                    store.state(k)[:N, 0] * bkd.dot(stiffness, store.state(k)[:N, 0])
                )
            )
            for k in range(store.ntimes())
        ]
        increases = [b - a for a, b in zip(energies, energies[1:])]
        assert max(increases) <= 1e-14 * energies[0]
        assert energies[-1] < 0.5 * energies[0]
