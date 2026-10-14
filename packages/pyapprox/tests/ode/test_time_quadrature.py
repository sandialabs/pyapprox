"""Tests for scheme-implied trajectory quadrature.

Pattern under test: every stepper reports the quadrature its scheme
implies via ``stepper.trajectory_quadrature(times)``. Consumers
integrating over a trajectory obtain the rule from the stepper of the
actual solve — hand-rolled weights are the bug this design makes
unrepresentable (a rule of the wrong order silently degrades the
scheme's convergence order).
"""

import numpy as np
import pytest

from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerHVP
from pyapprox.ode.explicit_steppers.heun import HeunHVP
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerHVP
from pyapprox.ode.implicit_steppers.crank_nicolson import CrankNicolsonHVP
from pyapprox.ode.implicit_steppers.implicit_midpoint import (
    ImplicitMidpointHVP,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.mass_matrix import ConstantDenseMassMatrix, IdentityMassMatrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.ode.time_quadrature import (
    MidpointTrajectoryQuadrature,
    NodalTrajectoryQuadrature,
    OneStepMassTelescope,
    StepResidualDeclaration,
    StepResidualRuleProtocol,
    TrajectoryQuadratureProtocol,
)
from pyapprox.util.rootfinding.newton import NewtonSolver

_NSTATES = 3


class _ToyResidual:
    """Minimal ODEResidualProtocol implementation: dy/dt = -y."""

    def __init__(self, bkd) -> None:
        self._bkd = bkd
        self._mass = IdentityMassMatrix(_NSTATES, bkd)

    def bkd(self):
        return self._bkd

    def __call__(self, state):
        return -state

    def mass_matrix(self):
        return self._mass


class _WrongLengthTelescope:
    def coefficients(self, times, bkd):
        return bkd.zeros((times.shape[0] + 1,))


class _ForcedCubicResidual(DefaultNewtonJacobianMixin):
    """M dy/dt = -y^3 + sin(3 t) b with a dense, non-identity M.

    Nonlinear and time-dependent, so ``F`` at the wrong state or the
    wrong time shows up in the summed step residuals.
    """

    def __init__(self, bkd) -> None:
        self._bkd = bkd
        self._mass = ConstantDenseMassMatrix(bkd.asarray([[2.0, 0.5], [0.5, 1.0]]), bkd)
        self._forcing = bkd.asarray([1.0, -0.5])
        self._time = 0.0

    def bkd(self):
        return self._bkd

    def set_time(self, time: float) -> None:
        self._time = time

    def __call__(self, state):
        return -(state**3) + np.sin(3.0 * self._time) * self._forcing

    def jacobian(self, state):
        return self._bkd.diag(-3.0 * state**2)

    def mass_matrix(self):
        return self._mass

    def nparams(self) -> int:
        return 0

    def set_param(self, param) -> None:
        pass

    def param_jacobian(self, state):
        return self._bkd.zeros((state.shape[0], 0))

    def initial_param_jacobian(self):
        return self._bkd.zeros((2, 0))


def _summed_step_residuals(bkd, stepper_class):
    """Solve, then rebuild the summed step residuals
    M Y c - sum_j w_j F(y_j, t_j) entirely from the stepper's own
    step-residual rule: its mass, telescope, stage states, stage times
    and weights."""
    residual = _ForcedCubicResidual(bkd)
    stepper = stepper_class(residual)
    newton = NewtonSolver(stepper)
    # Each step residual is then zero to roundoff, so the rebuilt sum
    # is too; the default tolerance (1e-7) would hide a small defect.
    newton.set_options(atol=1e-14, rtol=1e-14)
    integrator = TimeIntegrator(0.0, 0.5, 0.05, newton, verbosity=0)
    traj, times = integrator.solve(bkd.asarray([1.0, 0.5]))
    rule = stepper.trajectory_quadrature(times).step_residual_rule()
    assert rule is not None
    total = rule.mass_matrix().apply(bkd.dot(traj, rule.mass_telescope_coefficients()))
    stages = rule.sample(traj)
    stage_times = rule.sample_times()
    weights = rule.weights()
    for jj in range(rule.nsamples()):
        residual.set_time(bkd.to_float(stage_times[jj]))
        total = total - weights[jj] * residual(stages[:, jj])
    return total


def _uniform_times(bkd, ntimes=6, deltat=0.1):
    return bkd.asarray(np.linspace(0.0, deltat * (ntimes - 1), ntimes))


def _nonuniform_times(bkd):
    return bkd.asarray(np.array([0.0, 0.1, 0.25, 0.3, 0.55]))


class TestStepperTrajectoryQuadrature:
    @pytest.mark.parametrize(
        "stepper_class,expected_nodal",
        [
            # Scheme-implied rules: BE reconstructs right-constant,
            # FE left-constant, CN IS the trapezoid rule, Heun is the
            # explicit trapezoid rule.
            (BackwardEulerHVP, "right"),
            (ForwardEulerHVP, "left"),
            (CrankNicolsonHVP, "trapezoid"),
            (HeunHVP, "trapezoid"),
        ],
    )
    def test_nodal_schemes_report_their_rule(
        self, bkd, stepper_class, expected_nodal
    ) -> None:
        stepper = stepper_class(_ToyResidual(bkd))
        times_np = np.asarray(bkd.to_numpy(_nonuniform_times(bkd)))
        quadrature = stepper.trajectory_quadrature(_nonuniform_times(bkd))
        assert isinstance(quadrature, TrajectoryQuadratureProtocol)
        assert isinstance(quadrature, NodalTrajectoryQuadrature)
        assert quadrature.is_time_diagonal()
        deltas = np.diff(times_np)
        ntimes = times_np.shape[0]
        expected = np.zeros(ntimes)
        if expected_nodal == "right":
            expected[1:] = deltas
        elif expected_nodal == "left":
            expected[:-1] = deltas
        else:
            expected[:-1] += 0.5 * deltas
            expected[1:] += 0.5 * deltas
        bkd.assert_allclose(
            quadrature.nodal_weights(), bkd.asarray(expected), rtol=1e-14
        )

    def test_implicit_midpoint_reports_midpoint_rule(self, bkd) -> None:
        stepper = ImplicitMidpointHVP(_ToyResidual(bkd))
        quadrature = stepper.trajectory_quadrature(_uniform_times(bkd))
        assert isinstance(quadrature, MidpointTrajectoryQuadrature)
        assert not quadrature.is_time_diagonal()
        with pytest.raises(ValueError, match="couples adjacent"):
            quadrature.nodal_weights()

    @pytest.mark.parametrize(
        "stepper_class",
        [
            BackwardEulerHVP,
            ForwardEulerHVP,
            CrankNicolsonHVP,
            HeunHVP,
            ImplicitMidpointHVP,
        ],
    )
    def test_weights_sum_to_interval_length(self, bkd, stepper_class) -> None:
        """Every rule integrates the constant 1 exactly."""
        stepper = stepper_class(_ToyResidual(bkd))
        for times in (_uniform_times(bkd), _nonuniform_times(bkd)):
            quadrature = stepper.trajectory_quadrature(times)
            total = bkd.sum(quadrature.weights())
            span = times[-1] - times[0]
            bkd.assert_allclose(
                bkd.reshape(total, (1,)),
                bkd.reshape(span, (1,)),
                rtol=1e-14,
            )

    @pytest.mark.parametrize(
        "stepper_class",
        [BackwardEulerHVP, CrankNicolsonHVP, ImplicitMidpointHVP],
    )
    def test_sample_accumulate_adjoint_identity(self, bkd, stepper_class) -> None:
        """accumulate is the transpose of sample:
        <S u, x> == <u, S^T x> for random u, x."""
        stepper = stepper_class(_ToyResidual(bkd))
        times = _nonuniform_times(bkd)
        quadrature = stepper.trajectory_quadrature(times)
        rng = np.random.default_rng(3)
        traj = bkd.asarray(rng.normal(0.0, 1.0, (_NSTATES, quadrature.ntimes())))
        sampled_dir = bkd.asarray(
            rng.normal(0.0, 1.0, (_NSTATES, quadrature.nsamples()))
        )
        lhs = bkd.sum(quadrature.sample(traj) * sampled_dir)
        rhs = bkd.sum(traj * quadrature.accumulate(sampled_dir))
        bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-13)

    def test_trapezoid_exact_for_linear_integrand(self, bkd) -> None:
        """Trapezoid integrates a linear-in-time trajectory exactly:
        int_0^T t dt = T^2/2 with q the first component."""
        stepper = CrankNicolsonHVP(_ToyResidual(bkd))
        times = _nonuniform_times(bkd)
        quadrature = stepper.trajectory_quadrature(times)
        traj = bkd.reshape(times, (1, quadrature.ntimes()))
        sampled = quadrature.sample(traj)
        value = bkd.sum(quadrature.weights() * sampled[0, :])
        span = float(bkd.to_numpy(times[-1]))
        bkd.assert_allclose(
            bkd.reshape(value, (1,)),
            bkd.asarray(np.array([0.5 * span**2])),
            rtol=1e-14,
        )

    def test_midpoint_exact_for_linear_integrand(self, bkd) -> None:
        """The midpoint rule is also second order: exact on linears."""
        stepper = ImplicitMidpointHVP(_ToyResidual(bkd))
        times = _nonuniform_times(bkd)
        quadrature = stepper.trajectory_quadrature(times)
        traj = bkd.reshape(times, (1, quadrature.ntimes()))
        sampled = quadrature.sample(traj)
        value = bkd.sum(quadrature.weights() * sampled[0, :])
        span = float(bkd.to_numpy(times[-1]))
        bkd.assert_allclose(
            bkd.reshape(value, (1,)),
            bkd.asarray(np.array([0.5 * span**2])),
            rtol=1e-14,
        )


class TestSampleTimes:
    @pytest.mark.parametrize(
        "stepper_class,expected",
        [
            (BackwardEulerHVP, "right"),
            (ForwardEulerHVP, "left"),
            (CrankNicolsonHVP, "all"),
            (ImplicitMidpointHVP, "midpoints"),
        ],
    )
    def test_samples_are_timed_where_sampled(
        self, bkd, stepper_class, expected: str
    ) -> None:
        times = _nonuniform_times(bkd)
        quadrature = stepper_class(_ToyResidual(bkd)).trajectory_quadrature(times)
        expected_times = {
            "right": times[1:],
            "left": times[:-1],
            "all": times,
            "midpoints": 0.5 * (times[:-1] + times[1:]),
        }[expected]
        bkd.assert_allclose(quadrature.sample_times(), expected_times)


_ONE_STEP_STEPPERS = [
    BackwardEulerHVP,
    ForwardEulerHVP,
    CrankNicolsonHVP,
    ImplicitMidpointHVP,
]


class TestStepResidualRule:
    @pytest.mark.parametrize("stepper_class", _ONE_STEP_STEPPERS)
    def test_one_step_schemes_provide_it(self, bkd, stepper_class) -> None:
        residual = _ToyResidual(bkd)
        quadrature = stepper_class(residual).trajectory_quadrature(
            _nonuniform_times(bkd)
        )
        rule = quadrature.step_residual_rule()
        assert isinstance(rule, StepResidualRuleProtocol)
        assert rule.is_linear()
        # The mass the scheme steps with, not a copy or a recomputation.
        assert rule.mass_matrix() is residual.mass_matrix()
        bkd.assert_allclose(
            rule.mass_telescope_coefficients(),
            bkd.asarray([-1.0, 0.0, 0.0, 0.0, 1.0]),
        )
        bkd.assert_allclose(rule.sample_times(), quadrature.sample_times())
        bkd.assert_allclose(rule.weights(), quadrature.weights())

    def test_heun_provides_none(self, bkd) -> None:
        quadrature = HeunHVP(_ToyResidual(bkd)).trajectory_quadrature(
            _nonuniform_times(bkd)
        )
        assert quadrature.step_residual_rule() is None

    def test_undeclared_by_default(self, bkd) -> None:
        quadrature = NodalTrajectoryQuadrature(
            bkd.asarray([0, 1, 2, 3, 4], dtype=int),
            bkd.ones((5,)),
            _nonuniform_times(bkd),
            bkd,
        )
        assert quadrature.step_residual_rule() is None

    @pytest.mark.parametrize("stepper_class", _ONE_STEP_STEPPERS)
    def test_rule_reproduces_the_summed_steps(self, numpy_bkd, stepper_class) -> None:
        """At the solution every step residual vanishes, so their sum,
        rebuilt from the rule alone, vanishes to the solver tolerance."""
        total = _summed_step_residuals(numpy_bkd, stepper_class)
        numpy_bkd.assert_allclose(total, numpy_bkd.zeros((2,)), atol=1e-12)


class TestQuadratureValidation:
    def test_nodal_shape_validation(self, bkd) -> None:
        times = bkd.asarray([0.0, 1.0])
        with pytest.raises(ValueError, match="1D"):
            NodalTrajectoryQuadrature(
                bkd.asarray([[0]], dtype=int), bkd.ones((1,)), times, bkd
            )
        with pytest.raises(ValueError, match="entries"):
            NodalTrajectoryQuadrature(
                bkd.asarray([0, 1], dtype=int), bkd.ones((1,)), times, bkd
            )

    def test_midpoint_shape_validation(self, bkd) -> None:
        with pytest.raises(ValueError, match="ntimes - 1"):
            MidpointTrajectoryQuadrature(
                bkd.ones((3,)), bkd.asarray([0.0, 1.0, 2.0]), bkd
            )

    def test_rejects_a_telescope_of_the_wrong_length(self, bkd) -> None:
        with pytest.raises(ValueError, match="expected \\(3,\\)"):
            NodalTrajectoryQuadrature(
                bkd.asarray([0, 1, 2], dtype=int),
                bkd.ones((3,)),
                bkd.asarray([0.0, 1.0, 2.0]),
                bkd,
                StepResidualDeclaration(
                    _WrongLengthTelescope(), IdentityMassMatrix(1, bkd)
                ),
            )

    def test_rejects_a_bad_declaration(self, bkd) -> None:
        times = bkd.asarray([0.0, 1.0, 2.0])
        mass = IdentityMassMatrix(1, bkd)
        with pytest.raises(TypeError, match="StepResidualDeclaration"):
            MidpointTrajectoryQuadrature(
                bkd.ones((2,)),
                times,
                bkd,
                object(),  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="MassTelescopeProtocol"):
            MidpointTrajectoryQuadrature(
                bkd.ones((2,)),
                times,
                bkd,
                StepResidualDeclaration(object(), mass),  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="MassMatrixProtocol"):
            MidpointTrajectoryQuadrature(
                bkd.ones((2,)),
                times,
                bkd,
                StepResidualDeclaration(
                    OneStepMassTelescope(),
                    object(),  # type: ignore[arg-type]
                ),
            )
