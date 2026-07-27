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
from pyapprox.ode.time_quadrature import (
    MidpointTrajectoryQuadrature,
    NodalTrajectoryQuadrature,
    TrajectoryQuadratureProtocol,
)

_NSTATES = 3


class _ToyResidual:
    """Minimal ODEResidualProtocol implementation: dy/dt = -y."""

    def __init__(self, bkd) -> None:
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def __call__(self, state):
        return -state


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
        quadrature = stepper.trajectory_quadrature(
            _nonuniform_times(bkd)
        )
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
    def test_weights_sum_to_interval_length(
        self, bkd, stepper_class
    ) -> None:
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
    def test_sample_accumulate_adjoint_identity(
        self, bkd, stepper_class
    ) -> None:
        """accumulate is the transpose of sample:
        <S u, x> == <u, S^T x> for random u, x."""
        stepper = stepper_class(_ToyResidual(bkd))
        times = _nonuniform_times(bkd)
        quadrature = stepper.trajectory_quadrature(times)
        rng = np.random.default_rng(3)
        traj = bkd.asarray(
            rng.normal(0.0, 1.0, (_NSTATES, quadrature.ntimes()))
        )
        sampled_dir = bkd.asarray(
            rng.normal(0.0, 1.0, (_NSTATES, quadrature.nsamples()))
        )
        lhs = bkd.sum(quadrature.sample(traj) * sampled_dir)
        rhs = bkd.sum(traj * quadrature.accumulate(sampled_dir))
        bkd.assert_allclose(
            bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-13
        )

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


class TestQuadratureValidation:
    def test_nodal_shape_validation(self, bkd) -> None:
        with pytest.raises(ValueError, match="1D"):
            NodalTrajectoryQuadrature(
                bkd.asarray([[0]], dtype=int), bkd.ones((1,)), 2, bkd
            )
        with pytest.raises(ValueError, match="entries"):
            NodalTrajectoryQuadrature(
                bkd.asarray([0, 1], dtype=int), bkd.ones((1,)), 2, bkd
            )

    def test_midpoint_shape_validation(self, bkd) -> None:
        with pytest.raises(ValueError, match="ntimes - 1"):
            MidpointTrajectoryQuadrature(bkd.ones((3,)), 3, bkd)
