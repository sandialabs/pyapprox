"""Tests for TimeIntegrator."""

from typing import Any

import pytest

from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerAdjoint
from pyapprox.ode.functionals.time_integrated_weighted_l2 import (
    TimeIntegratedWeightedL2Functional,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.mass_matrix import IdentityMassMatrix
from pyapprox.ode.time_quadrature import trapezoidal_quadrature
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import EuclideanInnerProduct
from pyapprox.util.rootfinding.newton import NewtonSolver


class _DecayResidual:
    """dy/dt = -y. Minimal ODEResidualWithParamJacobianProtocol."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._mass = IdentityMassMatrix(1, bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, state: Array, time: float) -> Array:
        return -state

    def jacobian(self, state: Array, time: float) -> Array:
        return -self._bkd.eye(state.shape[0])

    def mass_matrix(self) -> IdentityMassMatrix:
        return self._mass

    def nparams(self) -> int:
        return 0

    def set_param(self, param: Array) -> None:
        pass

    def param_jacobian(self, state: Array, time: float) -> Array:
        return self._bkd.zeros((state.shape[0], 0))

    def initial_param_jacobian(self) -> Array:
        return self._bkd.zeros((1, 0))


class TestTimeIntegratorNtimes:
    @pytest.mark.parametrize(
        "init_time,final_time,deltat",
        [
            (0.0, 0.3, 0.01),
            (0.0, 1.0, 0.01),
            (0.0, 0.2, 0.05),
            (0.0, 1.0, 0.03),
            (0.5, 1.5, 0.1),
            (0.0, 0.1, 0.1),
        ],
    )
    def test_ntimes_matches_solve(self, numpy_bkd, init_time, final_time, deltat):
        bkd = numpy_bkd
        residual = _DecayResidual(bkd)
        stepper = ForwardEulerAdjoint(residual)
        newton = NewtonSolver(stepper)
        integrator = TimeIntegrator(init_time, final_time, deltat, newton)

        init_state = bkd.array([1.0])
        _, times = integrator.solve(init_state)
        assert integrator.ntimes() == times.shape[0]


class TestQuadratureInjection:
    """The integrator owns the scheme, so it injects the scheme's rule:
    on the planned times at set_functional, on the stepped times after
    solve. A caller's own rule, injected after the solve, wins."""

    def _setup(
        self, bkd: Backend[Array]
    ) -> tuple[TimeIntegrator[Array], ForwardEulerAdjoint[Array], Any]:
        stepper = ForwardEulerAdjoint(_DecayResidual(bkd))
        # A final step shorter than deltat: times 0, 0.1, 0.2, 0.25.
        integrator = TimeIntegrator(0.0, 0.25, 0.1, NewtonSolver(stepper))
        functional = TimeIntegratedWeightedL2Functional(
            EuclideanInnerProduct(1, bkd), 0, bkd
        )
        return integrator, stepper, functional

    def test_set_functional_needs_no_manual_quadrature(
        self, bkd: Backend[Array]
    ) -> None:
        integrator, _, functional = self._setup(bkd)
        integrator.set_functional(functional)
        assert functional.derivatives().state_jacobian is not None
        assert functional.time_quadrature().ntimes() == integrator.ntimes()

    def test_solve_injects_the_scheme_rule_on_stepped_times(
        self, bkd: Backend[Array]
    ) -> None:
        integrator, stepper, functional = self._setup(bkd)
        integrator.set_functional(functional)
        _, times = integrator.solve(bkd.array([1.0]))
        bkd.assert_allclose(times, bkd.array([0.0, 0.1, 0.2, 0.25]), rtol=1e-14)
        bkd.assert_allclose(
            functional.time_quadrature().nodal_weights(),
            stepper.trajectory_quadrature(times).nodal_weights(),
            rtol=1e-14,
        )

    def test_caller_rule_after_solve_wins(self, bkd: Backend[Array]) -> None:
        integrator, _, functional = self._setup(bkd)
        integrator.set_functional(functional)
        _, times = integrator.solve(bkd.array([1.0]))
        functional.set_time_quadrature(trapezoidal_quadrature(times, bkd))
        bkd.assert_allclose(
            functional.time_quadrature().nodal_weights(),
            trapezoidal_quadrature(times, bkd).nodal_weights(),
            rtol=1e-14,
        )
