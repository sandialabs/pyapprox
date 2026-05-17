"""Tests for TimeIntegrator."""

import pytest

from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerAdjoint
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.mass_matrix import IdentityMassMatrix
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver


class _DecayResidual:
    """dy/dt = -y. Minimal ODEResidualWithParamJacobianProtocol."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._mass = IdentityMassMatrix(1, bkd)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_time(self, time: float) -> None:
        pass

    def __call__(self, state: Array) -> Array:
        return -state

    def jacobian(self, state: Array) -> Array:
        return -self._bkd.eye(state.shape[0])

    def mass_matrix(self) -> IdentityMassMatrix:
        return self._mass

    def nparams(self) -> int:
        return 0

    def set_param(self, param: Array) -> None:
        pass

    def param_jacobian(self, state: Array) -> Array:
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
