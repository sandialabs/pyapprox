"""
Tests for TimeAdjointDerivativeChecker.

This module tests the derivative checking functionality at three levels:
1. ODE residual functions
2. Time residual functions
3. Full HVP accumulation
"""
import unittest
import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
from pyapprox.typing.pde.time.benchmarks.linear_ode import (
    LinearODEResidual,
    QuadraticODEResidual,
)
from pyapprox.typing.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.typing.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerResidual,
)
from pyapprox.typing.pde.time.implicit_steppers.integrator import TimeIntegrator
from pyapprox.typing.pde.time.functionals.endpoint import EndpointFunctional
from pyapprox.typing.pde.time.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.typing.pde.time.operator.check_derivatives import (
    TimeAdjointDerivativeChecker,
)


class TestTimeAdjointDerivativeChecker(unittest.TestCase):
    """Test the TimeAdjointDerivativeChecker with various ODE residuals."""

    def setUp(self):
        """Set up test fixtures."""
        self._bkd = NumpyBkd()
        self._nstates = 2
        self._nparams = 2
        self._deltat = 0.1
        self._nsteps = 5

    def _create_linear_ode_operator(self):
        """Create time adjoint operator with linear ODE."""
        Amat = self._bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        Bmat = self._bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        ode_residual = LinearODEResidual(Amat, Bmat, self._bkd)
        return self._create_operator(ode_residual)

    def _create_quadratic_ode_operator(self):
        """Create time adjoint operator with quadratic ODE."""
        Amat = self._bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        ode_residual = QuadraticODEResidual(Amat, self._bkd)
        return self._create_operator(ode_residual)

    def _create_operator(self, ode_residual):
        """Create time adjoint operator from ODE residual."""
        time_residual = HeunResidual(ode_residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0,
            final_time=self._nsteps * self._deltat,
            deltat=self._deltat,
            newton_solver=newton_solver,
        )
        functional = EndpointFunctional(
            state_idx=0,
            nstates=self._nstates,
            nparams=ode_residual.nparams(),
            bkd=self._bkd,
        )
        return TimeAdjointOperatorWithHVP(integrator, functional)

    def test_linear_ode_jacobian(self):
        """Test ODE Jacobian check with linear ODE."""
        operator = self._create_linear_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])

        operator._integrator._newton_solver._residual._residual.set_param(param)

        errors = checker.check_ode_jacobian(init_state, time=0.0, verbosity=0)
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            self.assertLess(min_err / max_err, 1e-6)

    def test_linear_ode_param_jacobian(self):
        """Test ODE param Jacobian check with linear ODE."""
        operator = self._create_linear_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])

        errors = checker.check_ode_param_jacobian(
            init_state, param, time=0.0, verbosity=0
        )
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            self.assertLess(min_err / max_err, 1e-6)

    def test_quadratic_ode_state_state_hvp(self):
        """Test ODE state-state HVP with quadratic ODE (non-zero HVP)."""
        operator = self._create_quadratic_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])
        adj_state = self._bkd.asarray([1.0, 0.0])

        operator._integrator._newton_solver._residual._residual.set_param(param)

        errors = checker.check_ode_state_state_hvp(
            init_state, adj_state, time=0.0, verbosity=0
        )
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            self.assertLess(min_err / max_err, 1e-6)

    def test_quadratic_ode_state_param_hvp(self):
        """Test ODE state-param HVP with quadratic ODE."""
        operator = self._create_quadratic_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])
        adj_state = self._bkd.asarray([1.0, 0.0])

        errors = checker.check_ode_state_param_hvp(
            init_state, param, adj_state, time=0.0, verbosity=0
        )
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            self.assertLess(min_err / max_err, 1e-6)

    def test_quadratic_ode_param_state_hvp(self):
        """Test ODE param-state HVP with quadratic ODE."""
        operator = self._create_quadratic_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])
        adj_state = self._bkd.asarray([1.0, 0.0])

        errors = checker.check_ode_param_state_hvp(
            init_state, param, adj_state, time=0.0, verbosity=0
        )
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            self.assertLess(min_err / max_err, 1e-6)

    def test_quadratic_ode_at_multiple_times(self):
        """Test ODE derivatives at multiple time points."""
        operator = self._create_quadratic_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])
        adj_state = self._bkd.asarray([1.0, 0.0])

        # Check at t=0, t=dt, t=2*dt
        for t in [0.0, self._deltat, 2 * self._deltat]:
            operator._integrator._newton_solver._residual._residual.set_param(
                param
            )

            errors = checker.check_ode_jacobian(init_state, time=t, verbosity=0)
            min_err = float(self._bkd.min(errors))
            max_err = float(self._bkd.max(errors))
            if max_err > 0:
                self.assertLess(
                    min_err / max_err, 1e-6, f"ODE jacobian failed at t={t}"
                )

            errors = checker.check_ode_param_jacobian(
                init_state, param, time=t, verbosity=0
            )
            min_err = float(self._bkd.min(errors))
            max_err = float(self._bkd.max(errors))
            if max_err > 0:
                self.assertLess(
                    min_err / max_err, 1e-6, f"ODE param_jacobian failed at t={t}"
                )


class TestTimeAdjointDerivativeCheckerTimeResidual(unittest.TestCase):
    """Test time residual derivative checks."""

    def setUp(self):
        """Set up test fixtures."""
        self._bkd = NumpyBkd()
        self._nstates = 2
        self._nparams = 2
        self._deltat = 0.1
        self._nsteps = 5

    def _create_quadratic_ode_operator(self):
        """Create time adjoint operator with quadratic ODE."""
        Amat = self._bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        ode_residual = QuadraticODEResidual(Amat, self._bkd)

        time_residual = HeunResidual(ode_residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0,
            final_time=self._nsteps * self._deltat,
            deltat=self._deltat,
            newton_solver=newton_solver,
        )
        functional = EndpointFunctional(
            state_idx=0,
            nstates=self._nstates,
            nparams=ode_residual.nparams(),
            bkd=self._bkd,
        )
        return TimeAdjointOperatorWithHVP(integrator, functional)

    def test_time_residual_param_jacobian(self):
        """Test time residual param Jacobian check."""
        operator = self._create_quadratic_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])

        # Get forward trajectory
        operator._integrator._newton_solver._residual._residual.set_param(param)
        operator.storage()._clear()
        fwd_sols, times = operator._get_forward_trajectory(init_state, param)

        fsol_nm1 = fwd_sols[:, 0]
        fsol_n = fwd_sols[:, 1]

        errors = checker.check_time_residual_param_jacobian(
            fsol_nm1, fsol_n, param, time=0.0, deltat=self._deltat, verbosity=0
        )
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            # Slightly relaxed tolerance for time residual derivatives due to
            # accumulated numerical error through time stepping
            self.assertLess(min_err / max_err, 1e-5)

    def test_time_residual_param_param_hvp(self):
        """Test time residual param-param HVP check."""
        operator = self._create_quadratic_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])
        adj_state = self._bkd.asarray([1.0, 0.0])

        # Get forward trajectory
        operator._integrator._newton_solver._residual._residual.set_param(param)
        operator.storage()._clear()
        fwd_sols, times = operator._get_forward_trajectory(init_state, param)

        fsol_nm1 = fwd_sols[:, 0]
        fsol_n = fwd_sols[:, 1]

        errors = checker.check_time_residual_param_param_hvp(
            fsol_nm1,
            fsol_n,
            param,
            adj_state,
            time=0.0,
            deltat=self._deltat,
            verbosity=0,
        )
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            # Slightly relaxed tolerance for param-param HVP due to
            # accumulated numerical error in second-order derivatives
            self.assertLess(min_err / max_err, 1e-5)


class TestTimeAdjointDerivativeCheckerFullOperator(unittest.TestCase):
    """Test full operator derivative checks."""

    def setUp(self):
        """Set up test fixtures."""
        self._bkd = NumpyBkd()
        self._nstates = 2
        self._nparams = 2
        self._deltat = 0.1
        self._nsteps = 5

    def _create_quadratic_ode_operator(self):
        """Create time adjoint operator with quadratic ODE."""
        Amat = self._bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        ode_residual = QuadraticODEResidual(Amat, self._bkd)

        time_residual = HeunResidual(ode_residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0,
            final_time=self._nsteps * self._deltat,
            deltat=self._deltat,
            newton_solver=newton_solver,
        )
        functional = EndpointFunctional(
            state_idx=0,
            nstates=self._nstates,
            nparams=ode_residual.nparams(),
            bkd=self._bkd,
        )
        return TimeAdjointOperatorWithHVP(integrator, functional)

    def test_full_jacobian(self):
        """Test full Jacobian check via adjoint method."""
        operator = self._create_quadratic_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])

        errors = checker.check_jacobian(init_state, param, verbosity=0)
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            self.assertLess(min_err / max_err, 1e-6)

    def test_full_hvp(self):
        """Test full HVP check via second-order adjoints."""
        operator = self._create_quadratic_ode_operator()
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = self._bkd.asarray([1.0, 0.5])
        param = self._bkd.asarray([[0.1], [0.2]])

        errors = checker.check_hvp(init_state, param, verbosity=0)
        min_err = float(self._bkd.min(errors))
        max_err = float(self._bkd.max(errors))
        if max_err > 0:
            self.assertLess(min_err / max_err, 1e-5)


if __name__ == "__main__":
    unittest.main()
