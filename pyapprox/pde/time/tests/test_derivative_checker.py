"""
Tests for TimeAdjointDerivativeChecker.

This module tests the derivative checking functionality at three levels:
1. ODE residual functions
2. Time residual functions
3. Full HVP accumulation
"""


from pyapprox.optimization.rootfinding.newton import NewtonSolver
from pyapprox.pde.time.benchmarks.linear_ode import (
    LinearODEResidual,
    QuadraticODEResidual,
)
from pyapprox.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.pde.time.functionals.endpoint import EndpointFunctional
from pyapprox.pde.time.implicit_steppers.integrator import TimeIntegrator
from pyapprox.pde.time.operator.check_derivatives import (
    TimeAdjointDerivativeChecker,
)
from pyapprox.pde.time.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestTimeAdjointDerivativeChecker:
    """Test the TimeAdjointDerivativeChecker with various ODE residuals."""

    def setup_method(self):
        """Set up test fixtures."""
        _bkd = NumpyBkd()
        self._nstates = 2
        self._nparams = 2
        self._deltat = 0.1
        self._nsteps = 5

    def _create_linear_ode_operator(self, bkd) :
        """Create time adjoint operator with linear ODE."""
        Amat = bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        Bmat = bkd.asarray([[1.0, 0.0], [0.0, 1.0]])
        ode_residual = LinearODEResidual(Amat, Bmat, bkd)
        return self._create_operator(bkd, ode_residual)

    def _create_quadratic_ode_operator(self, bkd) :
        """Create time adjoint operator with quadratic ODE."""
        Amat = bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        ode_residual = QuadraticODEResidual(Amat, bkd)
        return self._create_operator(bkd, ode_residual)

    def _create_operator(self, bkd, ode_residual) :
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
            bkd=bkd,
        )
        return TimeAdjointOperatorWithHVP(integrator, functional)

    def test_linear_ode_jacobian(self, bkd):
        """Test ODE Jacobian check with linear ODE."""
        operator = self._create_linear_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])

        operator._integrator._newton_solver._residual._residual.set_param(param)

        errors = checker.check_ode_jacobian(init_state, time=0.0, verbosity=0)
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            assert min_err / max_err < 1e-6

    def test_linear_ode_param_jacobian(self, bkd):
        """Test ODE param Jacobian check with linear ODE."""
        operator = self._create_linear_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])

        errors = checker.check_ode_param_jacobian(
            init_state, param, time=0.0, verbosity=0
        )
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            assert min_err / max_err < 1e-6

    def test_quadratic_ode_state_state_hvp(self, bkd):
        """Test ODE state-state HVP with quadratic ODE (non-zero HVP)."""
        operator = self._create_quadratic_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])
        adj_state = bkd.asarray([1.0, 0.0])

        operator._integrator._newton_solver._residual._residual.set_param(param)

        errors = checker.check_ode_state_state_hvp(
            init_state, adj_state, time=0.0, verbosity=0
        )
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            assert min_err / max_err < 1e-6

    def test_quadratic_ode_state_param_hvp(self, bkd):
        """Test ODE state-param HVP with quadratic ODE."""
        operator = self._create_quadratic_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])
        adj_state = bkd.asarray([1.0, 0.0])

        errors = checker.check_ode_state_param_hvp(
            init_state, param, adj_state, time=0.0, verbosity=0
        )
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            assert min_err / max_err < 1e-6

    def test_quadratic_ode_param_state_hvp(self, bkd):
        """Test ODE param-state HVP with quadratic ODE."""
        operator = self._create_quadratic_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])
        adj_state = bkd.asarray([1.0, 0.0])

        errors = checker.check_ode_param_state_hvp(
            init_state, param, adj_state, time=0.0, verbosity=0
        )
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            assert min_err / max_err < 1e-6

    def test_quadratic_ode_at_multiple_times(self, bkd):
        """Test ODE derivatives at multiple time points."""
        operator = self._create_quadratic_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])
        bkd.asarray([1.0, 0.0])

        # Check at t=0, t=dt, t=2*dt
        for t in [0.0, self._deltat, 2 * self._deltat]:
            operator._integrator._newton_solver._residual._residual.set_param(param)

            errors = checker.check_ode_jacobian(init_state, time=t, verbosity=0)
            min_err = float(bkd.min(errors))
            max_err = float(bkd.max(errors))
            if max_err > 0:
                assert min_err / max_err < 1e-6

            errors = checker.check_ode_param_jacobian(
                init_state, param, time=t, verbosity=0
            )
            min_err = float(bkd.min(errors))
            max_err = float(bkd.max(errors))
            if max_err > 0:
                assert min_err / max_err < 1e-6


class TestTimeAdjointDerivativeCheckerTimeResidual:
    """Test time residual derivative checks."""

    def setup_method(self):
        """Set up test fixtures."""
        _bkd = NumpyBkd()
        self._nstates = 2
        self._nparams = 2
        self._deltat = 0.1
        self._nsteps = 5

    def _create_quadratic_ode_operator(self, bkd) :
        """Create time adjoint operator with quadratic ODE."""
        Amat = bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        ode_residual = QuadraticODEResidual(Amat, bkd)

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
            bkd=bkd,
        )
        return TimeAdjointOperatorWithHVP(integrator, functional)

    def test_time_residual_param_jacobian(self, bkd):
        """Test time residual param Jacobian check."""
        operator = self._create_quadratic_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])

        # Get forward trajectory
        operator._integrator._newton_solver._residual._residual.set_param(param)
        operator.storage()._clear()
        fwd_sols, times = operator._get_forward_trajectory(init_state, param)

        fsol_nm1 = fwd_sols[:, 0]
        fsol_n = fwd_sols[:, 1]

        errors = checker.check_time_residual_param_jacobian(
            fsol_nm1, fsol_n, param, time=0.0, deltat=self._deltat, verbosity=0
        )
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            # Slightly relaxed tolerance for time residual derivatives due to
            # accumulated numerical error through time stepping
            assert min_err / max_err < 1e-5

    def test_time_residual_param_param_hvp(self, bkd):
        """Test time residual param-param HVP check."""
        operator = self._create_quadratic_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])
        adj_state = bkd.asarray([1.0, 0.0])

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
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            # Slightly relaxed tolerance for param-param HVP due to
            # accumulated numerical error in second-order derivatives
            assert min_err / max_err < 1e-5


class TestTimeAdjointDerivativeCheckerFullOperator:
    """Test full operator derivative checks."""

    def setup_method(self):
        """Set up test fixtures."""
        _bkd = NumpyBkd()
        self._nstates = 2
        self._nparams = 2
        self._deltat = 0.1
        self._nsteps = 5

    def _create_quadratic_ode_operator(self, bkd) :
        """Create time adjoint operator with quadratic ODE."""
        Amat = bkd.asarray([[-0.5, 0.1], [0.2, -0.3]])
        ode_residual = QuadraticODEResidual(Amat, bkd)

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
            bkd=bkd,
        )
        return TimeAdjointOperatorWithHVP(integrator, functional)

    def test_full_jacobian(self, bkd):
        """Test full Jacobian check via adjoint method."""
        operator = self._create_quadratic_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])

        errors = checker.check_jacobian(init_state, param, verbosity=0)
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            assert min_err / max_err < 1e-6

    def test_full_hvp(self, bkd):
        """Test full HVP check via second-order adjoints."""
        operator = self._create_quadratic_ode_operator(bkd)
        checker = TimeAdjointDerivativeChecker(operator)

        init_state = bkd.asarray([1.0, 0.5])
        param = bkd.asarray([[0.1], [0.2]])

        errors = checker.check_hvp(init_state, param, verbosity=0)
        min_err = float(bkd.min(errors))
        max_err = float(bkd.max(errors))
        if max_err > 0:
            assert min_err / max_err < 1e-5
