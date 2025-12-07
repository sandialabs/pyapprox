"""
Tests for ODE benchmark residuals.

Tests Jacobian accuracy for CoupledSpringsResidual, HastingsEcologyResidual,
and ChemicalReactionResidual using TimeAdjointDerivativeChecker.
"""
import unittest

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.time.benchmarks import (
    CoupledSpringsResidual,
    HastingsEcologyResidual,
    ChemicalReactionResidual,
)


class TestCoupledSpringsResidual(unittest.TestCase):
    """Test CoupledSpringsResidual."""

    def setUp(self):
        self._bkd = NumpyBkd()
        self._residual = CoupledSpringsResidual(self._bkd)
        # Parameters: m1, m2, k1, k2, L1, L2, b1, b2, x1_0, y1_0, x2_0, y2_0
        self._params = self._bkd.array([
            1.0, 1.5, 8.0, 40.0, 0.5, 1.0, 0.8, 0.5,
            0.5, 0.0, 2.25, 0.0
        ])
        self._residual.set_param(self._params)
        self._residual.set_time(0.0)
        self._state = self._bkd.array([0.5, 0.1, 2.3, -0.05])

    def test_shapes(self):
        """Test output shapes."""
        result = self._residual(self._state)
        self.assertEqual(result.shape, (4,))

        jac = self._residual.jacobian(self._state)
        self.assertEqual(jac.shape, (4, 4))

        param_jac = self._residual.param_jacobian(self._state)
        self.assertEqual(param_jac.shape, (4, 12))

        init_jac = self._residual.initial_param_jacobian()
        self.assertEqual(init_jac.shape, (4, 12))

    def test_initial_condition(self):
        """Test initial condition extraction."""
        init = self._residual.get_initial_condition()
        self._bkd.assert_allclose(init, self._params[8:], rtol=1e-14)

    def test_jacobians_with_checker(self):
        """Test Jacobians using TimeAdjointDerivativeChecker."""
        from pyapprox.typing.pde.time.implicit_steppers.backward_euler import (
            BackwardEulerResidual,
        )
        from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.typing.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )
        from pyapprox.typing.pde.time.functionals.endpoint import EndpointFunctional
        from pyapprox.typing.pde.time.operator import TimeAdjointOperatorWithHVP
        from pyapprox.typing.pde.time.operator.check_derivatives import (
            TimeAdjointDerivativeChecker,
        )

        time_residual = BackwardEulerResidual(self._residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0, final_time=0.5, deltat=0.1, newton_solver=newton_solver
        )
        functional = EndpointFunctional(
            state_idx=0, nstates=4, nparams=12, bkd=self._bkd
        )
        operator = TimeAdjointOperatorWithHVP(integrator, functional)
        checker = TimeAdjointDerivativeChecker(operator)

        param_2d = self._params[:, None]
        errors = checker.check_ode_jacobian(self._state, time=0.0)
        checker._assert_derivatives_close(errors, 1e-5, "jacobian")

        errors = checker.check_ode_param_jacobian(self._state, param_2d, time=0.0)
        checker._assert_derivatives_close(errors, 1e-5, "param_jacobian")


class TestHastingsEcologyResidual(unittest.TestCase):
    """Test HastingsEcologyResidual."""

    def setUp(self):
        self._bkd = NumpyBkd()
        self._residual = HastingsEcologyResidual(self._bkd)
        # Parameters: a1, b1, a2, b2, d1, d2, y1_0, y2_0, y3_0
        self._params = self._bkd.array([
            5.0, 3.0, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0
        ])
        self._residual.set_param(self._params)
        self._residual.set_time(0.0)
        self._state = self._bkd.array([0.8, 0.2, 8.0])

    def test_shapes(self):
        """Test output shapes."""
        result = self._residual(self._state)
        self.assertEqual(result.shape, (3,))

        jac = self._residual.jacobian(self._state)
        self.assertEqual(jac.shape, (3, 3))

        param_jac = self._residual.param_jacobian(self._state)
        self.assertEqual(param_jac.shape, (3, 9))

        init_jac = self._residual.initial_param_jacobian()
        self.assertEqual(init_jac.shape, (3, 9))

    def test_initial_condition(self):
        """Test initial condition extraction."""
        init = self._residual.get_initial_condition()
        self._bkd.assert_allclose(init, self._params[6:], rtol=1e-14)

    def test_jacobians_with_checker(self):
        """Test Jacobians using TimeAdjointDerivativeChecker."""
        from pyapprox.typing.pde.time.implicit_steppers.backward_euler import (
            BackwardEulerResidual,
        )
        from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.typing.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )
        from pyapprox.typing.pde.time.functionals.endpoint import EndpointFunctional
        from pyapprox.typing.pde.time.operator import TimeAdjointOperatorWithHVP
        from pyapprox.typing.pde.time.operator.check_derivatives import (
            TimeAdjointDerivativeChecker,
        )

        time_residual = BackwardEulerResidual(self._residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0, final_time=5.0, deltat=1.0, newton_solver=newton_solver
        )
        functional = EndpointFunctional(
            state_idx=0, nstates=3, nparams=9, bkd=self._bkd
        )
        operator = TimeAdjointOperatorWithHVP(integrator, functional)
        checker = TimeAdjointDerivativeChecker(operator)

        param_2d = self._params[:, None]
        errors = checker.check_ode_jacobian(self._state, time=0.0)
        checker._assert_derivatives_close(errors, 1e-5, "jacobian")

        errors = checker.check_ode_param_jacobian(self._state, param_2d, time=0.0)
        checker._assert_derivatives_close(errors, 1e-5, "param_jacobian")


class TestChemicalReactionResidual(unittest.TestCase):
    """Test ChemicalReactionResidual."""

    def setUp(self):
        self._bkd = NumpyBkd()
        self._residual = ChemicalReactionResidual(self._bkd)
        # Parameters: a, b, c, d, e, f
        self._params = self._bkd.array([1.6, 20.75, 0.04, 1.0, 0.36, 0.016])
        self._residual.set_param(self._params)
        self._residual.set_time(0.0)
        self._state = self._bkd.array([0.2, 0.3, 0.1])

    def test_shapes(self):
        """Test output shapes."""
        result = self._residual(self._state)
        self.assertEqual(result.shape, (3,))

        jac = self._residual.jacobian(self._state)
        self.assertEqual(jac.shape, (3, 3))

        param_jac = self._residual.param_jacobian(self._state)
        self.assertEqual(param_jac.shape, (3, 6))

        init_jac = self._residual.initial_param_jacobian()
        self.assertEqual(init_jac.shape, (3, 6))

    def test_initial_condition(self):
        """Test initial condition is zeros."""
        init = self._residual.get_initial_condition()
        expected = self._bkd.zeros(3)
        self._bkd.assert_allclose(init, expected, rtol=1e-14)

    def test_jacobians_with_checker(self):
        """Test Jacobians using TimeAdjointDerivativeChecker."""
        from pyapprox.typing.pde.time.implicit_steppers.backward_euler import (
            BackwardEulerResidual,
        )
        from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.typing.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )
        from pyapprox.typing.pde.time.functionals.endpoint import EndpointFunctional
        from pyapprox.typing.pde.time.operator import TimeAdjointOperatorWithHVP
        from pyapprox.typing.pde.time.operator.check_derivatives import (
            TimeAdjointDerivativeChecker,
        )

        time_residual = BackwardEulerResidual(self._residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0, final_time=1.0, deltat=0.1, newton_solver=newton_solver
        )
        functional = EndpointFunctional(
            state_idx=0, nstates=3, nparams=6, bkd=self._bkd
        )
        operator = TimeAdjointOperatorWithHVP(integrator, functional)
        checker = TimeAdjointDerivativeChecker(operator)

        param_2d = self._params[:, None]
        errors = checker.check_ode_jacobian(self._state, time=0.0)
        checker._assert_derivatives_close(errors, 1e-5, "jacobian")

        errors = checker.check_ode_param_jacobian(self._state, param_2d, time=0.0)
        checker._assert_derivatives_close(errors, 1e-5, "param_jacobian")


if __name__ == "__main__":
    unittest.main()
