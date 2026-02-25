"""
Tests for Lotka-Volterra ODE residual with HVP support.

Uses TimeAdjointDerivativeChecker for all derivative tests.
"""

import unittest

from pyapprox.pde.time.benchmarks.lotka_volterra import (
    LotkaVolterraResidual,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestLotkaVolterraResidual(unittest.TestCase):
    """Test the Lotka-Volterra residual class."""

    def setUp(self):
        """Set up test fixtures."""
        self._bkd = NumpyBkd()
        self._nspecies = 3
        self._residual = LotkaVolterraResidual(self._nspecies, self._bkd)

        self._params = self._bkd.array(
            [0.5, 0.6, 0.7, 1.0, 0.1, 0.2, 0.1, 1.0, 0.1, 0.2, 0.1, 1.0]
        )
        self._residual.set_param(self._params)
        self._residual.set_time(0.0)

        self._state = self._bkd.array([0.3, 0.4, 0.3])
        self._adj_state = self._bkd.array([1.0, 0.5, 0.2])

    def test_residual_shape(self):
        """Test that residual returns correct shape."""
        result = self._residual(self._state)
        self.assertEqual(result.shape, (self._nspecies,))

    def test_jacobian_shape(self):
        """Test that Jacobian returns correct shape."""
        result = self._residual.jacobian(self._state)
        self.assertEqual(result.shape, (self._nspecies, self._nspecies))

    def test_param_jacobian_shape(self):
        """Test that param Jacobian returns correct shape."""
        result = self._residual.param_jacobian(self._state)
        nparams = self._nspecies + self._nspecies**2
        self.assertEqual(result.shape, (self._nspecies, nparams))

    def test_hvp_shapes(self):
        """Test that all HVP methods return correct shapes."""
        wvec = self._bkd.array([0.1, 0.2, 0.15])
        vvec = self._bkd.ones(12) * 0.1

        ss = self._residual.state_state_hvp(self._state, self._adj_state, wvec)
        self.assertEqual(ss.shape, (self._nspecies,))

        sp = self._residual.state_param_hvp(self._state, self._adj_state, vvec)
        self.assertEqual(sp.shape, (self._nspecies,))

        ps = self._residual.param_state_hvp(self._state, self._adj_state, wvec)
        nparams = self._nspecies + self._nspecies**2
        self.assertEqual(ps.shape, (nparams,))

        pp = self._residual.param_param_hvp(self._state, self._adj_state, vvec)
        self.assertEqual(pp.shape, (nparams,))

    def test_mass_matrix_is_identity(self):
        """Test that mass matrix is identity."""
        mass = self._residual.mass_matrix(self._nspecies)
        expected = self._bkd.eye(self._nspecies)
        self._bkd.assert_allclose(mass, expected, rtol=1e-14, atol=1e-14)


class TestLotkaVolterraWithTimeAdjoint(unittest.TestCase):
    """Test Lotka-Volterra with TimeAdjointDerivativeChecker."""

    def setUp(self):
        """Set up test fixtures."""
        self._bkd = NumpyBkd()
        self._nspecies = 3
        self._deltat = 0.1
        self._nsteps = 5

        self._params = self._bkd.array(
            [0.5, 0.6, 0.7, 1.0, 0.1, 0.2, 0.1, 1.0, 0.1, 0.2, 0.1, 1.0]
        )
        self._init_state = self._bkd.array([0.3, 0.4, 0.3])

    def _create_operator_with_endpoint(self):
        """Create TimeAdjointOperatorWithHVP with EndpointFunctional."""
        from pyapprox.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.pde.time.functionals.endpoint import EndpointFunctional
        from pyapprox.pde.time.implicit_steppers.backward_euler import (
            BackwardEulerResidual,
        )
        from pyapprox.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )
        from pyapprox.pde.time.operator import TimeAdjointOperatorWithHVP

        residual = LotkaVolterraResidual(self._nspecies, self._bkd)
        residual.set_param(self._params)

        time_residual = BackwardEulerResidual(residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0,
            final_time=self._nsteps * self._deltat,
            deltat=self._deltat,
            newton_solver=newton_solver,
        )
        functional = EndpointFunctional(
            state_idx=0,
            nstates=self._nspecies,
            nparams=residual.nparams(),
            bkd=self._bkd,
        )
        return TimeAdjointOperatorWithHVP(integrator, functional)

    def _create_operator_with_mse(self):
        """Create TimeAdjointOperatorWithHVP with TransientMSEFunctional."""
        from pyapprox.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.pde.time.functionals.mse import TransientMSEFunctional
        from pyapprox.pde.time.implicit_steppers.backward_euler import (
            BackwardEulerResidual,
        )
        from pyapprox.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )
        from pyapprox.pde.time.operator import TimeAdjointOperatorWithHVP

        residual = LotkaVolterraResidual(self._nspecies, self._bkd)
        residual.set_param(self._params)

        time_residual = BackwardEulerResidual(residual)
        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=0.0,
            final_time=self._nsteps * self._deltat,
            deltat=self._deltat,
            newton_solver=newton_solver,
        )

        obs_times = self._bkd.asarray([2, 4], dtype=int)
        obs_tuples = [(0, obs_times)]
        functional = TransientMSEFunctional(
            nstates=self._nspecies,
            nresidual_params=residual.nparams(),
            obs_tuples=obs_tuples,
            noise_std=0.1,
            bkd=self._bkd,
        )
        functional.set_observations(self._bkd.array([0.25, 0.2]))

        return TimeAdjointOperatorWithHVP(integrator, functional)

    def test_ode_derivatives_with_endpoint(self):
        """Test ODE-level derivatives using TimeAdjointDerivativeChecker."""
        from pyapprox.pde.time.operator.check_derivatives import (
            TimeAdjointDerivativeChecker,
        )

        operator = self._create_operator_with_endpoint()
        checker = TimeAdjointDerivativeChecker(operator)
        param_2d = self._params[:, None]
        adj_state = self._bkd.array([1.0, 0.5, 0.2])

        # Check ODE jacobian
        errors = checker.check_ode_jacobian(self._init_state, time=0.0)
        checker._assert_derivatives_close(errors, 1e-5, "ODE jacobian")

        # Check ODE param jacobian
        errors = checker.check_ode_param_jacobian(self._init_state, param_2d, time=0.0)
        checker._assert_derivatives_close(errors, 1e-5, "ODE param_jacobian")

        # Check ODE HVPs
        errors = checker.check_ode_state_state_hvp(
            self._init_state, adj_state, time=0.0
        )
        checker._assert_derivatives_close(errors, 1e-5, "ODE state_state_hvp")

        errors = checker.check_ode_state_param_hvp(
            self._init_state, param_2d, adj_state, time=0.0
        )
        checker._assert_derivatives_close(errors, 1e-5, "ODE state_param_hvp")

        errors = checker.check_ode_param_state_hvp(
            self._init_state, param_2d, adj_state, time=0.0
        )
        checker._assert_derivatives_close(errors, 1e-5, "ODE param_state_hvp")

        errors = checker.check_ode_param_param_hvp(
            self._init_state, param_2d, adj_state, time=0.0
        )
        checker._assert_derivatives_close(errors, 1e-5, "ODE param_param_hvp")

    def test_endpoint_jacobian(self):
        """Test full Jacobian with EndpointFunctional."""
        from pyapprox.pde.time.operator.check_derivatives import (
            TimeAdjointDerivativeChecker,
        )

        operator = self._create_operator_with_endpoint()
        checker = TimeAdjointDerivativeChecker(operator)
        param_2d = self._params[:, None]

        errors = checker.check_jacobian(self._init_state, param_2d)
        checker._assert_derivatives_close(errors, 1e-5, "Jacobian")

    def test_endpoint_hvp(self):
        """Test full HVP with EndpointFunctional."""
        from pyapprox.pde.time.operator.check_derivatives import (
            TimeAdjointDerivativeChecker,
        )

        operator = self._create_operator_with_endpoint()
        checker = TimeAdjointDerivativeChecker(operator)
        param_2d = self._params[:, None]

        errors = checker.check_hvp(self._init_state, param_2d)
        checker._assert_derivatives_close(errors, 1e-5, "HVP")

    def test_mse_jacobian(self):
        """Test full Jacobian with TransientMSEFunctional."""
        from pyapprox.pde.time.operator.check_derivatives import (
            TimeAdjointDerivativeChecker,
        )

        operator = self._create_operator_with_mse()
        checker = TimeAdjointDerivativeChecker(operator)
        param_2d = self._params[:, None]

        errors = checker.check_jacobian(self._init_state, param_2d)
        checker._assert_derivatives_close(errors, 1e-5, "Jacobian")

    def test_mse_hvp(self):
        """Test full HVP with TransientMSEFunctional."""
        from pyapprox.pde.time.operator.check_derivatives import (
            TimeAdjointDerivativeChecker,
        )

        operator = self._create_operator_with_mse()
        checker = TimeAdjointDerivativeChecker(operator)
        param_2d = self._params[:, None]

        errors = checker.check_hvp(self._init_state, param_2d)
        checker._assert_derivatives_close(errors, 1e-5, "HVP")


if __name__ == "__main__":
    unittest.main()
