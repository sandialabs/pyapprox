"""
Test individual HVP components using DerivativeChecker.

This module tests each component of the HVP computation:
1. ODE residual HVP methods (state_state_hvp, state_param_hvp, etc.)
2. Time stepping residual HVP methods
3. Forward/backward ODE solutions against analytical (sympy) solutions

Uses DerivativeChecker with error_ratio tolerance 1e-6.
"""

import unittest
from typing import Generic

import numpy as np
import sympy as sp

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests
from pyapprox.pde.time.benchmarks.linear_ode import (
    QuadraticODEResidual,
)
from pyapprox.pde.time.implicit_steppers.backward_euler import (
    BackwardEulerResidual,
)
from pyapprox.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerResidual,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


# =============================================================================
# Sympy Analytical Solutions
# =============================================================================


def sympy_backward_euler_step(y_prev, p, dt, a):
    """
    Solve one Backward Euler step for quadratic ODE analytically.

    ODE: dy/dt = a*y + p[0]*y^2 + p[1]
    Backward Euler: y_n - y_{n-1} - dt*(a*y_n + p[0]*y_n^2 + p[1]) = 0

    Expanding:
        y_n - y_{n-1} - dt*a*y_n - dt*p[0]*y_n^2 - dt*p[1] = 0
        (1 - dt*a)*y_n - dt*p[0]*y_n^2 - y_{n-1} - dt*p[1] = 0

    Standard form: A*y^2 + B*y + C = 0 with A > 0:
        dt*p[0]*y_n^2 - (1 - dt*a)*y_n + (y_{n-1} + dt*p[1]) = 0
    """
    A = dt * p[0]
    B = -(1 - a * dt)
    C = y_prev + dt * p[1]

    if abs(A) < 1e-12:
        # Linear case: B*y + C = 0
        return -C / B

    disc = B**2 - 4 * A * C
    # Take the root that's closer to y_prev (stability)
    y_plus = (-B + np.sqrt(disc)) / (2 * A)
    y_minus = (-B - np.sqrt(disc)) / (2 * A)

    # Choose the root closer to y_prev
    if abs(y_plus - y_prev) < abs(y_minus - y_prev):
        return y_plus
    return y_minus


def sympy_forward_euler_step(y_prev, p, dt, a):
    """
    Solve one Forward Euler step for quadratic ODE.

    ODE: dy/dt = a*y + p[0]*y^2 + p[1]
    Forward Euler: y_n = y_{n-1} + dt*(a*y_{n-1} + p[0]*y_{n-1}^2 + p[1])
    """
    return y_prev + dt * (a * y_prev + p[0] * y_prev**2 + p[1])


def sympy_solve_forward(y0, p, dt, nsteps, a, method="backward_euler"):
    """Solve ODE forward in time with given method."""
    y = np.zeros(nsteps + 1)
    y[0] = y0

    step_func = (
        sympy_backward_euler_step
        if method == "backward_euler"
        else sympy_forward_euler_step
    )

    for n in range(nsteps):
        y[n + 1] = step_func(y[n], p, dt, a)

    return y


def sympy_adjoint_backward_euler(fwd_sols, dt, p, a, dQdy_final, dQdy_all=None):
    """
    Solve adjoint equations backward for Backward Euler.

    The Backward Euler residual is:
        R_n(y_n, y_{n-1}) = y_n - y_{n-1} - dt * f(y_n)

    where f(y) = a*y + p[0]*y^2 + p[1].

    Jacobians:
        dR_n/dy_n = 1 - dt*(a + 2*p[0]*y_n)    (diagonal term)
        dR_{n+1}/dy_n = -1                      (off-diagonal coupling)

    Adjoint equation at step n:
        (dR_n/dy_n)^T λ_n = -dR_{n+1}/dy_n^T λ_{n+1} - dQ/dy_n

    Note: R_{n+1} depends on y_n through the term -y_{n-1} in R_{n+1}.
    So dR_{n+1}/dy_n = -I (not +I).

    Parameters
    ----------
    fwd_sols : array
        Forward solutions at each time step.
    dt : float
        Time step size.
    p : list
        Parameters [p0, p1].
    a : float
        Linear coefficient.
    dQdy_final : float
        dQ/dy at final time.
    dQdy_all : array, optional
        dQ/dy at all time steps. If None, uses zeros for intermediate steps.
    """
    nsteps = len(fwd_sols) - 1
    lam = np.zeros(nsteps + 1)

    if dQdy_all is None:
        dQdy_all = np.zeros(nsteps + 1)
        dQdy_all[-1] = dQdy_final

    # Final time: (dR_N/dy_N)^T λ_N = -dQ/dy_N
    # dR_N/dy_N is evaluated at y_N
    dRdy_N = 1 - dt * (a + 2 * p[0] * fwd_sols[-1])
    lam[-1] = -dQdy_all[-1] / dRdy_N

    # Backward sweep from N-1 down to 1
    # Each step: (dR_n/dy_n)^T λ_n = -dR_{n+1}/dy_n^T λ_{n+1} - dQ/dy_n
    for n in range(nsteps - 1, 0, -1):
        # dR_n/dy_n at y_n
        dRn_dyn = 1 - dt * (a + 2 * p[0] * fwd_sols[n])
        # dR_{n+1}/dy_n = -1 (from the -y_{n-1} term in R_{n+1})
        dRnp1_dyn = -1
        lam[n] = (-dRnp1_dyn * lam[n + 1] - dQdy_all[n]) / dRn_dyn

    # At n=0: λ_0 is computed via adjoint_final_solution
    # Solve: M^T λ_0 = -B_1^T λ_1 - dQ/dy_0
    # For identity mass matrix: λ_0 = -(-1)*λ_1 - dQ/dy_0 = λ_1 - dQ/dy_0
    lam[0] = lam[1] - dQdy_all[0]

    return lam


# =============================================================================
# Wrapper for ODE Residual HVP Testing
# =============================================================================


class ODEResidualHVPWrapper(Generic[Array]):
    """
    Wrapper to test ODE residual HVP methods with DerivativeChecker.

    Tests the state_state_hvp method by wrapping:
    f(y) -> λ^T · df/dy(y) (the Jacobian contracted with adjoint)

    Then the HVP of this function is λ^T · d²f/dy² · w.
    """

    def __init__(
        self,
        ode_residual: QuadraticODEResidual[Array],
        adj_state: Array,
        bkd_: Backend[Array],
    ):
        self._residual = ode_residual
        self._adj_state = adj_state
        self._bkd = bkd_

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return self._adj_state.shape[0]

    def nvars(self) -> int:
        return self._adj_state.shape[0]  # nstates

    def __call__(self, state: Array) -> Array:
        """Return λ^T · df/dy(y) as a vector."""
        jac = self._residual.jacobian(state.flatten())
        return (jac.T @ self._adj_state.reshape(-1, 1)).reshape(1, -1)

    def jacobian(self, state: Array) -> Array:
        """Return d/dy [λ^T · df/dy] = λ^T · d²f/dy²."""
        # For quadratic ODE: df/dy = A + 2*p[0]*diag(y)
        # d²f/dy² is a tensor, but contracted with λ gives a matrix
        # For diagonal Hessian: d/dy_j [λ^T · df/dy]_i = λ_i * 2*p[0] * δ_{ij}
        nstates = state.shape[0] if state.ndim == 1 else state.shape[0]
        p0 = float(self._residual._param[0, 0])
        # Result is diagonal: 2*p[0]*diag(λ)
        return 2.0 * p0 * self._bkd.diag(self._adj_state.flatten())


class ODEResidualParamHVPWrapper(Generic[Array]):
    """
    Wrapper to test ODE residual param_state_hvp method.

    Tests: λ^T · d²f/dp dy · w
    """

    def __init__(
        self,
        ode_residual: QuadraticODEResidual[Array],
        adj_state: Array,
        state: Array,
        bkd_: Backend[Array],
    ):
        self._residual = ode_residual
        self._adj_state = adj_state
        self._state = state
        self._bkd = bkd_

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return self._residual.nparams()

    def nvars(self) -> int:
        return self._adj_state.shape[0]  # nstates

    def __call__(self, wvec: Array) -> Array:
        """Return λ^T · d²f/dp dy · w."""
        # For quadratic ODE: df/dp_0 = y^2, df/dp_1 = 1
        # d²f/dp_0 dy = 2*diag(y), d²f/dp_1 dy = 0
        # So λ^T · d²f/dp_0 dy · w = 2 * sum(λ * y * w)
        # And λ^T · d²f/dp_1 dy · w = 0
        wvec_flat = wvec.flatten()
        adj_flat = self._adj_state.flatten()
        state_flat = self._state.flatten()

        result = self._bkd.zeros((2, 1))
        result = self._bkd.copy(result)
        result[0, 0] = 2.0 * float(self._bkd.sum(adj_flat * state_flat * wvec_flat))
        result[1, 0] = 0.0
        return result.T  # (1, nparams)

    def jacobian(self, wvec: Array) -> Array:
        """Return d/dw [λ^T · d²f/dp dy · w]."""
        # d/dw_j [2 * sum_i(λ_i * y_i * w_i)] = 2 * λ_j * y_j
        adj_flat = self._adj_state.flatten()
        state_flat = self._state.flatten()

        # Result shape: (nparams, nstates)
        result = self._bkd.zeros((2, len(state_flat)))
        result = self._bkd.copy(result)
        result[0, :] = 2.0 * adj_flat * state_flat
        result[1, :] = 0.0
        return result


# =============================================================================
# Test Classes
# =============================================================================


class TestODEResidualHVP(Generic[Array], unittest.TestCase):
    """Test ODE residual HVP methods."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_quadratic_ode_state_state_hvp(self) -> None:
        """Test state_state_hvp using DerivativeChecker."""
        bkd = self.bkd()
        np.random.seed(42)

        nstates = 2
        Amat = bkd.asarray(np.array([[-1.0, 0.1], [0.1, -2.0]]))
        ode_residual = QuadraticODEResidual(Amat, bkd)

        # Set parameters
        param = bkd.asarray(np.array([[0.1], [0.5]]))
        ode_residual.set_param(param.flatten())

        # Create test state and adjoint
        state = bkd.asarray(np.array([0.5, 0.3]))
        adj_state = bkd.asarray(np.array([1.2, -0.8]))

        # Wrap for derivative checking
        wrapper = ODEResidualHVPWrapper(ode_residual, adj_state, bkd)

        # Check derivatives
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nstates, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            state.reshape(-1, 1),
            direction=direction,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        jac_ratio = float(checker.error_ratio(errors[0]))
        self.assertLess(
            jac_ratio,
            1e-6,
            f"ODE state_state_hvp Jacobian error ratio {jac_ratio:.2e} exceeds 1e-6",
        )

    def test_quadratic_ode_param_state_hvp(self) -> None:
        """Test param_state_hvp using DerivativeChecker."""
        bkd = self.bkd()
        np.random.seed(42)

        nstates = 2
        Amat = bkd.asarray(np.array([[-1.0, 0.1], [0.1, -2.0]]))
        ode_residual = QuadraticODEResidual(Amat, bkd)

        # Set parameters
        param = bkd.asarray(np.array([[0.1], [0.5]]))
        ode_residual.set_param(param.flatten())

        # Create test state and adjoint
        state = bkd.asarray(np.array([0.5, 0.3]))
        adj_state = bkd.asarray(np.array([1.2, -0.8]))

        # Wrap for derivative checking
        wrapper = ODEResidualParamHVPWrapper(ode_residual, adj_state, state, bkd)

        # Check derivatives
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nstates, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            bkd.asarray(np.random.randn(nstates, 1)),
            direction=direction,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        jac_ratio = float(checker.error_ratio(errors[0]))
        self.assertLess(
            jac_ratio,
            1e-6,
            f"ODE param_state_hvp Jacobian error ratio {jac_ratio:.2e} exceeds 1e-6",
        )


class TestAnalyticalSolutions(Generic[Array], unittest.TestCase):
    """Test numerical solutions against sympy analytical solutions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_backward_euler_forward_solve(self) -> None:
        """Test Backward Euler forward solve matches analytical solution."""
        bkd = self.bkd()
        from pyapprox.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )

        # Setup
        nstates = 1  # Scalar ODE for easier verification
        a = -0.5
        Amat = bkd.asarray(np.array([[a]]))
        ode_residual = QuadraticODEResidual(Amat, bkd)

        time_residual = BackwardEulerResidual(ode_residual)
        newton_solver = NewtonSolver(time_residual)

        init_time = 0.0
        final_time = 0.3
        deltat = 0.1
        integrator = TimeIntegrator(init_time, final_time, deltat, newton_solver)

        # Solve numerically
        param = bkd.asarray(np.array([[0.1], [0.2]]))
        ode_residual.set_param(param.flatten())
        init_state = bkd.asarray(np.array([1.0]))

        fwd_sols_num, times = integrator.solve(init_state)

        # Solve analytically with sympy formulas
        p = [0.1, 0.2]
        fwd_sols_ana = sympy_solve_forward(
            1.0, p, deltat, 3, a, method="backward_euler"
        )

        # Compare
        for n in range(len(times)):
            error = abs(float(fwd_sols_num[0, n]) - fwd_sols_ana[n])
            self.assertLess(
                error,
                1e-10,
                f"Backward Euler forward solve error at t={float(times[n]):.2f}: {error:.2e}",
            )

    def test_backward_euler_adjoint_solve(self) -> None:
        """Test Backward Euler adjoint solve matches analytical solution."""
        bkd = self.bkd()
        from pyapprox.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )
        from pyapprox.pde.time.functionals.endpoint import EndpointFunctional
        from pyapprox.pde.time.operator.time_adjoint_hvp import (
            TimeAdjointOperatorWithHVP,
        )

        # Setup
        nstates = 1
        a = -0.5
        Amat = bkd.asarray(np.array([[a]]))
        ode_residual = QuadraticODEResidual(Amat, bkd)
        nparams = ode_residual.nparams()

        time_residual = BackwardEulerResidual(ode_residual)
        newton_solver = NewtonSolver(time_residual)

        init_time = 0.0
        final_time = 0.3
        deltat = 0.1
        integrator = TimeIntegrator(init_time, final_time, deltat, newton_solver)

        # Endpoint functional: Q = y(T)
        functional = EndpointFunctional(
            state_idx=0, nstates=nstates, nparams=nparams, bkd=bkd
        )

        # Solve
        param = bkd.asarray(np.array([[0.1], [0.2]]))
        ode_residual.set_param(param.flatten())
        init_state = bkd.asarray(np.array([1.0]))

        # Forward solve
        fwd_sols_num, times = integrator.solve(init_state)
        integrator.set_functional(functional)

        # Adjoint solve
        adj_sols_num = integrator.solve_adjoint(fwd_sols_num, times, param)

        # Analytical adjoint
        p = [0.1, 0.2]
        fwd_sols_ana = fwd_sols_num[0, :].flatten()
        dQdy_final = 1.0  # dQ/dy = 1 for endpoint functional Q = y(T)
        adj_sols_ana = sympy_adjoint_backward_euler(fwd_sols_ana, deltat, p, a, dQdy_final)

        # Compare
        for n in range(len(times)):
            error = abs(float(adj_sols_num[0, n]) - adj_sols_ana[n])
            self.assertLess(
                error,
                1e-10,
                f"Backward Euler adjoint solve error at t={float(times[n]):.2f}: {error:.2e}",
            )

    def test_forward_euler_forward_solve(self) -> None:
        """Test Forward Euler forward solve matches analytical solution."""
        bkd = self.bkd()
        from pyapprox.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )

        # Setup
        nstates = 1
        a = -0.5
        Amat = bkd.asarray(np.array([[a]]))
        ode_residual = QuadraticODEResidual(Amat, bkd)

        time_residual = ForwardEulerResidual(ode_residual)
        newton_solver = NewtonSolver(time_residual)

        init_time = 0.0
        final_time = 0.3
        deltat = 0.1
        integrator = TimeIntegrator(init_time, final_time, deltat, newton_solver)

        # Solve numerically
        param = bkd.asarray(np.array([[0.1], [0.2]]))
        ode_residual.set_param(param.flatten())
        init_state = bkd.asarray(np.array([1.0]))

        fwd_sols_num, times = integrator.solve(init_state)

        # Solve analytically
        p = [0.1, 0.2]
        fwd_sols_ana = sympy_solve_forward(
            1.0, p, deltat, 3, a, method="forward_euler"
        )

        # Compare
        for n in range(len(times)):
            error = abs(float(fwd_sols_num[0, n]) - fwd_sols_ana[n])
            self.assertLess(
                error,
                1e-10,
                f"Forward Euler forward solve error at t={float(times[n]):.2f}: {error:.2e}",
            )


# =============================================================================
# Wrapper for Time Stepping Residual HVP Testing
# =============================================================================


class TimeResidualStateHVPWrapper(Generic[Array]):
    """
    Wrapper to test time stepping residual state_state_hvp method.

    Tests: λ^T · (dR/dy_n)
    The HVP is: λ^T · (d²R/dy_n²) · w
    """

    def __init__(
        self,
        time_residual,
        adj_state: Array,
        fsol_nm1: Array,
        bkd_: Backend[Array],
    ):
        self._time_residual = time_residual
        self._adj_state = adj_state
        self._fsol_nm1 = fsol_nm1
        self._bkd = bkd_

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return self._adj_state.shape[0]

    def nvars(self) -> int:
        return self._adj_state.shape[0]

    def __call__(self, state: Array) -> Array:
        """Return λ^T · (dR/dy_n) as a row vector."""
        jac = self._time_residual.jacobian(state.flatten())
        return (jac.T @ self._adj_state.reshape(-1, 1)).reshape(1, -1)

    def jacobian(self, state: Array) -> Array:
        """Return d/dy_n [λ^T · (dR/dy_n)]."""
        # For Backward Euler: dR/dy_n = M - dt*J_f
        # d²R/dy_n² = -dt * d²f/dy²
        # λ^T · d²R/dy_n² is what we need
        fsol_n = state.flatten()
        result = self._bkd.zeros((self.nqoi(), self.nvars()))
        result = self._bkd.copy(result)

        # Use HVP to compute each column
        for j in range(self.nvars()):
            ej = self._bkd.zeros((self.nvars(),))
            ej = self._bkd.copy(ej)
            ej[j] = 1.0
            hvp_col = self._time_residual.state_state_hvp(
                self._fsol_nm1, fsol_n, self._adj_state.flatten(), ej
            )
            result[:, j] = hvp_col
        return result


class TimeResidualParamHVPWrapper(Generic[Array]):
    """
    Wrapper to test time stepping residual param_state_hvp method.

    Tests: λ^T · (dR/dp) evaluated at various states
    The derivative w.r.t. state gives: λ^T · (d²R/dp dy_n)
    """

    def __init__(
        self,
        time_residual,
        adj_state: Array,
        fsol_nm1: Array,
        fsol_n: Array,
        bkd_: Backend[Array],
    ):
        self._time_residual = time_residual
        self._adj_state = adj_state
        self._fsol_nm1 = fsol_nm1
        self._fsol_n = fsol_n
        self._bkd = bkd_

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return self._time_residual._residual.nparams()

    def nvars(self) -> int:
        return self._adj_state.shape[0]

    def __call__(self, wvec: Array) -> Array:
        """Return λ^T · (d²R/dp dy_n) · w."""
        wvec_flat = wvec.flatten()
        hvp = self._time_residual.param_state_hvp(
            self._fsol_nm1, self._fsol_n, self._adj_state.flatten(), wvec_flat
        )
        return hvp.reshape(1, -1)

    def jacobian(self, wvec: Array) -> Array:
        """Return d/dw [λ^T · (d²R/dp dy_n) · w]."""
        nparams = self.nqoi()
        nstates = self.nvars()
        result = self._bkd.zeros((nparams, nstates))
        result = self._bkd.copy(result)

        for j in range(nstates):
            ej = self._bkd.zeros((nstates,))
            ej = self._bkd.copy(ej)
            ej[j] = 1.0
            hvp_col = self._time_residual.param_state_hvp(
                self._fsol_nm1, self._fsol_n, self._adj_state.flatten(), ej
            )
            result[:, j] = hvp_col
        return result


class TestTimeResidualHVP(Generic[Array], unittest.TestCase):
    """Test time stepping residual HVP methods."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_backward_euler_state_state_hvp(self) -> None:
        """Test BackwardEuler state_state_hvp using DerivativeChecker."""
        bkd = self.bkd()
        np.random.seed(42)

        # Setup
        nstates = 2
        a_diag = np.array([-0.5, -0.3])
        Amat = bkd.asarray(np.diag(a_diag))
        ode_residual = QuadraticODEResidual(Amat, bkd)

        param = bkd.asarray(np.array([[0.1], [0.2]]))
        ode_residual.set_param(param.flatten())

        time_residual = BackwardEulerResidual(ode_residual)

        # Set up the time step
        deltat = 0.1
        fsol_nm1 = bkd.asarray(np.array([1.0, 0.8]))
        time_residual.set_time(0.0, deltat, fsol_nm1)

        # Create test state and adjoint
        fsol_n = bkd.asarray(np.array([0.95, 0.75]))
        adj_state = bkd.asarray(np.array([1.2, -0.8]))

        # Wrap for derivative checking
        wrapper = TimeResidualStateHVPWrapper(
            time_residual, adj_state, fsol_nm1, bkd
        )

        # Check derivatives
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nstates, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            fsol_n.reshape(-1, 1),
            direction=direction,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        jac_ratio = float(checker.error_ratio(errors[0]))
        self.assertLess(
            jac_ratio,
            1e-6,
            f"BackwardEuler state_state_hvp error ratio {jac_ratio:.2e} exceeds 1e-6",
        )

    def test_backward_euler_param_state_hvp(self) -> None:
        """Test BackwardEuler param_state_hvp using DerivativeChecker."""
        bkd = self.bkd()
        np.random.seed(42)

        # Setup
        nstates = 2
        a_diag = np.array([-0.5, -0.3])
        Amat = bkd.asarray(np.diag(a_diag))
        ode_residual = QuadraticODEResidual(Amat, bkd)

        param = bkd.asarray(np.array([[0.1], [0.2]]))
        ode_residual.set_param(param.flatten())

        time_residual = BackwardEulerResidual(ode_residual)

        # Set up the time step
        deltat = 0.1
        fsol_nm1 = bkd.asarray(np.array([1.0, 0.8]))
        time_residual.set_time(0.0, deltat, fsol_nm1)

        # Create test state and adjoint
        fsol_n = bkd.asarray(np.array([0.95, 0.75]))
        adj_state = bkd.asarray(np.array([1.2, -0.8]))

        # Wrap for derivative checking
        wrapper = TimeResidualParamHVPWrapper(
            time_residual, adj_state, fsol_nm1, fsol_n, bkd
        )

        # Check derivatives
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nstates, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            bkd.asarray(np.random.randn(nstates, 1)),
            direction=direction,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        jac_ratio = float(checker.error_ratio(errors[0]))
        self.assertLess(
            jac_ratio,
            1e-6,
            f"BackwardEuler param_state_hvp error ratio {jac_ratio:.2e} exceeds 1e-6",
        )

    def test_forward_euler_state_state_hvp(self) -> None:
        """Test ForwardEuler state_state_hvp using DerivativeChecker.

        For Forward Euler:
            R_n(y_n, y_{n-1}) = y_n - y_{n-1} - Δt·f(y_{n-1})

        The HVP method computes d²R/dy_{n-1}² (not d²R/dy_n² which is 0).
        So we need to test the derivative w.r.t. y_{n-1}.
        """
        bkd = self.bkd()
        np.random.seed(42)

        # Setup
        nstates = 2
        a_diag = np.array([-0.5, -0.3])
        Amat = bkd.asarray(np.diag(a_diag))
        ode_residual = QuadraticODEResidual(Amat, bkd)

        param = bkd.asarray(np.array([[0.1], [0.2]]))
        ode_residual.set_param(param.flatten())

        time_residual = ForwardEulerResidual(ode_residual)

        # Set up the time step
        deltat = 0.1
        fsol_nm1 = bkd.asarray(np.array([1.0, 0.8]))
        fsol_n = bkd.asarray(np.array([0.95, 0.75]))  # Not used in this test
        adj_state = bkd.asarray(np.array([1.2, -0.8]))

        # Create a wrapper that varies y_{n-1} and computes λ^T·(dR/dy_{n-1})
        class ForwardEulerPrevStateWrapper:
            def __init__(self, time_residual, adj_state, deltat, bkd_):
                self._time_residual = time_residual
                self._adj_state = adj_state
                self._deltat = deltat
                self._bkd = bkd_
                self._fsol_n = fsol_n

            def bkd(self):
                return self._bkd

            def nqoi(self):
                return len(self._adj_state)

            def nvars(self):
                return len(self._adj_state)

            def __call__(self, y_nm1):
                # dR/dy_{n-1} = -M - dt * df/dy_{n-1}
                y_nm1_flat = y_nm1.flatten()
                self._time_residual.set_time(0.0, self._deltat, y_nm1_flat)
                # The off-diagonal jacobian gives dR_{n+1}/dy_n, which for FE is
                # -(M + dt*J). So dR/dy_{n-1} is similar.
                jac = (
                    -self._time_residual._residual.mass_matrix(len(y_nm1_flat))
                    - self._deltat * self._time_residual._residual.jacobian(y_nm1_flat)
                )
                return (jac.T @ self._adj_state.reshape(-1, 1)).reshape(1, -1)

            def jacobian(self, y_nm1):
                # d/dy_{n-1} [λ^T · dR/dy_{n-1}] = λ^T · d²R/dy_{n-1}²
                y_nm1_flat = y_nm1.flatten()
                self._time_residual.set_time(0.0, self._deltat, y_nm1_flat)
                result = self._bkd.zeros((self.nqoi(), self.nvars()))
                result = self._bkd.copy(result)
                for j in range(self.nvars()):
                    ej = self._bkd.zeros((self.nvars(),))
                    ej = self._bkd.copy(ej)
                    ej[j] = 1.0
                    hvp_col = self._time_residual.state_state_hvp(
                        y_nm1_flat, self._fsol_n, self._adj_state.flatten(), ej
                    )
                    result[:, j] = hvp_col
                return result

        wrapper = ForwardEulerPrevStateWrapper(time_residual, adj_state, deltat, bkd)

        # Check derivatives
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nstates, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            fsol_nm1.reshape(-1, 1),
            direction=direction,
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        jac_ratio = float(checker.error_ratio(errors[0]))
        self.assertLess(
            jac_ratio,
            1e-6,
            f"ForwardEuler state_state_hvp error ratio {jac_ratio:.2e} exceeds 1e-6",
        )


class TestODEResidualHVPNumpy(TestODEResidualHVP):
    """Test ODE residual HVP with NumPy backend."""

    __test__ = True

    def bkd(self) -> Backend:
        return NumpyBkd


class TestAnalyticalSolutionsNumpy(TestAnalyticalSolutions):
    """Test analytical solutions with NumPy backend."""

    __test__ = True

    def bkd(self) -> Backend:
        return NumpyBkd


class TestTimeResidualHVPNumpy(TestTimeResidualHVP):
    """Test time stepping residual HVP with NumPy backend."""

    __test__ = True

    def bkd(self) -> Backend:
        return NumpyBkd


if __name__ == "__main__":
    unittest.main()
