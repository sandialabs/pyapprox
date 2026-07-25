"""FD validation of the full-matrix forward-sensitivity solver.

A parameterized linear ODE ``M dy/dt = -diag(p) y + s`` with identity
and non-identity (dense) mass, integrated by implicit and explicit
steppers; the final sensitivity matrix ``W_T = dy(T)/dp`` is
DerivativeChecker-validated against FD of the forward solve.
"""

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.mass_matrix import (
    ConstantDenseMassMatrix,
    IdentityMassMatrix,
)
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.ode.operator.forward_sensitivity import (
    solve_final_forward_sensitivity,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.util.rootfinding.newton import NewtonSolver

_NSTATES = 3


class _ParamDecayResidual(DefaultNewtonJacobianMixin):
    """``M dy/dt = -diag(p) y + s``: linear ODE with per-state decay
    parameters (minimal implicit ODE residual with param jacobian)."""

    def __init__(self, bkd, mass, source):
        self._bkd = bkd
        self._mass = mass
        self._source = source
        self._param = bkd.zeros((_NSTATES,))

    def bkd(self):
        return self._bkd

    def set_time(self, time):
        pass

    def __call__(self, state):
        return -self._param * state + self._source

    def jacobian(self, state):
        return -self._bkd.diag(self._param)

    def mass_matrix(self):
        return self._mass

    def nparams(self):
        return _NSTATES

    def set_param(self, param):
        self._param = self._bkd.flatten(param)

    def param_jacobian(self, state):
        return -self._bkd.diag(state)

    def initial_param_jacobian(self):
        return self._bkd.zeros((_NSTATES, _NSTATES))


def _make_mass(bkd, kind):
    if kind == "identity":
        return IdentityMassMatrix(_NSTATES, bkd)
    mat = bkd.asarray(
        np.eye(_NSTATES) + 0.2 * np.diag(np.ones(_NSTATES - 1), 1)
        + 0.2 * np.diag(np.ones(_NSTATES - 1), -1)
    )
    return ConstantDenseMassMatrix(mat, bkd)


def _solve(bkd, residual, method, param, y0, final_time, deltat):
    residual.set_param(param)
    stepper = create_stepper(method, residual)
    newton = NewtonSolver(stepper)
    newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
    integrator = TimeIntegrator(0.0, final_time, deltat, newton)
    sols, times = integrator.solve(y0)
    return stepper, sols, times


class TestForwardSensitivityMatrix:
    @pytest.mark.parametrize("mass_kind", ["identity", "dense"])
    @pytest.mark.parametrize(
        "method", ["backward_euler", "crank_nicolson", "forward_euler", "heun"]
    )
    def test_final_sensitivity_matches_fd(
        self, bkd, method, mass_kind
    ) -> None:
        source = bkd.asarray(np.array([0.5, -0.3, 0.2]))
        residual = _ParamDecayResidual(
            bkd, _make_mass(bkd, mass_kind), source
        )
        y0 = bkd.asarray(np.array([1.0, -0.5, 0.8]))
        param_np = np.array([0.7, 1.2, 0.4])
        final_time, deltat = 0.5, 0.05

        def y_final_of_params(samples):
            results = []
            for ii in range(samples.shape[1]):
                _, sols, _ = _solve(
                    bkd, residual, method, samples[:, ii], y0,
                    final_time, deltat,
                )
                results.append(bkd.to_numpy(sols[:, -1]).copy())
            return bkd.asarray(np.stack(results, axis=1))

        def sensitivity_of_params(sample):
            stepper, sols, times = _solve(
                bkd, residual, method, sample[:, 0], y0,
                final_time, deltat,
            )
            return solve_final_forward_sensitivity(
                stepper, sols, times, bkd
            )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=_NSTATES,
            nvars=_NSTATES,
            fun=y_final_of_params,
            jacobian=sensitivity_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            bkd.asarray(param_np.reshape(-1, 1)), relative=True
        )[0]
        # One-sided FD of the iterative solve bottoms near 2e-8
        # (V-shaped eps sweeps verified for every method/mass config);
        # the roundoff side puts the ratio astride 1e-6. A genuine
        # sweep bug plateaus at O(1).
        err_min = float(bkd.to_numpy(bkd.min(errors)))
        assert err_min <= 1e-7
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 5e-6

    def test_nonuniform_dt(self, numpy_bkd) -> None:
        """Non-uniform final step: the sweep must rebuild the step
        context from the actual time points."""
        bkd = numpy_bkd
        source = bkd.asarray(np.array([0.5, -0.3, 0.2]))
        residual = _ParamDecayResidual(bkd, _make_mass(bkd, "dense"), source)
        y0 = bkd.asarray(np.array([1.0, -0.5, 0.8]))
        param_np = np.array([0.7, 1.2, 0.4])
        # T=0.35 with dt=0.1 leaves a short 0.05 last step.
        final_time, deltat = 0.35, 0.1

        def y_final_of_params(samples):
            results = []
            for ii in range(samples.shape[1]):
                _, sols, _ = _solve(
                    bkd, residual, "crank_nicolson", samples[:, ii], y0,
                    final_time, deltat,
                )
                results.append(bkd.to_numpy(sols[:, -1]).copy())
            return bkd.asarray(np.stack(results, axis=1))

        def sensitivity_of_params(sample):
            stepper, sols, times = _solve(
                bkd, residual, "crank_nicolson", sample[:, 0], y0,
                final_time, deltat,
            )
            return solve_final_forward_sensitivity(
                stepper, sols, times, bkd
            )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=_NSTATES,
            nvars=_NSTATES,
            fun=y_final_of_params,
            jacobian=sensitivity_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            bkd.asarray(param_np.reshape(-1, 1)), relative=True
        )[0]
        # One-sided FD of the iterative solve bottoms near 2e-8
        # (V-shaped eps sweeps verified for every method/mass config);
        # the roundoff side puts the ratio astride 1e-6. A genuine
        # sweep bug plateaus at O(1).
        err_min = float(bkd.to_numpy(bkd.min(errors)))
        assert err_min <= 1e-7
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 5e-6

    def test_short_trajectory_raises(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        source = bkd.zeros((_NSTATES,))
        residual = _ParamDecayResidual(
            bkd, _make_mass(bkd, "identity"), source
        )
        stepper = create_stepper("backward_euler", residual)
        with pytest.raises(ValueError, match="two time points"):
            solve_final_forward_sensitivity(
                stepper,
                bkd.zeros((_NSTATES, 1)),
                bkd.zeros((1,)),
                bkd,
            )
