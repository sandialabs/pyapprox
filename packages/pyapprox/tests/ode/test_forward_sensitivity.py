"""FD validation of the transient QoI Jacobian methods.

A parameterized linear ODE ``M dy/dt = -diag(p) y + s`` with identity
and non-identity (dense) mass, integrated by implicit and explicit
steppers. The tangent-linear Jacobian of the final state (``W_T =
dy(T)/dp``) and of a QoI observing several times are
DerivativeChecker-validated against FD of the forward solve, and the
adjoint method must reproduce the tangent-linear one.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.functionals.all_states_endpoint import (
    AllStatesEndpointFunctional,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.mass_matrix import (
    ConstantDenseMassMatrix,
    IdentityMassMatrix,
)
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.ode.operator.forward_sensitivity import (
    forward_sensitivity_jacobian,
)
from pyapprox.ode.operator.qoi_jacobian import (
    adjoint_jacobian,
    default_qoi_jacobian_method,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.util.backends.numpy import NumpyBkd
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
    return integrator, sols, times


def _final_states(bkd):
    return AllStatesEndpointFunctional(_NSTATES, _NSTATES, bkd)


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
            integrator, sols, times = _solve(
                bkd, residual, method, sample[:, 0], y0,
                final_time, deltat,
            )
            return forward_sensitivity_jacobian(
                integrator, _final_states(bkd), sols, times, sample
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
            integrator, sols, times = _solve(
                bkd, residual, "crank_nicolson", sample[:, 0], y0,
                final_time, deltat,
            )
            return forward_sensitivity_jacobian(
                integrator, _final_states(bkd), sols, times, sample
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

    def test_short_trajectory_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        source = bkd.zeros((_NSTATES,))
        residual = _ParamDecayResidual(
            bkd, _make_mass(bkd, "identity"), source
        )
        integrator, _, _ = _solve(
            bkd, residual, "backward_euler", bkd.ones((_NSTATES,)),
            bkd.ones((_NSTATES,)), 0.1, 0.05,
        )
        with pytest.raises(ValueError, match="two time points"):
            forward_sensitivity_jacobian(
                integrator,
                _final_states(bkd),
                bkd.zeros((_NSTATES, 1)),
                bkd.zeros((1,)),
                bkd.ones((_NSTATES, 1)),
            )


class _ObservedTimesFunctional:
    """``Q = vec[P_j y(t_{n_j})] + c q``: linear observations of the
    state at several times, plus one parameter ``q`` of the functional's
    own (ordered first) entering through the fixed vector ``c``."""

    def __init__(self, bkd, observations, shift):
        self._bkd = bkd
        self._observations = observations  # list of (time_idx, P_j)
        self._shift = shift  # c, shape (nqoi, 1)
        self._rows = [
            (time_idx, obs[ii : ii + 1, :])
            for time_idx, obs in observations
            for ii in range(obs.shape[0])
        ]

    def bkd(self):
        return self._bkd

    def nqoi(self):
        return len(self._rows)

    def nstates(self):
        return _NSTATES

    def nparams(self):
        return 1 + _NSTATES

    def nunique_params(self):
        return 1

    def __call__(self, sol, param):
        obs = [p_j @ sol[:, n_j : n_j + 1] for n_j, p_j in self._observations]
        return self._bkd.vstack(obs) + self._shift * param[0, 0]

    def state_jacobian(self, sol, param):
        raise NotImplementedError("vector QoI: use apply_state_jacobian")

    def param_jacobian(self, sol, param):
        bkd = self._bkd
        return bkd.hstack((self._shift, bkd.zeros((self.nqoi(), _NSTATES))))

    def apply_state_jacobian(self, sol, param, time_idx, wmat):
        blocks = [
            p_j @ wmat if n_j == time_idx
            else self._bkd.zeros((p_j.shape[0], wmat.shape[1]))
            for n_j, p_j in self._observations
        ]
        return self._bkd.vstack(blocks)

    def row_functional(self, qoi_idx):
        time_idx, row = self._rows[qoi_idx]
        return _ObservedRowFunctional(
            self._bkd, time_idx, row, self._shift[qoi_idx : qoi_idx + 1, :]
        )


class _ObservedRowFunctional:
    """One row of ``_ObservedTimesFunctional`` as a scalar functional."""

    def __init__(self, bkd, time_idx, row, shift):
        self._bkd = bkd
        self._time_idx = time_idx
        self._row = row
        self._shift = shift

    def bkd(self):
        return self._bkd

    def nqoi(self):
        return 1

    def nstates(self):
        return _NSTATES

    def nparams(self):
        return 1 + _NSTATES

    def nunique_params(self):
        return 1

    def __call__(self, sol, param):
        n = self._time_idx
        return self._row @ sol[:, n : n + 1] + self._shift * param[0, 0]

    def state_jacobian(self, sol, param):
        dqdu = self._bkd.copy(self._bkd.zeros(sol.shape))
        dqdu[:, self._time_idx] = self._row[0, :]
        return dqdu

    def param_jacobian(self, sol, param):
        bkd = self._bkd
        return bkd.hstack((self._shift, bkd.zeros((1, _NSTATES))))


def _observed_times_functional(bkd):
    rng = np.random.default_rng(3)
    return _ObservedTimesFunctional(
        bkd,
        [
            (4, bkd.asarray(rng.standard_normal((2, _NSTATES)))),
            (10, bkd.asarray(rng.standard_normal((1, _NSTATES)))),
        ],
        bkd.asarray(np.array([[1.0], [-2.0], [0.5]])),
    )


class TestQoIJacobianMethods:
    """Both methods on a QoI observing the state at two interior/final
    times with a functional-only parameter (T=0.5, dt=0.05: indices 4 and
    10 are t=0.2 and T)."""

    def _setup(self, bkd, method):
        source = bkd.asarray(np.array([0.5, -0.3, 0.2]))
        residual = _ParamDecayResidual(bkd, _make_mass(bkd, "dense"), source)
        y0 = bkd.asarray(np.array([1.0, -0.5, 0.8]))
        functional = _observed_times_functional(bkd)

        def qoi(samples):
            cols = []
            for ii in range(samples.shape[1]):
                sample = samples[:, ii : ii + 1]
                _, sols, _ = _solve(
                    bkd, residual, method, sample[1:, 0], y0, 0.5, 0.05
                )
                cols.append(functional(sols, sample))
            return bkd.hstack(cols)

        def jacobian_by(jacobian_method):
            def jacobian(sample):
                integrator, sols, times = _solve(
                    bkd, residual, method, sample[1:, 0], y0, 0.5, 0.05
                )
                return jacobian_method(
                    integrator, functional, sols, times, sample
                )
            return jacobian

        sample = bkd.asarray(np.array([[0.3], [0.7], [1.2], [0.4]]))
        return functional, qoi, jacobian_by, sample

    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_tangent_linear_matches_fd(self, bkd, method: str) -> None:
        functional, qoi, jacobian_by, sample = self._setup(bkd, method)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=functional.nqoi(),
            nvars=functional.nparams(),
            fun=qoi,
            jacobian=jacobian_by(forward_sensitivity_jacobian),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            sample, relative=True,
            weights=bkd.ones((functional.nqoi(), 1)),
        )[0]
        # Same one-sided FD floor as the final-time check above.
        assert float(bkd.to_numpy(bkd.min(errors))) <= 1e-7
        assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 5e-6

    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_adjoint_matches_tangent_linear(self, bkd, method: str) -> None:
        _, _, jacobian_by, sample = self._setup(bkd, method)
        bkd.assert_allclose(
            jacobian_by(adjoint_jacobian)(sample),
            jacobian_by(forward_sensitivity_jacobian)(sample),
            rtol=1e-10,
            atol=1e-13,
        )

    def test_unsupported_functionals_raise(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        source = bkd.asarray(np.array([0.5, -0.3, 0.2]))
        residual = _ParamDecayResidual(bkd, _make_mass(bkd, "dense"), source)
        integrator, sols, times = _solve(
            bkd, residual, "backward_euler", bkd.ones((_NSTATES,)),
            bkd.ones((_NSTATES,)), 0.1, 0.05,
        )
        param = bkd.ones((_NSTATES, 1))
        # A scalar functional without a state-Jacobian action.
        scalar = EndpointFunctional(0, _NSTATES, _NSTATES, bkd)
        with pytest.raises(TypeError, match="StateJacobianAction"):
            forward_sensitivity_jacobian(
                integrator, scalar, sols, times, param
            )
        # A vector functional without scalar rows.
        no_rows = _NoRowsFunctional(_final_states(bkd))
        with pytest.raises(TypeError, match="WithRowsProtocol"):
            adjoint_jacobian(integrator, no_rows, sols, times, param)

    def test_default_method(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        scalar = EndpointFunctional(0, _NSTATES, _NSTATES, bkd)
        vector = _final_states(bkd)
        assert default_qoi_jacobian_method(scalar, None) is adjoint_jacobian
        assert (
            default_qoi_jacobian_method(vector, None)
            is forward_sensitivity_jacobian
        )
        assert (
            default_qoi_jacobian_method(vector, adjoint_jacobian)
            is adjoint_jacobian
        )


class _NoRowsFunctional:
    """Vector functional exposing everything but ``row_functional``."""

    def __init__(self, inner):
        self._inner = inner

    def bkd(self):
        return self._inner.bkd()

    def nqoi(self):
        return self._inner.nqoi()

    def nstates(self):
        return self._inner.nstates()

    def nparams(self):
        return self._inner.nparams()

    def nunique_params(self):
        return self._inner.nunique_params()

    def __call__(self, sol, param):
        return self._inner(sol, param)

    def state_jacobian(self, sol, param):
        return self._inner.state_jacobian(sol, param)

    def param_jacobian(self, sol, param):
        return self._inner.param_jacobian(sol, param)
