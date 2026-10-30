"""Tests for TrajectoryMatchingLoss and TrajectoryMatchingFitter.

A polynomial expansion fitted to the Van der Pol vector field (nine terms,
the six exact monomials plus three zero-coefficient ones, two outputs
over (x1, x2, mu)) is the learned function, wrapped for batched
integration:

- the loss vanishes at the fitted coefficients and grows away from them;
- its gradient passes ``DerivativeChecker`` (forward Euler, T = 0.3,
  dt 0.01) and, batched over three values of mu, equals the sum of the
  single-trajectory gradients to 1e-8;
- on two trajectories (mu 0.7 and 1.3), the loss value and gradient equal
  the earlier time integration's (its stepper, Newton and adjoint) to
  1e-10, relative, for forward and backward Euler and Crank-Nicolson,
  Newton at atol 1e-13 on both;
- the fitter recovers all coefficients to 1e-9 from noiseless data
  (slow).
"""

from typing import Any, Callable, List

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerAdjoint
from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerAdjoint
from pyapprox.ode.implicit_steppers.crank_nicolson import CrankNicolsonAdjoint
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.protocols import TimeSteppingResidualProtocol
from pyapprox.ode.stepping_methods import RungeKutta, SteppingMethodProtocol
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.fitters import (
    TrajectoryMatchingFitter,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.surrogates.dynamical_systems import solve_batched, step_times
from tests._helpers.surrogates.least_squares import least_squares_fit


def _vdp_rhs(states: Array, mu: Array, bkd: Backend[Array]) -> Array:
    x1 = states[0:1, :]
    x2 = states[1:2, :]
    return bkd.vstack([x2, mu * (1 - x1**2) * x2 - x1])


# VDP exact monomials: (0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,1,1), (2,1,1)
# Plus 3 extra terms to test identifiability of zero-coefficient directions
_VDP_INDICES = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [2, 1, 1],
    [1, 1, 0], [2, 0, 0], [0, 2, 0],
], dtype=np.int64).T


def _build_fitted_expansion(
    bkd: Backend[Array], nvars: int = 3, nqoi: int = 2, nsamples: int = 1000
) -> BasisExpansion[Array]:
    """The expansion fitted to the Van der Pol field by least squares."""
    marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = bkd.asarray(_VDP_INDICES, dtype=int)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    expansion = BasisExpansion(basis, bkd, nqoi=nqoi)

    rng = np.random.RandomState(42)
    train_states = bkd.array(rng.uniform(-3, 3, (nvars, nsamples)))
    x1x2 = train_states[:2, :]
    mu_row = train_states[2:3, :]
    train_derivs = _vdp_rhs(x1x2, mu=mu_row, bkd=bkd)

    return least_squares_fit(expansion, train_states, train_derivs)


def _all_observed(bkd: Backend[Array], trajectory: Array) -> List[tuple[int, Array]]:
    """Every state at every stored time."""
    time_indices = bkd.arange(trajectory.shape[1])
    return [(i, time_indices) for i in range(trajectory.shape[0])]


def _build_loss(
    wrapper: BatchedBoundODEResidual[Array],
    init_states: Array,
    final_time: float,
    deltat: float,
    bkd: Backend[Array],
    obs_perturbation: float = 0.01,
) -> TrajectoryMatchingLoss[Array]:
    method: SteppingMethodProtocol[Array, Array] = RungeKutta(forward_euler())
    times = step_times(0.0, final_time, deltat)
    fwd_sols = solve_batched(wrapper, method, init_states, times)[0]
    nstates, ntimes = fwd_sols.shape
    flat_obs = bkd.reshape(fwd_sols, (nstates * ntimes,))
    if obs_perturbation > 0:
        rng = np.random.RandomState(123)
        flat_obs = flat_obs + bkd.array(rng.randn(nstates * ntimes) * obs_perturbation)

    return TrajectoryMatchingLoss(
        wrapper=wrapper,
        method=method,
        times=times,
        init_states=init_states,
        obs_tuples=_all_observed(bkd, fwd_sols),
        observations=flat_obs,
        noise_std=1.0,
    )


def _column(bkd: Backend[Array], eta: Array) -> Array:
    return bkd.reshape(eta, (eta.shape[0], 1))


class TestTrajectoryMatchingLoss:
    def test_loss_near_zero_at_fitted_params(self, bkd: Backend[Array]) -> None:
        fitted = _build_fitted_expansion(bkd)
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_loss(wrapper, init_state, 0.3, 0.01, bkd, obs_perturbation=0.0)

        sample = _column(bkd, fitted.hyp_list().get_active_values())
        val = bkd.to_float(loss(sample)[0, 0])
        assert val < 1e-20, f"loss at fitted params should be ~0, got {val:.2e}"

    def test_loss_increases_at_perturbed_params(self, bkd: Backend[Array]) -> None:
        fitted = _build_fitted_expansion(bkd)
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_loss(wrapper, init_state, 0.3, 0.01, bkd, obs_perturbation=0.0)

        eta = fitted.hyp_list().get_active_values()
        val_opt = bkd.to_float(loss(_column(bkd, eta))[0, 0])

        rng = np.random.RandomState(7)
        perturbed = eta + bkd.array(rng.randn(eta.shape[0]) * 0.01)
        val_perturbed = bkd.to_float(loss(_column(bkd, perturbed))[0, 0])
        assert val_perturbed > val_opt

    @pytest.mark.slow_on("TorchBkd")
    def test_gradient_via_derivative_checker(self, bkd: Backend[Array]) -> None:
        fitted = _build_fitted_expansion(bkd)
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_loss(wrapper, init_state, 0.3, 0.01, bkd)
        sample = _column(bkd, fitted.hyp_list().get_active_values())

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert bkd.to_float(ratio) <= 2e-6, f"error_ratio={bkd.to_float(ratio):.2e}"

    @pytest.mark.slow_on("TorchBkd")
    def test_batched_gradient_equals_sum_of_singles(
        self, bkd: Backend[Array]
    ) -> None:
        fitted = _build_fitted_expansion(bkd)

        mu_vals = [0.5, 1.0, 1.5]
        k = len(mu_vals)
        n_dynamic = 2
        method: SteppingMethodProtocol[Array, Array] = RungeKutta(forward_euler())
        times = step_times(0.0, 0.2, 0.01)

        # Generate target observations from a slightly perturbed expansion
        eta_true = fitted.hyp_list().get_active_values()
        rng = np.random.RandomState(99)
        eta_target = eta_true + bkd.array(rng.randn(eta_true.shape[0]) * 0.02)

        # Build batched loss with deterministic observations
        mu_batch = bkd.array([mu_vals])
        wrapper = BatchedBoundODEResidual(
            fitted, n_dynamic=n_dynamic, mu_batch=mu_batch
        )
        init_flat = bkd.array([0.5, 0.3] * k)

        # Generate obs at perturbed params
        wrapper.set_param(eta_target)
        fwd_target = solve_batched(wrapper, method, init_flat, times)[0]
        nstates_b, ntimes_b = fwd_target.shape
        flat_obs_b = bkd.reshape(fwd_target, (nstates_b * ntimes_b,))

        # Restore and build batched loss
        wrapper.set_param(eta_true)
        loss_batched = TrajectoryMatchingLoss(
            wrapper=wrapper,
            method=method,
            times=times,
            init_states=init_flat,
            obs_tuples=_all_observed(bkd, fwd_target),
            observations=flat_obs_b,
            noise_std=1.0,
        )

        sample = _column(bkd, eta_true)
        grad_batched = loss_batched.jacobian(sample)

        # Sum of single-trajectory gradients with matching observations
        obs_time_indices = bkd.arange(ntimes_b)
        grad_sum = bkd.zeros((1, eta_true.shape[0]))
        for i, mu_val in enumerate(mu_vals):
            single_wrapper = BatchedBoundODEResidual(
                fitted, n_dynamic=n_dynamic, mu_batch=bkd.array([[mu_val]])
            )
            # Extract this trajectory's observations from the batched target
            single_obs = bkd.reshape(
                fwd_target[i * n_dynamic : (i + 1) * n_dynamic, :],
                (n_dynamic * ntimes_b,),
            )
            single_loss = TrajectoryMatchingLoss(
                wrapper=single_wrapper,
                method=method,
                times=times,
                init_states=bkd.array([0.5, 0.3]),
                obs_tuples=[(j, obs_time_indices) for j in range(n_dynamic)],
                observations=single_obs,
                noise_std=1.0,
            )
            grad_sum = grad_sum + single_loss.jacobian(sample)

        bkd.assert_allclose(grad_batched, grad_sum, rtol=1e-8)


_OLD = [
    (forward_euler, ForwardEulerAdjoint),
    (backward_euler, BackwardEulerAdjoint),
    (crank_nicolson, CrankNicolsonAdjoint),
]


class TestAgainstTheOldPath:
    """The loss and gradient equal the earlier time integration's on two
    trajectories (mu 0.7, 1.3) from (0.5, 0.3) and (-0.4, 0.8), T = 0.2,
    dt 0.02, at perturbed coefficients against noisy observations."""

    @pytest.mark.parametrize(
        "make, old_stepper", _OLD, ids=[make().name for make, _ in _OLD]
    )
    def test_value_and_gradient(
        self,
        bkd: Backend[Array],
        make: Callable[[], ButcherTableau],
        old_stepper: Callable[
            [BatchedBoundODEResidual[Array]], TimeSteppingResidualProtocol[Array]
        ],
    ) -> None:
        fitted = _build_fitted_expansion(bkd)
        wrapper = BatchedBoundODEResidual(
            fitted, n_dynamic=2, mu_batch=bkd.array([[0.7, 1.3]])
        )
        init_flat = bkd.array([0.5, 0.3, -0.4, 0.8])
        final_time, deltat = 0.2, 0.02
        times = step_times(0.0, final_time, deltat)
        # Noisy observations of the fitted model's forward-Euler trajectory.
        euler: SteppingMethodProtocol[Array, Array] = RungeKutta(forward_euler())
        target = solve_batched(wrapper, euler, init_flat, times)[0]
        obs_tuples = _all_observed(bkd, target)
        rng = np.random.RandomState(123)
        observations = bkd.reshape(target, (-1,)) + bkd.array(
            rng.randn(target.shape[0] * target.shape[1]) * 0.01
        )
        newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        tableau = make()
        loss = TrajectoryMatchingLoss(
            wrapper=wrapper,
            method=RungeKutta(tableau, None if tableau.is_explicit() else newton),
            times=times,
            init_states=init_flat,
            obs_tuples=obs_tuples,
            observations=observations,
        )
        eta = fitted.hyp_list().get_active_values()
        sample = _column(bkd, eta + bkd.array(rng.randn(eta.shape[0]) * 0.05))

        # The earlier path: its stepper, Newton, integrator and adjoint.
        old_newton = NewtonSolver(old_stepper(wrapper))
        old_newton.set_options(maxiters=30, atol=1e-13, rtol=0.0)
        integrator = TimeIntegrator(0.0, final_time, deltat, old_newton)
        functional = TransientMSEFunctional(
            nstates=4,
            nresidual_params=wrapper.nparams(),
            obs_tuples=obs_tuples,
            bkd=bkd,
            noise_std=1.0,
        )
        functional.set_observations(observations)
        integrator.set_functional(functional)
        wrapper.set_param(sample[:, 0])
        old_sols, old_times = integrator.solve(init_flat)
        old_value = functional(old_sols, sample)
        old_gradient = integrator.gradient(old_sols, old_times, sample)

        bkd.assert_allclose(loss(sample), old_value, rtol=1e-10)
        bkd.assert_allclose(loss.jacobian(sample), old_gradient, rtol=1e-10)


class TestTrajectoryMatchingFitter:
    @pytest.mark.slow_on("*")
    def test_optimizer_recovers_params(self, numpy_bkd: NumpyBkd) -> None:
        """Recover all 18 params from noiseless trajectory data.

        Identifiability requires diverse ICs and long T so the Hessian has no
        null directions. If this test fails after changing the data config,
        check the Hessian spectrum first — null eigenvalues mean the new config
        does not excite all basis functions.
        """
        bkd = numpy_bkd
        fitted = _build_fitted_expansion(bkd)
        true_eta = bkd.to_numpy(fitted.hyp_list().get_active_values()).copy()

        mu_vals = [0.5, 0.8, 1.0, 1.5, 2.0]
        ic_list = [0.5, 0.3, 1.5, 1.0, 2.5, 0.5, -1.0, 2.0, 0.0, -1.5]
        mu_batch = bkd.array([mu_vals])
        init_flat = bkd.array(ic_list)

        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)
        method: SteppingMethodProtocol[NDArray[Any], NDArray[Any]] = RungeKutta(
            forward_euler()
        )
        times = step_times(0.0, 1.0, 0.1)
        fwd_sols = solve_batched(wrapper, method, init_flat, times)[0]
        nstates, ntimes = fwd_sols.shape
        observations = bkd.reshape(fwd_sols, (nstates * ntimes,))

        # Verify all parameters are identifiable (no null Hessian directions)
        loss = TrajectoryMatchingLoss(
            wrapper=wrapper,
            method=method,
            times=times,
            init_states=init_flat,
            obs_tuples=_all_observed(bkd, fwd_sols),
            observations=observations,
            noise_std=1.0,
        )
        nparams = true_eta.shape[0]
        eps = 1e-5
        H = np.zeros((nparams, nparams))
        for i in range(nparams):
            e_plus = true_eta.copy()
            e_plus[i] += eps
            e_minus = true_eta.copy()
            e_minus[i] -= eps
            g_plus = bkd.to_numpy(
                loss.jacobian(bkd.reshape(bkd.array(e_plus), (nparams, 1)))
            )
            g_minus = bkd.to_numpy(
                loss.jacobian(bkd.reshape(bkd.array(e_minus), (nparams, 1)))
            )
            H[i, :] = (g_plus - g_minus).flatten() / (2 * eps)
        eigvals = np.linalg.eigvalsh(H)
        min_eig = eigvals[0]
        assert min_eig > 1e-3, (
            f"Hessian has null direction (min_eig={min_eig:.2e}); "
            "data config does not identify all parameters"
        )

        # Absolute perturbation (5% of max coef magnitude)
        rng = np.random.RandomState(0)
        scale = max(np.max(np.abs(true_eta)), 1.0)
        perturbed_eta = true_eta + 0.05 * scale * rng.randn(true_eta.shape[0])
        fitted.hyp_list().set_active_values(bkd.array(perturbed_eta))

        fitter = TrajectoryMatchingFitter(wrapper, method, times)
        optimizer: ScipyTrustConstrOptimizer[NDArray[Any]] = ScipyTrustConstrOptimizer(
            maxiter=500, gtol=1e-14
        )
        fitter.set_optimizer(optimizer)
        result = fitter.fit(init_flat, observations)

        recovered = bkd.to_numpy(result.surrogate().hyp_list().get_active_values())
        rel_err = np.linalg.norm(recovered - true_eta) / np.linalg.norm(true_eta)
        assert rel_err < 1e-9, f"relative parameter error {rel_err:.2e} exceeds 1e-9"
