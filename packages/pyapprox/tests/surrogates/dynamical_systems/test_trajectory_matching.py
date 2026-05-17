"""Tests for TrajectoryMatchingLoss."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerAdjoint
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)
from pyapprox.util.rootfinding.newton import NewtonSolver


def _vdp_rhs(states, mu, bkd):
    x1 = states[0:1, :]
    x2 = states[1:2, :]
    return bkd.vstack([x2, mu * (1 - x1**2) * x2 - x1])


def _build_fitted_expansion(bkd, nvars, nqoi, max_level, train_mu_values):
    marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    expansion = BasisExpansion(basis, bkd, nqoi=nqoi)

    rng = np.random.RandomState(42)
    n_per_mu = 200
    all_states, all_derivs = [], []
    for mu in train_mu_values:
        s2d = bkd.array(rng.uniform(-2, 2, (2, n_per_mu)))
        mu_row = bkd.full((1, n_per_mu), mu)
        all_states.append(bkd.vstack([s2d, mu_row]))
        all_derivs.append(_vdp_rhs(s2d, mu=mu, bkd=bkd))
    train_states = bkd.hstack(all_states)
    train_derivs = bkd.hstack(all_derivs)

    fitter = LeastSquaresFitter(bkd)
    return fitter.fit(expansion, train_states, train_derivs).surrogate()


def _build_loss(wrapper, init_states, final_time, deltat, bkd, obs_perturbation=0.01):
    stepper = BackwardEulerAdjoint(wrapper)
    newton = NewtonSolver(stepper)
    newton.set_options(maxiters=30, atol=1e-10, rtol=1e-10)
    integrator = TimeIntegrator(0.0, final_time, deltat, newton)

    fwd_sols, times = integrator.solve(init_states)
    nstates = fwd_sols.shape[0]
    ntimes = fwd_sols.shape[1]
    obs_time_indices = bkd.arange(ntimes)
    obs_tuples = [(i, obs_time_indices) for i in range(nstates)]

    flat_obs = bkd.reshape(fwd_sols, (nstates * ntimes,))
    if obs_perturbation > 0:
        rng = np.random.RandomState(123)
        flat_obs = flat_obs + bkd.array(rng.randn(nstates * ntimes) * obs_perturbation)

    return TrajectoryMatchingLoss(
        wrapper=wrapper, integrator=integrator,
        init_states=init_states, obs_tuples=obs_tuples,
        observations=flat_obs, noise_std=1.0,
    )


class TestTrajectoryMatchingLoss:
    def test_loss_near_zero_at_fitted_params(self, bkd):
        fitted = _build_fitted_expansion(bkd, 3, 2, 3, [1.0])
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_loss(wrapper, init_state, 0.3, 0.01, bkd, obs_perturbation=0.0)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))
        val = float(loss(sample)[0, 0])
        assert val < 1e-20, f"loss at fitted params should be ~0, got {val:.2e}"

    def test_loss_increases_at_perturbed_params(self, bkd):
        fitted = _build_fitted_expansion(bkd, 3, 2, 3, [1.0])
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_loss(wrapper, init_state, 0.3, 0.01, bkd, obs_perturbation=0.0)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))
        val_opt = float(loss(sample)[0, 0])

        rng = np.random.RandomState(7)
        perturbed = eta + bkd.array(rng.randn(eta.shape[0]) * 0.01)
        val_perturbed = float(loss(bkd.reshape(perturbed, (perturbed.shape[0], 1)))[0, 0])
        assert val_perturbed > val_opt

    @pytest.mark.slow_on("TorchBkd")
    def test_gradient_via_derivative_checker(self, bkd):
        fitted = _build_fitted_expansion(bkd, 3, 2, 3, [1.0])
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_loss(wrapper, init_state, 0.3, 0.01, bkd)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 1e-6, f"error_ratio={float(ratio):.2e}"

    @pytest.mark.slow_on("TorchBkd")
    def test_batched_gradient_equals_sum_of_singles(self, bkd):
        fitted = _build_fitted_expansion(bkd, 3, 2, 3, [0.5, 1.0, 1.5])

        mu_vals = [0.5, 1.0, 1.5]
        k = len(mu_vals)
        n_dynamic = 2

        # Generate target observations from a slightly perturbed expansion
        eta_true = fitted.hyp_list().get_active_values()
        rng = np.random.RandomState(99)
        eta_target = eta_true + bkd.array(rng.randn(eta_true.shape[0]) * 0.02)

        # Build batched loss with deterministic observations
        mu_batch = bkd.array([mu_vals])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=n_dynamic, mu_batch=mu_batch)
        init_flat = bkd.array([0.5, 0.3] * k)

        # Generate obs at perturbed params
        wrapper.set_param(eta_target)
        stepper_b = BackwardEulerAdjoint(wrapper)
        newton_b = NewtonSolver(stepper_b)
        newton_b.set_options(maxiters=30, atol=1e-10, rtol=1e-10)
        integrator_b = TimeIntegrator(0.0, 0.2, 0.01, newton_b)
        fwd_target, times_b = integrator_b.solve(init_flat)
        nstates_b = fwd_target.shape[0]
        ntimes_b = fwd_target.shape[1]
        obs_time_indices_b = bkd.arange(ntimes_b)
        obs_tuples_b = [(i, obs_time_indices_b) for i in range(nstates_b)]
        flat_obs_b = bkd.reshape(fwd_target, (nstates_b * ntimes_b,))

        # Restore and build batched loss
        wrapper.set_param(eta_true)
        loss_batched = TrajectoryMatchingLoss(
            wrapper=wrapper, integrator=integrator_b,
            init_states=init_flat, obs_tuples=obs_tuples_b,
            observations=flat_obs_b, noise_std=1.0,
        )

        sample = bkd.reshape(eta_true, (eta_true.shape[0], 1))
        grad_batched = loss_batched.jacobian(sample)

        # Sum of single-trajectory gradients with matching observations
        grad_sum = bkd.zeros((1, eta_true.shape[0]))
        for i, mu_val in enumerate(mu_vals):
            single_mu = bkd.array([[mu_val]])
            single_wrapper = BatchedBoundODEResidual(
                fitted, n_dynamic=n_dynamic, mu_batch=single_mu
            )
            single_init = bkd.array([0.5, 0.3])

            # Extract this trajectory's observations from the batched target
            single_obs = bkd.reshape(
                fwd_target[i * n_dynamic:(i + 1) * n_dynamic, :],
                (n_dynamic * ntimes_b,)
            )
            stepper_s = BackwardEulerAdjoint(single_wrapper)
            newton_s = NewtonSolver(stepper_s)
            newton_s.set_options(maxiters=30, atol=1e-10, rtol=1e-10)
            integrator_s = TimeIntegrator(0.0, 0.2, 0.01, newton_s)
            obs_tuples_s = [(j, obs_time_indices_b) for j in range(n_dynamic)]

            single_loss = TrajectoryMatchingLoss(
                wrapper=single_wrapper, integrator=integrator_s,
                init_states=single_init, obs_tuples=obs_tuples_s,
                observations=single_obs, noise_std=1.0,
            )
            grad_sum = grad_sum + single_loss.jacobian(sample)

        bkd.assert_allclose(grad_batched, grad_sum, rtol=1e-8)

    @pytest.mark.slow_on("*")
    def test_optimizer_recovers_params(self, numpy_bkd):
        bkd = numpy_bkd
        fitted = _build_fitted_expansion(bkd, 3, 2, 4, [0.5, 1.0, 1.5, 2.0])
        true_eta = bkd.to_numpy(fitted.hyp_list().get_active_values()).copy()

        mu_batch = bkd.array([[0.8, 1.2]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_flat = bkd.array([0.5, 0.3, 0.5, 0.3])
        loss = _build_loss(wrapper, init_flat, 0.3, 0.01, bkd, obs_perturbation=0.0)

        # Perturb 5% and optimize
        rng = np.random.RandomState(0)
        perturbed_eta = true_eta * (1.0 + 0.05 * rng.randn(true_eta.shape[0]))

        from scipy.optimize import minimize

        def scipy_loss_and_grad(x):
            x_arr = bkd.array(x)
            sample = bkd.reshape(x_arr, (x_arr.shape[0], 1))
            l = float(loss(sample)[0, 0])
            g = bkd.to_numpy(loss.jacobian(sample)[0, :])
            return l, g

        result = minimize(
            scipy_loss_and_grad, perturbed_eta,
            jac=True, method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-10},
        )

        recovered = result.x
        rel_err = np.linalg.norm(recovered - true_eta) / np.linalg.norm(true_eta)
        assert rel_err < 0.05, f"relative parameter error {rel_err:.4f} exceeds 0.05"
