"""Tests for TrajectoryMatchingLoss."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerAdjoint
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.fitters import (
    TrajectoryMatchingFitter,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.util.rootfinding.newton import NewtonSolver


def _vdp_rhs(states, mu, bkd):
    x1 = states[0:1, :]
    x2 = states[1:2, :]
    return bkd.vstack([x2, mu * (1 - x1**2) * x2 - x1])


# VDP exact monomials: (0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,1,1), (2,1,1)
# Plus 3 extra terms to test identifiability of zero-coefficient directions
_VDP_INDICES = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [2, 1, 1],
    [1, 1, 0], [2, 0, 0], [0, 2, 0],
], dtype=np.int64).T


def _build_fitted_expansion(bkd, nvars=3, nqoi=2, nsamples=1000):
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

    fitter = LeastSquaresFitter(bkd)
    return fitter.fit(expansion, train_states, train_derivs).surrogate()


def _build_loss(wrapper, init_states, final_time, deltat, bkd, obs_perturbation=0.01):
    stepper = ForwardEulerAdjoint(wrapper)
    newton = NewtonSolver(stepper)
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
        fitted = _build_fitted_expansion(bkd)
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_loss(wrapper, init_state, 0.3, 0.01, bkd, obs_perturbation=0.0)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))
        val = float(loss(sample)[0, 0])
        assert val < 1e-20, f"loss at fitted params should be ~0, got {val:.2e}"

    def test_loss_increases_at_perturbed_params(self, bkd):
        fitted = _build_fitted_expansion(bkd)
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
        fitted = _build_fitted_expansion(bkd)
        mu_batch = bkd.array([[1.0]])
        wrapper = BatchedBoundODEResidual(fitted, n_dynamic=2, mu_batch=mu_batch)

        init_state = bkd.array([0.5, 0.3])
        loss = _build_loss(wrapper, init_state, 0.3, 0.01, bkd)

        eta = fitted.hyp_list().get_active_values()
        sample = bkd.reshape(eta, (eta.shape[0], 1))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 2e-6, f"error_ratio={float(ratio):.2e}"

    @pytest.mark.slow_on("TorchBkd")
    def test_batched_gradient_equals_sum_of_singles(self, bkd):
        fitted = _build_fitted_expansion(bkd)

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
        stepper_b = ForwardEulerAdjoint(wrapper)
        newton_b = NewtonSolver(stepper_b)
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
            stepper_s = ForwardEulerAdjoint(single_wrapper)
            newton_s = NewtonSolver(stepper_s)
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
        stepper = ForwardEulerAdjoint(wrapper)
        newton = NewtonSolver(stepper)
        integrator = TimeIntegrator(0.0, 1.0, 0.1, newton)
        fwd_sols, _ = integrator.solve(init_flat)
        nstates = fwd_sols.shape[0]
        ntimes = fwd_sols.shape[1]
        observations = bkd.reshape(fwd_sols, (nstates * ntimes,))

        # Verify all parameters are identifiable (no null Hessian directions)
        obs_time_indices = bkd.arange(ntimes)
        obs_tuples = [(i, obs_time_indices) for i in range(nstates)]
        loss = TrajectoryMatchingLoss(
            wrapper=wrapper, integrator=integrator,
            init_states=init_flat, obs_tuples=obs_tuples,
            observations=observations, noise_std=1.0,
        )
        nparams = true_eta.shape[0]
        eps = 1e-5
        H = np.zeros((nparams, nparams))
        for i in range(nparams):
            e_plus = true_eta.copy(); e_plus[i] += eps
            e_minus = true_eta.copy(); e_minus[i] -= eps
            g_plus = bkd.to_numpy(loss.jacobian(
                bkd.reshape(bkd.array(e_plus), (nparams, 1))))
            g_minus = bkd.to_numpy(loss.jacobian(
                bkd.reshape(bkd.array(e_minus), (nparams, 1))))
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

        fitter = TrajectoryMatchingFitter(wrapper, integrator)
        fitter.set_optimizer(ScipyTrustConstrOptimizer(maxiter=500, gtol=1e-14))
        result = fitter.fit(init_flat, observations)

        recovered = bkd.to_numpy(
            result.surrogate().hyp_list().get_active_values()
        )
        rel_err = np.linalg.norm(recovered - true_eta) / np.linalg.norm(true_eta)
        assert rel_err < 1e-9, f"relative parameter error {rel_err:.2e} exceeds 1e-9"
