"""
Tests for LogNormalDataMeanQoIAVaRStdDevSAAObjective.

Validation triangle:
1. degree-1 values: converges to the piecewise-Gaussian closed form as
   the outer rule refines (the two implementations share no E_y code);
2. gradients: DerivativeChecker on torch validates that the detached
   per-node rankings preserve the exact envelope gradient;
3. inner loop: with SHARED outer data the gap to the double-loop
   pipeline estimator isolates inner-loop error, which must decay at
   the MC rate, plus the O(1/delta) smoothing floor ordering.
"""

import numpy as np

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    LogNormalDataMeanQoIAVaRStdDevSAAObjective,
    ReparameterizedOuterData,
)
from pyapprox.expdesign.objective import create_prediction_oed_objective
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.markers import slow_test


def _build_problem(bkd, nobs=5, npred=6, degree=1):
    nparams = degree + 1
    obs_x = np.linspace(-1, 1, nobs)
    obs_mat = bkd.asarray(
        np.column_stack([obs_x**d for d in range(nparams)])
    )
    pred_x = np.linspace(-2 / 3, 2 / 3, npred)
    qoi_mat = bkd.asarray(
        np.column_stack([pred_x**d for d in range(nparams)])
    )
    prior_mean = bkd.zeros((nparams, 1))
    prior_cov = bkd.asarray(0.25 * np.eye(nparams))
    noise_variances = bkd.asarray(0.25 * np.ones(nobs))
    return obs_mat, prior_mean, prior_cov, qoi_mat, noise_variances


def _saa_objective(bkd, alpha, nouter, seed, degree=1, nobs=5, npred=6):
    obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = _build_problem(
        bkd, nobs=nobs, npred=npred, degree=degree
    )
    outer_data = ReparameterizedOuterData.from_prior_draws(
        obs_mat, prior_mean, prior_cov, noise_var, nouter, seed, bkd
    )
    return LogNormalDataMeanQoIAVaRStdDevSAAObjective(
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, alpha,
        outer_data, bkd,
    )


class TestLogNormalDataMeanQoIAVaRStdDevSAAObjective:

    def test_matches_closed_form_degree1(self, bkd):
        """Outer refinement converges to the piecewise-Gaussian closed
        form (fractional alpha so the tail rule is exercised)."""
        alpha = 0.45
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        weights = bkd.full((nobs, 1), 1.0 / nobs)

        exact_obj = ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
            prior_mean, prior_cov, qoi_mat, alpha, bkd
        )
        exact_obj.set_observation_matrix(obs_mat)
        exact_obj.set_noise_covariance(
            bkd.diag(noise_var / weights[:, 0])
        )
        exact = exact_obj.value()

        errors = []
        for nouter in [2000, 32_000]:
            obj = _saa_objective(bkd, alpha, nouter, seed=3)
            errors.append(abs(obj.value(weights) - exact) / exact)
        assert errors[-1] < errors[0], errors
        assert errors[-1] < 1e-2, errors

    def test_derivatives_pass_derivative_checker(self, torch_bkd):
        """The detached-ranking envelope gradient is exact: validate the
        autograd bundle against finite differences."""
        bkd = torch_bkd
        obj = _saa_objective(bkd, alpha=0.45, nouter=200, seed=5)
        nobs = obj.nvars()
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        checker = DerivativeChecker(obj)
        # steps capped so perturbed weights stay strictly positive
        fd_eps = bkd.flip(bkd.logspace(-12, -2, 11))
        errors = checker.check_derivatives(
            weights, fd_eps=fd_eps, relative=True, verbosity=0
        )
        min_error = float(bkd.min(errors[0]))
        assert min_error < 1e-6, min_error

    @slow_test
    def test_inner_loop_mc_rate_and_smoothing_floor(self, bkd):
        """Shared outer data isolates the pipeline's inner-loop error:
        its RMS over inner replicates must decay at the MC rate, and at
        fixed inner size the smoothed-AVaR bias must shrink as delta
        grows (the O(1/delta) floor)."""
        if not isinstance(bkd, NumpyBkd):
            import pytest

            pytest.skip("pipeline objective is exercised on numpy")

        alpha, nobs, npred, nouter = 0.45, 5, 10, 500
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd, nobs=nobs, npred=npred)
        )
        weights = bkd.full((nobs, 1), 1.0 / nobs)

        rng = np.random.default_rng(17)
        prior_chol = np.linalg.cholesky(bkd.to_numpy(prior_cov))
        theta_out = prior_chol @ rng.standard_normal((2, nouter))
        outer_shapes = bkd.to_numpy(obs_mat) @ theta_out
        latent = rng.standard_normal((nobs, nouter))

        outer_data = ReparameterizedOuterData(
            bkd.asarray(outer_shapes), bkd.asarray(latent), noise_var,
            bkd,
        )
        saa = LogNormalDataMeanQoIAVaRStdDevSAAObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, alpha,
            outer_data, bkd,
        )
        u_ref = saa.value(weights)

        def pipeline_value(ninner, delta, seed):
            inner_rng = np.random.default_rng(seed)
            theta_in = prior_chol @ inner_rng.standard_normal(
                (2, ninner)
            )
            inner_shapes = bkd.to_numpy(obs_mat) @ theta_in
            qoi_vals = np.exp(bkd.to_numpy(qoi_mat) @ theta_in).T
            objective = create_prediction_oed_objective(
                noise_var,
                bkd.asarray(outer_shapes),
                bkd.asarray(inner_shapes),
                bkd.asarray(latent),
                bkd.asarray(qoi_vals),
                bkd,
                deviation_type="stdev",
                risk_type="avar",
                noise_stat_type="mean",
                risk_kwargs={"alpha": alpha, "delta": delta},
            )
            return float(bkd.to_numpy(objective(weights)).flat[0])

        nreps = 8
        ninners = np.array([250, 1000, 4000, 16_000, 64_000])
        rms = []
        for ninner in ninners:
            gaps = [
                pipeline_value(int(ninner), 1e6, seed=100 * r + 7)
                - u_ref
                for r in range(nreps)
            ]
            rms.append(float(np.sqrt(np.mean(np.square(gaps)))))
        rms_arr = np.array(rms)

        assert np.all(np.diff(rms_arr) < 0), rms_arr
        slope = np.polyfit(np.log(ninners), np.log(rms_arr), 1)[0]
        assert -0.55 < slope < -0.45, (slope, rms_arr)

        # Smoothing floor: at fixed inner size the delta=20 bias
        # dominates the delta=1e6 residual
        gap_small_delta = abs(
            pipeline_value(16_000, 20, seed=7) - u_ref
        )
        gap_large_delta = abs(
            pipeline_value(16_000, 1e6, seed=7) - u_ref
        )
        assert gap_small_delta > gap_large_delta, (
            gap_small_delta, gap_large_delta,
        )
