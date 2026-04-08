"""
Tests for LogNormalDataMeanQoIAVaRStdDevObjective (differentiable objective).

Verifies protocol satisfaction, shape correctness, finite differences vs
jacobian, autograd compatibility, and consistency with the diagnostics class.
"""

import numpy as np
import pytest

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    LogNormalDataMeanQoIAVaRStdDevObjective,
)
from tests._helpers.markers import slow_test


def _build_problem(bkd, nobs=3, npred=4):
    """Build degree-1 basis with general QoI locations."""
    np.random.seed(42)
    nvars = 2

    obs_mat = bkd.asarray(np.random.randn(nobs, nvars))
    prior_mean = bkd.zeros((nvars, 1))
    prior_cov = bkd.asarray(np.eye(nvars) * 0.5)
    noise_variances = bkd.asarray(np.random.uniform(0.1, 0.5, nobs))

    x_vals = np.linspace(-1.5, 1.5, npred)
    qoi_mat = bkd.asarray(np.column_stack([np.ones(npred), x_vals]))

    return obs_mat, prior_mean, prior_cov, qoi_mat, noise_variances


class TestLogNormalDataMeanQoIAVaRStdDevObjective:

    def test_call_returns_correct_shape(self, bkd):
        """Test __call__ returns (1, 1) array."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        nobs = obs_mat.shape[0]
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        result = obj(weights)
        assert result.shape == (1, 1)

    def test_positive_value(self, bkd):
        """Test objective returns positive value."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        nobs = obs_mat.shape[0]
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        result = obj(weights)
        assert float(bkd.to_numpy(result).flat[0]) > 0.0

    def test_jacobian_returns_correct_shape(self, bkd):
        """Test jacobian returns (1, nobs) array."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        jac = obj.jacobian(weights)
        assert jac.shape == (1, nobs)

    def test_jacobian_finite_values(self, bkd):
        """Test jacobian returns finite values."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        jac = obj.jacobian(weights)
        jac_np = bkd.to_numpy(jac)
        assert np.all(np.isfinite(jac_np))

    def test_nvars_nqoi(self, bkd):
        """Test nvars and nqoi accessors."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        assert obj.nvars() == obs_mat.shape[0]
        assert obj.nqoi() == 1

    def test_matches_diagnostics_class(self, bkd):
        """Test differentiable objective matches non-differentiable version."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        alpha = 0.5
        weights = bkd.full((nobs, 1), 1.0 / nobs)

        # Differentiable objective
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, alpha, bkd
        )
        obj_val = float(bkd.to_numpy(obj(weights)).flat[0])

        # Non-differentiable diagnostics class
        noise_cov = bkd.diag(noise_var / weights[:, 0])
        diag_utility = ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
            prior_mean, prior_cov, qoi_mat, alpha, bkd
        )
        diag_utility.set_observation_matrix(obs_mat)
        diag_utility.set_noise_covariance(noise_cov)
        diag_val = diag_utility.value()

        bkd.assert_allclose(
            bkd.asarray([obj_val]),
            bkd.asarray([diag_val]),
            rtol=1e-8,
        )

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75])
    def test_alpha_values(self, bkd, alpha):
        """Test objective computes for several alpha values."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, alpha, bkd
        )
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        result = obj(weights)
        val = float(bkd.to_numpy(result).flat[0])
        assert np.isfinite(val) and val > 0.0

    def test_jacobian_matches_autograd(self, torch_bkd):
        """Test analytical jacobian vs torch.autograd.functional.jacobian."""
        import torch

        bkd = torch_bkd
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        weights = bkd.full((nobs, 1), 1.0 / nobs)

        # Autograd jacobian
        def f(w):
            return obj(w.reshape(nobs, 1)).reshape(1)

        auto_jac = torch.autograd.functional.jacobian(
            f, weights.reshape(nobs)
        )  # (1, nobs)

        # Finite difference jacobian from objective
        fd_jac = obj.jacobian(weights)

        bkd.assert_allclose(fd_jac, auto_jac, rtol=1e-4)
