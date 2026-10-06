"""Tests for LinearGaussianNuisanceLognormalBenchmark.

The ground-truth methods are checked against the relaxed-weight functions
of ``pyapprox.expdesign.analytical`` on the benchmark's matrices and,
independently, against the conjugate Gaussian OED classes on the selected
rows of a binary design.
"""

import numpy as np
import pytest

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDExpectedPushforwardKLDivergence,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
    lognormal_goal_mg_blocks,
    relaxed_linear_target_covariance,
    relaxed_linear_target_eig,
    relaxed_lognormal_expected_variance,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.expdesign import (
    LinearGaussianNuisanceLognormalBenchmark,
    build_linear_gaussian_nuisance_lognormal_benchmark,
)


class TestLinearGaussianNuisanceLognormalBenchmark:
    """Sizes nm=3, na=2, nb=1 on 6 candidate sensors with 2 QoIs."""

    _nobs, _nm, _na, _nb, _nqoi = 6, 3, 2, 1, 2

    def _benchmark(
        self, bkd: Backend[Array]
    ) -> LinearGaussianNuisanceLognormalBenchmark[Array]:
        return build_linear_gaussian_nuisance_lognormal_benchmark(
            self._nobs, self._nm, self._na, self._nb, self._nqoi, bkd, seed=3
        )

    def _weights(self, bkd: Backend[Array]) -> Array:
        return bkd.asarray([[0.7], [0.0], [0.2], [1.0], [0.5], [0.9]])

    def test_shapes(self, bkd: Backend[Array]) -> None:
        bench = self._benchmark(bkd)
        nvars = self._nm + self._na + self._nb
        assert bench.obs_matrix().shape == (self._nobs, nvars)
        assert bench.qoi_matrix().shape == (self._nqoi, nvars)
        assert bench.prior_mean().shape == (nvars, 1)
        assert bench.prior_covariance().shape == (nvars, nvars)
        assert bench.noise_covariance().shape == (self._nobs, self._nobs)
        assert bench.problem().nobs() == self._nobs
        assert bench.problem().nparams() == nvars
        weights = self._weights(bkd)
        assert bench.exact_goal_expected_posterior_variance(weights).shape == (
            self._nqoi,
            1,
        )
        assert bench.exact_goal_eig(weights).shape == (1, 1)
        assert bench.exact_param_posterior_covariance(weights).shape == (
            self._nm,
            self._nm,
        )

    def test_prediction_nuisance_does_not_enter_data(self, bkd: Backend[Array]) -> None:
        bench = self._benchmark(bkd)
        nb_cols = bench.obs_matrix()[:, self._nm + self._na :]
        bkd.assert_allclose(nb_cols, bkd.zeros((self._nobs, self._nb)))

    def test_evaluate_both_matches_matrices(self, bkd: Backend[Array]) -> None:
        bench = self._benchmark(bkd)
        samples = bench.prior().rvs(7)
        obs, qoi = bench.evaluate_both(samples)
        bkd.assert_allclose(obs, bkd.dot(bench.obs_matrix(), samples), rtol=1e-12)
        bkd.assert_allclose(
            qoi, bkd.exp(bkd.dot(bench.qoi_matrix(), samples)), rtol=1e-12
        )

    def test_ground_truth_matches_analytical_functions(
        self, bkd: Backend[Array]
    ) -> None:
        bench = self._benchmark(bkd)
        weights = self._weights(bkd)
        obs_mat, qoi_mat = bench.obs_matrix(), bench.qoi_matrix()
        mean, cov = bench.prior_mean(), bench.prior_covariance()
        noise = bench.noise_covariance()
        bkd.assert_allclose(
            bench.exact_goal_expected_posterior_variance(weights),
            relaxed_lognormal_expected_variance(
                qoi_mat, obs_mat, mean, cov, noise, weights, bkd
            ),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            bench.exact_goal_eig(weights),
            relaxed_linear_target_eig(qoi_mat, obs_mat, cov, noise, weights, bkd),
            rtol=1e-12,
        )
        param_mat = bkd.eye(cov.shape[0])[: self._nm]
        bkd.assert_allclose(
            bench.exact_param_posterior_covariance(weights),
            relaxed_linear_target_covariance(
                param_mat, obs_mat, cov, noise, weights, bkd
            ),
            rtol=1e-12,
        )
        blocks = bench.exact_mg_blocks()
        expected = lognormal_goal_mg_blocks(obs_mat, qoi_mat, mean, cov, bkd)
        bkd.assert_allclose(blocks.qoi_cov, expected.qoi_cov, rtol=1e-12)
        bkd.assert_allclose(blocks.qoi_obs_cov, expected.qoi_obs_cov, rtol=1e-12)

    _rows = [0, 3, 5]

    def _binary_weights(self, bkd: Backend[Array]) -> Array:
        weights = np.zeros((self._nobs, 1))
        weights[self._rows] = 1.0
        return bkd.asarray(weights)

    def test_binary_design_matches_existing_classes(self, bkd: Backend[Array]) -> None:
        """At binary weights the values are those of the selected sensors."""
        bench = self._benchmark(bkd)
        weights = self._binary_weights(bkd)
        obs_rows = bench.obs_matrix()[self._rows]
        noise_rows = bench.noise_covariance()[self._rows][:, self._rows]
        kl = ConjugateGaussianOEDExpectedPushforwardKLDivergence(
            bench.prior_mean(), bench.prior_covariance(), bench.qoi_matrix(), bkd
        )
        kl.set_observation_matrix(obs_rows)
        kl.set_noise_covariance(noise_rows)
        bkd.assert_allclose(
            bench.exact_goal_eig(weights)[0],
            bkd.asarray([float(kl.value())]),
            rtol=1e-10,
        )
        variance = bench.exact_goal_expected_posterior_variance(weights)
        for ii in range(self._nqoi):
            util = ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance(
                bench.prior_mean(),
                bench.prior_covariance(),
                bench.qoi_matrix()[ii : ii + 1],
                bkd,
            )
            util.set_observation_matrix(obs_rows)
            util.set_noise_covariance(noise_rows)
            bkd.assert_allclose(
                variance[ii], bkd.asarray([float(util.value())]), rtol=1e-10
            )

    def test_nuisance_free_variant(self, bkd: Backend[Array]) -> None:
        """Ignoring the nuisances understates the posterior uncertainty."""
        bench = self._benchmark(bkd)
        ignoring = bench.nuisance_free()
        assert ignoring.nparams() == self._nm
        assert ignoring.nobs_nuisance() == 0
        assert ignoring.npred_nuisance() == 0
        bkd.assert_allclose(ignoring.obs_matrix(), bench.obs_matrix()[:, : self._nm])
        weights = self._weights(bkd)
        true_var = bkd.to_numpy(bench.exact_goal_expected_posterior_variance(weights))
        believed_var = bkd.to_numpy(
            ignoring.exact_goal_expected_posterior_variance(weights)
        )
        assert np.all(believed_var < true_var)
        true_cov = bkd.to_numpy(bench.exact_param_posterior_covariance(weights))
        believed_cov = bkd.to_numpy(ignoring.exact_param_posterior_covariance(weights))
        assert np.trace(believed_cov) < np.trace(true_cov)

    def test_rejects_nonpositive_std(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            build_linear_gaussian_nuisance_lognormal_benchmark(
                self._nobs,
                self._nm,
                self._na,
                self._nb,
                self._nqoi,
                bkd,
                obs_nuisance_std=0.0,
            )
