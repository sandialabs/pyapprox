"""
Tests for likelihood functions.
"""

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.covariance import (
    DenseCholeskyCovarianceOperator,
)
from pyapprox.probability.likelihood import (
    DiagonalGaussianLogLikelihood,
    GaussianLogLikelihood,
)


class TestGaussianLogLikelihood:
    """Tests for GaussianLogLikelihood."""

    def _setup(self, bkd):
        nobs = 3
        noise_cov = 0.01 * np.eye(nobs)
        noise_op = DenseCholeskyCovarianceOperator(
            bkd.asarray(noise_cov), bkd
        )
        likelihood = GaussianLogLikelihood(noise_op, bkd)
        return nobs, noise_cov, noise_op, likelihood

    def test_nobs(self, bkd) -> None:
        """Test nobs returns correct dimension."""
        _, _, _, likelihood = self._setup(bkd)
        assert likelihood.nobs() == 3

    def test_noise_covariance_operator(self, bkd) -> None:
        """Test noise covariance operator accessor."""
        _, _, _, likelihood = self._setup(bkd)
        assert likelihood.noise_covariance_operator() is not None

    def test_set_observations(self, bkd) -> None:
        """Test setting observations."""
        _, _, _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        likelihood.set_observations(obs)
        # Should not raise

    def test_logpdf_at_observations(self, bkd) -> None:
        """Test logpdf at exact observations is maximum."""
        _, _, _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        likelihood.set_observations(obs)

        # At observations, logpdf should be maximum
        logpdf_at_obs = likelihood.logpdf(obs)

        # Perturbed
        model_perturbed = obs + 0.1
        logpdf_perturbed = likelihood.logpdf(model_perturbed)

        assert float(logpdf_at_obs) > float(logpdf_perturbed)

    def test_logpdf_vs_scipy(self, bkd) -> None:
        """Test logpdf matches scipy multivariate normal."""
        _, noise_cov, _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])  # Shape: (nobs, 1)
        likelihood.set_observations(obs)

        model = bkd.asarray([[1.01], [1.99], [3.02]])  # Shape: (nobs, 1)

        # Our likelihood
        logpdf_ours = likelihood.logpdf(model)

        # Scipy (note: scipy uses obs as the random variable, model as mean)
        model_np = bkd.to_numpy(model).flatten()
        scipy_dist = stats.multivariate_normal(model_np, noise_cov)
        obs_np = bkd.to_numpy(obs).flatten()
        logpdf_scipy = scipy_dist.logpdf(obs_np)

        assert bkd.allclose(
            bkd.flatten(logpdf_ours),
            bkd.asarray([logpdf_scipy]),
            rtol=1e-12,
        )

    def test_logpdf_batch(self, bkd) -> None:
        """Test logpdf with multiple model samples."""
        _, _, _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        likelihood.set_observations(obs)

        model = bkd.asarray([[1.0, 1.1, 0.9], [2.0, 2.1, 1.9], [3.0, 3.1, 2.9]])
        logpdf = likelihood.logpdf(model)
        # logpdf returns (1, nsamples)
        assert logpdf.shape == (1, 3)

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs returns correct shape."""
        _, _, _, likelihood = self._setup(bkd)
        model = bkd.asarray([[1.0], [2.0], [3.0]])
        samples = likelihood.rvs(model, nsamples=10)
        assert samples.shape == (3, 10)

    def test_rvs_mean(self, bkd) -> None:
        """Test rvs has correct mean."""
        _, _, _, likelihood = self._setup(bkd)
        np.random.seed(42)
        model = bkd.asarray([[1.0], [2.0], [3.0]])
        samples = likelihood.rvs(model, nsamples=10000)

        mean = bkd.mean(samples, axis=1)
        expected = bkd.flatten(model)
        assert bkd.allclose(mean, expected, atol=0.1)

    def test_gradient_shape(self, bkd) -> None:
        """Test gradient has correct shape."""
        _, _, _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        likelihood.set_observations(obs)

        model = bkd.asarray([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        grad = likelihood.gradient(model)

        assert grad.shape == (3, 2)

    def test_gradient_at_observations_zero(self, bkd) -> None:
        """Test gradient at observations is zero."""
        _, _, _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        likelihood.set_observations(obs)

        grad = likelihood.gradient(obs)
        expected = bkd.zeros((3, 1))
        assert bkd.allclose(grad, expected, atol=1e-10)

    def test_observations_not_set_raises(self, bkd) -> None:
        """Test logpdf without observations raises error."""
        _, _, _, likelihood = self._setup(bkd)
        model = bkd.asarray([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError):
            likelihood.logpdf(model)


class TestGaussianLogLikelihoodCorrelated:
    """Tests for GaussianLogLikelihood with correlated noise."""

    def _setup(self, bkd):
        nobs = 2
        noise_cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        noise_op = DenseCholeskyCovarianceOperator(
            bkd.asarray(noise_cov), bkd
        )
        likelihood = GaussianLogLikelihood(noise_op, bkd)
        return nobs, noise_cov, noise_op, likelihood

    def test_logpdf_vs_scipy(self, bkd) -> None:
        """Test logpdf matches scipy for correlated noise."""
        _, noise_cov, _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0]])  # Shape: (nobs, 1)
        likelihood.set_observations(obs)

        model = bkd.asarray([[1.1], [1.9]])  # Shape: (nobs, 1)

        logpdf_ours = likelihood.logpdf(model)

        model_np = bkd.to_numpy(model).flatten()
        scipy_dist = stats.multivariate_normal(model_np, noise_cov)
        obs_np = bkd.to_numpy(obs).flatten()
        logpdf_scipy = scipy_dist.logpdf(obs_np)

        assert bkd.allclose(
            bkd.flatten(logpdf_ours),
            bkd.asarray([logpdf_scipy]),
            rtol=1e-12,
        )


class TestDiagonalGaussianLogLikelihood:
    """Tests for DiagonalGaussianLogLikelihood."""

    def _setup(self, bkd):
        noise_var = np.array([0.01, 0.02, 0.01])
        likelihood = DiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd
        )
        return noise_var, likelihood

    def test_nobs(self, bkd) -> None:
        """Test nobs returns correct dimension."""
        _, likelihood = self._setup(bkd)
        assert likelihood.nobs() == 3

    def test_logpdf_at_observations(self, bkd) -> None:
        """Test logpdf at exact observations is maximum."""
        _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        likelihood.set_observations(obs)

        logpdf_at_obs = likelihood.logpdf(obs)
        model_perturbed = obs + 0.1
        logpdf_perturbed = likelihood.logpdf(model_perturbed)

        assert float(logpdf_at_obs) > float(logpdf_perturbed)

    def test_logpdf_vs_full_gaussian(self, bkd) -> None:
        """Test diagonal matches full Gaussian with diagonal cov."""
        noise_var, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        likelihood.set_observations(obs)

        model = bkd.asarray([[1.01], [1.99], [3.02]])

        # Full Gaussian likelihood
        noise_cov = bkd.diag(bkd.asarray(noise_var))
        noise_op = DenseCholeskyCovarianceOperator(noise_cov, bkd)
        full_likelihood = GaussianLogLikelihood(noise_op, bkd)
        full_likelihood.set_observations(obs)

        logpdf_diag = likelihood.logpdf(model)
        logpdf_full = full_likelihood.logpdf(model)

        assert bkd.allclose(logpdf_diag, logpdf_full, rtol=1e-12)

    def test_rvs_shape(self, bkd) -> None:
        """Test rvs returns correct shape."""
        _, likelihood = self._setup(bkd)
        model = bkd.asarray([[1.0], [2.0], [3.0]])
        samples = likelihood.rvs(model, nsamples=10)
        assert samples.shape == (3, 10)

    def test_gradient_shape(self, bkd) -> None:
        """Test gradient has correct shape."""
        _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        likelihood.set_observations(obs)

        model = bkd.asarray([[1.0], [2.0], [3.0]])
        grad = likelihood.gradient(model)

        assert grad.shape == (3, 1)

    def test_logpdf_vectorized_shape(self, bkd) -> None:
        """Test vectorized logpdf returns correct shape."""
        _, likelihood = self._setup(bkd)
        model = bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]])
        obs = bkd.asarray([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        logpdf = likelihood.logpdf_vectorized(model, obs)
        assert logpdf.shape == (3, 2)

    def test_logpdf_vectorized_matches_loop(self, bkd) -> None:
        """Test vectorized matches sequential set_observations + logpdf."""
        _, likelihood = self._setup(bkd)
        model = bkd.asarray([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        obs = bkd.asarray([[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]])
        logpdf_vec = likelihood.logpdf_vectorized(model, obs)

        for j in range(obs.shape[1]):
            likelihood.set_observations(obs[:, j : j + 1])
            logpdf_seq = likelihood.logpdf(model)
            bkd.assert_allclose(
                bkd.reshape(logpdf_vec[:, j], (1, -1)),
                logpdf_seq,
                rtol=1e-12,
            )

    def test_logpdf_vectorized_matches_full(self, bkd) -> None:
        """Test diagonal vectorized matches full GaussianLogLikelihood."""
        noise_var, likelihood = self._setup(bkd)
        noise_cov = bkd.diag(bkd.asarray(noise_var))
        noise_op = DenseCholeskyCovarianceOperator(noise_cov, bkd)
        full_lik = GaussianLogLikelihood(noise_op, bkd)

        model = bkd.asarray([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        obs = bkd.asarray([[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]])

        logpdf_diag = likelihood.logpdf_vectorized(model, obs)
        logpdf_full = full_lik.logpdf_vectorized(model, obs)

        bkd.assert_allclose(logpdf_diag, logpdf_full, rtol=1e-12)


class TestDesignWeights:
    """Tests for design weights functionality."""

    def _setup(self, bkd):
        noise_var = bkd.asarray([0.01, 0.01, 0.01])
        likelihood = DiagonalGaussianLogLikelihood(noise_var, bkd)
        return noise_var, likelihood

    def test_set_design_weights(self, bkd) -> None:
        """Test setting design weights."""
        _, likelihood = self._setup(bkd)
        weights = bkd.asarray([1.0, 0.5, 0.25])
        likelihood.set_design_weights(weights)
        # Should not raise

    def test_weights_affect_logpdf(self, bkd) -> None:
        """Test design weights affect logpdf."""
        _, likelihood = self._setup(bkd)
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        model = bkd.asarray([[1.1], [2.1], [3.1]])  # Same error per obs

        likelihood.set_observations(obs)

        logpdf_no_weights = likelihood.logpdf(model)

        # Zero out third observation
        weights = bkd.asarray([1.0, 1.0, 0.0])
        likelihood.set_design_weights(weights)
        logpdf_with_weights = likelihood.logpdf(model)

        # With zero weight on third obs, logpdf should be higher
        assert float(logpdf_with_weights) > float(logpdf_no_weights)


class TestVectorizedLogLikelihood:
    """Tests for vectorized log-likelihood evaluation."""

    def _setup(self, bkd):
        noise_cov = 0.01 * np.eye(2)
        noise_op = DenseCholeskyCovarianceOperator(
            bkd.asarray(noise_cov), bkd
        )
        likelihood = GaussianLogLikelihood(noise_op, bkd)
        return noise_cov, noise_op, likelihood

    def test_vectorized_shape(self, bkd) -> None:
        """Test vectorized logpdf returns correct shape."""
        _, _, likelihood = self._setup(bkd)
        model = bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])
        obs = bkd.asarray([[1.0, 1.5], [2.0, 2.5]])

        logpdf = likelihood.logpdf_vectorized(model, obs)
        assert logpdf.shape == (3, 2)

    def test_vectorized_matches_loop(self, bkd) -> None:
        """Test vectorized matches sequential evaluation."""
        _, _, likelihood = self._setup(bkd)
        model = bkd.asarray([[1.0, 1.1], [2.0, 2.1]])
        obs = bkd.asarray([[1.0, 1.2], [2.0, 2.2]])

        logpdf_vec = likelihood.logpdf_vectorized(model, obs)

        # Sequential
        for j in range(obs.shape[1]):
            likelihood.set_observations(obs[:, j : j + 1])
            logpdf_seq = likelihood.logpdf(model)
            assert bkd.allclose(logpdf_vec[:, j], logpdf_seq, rtol=1e-12)


class TestParallelDiagonalGaussianLogLikelihood:
    """Tests for ParallelDiagonalGaussianLogLikelihood."""

    def _setup(self, bkd):
        noise_var = np.array([0.01, 0.02, 0.01])
        return noise_var

    def test_sequential_matches_base(self, bkd) -> None:
        """Test nprocs=1 matches base class."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        base = DiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd
        )
        parallel = ParallelDiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd, nprocs=1
        )

        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        base.set_observations(obs)
        parallel.set_observations(obs)

        model = bkd.asarray([[1.01], [1.99], [3.02]])

        logpdf_base = base.logpdf(model)
        logpdf_parallel = parallel.logpdf(model)

        assert bkd.allclose(logpdf_base, logpdf_parallel, rtol=1e-12)

    def test_parallel_matches_sequential(self, bkd) -> None:
        """Test parallel logpdf_vectorized matches sequential."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        seq_lik = ParallelDiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd, nprocs=1
        )
        par_lik = ParallelDiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd, nprocs=2
        )

        model = bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]])
        obs = bkd.asarray(
            [[1.0, 1.5, 1.2, 1.3], [2.0, 2.5, 2.2, 2.3], [3.0, 3.5, 3.2, 3.3]]
        )

        result_seq = seq_lik.logpdf_vectorized(model, obs)
        result_par = par_lik.logpdf_vectorized(model, obs)

        assert result_seq.shape == result_par.shape
        assert bkd.allclose(result_seq, result_par, rtol=1e-12)

    def test_logpdf_vectorized_shape(self, bkd) -> None:
        """Test logpdf_vectorized returns correct shape."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        lik = ParallelDiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd, nprocs=2
        )

        n_model = 5
        n_obs = 8
        model = bkd.asarray(np.random.randn(3, n_model))
        obs = bkd.asarray(np.random.randn(3, n_obs))

        result = lik.logpdf_vectorized(model, obs)

        assert result.shape == (n_model, n_obs)

    def test_nprocs(self, bkd) -> None:
        """Test nprocs accessor."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        lik = ParallelDiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd, nprocs=4
        )
        assert lik.nprocs() == 4

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        lik = ParallelDiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd, nprocs=4
        )
        repr_str = repr(lik)
        assert "ParallelDiagonalGaussianLogLikelihood" in repr_str
        assert "nprocs=4" in repr_str


class TestMultiExperimentLogLikelihood:
    """Tests for MultiExperimentLogLikelihood."""

    def _setup(self, bkd):
        noise_var = np.array([0.01, 0.02, 0.01])
        return noise_var

    def test_logpdf_shape(self, bkd) -> None:
        """Test logpdf returns (1, nsamples)."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        lik = DiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd
        )
        obs = bkd.asarray([[1.0, 1.2, 1.1], [2.0, 2.2, 2.1], [3.0, 3.2, 3.1]])
        multi = MultiExperimentLogLikelihood(lik, obs, bkd)

        model = bkd.asarray([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        result = multi.logpdf(model)
        assert result.shape == (1, 2)

    def test_single_experiment_matches_base(self, bkd) -> None:
        """Test nexperiments=1 matches base logpdf."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        lik = DiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd
        )
        obs = bkd.asarray([[1.0], [2.0], [3.0]])
        multi = MultiExperimentLogLikelihood(lik, obs, bkd)

        model = bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]])
        result_multi = multi.logpdf(model)

        lik.set_observations(obs)
        result_base = lik.logpdf(model)

        bkd.assert_allclose(result_multi, result_base, rtol=1e-12)

    def test_multi_experiment_matches_loop(self, bkd) -> None:
        """Test sum matches iterating over experiments."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        lik = DiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd
        )
        obs = bkd.asarray([[1.0, 1.2, 1.1], [2.0, 2.2, 2.1], [3.0, 3.2, 3.1]])
        multi = MultiExperimentLogLikelihood(lik, obs, bkd)

        model = bkd.asarray([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        result_multi = multi.logpdf(model)

        # Manual loop
        total = bkd.zeros((1, 2))
        for j in range(obs.shape[1]):
            lik.set_observations(obs[:, j : j + 1])
            total = total + lik.logpdf(model)

        bkd.assert_allclose(result_multi, total, rtol=1e-12)

    def test_nobs_nexperiments(self, bkd) -> None:
        """Test accessor methods."""
        noise_var = self._setup(bkd)
        from pyapprox.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        lik = DiagonalGaussianLogLikelihood(
            bkd.asarray(noise_var), bkd
        )
        obs = bkd.asarray([[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]])
        multi = MultiExperimentLogLikelihood(lik, obs, bkd)

        assert multi.nobs() == 3
        assert multi.nexperiments() == 2

    def test_invalid_base_raises(self, bkd) -> None:
        """Test protocol check fires for invalid base."""
        from pyapprox.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        obs = bkd.asarray([[1.0], [2.0], [3.0]])

        class FakeLikelihood:
            pass

        with pytest.raises(TypeError):
            MultiExperimentLogLikelihood(FakeLikelihood(), obs, bkd)
