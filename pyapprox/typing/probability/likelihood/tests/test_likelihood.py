"""
Tests for likelihood functions.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.probability.covariance import (
    DenseCholeskyCovarianceOperator,
)
from pyapprox.typing.probability.likelihood import (
    GaussianLogLikelihood,
    DiagonalGaussianLogLikelihood,
)


class TestGaussianLogLikelihood(Generic[Array], unittest.TestCase):
    """Tests for GaussianLogLikelihood."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.nobs = 3
        self.noise_cov = 0.01 * np.eye(self.nobs)
        self.noise_op = DenseCholeskyCovarianceOperator(
            self._bkd.asarray(self.noise_cov), self._bkd
        )
        self.likelihood = GaussianLogLikelihood(self.noise_op, self._bkd)

    def test_nobs(self) -> None:
        """Test nobs returns correct dimension."""
        self.assertEqual(self.likelihood.nobs(), 3)

    def test_noise_covariance_operator(self) -> None:
        """Test noise covariance operator accessor."""
        self.assertIsNotNone(self.likelihood.noise_covariance_operator())

    def test_set_observations(self) -> None:
        """Test setting observations."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)
        # Should not raise

    def test_logpdf_at_observations(self) -> None:
        """Test logpdf at exact observations is maximum."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        # At observations, logpdf should be maximum
        logpdf_at_obs = self.likelihood.logpdf(obs)

        # Perturbed
        model_perturbed = obs + 0.1
        logpdf_perturbed = self.likelihood.logpdf(model_perturbed)

        self.assertGreater(float(logpdf_at_obs), float(logpdf_perturbed))

    def test_logpdf_vs_scipy(self) -> None:
        """Test logpdf matches scipy multivariate normal."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])  # Shape: (nobs, 1)
        self.likelihood.set_observations(obs)

        model = self._bkd.asarray([[1.01], [1.99], [3.02]])  # Shape: (nobs, 1)

        # Our likelihood
        logpdf_ours = self.likelihood.logpdf(model)

        # Scipy (note: scipy uses obs as the random variable, model as mean)
        model_np = self._bkd.to_numpy(model).flatten()
        scipy_dist = stats.multivariate_normal(model_np, self.noise_cov)
        obs_np = self._bkd.to_numpy(obs).flatten()
        logpdf_scipy = scipy_dist.logpdf(obs_np)

        self.assertTrue(
            self._bkd.allclose(
                self._bkd.flatten(logpdf_ours),
                self._bkd.asarray([logpdf_scipy]),
                rtol=1e-12,
            )
        )

    def test_logpdf_batch(self) -> None:
        """Test logpdf with multiple model samples."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        model = self._bkd.asarray(
            [[1.0, 1.1, 0.9], [2.0, 2.1, 1.9], [3.0, 3.1, 2.9]]
        )
        logpdf = self.likelihood.logpdf(model)
        # logpdf returns (1, nsamples)
        self.assertEqual(logpdf.shape, (1, 3))

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        model = self._bkd.asarray([[1.0], [2.0], [3.0]])
        samples = self.likelihood.rvs(model, nsamples=10)
        self.assertEqual(samples.shape, (3, 10))

    def test_rvs_mean(self) -> None:
        """Test rvs has correct mean."""
        np.random.seed(42)
        model = self._bkd.asarray([[1.0], [2.0], [3.0]])
        samples = self.likelihood.rvs(model, nsamples=10000)

        mean = self._bkd.mean(samples, axis=1)
        expected = self._bkd.flatten(model)
        self.assertTrue(self._bkd.allclose(mean, expected, atol=0.1))

    def test_gradient_shape(self) -> None:
        """Test gradient has correct shape."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        model = self._bkd.asarray([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        grad = self.likelihood.gradient(model)

        self.assertEqual(grad.shape, (3, 2))

    def test_gradient_at_observations_zero(self) -> None:
        """Test gradient at observations is zero."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        grad = self.likelihood.gradient(obs)
        expected = self._bkd.zeros((3, 1))
        self.assertTrue(self._bkd.allclose(grad, expected, atol=1e-10))

    def test_observations_not_set_raises(self) -> None:
        """Test logpdf without observations raises error."""
        model = self._bkd.asarray([[1.0], [2.0], [3.0]])
        with self.assertRaises(ValueError):
            self.likelihood.logpdf(model)


class TestGaussianLogLikelihoodNumpy(TestGaussianLogLikelihood[NDArray[Any]]):
    """NumPy backend tests for GaussianLogLikelihood."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianLogLikelihoodTorch(TestGaussianLogLikelihood[torch.Tensor]):
    """PyTorch backend tests for GaussianLogLikelihood."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestGaussianLogLikelihoodCorrelated(Generic[Array], unittest.TestCase):
    """Tests for GaussianLogLikelihood with correlated noise."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.nobs = 2
        self.noise_cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.noise_op = DenseCholeskyCovarianceOperator(
            self._bkd.asarray(self.noise_cov), self._bkd
        )
        self.likelihood = GaussianLogLikelihood(self.noise_op, self._bkd)

    def test_logpdf_vs_scipy(self) -> None:
        """Test logpdf matches scipy for correlated noise."""
        obs = self._bkd.asarray([[1.0], [2.0]])  # Shape: (nobs, 1)
        self.likelihood.set_observations(obs)

        model = self._bkd.asarray([[1.1], [1.9]])  # Shape: (nobs, 1)

        logpdf_ours = self.likelihood.logpdf(model)

        model_np = self._bkd.to_numpy(model).flatten()
        scipy_dist = stats.multivariate_normal(model_np, self.noise_cov)
        obs_np = self._bkd.to_numpy(obs).flatten()
        logpdf_scipy = scipy_dist.logpdf(obs_np)

        self.assertTrue(
            self._bkd.allclose(
                self._bkd.flatten(logpdf_ours),
                self._bkd.asarray([logpdf_scipy]),
                rtol=1e-12,
            )
        )


class TestGaussianLogLikelihoodCorrelatedNumpy(
    TestGaussianLogLikelihoodCorrelated[NDArray[Any]]
):
    """NumPy backend tests for GaussianLogLikelihood with correlated noise."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianLogLikelihoodCorrelatedTorch(
    TestGaussianLogLikelihoodCorrelated[torch.Tensor]
):
    """PyTorch backend tests for GaussianLogLikelihood with correlated noise."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDiagonalGaussianLogLikelihood(Generic[Array], unittest.TestCase):
    """Tests for DiagonalGaussianLogLikelihood."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.noise_var = np.array([0.01, 0.02, 0.01])
        self.likelihood = DiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd
        )

    def test_nobs(self) -> None:
        """Test nobs returns correct dimension."""
        self.assertEqual(self.likelihood.nobs(), 3)

    def test_logpdf_at_observations(self) -> None:
        """Test logpdf at exact observations is maximum."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        logpdf_at_obs = self.likelihood.logpdf(obs)
        model_perturbed = obs + 0.1
        logpdf_perturbed = self.likelihood.logpdf(model_perturbed)

        self.assertGreater(float(logpdf_at_obs), float(logpdf_perturbed))

    def test_logpdf_vs_full_gaussian(self) -> None:
        """Test diagonal matches full Gaussian with diagonal cov."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        model = self._bkd.asarray([[1.01], [1.99], [3.02]])

        # Full Gaussian likelihood
        noise_cov = self._bkd.diag(self._bkd.asarray(self.noise_var))
        noise_op = DenseCholeskyCovarianceOperator(noise_cov, self._bkd)
        full_likelihood = GaussianLogLikelihood(noise_op, self._bkd)
        full_likelihood.set_observations(obs)

        logpdf_diag = self.likelihood.logpdf(model)
        logpdf_full = full_likelihood.logpdf(model)

        self.assertTrue(self._bkd.allclose(logpdf_diag, logpdf_full, rtol=1e-12))

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        model = self._bkd.asarray([[1.0], [2.0], [3.0]])
        samples = self.likelihood.rvs(model, nsamples=10)
        self.assertEqual(samples.shape, (3, 10))

    def test_gradient_shape(self) -> None:
        """Test gradient has correct shape."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        self.likelihood.set_observations(obs)

        model = self._bkd.asarray([[1.0], [2.0], [3.0]])
        grad = self.likelihood.gradient(model)

        self.assertEqual(grad.shape, (3, 1))

    def test_logpdf_vectorized_shape(self) -> None:
        """Test vectorized logpdf returns correct shape."""
        model = self._bkd.asarray(
            [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        )
        obs = self._bkd.asarray(
            [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]
        )
        logpdf = self.likelihood.logpdf_vectorized(model, obs)
        self.assertEqual(logpdf.shape, (3, 2))

    def test_logpdf_vectorized_matches_loop(self) -> None:
        """Test vectorized matches sequential set_observations + logpdf."""
        model = self._bkd.asarray(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
        )
        obs = self._bkd.asarray(
            [[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]]
        )
        logpdf_vec = self.likelihood.logpdf_vectorized(model, obs)

        for j in range(obs.shape[1]):
            self.likelihood.set_observations(obs[:, j:j+1])
            logpdf_seq = self.likelihood.logpdf(model)
            self._bkd.assert_allclose(
                self._bkd.reshape(logpdf_vec[:, j], (1, -1)),
                logpdf_seq,
                rtol=1e-12,
            )

    def test_logpdf_vectorized_matches_full(self) -> None:
        """Test diagonal vectorized matches full GaussianLogLikelihood."""
        noise_cov = self._bkd.diag(self._bkd.asarray(self.noise_var))
        noise_op = DenseCholeskyCovarianceOperator(noise_cov, self._bkd)
        full_lik = GaussianLogLikelihood(noise_op, self._bkd)

        model = self._bkd.asarray(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
        )
        obs = self._bkd.asarray(
            [[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]]
        )

        logpdf_diag = self.likelihood.logpdf_vectorized(model, obs)
        logpdf_full = full_lik.logpdf_vectorized(model, obs)

        self._bkd.assert_allclose(logpdf_diag, logpdf_full, rtol=1e-12)


class TestDiagonalGaussianLogLikelihoodNumpy(
    TestDiagonalGaussianLogLikelihood[NDArray[Any]]
):
    """NumPy backend tests for DiagonalGaussianLogLikelihood."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDiagonalGaussianLogLikelihoodTorch(
    TestDiagonalGaussianLogLikelihood[torch.Tensor]
):
    """PyTorch backend tests for DiagonalGaussianLogLikelihood."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestDesignWeights(Generic[Array], unittest.TestCase):
    """Tests for design weights functionality."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.noise_var = self._bkd.asarray([0.01, 0.01, 0.01])
        self.likelihood = DiagonalGaussianLogLikelihood(self.noise_var, self._bkd)

    def test_set_design_weights(self) -> None:
        """Test setting design weights."""
        weights = self._bkd.asarray([1.0, 0.5, 0.25])
        self.likelihood.set_design_weights(weights)
        # Should not raise

    def test_weights_affect_logpdf(self) -> None:
        """Test design weights affect logpdf."""
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        model = self._bkd.asarray([[1.1], [2.1], [3.1]])  # Same error per obs

        self.likelihood.set_observations(obs)

        logpdf_no_weights = self.likelihood.logpdf(model)

        # Zero out third observation
        weights = self._bkd.asarray([1.0, 1.0, 0.0])
        self.likelihood.set_design_weights(weights)
        logpdf_with_weights = self.likelihood.logpdf(model)

        # With zero weight on third obs, logpdf should be higher
        self.assertGreater(float(logpdf_with_weights), float(logpdf_no_weights))


class TestDesignWeightsNumpy(TestDesignWeights[NDArray[Any]]):
    """NumPy backend tests for design weights."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDesignWeightsTorch(TestDesignWeights[torch.Tensor]):
    """PyTorch backend tests for design weights."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestVectorizedLogLikelihood(Generic[Array], unittest.TestCase):
    """Tests for vectorized log-likelihood evaluation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.noise_cov = 0.01 * np.eye(2)
        self.noise_op = DenseCholeskyCovarianceOperator(
            self._bkd.asarray(self.noise_cov), self._bkd
        )
        self.likelihood = GaussianLogLikelihood(self.noise_op, self._bkd)

    def test_vectorized_shape(self) -> None:
        """Test vectorized logpdf returns correct shape."""
        model = self._bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])
        obs = self._bkd.asarray([[1.0, 1.5], [2.0, 2.5]])

        logpdf = self.likelihood.logpdf_vectorized(model, obs)
        self.assertEqual(logpdf.shape, (3, 2))

    def test_vectorized_matches_loop(self) -> None:
        """Test vectorized matches sequential evaluation."""
        model = self._bkd.asarray([[1.0, 1.1], [2.0, 2.1]])
        obs = self._bkd.asarray([[1.0, 1.2], [2.0, 2.2]])

        logpdf_vec = self.likelihood.logpdf_vectorized(model, obs)

        # Sequential
        for j in range(obs.shape[1]):
            self.likelihood.set_observations(obs[:, j : j + 1])
            logpdf_seq = self.likelihood.logpdf(model)
            self.assertTrue(
                self._bkd.allclose(logpdf_vec[:, j], logpdf_seq, rtol=1e-12)
            )


class TestVectorizedLogLikelihoodNumpy(TestVectorizedLogLikelihood[NDArray[Any]]):
    """NumPy backend tests for vectorized log-likelihood."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestVectorizedLogLikelihoodTorch(TestVectorizedLogLikelihood[torch.Tensor]):
    """PyTorch backend tests for vectorized log-likelihood."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestParallelDiagonalGaussianLogLikelihood(Generic[Array], unittest.TestCase):
    """Tests for ParallelDiagonalGaussianLogLikelihood."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.noise_var = np.array([0.01, 0.02, 0.01])

    def test_sequential_matches_base(self) -> None:
        """Test nprocs=1 matches base class."""
        from pyapprox.typing.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        base = DiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd
        )
        parallel = ParallelDiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd, nprocs=1
        )

        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        base.set_observations(obs)
        parallel.set_observations(obs)

        model = self._bkd.asarray([[1.01], [1.99], [3.02]])

        logpdf_base = base.logpdf(model)
        logpdf_parallel = parallel.logpdf(model)

        self.assertTrue(self._bkd.allclose(logpdf_base, logpdf_parallel, rtol=1e-12))

    def test_parallel_matches_sequential(self) -> None:
        """Test parallel logpdf_vectorized matches sequential."""
        from pyapprox.typing.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        seq_lik = ParallelDiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd, nprocs=1
        )
        par_lik = ParallelDiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd, nprocs=2
        )

        model = self._bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]])
        obs = self._bkd.asarray([[1.0, 1.5, 1.2, 1.3], [2.0, 2.5, 2.2, 2.3], [3.0, 3.5, 3.2, 3.3]])

        result_seq = seq_lik.logpdf_vectorized(model, obs)
        result_par = par_lik.logpdf_vectorized(model, obs)

        self.assertEqual(result_seq.shape, result_par.shape)
        self.assertTrue(self._bkd.allclose(result_seq, result_par, rtol=1e-12))

    def test_logpdf_vectorized_shape(self) -> None:
        """Test logpdf_vectorized returns correct shape."""
        from pyapprox.typing.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        lik = ParallelDiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd, nprocs=2
        )

        n_model = 5
        n_obs = 8
        model = self._bkd.asarray(np.random.randn(3, n_model))
        obs = self._bkd.asarray(np.random.randn(3, n_obs))

        result = lik.logpdf_vectorized(model, obs)

        self.assertEqual(result.shape, (n_model, n_obs))

    def test_nprocs(self) -> None:
        """Test nprocs accessor."""
        from pyapprox.typing.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        lik = ParallelDiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd, nprocs=4
        )
        self.assertEqual(lik.nprocs(), 4)

    def test_repr(self) -> None:
        """Test string representation."""
        from pyapprox.typing.probability.likelihood.parallel_diagonal_gaussian import (
            ParallelDiagonalGaussianLogLikelihood,
        )

        lik = ParallelDiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd, nprocs=4
        )
        repr_str = repr(lik)
        self.assertIn("ParallelDiagonalGaussianLogLikelihood", repr_str)
        self.assertIn("nprocs=4", repr_str)


class TestParallelDiagonalGaussianLogLikelihoodNumpy(
    TestParallelDiagonalGaussianLogLikelihood[NDArray[Any]]
):
    """NumPy backend tests for ParallelDiagonalGaussianLogLikelihood."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestParallelDiagonalGaussianLogLikelihoodTorch(
    TestParallelDiagonalGaussianLogLikelihood[torch.Tensor]
):
    """PyTorch backend tests for ParallelDiagonalGaussianLogLikelihood."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestMultiExperimentLogLikelihood(Generic[Array], unittest.TestCase):
    """Tests for MultiExperimentLogLikelihood."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.noise_var = np.array([0.01, 0.02, 0.01])

    def test_logpdf_shape(self) -> None:
        """Test logpdf returns (1, nsamples)."""
        from pyapprox.typing.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        lik = DiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd
        )
        obs = self._bkd.asarray(
            [[1.0, 1.2, 1.1], [2.0, 2.2, 2.1], [3.0, 3.2, 3.1]]
        )
        multi = MultiExperimentLogLikelihood(lik, obs, self._bkd)

        model = self._bkd.asarray(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
        )
        result = multi.logpdf(model)
        self.assertEqual(result.shape, (1, 2))

    def test_single_experiment_matches_base(self) -> None:
        """Test nexperiments=1 matches base logpdf."""
        from pyapprox.typing.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        lik = DiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd
        )
        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])
        multi = MultiExperimentLogLikelihood(lik, obs, self._bkd)

        model = self._bkd.asarray(
            [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        )
        result_multi = multi.logpdf(model)

        lik.set_observations(obs)
        result_base = lik.logpdf(model)

        self._bkd.assert_allclose(result_multi, result_base, rtol=1e-12)

    def test_multi_experiment_matches_loop(self) -> None:
        """Test sum matches iterating over experiments."""
        from pyapprox.typing.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        lik = DiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd
        )
        obs = self._bkd.asarray(
            [[1.0, 1.2, 1.1], [2.0, 2.2, 2.1], [3.0, 3.2, 3.1]]
        )
        multi = MultiExperimentLogLikelihood(lik, obs, self._bkd)

        model = self._bkd.asarray(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
        )
        result_multi = multi.logpdf(model)

        # Manual loop
        total = self._bkd.zeros((1, 2))
        for j in range(obs.shape[1]):
            lik.set_observations(obs[:, j:j+1])
            total = total + lik.logpdf(model)

        self._bkd.assert_allclose(result_multi, total, rtol=1e-12)

    def test_nobs_nexperiments(self) -> None:
        """Test accessor methods."""
        from pyapprox.typing.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        lik = DiagonalGaussianLogLikelihood(
            self._bkd.asarray(self.noise_var), self._bkd
        )
        obs = self._bkd.asarray(
            [[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]]
        )
        multi = MultiExperimentLogLikelihood(lik, obs, self._bkd)

        self.assertEqual(multi.nobs(), 3)
        self.assertEqual(multi.nexperiments(), 2)

    def test_invalid_base_raises(self) -> None:
        """Test protocol check fires for invalid base."""
        from pyapprox.typing.probability.likelihood import (
            MultiExperimentLogLikelihood,
        )

        obs = self._bkd.asarray([[1.0], [2.0], [3.0]])

        class FakeLikelihood:
            pass

        with self.assertRaises(TypeError):
            MultiExperimentLogLikelihood(FakeLikelihood(), obs, self._bkd)


class TestMultiExperimentLogLikelihoodNumpy(
    TestMultiExperimentLogLikelihood[NDArray[Any]]
):
    """NumPy backend tests for MultiExperimentLogLikelihood."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultiExperimentLogLikelihoodTorch(
    TestMultiExperimentLogLikelihood[torch.Tensor]
):
    """PyTorch backend tests for MultiExperimentLogLikelihood."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
