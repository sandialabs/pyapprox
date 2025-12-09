"""
Tests for full-rank Laplace posterior.
"""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.inverse.laplace.full_rank import DenseLaplacePosterior
from pyapprox.typing.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior


class TestDenseLaplacePosteriorBase(Generic[Array], unittest.TestCase):
    """Base test class for DenseLaplacePosterior."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        self.nvars = 3

        # MAP point
        self.map_point = self.bkd().asarray([[1.0], [2.0], [3.0]])

        # Prior precision
        self.prior_precision = self.bkd().asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        # Likelihood Hessian (from some nonlinear model)
        self.likelihood_hessian = self.bkd().asarray([
            [2.0, 0.5, 0.1],
            [0.5, 3.0, 0.2],
            [0.1, 0.2, 1.0],
        ])

        self.laplace = DenseLaplacePosterior(
            self.map_point,
            self.prior_precision,
            self.likelihood_hessian,
            self.bkd(),
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.laplace.nvars(), self.nvars)

    def test_posterior_mean_is_map(self) -> None:
        """Test posterior mean is the MAP point."""
        mean = self.laplace.posterior_mean()
        mean_np = self.bkd().to_numpy(mean)
        map_np = self.bkd().to_numpy(self.map_point)
        np.testing.assert_array_equal(mean_np, map_np)

    def test_posterior_covariance_shape(self) -> None:
        """Test posterior covariance has correct shape."""
        self.laplace.compute()
        cov = self.laplace.posterior_covariance()
        self.assertEqual(cov.shape, (self.nvars, self.nvars))

    def test_posterior_covariance_symmetric(self) -> None:
        """Test posterior covariance is symmetric."""
        self.laplace.compute()
        cov = self.laplace.posterior_covariance()
        cov_np = self.bkd().to_numpy(cov)
        np.testing.assert_array_almost_equal(cov_np, cov_np.T)

    def test_posterior_covariance_positive_definite(self) -> None:
        """Test posterior covariance is positive definite."""
        self.laplace.compute()
        cov = self.laplace.posterior_covariance()
        cov_np = self.bkd().to_numpy(cov)
        eigenvalues = np.linalg.eigvalsh(cov_np)
        self.assertTrue(all(eigenvalues > 0))

    def test_covariance_diagonal_shape(self) -> None:
        """Test covariance diagonal has correct shape."""
        self.laplace.compute()
        diag = self.laplace.covariance_diagonal()
        self.assertEqual(diag.shape, (self.nvars,))

    def test_covariance_diagonal_matches_matrix(self) -> None:
        """Test diagonal matches full matrix diagonal."""
        self.laplace.compute()
        diag = self.laplace.covariance_diagonal()
        cov = self.laplace.posterior_covariance()
        diag_np = self.bkd().to_numpy(diag)
        cov_diag_np = np.diag(self.bkd().to_numpy(cov))
        np.testing.assert_array_almost_equal(diag_np, cov_diag_np)

    def test_compute_not_called_raises(self) -> None:
        """Test accessing covariance before compute raises error."""
        with self.assertRaises(RuntimeError):
            self.laplace.posterior_covariance()

    def test_posterior_variable_returns_gaussian(self) -> None:
        """Test posterior_variable returns a Gaussian distribution."""
        self.laplace.compute()
        post = self.laplace.posterior_variable()
        self.assertTrue(hasattr(post, 'logpdf'))
        self.assertTrue(hasattr(post, 'rvs'))


class TestDenseLaplacePosteriorVsConjugate(Generic[Array], unittest.TestCase):
    """Test Laplace matches conjugate for linear Gaussian model."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_matches_conjugate_for_linear_model(self) -> None:
        """
        For linear Gaussian model, Laplace should match conjugate solution.

        The log-likelihood Hessian for a linear model y = Ax + noise with
        noise ~ N(0, Sigma) is A^T @ Sigma^{-1} @ A, which is constant.
        """
        nvars = 2
        nobs = 3

        # Linear model matrix
        A = self.bkd().asarray([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])

        # Prior
        prior_mean = self.bkd().zeros((nvars, 1))
        prior_cov = self.bkd().eye(nvars)
        prior_precision = self.bkd().eye(nvars)

        # Noise
        noise_var = 0.1
        noise_cov = noise_var * self.bkd().eye(nobs)
        noise_precision = (1.0 / noise_var) * self.bkd().eye(nobs)

        # Observations
        obs = self.bkd().asarray([[1.0], [1.5], [2.0]])

        # Conjugate solution
        conjugate = DenseGaussianConjugatePosterior(
            A, prior_mean, prior_cov, noise_cov, self.bkd()
        )
        conjugate.compute(obs)
        conj_mean = self.bkd().to_numpy(conjugate.posterior_mean())
        conj_cov = self.bkd().to_numpy(conjugate.posterior_covariance())

        # Laplace solution
        # For linear model, likelihood Hessian = A^T @ noise_precision @ A
        likelihood_hessian = A.T @ noise_precision @ A

        laplace = DenseLaplacePosterior(
            conjugate.posterior_mean(),  # Use conjugate MAP as starting point
            prior_precision,
            likelihood_hessian,
            self.bkd(),
        )
        laplace.compute()
        laplace_cov = self.bkd().to_numpy(laplace.posterior_covariance())

        # Covariances should match
        np.testing.assert_array_almost_equal(laplace_cov, conj_cov, decimal=5)


class TestDenseLaplacePosteriorValidation(Generic[Array], unittest.TestCase):
    """Test input validation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_wrong_map_shape_raises(self) -> None:
        """Test wrong MAP point shape raises error."""
        map_point = self.bkd().zeros((3, 2))  # Wrong shape
        prior_precision = self.bkd().eye(3)
        likelihood_hessian = self.bkd().eye(3)

        with self.assertRaises(ValueError):
            DenseLaplacePosterior(
                map_point, prior_precision, likelihood_hessian, self.bkd()
            )

    def test_wrong_prior_precision_shape_raises(self) -> None:
        """Test wrong prior precision shape raises error."""
        map_point = self.bkd().zeros((3, 1))
        prior_precision = self.bkd().eye(2)  # Wrong shape
        likelihood_hessian = self.bkd().eye(3)

        with self.assertRaises(ValueError):
            DenseLaplacePosterior(
                map_point, prior_precision, likelihood_hessian, self.bkd()
            )

    def test_wrong_likelihood_hessian_shape_raises(self) -> None:
        """Test wrong likelihood Hessian shape raises error."""
        map_point = self.bkd().zeros((3, 1))
        prior_precision = self.bkd().eye(3)
        likelihood_hessian = self.bkd().eye(2)  # Wrong shape

        with self.assertRaises(ValueError):
            DenseLaplacePosterior(
                map_point, prior_precision, likelihood_hessian, self.bkd()
            )


# NumPy backend tests
class TestDenseLaplacePosteriorNumpy(TestDenseLaplacePosteriorBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDenseLaplacePosteriorVsConjugateNumpy(
    TestDenseLaplacePosteriorVsConjugate[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestDenseLaplacePosteriorValidationNumpy(
    TestDenseLaplacePosteriorValidation[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestDenseLaplacePosteriorTorch(TestDenseLaplacePosteriorBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestDenseLaplacePosteriorVsConjugateTorch(
    TestDenseLaplacePosteriorVsConjugate[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestDenseLaplacePosteriorValidationTorch(
    TestDenseLaplacePosteriorValidation[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
