"""
Tests for MCMC diagnostic functions.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.inverse.sampling.diagnostics import (
    autocorrelation,
    integrated_autocorrelation_time,
    effective_sample_size,
    rhat,
    compute_diagnostics,
    MCMCDiagnostics,
)


class TestAutocorrelationBase(Generic[Array], unittest.TestCase):
    """Base test class for autocorrelation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_lag_zero_is_one(self) -> None:
        """Test autocorrelation at lag 0 is always 1."""
        samples = self.bkd().asarray(np.random.randn(2, 100))
        acf = autocorrelation(samples, max_lag=10, bkd=self.bkd())
        acf_np = self.bkd().to_numpy(acf)
        np.testing.assert_allclose(acf_np[:, 0], 1.0, rtol=1e-10)

    def test_independent_samples_low_autocorrelation(self) -> None:
        """Test independent samples have low autocorrelation at lag > 0."""
        np.random.seed(42)
        samples = self.bkd().asarray(np.random.randn(1, 1000))
        acf = autocorrelation(samples, max_lag=50, bkd=self.bkd())
        acf_np = self.bkd().to_numpy(acf)
        # For independent samples, autocorrelation should be near 0
        np.testing.assert_array_less(np.abs(acf_np[0, 10:]), 0.15)

    def test_ar1_process(self) -> None:
        """Test autocorrelation for AR(1) process matches theory."""
        np.random.seed(123)
        n = 5000
        phi = 0.7  # AR(1) coefficient

        # Generate AR(1): x_t = phi * x_{t-1} + epsilon_t
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + np.random.randn()

        samples = self.bkd().asarray(x.reshape(1, -1))
        acf = autocorrelation(samples, max_lag=10, bkd=self.bkd())
        acf_np = self.bkd().to_numpy(acf)[0, :]

        # Theoretical autocorrelation for AR(1): rho_k = phi^k
        for k in range(1, 6):
            expected = phi**k
            np.testing.assert_allclose(acf_np[k], expected, atol=0.1)

    def test_1d_input(self) -> None:
        """Test autocorrelation works with 1D input."""
        samples = self.bkd().asarray(np.random.randn(100))
        acf = autocorrelation(samples, max_lag=10, bkd=self.bkd())
        self.assertEqual(acf.shape, (11,))


class TestESSBase(Generic[Array], unittest.TestCase):
    """Base test class for effective sample size."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_independent_samples_ess_near_n(self) -> None:
        """Test ESS is close to n for independent samples."""
        np.random.seed(42)
        n = 1000
        samples = self.bkd().asarray(np.random.randn(2, n))
        ess = effective_sample_size(samples, bkd=self.bkd())
        ess_np = self.bkd().to_numpy(ess)
        # ESS should be close to n for independent samples
        self.assertTrue(np.all(ess_np > 0.5 * n))

    def test_correlated_samples_lower_ess(self) -> None:
        """Test correlated samples have lower ESS."""
        np.random.seed(123)
        n = 2000
        phi = 0.9  # High autocorrelation

        # Generate AR(1)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + np.random.randn()

        samples = self.bkd().asarray(x.reshape(1, -1))
        ess = effective_sample_size(samples, bkd=self.bkd())
        ess_val = float(self.bkd().to_numpy(ess)[0])
        # ESS should be much less than n for highly correlated samples
        self.assertLess(ess_val, 0.3 * n)

    def test_ess_positive(self) -> None:
        """Test ESS is always positive."""
        samples = self.bkd().asarray(np.random.randn(3, 500))
        ess = effective_sample_size(samples, bkd=self.bkd())
        ess_np = self.bkd().to_numpy(ess)
        self.assertTrue(np.all(ess_np > 0))


class TestRhatBase(Generic[Array], unittest.TestCase):
    """Base test class for R-hat."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_identical_chains_rhat_one(self) -> None:
        """Test R-hat is approximately 1 for converged chains."""
        np.random.seed(42)
        # Two chains from same distribution
        chain1 = self.bkd().asarray(np.random.randn(2, 500))
        chain2 = self.bkd().asarray(np.random.randn(2, 500))
        rhat_val = rhat([chain1, chain2], bkd=self.bkd())
        rhat_np = self.bkd().to_numpy(rhat_val)
        # R-hat should be close to 1
        np.testing.assert_array_less(rhat_np, 1.1)

    def test_divergent_chains_high_rhat(self) -> None:
        """Test R-hat is high for divergent chains."""
        # Two chains with different means
        chain1 = self.bkd().asarray(np.random.randn(1, 500))
        chain2 = self.bkd().asarray(np.random.randn(1, 500) + 5.0)
        rhat_val = rhat([chain1, chain2], bkd=self.bkd())
        rhat_np = self.bkd().to_numpy(rhat_val)
        # R-hat should be much greater than 1
        self.assertGreater(rhat_np[0], 1.5)

    def test_requires_two_chains(self) -> None:
        """Test R-hat raises error with fewer than 2 chains."""
        chain = self.bkd().asarray(np.random.randn(2, 100))
        with self.assertRaises(ValueError):
            rhat([chain], bkd=self.bkd())

    def test_multiple_chains(self) -> None:
        """Test R-hat works with more than 2 chains."""
        np.random.seed(42)
        chains = [self.bkd().asarray(np.random.randn(2, 300)) for _ in range(4)]
        rhat_val = rhat(chains, bkd=self.bkd())
        rhat_np = self.bkd().to_numpy(rhat_val)
        np.testing.assert_array_less(rhat_np, 1.1)


class TestComputeDiagnosticsBase(Generic[Array], unittest.TestCase):
    """Base test class for compute_diagnostics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_returns_dataclass(self) -> None:
        """Test compute_diagnostics returns MCMCDiagnostics."""
        samples = self.bkd().asarray(np.random.randn(2, 500))
        diag = compute_diagnostics(samples, bkd=self.bkd())
        self.assertIsInstance(diag, MCMCDiagnostics)

    def test_no_rhat_without_other_chains(self) -> None:
        """Test rhat is None when no other chains provided."""
        samples = self.bkd().asarray(np.random.randn(2, 500))
        diag = compute_diagnostics(samples, bkd=self.bkd())
        self.assertIsNone(diag.rhat)

    def test_rhat_with_other_chains(self) -> None:
        """Test rhat is computed when other chains provided."""
        np.random.seed(42)
        samples = self.bkd().asarray(np.random.randn(2, 500))
        other = [self.bkd().asarray(np.random.randn(2, 500))]
        diag = compute_diagnostics(samples, bkd=self.bkd(), other_chains=other)
        self.assertIsNotNone(diag.rhat)


# NumPy backend tests
class TestAutocorrelationNumpy(TestAutocorrelationBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestESSNumpy(TestESSBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestRhatNumpy(TestRhatBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestComputeDiagnosticsNumpy(TestComputeDiagnosticsBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestAutocorrelationTorch(TestAutocorrelationBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestESSTorch(TestESSBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestRhatTorch(TestRhatBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestComputeDiagnosticsTorch(TestComputeDiagnosticsBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


from pyapprox.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
