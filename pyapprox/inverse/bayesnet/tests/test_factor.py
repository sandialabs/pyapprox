"""
Tests for GaussianFactor with variable ID tracking.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.inverse.bayesnet.factor import GaussianFactor
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestGaussianFactorBase(Generic[Array], unittest.TestCase):
    """Base tests for GaussianFactor."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_from_moments(self) -> None:
        """Test creating factor from moments."""
        mean = self.bkd().asarray(np.array([1.0, 2.0]))
        cov = self.bkd().asarray(np.array([[1.0, 0.3], [0.3, 1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0], nvars_per_var=[2], bkd=self.bkd()
        )

        self.assertEqual(factor.var_ids(), [0])
        self.assertEqual(factor.nvars_per_var(), [2])
        self.assertEqual(factor.total_dims(), 2)

    def test_to_moments(self) -> None:
        """Test converting back to moments."""
        mean = self.bkd().asarray(np.array([1.0, 2.0]))
        cov = self.bkd().asarray(np.array([[1.0, 0.3], [0.3, 1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0], nvars_per_var=[2], bkd=self.bkd()
        )

        recovered_mean, recovered_cov = factor.to_moments()

        np.testing.assert_allclose(
            self.bkd().to_numpy(recovered_mean),
            self.bkd().to_numpy(mean),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            self.bkd().to_numpy(recovered_cov),
            self.bkd().to_numpy(cov),
            rtol=1e-6,
        )

    def test_multiply_same_scope(self) -> None:
        """Test multiplying factors with same scope."""
        # Two Gaussians with same variables
        mean1 = self.bkd().asarray(np.array([0.0]))
        cov1 = self.bkd().asarray(np.array([[1.0]]))
        mean2 = self.bkd().asarray(np.array([2.0]))
        cov2 = self.bkd().asarray(np.array([[2.0]]))

        f1 = GaussianFactor.from_moments(
            mean1, cov1, var_ids=[0], nvars_per_var=[1], bkd=self.bkd()
        )
        f2 = GaussianFactor.from_moments(
            mean2, cov2, var_ids=[0], nvars_per_var=[1], bkd=self.bkd()
        )

        product = f1.multiply(f2)

        # Product precision = 1/1 + 1/2 = 1.5
        # Product variance = 1/1.5 = 2/3
        # Product mean = (2/3) * (0/1 + 2/2) = (2/3) * 1 = 2/3
        mean_prod, cov_prod = product.to_moments()
        expected_var = 2.0 / 3.0
        expected_mean = 2.0 / 3.0

        np.testing.assert_allclose(
            float(self.bkd().to_numpy(cov_prod)[0, 0]),
            expected_var,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            float(self.bkd().to_numpy(mean_prod)[0]),
            expected_mean,
            rtol=1e-6,
        )

    def test_multiply_different_scope(self) -> None:
        """Test multiplying factors with different scopes."""
        mean1 = self.bkd().asarray(np.array([1.0]))
        cov1 = self.bkd().asarray(np.array([[1.0]]))
        mean2 = self.bkd().asarray(np.array([2.0]))
        cov2 = self.bkd().asarray(np.array([[1.0]]))

        f1 = GaussianFactor.from_moments(
            mean1, cov1, var_ids=[0], nvars_per_var=[1], bkd=self.bkd()
        )
        f2 = GaussianFactor.from_moments(
            mean2, cov2, var_ids=[1], nvars_per_var=[1], bkd=self.bkd()
        )

        product = f1.multiply(f2)

        # Product should be over both variables
        self.assertEqual(set(product.var_ids()), {0, 1})
        self.assertEqual(product.total_dims(), 2)

        # Variables should be independent
        mean_prod, cov_prod = product.to_moments()
        cov_np = self.bkd().to_numpy(cov_prod)
        self.assertAlmostEqual(cov_np[0, 1], 0.0, places=6)

    def test_marginalize_vars(self) -> None:
        """Test marginalizing out variables."""
        # Joint over two independent variables
        mean = self.bkd().asarray(np.array([1.0, 2.0]))
        cov = self.bkd().asarray(np.array([[1.0, 0.0], [0.0, 2.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0, 1], nvars_per_var=[1, 1], bkd=self.bkd()
        )

        # Marginalize out variable 1
        marginal = factor.marginalize_vars([1])

        self.assertEqual(marginal.var_ids(), [0])
        mean_marg, cov_marg = marginal.to_moments()

        np.testing.assert_allclose(self.bkd().to_numpy(mean_marg), [1.0], rtol=1e-6)
        np.testing.assert_allclose(self.bkd().to_numpy(cov_marg), [[1.0]], rtol=1e-6)

    def test_condition_vars(self) -> None:
        """Test conditioning on variables."""
        # Joint Gaussian
        mean = self.bkd().asarray(np.array([0.0, 0.0]))
        cov = self.bkd().asarray(np.array([[1.0, 0.5], [0.5, 1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0, 1], nvars_per_var=[1, 1], bkd=self.bkd()
        )

        # Condition on variable 1 = 1.0
        value = self.bkd().asarray(np.array([1.0]))
        conditional = factor.condition_vars([1], value)

        self.assertEqual(conditional.var_ids(), [0])

        # Conditional mean: 0 + 0.5 * 1 = 0.5
        # Conditional var: 1 - 0.5^2 = 0.75
        mean_cond, cov_cond = conditional.to_moments()

        np.testing.assert_allclose(self.bkd().to_numpy(mean_cond), [0.5], rtol=1e-6)
        np.testing.assert_allclose(self.bkd().to_numpy(cov_cond), [[0.75]], rtol=1e-6)

    def test_expand_scope(self) -> None:
        """Test expanding factor scope."""
        mean = self.bkd().asarray(np.array([1.0]))
        cov = self.bkd().asarray(np.array([[1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0], nvars_per_var=[1], bkd=self.bkd()
        )

        expanded = factor.expand_scope(
            target_var_ids=[0, 1], target_nvars_per_var=[1, 1]
        )

        self.assertEqual(set(expanded.var_ids()), {0, 1})
        self.assertEqual(expanded.total_dims(), 2)

        # New variable should have zero precision (vacuous information)
        # Verify the precision matrix structure
        prec = self.bkd().to_numpy(expanded.canonical().precision())

        # Original variable keeps its precision
        np.testing.assert_allclose(prec[0, 0], 1.0, rtol=1e-6)

        # New variable has zero precision (vacuous)
        self.assertAlmostEqual(prec[1, 1], 0.0, places=6)

        # Cross terms are zero
        self.assertAlmostEqual(prec[0, 1], 0.0, places=6)
        self.assertAlmostEqual(prec[1, 0], 0.0, places=6)

    def test_repr(self) -> None:
        """Test string representation."""
        mean = self.bkd().asarray(np.array([0.0]))
        cov = self.bkd().asarray(np.array([[1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[5], nvars_per_var=[1], bkd=self.bkd()
        )

        repr_str = repr(factor)
        self.assertIn("GaussianFactor", repr_str)
        self.assertIn("5", repr_str)


# NumPy backend tests
class TestGaussianFactorNumpy(TestGaussianFactorBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestGaussianFactorTorch(TestGaussianFactorBase[torch.Tensor]):
    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
