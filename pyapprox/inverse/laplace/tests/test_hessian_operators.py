"""
Tests for Hessian operators.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.inverse.laplace.hessian_operators import (
    ApplyNegLogLikelihoodHessian,
    PriorConditionedHessianMatVec,
)
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestApplyNegLogLikelihoodHessianBase(Generic[Array], unittest.TestCase):
    """Base test class for ApplyNegLogLikelihoodHessian."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        self.nvars = 3

        # Define a simple quadratic Hessian (constant)
        self.hess_matrix = self.bkd().asarray(
            [
                [2.0, 0.5, 0.1],
                [0.5, 3.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )

        def apply_hess(sample: Array, vec: Array) -> Array:
            return self.hess_matrix @ vec

        self.op = ApplyNegLogLikelihoodHessian(apply_hess, self.nvars, self.bkd())
        self.sample = self.bkd().zeros((self.nvars, 1))
        self.op.set_sample(self.sample)

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.op.nvars(), self.nvars)

    def test_apply_single_vector(self) -> None:
        """Test applying to a single vector."""
        vec = self.bkd().asarray([[1.0], [0.0], [0.0]])
        result = self.op.apply(vec)
        self.assertEqual(result.shape, (self.nvars, 1))

    def test_apply_multiple_vectors(self) -> None:
        """Test applying to multiple vectors."""
        vecs = self.bkd().eye(self.nvars)
        result = self.op.apply(vecs)
        self.assertEqual(result.shape, (self.nvars, self.nvars))

    def test_apply_negates_hessian(self) -> None:
        """Test that apply returns negative of Hessian."""
        vecs = self.bkd().eye(self.nvars)
        result = self.op.apply(vecs)
        result_np = self.bkd().to_numpy(result)
        hess_np = self.bkd().to_numpy(self.hess_matrix)
        np.testing.assert_array_almost_equal(result_np, -hess_np)

    def test_set_sample_required(self) -> None:
        """Test that set_sample must be called before apply."""

        def apply_hess(sample: Array, vec: Array) -> Array:
            return vec

        op = ApplyNegLogLikelihoodHessian(apply_hess, 2, self.bkd())
        vec = self.bkd().ones((2, 1))
        with self.assertRaises(RuntimeError):
            op.apply(vec)

    def test_wrong_sample_shape_raises(self) -> None:
        """Test wrong sample shape raises error."""
        bad_sample = self.bkd().zeros((5, 1))
        with self.assertRaises(ValueError):
            self.op.set_sample(bad_sample)


class TestPriorConditionedHessianMatVecBase(Generic[Array], unittest.TestCase):
    """Base test class for PriorConditionedHessianMatVec."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def setUp(self) -> None:
        self.nvars = 3

        # Prior covariance
        self.prior_cov = self.bkd().asarray(
            [
                [1.0, 0.3, 0.1],
                [0.3, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        self.prior_sqrt = DenseCholeskyCovarianceOperator(self.prior_cov, self.bkd())

        # Misfit Hessian
        self.hess_matrix = self.bkd().asarray(
            [
                [2.0, 0.5, 0.1],
                [0.5, 3.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )

        def apply_hess(vecs: Array) -> Array:
            return self.hess_matrix @ vecs

        self.op = PriorConditionedHessianMatVec(self.prior_sqrt, apply_hess)

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        self.assertEqual(self.op.nvars(), self.nvars)

    def test_apply_shape(self) -> None:
        """Test apply returns correct shape."""
        vecs = self.bkd().eye(self.nvars)
        result = self.op.apply(vecs)
        self.assertEqual(result.shape, (self.nvars, self.nvars))

    def test_apply_transpose_equals_apply(self) -> None:
        """Test that apply_transpose equals apply (symmetric)."""
        vecs = self.bkd().asarray([[1.0], [2.0], [3.0]])
        result = self.op.apply(vecs)
        result_t = self.op.apply_transpose(vecs)
        result_np = self.bkd().to_numpy(result)
        result_t_np = self.bkd().to_numpy(result_t)
        np.testing.assert_array_almost_equal(result_np, result_t_np)

    def test_result_is_symmetric(self) -> None:
        """Test that L^T H L is symmetric."""
        identity = self.bkd().eye(self.nvars)
        result = self.op.apply(identity)
        result_np = self.bkd().to_numpy(result)
        np.testing.assert_array_almost_equal(result_np, result_np.T)

    def test_matches_explicit_computation(self) -> None:
        """Test against explicit L^T @ H @ L computation."""
        L = self.bkd().cholesky(self.prior_cov)
        expected = L.T @ self.hess_matrix @ L

        identity = self.bkd().eye(self.nvars)
        result = self.op.apply(identity)

        result_np = self.bkd().to_numpy(result)
        expected_np = self.bkd().to_numpy(expected)
        np.testing.assert_array_almost_equal(result_np, expected_np)


# NumPy backend tests
class TestApplyNegLogLikelihoodHessianNumpy(
    TestApplyNegLogLikelihoodHessianBase[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestPriorConditionedHessianMatVecNumpy(
    TestPriorConditionedHessianMatVecBase[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch backend tests
class TestApplyNegLogLikelihoodHessianTorch(
    TestApplyNegLogLikelihoodHessianBase[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


class TestPriorConditionedHessianMatVecTorch(
    TestPriorConditionedHessianMatVecBase[torch.Tensor]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
