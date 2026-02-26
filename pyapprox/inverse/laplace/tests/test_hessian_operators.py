"""
Tests for Hessian operators.
"""

import pytest

import numpy as np

from pyapprox.inverse.laplace.hessian_operators import (
    ApplyNegLogLikelihoodHessian,
    PriorConditionedHessianMatVec,
)
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator


class TestApplyNegLogLikelihoodHessianBase:
    """Base test class for ApplyNegLogLikelihoodHessian."""

    def _make_op(self, bkd):
        """Create operator for tests."""
        nvars = 3

        # Define a simple quadratic Hessian (constant)
        hess_matrix = bkd.asarray(
            [
                [2.0, 0.5, 0.1],
                [0.5, 3.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )

        def apply_hess(sample, vec):
            return hess_matrix @ vec

        op = ApplyNegLogLikelihoodHessian(apply_hess, nvars, bkd)
        sample = bkd.zeros((nvars, 1))
        op.set_sample(sample)
        return op, nvars, hess_matrix

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        op, nvars, _ = self._make_op(bkd)
        assert op.nvars() == nvars

    def test_apply_single_vector(self, bkd) -> None:
        """Test applying to a single vector."""
        op, nvars, _ = self._make_op(bkd)
        vec = bkd.asarray([[1.0], [0.0], [0.0]])
        result = op.apply(vec)
        assert result.shape == (nvars, 1)

    def test_apply_multiple_vectors(self, bkd) -> None:
        """Test applying to multiple vectors."""
        op, nvars, _ = self._make_op(bkd)
        vecs = bkd.eye(nvars)
        result = op.apply(vecs)
        assert result.shape == (nvars, nvars)

    def test_apply_negates_hessian(self, bkd) -> None:
        """Test that apply returns negative of Hessian."""
        op, nvars, hess_matrix = self._make_op(bkd)
        vecs = bkd.eye(nvars)
        result = op.apply(vecs)
        result_np = bkd.to_numpy(result)
        hess_np = bkd.to_numpy(hess_matrix)
        np.testing.assert_array_almost_equal(result_np, -hess_np)

    def test_set_sample_required(self, bkd) -> None:
        """Test that set_sample must be called before apply."""

        def apply_hess(sample, vec):
            return vec

        op = ApplyNegLogLikelihoodHessian(apply_hess, 2, bkd)
        vec = bkd.ones((2, 1))
        with pytest.raises(RuntimeError):
            op.apply(vec)

    def test_wrong_sample_shape_raises(self, bkd) -> None:
        """Test wrong sample shape raises error."""
        op, _, _ = self._make_op(bkd)
        bad_sample = bkd.zeros((5, 1))
        with pytest.raises(ValueError):
            op.set_sample(bad_sample)


class TestPriorConditionedHessianMatVecBase:
    """Base test class for PriorConditionedHessianMatVec."""

    def _make_op(self, bkd):
        """Create operator for tests."""
        nvars = 3

        # Prior covariance
        prior_cov = bkd.asarray(
            [
                [1.0, 0.3, 0.1],
                [0.3, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )
        prior_sqrt = DenseCholeskyCovarianceOperator(prior_cov, bkd)

        # Misfit Hessian
        hess_matrix = bkd.asarray(
            [
                [2.0, 0.5, 0.1],
                [0.5, 3.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )

        def apply_hess(vecs):
            return hess_matrix @ vecs

        op = PriorConditionedHessianMatVec(prior_sqrt, apply_hess)
        return op, nvars, prior_cov, hess_matrix

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct value."""
        op, nvars, _, _ = self._make_op(bkd)
        assert op.nvars() == nvars

    def test_apply_shape(self, bkd) -> None:
        """Test apply returns correct shape."""
        op, nvars, _, _ = self._make_op(bkd)
        vecs = bkd.eye(nvars)
        result = op.apply(vecs)
        assert result.shape == (nvars, nvars)

    def test_apply_transpose_equals_apply(self, bkd) -> None:
        """Test that apply_transpose equals apply (symmetric)."""
        op, _, _, _ = self._make_op(bkd)
        vecs = bkd.asarray([[1.0], [2.0], [3.0]])
        result = op.apply(vecs)
        result_t = op.apply_transpose(vecs)
        result_np = bkd.to_numpy(result)
        result_t_np = bkd.to_numpy(result_t)
        np.testing.assert_array_almost_equal(result_np, result_t_np)

    def test_result_is_symmetric(self, bkd) -> None:
        """Test that L^T H L is symmetric."""
        op, nvars, _, _ = self._make_op(bkd)
        identity = bkd.eye(nvars)
        result = op.apply(identity)
        result_np = bkd.to_numpy(result)
        np.testing.assert_array_almost_equal(result_np, result_np.T)

    def test_matches_explicit_computation(self, bkd) -> None:
        """Test against explicit L^T @ H @ L computation."""
        op, nvars, prior_cov, hess_matrix = self._make_op(bkd)
        L = bkd.cholesky(prior_cov)
        expected = L.T @ hess_matrix @ L

        identity = bkd.eye(nvars)
        result = op.apply(identity)

        result_np = bkd.to_numpy(result)
        expected_np = bkd.to_numpy(expected)
        np.testing.assert_array_almost_equal(result_np, expected_np)
