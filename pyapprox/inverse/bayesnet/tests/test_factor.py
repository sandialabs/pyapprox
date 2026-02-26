"""
Tests for GaussianFactor with variable ID tracking.
"""

import pytest

import numpy as np

from pyapprox.inverse.bayesnet.factor import GaussianFactor


class TestGaussianFactorBase:
    """Base tests for GaussianFactor."""

    def test_from_moments(self, bkd) -> None:
        """Test creating factor from moments."""
        mean = bkd.asarray(np.array([1.0, 2.0]))
        cov = bkd.asarray(np.array([[1.0, 0.3], [0.3, 1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0], nvars_per_var=[2], bkd=bkd
        )

        assert factor.var_ids() == [0]
        assert factor.nvars_per_var() == [2]
        assert factor.total_dims() == 2

    def test_to_moments(self, bkd) -> None:
        """Test converting back to moments."""
        mean = bkd.asarray(np.array([1.0, 2.0]))
        cov = bkd.asarray(np.array([[1.0, 0.3], [0.3, 1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0], nvars_per_var=[2], bkd=bkd
        )

        recovered_mean, recovered_cov = factor.to_moments()

        np.testing.assert_allclose(
            bkd.to_numpy(recovered_mean),
            bkd.to_numpy(mean),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            bkd.to_numpy(recovered_cov),
            bkd.to_numpy(cov),
            rtol=1e-6,
        )

    def test_multiply_same_scope(self, bkd) -> None:
        """Test multiplying factors with same scope."""
        # Two Gaussians with same variables
        mean1 = bkd.asarray(np.array([0.0]))
        cov1 = bkd.asarray(np.array([[1.0]]))
        mean2 = bkd.asarray(np.array([2.0]))
        cov2 = bkd.asarray(np.array([[2.0]]))

        f1 = GaussianFactor.from_moments(
            mean1, cov1, var_ids=[0], nvars_per_var=[1], bkd=bkd
        )
        f2 = GaussianFactor.from_moments(
            mean2, cov2, var_ids=[0], nvars_per_var=[1], bkd=bkd
        )

        product = f1.multiply(f2)

        # Product precision = 1/1 + 1/2 = 1.5
        # Product variance = 1/1.5 = 2/3
        # Product mean = (2/3) * (0/1 + 2/2) = (2/3) * 1 = 2/3
        mean_prod, cov_prod = product.to_moments()
        expected_var = 2.0 / 3.0
        expected_mean = 2.0 / 3.0

        np.testing.assert_allclose(
            float(bkd.to_numpy(cov_prod)[0, 0]),
            expected_var,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            float(bkd.to_numpy(mean_prod)[0]),
            expected_mean,
            rtol=1e-6,
        )

    def test_multiply_different_scope(self, bkd) -> None:
        """Test multiplying factors with different scopes."""
        mean1 = bkd.asarray(np.array([1.0]))
        cov1 = bkd.asarray(np.array([[1.0]]))
        mean2 = bkd.asarray(np.array([2.0]))
        cov2 = bkd.asarray(np.array([[1.0]]))

        f1 = GaussianFactor.from_moments(
            mean1, cov1, var_ids=[0], nvars_per_var=[1], bkd=bkd
        )
        f2 = GaussianFactor.from_moments(
            mean2, cov2, var_ids=[1], nvars_per_var=[1], bkd=bkd
        )

        product = f1.multiply(f2)

        # Product should be over both variables
        assert set(product.var_ids()) == {0, 1}
        assert product.total_dims() == 2

        # Variables should be independent
        mean_prod, cov_prod = product.to_moments()
        cov_np = bkd.to_numpy(cov_prod)
        assert cov_np[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_marginalize_vars(self, bkd) -> None:
        """Test marginalizing out variables."""
        # Joint over two independent variables
        mean = bkd.asarray(np.array([1.0, 2.0]))
        cov = bkd.asarray(np.array([[1.0, 0.0], [0.0, 2.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0, 1], nvars_per_var=[1, 1], bkd=bkd
        )

        # Marginalize out variable 1
        marginal = factor.marginalize_vars([1])

        assert marginal.var_ids() == [0]
        mean_marg, cov_marg = marginal.to_moments()

        np.testing.assert_allclose(bkd.to_numpy(mean_marg), [1.0], rtol=1e-6)
        np.testing.assert_allclose(bkd.to_numpy(cov_marg), [[1.0]], rtol=1e-6)

    def test_condition_vars(self, bkd) -> None:
        """Test conditioning on variables."""
        # Joint Gaussian
        mean = bkd.asarray(np.array([0.0, 0.0]))
        cov = bkd.asarray(np.array([[1.0, 0.5], [0.5, 1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0, 1], nvars_per_var=[1, 1], bkd=bkd
        )

        # Condition on variable 1 = 1.0
        value = bkd.asarray(np.array([1.0]))
        conditional = factor.condition_vars([1], value)

        assert conditional.var_ids() == [0]

        # Conditional mean: 0 + 0.5 * 1 = 0.5
        # Conditional var: 1 - 0.5^2 = 0.75
        mean_cond, cov_cond = conditional.to_moments()

        np.testing.assert_allclose(bkd.to_numpy(mean_cond), [0.5], rtol=1e-6)
        np.testing.assert_allclose(bkd.to_numpy(cov_cond), [[0.75]], rtol=1e-6)

    def test_expand_scope(self, bkd) -> None:
        """Test expanding factor scope."""
        mean = bkd.asarray(np.array([1.0]))
        cov = bkd.asarray(np.array([[1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[0], nvars_per_var=[1], bkd=bkd
        )

        expanded = factor.expand_scope(
            target_var_ids=[0, 1], target_nvars_per_var=[1, 1]
        )

        assert set(expanded.var_ids()) == {0, 1}
        assert expanded.total_dims() == 2

        # New variable should have zero precision (vacuous information)
        # Verify the precision matrix structure
        prec = bkd.to_numpy(expanded.canonical().precision())

        # Original variable keeps its precision
        np.testing.assert_allclose(prec[0, 0], 1.0, rtol=1e-6)

        # New variable has zero precision (vacuous)
        assert prec[1, 1] == pytest.approx(0.0, abs=1e-6)

        # Cross terms are zero
        assert prec[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert prec[1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        mean = bkd.asarray(np.array([0.0]))
        cov = bkd.asarray(np.array([[1.0]]))

        factor = GaussianFactor.from_moments(
            mean, cov, var_ids=[5], nvars_per_var=[1], bkd=bkd
        )

        repr_str = repr(factor)
        assert "GaussianFactor" in repr_str
        assert "5" in repr_str
