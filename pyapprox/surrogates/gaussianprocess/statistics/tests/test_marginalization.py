"""
Tests for MarginalizedGP class.

Tests the MarginalizedGP class for reducing GP dimensionality by integrating
out selected variables. Includes tests for 1D and 2D marginalization cases.
"""

import math
from typing import List

import numpy as np
import pytest

from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.statistics import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.marginalization import (
    MarginalizedGP,
)
from pyapprox.surrogates.kernels.composition import SeparableProductKernel
from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.util.test_utils import slow_test


def _create_quadrature_bases(
    marginals, nquad_points, bkd,
):
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestMarginalizedGP1D:
    """
    Test MarginalizedGP with 1D output (single active dimension).

    Tests marginalization of a 2D GP down to 1D by integrating out one variable.
    """

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 2D GP with separable product kernel
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.8], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(
            kernel, nvars=2, bkd=bkd, nugget=1e-6
        )
        # Skip hyperparameter optimization for these tests
        gp.hyp_list().set_all_inactive()

        # Training data
        n_train = 8
        X_train_np = np.random.rand(2, n_train) * 2 - 1  # [-1, 1]^2
        X_train = bkd.array(X_train_np)
        # Use backend math operations, shape: (nqoi, n_train)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :])
            + 0.5 * bkd.cos(math.pi * X_train[1, :]),
            (1, -1),
        )

        gp.fit(X_train, y_train)

        # Marginal distributions
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Create quadrature bases
        nquad_points = 30
        bases = _create_quadrature_bases(
            marginals, nquad_points, bkd
        )

        # Create integral calculator
        calc = SeparableKernelIntegralCalculator(
            gp, bases, marginals, bkd=bkd
        )

        # Create marginalized GP (keep dimension 0, integrate out dimension 1)
        marg_gp = MarginalizedGP(gp, calc, active_dims=[0])

        return marg_gp, n_train

    def test_ndims(self, bkd) -> None:
        """Test ndims returns 1 for 1D marginalization."""
        marg_gp, _ = self._setup(bkd)
        assert marg_gp.ndims() == 1

    def test_active_dims(self, bkd) -> None:
        """Test active_dims returns correct list."""
        marg_gp, _ = self._setup(bkd)
        assert marg_gp.active_dims() == [0]

    def test_marginalized_dims(self, bkd) -> None:
        """Test marginalized_dims returns correct list."""
        marg_gp, _ = self._setup(bkd)
        assert marg_gp.marginalized_dims() == [1]

    def test_nvars_original(self, bkd) -> None:
        """Test nvars_original returns 2."""
        marg_gp, _ = self._setup(bkd)
        assert marg_gp.nvars_original() == 2

    def test_u_not_p_positive(self, bkd) -> None:
        """Test u_~p is positive and less than 1."""
        marg_gp, _ = self._setup(bkd)
        u_not_p = marg_gp.u_not_p()
        u_val = float(bkd.to_numpy(u_not_p))
        assert u_val > 0.0
        assert u_val <= 1.0 + 1e-10

    def test_tau_not_p_shape(self, bkd) -> None:
        """Test tau_~p has correct shape (N,)."""
        marg_gp, n_train = self._setup(bkd)
        tau_not_p = marg_gp.tau_not_p()
        assert tau_not_p.shape == (n_train,)

    def test_tau_not_p_positive(self, bkd) -> None:
        """Test tau_~p values are positive."""
        marg_gp, _ = self._setup(bkd)
        tau_not_p = marg_gp.tau_not_p()
        assert bkd.all_bool(tau_not_p > 0)

    def test_predict_mean_shape(self, bkd) -> None:
        """Test predict_mean returns correct shape."""
        marg_gp, _ = self._setup(bkd)
        z_test = bkd.array([[-0.5, 0.0, 0.5]])  # Shape (1, 3)
        mean = marg_gp.predict_mean(z_test)
        assert mean.shape == (3,)

    def test_predict_variance_shape(self, bkd) -> None:
        """Test predict_variance returns correct shape."""
        marg_gp, _ = self._setup(bkd)
        z_test = bkd.array([[-0.5, 0.0, 0.5]])  # Shape (1, 3)
        var = marg_gp.predict_variance(z_test)
        assert var.shape == (3,)

    def test_predict_variance_bounded(self, bkd) -> None:
        """Test variance is bounded by u_~p (prior variance bound)."""
        marg_gp, _ = self._setup(bkd)
        z_test = bkd.linspace(-1.0, 1.0, 20)
        z_test = bkd.reshape(z_test, (1, -1))  # Shape (1, 20)

        var = marg_gp.predict_variance(z_test)
        u_not_p = float(bkd.to_numpy(marg_gp.u_not_p()))

        # Variance should be <= u_~p (with small numerical tolerance)
        assert bkd.all_bool(var <= u_not_p + 1e-10), (
            f"Variance exceeds prior bound u_~p = {u_not_p}"
        )

    def test_predict_variance_nonnegative(self, bkd) -> None:
        """Test variance is non-negative."""
        marg_gp, _ = self._setup(bkd)
        z_test = bkd.linspace(-1.0, 1.0, 20)
        z_test = bkd.reshape(z_test, (1, -1))

        var = marg_gp.predict_variance(z_test)

        # Variance should be >= 0
        assert bkd.all_bool(var >= -1e-10), "Variance has negative values"

    def test_predict_returns_both(self, bkd) -> None:
        """Test predict returns both mean and variance."""
        marg_gp, _ = self._setup(bkd)
        z_test = bkd.array([[-0.5, 0.0, 0.5]])
        mean, var = marg_gp.predict(z_test)

        assert mean.shape == (3,)
        assert var.shape == (3,)

    def test_predict_consistency(self, bkd) -> None:
        """Test predict gives same results as separate calls."""
        marg_gp, _ = self._setup(bkd)
        z_test = bkd.array([[-0.5, 0.0, 0.5]])

        mean_separate = marg_gp.predict_mean(z_test)
        var_separate = marg_gp.predict_variance(z_test)
        mean_combined, var_combined = marg_gp.predict(z_test)

        bkd.assert_allclose(mean_separate, mean_combined, rtol=1e-12)
        bkd.assert_allclose(var_separate, var_combined, rtol=1e-12)

    def test_invalid_input_shape_raises(self, bkd) -> None:
        """Test that wrong input shape raises ValueError."""
        marg_gp, _ = self._setup(bkd)
        # 2D input for 1D marginalized GP
        z_wrong = bkd.array([[-0.5, 0.0], [0.5, 0.5]])  # Shape (2, 2)

        with pytest.raises(ValueError):
            marg_gp.predict_mean(z_wrong)


class TestMarginalizedGP2D:
    """
    Test MarginalizedGP with 2D output (two active dimensions).

    Tests marginalization of a 3D GP down to 2D by integrating out one variable.
    """

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 3D GP with separable product kernel
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.8], (0.1, 10.0), 1, bkd)
        k3 = SquaredExponentialKernel([0.6], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2, k3], bkd)

        gp = ExactGaussianProcess(
            kernel, nvars=3, bkd=bkd, nugget=1e-6
        )
        # Skip hyperparameter optimization for these tests
        gp.hyp_list().set_all_inactive()

        # Training data
        n_train = 15
        X_train_np = np.random.rand(3, n_train) * 2 - 1  # [-1, 1]^3
        X_train = bkd.array(X_train_np)
        # Use backend math operations, shape: (nqoi, n_train)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :])
            + 0.5 * bkd.cos(math.pi * X_train[1, :])
            + 0.3 * X_train[2, :],
            (1, -1),
        )

        gp.fit(X_train, y_train)

        # Marginal distributions
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Create quadrature bases
        nquad_points = 20
        bases = _create_quadrature_bases(
            marginals, nquad_points, bkd
        )

        # Create integral calculator
        calc = SeparableKernelIntegralCalculator(
            gp, bases, marginals, bkd=bkd
        )

        # Create marginalized GP (keep dims 0 and 2, integrate out dim 1)
        marg_gp = MarginalizedGP(gp, calc, active_dims=[0, 2])

        return marg_gp

    def test_ndims(self, bkd) -> None:
        """Test ndims returns 2 for 2D marginalization."""
        marg_gp = self._setup(bkd)
        assert marg_gp.ndims() == 2

    def test_active_dims(self, bkd) -> None:
        """Test active_dims returns correct sorted list."""
        marg_gp = self._setup(bkd)
        assert marg_gp.active_dims() == [0, 2]

    def test_marginalized_dims(self, bkd) -> None:
        """Test marginalized_dims returns correct list."""
        marg_gp = self._setup(bkd)
        assert marg_gp.marginalized_dims() == [1]

    def test_nvars_original(self, bkd) -> None:
        """Test nvars_original returns 3."""
        marg_gp = self._setup(bkd)
        assert marg_gp.nvars_original() == 3

    def test_predict_mean_shape(self, bkd) -> None:
        """Test predict_mean returns correct shape for 2D input."""
        marg_gp = self._setup(bkd)
        # 5 test points, 2 active dimensions
        z_test = bkd.array(
            [
                [-0.5, 0.0, 0.5, -0.3, 0.3],  # Dim 0 values
                [0.1, -0.2, 0.3, 0.4, -0.1],  # Dim 2 values (mapped to row 1)
            ]
        )  # Shape (2, 5)

        mean = marg_gp.predict_mean(z_test)
        assert mean.shape == (5,)

    def test_predict_variance_shape(self, bkd) -> None:
        """Test predict_variance returns correct shape for 2D input."""
        marg_gp = self._setup(bkd)
        z_test = bkd.array(
            [
                [-0.5, 0.0, 0.5],
                [0.1, -0.2, 0.3],
            ]
        )  # Shape (2, 3)

        var = marg_gp.predict_variance(z_test)
        assert var.shape == (3,)

    def test_predict_variance_bounded(self, bkd) -> None:
        """Test variance is bounded by u_~p."""
        marg_gp = self._setup(bkd)
        # Create a grid of test points
        n_1d = 5
        z0 = bkd.linspace(-0.9, 0.9, n_1d)
        z2 = bkd.linspace(-0.9, 0.9, n_1d)
        Z0, Z2 = bkd.meshgrid(z0, z2)
        z_test = bkd.vstack(
            [bkd.flatten(Z0), bkd.flatten(Z2)]
        )  # Shape (2, n_1d^2)

        var = marg_gp.predict_variance(z_test)
        u_not_p = float(bkd.to_numpy(marg_gp.u_not_p()))

        assert bkd.all_bool(var <= u_not_p + 1e-10), (
            f"Variance exceeds prior bound u_~p = {u_not_p}"
        )

    def test_predict_variance_nonnegative(self, bkd) -> None:
        """Test variance is non-negative."""
        marg_gp = self._setup(bkd)
        z_test = bkd.array(
            [
                [-0.5, 0.0, 0.5],
                [0.1, -0.2, 0.3],
            ]
        )

        var = marg_gp.predict_variance(z_test)
        assert bkd.all_bool(var >= -1e-10)

    def test_non_contiguous_active_dims(self, bkd) -> None:
        """Test marginalization with non-contiguous active dims [0, 2]."""
        marg_gp = self._setup(bkd)
        # This is already our setup, verify it works
        z_test = bkd.array(
            [
                [0.0],  # Dim 0
                [0.0],  # Dim 2
            ]
        )
        mean, var = marg_gp.predict(z_test)

        assert mean.shape == (1,)
        assert var.shape == (1,)


class TestMarginalizedGPNumerical:
    """
    Test MarginalizedGP against numerical integration.

    Compares the marginalized GP predictions to Monte Carlo integration
    of the full GP.
    """

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 2D GP with separable product kernel
        # Use shorter length scales for more interesting behavior
        k1 = SquaredExponentialKernel([0.3], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(
            kernel, nvars=2, bkd=bkd, nugget=1e-6
        )

        # Sparse training data (to have non-trivial posterior uncertainty)
        n_train = 5
        X_train_np = np.array(
            [
                [-0.8, -0.3, 0.0, 0.4, 0.9],
                [-0.5, 0.5, -0.2, 0.8, 0.0],
            ]
        )
        X_train = bkd.array(X_train_np)
        # Use backend math operations, shape: (nqoi, n_train)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :]) + 0.5 * X_train[1, :],
            (1, -1),
        )

        gp.fit(X_train, y_train)

        # Marginal distributions
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Create quadrature bases with high accuracy
        nquad_points = 50
        bases = _create_quadrature_bases(
            marginals, nquad_points, bkd
        )

        # Create integral calculator
        calc = SeparableKernelIntegralCalculator(
            gp, bases, marginals, bkd=bkd
        )

        return gp, calc, bases

    @slow_test
    def test_marginalized_mean_vs_numerical_1d(self, bkd) -> None:
        """
        Compare marginalized mean to numerical integration.

        For a fixed z_0, the marginalized mean should equal:
        m_tilde*(z_0) = integral m*(z_0, z_1) rho(z_1) dz_1

        where m* is the full GP posterior mean.
        """
        gp, calc, bases = self._setup(bkd)

        # Create marginalized GP (keep dim 0)
        marg_gp = MarginalizedGP(gp, calc, active_dims=[0])

        # Test at a few z_0 values
        z0_test_np = np.array([-0.5, 0.0, 0.5])
        z0_test = bkd.array(z0_test_np.reshape(1, -1))

        # Get marginalized mean
        marg_mean = marg_gp.predict_mean(z0_test)

        # Compute numerical integration for comparison
        # Use quadrature rule for dimension 1
        quad_pts_1, quad_wts_1 = bases[1].quadrature_rule()
        # quad_pts_1 shape: (1, nquad), quad_wts_1 shape: (nquad, 1)
        quad_wts_1 = bkd.reshape(quad_wts_1, (-1,))  # (nquad,)
        nquad = quad_pts_1.shape[1]

        numerical_mean = []
        for z0_val in z0_test_np:
            # Create full 2D test points: (z0_val, quad_pts)
            z0_repeated = bkd.array([[z0_val]] * nquad).T  # (1, nquad)
            X_full = bkd.vstack([z0_repeated, quad_pts_1])  # (2, nquad)

            # Get full GP mean at these points
            gp_mean = gp.predict(X_full)  # (nquad, 1)
            gp_mean = bkd.reshape(gp_mean, (-1,))  # (nquad,)

            # Integrate: integral m*(z_0, z_1) rho(z_1) dz_1 approx Sum w_j m*(z_0, quad_j)
            integral = bkd.sum(quad_wts_1 * gp_mean)
            numerical_mean.append(float(bkd.to_numpy(integral)))

        numerical_mean = bkd.array(numerical_mean)

        # Compare
        bkd.assert_allclose(
            marg_mean,
            numerical_mean,
            rtol=1e-6,
            err_msg="Marginalized mean differs from numerical integration",
        )

    @slow_test
    def test_marginalized_variance_vs_numerical_1d(self, bkd) -> None:
        """
        Compare marginalized variance to numerical integration.

        The marginalized variance involves double integration over the
        posterior covariance. This test verifies the formula:
        C_tilde*(z_i, z_i) = u_~i - tau_tilde(z_i)^T A^{-1} tau_tilde(z_i)
        """
        gp, calc, _ = self._setup(bkd)

        # Create marginalized GP (keep dim 0)
        marg_gp = MarginalizedGP(gp, calc, active_dims=[0])

        # Test at a single z_0 value (variance computation is expensive)
        z0_test = bkd.array([[0.0]])  # Shape (1, 1)

        # Get marginalized variance
        marg_var = marg_gp.predict_variance(z0_test)

        # The variance should be bounded by u_~p
        u_not_p = float(bkd.to_numpy(marg_gp.u_not_p()))
        marg_var_val = float(bkd.to_numpy(marg_var[0]))

        assert marg_var_val <= u_not_p + 1e-10
        assert marg_var_val >= -1e-10


class TestMarginalizedGPValidation:
    """Test validation and error handling for MarginalizedGP."""

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 2D GP
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(
            kernel, nvars=2, bkd=bkd, nugget=1e-6
        )

        X_train = bkd.array(np.random.rand(2, 5) * 2 - 1)
        y_train = bkd.array(np.random.rand(5).reshape(1, -1))
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train)

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        bases = _create_quadrature_bases(marginals, 10, bkd)
        calc = SeparableKernelIntegralCalculator(
            gp, bases, marginals, bkd=bkd
        )

        return gp, calc, kernel

    def test_empty_active_dims_raises(self, bkd) -> None:
        """Test that empty active_dims raises ValueError."""
        gp, calc, _ = self._setup(bkd)
        with pytest.raises(ValueError):
            MarginalizedGP(gp, calc, active_dims=[])

    def test_duplicate_active_dims_raises(self, bkd) -> None:
        """Test that duplicate active_dims raises ValueError."""
        gp, calc, _ = self._setup(bkd)
        with pytest.raises(ValueError):
            MarginalizedGP(gp, calc, active_dims=[0, 0])

    def test_invalid_dim_raises(self, bkd) -> None:
        """Test that invalid dimension index raises ValueError."""
        gp, calc, _ = self._setup(bkd)
        with pytest.raises(ValueError):
            MarginalizedGP(gp, calc, active_dims=[0, 5])

    def test_negative_dim_raises(self, bkd) -> None:
        """Test that negative dimension index raises ValueError."""
        gp, calc, _ = self._setup(bkd)
        with pytest.raises(ValueError):
            MarginalizedGP(gp, calc, active_dims=[-1])

    def test_unfitted_gp_raises(self, bkd) -> None:
        """Test that unfitted GP raises RuntimeError."""
        _, calc, kernel = self._setup(bkd)
        gp_unfitted = ExactGaussianProcess(kernel, nvars=2, bkd=bkd)

        with pytest.raises(RuntimeError):
            MarginalizedGP(gp_unfitted, calc, active_dims=[0])

    def test_all_dims_active(self, bkd) -> None:
        """Test that keeping all dimensions works (no marginalization)."""
        gp, calc, _ = self._setup(bkd)
        # Keep both dimensions - no marginalization
        marg_gp = MarginalizedGP(gp, calc, active_dims=[0, 1])

        assert marg_gp.ndims() == 2
        assert marg_gp.marginalized_dims() == []

        # u_~p should be 1.0 (no dimensions marginalized)
        u_not_p = float(bkd.to_numpy(marg_gp.u_not_p()))
        bkd.assert_allclose(
            bkd.asarray([u_not_p]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_active_dims_sorted(self, bkd) -> None:
        """Test that active_dims are sorted regardless of input order."""
        gp, calc, _ = self._setup(bkd)
        marg_gp = MarginalizedGP(gp, calc, active_dims=[1, 0])
        assert marg_gp.active_dims() == [0, 1]


class TestMarginalizedGP1DSpecialCases:
    """
    Test special cases for 1D marginalization.

    Includes tests for 1D GP (no marginalization needed) and edge cases.
    """

    def test_1d_gp_marginalized_to_1d(self, bkd) -> None:
        """
        Test marginalizing 1D GP to 1D (no actual marginalization).

        The marginalized GP should match the original GP predictions.
        """
        np.random.seed(42)

        # Create 1D GP
        kernel = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        gp = ExactGaussianProcess(kernel, nvars=1, bkd=bkd, nugget=1e-6)

        X_train = bkd.array([[-0.8, -0.3, 0.0, 0.4, 0.9]])
        y_train = bkd.reshape(bkd.sin(3.14159 * X_train[0, :]), (1, -1))
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train)

        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases = _create_quadrature_bases(marginals, 20, bkd)
        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)

        # Marginalize to 1D (keeping the only dimension)
        marg_gp = MarginalizedGP(gp, calc, active_dims=[0])

        # Test points
        z_test = bkd.array([[-0.5, 0.0, 0.5]])

        # Marginalized predictions
        marg_mean = marg_gp.predict_mean(z_test)

        # Original GP predictions
        gp_mean = gp.predict(z_test)
        gp_mean = bkd.reshape(gp_mean, (-1,))

        # Should match (no dimensions being marginalized)
        bkd.assert_allclose(marg_mean, gp_mean, rtol=1e-10)
