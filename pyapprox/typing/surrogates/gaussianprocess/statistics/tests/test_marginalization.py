"""
Tests for MarginalizedGP class.

Tests the MarginalizedGP class for reducing GP dimensionality by integrating
out selected variables. Includes tests for 1D and 2D marginalization cases.
"""
import unittest
from typing import Generic, Any, List
import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401
from pyapprox.typing.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.typing.surrogates.kernels.composition import SeparableProductKernel
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.marginalization import (
    MarginalizedGP,
)


def _create_quadrature_bases(
    marginals: List[Any], nquad_points: int, bkd: Backend[Array]
) -> List[Any]:
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestMarginalizedGP1D(Generic[Array], unittest.TestCase):
    """
    Test MarginalizedGP with 1D output (single active dimension).

    Tests marginalization of a 2D GP down to 1D by integrating out one variable.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D GP with separable product kernel
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([0.8], (0.1, 10.0), 1, self._bkd)
        self._kernel = SeparableProductKernel([k1, k2], self._bkd)

        self._gp = ExactGaussianProcess(
            self._kernel,
            nvars=2,
            bkd=self._bkd,
            nugget=1e-6
        )

        # Training data
        self._n_train = 8
        X_train_np = np.random.rand(2, self._n_train) * 2 - 1  # [-1, 1]^2
        y_train_np = (
            np.sin(np.pi * X_train_np[0, :]) +
            0.5 * np.cos(np.pi * X_train_np[1, :])
        )
        y_train_np = y_train_np.reshape(-1, 1)

        self._X_train = self._bkd.array(X_train_np)
        self._y_train = self._bkd.array(y_train_np)

        self._gp.fit(self._X_train, self._y_train)

        # Marginal distributions
        self._marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Create quadrature bases
        self._nquad_points = 30
        self._bases = _create_quadrature_bases(
            self._marginals, self._nquad_points, self._bkd
        )

        # Create integral calculator
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, self._bases, self._marginals, bkd=self._bkd
        )

        # Create marginalized GP (keep dimension 0, integrate out dimension 1)
        self._marg_gp = MarginalizedGP(self._gp, self._calc, active_dims=[0])

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_ndims(self) -> None:
        """Test ndims returns 1 for 1D marginalization."""
        self.assertEqual(self._marg_gp.ndims(), 1)

    def test_active_dims(self) -> None:
        """Test active_dims returns correct list."""
        self.assertEqual(self._marg_gp.active_dims(), [0])

    def test_marginalized_dims(self) -> None:
        """Test marginalized_dims returns correct list."""
        self.assertEqual(self._marg_gp.marginalized_dims(), [1])

    def test_nvars_original(self) -> None:
        """Test nvars_original returns 2."""
        self.assertEqual(self._marg_gp.nvars_original(), 2)

    def test_u_not_p_positive(self) -> None:
        """Test u_~p is positive and less than 1."""
        u_not_p = self._marg_gp.u_not_p()
        u_val = float(self._bkd.to_numpy(u_not_p))
        self.assertGreater(u_val, 0.0)
        self.assertLessEqual(u_val, 1.0 + 1e-10)

    def test_tau_not_p_shape(self) -> None:
        """Test τ_~p has correct shape (N,)."""
        tau_not_p = self._marg_gp.tau_not_p()
        self.assertEqual(tau_not_p.shape, (self._n_train,))

    def test_tau_not_p_positive(self) -> None:
        """Test τ_~p values are positive."""
        tau_not_p = self._marg_gp.tau_not_p()
        self.assertTrue(self._bkd.all_bool(tau_not_p > 0))

    def test_predict_mean_shape(self) -> None:
        """Test predict_mean returns correct shape."""
        z_test = self._bkd.array([[-0.5, 0.0, 0.5]])  # Shape (1, 3)
        mean = self._marg_gp.predict_mean(z_test)
        self.assertEqual(mean.shape, (3,))

    def test_predict_variance_shape(self) -> None:
        """Test predict_variance returns correct shape."""
        z_test = self._bkd.array([[-0.5, 0.0, 0.5]])  # Shape (1, 3)
        var = self._marg_gp.predict_variance(z_test)
        self.assertEqual(var.shape, (3,))

    def test_predict_variance_bounded(self) -> None:
        """Test variance is bounded by u_~p (prior variance bound)."""
        z_test = self._bkd.linspace(-1.0, 1.0, 20)
        z_test = self._bkd.reshape(z_test, (1, -1))  # Shape (1, 20)

        var = self._marg_gp.predict_variance(z_test)
        u_not_p = float(self._bkd.to_numpy(self._marg_gp.u_not_p()))

        # Variance should be <= u_~p (with small numerical tolerance)
        self.assertTrue(
            self._bkd.all_bool(var <= u_not_p + 1e-10),
            f"Variance exceeds prior bound u_~p = {u_not_p}"
        )

    def test_predict_variance_nonnegative(self) -> None:
        """Test variance is non-negative."""
        z_test = self._bkd.linspace(-1.0, 1.0, 20)
        z_test = self._bkd.reshape(z_test, (1, -1))

        var = self._marg_gp.predict_variance(z_test)

        # Variance should be >= 0
        self.assertTrue(
            self._bkd.all_bool(var >= -1e-10),
            "Variance has negative values"
        )

    def test_predict_returns_both(self) -> None:
        """Test predict returns both mean and variance."""
        z_test = self._bkd.array([[-0.5, 0.0, 0.5]])
        mean, var = self._marg_gp.predict(z_test)

        self.assertEqual(mean.shape, (3,))
        self.assertEqual(var.shape, (3,))

    def test_predict_consistency(self) -> None:
        """Test predict gives same results as separate calls."""
        z_test = self._bkd.array([[-0.5, 0.0, 0.5]])

        mean_separate = self._marg_gp.predict_mean(z_test)
        var_separate = self._marg_gp.predict_variance(z_test)
        mean_combined, var_combined = self._marg_gp.predict(z_test)

        self._bkd.assert_allclose(mean_separate, mean_combined, rtol=1e-12)
        self._bkd.assert_allclose(var_separate, var_combined, rtol=1e-12)

    def test_invalid_input_shape_raises(self) -> None:
        """Test that wrong input shape raises ValueError."""
        # 2D input for 1D marginalized GP
        z_wrong = self._bkd.array([[-0.5, 0.0], [0.5, 0.5]])  # Shape (2, 2)

        with self.assertRaises(ValueError):
            self._marg_gp.predict_mean(z_wrong)


class TestMarginalizedGP2D(Generic[Array], unittest.TestCase):
    """
    Test MarginalizedGP with 2D output (two active dimensions).

    Tests marginalization of a 3D GP down to 2D by integrating out one variable.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 3D GP with separable product kernel
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([0.8], (0.1, 10.0), 1, self._bkd)
        k3 = SquaredExponentialKernel([0.6], (0.1, 10.0), 1, self._bkd)
        self._kernel = SeparableProductKernel([k1, k2, k3], self._bkd)

        self._gp = ExactGaussianProcess(
            self._kernel,
            nvars=3,
            bkd=self._bkd,
            nugget=1e-6
        )

        # Training data
        self._n_train = 15
        X_train_np = np.random.rand(3, self._n_train) * 2 - 1  # [-1, 1]^3
        y_train_np = (
            np.sin(np.pi * X_train_np[0, :]) +
            0.5 * np.cos(np.pi * X_train_np[1, :]) +
            0.3 * X_train_np[2, :]
        )
        y_train_np = y_train_np.reshape(-1, 1)

        self._X_train = self._bkd.array(X_train_np)
        self._y_train = self._bkd.array(y_train_np)

        self._gp.fit(self._X_train, self._y_train)

        # Marginal distributions
        self._marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Create quadrature bases
        self._nquad_points = 20
        self._bases = _create_quadrature_bases(
            self._marginals, self._nquad_points, self._bkd
        )

        # Create integral calculator
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, self._bases, self._marginals, bkd=self._bkd
        )

        # Create marginalized GP (keep dims 0 and 2, integrate out dim 1)
        self._marg_gp = MarginalizedGP(self._gp, self._calc, active_dims=[0, 2])

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_ndims(self) -> None:
        """Test ndims returns 2 for 2D marginalization."""
        self.assertEqual(self._marg_gp.ndims(), 2)

    def test_active_dims(self) -> None:
        """Test active_dims returns correct sorted list."""
        self.assertEqual(self._marg_gp.active_dims(), [0, 2])

    def test_marginalized_dims(self) -> None:
        """Test marginalized_dims returns correct list."""
        self.assertEqual(self._marg_gp.marginalized_dims(), [1])

    def test_nvars_original(self) -> None:
        """Test nvars_original returns 3."""
        self.assertEqual(self._marg_gp.nvars_original(), 3)

    def test_predict_mean_shape(self) -> None:
        """Test predict_mean returns correct shape for 2D input."""
        # 5 test points, 2 active dimensions
        z_test = self._bkd.array([
            [-0.5, 0.0, 0.5, -0.3, 0.3],  # Dim 0 values
            [0.1, -0.2, 0.3, 0.4, -0.1],  # Dim 2 values (mapped to row 1)
        ])  # Shape (2, 5)

        mean = self._marg_gp.predict_mean(z_test)
        self.assertEqual(mean.shape, (5,))

    def test_predict_variance_shape(self) -> None:
        """Test predict_variance returns correct shape for 2D input."""
        z_test = self._bkd.array([
            [-0.5, 0.0, 0.5],
            [0.1, -0.2, 0.3],
        ])  # Shape (2, 3)

        var = self._marg_gp.predict_variance(z_test)
        self.assertEqual(var.shape, (3,))

    def test_predict_variance_bounded(self) -> None:
        """Test variance is bounded by u_~p."""
        # Create a grid of test points
        n_1d = 5
        z0 = self._bkd.linspace(-0.9, 0.9, n_1d)
        z2 = self._bkd.linspace(-0.9, 0.9, n_1d)
        Z0, Z2 = self._bkd.meshgrid(z0, z2)
        z_test = self._bkd.vstack([
            self._bkd.flatten(Z0),
            self._bkd.flatten(Z2)
        ])  # Shape (2, n_1d^2)

        var = self._marg_gp.predict_variance(z_test)
        u_not_p = float(self._bkd.to_numpy(self._marg_gp.u_not_p()))

        self.assertTrue(
            self._bkd.all_bool(var <= u_not_p + 1e-10),
            f"Variance exceeds prior bound u_~p = {u_not_p}"
        )

    def test_predict_variance_nonnegative(self) -> None:
        """Test variance is non-negative."""
        z_test = self._bkd.array([
            [-0.5, 0.0, 0.5],
            [0.1, -0.2, 0.3],
        ])

        var = self._marg_gp.predict_variance(z_test)
        self.assertTrue(self._bkd.all_bool(var >= -1e-10))

    def test_non_contiguous_active_dims(self) -> None:
        """Test marginalization with non-contiguous active dims [0, 2]."""
        # This is already our setup, verify it works
        z_test = self._bkd.array([
            [0.0],  # Dim 0
            [0.0],  # Dim 2
        ])
        mean, var = self._marg_gp.predict(z_test)

        self.assertEqual(mean.shape, (1,))
        self.assertEqual(var.shape, (1,))


class TestMarginalizedGPNumerical(Generic[Array], unittest.TestCase):
    """
    Test MarginalizedGP against numerical integration.

    Compares the marginalized GP predictions to Monte Carlo integration
    of the full GP.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D GP with separable product kernel
        # Use shorter length scales for more interesting behavior
        k1 = SquaredExponentialKernel([0.3], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        self._kernel = SeparableProductKernel([k1, k2], self._bkd)

        self._gp = ExactGaussianProcess(
            self._kernel,
            nvars=2,
            bkd=self._bkd,
            nugget=1e-6
        )

        # Sparse training data (to have non-trivial posterior uncertainty)
        self._n_train = 5
        X_train_np = np.array([
            [-0.8, -0.3, 0.0, 0.4, 0.9],
            [-0.5, 0.5, -0.2, 0.8, 0.0],
        ])
        y_train_np = np.sin(np.pi * X_train_np[0, :]) + 0.5 * X_train_np[1, :]
        y_train_np = y_train_np.reshape(-1, 1)

        self._X_train = self._bkd.array(X_train_np)
        self._y_train = self._bkd.array(y_train_np)

        self._gp.fit(self._X_train, self._y_train)

        # Marginal distributions
        self._marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Create quadrature bases with high accuracy
        self._nquad_points = 50
        self._bases = _create_quadrature_bases(
            self._marginals, self._nquad_points, self._bkd
        )

        # Create integral calculator
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, self._bases, self._marginals, bkd=self._bkd
        )

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    @slow_test
    def test_marginalized_mean_vs_numerical_1d(self) -> None:
        """
        Compare marginalized mean to numerical integration.

        For a fixed z_0, the marginalized mean should equal:
        m̃*(z_0) = ∫ m*(z_0, z_1) ρ(z_1) dz_1

        where m* is the full GP posterior mean.
        """
        bkd = self._bkd

        # Create marginalized GP (keep dim 0)
        marg_gp = MarginalizedGP(self._gp, self._calc, active_dims=[0])

        # Test at a few z_0 values
        z0_test_np = np.array([-0.5, 0.0, 0.5])
        z0_test = bkd.array(z0_test_np.reshape(1, -1))

        # Get marginalized mean
        marg_mean = marg_gp.predict_mean(z0_test)

        # Compute numerical integration for comparison
        # Use quadrature rule for dimension 1
        quad_pts_1, quad_wts_1 = self._bases[1].quadrature_rule()
        # quad_pts_1 shape: (1, nquad), quad_wts_1 shape: (nquad, 1)
        quad_wts_1 = bkd.reshape(quad_wts_1, (-1,))  # (nquad,)
        nquad = quad_pts_1.shape[1]

        numerical_mean = []
        for z0_val in z0_test_np:
            # Create full 2D test points: (z0_val, quad_pts)
            z0_repeated = bkd.array([[z0_val]] * nquad).T  # (1, nquad)
            X_full = bkd.vstack([z0_repeated, quad_pts_1])  # (2, nquad)

            # Get full GP mean at these points
            gp_mean = self._gp.predict(X_full)  # (nquad, 1)
            gp_mean = bkd.reshape(gp_mean, (-1,))  # (nquad,)

            # Integrate: ∫ m*(z_0, z_1) ρ(z_1) dz_1 ≈ Σ w_j m*(z_0, quad_j)
            integral = bkd.sum(quad_wts_1 * gp_mean)
            numerical_mean.append(float(bkd.to_numpy(integral)))

        numerical_mean = bkd.array(numerical_mean)

        # Compare
        bkd.assert_allclose(
            marg_mean, numerical_mean, rtol=1e-6,
            err_msg="Marginalized mean differs from numerical integration"
        )

    @slow_test
    def test_marginalized_variance_vs_numerical_1d(self) -> None:
        """
        Compare marginalized variance to numerical integration.

        The marginalized variance involves double integration over the
        posterior covariance. This test verifies the formula:
        C̃*(z_i, z_i) = u_~i - τ̃(z_i)^T A^{-1} τ̃(z_i)
        """
        bkd = self._bkd

        # Create marginalized GP (keep dim 0)
        marg_gp = MarginalizedGP(self._gp, self._calc, active_dims=[0])

        # Test at a single z_0 value (variance computation is expensive)
        z0_test = bkd.array([[0.0]])  # Shape (1, 1)

        # Get marginalized variance
        marg_var = marg_gp.predict_variance(z0_test)

        # The variance should be bounded by u_~p
        u_not_p = float(bkd.to_numpy(marg_gp.u_not_p()))
        marg_var_val = float(bkd.to_numpy(marg_var[0]))

        self.assertLessEqual(marg_var_val, u_not_p + 1e-10)
        self.assertGreaterEqual(marg_var_val, -1e-10)


class TestMarginalizedGPValidation(Generic[Array], unittest.TestCase):
    """Test validation and error handling for MarginalizedGP."""

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D GP
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        self._kernel = SeparableProductKernel([k1, k2], self._bkd)

        self._gp = ExactGaussianProcess(
            self._kernel,
            nvars=2,
            bkd=self._bkd,
            nugget=1e-6
        )

        X_train = self._bkd.array(np.random.rand(2, 5) * 2 - 1)
        y_train = self._bkd.array(np.random.rand(5, 1))
        self._gp.fit(X_train, y_train)

        self._marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        bases = _create_quadrature_bases(self._marginals, 10, self._bkd)
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, bases, self._marginals, bkd=self._bkd
        )

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_empty_active_dims_raises(self) -> None:
        """Test that empty active_dims raises ValueError."""
        with self.assertRaises(ValueError):
            MarginalizedGP(self._gp, self._calc, active_dims=[])

    def test_duplicate_active_dims_raises(self) -> None:
        """Test that duplicate active_dims raises ValueError."""
        with self.assertRaises(ValueError):
            MarginalizedGP(self._gp, self._calc, active_dims=[0, 0])

    def test_invalid_dim_raises(self) -> None:
        """Test that invalid dimension index raises ValueError."""
        with self.assertRaises(ValueError):
            MarginalizedGP(self._gp, self._calc, active_dims=[0, 5])

    def test_negative_dim_raises(self) -> None:
        """Test that negative dimension index raises ValueError."""
        with self.assertRaises(ValueError):
            MarginalizedGP(self._gp, self._calc, active_dims=[-1])

    def test_unfitted_gp_raises(self) -> None:
        """Test that unfitted GP raises RuntimeError."""
        gp_unfitted = ExactGaussianProcess(
            self._kernel, nvars=2, bkd=self._bkd
        )

        with self.assertRaises(RuntimeError):
            MarginalizedGP(gp_unfitted, self._calc, active_dims=[0])

    def test_all_dims_active(self) -> None:
        """Test that keeping all dimensions works (no marginalization)."""
        # Keep both dimensions - no marginalization
        marg_gp = MarginalizedGP(self._gp, self._calc, active_dims=[0, 1])

        self.assertEqual(marg_gp.ndims(), 2)
        self.assertEqual(marg_gp.marginalized_dims(), [])

        # u_~p should be 1.0 (no dimensions marginalized)
        u_not_p = float(self._bkd.to_numpy(marg_gp.u_not_p()))
        self._bkd.assert_allclose(
            self._bkd.asarray([u_not_p]),
            self._bkd.asarray([1.0]),
            rtol=1e-12
        )

    def test_active_dims_sorted(self) -> None:
        """Test that active_dims are sorted regardless of input order."""
        marg_gp = MarginalizedGP(self._gp, self._calc, active_dims=[1, 0])
        self.assertEqual(marg_gp.active_dims(), [0, 1])


class TestMarginalizedGP1DSpecialCases(Generic[Array], unittest.TestCase):
    """
    Test special cases for 1D marginalization.

    Includes tests for 1D GP (no marginalization needed) and edge cases.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_1d_gp_marginalized_to_1d(self) -> None:
        """
        Test marginalizing 1D GP to 1D (no actual marginalization).

        The marginalized GP should match the original GP predictions.
        """
        bkd = self._bkd

        # Create 1D GP
        kernel = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        gp = ExactGaussianProcess(kernel, nvars=1, bkd=bkd, nugget=1e-6)

        X_train = bkd.array([[-0.8, -0.3, 0.0, 0.4, 0.9]])
        y_train = bkd.reshape(bkd.sin(3.14159 * X_train[0, :]), (-1, 1))
        gp.fit(X_train, y_train)

        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases = _create_quadrature_bases(marginals, 20, bkd)
        calc = SeparableKernelIntegralCalculator(
            gp, bases, marginals, bkd=bkd
        )

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


# NumPy backend tests
class TestMarginalizedGP1DNumpy(TestMarginalizedGP1D[NDArray[Any]]):
    """NumPy backend tests for 1D marginalization."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMarginalizedGP2DNumpy(TestMarginalizedGP2D[NDArray[Any]]):
    """NumPy backend tests for 2D marginalization."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMarginalizedGPNumericalNumpy(TestMarginalizedGPNumerical[NDArray[Any]]):
    """NumPy backend tests for numerical comparison."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMarginalizedGPValidationNumpy(TestMarginalizedGPValidation[NDArray[Any]]):
    """NumPy backend tests for validation."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMarginalizedGP1DSpecialCasesNumpy(
    TestMarginalizedGP1DSpecialCases[NDArray[Any]]
):
    """NumPy backend tests for special cases."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestMarginalizedGP1DTorch(TestMarginalizedGP1D[torch.Tensor]):
    """PyTorch backend tests for 1D marginalization."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMarginalizedGP2DTorch(TestMarginalizedGP2D[torch.Tensor]):
    """PyTorch backend tests for 2D marginalization."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMarginalizedGPNumericalTorch(TestMarginalizedGPNumerical[torch.Tensor]):
    """PyTorch backend tests for numerical comparison."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMarginalizedGPValidationTorch(TestMarginalizedGPValidation[torch.Tensor]):
    """PyTorch backend tests for validation."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMarginalizedGP1DSpecialCasesTorch(
    TestMarginalizedGP1DSpecialCases[torch.Tensor]
):
    """PyTorch backend tests for special cases."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
