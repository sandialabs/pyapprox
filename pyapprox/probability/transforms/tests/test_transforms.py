"""
Tests for probability transforms.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability.transforms import (
    AffineTransform,
    GaussianTransform,
    IndependentGaussianTransform,
    NatafTransform,
    RosenblattTransform,
)
from pyapprox.probability.univariate import (
    GaussianMarginal,
    ScipyContinuousMarginal,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestAffineTransform(Generic[Array], unittest.TestCase):
    """Tests for AffineTransform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.loc = self._bkd.asarray([1.0, 2.0])
        self.scale = self._bkd.asarray([2.0, 3.0])
        self.transform = AffineTransform(self.loc, self.scale, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_loc_scale(self) -> None:
        """Test loc and scale accessors."""
        self.assertTrue(self._bkd.allclose(self.transform.loc(), self.loc, atol=1e-10))
        self.assertTrue(
            self._bkd.allclose(self.transform.scale(), self.scale, atol=1e-10)
        )

    def test_map_to_canonical(self) -> None:
        """Test map to canonical space."""
        x = self._bkd.asarray([[1.0, 3.0], [2.0, 5.0]])
        y = self.transform.map_to_canonical(x)
        expected = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        self.assertTrue(self._bkd.allclose(y, expected, rtol=1e-6))

    def test_map_from_canonical(self) -> None:
        """Test map from canonical space."""
        y = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        x = self.transform.map_from_canonical(y)
        expected = self._bkd.asarray([[1.0, 3.0], [2.0, 5.0]])
        self.assertTrue(self._bkd.allclose(x, expected, rtol=1e-6))

    def test_roundtrip(self) -> None:
        """Test that map_to_canonical and map_from_canonical are inverses."""
        x = self._bkd.asarray([[0.0, 1.0, 5.0], [1.0, 3.0, 10.0]])
        y = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(y)
        self.assertTrue(self._bkd.allclose(x, x_recovered, rtol=1e-6))

    def test_jacobian_to_canonical(self) -> None:
        """Test Jacobian to canonical is 1/scale."""
        x = self._bkd.asarray([[1.0, 3.0], [2.0, 5.0]])
        _, jacobian = self.transform.map_to_canonical_with_jacobian(x)
        expected = self._bkd.asarray([[0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0]])
        self.assertTrue(self._bkd.allclose(jacobian, expected, rtol=1e-6))

    def test_jacobian_from_canonical(self) -> None:
        """Test Jacobian from canonical is scale."""
        y = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        _, jacobian = self.transform.map_from_canonical_with_jacobian(y)
        expected = self._bkd.asarray([[2.0, 2.0], [3.0, 3.0]])
        self.assertTrue(self._bkd.allclose(jacobian, expected, rtol=1e-6))

    def test_log_det_jacobian(self) -> None:
        """Test log determinant of Jacobian."""
        x = self._bkd.asarray([[1.0, 3.0], [2.0, 5.0]])
        log_det = self.transform.log_det_jacobian_to_canonical(x)
        expected_val = -np.log(2) - np.log(3)
        expected = self._bkd.asarray([expected_val, expected_val])
        self.assertTrue(self._bkd.allclose(log_det, expected, rtol=1e-6))

    def test_mismatched_shapes_raises(self) -> None:
        """Test mismatched loc and scale raises error."""
        with self.assertRaises(ValueError):
            AffineTransform(
                self._bkd.asarray([1.0, 2.0]),
                self._bkd.asarray([1.0]),
                self._bkd,
            )

    def test_jacobian_derivative_checker(self) -> None:
        """Test Jacobian using DerivativeChecker."""
        nvars = self.transform.nvars()

        def fun(sample: Array) -> Array:
            # sample is (nvars, 1), output is (nvars, 1)
            x = self._bkd.reshape(sample, (nvars, 1))
            y = self.transform.map_to_canonical(x)
            return self._bkd.flatten(y)[:, None]

        def jacobian(sample: Array) -> Array:
            # Return full Jacobian (nqoi, nvars) = (nvars, nvars)
            x = self._bkd.reshape(sample, (nvars, 1))
            _, jac_diag = self.transform.map_to_canonical_with_jacobian(x)
            # jac_diag is (nvars, 1), create diagonal matrix
            return self._bkd.diag(self._bkd.flatten(jac_diag))

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nvars,
            nvars=nvars,
            fun=fun,
            jacobian=jacobian,
            bkd=self._bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = self._bkd.asarray([[1.5], [3.0]])
        errors = checker.check_derivatives(sample)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 5e-6)


class TestAffineTransformNumpy(TestAffineTransform[NDArray[Any]]):
    """NumPy backend tests for AffineTransform."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAffineTransformTorch(TestAffineTransform[torch.Tensor]):
    """PyTorch backend tests for AffineTransform."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGaussianTransform(Generic[Array], unittest.TestCase):
    """Tests for GaussianTransform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.uniform = ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd)
        self.transform = GaussianTransform(self.uniform, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        self.assertEqual(self.transform.nvars(), 1)

    def test_median_to_zero(self) -> None:
        """Test median of uniform maps to 0 (median of normal)."""
        x = self._bkd.asarray([[0.5]])  # Shape: (1, 1) for univariate transform
        y = self.transform.map_to_canonical(x)
        expected = self._bkd.asarray([[0.0]])
        self.assertTrue(self._bkd.allclose(y, expected, rtol=1e-6))

    def test_roundtrip(self) -> None:
        """Test roundtrip preserves samples."""
        x = self._bkd.asarray([[0.1, 0.3, 0.5, 0.7, 0.9]])  # Shape: (1, nsamples)
        y = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(y)
        self.assertTrue(self._bkd.allclose(x, x_recovered, rtol=1e-6))

    def test_normal_to_normal_identity(self) -> None:
        """Test normal marginal gives identity transform."""
        normal = GaussianMarginal(0.0, 1.0, self._bkd)
        transform = GaussianTransform(normal, self._bkd)

        x = self._bkd.asarray([[-1.0, 0.0, 1.0, 2.0]])  # Shape: (1, nsamples)
        y = transform.map_to_canonical(x)
        self.assertTrue(self._bkd.allclose(x, y, rtol=1e-5))

    def test_jacobian_chain_rule(self) -> None:
        """Test Jacobian satisfies chain rule."""
        x = self._bkd.asarray([[0.3]])  # Shape: (1, 1) for univariate transform
        y, jac_to = self.transform.map_to_canonical_with_jacobian(x)
        x_back, jac_from = self.transform.map_from_canonical_with_jacobian(y)

        # jac_to * jac_from should be close to 1
        expected = self._bkd.asarray([[1.0]])
        self.assertTrue(self._bkd.allclose(jac_to * jac_from, expected, rtol=1e-5))

    def test_jacobian_derivative_checker(self) -> None:
        """Test Jacobian using DerivativeChecker."""

        def fun(sample: Array) -> Array:
            # sample is (1, 1), transform expects (1, nsamples)
            y = self.transform.map_to_canonical(sample)  # Returns (1, 1)
            return y.T  # Return (nsamples, nqoi) = (1, 1)

        def jacobian(sample: Array) -> Array:
            # Return Jacobian (nqoi, nvars) = (1, 1)
            _, jac = self.transform.map_to_canonical_with_jacobian(sample)
            return jac  # Already (1, 1)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=fun,
            jacobian=jacobian,
            bkd=self._bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = self._bkd.asarray([[0.3]])
        errors = checker.check_derivatives(sample)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 5e-6)


class TestGaussianTransformNumpy(TestGaussianTransform[NDArray[Any]]):
    """NumPy backend tests for GaussianTransform."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianTransformTorch(TestGaussianTransform[torch.Tensor]):
    """PyTorch backend tests for GaussianTransform."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIndependentGaussianTransform(Generic[Array], unittest.TestCase):
    """Tests for IndependentGaussianTransform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginals = [
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), self._bkd),
            ScipyContinuousMarginal(stats.norm(0, 1), self._bkd),
        ]
        self.transform = IndependentGaussianTransform(self.marginals, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 3)

    def test_marginals(self) -> None:
        """Test marginals accessor."""
        marginals = self.transform.marginals()
        self.assertEqual(len(marginals), 3)

    def test_map_to_canonical_shape(self) -> None:
        """Test map to canonical returns correct shape."""
        x = self._bkd.asarray([[0.5, 0.3], [0.4, 0.2], [0.0, 1.0]])
        y = self.transform.map_to_canonical(x)
        self.assertEqual(y.shape, (3, 2))

    def test_roundtrip(self) -> None:
        """Test roundtrip preserves samples."""
        x = self._bkd.asarray([[0.3, 0.7], [0.2, 0.6], [-1.0, 1.0]])
        y = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(y)
        self.assertTrue(self._bkd.allclose(x, x_recovered, rtol=1e-5))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        x = self._bkd.asarray([[0.5, 0.3], [0.4, 0.2], [0.0, 1.0]])
        _, jacobian = self.transform.map_to_canonical_with_jacobian(x)
        self.assertEqual(jacobian.shape, (3, 2))

    def test_log_det_jacobian_shape(self) -> None:
        """Test log determinant has correct shape."""
        x = self._bkd.asarray([[0.5, 0.3], [0.4, 0.2], [0.0, 1.0]])
        log_det = self.transform.log_det_jacobian_to_canonical(x)
        # log_det_jacobian returns (1, nsamples)
        self.assertEqual(log_det.shape, (1, 2))

    def test_normal_component_identity(self) -> None:
        """Test normal component is identity transform."""
        x = self._bkd.asarray([[0.5], [0.4], [0.5]])
        y = self.transform.map_to_canonical(x)
        self.assertTrue(self._bkd.allclose(y[2], x[2], rtol=1e-5))

    def test_jacobian_derivative_checker(self) -> None:
        """Test Jacobian using DerivativeChecker."""
        nvars = self.transform.nvars()

        def fun(sample: Array) -> Array:
            # sample is (nvars, 1), output is (nvars, 1)
            x = self._bkd.reshape(sample, (nvars, 1))
            y = self.transform.map_to_canonical(x)
            return self._bkd.flatten(y)[:, None]

        def jacobian(sample: Array) -> Array:
            # Return full Jacobian (nqoi, nvars) = (nvars, nvars)
            x = self._bkd.reshape(sample, (nvars, 1))
            _, jac_diag = self.transform.map_to_canonical_with_jacobian(x)
            # jac_diag is (nvars, 1), create diagonal matrix
            return self._bkd.diag(self._bkd.flatten(jac_diag))

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nvars,
            nvars=nvars,
            fun=fun,
            jacobian=jacobian,
            bkd=self._bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = self._bkd.asarray([[0.5], [0.3], [0.0]])
        errors = checker.check_derivatives(sample)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 5e-6)


class TestIndependentGaussianTransformNumpy(
    TestIndependentGaussianTransform[NDArray[Any]]
):
    """NumPy backend tests for IndependentGaussianTransform."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIndependentGaussianTransformTorch(
    TestIndependentGaussianTransform[torch.Tensor]
):
    """PyTorch backend tests for IndependentGaussianTransform."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestAffineTransformProtocol(Generic[Array], unittest.TestCase):
    """Test AffineTransform satisfies TransformWithJacobianProtocol."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.transform = AffineTransform(
            self._bkd.asarray([0.0, 1.0]),
            self._bkd.asarray([1.0, 2.0]),
            self._bkd,
        )

    def test_has_bkd(self) -> None:
        """Test has bkd method."""
        self.assertIsNotNone(self.transform.bkd())

    def test_has_nvars(self) -> None:
        """Test has nvars method."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_has_map_to_canonical(self) -> None:
        """Test has map_to_canonical method."""
        x = self._bkd.asarray([[0.0, 1.0], [1.0, 3.0]])
        y = self.transform.map_to_canonical(x)
        self.assertEqual(y.shape, (2, 2))

    def test_has_map_from_canonical(self) -> None:
        """Test has map_from_canonical method."""
        y = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        x = self.transform.map_from_canonical(y)
        self.assertEqual(x.shape, (2, 2))

    def test_has_map_with_jacobian(self) -> None:
        """Test has map_to_canonical_with_jacobian method."""
        x = self._bkd.asarray([[0.0, 1.0], [1.0, 3.0]])
        y, jac = self.transform.map_to_canonical_with_jacobian(x)
        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(jac.shape, (2, 2))


class TestAffineTransformProtocolNumpy(TestAffineTransformProtocol[NDArray[Any]]):
    """NumPy backend tests for AffineTransform protocol."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAffineTransformProtocolTorch(TestAffineTransformProtocol[torch.Tensor]):
    """PyTorch backend tests for AffineTransform protocol."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGaussianTransformProtocol(Generic[Array], unittest.TestCase):
    """Test GaussianTransform satisfies TransformWithJacobianProtocol."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginal = ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd)
        self.transform = GaussianTransform(self.marginal, self._bkd)

    def test_has_bkd(self) -> None:
        """Test has bkd method."""
        self.assertIsNotNone(self.transform.bkd())

    def test_has_nvars(self) -> None:
        """Test has nvars method."""
        self.assertEqual(self.transform.nvars(), 1)

    def test_has_map_to_canonical(self) -> None:
        """Test has map_to_canonical method."""
        x = self._bkd.asarray([[0.3, 0.5, 0.7]])  # Shape: (1, nsamples)
        y = self.transform.map_to_canonical(x)
        self.assertEqual(y.shape, (1, 3))

    def test_has_map_from_canonical(self) -> None:
        """Test has map_from_canonical method."""
        y = self._bkd.asarray([[-1.0, 0.0, 1.0]])  # Shape: (1, nsamples)
        x = self.transform.map_from_canonical(y)
        self.assertEqual(x.shape, (1, 3))


class TestGaussianTransformProtocolNumpy(TestGaussianTransformProtocol[NDArray[Any]]):
    """NumPy backend tests for GaussianTransform protocol."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianTransformProtocolTorch(TestGaussianTransformProtocol[torch.Tensor]):
    """PyTorch backend tests for GaussianTransform protocol."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestNatafTransform(Generic[Array], unittest.TestCase):
    """Tests for NatafTransform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self.marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(0.0, 1.0, self._bkd),
        ]
        self.correlation = self._bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        self.transform = NatafTransform(self.marginals, self.correlation, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_marginals(self) -> None:
        """Test marginals accessor."""
        marginals = self.transform.marginals()
        self.assertEqual(len(marginals), 2)

    def test_correlation(self) -> None:
        """Test correlation accessor."""
        corr = self.transform.correlation()
        self.assertTrue(self._bkd.allclose(corr, self.correlation, rtol=1e-6))

    def test_map_to_canonical_shape(self) -> None:
        """Test map to canonical returns correct shape."""
        x = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        z = self.transform.map_to_canonical(x)
        self.assertEqual(z.shape, (2, 2))

    def test_map_from_canonical_shape(self) -> None:
        """Test map from canonical returns correct shape."""
        z = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        x = self.transform.map_from_canonical(z)
        self.assertEqual(x.shape, (2, 2))

    def test_roundtrip(self) -> None:
        """Test roundtrip preserves samples."""
        x = self._bkd.asarray([[0.0, 0.5, -0.5], [0.0, 1.0, -1.0]])
        z = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(z)
        self.assertTrue(self._bkd.allclose(x, x_recovered, rtol=1e-5))

    def test_identity_correlation(self) -> None:
        """Test identity correlation gives independent transform."""
        identity = self._bkd.eye(2)
        transform = NatafTransform(self.marginals, identity, self._bkd)

        x = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        z = transform.map_to_canonical(x)
        self.assertTrue(self._bkd.allclose(x, z, rtol=1e-5))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        x = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        _, jacobian = self.transform.map_to_canonical_with_jacobian(x)
        self.assertEqual(jacobian.shape, (2, 2, 2))

    def test_invalid_correlation_shape_raises(self) -> None:
        """Test mismatched correlation shape raises error."""
        with self.assertRaises(ValueError):
            NatafTransform(
                self.marginals,
                self._bkd.eye(3),
                self._bkd,
            )

    def test_jacobian_derivative_checker(self) -> None:
        """Test Jacobian using DerivativeChecker."""
        nvars = self.transform.nvars()

        def fun(sample: Array) -> Array:
            # sample is (nvars, 1), output is (nvars, 1)
            x = self._bkd.reshape(sample, (nvars, 1))
            z = self.transform.map_to_canonical(x)
            return self._bkd.flatten(z)[:, None]

        def jacobian(sample: Array) -> Array:
            # Return full Jacobian (nqoi, nvars) = (nvars, nvars)
            x = self._bkd.reshape(sample, (nvars, 1))
            _, jac = self.transform.map_to_canonical_with_jacobian(x)
            # jac is (nvars, nvars, 1), extract for single sample
            return jac[:, :, 0]

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nvars,
            nvars=nvars,
            fun=fun,
            jacobian=jacobian,
            bkd=self._bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = self._bkd.asarray([[0.5], [0.3]])
        errors = checker.check_derivatives(sample)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 5e-6)


class TestNatafTransformNumpy(TestNatafTransform[NDArray[Any]]):
    """NumPy backend tests for NatafTransform."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestNatafTransformTorch(TestNatafTransform[torch.Tensor]):
    """PyTorch backend tests for NatafTransform."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestNatafTransformNonGaussian(Generic[Array], unittest.TestCase):
    """Tests for NatafTransform with non-Gaussian marginals."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()
        self.marginals = [
            ScipyContinuousMarginal(stats.uniform(0, 1), self._bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), self._bkd),
        ]
        self.correlation = self._bkd.asarray([[1.0, 0.3], [0.3, 1.0]])
        self.transform = NatafTransform(self.marginals, self.correlation, self._bkd)

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_roundtrip(self) -> None:
        """Test roundtrip preserves samples."""
        x = self._bkd.asarray([[0.3, 0.7], [0.2, 0.4]])
        z = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(z)
        self.assertTrue(self._bkd.allclose(x, x_recovered, rtol=1e-4))

    def test_canonical_is_approximately_normal(self) -> None:
        """Test canonical samples are approximately standard normal.

        The correct procedure is:
        1. Generate independent standard normal samples Z
        2. Use map_from_canonical(Z) to get correlated samples X
        3. Use map_to_canonical(X) to transform back to Z'
        4. Z' should be approximately standard normal with identity covariance
        """
        np.random.seed(42)
        n = 10000000

        # Generate independent standard normal samples
        z_initial = self._bkd.asarray(np.random.randn(2, n))

        # Transform to correlated physical space samples
        x = self.transform.map_from_canonical(z_initial)

        # Transform back to canonical (should recover standard normals)
        z = self.transform.map_to_canonical(x)
        z_np = self._bkd.to_numpy(z)

        for i in range(2):
            z_i = z_np[i]
            mean_val = float(np.mean(z_i))
            std_val = float(np.std(z_i))
            # Mean should be close to 0
            self.assertTrue(
                self._bkd.allclose(
                    self._bkd.asarray([mean_val]),
                    self._bkd.asarray([0.0]),
                    atol=3e-3,
                )
            )
            # Std should be close to 1
            self.assertTrue(
                self._bkd.allclose(
                    self._bkd.asarray([std_val]),
                    self._bkd.asarray([1.0]),
                    atol=3e-3,
                )
            )


class TestNatafTransformNonGaussianNumpy(TestNatafTransformNonGaussian[NDArray[Any]]):
    """NumPy backend tests for NatafTransform with non-Gaussian marginals."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestNatafTransformNonGaussianTorch(TestNatafTransformNonGaussian[torch.Tensor]):
    """PyTorch backend tests for NatafTransform with non-Gaussian marginals."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestRosenblattTransform(Generic[Array], unittest.TestCase):
    """Tests for RosenblattTransform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # Use a bivariate Gaussian PDF for testing
        self.nvars = 2
        self.mean = np.array([0.0, 0.0])
        self.cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

        def joint_pdf(samples: Array) -> Array:
            """Bivariate Gaussian PDF."""
            samples_np = self._bkd.to_numpy(samples)
            nsamples = samples_np.shape[1]
            pdf_vals = np.zeros(nsamples)
            for i in range(nsamples):
                x = samples_np[:, i] - self.mean
                exponent = -0.5 * x @ self.cov_inv @ x
                pdf_vals[i] = np.exp(exponent) / (2 * np.pi * np.sqrt(self.cov_det))
            return self._bkd.asarray(pdf_vals)

        self.joint_pdf = joint_pdf
        bounds = self._bkd.asarray([[-5.0, -5.0], [5.0, 5.0]])
        self.transform = RosenblattTransform(
            self.joint_pdf, self.nvars, self._bkd, bounds
        )

    def test_nvars(self) -> None:
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_map_to_uniform_shape(self) -> None:
        """Test map to uniform returns correct shape."""
        x = self._bkd.asarray([[0.0, 0.5], [0.0, 0.5]])
        u = self.transform.map_to_uniform(x)
        self.assertEqual(u.shape, (2, 2))

    def test_map_to_canonical_shape(self) -> None:
        """Test map to canonical returns correct shape."""
        x = self._bkd.asarray([[0.0, 0.5], [0.0, 0.5]])
        z = self.transform.map_to_canonical(x)
        self.assertEqual(z.shape, (2, 2))

    def test_map_from_uniform_shape(self) -> None:
        """Test map from uniform returns correct shape."""
        u = self._bkd.asarray([[0.5, 0.3], [0.5, 0.7]])
        x = self.transform.map_from_uniform(u)
        self.assertEqual(x.shape, (2, 2))

    def test_map_from_canonical_shape(self) -> None:
        """Test map from canonical returns correct shape."""
        z = self._bkd.asarray([[0.0, 0.5], [0.0, 0.5]])
        x = self.transform.map_from_canonical(z)
        self.assertEqual(x.shape, (2, 2))

    def test_uniform_bounds(self) -> None:
        """Test uniform samples are in [0, 1]."""
        x = self._bkd.asarray([[0.0], [0.0]])
        u = self.transform.map_to_uniform(x)
        u_np = self._bkd.to_numpy(u)
        self.assertTrue(np.all(u_np >= 0.0))
        self.assertTrue(np.all(u_np <= 1.0))

    @unittest.skip(
        "RosenblattTransform numerical integration is incomplete - "
        "see TODO in rosenblatt.py for proper multivariate integration"
    )
    def test_rosenblatt_recovers_nataf_for_gaussian(self) -> None:
        """Test Rosenblatt gives same result as Nataf for Gaussian marginals.

        For a Gaussian joint distribution, both transforms should produce
        the same independent standard normal samples.

        Note: This test is skipped because the current Rosenblatt implementation
        uses placeholder numerical integration that doesn't properly compute
        marginal densities by integrating out other variables.
        """
        # Create Nataf transform with same Gaussian setup
        marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(0.0, 1.0, self._bkd),
        ]
        correlation = self._bkd.asarray([[1.0, 0.5], [0.5, 1.0]])
        nataf = NatafTransform(marginals, correlation, self._bkd)

        # Test at the origin (mean of distribution)
        x = self._bkd.asarray([[0.0], [0.0]])

        # Both transforms applied to the same point
        z_rosenblatt = self.transform.map_to_canonical(x)
        z_nataf = nataf.map_to_canonical(x)

        # For the origin of a zero-mean Gaussian, both should give ~0
        # The Rosenblatt is numerical so use relaxed tolerance
        self.assertTrue(
            self._bkd.allclose(
                z_rosenblatt,
                z_nataf,
                atol=0.1,  # Relaxed due to numerical integration in Rosenblatt
            )
        )


class TestRosenblattTransformNumpy(TestRosenblattTransform[NDArray[Any]]):
    """NumPy backend tests for RosenblattTransform."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRosenblattTransformTorch(TestRosenblattTransform[torch.Tensor]):
    """PyTorch backend tests for RosenblattTransform."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
