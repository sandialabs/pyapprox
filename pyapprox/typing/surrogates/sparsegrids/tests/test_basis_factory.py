"""Tests for basis factory system.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.probability.univariate import (
    BetaMarginal,
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
    LejaLagrangeFactory,
    PrebuiltBasisFactory,
    create_basis_factories,
    create_bases_from_marginals,
    get_bounds_from_marginal,
    get_transform_from_marginal,
)


# =============================================================================
# GaussLagrangeFactory tests
# =============================================================================


class TestGaussLagrangeFactory(Generic[Array], unittest.TestCase):
    """Tests for GaussLagrangeFactory."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_uniform_marginal_quadrature_in_user_domain(self) -> None:
        """Test that Uniform[0,1] returns quadrature points in [0,1]."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Samples should be in [0, 1], not [-1, 1]
        min_val = float(self._bkd.min(samples))
        max_val = float(self._bkd.max(samples))

        self.assertGreaterEqual(min_val, 0.0)
        self.assertLessEqual(max_val, 1.0)

        # For Gauss-Legendre on [0,1] with 5 points, samples should NOT
        # include exactly 0 or 1 (those are boundary points)
        self.assertGreater(min_val, 0.0)
        self.assertLess(max_val, 1.0)

    def test_uniform_marginal_integration_exact(self) -> None:
        """Test that integration of x over [0,1] gives mean=0.5."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = x over [0, 1] with uniform density
        # Expected mean = 0.5
        x = samples[0, :]
        mean = float(self._bkd.sum(x * weights[:, 0]))

        self._bkd.assert_allclose(
            self._bkd.asarray([mean]),
            self._bkd.asarray([0.5]),
            rtol=1e-12,
        )

    def test_uniform_marginal_variance_exact(self) -> None:
        """Test that integration of (x-0.5)^2 over [0,1] gives variance=1/12."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = (x - 0.5)^2 over [0, 1] with uniform density
        # Expected variance = 1/12
        x = samples[0, :]
        variance = float(self._bkd.sum((x - 0.5) ** 2 * weights[:, 0]))

        self._bkd.assert_allclose(
            self._bkd.asarray([variance]),
            self._bkd.asarray([1.0 / 12.0]),
            rtol=1e-12,
        )

    def test_gaussian_marginal_quadrature_in_user_domain(self) -> None:
        """Test that N(5, 2^2) returns quadrature points centered at 5."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Samples should be centered around mean=5
        sample_mean = float(self._bkd.mean(samples))
        self.assertGreater(sample_mean, 3.0)  # Not centered at 0
        self.assertLess(sample_mean, 7.0)

    def test_gaussian_marginal_integration_mean(self) -> None:
        """Test that integration of x gives mean=5 for N(5, 2^2)."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(10)  # More points for Hermite quadrature
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = x
        # Expected mean = 5
        x = samples[0, :]
        mean = float(self._bkd.sum(x * weights[:, 0]))

        self._bkd.assert_allclose(
            self._bkd.asarray([mean]),
            self._bkd.asarray([5.0]),
            rtol=1e-10,
        )

    def test_gaussian_marginal_integration_variance(self) -> None:
        """Test that integration of (x-5)^2 gives variance=4 for N(5, 2^2)."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(10)
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = (x - 5)^2
        # Expected variance = 4
        x = samples[0, :]
        variance = float(self._bkd.sum((x - 5.0) ** 2 * weights[:, 0]))

        self._bkd.assert_allclose(
            self._bkd.asarray([variance]),
            self._bkd.asarray([4.0]),
            rtol=1e-10,
        )

    def test_beta_marginal_quadrature_in_user_domain(self) -> None:
        """Test that Beta(2, 5) returns quadrature points in [0, 1]."""
        marginal = BetaMarginal(alpha=2.0, beta=5.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Samples should be in [0, 1]
        min_val = float(self._bkd.min(samples))
        max_val = float(self._bkd.max(samples))

        self.assertGreaterEqual(min_val, 0.0)
        self.assertLessEqual(max_val, 1.0)

    def test_beta_marginal_integration_mean(self) -> None:
        """Test that integration of x gives correct mean for Beta(2, 5)."""
        alpha, beta = 2.0, 5.0
        marginal = BetaMarginal(alpha=alpha, beta=beta, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(10)
        samples, weights = basis.quadrature_rule()

        # Expected mean = alpha / (alpha + beta) = 2/7
        expected_mean = alpha / (alpha + beta)
        x = samples[0, :]
        computed_mean = float(self._bkd.sum(x * weights[:, 0]))

        self._bkd.assert_allclose(
            self._bkd.asarray([computed_mean]),
            self._bkd.asarray([expected_mean]),
            rtol=1e-10,
        )

    def test_factory_creates_independent_bases(self) -> None:
        """Test that each create_basis() call returns independent bases."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        basis1 = factory.create_basis()
        basis2 = factory.create_basis()

        # Set different nterms
        basis1.set_nterms(3)
        basis2.set_nterms(7)

        # They should have different numbers of samples
        samples1, _ = basis1.quadrature_rule()
        samples2, _ = basis2.quadrature_rule()

        self.assertEqual(samples1.shape[1], 3)
        self.assertEqual(samples2.shape[1], 7)

    def test_factory_implements_protocol(self) -> None:
        """Test that factory implements BasisFactoryProtocol."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)

        self.assertIsInstance(factory, BasisFactoryProtocol)


class TestGaussLagrangeFactoryNumpy(TestGaussLagrangeFactory[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussLagrangeFactoryTorch(TestGaussLagrangeFactory[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# PrebuiltBasisFactory tests
# =============================================================================


class TestPrebuiltBasisFactory(Generic[Array], unittest.TestCase):
    """Tests for PrebuiltBasisFactory."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_wraps_existing_basis(self) -> None:
        """Test that PrebuiltBasisFactory wraps an existing basis."""
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)

        created = factory.create_basis()
        self.assertIs(created, basis)

    def test_factory_implements_protocol(self) -> None:
        """Test that factory implements BasisFactoryProtocol."""
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)

        self.assertIsInstance(factory, BasisFactoryProtocol)

    def test_returns_same_backend(self) -> None:
        """Test that factory returns same backend as wrapped basis."""
        basis = LegendrePolynomial1D(self._bkd)
        factory = PrebuiltBasisFactory(basis)

        self.assertIs(factory.bkd(), basis.bkd())


class TestPrebuiltBasisFactoryNumpy(TestPrebuiltBasisFactory[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPrebuiltBasisFactoryTorch(TestPrebuiltBasisFactory[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Helper function tests
# =============================================================================


class TestHelperFunctions(Generic[Array], unittest.TestCase):
    """Tests for helper functions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_get_bounds_uniform(self) -> None:
        """Test get_bounds_from_marginal for UniformMarginal."""
        marginal = UniformMarginal(lower=2.0, upper=5.0, bkd=self._bkd)
        lb, ub = get_bounds_from_marginal(marginal)

        self.assertEqual(lb, 2.0)
        self.assertEqual(ub, 5.0)

    def test_get_bounds_gaussian(self) -> None:
        """Test get_bounds_from_marginal for GaussianMarginal."""
        marginal = GaussianMarginal(mean=0.0, stdev=1.0, bkd=self._bkd)
        lb, ub = get_bounds_from_marginal(marginal, eps=1e-6)

        # For standard normal, eps=1e-6 gives approximately [-4.75, 4.75]
        self.assertLess(lb, -4.0)
        self.assertGreater(ub, 4.0)

    def test_get_transform_uniform(self) -> None:
        """Test get_transform_from_marginal for UniformMarginal."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        transform = get_transform_from_marginal(marginal, self._bkd)

        # Map canonical [-1, 1] to user [0, 1]
        canonical = self._bkd.asarray([[-1.0, 0.0, 1.0]])
        user = transform.map_from_canonical(canonical)

        expected = self._bkd.asarray([[0.0, 0.5, 1.0]])
        self._bkd.assert_allclose(user, expected, rtol=1e-12)

    def test_get_transform_gaussian(self) -> None:
        """Test get_transform_from_marginal for GaussianMarginal."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=self._bkd)
        transform = get_transform_from_marginal(marginal, self._bkd)

        # Map canonical N(0,1) points to user N(5, 4)
        # z = -1, 0, 1 -> x = 5 + 2*z = 3, 5, 7
        canonical = self._bkd.asarray([[-1.0, 0.0, 1.0]])
        user = transform.map_from_canonical(canonical)

        expected = self._bkd.asarray([[3.0, 5.0, 7.0]])
        self._bkd.assert_allclose(user, expected, rtol=1e-12)

    def test_create_basis_factories_gauss(self) -> None:
        """Test create_basis_factories with gauss type."""
        marginals = [
            UniformMarginal(0.0, 1.0, self._bkd),
            GaussianMarginal(0.0, 1.0, self._bkd),
        ]
        factories = create_basis_factories(marginals, self._bkd, "gauss")

        self.assertEqual(len(factories), 2)
        for factory in factories:
            self.assertIsInstance(factory, BasisFactoryProtocol)
            self.assertIsInstance(factory, GaussLagrangeFactory)

    def test_create_bases_from_marginals(self) -> None:
        """Test create_bases_from_marginals convenience function."""
        marginals = [
            UniformMarginal(0.0, 1.0, self._bkd),
            UniformMarginal(0.0, 1.0, self._bkd),
        ]
        bases = create_bases_from_marginals(marginals, self._bkd)

        self.assertEqual(len(bases), 2)
        for basis in bases:
            basis.set_nterms(5)
            samples, weights = basis.quadrature_rule()
            self.assertEqual(samples.shape[1], 5)


class TestHelperFunctionsNumpy(TestHelperFunctions[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestHelperFunctionsTorch(TestHelperFunctions[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# LejaLagrangeFactory tests (basic - Leja is expensive)
# =============================================================================


class TestLejaLagrangeFactory(Generic[Array], unittest.TestCase):
    """Basic tests for LejaLagrangeFactory."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_factory_implements_protocol(self) -> None:
        """Test that factory implements BasisFactoryProtocol."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        factory = LejaLagrangeFactory(marginal, self._bkd)

        self.assertIsInstance(factory, BasisFactoryProtocol)

    def test_uniform_marginal_basic(self) -> None:
        """Test basic Leja factory creation for Uniform marginal."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        factory = LejaLagrangeFactory(marginal, self._bkd)

        basis = factory.create_basis()
        basis.set_nterms(3)  # Small number for speed
        samples, weights = basis.quadrature_rule()

        # Samples should be in [0, 1]
        min_val = float(self._bkd.min(samples))
        max_val = float(self._bkd.max(samples))

        self.assertGreaterEqual(min_val, 0.0)
        self.assertLessEqual(max_val, 1.0)

    def test_leja_caching(self) -> None:
        """Test that Leja sequence is cached across create_basis calls."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=self._bkd)
        factory = LejaLagrangeFactory(marginal, self._bkd)

        # First call creates the Leja sequence
        basis1 = factory.create_basis()
        basis1.set_nterms(3)
        samples1, _ = basis1.quadrature_rule()

        # Second call should reuse the cached sequence
        basis2 = factory.create_basis()
        basis2.set_nterms(3)
        samples2, _ = basis2.quadrature_rule()

        # Same samples (nested property of Leja)
        self._bkd.assert_allclose(samples1, samples2, rtol=1e-12)


class TestLejaLagrangeFactoryNumpy(TestLejaLagrangeFactory[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLejaLagrangeFactoryTorch(TestLejaLagrangeFactory[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
