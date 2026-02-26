"""Tests for basis factory system.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import pytest

from pyapprox.probability.univariate import (
    BetaMarginal,
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.surrogates.affine.univariate.lagrange import LagrangeBasis1D
from pyapprox.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    ClenshawCurtisLagrangeFactory,
    GaussLagrangeFactory,
    LejaLagrangeFactory,
    PrebuiltBasisFactory,
    create_bases_from_marginals,
    create_basis_factories,
    get_bounds_from_marginal,
    get_registered_basis_types,
    get_transform_from_marginal,
)

# =============================================================================
# GaussLagrangeFactory tests
# =============================================================================


class TestGaussLagrangeFactory:
    """Tests for GaussLagrangeFactory."""

    def test_uniform_marginal_quadrature_in_user_domain(self, bkd) -> None:
        """Test that Uniform[0,1] returns quadrature points in [0,1]."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Samples should be in [0, 1], not [-1, 1]
        min_val = float(bkd.min(samples))
        max_val = float(bkd.max(samples))

        assert min_val >= 0.0
        assert max_val <= 1.0

        # For Gauss-Legendre on [0,1] with 5 points, samples should NOT
        # include exactly 0 or 1 (those are boundary points)
        assert min_val > 0.0
        assert max_val < 1.0

    def test_uniform_marginal_integration_exact(self, bkd) -> None:
        """Test that integration of x over [0,1] gives mean=0.5."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = x over [0, 1] with uniform density
        # Expected mean = 0.5
        x = samples[0, :]
        mean = float(bkd.sum(x * weights[:, 0]))

        bkd.assert_allclose(
            bkd.asarray([mean]),
            bkd.asarray([0.5]),
            rtol=1e-12,
        )

    def test_uniform_marginal_variance_exact(self, bkd) -> None:
        """Test that integration of (x-0.5)^2 over [0,1] gives variance=1/12."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = (x - 0.5)^2 over [0, 1] with uniform density
        # Expected variance = 1/12
        x = samples[0, :]
        variance = float(bkd.sum((x - 0.5) ** 2 * weights[:, 0]))

        bkd.assert_allclose(
            bkd.asarray([variance]),
            bkd.asarray([1.0 / 12.0]),
            rtol=1e-12,
        )

    def test_gaussian_marginal_quadrature_in_user_domain(self, bkd) -> None:
        """Test that N(5, 2^2) returns quadrature points centered at 5."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Samples should be centered around mean=5
        sample_mean = float(bkd.mean(samples))
        assert sample_mean > 3.0  # Not centered at 0
        assert sample_mean < 7.0

    def test_gaussian_marginal_integration_mean(self, bkd) -> None:
        """Test that integration of x gives mean=5 for N(5, 2^2)."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(10)  # More points for Hermite quadrature
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = x
        # Expected mean = 5
        x = samples[0, :]
        mean = float(bkd.sum(x * weights[:, 0]))

        bkd.assert_allclose(
            bkd.asarray([mean]),
            bkd.asarray([5.0]),
            rtol=1e-10,
        )

    def test_gaussian_marginal_integration_variance(self, bkd) -> None:
        """Test that integration of (x-5)^2 gives variance=4 for N(5, 2^2)."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(10)
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = (x - 5)^2
        # Expected variance = 4
        x = samples[0, :]
        variance = float(bkd.sum((x - 5.0) ** 2 * weights[:, 0]))

        bkd.assert_allclose(
            bkd.asarray([variance]),
            bkd.asarray([4.0]),
            rtol=1e-10,
        )

    def test_beta_marginal_quadrature_in_user_domain(self, bkd) -> None:
        """Test that Beta(2, 5) returns quadrature points in [0, 1]."""
        marginal = BetaMarginal(alpha=2.0, beta=5.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Samples should be in [0, 1]
        min_val = float(bkd.min(samples))
        max_val = float(bkd.max(samples))

        assert min_val >= 0.0
        assert max_val <= 1.0

    def test_beta_marginal_integration_mean(self, bkd) -> None:
        """Test that integration of x gives correct mean for Beta(2, 5)."""
        alpha, beta = 2.0, 5.0
        marginal = BetaMarginal(alpha=alpha, beta=beta, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(10)
        samples, weights = basis.quadrature_rule()

        # Expected mean = alpha / (alpha + beta) = 2/7
        expected_mean = alpha / (alpha + beta)
        x = samples[0, :]
        computed_mean = float(bkd.sum(x * weights[:, 0]))

        bkd.assert_allclose(
            bkd.asarray([computed_mean]),
            bkd.asarray([expected_mean]),
            rtol=1e-10,
        )

    def test_factory_creates_independent_bases(self, bkd) -> None:
        """Test that each create_basis() call returns independent bases."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        basis1 = factory.create_basis()
        basis2 = factory.create_basis()

        # Set different nterms
        basis1.set_nterms(3)
        basis2.set_nterms(7)

        # They should have different numbers of samples
        samples1, _ = basis1.quadrature_rule()
        samples2, _ = basis2.quadrature_rule()

        assert samples1.shape[1] == 3
        assert samples2.shape[1] == 7

    def test_factory_implements_protocol(self, bkd) -> None:
        """Test that factory implements BasisFactoryProtocol."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = GaussLagrangeFactory(marginal, bkd)

        assert isinstance(factory, BasisFactoryProtocol)


# =============================================================================
# PrebuiltBasisFactory tests
# =============================================================================


class TestPrebuiltBasisFactory:
    """Tests for PrebuiltBasisFactory."""

    def test_wraps_existing_basis(self, bkd) -> None:
        """Test that PrebuiltBasisFactory creates LagrangeBasis1D from basis quadrature.

        PrebuiltBasisFactory extracts the quadrature rule from the wrapped basis
        and creates fresh LagrangeBasis1D instances each time. This ensures
        independent state for each subspace in a sparse grid.
        """
        basis = LegendrePolynomial1D(bkd)
        factory = PrebuiltBasisFactory(basis)

        # create_basis() returns LagrangeBasis1D, not the original basis
        created = factory.create_basis()
        assert isinstance(created, LagrangeBasis1D)

        # Each call creates an independent instance
        created2 = factory.create_basis()
        assert created is not created2

        # The bases should have independent state
        created.set_nterms(3)
        created2.set_nterms(5)
        assert created.nterms() == 3
        assert created2.nterms() == 5

    def test_factory_implements_protocol(self, bkd) -> None:
        """Test that factory implements BasisFactoryProtocol."""
        basis = LegendrePolynomial1D(bkd)
        factory = PrebuiltBasisFactory(basis)

        assert isinstance(factory, BasisFactoryProtocol)

    def test_returns_same_backend(self, bkd) -> None:
        """Test that factory returns same backend as wrapped basis."""
        basis = LegendrePolynomial1D(bkd)
        factory = PrebuiltBasisFactory(basis)

        assert factory.bkd() is basis.bkd()


# =============================================================================
# Helper function tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_bounds_uniform(self, bkd) -> None:
        """Test get_bounds_from_marginal for UniformMarginal."""
        marginal = UniformMarginal(lower=2.0, upper=5.0, bkd=bkd)
        lb, ub = get_bounds_from_marginal(marginal)

        assert lb == 2.0
        assert ub == 5.0

    def test_get_bounds_gaussian(self, bkd) -> None:
        """Test get_bounds_from_marginal for GaussianMarginal."""
        marginal = GaussianMarginal(mean=0.0, stdev=1.0, bkd=bkd)
        lb, ub = get_bounds_from_marginal(marginal, eps=1e-6)

        # For standard normal, eps=1e-6 gives approximately [-4.75, 4.75]
        assert lb < -4.0
        assert ub > 4.0

    def test_get_transform_uniform(self, bkd) -> None:
        """Test get_transform_from_marginal for UniformMarginal."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        transform = get_transform_from_marginal(marginal, bkd)

        # Map canonical [-1, 1] to user [0, 1]
        canonical = bkd.asarray([[-1.0, 0.0, 1.0]])
        user = transform.map_from_canonical(canonical)

        expected = bkd.asarray([[0.0, 0.5, 1.0]])
        bkd.assert_allclose(user, expected, rtol=1e-12)

    def test_get_transform_gaussian(self, bkd) -> None:
        """Test get_transform_from_marginal for GaussianMarginal."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=bkd)
        transform = get_transform_from_marginal(marginal, bkd)

        # Map canonical N(0,1) points to user N(5, 4)
        # z = -1, 0, 1 -> x = 5 + 2*z = 3, 5, 7
        canonical = bkd.asarray([[-1.0, 0.0, 1.0]])
        user = transform.map_from_canonical(canonical)

        expected = bkd.asarray([[3.0, 5.0, 7.0]])
        bkd.assert_allclose(user, expected, rtol=1e-12)

    def test_create_basis_factories_gauss(self, bkd) -> None:
        """Test create_basis_factories with gauss type."""
        marginals = [
            UniformMarginal(0.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
        ]
        factories = create_basis_factories(marginals, bkd, "gauss")

        assert len(factories) == 2
        for factory in factories:
            assert isinstance(factory, BasisFactoryProtocol)
            assert isinstance(factory, GaussLagrangeFactory)

    def test_create_bases_from_marginals(self, bkd) -> None:
        """Test create_bases_from_marginals convenience function."""
        marginals = [
            UniformMarginal(0.0, 1.0, bkd),
            UniformMarginal(0.0, 1.0, bkd),
        ]
        bases = create_bases_from_marginals(marginals, bkd)

        assert len(bases) == 2
        for basis in bases:
            basis.set_nterms(5)
            samples, weights = basis.quadrature_rule()
            assert samples.shape[1] == 5

    def test_factory_sharing_identical_marginals(self, bkd) -> None:
        """Identical marginals share the same factory instance."""
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),  # Same
            UniformMarginal(0.0, 2.0, bkd),  # Different
        ]

        for basis_type in ["gauss", "leja"]:
            factories = create_basis_factories(marginals, bkd, basis_type)
            assert factories[0] is factories[1]
            assert factories[0] is not factories[2]

    def test_factory_sharing_different_types(self, bkd) -> None:
        """Different marginal types get different factories."""
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
        ]

        factories = create_basis_factories(marginals, bkd, "gauss")
        assert factories[0] is not factories[1]

    def test_leja_sequence_shared_across_dimensions(self, bkd) -> None:
        """Verify Leja sequence is computed once for identical dimensions."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(5)]

        factories = create_basis_factories(marginals, bkd, "leja")

        # All should be the exact same instance
        for i in range(1, 5):
            assert factories[0] is factories[i]

        # Create bases and verify they share the Leja sequence
        bases = [f.create_basis() for f in factories]
        for b in bases:
            b.set_nterms(5)

        # Get samples - should be identical since sharing same Leja sequence
        samples_list = [b.quadrature_rule()[0] for b in bases]
        for s in samples_list[1:]:
            bkd.assert_allclose(samples_list[0], s)


# =============================================================================
# LejaLagrangeFactory tests (basic - Leja is expensive)
# =============================================================================


class TestLejaLagrangeFactory:
    """Basic tests for LejaLagrangeFactory."""

    def test_factory_implements_protocol(self, bkd) -> None:
        """Test that factory implements BasisFactoryProtocol."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = LejaLagrangeFactory(marginal, bkd)

        assert isinstance(factory, BasisFactoryProtocol)

    def test_uniform_marginal_basic(self, bkd) -> None:
        """Test basic Leja factory creation for Uniform marginal."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = LejaLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(3)  # Small number for speed
        samples, weights = basis.quadrature_rule()

        # Samples should be in [0, 1]
        min_val = float(bkd.min(samples))
        max_val = float(bkd.max(samples))

        assert min_val >= 0.0
        assert max_val <= 1.0

    def test_leja_caching(self, bkd) -> None:
        """Test that Leja sequence is cached across create_basis calls."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = LejaLagrangeFactory(marginal, bkd)

        # First call creates the Leja sequence
        basis1 = factory.create_basis()
        basis1.set_nterms(3)
        samples1, _ = basis1.quadrature_rule()

        # Second call should reuse the cached sequence
        basis2 = factory.create_basis()
        basis2.set_nterms(3)
        samples2, _ = basis2.quadrature_rule()

        # Same samples (nested property of Leja)
        bkd.assert_allclose(samples1, samples2, rtol=1e-12)


# =============================================================================
# ClenshawCurtisLagrangeFactory tests
# =============================================================================


class TestClenshawCurtisLagrangeFactory:
    """Tests for ClenshawCurtisLagrangeFactory."""

    def test_factory_implements_protocol(self, bkd) -> None:
        """Test that factory implements BasisFactoryProtocol."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)

        assert isinstance(factory, BasisFactoryProtocol)

    def test_uniform_marginal_quadrature_in_user_domain(self, bkd) -> None:
        """Test that Uniform[0,1] returns quadrature points in [0,1]."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)  # 2^2 + 1 = 5
        samples, weights = basis.quadrature_rule()

        # Samples should be in [0, 1]
        min_val = float(bkd.min(samples))
        max_val = float(bkd.max(samples))

        assert min_val >= 0.0
        assert max_val <= 1.0

    def test_uniform_marginal_integration_mean(self, bkd) -> None:
        """Test that integration of x over [0,1] gives mean=0.5."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)  # 2^2 + 1 = 5
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = x over [0, 1] with uniform density
        # Expected mean = 0.5
        x = samples[0, :]
        mean = float(bkd.sum(x * weights[:, 0]))

        bkd.assert_allclose(
            bkd.asarray([mean]),
            bkd.asarray([0.5]),
            rtol=1e-12,
        )

    def test_uniform_marginal_integration_variance(self, bkd) -> None:
        """Test that integration of (x-0.5)^2 over [0,1] gives variance=1/12."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(9)  # 2^3 + 1 = 9 for higher accuracy
        samples, weights = basis.quadrature_rule()

        # Integrate f(x) = (x - 0.5)^2 over [0, 1] with uniform density
        # Expected variance = 1/12
        x = samples[0, :]
        variance = float(bkd.sum((x - 0.5) ** 2 * weights[:, 0]))

        bkd.assert_allclose(
            bkd.asarray([variance]),
            bkd.asarray([1.0 / 12.0]),
            rtol=1e-12,
        )

    def test_points_are_nested(self, bkd) -> None:
        """Test that CC points at level l are subset of points at level l+1."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)
        basis = factory.create_basis()

        # Test nesting for levels 0 through 3
        for npoints_curr, npoints_next in [(1, 3), (3, 5), (5, 9)]:
            basis.set_nterms(npoints_curr)
            pts_curr, _ = basis.quadrature_rule()

            basis.set_nterms(npoints_next)
            pts_next, _ = basis.quadrature_rule()

            # Every point at current level should exist at next level
            for i in range(npoints_curr):
                pt = float(pts_curr[0, i])
                found = any(
                    abs(float(pts_next[0, j]) - pt) < 1e-12 for j in range(npoints_next)
                )
                assert found, (
                    f"Point {pt} at level with {npoints_curr} pts "
                    f"not found at level with {npoints_next} pts"
                )

    def test_weights_sum_to_one(self, bkd) -> None:
        """Test that weights sum to 1 (probability measure)."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)
        basis = factory.create_basis()

        for npoints in [1, 3, 5, 9, 17]:
            basis.set_nterms(npoints)
            _, weights = basis.quadrature_rule()
            weight_sum = float(bkd.sum(weights))
            bkd.assert_allclose(
                bkd.asarray([weight_sum]),
                bkd.asarray([1.0]),
                rtol=1e-12,
            )

    def test_gaussian_marginal_user_domain(self, bkd) -> None:
        """Test that N(5, 2^2) returns points centered around mean."""
        marginal = GaussianMarginal(mean=5.0, stdev=2.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)

        basis = factory.create_basis()
        basis.set_nterms(5)
        samples, weights = basis.quadrature_rule()

        # Samples should be centered around mean=5
        # The CC points on [-1, 1] get transformed to [5-2, 5+2] = [3, 7]
        min_val = float(bkd.min(samples))
        max_val = float(bkd.max(samples))

        assert min_val > 2.0  # At least 3.0 minus some margin
        assert max_val < 8.0  # At most 7.0 plus some margin

    def test_factory_creates_independent_bases(self, bkd) -> None:
        """Test that each create_basis() call returns independent bases."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)

        basis1 = factory.create_basis()
        basis2 = factory.create_basis()

        # Set different nterms (both must be valid CC sizes: 1, 3, 5, 9, ...)
        basis1.set_nterms(3)
        basis2.set_nterms(9)

        # They should have different numbers of samples
        samples1, _ = basis1.quadrature_rule()
        samples2, _ = basis2.quadrature_rule()

        assert samples1.shape[1] == 3
        assert samples2.shape[1] == 9

    def test_quadrature_caching(self, bkd) -> None:
        """Test that CC quadrature rule is cached for efficiency."""
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
        factory = ClenshawCurtisLagrangeFactory(marginal, bkd)

        # Create two bases from the same factory
        basis1 = factory.create_basis()
        basis2 = factory.create_basis()

        basis1.set_nterms(5)
        basis2.set_nterms(5)

        samples1, weights1 = basis1.quadrature_rule()
        samples2, weights2 = basis2.quadrature_rule()

        # Same samples and weights due to caching
        bkd.assert_allclose(samples1, samples2, rtol=1e-12)
        bkd.assert_allclose(weights1, weights2, rtol=1e-12)


# =============================================================================
# Registry pattern tests
# =============================================================================


class TestBasisFactoryRegistry:
    """Tests for basis factory registry pattern."""

    def test_get_registered_basis_types(self, bkd) -> None:
        """Test that get_registered_basis_types returns expected types."""
        types = get_registered_basis_types()

        # Check that all built-in types are registered
        assert "gauss" in types
        assert "leja" in types
        assert "clenshaw_curtis" in types
        assert "piecewise_linear" in types
        assert "piecewise_quadratic" in types
        assert "piecewise_cubic" in types

        # Check that result is sorted
        assert types == sorted(types)

    def test_create_basis_factories_gauss(self, bkd) -> None:
        """Test create_basis_factories with gauss type via registry."""
        marginals = [UniformMarginal(0.0, 1.0, bkd)]
        factories = create_basis_factories(marginals, bkd, "gauss")

        assert len(factories) == 1
        assert isinstance(factories[0], GaussLagrangeFactory)

    def test_create_basis_factories_leja(self, bkd) -> None:
        """Test create_basis_factories with leja type via registry."""
        marginals = [UniformMarginal(0.0, 1.0, bkd)]
        factories = create_basis_factories(marginals, bkd, "leja")

        assert len(factories) == 1
        assert isinstance(factories[0], LejaLagrangeFactory)

    def test_create_basis_factories_clenshaw_curtis(self, bkd) -> None:
        """Test create_basis_factories with clenshaw_curtis type via registry."""
        marginals = [UniformMarginal(0.0, 1.0, bkd)]
        factories = create_basis_factories(marginals, bkd, "clenshaw_curtis")

        assert len(factories) == 1
        assert isinstance(factories[0], ClenshawCurtisLagrangeFactory)

    def test_create_basis_factories_unknown_type(self, bkd) -> None:
        """Test that unknown basis_type raises ValueError with helpful message."""
        marginals = [UniformMarginal(0.0, 1.0, bkd)]

        with pytest.raises(ValueError) as context:
            create_basis_factories(marginals, bkd, "unknown_type")

        error_msg = str(context.value)
        assert "unknown_type" in error_msg
        assert "Available" in error_msg
