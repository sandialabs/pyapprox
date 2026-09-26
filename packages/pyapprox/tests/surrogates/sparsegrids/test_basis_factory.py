"""Tests for basis factory system.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import pytest
from pyapprox.probability.univariate import (
    BetaMarginal,
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.leja.univariate import (
    LejaSequence1D,
    TwoPointLejaObjective,
)
from pyapprox.surrogates.affine.leja.weighting import ChristoffelWeighting
from pyapprox.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.surrogates.affine.univariate.lagrange import LagrangeBasis1D
from pyapprox.surrogates.affine.univariate.registry import (
    _lookup_analytical,
)
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


class TestLejaDomainHandling:
    """Leja points land in, and spread across, the marginal's support.

    The sequence is generated on the polynomial's canonical domain and
    then mapped to the user domain, so these use marginals whose
    support differs from that canonical domain. A marginal where the
    two coincide exercises the mapping as an identity and so says
    nothing about it.

    Generating a sequence costs roughly a second per handful of points
    and grows superlinearly, so the counts here are the smallest that
    still separate a correctly placed sequence from a misplaced one.
    """

    # Enough for a sequence to have spread over its support and for the
    # Lebesgue constant to be meaningful, while staying inside the
    # superlinear cost of generating one.
    _NTERMS = 20

    @staticmethod
    def _points(bkd, marginal, nterms: int):
        basis = LejaLagrangeFactory(marginal, bkd).create_basis()
        basis.set_nterms(nterms)
        return bkd.flatten(basis.quadrature_rule()[0])

    @pytest.mark.parametrize(
        "lower,upper", [(0.0, 1.0), (2.0, 4.0), (-5.0, -3.0)]
    )
    def test_bounded_points_span_their_support(
        self, bkd, lower: float, upper: float
    ) -> None:
        """Inside the support, and reaching out towards both ends.

        Both halves are asserted from one sequence because neither
        alone is worth much: containment passes for points crowded into
        a corner, and spread passes for points outside the support.
        """
        points = self._points(
            bkd,
            UniformMarginal(lower=lower, upper=upper, bkd=bkd),
            self._NTERMS,
        )
        low = float(bkd.min(points))
        high = float(bkd.max(points))
        width = upper - lower
        assert low >= lower - 1e-9
        assert high <= upper + 1e-9
        # Leja takes the endpoints early, so even a short sequence
        # reaches close to both.
        assert low < lower + 0.05 * width
        assert high > upper - 0.05 * width

    def test_uniform_points_are_affine_images_of_each_other(
        self, bkd
    ) -> None:
        """The sharpest available check on where the points go.

        Leja selection commutes with an affine reparameterization, so
        two Uniform marginals give point sets related by the exact
        affine map between their supports. This pins every point rather
        than the extremes, needs no claim about what the distribution
        should be, and holds term by term rather than asymptotically.
        """
        reference = self._points(
            bkd,
            UniformMarginal(lower=-1.0, upper=1.0, bkd=bkd),
            self._NTERMS,
        )
        for lower, upper in [(0.0, 1.0), (2.0, 4.0), (-5.0, -3.0)]:
            mapped = self._points(
                bkd,
                UniformMarginal(lower=lower, upper=upper, bkd=bkd),
                self._NTERMS,
            )
            # canonical [-1, 1] -> [lower, upper]
            expected = lower + (reference + 1.0) * (upper - lower) / 2.0
            bkd.assert_allclose(mapped, expected, atol=1e-12)

    def test_gaussian_points_are_affine_images_of_standard(
        self, bkd
    ) -> None:
        """The same equivariance where the support is unbounded."""
        reference = self._points(
            bkd, GaussianMarginal(0.0, 1.0, bkd), self._NTERMS
        )
        mapped = self._points(
            bkd, GaussianMarginal(5.0, 2.0, bkd), self._NTERMS
        )
        bkd.assert_allclose(mapped, 5.0 + 2.0 * reference, atol=1e-10)

    def test_interpolation_is_well_conditioned(self, bkd) -> None:
        """The property Christoffel weighting is chosen to provide.

        The Lebesgue constant, max over x of sum_i |L_i(x)|, bounds how
        much the interpolant amplifies its data. For Christoffel
        weighted nodes it grows slowly with the node count; nodes
        crowded into part of their domain make it explode, so a loose
        bound separates the two without pinning a value that depends on
        the count.
        """
        basis = LejaLagrangeFactory(
            UniformMarginal(lower=0.0, upper=1.0, bkd=bkd), bkd
        ).create_basis()
        basis.set_nterms(self._NTERMS)
        nodes = bkd.flatten(basis.quadrature_rule()[0])
        grid = bkd.linspace(
            float(bkd.min(nodes)), float(bkd.max(nodes)), 200
        )
        values = basis(bkd.reshape(grid, (1, -1)))
        constant = float(bkd.max(bkd.sum(bkd.abs(values), axis=1)))
        assert constant < 50.0

    def test_semibounded_points_respect_their_lower_bound(
        self, bkd
    ) -> None:
        """A Gamma marginal is bounded below and unbounded above."""
        from pyapprox.probability.univariate import GammaMarginal

        points = self._points(bkd, GammaMarginal(2.0, bkd=bkd), self._NTERMS)
        assert float(bkd.min(points)) >= 0.0

    def test_beta_points_lie_in_support(self, bkd) -> None:
        """A skewed bounded marginal, whose transform is not affine."""
        points = self._points(bkd, BetaMarginal(2.0, 5.0, bkd), self._NTERMS)
        assert float(bkd.min(points)) >= 0.0
        assert float(bkd.max(points)) <= 1.0

    @pytest.mark.parametrize(
        "objective_class", [None, TwoPointLejaObjective]
    )
    def test_every_objective_searches_the_canonical_domain(
        self, numpy_bkd, objective_class
    ) -> None:
        """Bounds reach the objective and the initial point together.

        ``LejaSequence1D`` passes one ``bounds`` to whichever objective
        it is given and also seeds the sequence at their midpoint, so
        the domain a sequence searches is a property of the sequence
        rather than of the objective. The tests above go through
        ``LejaLagrangeFactory``, which always takes the default
        objective; this covers the alternative against the same
        contract, so a second objective cannot silently search
        elsewhere.

        One backend is enough: the question is which interval is
        searched, which no backend affects.
        """
        marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=numpy_bkd)
        entry = _lookup_analytical(marginal)
        assert entry is not None
        sequence = LejaSequence1D(
            numpy_bkd,
            entry.polynomial_factory(marginal, numpy_bkd),
            ChristoffelWeighting(numpy_bkd),
            bounds=(-1.0, 1.0),
            objective_class=objective_class,
        )
        # The sequence searches the canonical domain, so its points
        # must span it rather than sit in part of it.
        points = numpy_bkd.flatten(sequence.quadrature_rule(7)[0])
        assert float(numpy_bkd.min(points)) < -0.9
        assert float(numpy_bkd.max(points)) > 0.9


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
