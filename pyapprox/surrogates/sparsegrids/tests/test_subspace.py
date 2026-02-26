"""Tests for TensorProductSubspace and basis_setup utilities."""

import pytest

from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.surrogates.affine.univariate.globalpoly import (
    HermitePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    DynamicPiecewiseBasis,
    EquidistantNodeGenerator,
    PiecewiseQuadratic,
)
from pyapprox.surrogates.sparsegrids import TensorProductSubspace
from pyapprox.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
    LejaLagrangeFactory,
    PiecewiseFactory,
)
from pyapprox.surrogates.sparsegrids.basis_setup import (
    compute_npts_from_growth_rule,
    create_lagrange_from_quadrature,
    get_quadrature_rule,
)


class TestBasisSetup:
    """Tests for basis_setup.py helper functions."""

    def test_compute_npts_from_growth_rule(self, bkd):
        """Test growth rule application to multi-index."""
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1 for l > 0
        index = bkd.asarray([0, 1, 2, 3])
        npts = compute_npts_from_growth_rule(index, growth)
        assert npts == [1, 2, 3, 4]

    def test_create_lagrange_from_quadrature(self, bkd):
        """Test Lagrange basis creation from quadrature rule callable."""
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(5)  # Must set nterms before quadrature
        quad_rule = get_quadrature_rule(poly)
        lagrange = create_lagrange_from_quadrature(bkd, quad_rule)

        # Get 5-point quadrature
        samples, weights = quad_rule(5)
        assert samples.shape == (1, 5)
        assert weights.shape[0] == 5

        # Lagrange basis should work at these points
        lagrange.set_nterms(5)
        vals = lagrange(samples)
        assert vals.shape == (5, 5)

    def test_get_quadrature_rule_legendre(self, bkd):
        """Test extracting quadrature rule from Legendre polynomial."""
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(3)  # Must set nterms before quadrature
        quad_rule = get_quadrature_rule(poly)

        # Should return callable
        samples, weights = quad_rule(3)
        assert samples.shape == (1, 3)
        assert weights.shape[0] == 3

        # Weights should sum to 1 for probability measure
        bkd.assert_allclose(
            bkd.sum(weights), bkd.asarray(1.0), rtol=1e-12
        )

    def test_get_quadrature_rule_hermite(self, bkd):
        """Test extracting Gauss-Hermite quadrature rule."""
        hermite = HermitePolynomial1D(bkd)
        hermite.set_nterms(5)  # Must set nterms before quadrature
        quad_rule = get_quadrature_rule(hermite)

        # Get 5-point Gauss-Hermite quadrature
        samples, weights = quad_rule(5)
        assert samples.shape == (1, 5)
        assert weights.shape[0] == 5

        # Flatten weights for computation (may be (npts,1) or (npts,))
        weights_flat = bkd.flatten(weights)

        # For standard normal (probability measure), weights sum to 1
        bkd.assert_allclose(
            bkd.sum(weights_flat), bkd.asarray(1.0), rtol=1e-12
        )

        # Test quadrature exactness: E[x^2] = 1 for standard normal
        samples_1d = samples[0, :]
        integrand = samples_1d**2
        integral = bkd.sum(integrand * weights_flat)
        bkd.assert_allclose(integral, bkd.asarray(1.0), rtol=1e-10)

    def test_get_quadrature_rule_piecewise(self, bkd):
        """Test extracting quadrature rule from DynamicPiecewiseBasis.

        For DynamicPiecewiseBasis, the npoints argument is used to set
        the number of nodes (via set_nterms), enabling use with sparse grids.

        Per CLAUDE.md conventions, quadrature rules return 2D shapes:
        - points: (1, nterms) matching (nvars, nsamples) with nvars=1
        - weights: (nterms, 1) matching (nqoi, nsamples) with nqoi=1
        """
        # Create dynamic piecewise quadratic with node generator
        node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
        piecewise = DynamicPiecewiseBasis(bkd, PiecewiseQuadratic, node_gen)
        quad_rule = get_quadrature_rule(piecewise)

        # npoints determines number of nodes for DynamicPiecewiseBasis
        # Shapes follow 2D convention: points (1, n), weights (n, 1)
        samples, weights = quad_rule(5)
        assert samples.shape == (1, 5)
        assert weights.shape == (5, 1)

        # Can request different number of points
        samples7, weights7 = quad_rule(7)
        assert samples7.shape == (1, 7)
        assert weights7.shape == (7, 1)

    def test_get_quadrature_rule_error(self, bkd):
        """Test error raised for unsupported basis type."""

        class FakeBasis:
            pass

        with pytest.raises(TypeError):
            get_quadrature_rule(FakeBasis())


class TestTensorProductSubspace:
    """Tests for TensorProductSubspace."""

    def test_subspace_samples(self, bkd):
        """Test that subspace generates correct number of samples."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        # Level (1, 2) -> 2 x 3 = 6 samples
        index = bkd.asarray([1, 2])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        assert subspace.nsamples() == 6
        assert subspace.get_samples().shape == (2, 6)

    def test_subspace_single_dimension(self, bkd):
        """Test 1D subspace at various levels."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        for level in range(4):
            index = bkd.asarray([level])
            subspace = TensorProductSubspace(bkd, index, [factory], growth)
            expected_npts = growth(level)
            assert subspace.nsamples() == expected_npts

    def test_subspace_interpolation(self, bkd):
        """Test that subspace interpolates exactly for polynomials."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1

        # Level (2, 2) -> 3 x 3 = 9 samples, can interpolate degree 2 exactly
        index = bkd.asarray([2, 2])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        # Test function: f(x, y) = x^2 + y
        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + y, (1, -1))
        subspace.set_values(values)

        # Test at new points
        test_pts = bkd.asarray([[0.3, -0.5, 0.7], [0.2, 0.4, -0.3]])
        result = subspace(test_pts)
        x_t, y_t = test_pts[0, :], test_pts[1, :]
        expected = bkd.reshape(x_t**2 + y_t, (1, -1))

        bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_subspace_polynomial_exactness_by_degree(self, bkd):
        """Test polynomial exactness: degree-d polynomial exact at level d."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Test various polynomial degrees
        for degree in range(1, 5):
            index = bkd.asarray([degree, degree])
            subspace = TensorProductSubspace(
                bkd, index, [factory, factory], growth
            )

            # Polynomial of total degree = degree
            samples = subspace.get_samples()
            x, y = samples[0, :], samples[1, :]
            values = bkd.reshape(x**degree + y**degree, (1, -1))
            subspace.set_values(values)

            # Should interpolate exactly
            test_pts = bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
            result = subspace(test_pts)
            x_t, y_t = test_pts[0, :], test_pts[1, :]
            expected = bkd.reshape(x_t**degree + y_t**degree, (1, -1))

            bkd.assert_allclose(result, expected, rtol=1e-9)

    def test_subspace_jacobian_polynomial(self, bkd):
        """Test Jacobian computation for polynomial function."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Level (2, 2) -> can interpolate degree 2
        index = bkd.asarray([2, 2])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        # f(x, y) = x^2 + 2*x*y + y^2 = (x + y)^2
        # df/dx = 2x + 2y, df/dy = 2x + 2y
        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + 2 * x * y + y**2, (1, -1))
        subspace.set_values(values)

        # Test Jacobian at a point
        test_pt = bkd.asarray([[0.3], [0.5]])
        jac = subspace.jacobian(test_pt)

        # Expected: [2*0.3 + 2*0.5, 2*0.3 + 2*0.5] = [1.6, 1.6]
        expected_jac = bkd.asarray([[1.6, 1.6]])
        bkd.assert_allclose(jac, expected_jac, rtol=1e-8)

    def test_subspace_hessian_polynomial(self, bkd):
        """Test Hessian computation for polynomial function."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Level (2, 2) -> can interpolate degree 2
        index = bkd.asarray([2, 2])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        # f(x, y) = x^2 + x*y + y^2
        # Hessian: [[2, 1], [1, 2]]
        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + x * y + y**2, (1, -1))
        subspace.set_values(values)

        # Test Hessian at a point
        test_pt = bkd.asarray([[0.3], [0.5]])
        hess = subspace.hessian(test_pt)

        expected_hess = bkd.asarray([[2.0, 1.0], [1.0, 2.0]])
        bkd.assert_allclose(hess, expected_hess, rtol=1e-8)

    def test_subspace_hvp_polynomial(self, bkd):
        """Test Hessian-vector product matches explicit Hessian @ vec."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        index = bkd.asarray([2, 2])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        # f(x, y) = x^2 + x*y + y^2
        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + x * y + y**2, (1, -1))
        subspace.set_values(values)

        test_pt = bkd.asarray([[0.3], [0.5]])
        vec = bkd.asarray([[1.0], [2.0]])

        # Compute HVP
        hvp_result = subspace.hvp(test_pt, vec)

        # Compare with explicit Hessian @ vec
        hess = subspace.hessian(test_pt)
        expected = hess @ vec

        bkd.assert_allclose(hvp_result, expected, rtol=1e-10)

    def test_subspace_integrate_monomial_even(self, bkd):
        """Test quadrature exactness for even monomial."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Level 2 -> 3 points, can integrate degree 4 exactly (2n-1 = 5)
        index = bkd.asarray([2, 2])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        # f(x, y) = x^2 * y^2
        # With probability measure (weights sum to 1):
        # E[x^2] = 1/3, E[y^2] = 1/3
        # Integral = E[x^2 * y^2] = E[x^2] * E[y^2] = 1/9
        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 * y**2, (1, -1))
        subspace.set_values(values)

        integral = subspace.integrate()
        expected = bkd.asarray([1.0 / 9.0])
        bkd.assert_allclose(integral, expected, rtol=1e-10)

    def test_subspace_integrate_odd_zero(self, bkd):
        """Test that odd functions integrate to zero on symmetric domain."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        index = bkd.asarray([3, 3])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        # f(x, y) = x^3 (odd in x)
        samples = subspace.get_samples()
        x = samples[0, :]
        values = bkd.reshape(x**3, (1, -1))
        subspace.set_values(values)

        integral = subspace.integrate()
        expected = bkd.asarray([0.0])
        bkd.assert_allclose(integral, expected, atol=1e-14)

    def test_subspace_quadrature_weights_shape(self, bkd):
        """Test quadrature weights have correct shape."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        index = bkd.asarray([2, 3])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        weights = subspace.get_quadrature_weights()
        # 3 x 4 = 12 samples
        assert weights.shape == (12,)

    def test_subspace_hermite_integration(self, bkd):
        """Test subspace with Gauss-Hermite quadrature.

        Integrates f(x, y) = x^2 + y^2 over standard normal distribution.
        Expected: E[x^2] + E[y^2] = 1 + 1 = 2
        """
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Level (2, 2) -> 3 x 3 = 9 points
        index = bkd.asarray([2, 2])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + y**2, (1, -1))
        subspace.set_values(values)

        integral = subspace.integrate()
        # E[x^2 + y^2] = E[x^2] + E[y^2] = 1 + 1 = 2
        expected = bkd.asarray([2.0])
        bkd.assert_allclose(integral, expected, rtol=1e-10)

    def test_subspace_hermite_higher_moments(self, bkd):
        """Test Gauss-Hermite integrates higher moments correctly.

        For standard normal: E[x^4] = 3 (fourth moment)
        """
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Level 3 -> 4 points, can integrate degree 6 exactly
        index = bkd.asarray([3])
        subspace = TensorProductSubspace(bkd, index, [factory], growth)

        samples = subspace.get_samples()
        x = samples[0, :]
        values = bkd.reshape(x**4, (1, -1))
        subspace.set_values(values)

        integral = subspace.integrate()
        # E[x^4] = 3 for standard normal
        expected = bkd.asarray([3.0])
        bkd.assert_allclose(integral, expected, rtol=1e-10)

    def test_subspace_leja_legendre_interpolation(self, bkd):
        """Test subspace with Leja sequence on Legendre polynomial.

        Uses Leja points instead of Gauss quadrature for interpolation.
        """
        # Skip if lstsq is not implemented in backend (required by LejaSequence1D)
        if not hasattr(bkd, "lstsq"):
            pytest.skip("Backend does not support lstsq")
        # Test lstsq returns valid result (not None)
        test_A = bkd.asarray([[1.0]])
        test_b = bkd.asarray([[1.0]])
        if bkd.lstsq(test_A, test_b) is None:
            pytest.skip("Backend lstsq returns None (not implemented)")

        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = LejaLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)

        # Level 3 -> 4 points
        index = bkd.asarray([3, 3])
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        # Test polynomial exactness: f(x, y) = x^2 + y^2
        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 + y**2, (1, -1))
        subspace.set_values(values)

        # Should interpolate exactly
        test_pts = bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        result = subspace(test_pts)
        x_t, y_t = test_pts[0, :], test_pts[1, :]
        expected = bkd.reshape(x_t**2 + y_t**2, (1, -1))
        bkd.assert_allclose(result, expected, rtol=1e-9)

    def test_subspace_piecewise_quadratic_interpolation(self, bkd):
        """Test subspace with piecewise quadratic basis.

        Uses Simpson's rule quadrature for integration of exactly
        representable functions.
        """
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = PiecewiseFactory(marginal, bkd, poly_type="quadratic")

        # Growth rule: level 2 -> 5 nodes (must be odd for quadratic)
        growth = LinearGrowthRule(scale=2, shift=1)  # n(l) = 2*l + 1

        index = bkd.asarray([2])  # level 2 -> 5 nodes
        subspace = TensorProductSubspace(bkd, index, [factory], growth)

        # Test function: f(x) = x^2 (quadratic, exactly representable)
        samples = subspace.get_samples()
        x = samples[0, :]
        values = bkd.reshape(x**2, (1, -1))
        subspace.set_values(values)

        # Integrate x^2 over [-1, 1] with uniform measure
        # Integral = int_{-1}^{1} x^2 dx = 2/3
        integral = subspace.integrate()
        expected = bkd.asarray([2.0 / 3.0])
        bkd.assert_allclose(integral, expected, rtol=1e-10)

    def test_subspace_piecewise_quadratic_2d(self, bkd):
        """Test 2D piecewise quadratic integration.

        Integrates f(x, y) = x^2 * y^2 over [-1, 1]^2.
        Integral = (2/3) * (2/3) = 4/9
        """
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = PiecewiseFactory(marginal, bkd, poly_type="quadratic")

        # Growth rule: level 2 -> 5 nodes (must be odd for quadratic)
        growth = LinearGrowthRule(scale=2, shift=1)  # n(l) = 2*l + 1

        index = bkd.asarray([2, 2])  # level 2 -> 5 nodes each dim
        subspace = TensorProductSubspace(bkd, index, [factory, factory], growth)

        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 * y**2, (1, -1))
        subspace.set_values(values)

        integral = subspace.integrate()
        # int x^2 dx * int y^2 dy = (2/3) * (2/3) = 4/9
        expected = bkd.asarray([4.0 / 9.0])
        bkd.assert_allclose(integral, expected, rtol=1e-10)

    def test_subspace_mixed_piecewise_gauss(self, bkd):
        """Test 2D mixed subspace: piecewise quadratic + Gauss-Legendre.

        Dimension 0: Piecewise quadratic on [-1, 1]
        Dimension 1: Gauss-Legendre quadrature

        Integrates f(x, y) = x^2 * y^2.
        For piecewise (uniform measure): int x^2 dx = 2/3
        For Gauss-Legendre (probability measure): E[y^2] = 1/3
        Combined: (2/3) * (1/3) = 2/9
        """
        # Dimension 0: Dynamic piecewise quadratic
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        piecewise_factory = PiecewiseFactory(marginal, bkd, poly_type="quadratic")

        # Dimension 1: Gauss-Legendre (probability measure)
        gauss_factory = GaussLagrangeFactory(marginal, bkd)

        # Growth rule: level 2 -> 5 nodes for piecewise, level+1 for Gauss
        growth = LinearGrowthRule(scale=2, shift=1)  # n(l) = 2*l + 1

        # Index: (2, 2) means piecewise at level 2 (5 pts), Gauss at level 2 (5 pts)
        # But we use same growth for both - piecewise needs odd, Gauss is flexible
        index = bkd.asarray([2, 2])
        subspace = TensorProductSubspace(
            bkd, index, [piecewise_factory, gauss_factory], growth
        )

        samples = subspace.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = bkd.reshape(x**2 * y**2, (1, -1))
        subspace.set_values(values)

        integral = subspace.integrate()
        # piecewise x^2: 2/3, gauss E[y^2]: 1/3
        # Combined: 2/9
        expected = bkd.asarray([2.0 / 9.0])
        bkd.assert_allclose(integral, expected, rtol=1e-10)


class TestTensorProductSubspaceProtocolValidation:
    """Tests for runtime protocol validation in TensorProductSubspace."""

    def _setup_factories(self, bkd):
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factory = GaussLagrangeFactory(marginal, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        return factory, growth

    def test_invalid_bkd_raises_typeerror(self, bkd):
        """Verify TypeError raised when bkd doesn't satisfy Backend."""
        factory, growth = self._setup_factories(bkd)
        index = bkd.asarray([1, 2])
        with pytest.raises(TypeError) as ctx:
            TensorProductSubspace(
                "not a backend",  # type: ignore[arg-type]
                index,
                [factory, factory],
                growth,
            )
        assert "Backend" in str(ctx.value)
        assert "str" in str(ctx.value)

    def test_invalid_basis_factories_raises_typeerror(self, bkd):
        """Verify TypeError raised when basis_factories don't satisfy protocol."""
        factory, growth = self._setup_factories(bkd)
        index = bkd.asarray([1, 2])
        with pytest.raises(TypeError) as ctx:
            TensorProductSubspace(
                bkd,
                index,
                ["not a factory", "also not"],  # type: ignore[list-item]
                growth,
            )
        assert "BasisFactoryProtocol" in str(ctx.value)

    def test_invalid_growth_rules_raises_typeerror(self, bkd):
        """Verify TypeError raised when growth_rules don't satisfy protocol."""
        factory, growth = self._setup_factories(bkd)
        index = bkd.asarray([1, 2])
        with pytest.raises(TypeError) as ctx:
            TensorProductSubspace(
                bkd,
                index,
                [factory, factory],
                "not a growth rule",  # type: ignore[arg-type]
            )
        assert "IndexGrowthRuleProtocol" in str(ctx.value)
