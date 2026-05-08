"""Tests for quadrature module."""

from pyapprox.util.backends.numpy import NumpyBkd


class TestTensorProductQuadrature:
    """Tests for TensorProductQuadratureRule."""

    def test_1d_gauss_legendre(self):
        """Test 1D Gauss-Legendre quadrature."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.surrogates.quadrature import (
            TensorProductQuadratureRule,
        )

        basis = LegendrePolynomial1D(bkd)
        basis.set_nterms(5)

        rule = TensorProductQuadratureRule(
            bkd,
            [lambda n, b=basis: b.gauss_quadrature_rule(n)],
            [3],
        )

        samples, weights = rule()
        assert samples.shape == (1, 3)
        assert weights.shape == (3,)

        # Weights sum to 1 for orthonormal polynomials (integrates P_0 = 1)
        bkd.assert_allclose(
            bkd.asarray([float(bkd.sum(weights))]),
            bkd.asarray([1.0]),
            atol=1e-7,
        )

    def test_2d_tensor_product(self):
        """Test 2D tensor product quadrature."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.surrogates.quadrature import (
            TensorProductQuadratureRule,
        )

        basis = LegendrePolynomial1D(bkd)
        basis.set_nterms(5)

        def quad_rule(n, b=basis):
            return b.gauss_quadrature_rule(n)

        rule = TensorProductQuadratureRule(
            bkd,
            [quad_rule, quad_rule],
            [3, 4],
        )

        samples, weights = rule()
        assert samples.shape == (2, 12)
        assert weights.shape == (12,)

        # Weights sum to 1 for orthonormal polynomials (integrates P_0^2 = 1)
        bkd.assert_allclose(
            bkd.asarray([float(bkd.sum(weights))]),
            bkd.asarray([1.0]),
            atol=1e-7,
        )

    def test_integrate_polynomial(self):
        """Test integrating a polynomial exactly."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.surrogates.quadrature import (
            TensorProductQuadratureRule,
        )

        basis = LegendrePolynomial1D(bkd)
        basis.set_nterms(5)

        def quad_rule(n, b=basis):
            return b.gauss_quadrature_rule(n)

        rule = TensorProductQuadratureRule(
            bkd,
            [quad_rule, quad_rule],
            [3, 3],
        )

        # Integrate f(x, y) = x^2 + y^2 over [-1, 1]^2
        # With orthonormal weights (sum to 1), result is scaled by 1/4
        # True integral = 2 * (2/3) = 4/3, but with orthonormal: (4/3) / 4 = 1/3
        # Actually: integral of (x^2 + y^2) with orthonormal = sum(w * f)
        # where weights integrate P_0^2 = 1, so we get 2 * E[X^2] = 2 * (1/3) = 2/3
        def func(samples):
            x, y = samples[0, :], samples[1, :]
            return (x**2 + y**2)[:, None]

        result = rule.integrate(func)
        expected = 2.0 / 3.0
        bkd.assert_allclose(
            bkd.asarray([float(result[0])]),
            bkd.asarray([expected]),
            atol=1e-10,
        )


class TestStroudCubature:
    """Tests for Stroud cubature rules."""

    def test_cd2_nsamples(self):
        """Test CdD2 has correct number of samples."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.quadrature import StroudCdD2

        for d in [2, 3, 5]:
            rule = StroudCdD2(bkd, nvars=d)
            assert rule.nsamples() == 2 * d

    def test_cd3_nsamples(self):
        """Test CdD3 has correct number of samples."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.quadrature import StroudCdD3

        for d in [2, 3, 4]:
            rule = StroudCdD3(bkd, nvars=d)
            assert rule.nsamples() == 2**d

    def test_cd5_nsamples(self):
        """Test CdD5 has correct number of samples."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.quadrature import StroudCdD5

        for d in [2, 3, 4]:
            rule = StroudCdD5(bkd, nvars=d)
            assert rule.nsamples() == 2 * d * d + 1

    def test_cd2_integrate_constant(self):
        """Test CdD2 integrates constant exactly."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.quadrature import StroudCdD2

        rule = StroudCdD2(bkd, nvars=3)

        def const(samples):
            return bkd.ones((samples.shape[1], 1))

        result = rule.integrate(const)
        expected = 8.0  # Volume of [-1, 1]^3
        bkd.assert_allclose(
            bkd.asarray([float(result[0])]),
            bkd.asarray([expected]),
            atol=1e-10,
        )

    def test_cd2_integrate_linear(self):
        """Test CdD2 integrates linear function exactly."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.quadrature import StroudCdD2

        rule = StroudCdD2(bkd, nvars=3)

        # Integral of x + y + z over [-1, 1]^3 = 0
        def linear(samples):
            return bkd.sum(samples, axis=0)[:, None]

        result = rule.integrate(linear)
        bkd.assert_allclose(
            bkd.asarray([float(result[0])]),
            bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_cd3_integrate_quadratic(self):
        """Test CdD3 integrates quadratic exactly."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.quadrature import StroudCdD3

        rule = StroudCdD3(bkd, nvars=2)

        # Integral of x^2 + y^2 over [-1, 1]^2 = 2 * (2/3) * 2 = 8/3
        def quadratic(samples):
            return (samples[0, :] ** 2 + samples[1, :] ** 2)[:, None]

        result = rule.integrate(quadratic)
        expected = 8.0 / 3.0
        bkd.assert_allclose(
            bkd.asarray([float(result[0])]),
            bkd.asarray([expected]),
            atol=1e-10,
        )

    def test_cd5_integrate_quartic(self):
        """Test CdD5 integrates quartic exactly."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.quadrature import StroudCdD5

        rule = StroudCdD5(bkd, nvars=2)

        # Integral of x^4 + y^4 over [-1, 1]^2
        # = 2 * (integral of x^4 from -1 to 1) * (integral of 1 from -1 to 1)
        # = 2 * (2/5) * 2 = 8/5
        def quartic(samples):
            return (samples[0, :] ** 4 + samples[1, :] ** 4)[:, None]

        result = rule.integrate(quartic)
        expected = 8.0 / 5.0
        bkd.assert_allclose(
            bkd.asarray([float(result[0])]),
            bkd.asarray([expected]),
            atol=1e-10,
        )


class TestParameterizedQuadrature:
    """Tests for ParameterizedTensorProductQuadratureRule."""

    def test_levels(self):
        """Test that different levels produce different sample counts."""
        bkd = NumpyBkd()
        from pyapprox.surrogates.affine.indices import (
            LinearGrowthRule,
        )
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.surrogates.quadrature import (
            ParameterizedTensorProductQuadratureRule,
        )

        basis = LegendrePolynomial1D(bkd)
        basis.set_nterms(10)

        def quad_rule(n, b=basis):
            return b.gauss_quadrature_rule(n)

        growth = LinearGrowthRule(scale=1, shift=1)

        rule = ParameterizedTensorProductQuadratureRule(
            bkd,
            [quad_rule, quad_rule],
            growth,
        )

        # Level 0: 1x1 = 1 sample
        samples0, _ = rule(0)
        assert samples0.shape[1] == 1

        # Level 1: 2x2 = 4 samples
        samples1, _ = rule(1)
        assert samples1.shape[1] == 4

        # Level 2: 3x3 = 9 samples
        samples2, _ = rule(2)
        assert samples2.shape[1] == 9
