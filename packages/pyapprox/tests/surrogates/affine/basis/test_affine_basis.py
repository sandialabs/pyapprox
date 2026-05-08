"""Tests for multivariate basis module."""

import numpy as np

from pyapprox.probability import (
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.basis import (
    FixedTensorProductQuadratureRule,
    OrthonormalPolynomialBasis,
    TensorProductQuadratureRule,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import (
    create_bases_1d,
)


class TestMultiIndexBasis:
    """Test MultiIndexBasis class."""

    def _create_basis(self, bkd, nvars: int, max_level: int):
        """Helper to create a Legendre basis with hyperbolic indices."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_basic_properties(self, bkd):
        """Test basic properties."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        assert basis.nvars() == 2
        assert basis.nterms() > 0
        assert basis.jacobian_supported()
        assert basis.hessian_supported()

    def test_evaluation_shape(self, bkd):
        """Test basis evaluation shape."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        values = basis(samples)
        assert values.shape == (nsamples, basis.nterms())

    def test_jacobian_batch_shape(self, bkd):
        """Test Jacobian computation shape."""
        basis = self._create_basis(bkd, nvars=3, max_level=2)
        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))

        jac = basis.jacobian_batch(samples)
        assert jac.shape == (nsamples, basis.nterms(), 3)

    def test_hessian_batch_shape(self, bkd):
        """Test Hessian computation shape."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        nsamples = 8
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        hess = basis.hessian_batch(samples)
        assert hess.shape == (nsamples, basis.nterms(), 2, 2)

    def test_jacobian_batch_finite_difference(self, bkd):
        """Test Jacobian accuracy via finite differences."""
        basis = self._create_basis(bkd, nvars=2, max_level=3)
        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        jac = basis.jacobian_batch(samples)

        # Finite difference check
        eps = 1e-7
        for dd in range(2):
            samples_plus = bkd.copy(samples)
            samples_minus = bkd.copy(samples)
            samples_plus[dd, :] += eps
            samples_minus[dd, :] -= eps

            fd_jac = (basis(samples_plus) - basis(samples_minus)) / (2 * eps)
            bkd.assert_allclose(jac[:, :, dd], fd_jac, rtol=1e-5, atol=1e-7)

    def test_hessian_batch_finite_difference(self, bkd):
        """Test Hessian accuracy via finite differences."""
        basis = self._create_basis(bkd, nvars=2, max_level=2)
        nsamples = 3
        samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, nsamples)))

        hess = basis.hessian_batch(samples)

        # Finite difference check on Jacobian
        eps = 1e-6
        for dd in range(2):
            samples_plus = bkd.copy(samples)
            samples_minus = bkd.copy(samples)
            samples_plus[dd, :] += eps
            samples_minus[dd, :] -= eps

            jac_plus = basis.jacobian_batch(samples_plus)
            jac_minus = basis.jacobian_batch(samples_minus)

            fd_hess_row = (jac_plus - jac_minus) / (2 * eps)
            bkd.assert_allclose(hess[:, :, dd, :], fd_hess_row, rtol=1e-4, atol=1e-6)

    def test_hessian_batch_symmetry(self, bkd):
        """Test that Hessian is symmetric."""
        basis = self._create_basis(bkd, nvars=3, max_level=2)
        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))

        hess = basis.hessian_batch(samples)

        for dd in range(3):
            for kk in range(dd + 1, 3):
                bkd.assert_allclose(hess[:, :, dd, kk], hess[:, :, kk, dd])

    def test_constant_basis(self, bkd):
        """Test that first basis function is constant."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        # Just the zero index
        indices = bkd.zeros((2, 1), dtype=bkd.int64_dtype())
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        values = basis(samples)
        # All values should be the same (constant)
        bkd.assert_allclose(values, values[0:1, :] * bkd.ones((nsamples, 1)))


class TestOrthonormalPolynomialBasis:
    """Test OrthonormalPolynomialBasis class."""

    def test_univariate_quadrature(self, bkd):
        """Test univariate quadrature access."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        for b in bases_1d:
            b.set_nterms(5)
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        pts, wts = basis.univariate_quadrature(0, npoints=5)
        assert pts.shape == (1, 5)
        assert wts.shape[0] == 5

    def test_tensor_product_quadrature(self, bkd):
        """Test tensor product quadrature rule."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        for b in bases_1d:
            b.set_nterms(4)
        indices = compute_hyperbolic_indices(2, 2, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        pts, wts = basis.tensor_product_quadrature([3, 4])
        assert pts.shape == (2, 12)  # 3 * 4 = 12
        assert wts.shape[0] == 12

        # Weights sum to 1 (probability measure normalization)
        bkd.assert_allclose(bkd.sum(wts), bkd.asarray(1.0), atol=1e-12)

    def test_orthonormality(self, bkd):
        """Test orthonormality using quadrature."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        # Use more points than needed for exact integration
        npoints_1d = [8, 8]
        pts, wts = basis.tensor_product_quadrature(npoints_1d)

        # Evaluate basis at quadrature points
        values = basis(pts)  # (npoints, nterms)

        # Compute inner product matrix: Phi^T W Phi
        # This should be identity for orthonormal polynomials
        weighted_values = values * bkd.reshape(wts, (-1, 1))
        gram = bkd.dot(values.T, weighted_values)

        expected = bkd.eye(basis.nterms())
        bkd.assert_allclose(gram, expected, atol=1e-10)


class TestTensorProductQuadratureRule:
    """Test TensorProductQuadratureRule class."""

    def test_quadrature_shape(self, bkd):
        """Test quadrature points and weights shape."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        bases_1d = create_bases_1d(marginals, bkd)
        npoints_1d = [3, 4, 5]

        quad = TensorProductQuadratureRule(bases_1d, npoints_1d, bkd)
        pts, wts = quad()

        assert quad.nvars() == 3
        assert quad.npoints() == 60  # 3*4*5
        assert pts.shape == (3, 60)
        assert wts.shape[0] == 60

    def test_integrate_polynomial(self, bkd):
        """Test integration of polynomial function.

        The quadrature is normalized for the probability measure (uniform on
        [-1,1]^2 with total weight 1), so it computes expected values.
        """
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        npoints_1d = [5, 5]

        quad = TensorProductQuadratureRule(bases_1d, npoints_1d, bkd)

        # Compute E[x^2 * y^2] for uniform on [-1, 1]^2
        # E[x^2] = 1/3, E[y^2] = 1/3, so E[x^2 * y^2] = 1/9
        def func(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**2 * y**2, (-1, 1))

        integral = quad.integrate(func)
        expected = 1.0 / 9.0  # Expected value, not Lebesgue integral
        bkd.assert_allclose(integral, bkd.asarray([expected]), atol=1e-12)

    def test_integrate_constant(self, bkd):
        """Test integration of constant function.

        The quadrature is normalized for the probability measure, so
        the integral of f=1 gives 1.
        """
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        npoints_1d = [3, 3]

        quad = TensorProductQuadratureRule(bases_1d, npoints_1d, bkd)

        # Integral of constant 1 with probability weights = 1
        def func(samples):
            return bkd.ones((samples.shape[1], 1))

        integral = quad.integrate(func)
        bkd.assert_allclose(integral, bkd.asarray([1.0]), atol=1e-12)


class TestFixedTensorProductQuadratureRule:
    """Test FixedTensorProductQuadratureRule class."""

    def test_basic_properties(self, bkd):
        """Test basic properties."""
        points = bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        weights = bkd.asarray([0.5, 0.5])

        quad = FixedTensorProductQuadratureRule(points, weights, bkd)

        assert quad.nvars() == 2
        assert quad.npoints() == 2

        pts, wts = quad()
        bkd.assert_allclose(pts, points)
        bkd.assert_allclose(wts, weights)


class TestMixedBases:
    """Test mixing different univariate bases."""

    def test_mixed_legendre_hermite(self, bkd):
        """Test basis with Legendre and Hermite polynomials."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd), GaussianMarginal(0.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(2, 2, 1.0, bkd)

        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        # Should work with samples in appropriate ranges
        nsamples = 10
        # Legendre: [-1, 1], Hermite: any real
        samples = bkd.asarray(
            [np.random.uniform(-1, 1, nsamples), np.random.normal(0, 1, nsamples)]
        )

        values = basis(samples)
        assert values.shape == (nsamples, basis.nterms())

        # Jacobians should work
        jac = basis.jacobian_batch(samples)
        assert jac.shape == (nsamples, basis.nterms(), 2)
