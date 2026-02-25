"""Tests for multivariate basis module."""

import unittest
from typing import Type

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend

from pyapprox.surrogates.affine.univariate import (
    create_bases_1d,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import (
    MultiIndexBasis,
    OrthonormalPolynomialBasis,
    TensorProductQuadratureRule,
    FixedTensorProductQuadratureRule,
)
from pyapprox.probability import (
    UniformMarginal,
    GaussianMarginal,
)


class _BaseBasisTest:
    """Base class for basis tests. Not run directly."""

    __test__ = False
    bkd_class: Type[Backend[NDArray]] = NumpyBkd

    def setUp(self):
        self.bkd = self.bkd_class()


class TestMultiIndexBasis(_BaseBasisTest, unittest.TestCase):
    """Test MultiIndexBasis class."""

    __test__ = True

    def _create_basis(self, nvars: int, max_level: int):
        """Helper to create a Legendre basis with hyperbolic indices."""
        bkd = self.bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        return OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    def test_basic_properties(self):
        """Test basic properties."""
        basis = self._create_basis(nvars=2, max_level=3)
        self.assertEqual(basis.nvars(), 2)
        self.assertGreater(basis.nterms(), 0)
        self.assertTrue(basis.jacobian_supported())
        self.assertTrue(basis.hessian_supported())

    def test_evaluation_shape(self):
        """Test basis evaluation shape."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=3)
        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        values = basis(samples)
        self.assertEqual(values.shape, (nsamples, basis.nterms()))

    def test_jacobian_batch_shape(self):
        """Test Jacobian computation shape."""
        bkd = self.bkd
        basis = self._create_basis(nvars=3, max_level=2)
        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))

        jac = basis.jacobian_batch(samples)
        self.assertEqual(jac.shape, (nsamples, basis.nterms(), 3))

    def test_hessian_batch_shape(self):
        """Test Hessian computation shape."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=3)
        nsamples = 8
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        hess = basis.hessian_batch(samples)
        self.assertEqual(hess.shape, (nsamples, basis.nterms(), 2, 2))

    def test_jacobian_batch_finite_difference(self):
        """Test Jacobian accuracy via finite differences."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=3)
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

    def test_hessian_batch_finite_difference(self):
        """Test Hessian accuracy via finite differences."""
        bkd = self.bkd
        basis = self._create_basis(nvars=2, max_level=2)
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
            bkd.assert_allclose(
                hess[:, :, dd, :], fd_hess_row, rtol=1e-4, atol=1e-6
            )

    def test_hessian_batch_symmetry(self):
        """Test that Hessian is symmetric."""
        bkd = self.bkd
        basis = self._create_basis(nvars=3, max_level=2)
        nsamples = 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))

        hess = basis.hessian_batch(samples)

        for dd in range(3):
            for kk in range(dd + 1, 3):
                bkd.assert_allclose(hess[:, :, dd, kk], hess[:, :, kk, dd])

    def test_constant_basis(self):
        """Test that first basis function is constant."""
        bkd = self.bkd
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


class TestOrthonormalPolynomialBasis(_BaseBasisTest, unittest.TestCase):
    """Test OrthonormalPolynomialBasis class."""

    __test__ = True

    def test_univariate_quadrature(self):
        """Test univariate quadrature access."""
        bkd = self.bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        for b in bases_1d:
            b.set_nterms(5)
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        pts, wts = basis.univariate_quadrature(0, npoints=5)
        self.assertEqual(pts.shape, (1, 5))
        self.assertEqual(wts.shape[0], 5)

    def test_tensor_product_quadrature(self):
        """Test tensor product quadrature rule."""
        bkd = self.bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        for b in bases_1d:
            b.set_nterms(4)
        indices = compute_hyperbolic_indices(2, 2, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        pts, wts = basis.tensor_product_quadrature([3, 4])
        self.assertEqual(pts.shape, (2, 12))  # 3 * 4 = 12
        self.assertEqual(wts.shape[0], 12)

        # Weights sum to 1 (probability measure normalization)
        bkd.assert_allclose(bkd.sum(wts), 1.0, atol=1e-12)

    def test_orthonormality(self):
        """Test orthonormality using quadrature."""
        bkd = self.bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(2, 3, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        # Use more points than needed for exact integration
        npoints_1d = [8, 8]
        pts, wts = basis.tensor_product_quadrature(npoints_1d)

        # Evaluate basis at quadrature points
        values = basis(pts)  # (npoints, nterms)

        # Compute inner product matrix: Φᵀ W Φ
        # This should be identity for orthonormal polynomials
        weighted_values = values * bkd.reshape(wts, (-1, 1))
        gram = bkd.dot(values.T, weighted_values)

        expected = bkd.eye(basis.nterms())
        bkd.assert_allclose(gram, expected, atol=1e-10)


class TestTensorProductQuadratureRule(_BaseBasisTest, unittest.TestCase):
    """Test TensorProductQuadratureRule class."""

    __test__ = True

    def test_quadrature_shape(self):
        """Test quadrature points and weights shape."""
        bkd = self.bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        bases_1d = create_bases_1d(marginals, bkd)
        npoints_1d = [3, 4, 5]

        quad = TensorProductQuadratureRule(bases_1d, npoints_1d, bkd)
        pts, wts = quad()

        self.assertEqual(quad.nvars(), 3)
        self.assertEqual(quad.npoints(), 60)  # 3*4*5
        self.assertEqual(pts.shape, (3, 60))
        self.assertEqual(wts.shape[0], 60)

    def test_integrate_polynomial(self):
        """Test integration of polynomial function.

        The quadrature is normalized for the probability measure (uniform on
        [-1,1]^2 with total weight 1), so it computes expected values.
        """
        bkd = self.bkd
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

    def test_integrate_constant(self):
        """Test integration of constant function.

        The quadrature is normalized for the probability measure, so
        the integral of f=1 gives 1.
        """
        bkd = self.bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        npoints_1d = [3, 3]

        quad = TensorProductQuadratureRule(bases_1d, npoints_1d, bkd)

        # Integral of constant 1 with probability weights = 1
        def func(samples):
            return bkd.ones((samples.shape[1], 1))

        integral = quad.integrate(func)
        bkd.assert_allclose(integral, bkd.asarray([1.0]), atol=1e-12)


class TestFixedTensorProductQuadratureRule(_BaseBasisTest, unittest.TestCase):
    """Test FixedTensorProductQuadratureRule class."""

    __test__ = True

    def test_basic_properties(self):
        """Test basic properties."""
        bkd = self.bkd
        points = bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        weights = bkd.asarray([0.5, 0.5])

        quad = FixedTensorProductQuadratureRule(points, weights, bkd)

        self.assertEqual(quad.nvars(), 2)
        self.assertEqual(quad.npoints(), 2)

        pts, wts = quad()
        bkd.assert_allclose(pts, points)
        bkd.assert_allclose(wts, weights)


class TestMixedBases(_BaseBasisTest, unittest.TestCase):
    """Test mixing different univariate bases."""

    __test__ = True

    def test_mixed_legendre_hermite(self):
        """Test basis with Legendre and Hermite polynomials."""
        bkd = self.bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd), GaussianMarginal(0.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(2, 2, 1.0, bkd)

        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

        # Should work with samples in appropriate ranges
        nsamples = 10
        # Legendre: [-1, 1], Hermite: any real
        samples = bkd.asarray([
            np.random.uniform(-1, 1, nsamples),
            np.random.normal(0, 1, nsamples)
        ])

        values = basis(samples)
        self.assertEqual(values.shape, (nsamples, basis.nterms()))

        # Jacobians should work
        jac = basis.jacobian_batch(samples)
        self.assertEqual(jac.shape, (nsamples, basis.nterms(), 2))


if __name__ == "__main__":
    unittest.main()
