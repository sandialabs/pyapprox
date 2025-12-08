"""Tests for Leja and Fekete point selection."""

import unittest
import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd


class TestChristoffelWeighting(unittest.TestCase):
    """Tests for ChristoffelWeighting."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_weights_shape(self):
        """Test that weights have correct shape."""
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(self.bkd)
        samples = self.bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self.bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        self.assertEqual(weights.shape, (3, 1))

    def test_weights_positive(self):
        """Test that weights are positive."""
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(self.bkd)
        samples = self.bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self.bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        self.assertTrue(np.all(self.bkd.to_numpy(weights) > 0))

    def test_jacobian_shape(self):
        """Test that Jacobian has correct shape."""
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(self.bkd)
        samples = self.bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self.bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        basis_jac = self.bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        jac = weighting.jacobian(samples, basis_values, basis_jac)

        self.assertEqual(jac.shape, (3, 1))


class TestPDFWeighting(unittest.TestCase):
    """Tests for PDFWeighting."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_weights_shape(self):
        """Test that PDF weights have correct shape."""
        from pyapprox.typing.surrogates.affine.leja import PDFWeighting
        from scipy import stats

        rv = stats.uniform(-1, 2)  # Uniform on [-1, 1]
        weighting = PDFWeighting(self.bkd, rv.pdf)
        samples = self.bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self.bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        self.assertEqual(weights.shape, (3, 1))

    def test_weights_match_pdf(self):
        """Test that weights match the PDF values."""
        from pyapprox.typing.surrogates.affine.leja import PDFWeighting
        from scipy import stats

        rv = stats.norm(0, 1)
        weighting = PDFWeighting(self.bkd, rv.pdf)
        samples = self.bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self.bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        expected = rv.pdf(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            self.bkd.to_numpy(weights)[:, 0], expected, rtol=1e-10
        )


class TestCompositeWeighting(unittest.TestCase):
    """Tests for CompositeWeighting."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_composite_product(self):
        """Test that composite weighting is product of individual weights."""
        from pyapprox.typing.surrogates.affine.leja import (
            ChristoffelWeighting,
            PDFWeighting,
            CompositeWeighting,
        )
        from scipy import stats

        rv = stats.norm(0, 1)
        christoffel = ChristoffelWeighting(self.bkd)
        pdf = PDFWeighting(self.bkd, rv.pdf)
        composite = CompositeWeighting(self.bkd, christoffel, pdf)

        samples = self.bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self.bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])

        w1 = christoffel(samples, basis_values)
        w2 = pdf(samples, basis_values)
        w_composite = composite(samples, basis_values)

        expected = self.bkd.to_numpy(w1) * self.bkd.to_numpy(w2)
        np.testing.assert_allclose(
            self.bkd.to_numpy(w_composite), expected, rtol=1e-10
        )


class TestPivotedLUFactorizer(unittest.TestCase):
    """Tests for PivotedLUFactorizer."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_factorize_basic(self):
        """Test basic LU factorization."""
        from pyapprox.typing.util.linalg import PivotedLUFactorizer

        A = self.bkd.asarray([[4.0, 3.0], [6.0, 3.0]])
        factorizer = PivotedLUFactorizer(self.bkd, A)
        L, U = factorizer.factorize(2)

        # L should be lower triangular with unit diagonal
        self.assertEqual(L.shape, (2, 2))
        self.assertEqual(U.shape, (2, 2))

    def test_factorize_recovers_matrix(self):
        """Test that L @ U @ P recovers original matrix."""
        from pyapprox.typing.util.linalg import (
            PivotedLUFactorizer,
            get_final_pivots_from_sequential_pivots,
        )

        A = self.bkd.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
        )
        factorizer = PivotedLUFactorizer(self.bkd, A)
        L, U = factorizer.factorize(3)

        # Reconstruct and compare
        LU = L @ U
        pivots = factorizer.pivots()
        A_permuted = A[pivots, :]
        np.testing.assert_allclose(
            self.bkd.to_numpy(LU), self.bkd.to_numpy(A_permuted[:3, :3]), rtol=1e-10
        )


class TestPivotedQRFactorizer(unittest.TestCase):
    """Tests for PivotedQRFactorizer."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_select_points(self):
        """Test selecting points via QR factorization."""
        from pyapprox.typing.util.linalg import PivotedQRFactorizer

        # Create basis matrix
        basis_mat = self.bkd.asarray(
            [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, -0.5], [1.0, -1.0]]
        )
        factorizer = PivotedQRFactorizer(self.bkd)
        pivots = factorizer.select_points(basis_mat, 2)

        self.assertEqual(len(pivots), 2)
        # Pivots should be valid indices
        for p in pivots:
            self.assertTrue(0 <= int(p) < 5)


class TestFeketeSampler(unittest.TestCase):
    """Tests for Fekete point sampling."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_fekete_sample_count(self):
        """Test that Fekete sampler returns correct number of points."""
        from pyapprox.typing.surrogates.affine.leja import FeketeSampler
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import HyperbolicIndexGenerator

        # Create a simple 1D basis
        poly = LegendrePolynomial1D(self.bkd)
        poly.set_nterms(5)

        # Create multivariate basis wrapper
        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, self.bkd)
        basis = MultiIndexBasis([poly], self.bkd, idx_gen.get_indices())

        # Generate candidates
        candidates = self.bkd.linspace(-1, 1, 20)[None, :]

        sampler = FeketeSampler(self.bkd, basis, candidates)
        selected = sampler.sample(5)

        self.assertEqual(selected.shape, (1, 5))


class TestLejaSampler(unittest.TestCase):
    """Tests for multivariate Leja sampling."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_leja_sample_count(self):
        """Test that Leja sampler returns correct number of points."""
        from pyapprox.typing.surrogates.affine.leja import LejaSampler
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import HyperbolicIndexGenerator

        # Create a simple 1D basis
        poly = LegendrePolynomial1D(self.bkd)
        poly.set_nterms(5)

        # Create multivariate basis wrapper
        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, self.bkd)
        basis = MultiIndexBasis([poly], self.bkd, idx_gen.get_indices())

        # Generate candidates
        candidates = self.bkd.linspace(-1, 1, 20)[None, :]

        sampler = LejaSampler(self.bkd, basis, candidates)
        selected = sampler.sample(5)

        self.assertEqual(selected.shape, (1, 5))

    def test_leja_incremental(self):
        """Test incremental Leja sampling."""
        from pyapprox.typing.surrogates.affine.leja import LejaSampler
        from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
        from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
        from pyapprox.typing.surrogates.affine.indices import HyperbolicIndexGenerator

        poly = LegendrePolynomial1D(self.bkd)
        poly.set_nterms(5)

        idx_gen = HyperbolicIndexGenerator(1, 4, 1.0, self.bkd)
        basis = MultiIndexBasis([poly], self.bkd, idx_gen.get_indices())

        candidates = self.bkd.linspace(-1, 1, 20)[None, :]

        sampler = LejaSampler(self.bkd, basis, candidates)

        # Sample 3 points
        sampler.sample(3)
        self.assertEqual(sampler.nsamples(), 3)

        # Add 2 more incrementally
        new_samples = sampler.sample_incremental(2)
        self.assertEqual(new_samples.shape, (1, 2))
        self.assertEqual(sampler.nsamples(), 5)


if __name__ == "__main__":
    unittest.main()
