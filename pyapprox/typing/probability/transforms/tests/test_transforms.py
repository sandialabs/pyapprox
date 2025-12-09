"""
Tests for probability transforms.
"""

import unittest
import numpy as np
from scipy import stats

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.probability.univariate import (
    ScipyContinuousMarginal,
    GaussianMarginal,
)
from pyapprox.typing.probability.transforms import (
    AffineTransform,
    GaussianTransform,
    IndependentGaussianTransform,
    NatafTransform,
)


class TestAffineTransform(unittest.TestCase):
    """Tests for AffineTransform."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.loc = np.array([1.0, 2.0])
        self.scale = np.array([2.0, 3.0])
        self.transform = AffineTransform(self.loc, self.scale, self.bkd)

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_loc_scale(self):
        """Test loc and scale accessors."""
        np.testing.assert_array_equal(self.transform.loc(), self.loc)
        np.testing.assert_array_equal(self.transform.scale(), self.scale)

    def test_map_to_canonical(self):
        """Test map to canonical space."""
        # x = [1, 2] -> y = [0, 0] (at loc, canonical is 0)
        # x = [3, 5] -> y = [1, 1]
        x = np.array([[1.0, 3.0], [2.0, 5.0]])
        y = self.transform.map_to_canonical(x)
        expected = np.array([[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(y, expected)

    def test_map_from_canonical(self):
        """Test map from canonical space."""
        # y = [0, 0] -> x = [1, 2]
        # y = [1, 1] -> x = [3, 5]
        y = np.array([[0.0, 1.0], [0.0, 1.0]])
        x = self.transform.map_from_canonical(y)
        expected = np.array([[1.0, 3.0], [2.0, 5.0]])
        np.testing.assert_array_almost_equal(x, expected)

    def test_roundtrip(self):
        """Test that map_to_canonical and map_from_canonical are inverses."""
        x = np.array([[0.0, 1.0, 5.0], [1.0, 3.0, 10.0]])
        y = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(y)
        np.testing.assert_array_almost_equal(x, x_recovered)

    def test_jacobian_to_canonical(self):
        """Test Jacobian to canonical is 1/scale."""
        x = np.array([[1.0, 3.0], [2.0, 5.0]])
        _, jacobian = self.transform.map_to_canonical_with_jacobian(x)
        expected = np.array([[0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0]])
        np.testing.assert_array_almost_equal(jacobian, expected)

    def test_jacobian_from_canonical(self):
        """Test Jacobian from canonical is scale."""
        y = np.array([[0.0, 1.0], [0.0, 1.0]])
        _, jacobian = self.transform.map_from_canonical_with_jacobian(y)
        expected = np.array([[2.0, 2.0], [3.0, 3.0]])
        np.testing.assert_array_almost_equal(jacobian, expected)

    def test_log_det_jacobian(self):
        """Test log determinant of Jacobian."""
        x = np.array([[1.0, 3.0], [2.0, 5.0]])
        log_det = self.transform.log_det_jacobian_to_canonical(x)
        # log(1/2) + log(1/3) = -log(2) - log(3)
        expected = -np.log(2) - np.log(3)
        np.testing.assert_array_almost_equal(log_det, [expected, expected])

    def test_mismatched_shapes_raises(self):
        """Test mismatched loc and scale raises error."""
        with self.assertRaises(ValueError):
            AffineTransform(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                self.bkd,
            )


class TestGaussianTransform(unittest.TestCase):
    """Tests for GaussianTransform."""

    def setUp(self):
        self.bkd = NumpyBkd()
        # Uniform [0, 1] marginal
        self.uniform = ScipyContinuousMarginal(stats.uniform(0, 1), self.bkd)
        self.transform = GaussianTransform(self.uniform, self.bkd)

    def test_nvars(self):
        """Test nvars returns 1."""
        self.assertEqual(self.transform.nvars(), 1)

    def test_median_to_zero(self):
        """Test median of uniform maps to 0 (median of normal)."""
        x = np.array([0.5])
        y = self.transform.map_to_canonical(x)
        # 0.5 is median of U[0,1], should map to 0 (median of N(0,1))
        np.testing.assert_array_almost_equal(y, [0.0])

    def test_roundtrip(self):
        """Test roundtrip preserves samples."""
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(y)
        np.testing.assert_array_almost_equal(x, x_recovered)

    def test_normal_to_normal_identity(self):
        """Test normal marginal gives identity transform."""
        normal = GaussianMarginal(0.0, 1.0, self.bkd)
        transform = GaussianTransform(normal, self.bkd)

        x = np.array([-1.0, 0.0, 1.0, 2.0])
        y = transform.map_to_canonical(x)
        np.testing.assert_array_almost_equal(x, y, decimal=5)

    def test_jacobian_chain_rule(self):
        """Test Jacobian satisfies chain rule."""
        x = np.array([0.3])
        y, jac_to = self.transform.map_to_canonical_with_jacobian(x)
        x_back, jac_from = self.transform.map_from_canonical_with_jacobian(y)

        # jac_to * jac_from should be close to 1
        np.testing.assert_array_almost_equal(jac_to * jac_from, [1.0], decimal=5)


class TestIndependentGaussianTransform(unittest.TestCase):
    """Tests for IndependentGaussianTransform."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.marginals = [
            ScipyContinuousMarginal(stats.uniform(0, 1), self.bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), self.bkd),
            ScipyContinuousMarginal(stats.norm(0, 1), self.bkd),
        ]
        self.transform = IndependentGaussianTransform(self.marginals, self.bkd)

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 3)

    def test_marginals(self):
        """Test marginals accessor."""
        marginals = self.transform.marginals()
        self.assertEqual(len(marginals), 3)

    def test_map_to_canonical_shape(self):
        """Test map to canonical returns correct shape."""
        x = np.array([[0.5, 0.3], [0.4, 0.2], [0.0, 1.0]])
        y = self.transform.map_to_canonical(x)
        self.assertEqual(y.shape, (3, 2))

    def test_roundtrip(self):
        """Test roundtrip preserves samples."""
        x = np.array([[0.3, 0.7], [0.2, 0.6], [-1.0, 1.0]])
        y = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(y)
        np.testing.assert_array_almost_equal(x, x_recovered, decimal=5)

    def test_jacobian_shape(self):
        """Test Jacobian has correct shape."""
        x = np.array([[0.5, 0.3], [0.4, 0.2], [0.0, 1.0]])
        _, jacobian = self.transform.map_to_canonical_with_jacobian(x)
        self.assertEqual(jacobian.shape, (3, 2))

    def test_log_det_jacobian_shape(self):
        """Test log determinant has correct shape."""
        x = np.array([[0.5, 0.3], [0.4, 0.2], [0.0, 1.0]])
        log_det = self.transform.log_det_jacobian_to_canonical(x)
        self.assertEqual(log_det.shape, (2,))

    def test_normal_component_identity(self):
        """Test normal component is identity transform."""
        x = np.array([[0.5], [0.4], [0.5]])  # Third is N(0,1)
        y = self.transform.map_to_canonical(x)
        # Third component should be approximately unchanged
        np.testing.assert_array_almost_equal(y[2], x[2], decimal=5)


class TestAffineTransformProtocol(unittest.TestCase):
    """Test AffineTransform satisfies TransformWithJacobianProtocol."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.transform = AffineTransform(
            np.array([0.0, 1.0]),
            np.array([1.0, 2.0]),
            self.bkd,
        )

    def test_has_bkd(self):
        """Test has bkd method."""
        self.assertIsNotNone(self.transform.bkd())

    def test_has_nvars(self):
        """Test has nvars method."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_has_map_to_canonical(self):
        """Test has map_to_canonical method."""
        x = np.array([[0.0, 1.0], [1.0, 3.0]])
        y = self.transform.map_to_canonical(x)
        self.assertEqual(y.shape, (2, 2))

    def test_has_map_from_canonical(self):
        """Test has map_from_canonical method."""
        y = np.array([[0.0, 1.0], [0.0, 1.0]])
        x = self.transform.map_from_canonical(y)
        self.assertEqual(x.shape, (2, 2))

    def test_has_map_with_jacobian(self):
        """Test has map_to_canonical_with_jacobian method."""
        x = np.array([[0.0, 1.0], [1.0, 3.0]])
        y, jac = self.transform.map_to_canonical_with_jacobian(x)
        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(jac.shape, (2, 2))


class TestGaussianTransformProtocol(unittest.TestCase):
    """Test GaussianTransform satisfies TransformWithJacobianProtocol."""

    def setUp(self):
        self.bkd = NumpyBkd()
        self.marginal = ScipyContinuousMarginal(stats.uniform(0, 1), self.bkd)
        self.transform = GaussianTransform(self.marginal, self.bkd)

    def test_has_bkd(self):
        """Test has bkd method."""
        self.assertIsNotNone(self.transform.bkd())

    def test_has_nvars(self):
        """Test has nvars method."""
        self.assertEqual(self.transform.nvars(), 1)

    def test_has_map_to_canonical(self):
        """Test has map_to_canonical method."""
        x = np.array([0.3, 0.5, 0.7])
        y = self.transform.map_to_canonical(x)
        self.assertEqual(y.shape, (3,))

    def test_has_map_from_canonical(self):
        """Test has map_from_canonical method."""
        y = np.array([-1.0, 0.0, 1.0])
        x = self.transform.map_from_canonical(y)
        self.assertEqual(x.shape, (3,))


class TestNatafTransform(unittest.TestCase):
    """Tests for NatafTransform."""

    def setUp(self):
        self.bkd = NumpyBkd()
        # Two normal marginals with correlation
        self.marginals = [
            GaussianMarginal(0.0, 1.0, self.bkd),
            GaussianMarginal(0.0, 1.0, self.bkd),
        ]
        # Correlation matrix
        self.correlation = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.transform = NatafTransform(
            self.marginals, self.correlation, self.bkd
        )

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_marginals(self):
        """Test marginals accessor."""
        marginals = self.transform.marginals()
        self.assertEqual(len(marginals), 2)

    def test_correlation(self):
        """Test correlation accessor."""
        corr = self.transform.correlation()
        np.testing.assert_array_almost_equal(corr, self.correlation)

    def test_map_to_canonical_shape(self):
        """Test map to canonical returns correct shape."""
        x = np.array([[0.0, 1.0], [0.0, 1.0]])
        z = self.transform.map_to_canonical(x)
        self.assertEqual(z.shape, (2, 2))

    def test_map_from_canonical_shape(self):
        """Test map from canonical returns correct shape."""
        z = np.array([[0.0, 1.0], [0.0, 1.0]])
        x = self.transform.map_from_canonical(z)
        self.assertEqual(x.shape, (2, 2))

    def test_roundtrip(self):
        """Test roundtrip preserves samples."""
        x = np.array([[0.0, 0.5, -0.5], [0.0, 1.0, -1.0]])
        z = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(z)
        np.testing.assert_array_almost_equal(x, x_recovered, decimal=5)

    def test_identity_correlation(self):
        """Test identity correlation gives independent transform."""
        identity = np.eye(2)
        transform = NatafTransform(self.marginals, identity, self.bkd)

        x = np.array([[0.0, 1.0], [0.0, 1.0]])
        z = transform.map_to_canonical(x)
        # With identity correlation, output should equal input
        # since marginals are already standard normal
        np.testing.assert_array_almost_equal(x, z, decimal=5)

    def test_jacobian_shape(self):
        """Test Jacobian has correct shape."""
        x = np.array([[0.0, 1.0], [0.0, 1.0]])
        _, jacobian = self.transform.map_to_canonical_with_jacobian(x)
        self.assertEqual(jacobian.shape, (2, 2, 2))  # (nvars, nvars, nsamples)

    def test_invalid_correlation_shape_raises(self):
        """Test mismatched correlation shape raises error."""
        with self.assertRaises(ValueError):
            NatafTransform(
                self.marginals,
                np.eye(3),  # Wrong size
                self.bkd,
            )


class TestNatafTransformNonGaussian(unittest.TestCase):
    """Tests for NatafTransform with non-Gaussian marginals."""

    def setUp(self):
        self.bkd = NumpyBkd()
        # Uniform and beta marginals
        self.marginals = [
            ScipyContinuousMarginal(stats.uniform(0, 1), self.bkd),
            ScipyContinuousMarginal(stats.beta(2, 5), self.bkd),
        ]
        self.correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
        self.transform = NatafTransform(
            self.marginals, self.correlation, self.bkd
        )

    def test_nvars(self):
        """Test nvars returns correct dimension."""
        self.assertEqual(self.transform.nvars(), 2)

    def test_roundtrip(self):
        """Test roundtrip preserves samples."""
        x = np.array([[0.3, 0.7], [0.2, 0.4]])
        z = self.transform.map_to_canonical(x)
        x_recovered = self.transform.map_from_canonical(z)
        np.testing.assert_array_almost_equal(x, x_recovered, decimal=4)

    def test_canonical_is_approximately_normal(self):
        """Test canonical samples are approximately standard normal."""
        # Generate samples from marginals
        np.random.seed(42)
        n = 1000
        x = np.array([
            np.random.uniform(0, 1, n),
            np.random.beta(2, 5, n),
        ])

        # Transform to canonical
        z = self.transform.map_to_canonical(x)

        # Check that each component is approximately standard normal
        for i in range(2):
            z_i = z[i]
            # Mean should be close to 0 (within 0.2)
            self.assertLess(abs(float(np.mean(z_i))), 0.2)
            # Std should be close to 1 (within 0.2)
            self.assertLess(abs(float(np.std(z_i)) - 1.0), 0.2)


if __name__ == "__main__":
    unittest.main()
