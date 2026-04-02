"""Tests for normal CDF/PDF utility functions."""

from scipy.stats import norm

from pyapprox.optimization.bayesian.math_utils import normal_cdf, normal_pdf


class TestMathUtils:
    def test_normal_cdf(self, bkd) -> None:
        x_vals = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        x = bkd.array(x_vals)
        result = normal_cdf(x, bkd)
        expected = bkd.array(norm.cdf(x_vals))
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_normal_pdf(self, bkd) -> None:
        x_vals = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        x = bkd.array(x_vals)
        result = normal_pdf(x, bkd)
        expected = bkd.array(norm.pdf(x_vals))
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_normal_cdf_extreme_values(self, bkd) -> None:
        x = bkd.array([-10.0, 10.0])
        result = normal_cdf(x, bkd)
        bkd.assert_allclose(result, bkd.array([0.0, 1.0]), atol=1e-15)

    def test_normal_pdf_symmetry(self, bkd) -> None:
        x = bkd.array([1.5, -1.5])
        result = normal_pdf(x, bkd)
        bkd.assert_allclose(
            bkd.reshape(result[0:1], (1,)),
            bkd.reshape(result[1:2], (1,)),
            rtol=1e-14,
        )

    def test_normal_pdf_at_zero(self, bkd) -> None:
        x = bkd.array([0.0])
        result = normal_pdf(x, bkd)
        expected = bkd.array([norm.pdf(0.0)])
        bkd.assert_allclose(result, expected, rtol=1e-14)
