"""Tests for LinearKernel."""

import numpy as np

from pyapprox.surrogates.kernels.linear import LinearKernel


class TestLinearKernel:
    def test_shape(self, bkd):
        kernel = LinearKernel(
            signal_variance=1.0, signal_variance_bounds=(0.1, 10.0),
            nvars=2, bkd=bkd, fixed=True,
        )
        X1 = bkd.array(np.random.RandomState(0).randn(2, 5))
        X2 = bkd.array(np.random.RandomState(1).randn(2, 7))
        K = kernel(X1, X2)
        assert K.shape == (5, 7)

    def test_value(self, bkd):
        """k(x, x') = sv * x.T @ x'."""
        kernel = LinearKernel(
            signal_variance=2.0, signal_variance_bounds=(0.1, 10.0),
            nvars=2, bkd=bkd, fixed=True,
        )
        X1 = bkd.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        X2 = bkd.array([[5.0], [6.0]])             # (2, 1)
        K = kernel(X1, X2)
        expected = bkd.array([[46.0], [68.0]])
        bkd.assert_allclose(K, expected, rtol=1e-12)

    def test_diag(self, bkd):
        kernel = LinearKernel(
            signal_variance=1.5, signal_variance_bounds=(0.1, 10.0),
            nvars=2, bkd=bkd, fixed=True,
        )
        X = bkd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        diag = kernel.diag(X)
        expected = bkd.array([25.5, 43.5, 67.5])
        bkd.assert_allclose(diag, expected, rtol=1e-12)
