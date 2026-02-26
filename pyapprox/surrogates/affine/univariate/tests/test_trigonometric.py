"""Tests for TrigonometricPolynomial1D.

Ports the trigonometric polynomial test from legacy
pyapprox/surrogates/affine/tests/test_basis.py.
"""

import numpy as np
import pytest

from pyapprox.surrogates.affine.expansions.trigonometric import (
    TrigonometricExpansion,
)
from pyapprox.surrogates.affine.univariate.trigonometric import (
    TrigonometricPolynomial1D,
)


class TestTrigonometricBasis:

    def test_trigonometric_polynomial(self, bkd) -> None:
        """Port of legacy test_trigonometric_polynomial.

        Build trig basis with bounds=[-pi, pi], nterms=5.
        Set coefs [1, -4, 0, 0, 6] for f(x) = 1 - 4*cos(x) + 6*sin(2x).
        Verify evaluation matches at 11 test points.
        """
        bounds = bkd.array([-np.pi, np.pi])
        nterms = 5

        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        trig_exp = TrigonometricExpansion(trig_basis, bkd)

        def fun(xx):
            return 1 - 4 * bkd.cos(xx.T) + 6 * bkd.sin(2 * xx.T)

        trig_coefs = bkd.array([1.0, -4.0, 0.0, 0.0, 6.0])[:, None]
        trig_exp.set_coefficients(trig_coefs)

        test_samples = bkd.linspace(-np.pi, np.pi, 11)[None, :]
        bkd.assert_allclose(
            trig_exp(test_samples),
            fun(test_samples).T,
        )

    def test_trigonometric_expansion_coefficients(self, bkd) -> None:
        """Test set/get coefficients roundtrip and multi-QoI evaluation."""
        bounds = bkd.array([0.0, 2 * np.pi])
        nterms = 3

        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        trig_exp = TrigonometricExpansion(trig_basis, bkd)

        # Single QoI
        coef1 = bkd.array([2.0, 1.0, 3.0])[:, None]
        trig_exp.set_coefficients(coef1)
        bkd.assert_allclose(trig_exp.get_coefficients(), coef1)
        assert trig_exp.nqoi() == 1

        # Multi QoI
        coef2 = bkd.array([[1.0, 0.5], [2.0, -1.0], [0.0, 3.0]])
        trig_exp.set_coefficients(coef2)
        bkd.assert_allclose(trig_exp.get_coefficients(), coef2)
        assert trig_exp.nqoi() == 2

        # Evaluate
        test_samples = bkd.linspace(0, 2 * np.pi, 7)[None, :]
        result = trig_exp(test_samples)
        assert result.shape == (2, 7)

    def test_nterms_must_be_odd(self, bkd) -> None:
        """Test that nterms must be odd."""
        bounds = bkd.array([-np.pi, np.pi])
        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        with pytest.raises(ValueError):
            trig_basis.set_nterms(4)

    def test_basis_matrix_shape(self, bkd) -> None:
        """Test basis matrix has correct shape."""
        bounds = bkd.array([-np.pi, np.pi])
        nterms = 7
        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        samples = bkd.linspace(-np.pi, np.pi, 20)[None, :]
        basis_mat = trig_basis(samples)
        assert basis_mat.shape == (20, nterms)

    def test_jacobian_batch(self, bkd) -> None:
        """Test analytical derivatives against finite differences."""
        bounds = bkd.array([-np.pi, np.pi])
        nterms = 5
        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        samples = bkd.array([[0.5, 1.0, -0.5]])
        h = 1e-7
        jac = trig_basis.jacobian_batch(samples)

        for i in range(samples.shape[1]):
            x = samples[:, i : i + 1]
            xph = x + h
            xmh = x - h
            fd = (trig_basis(xph) - trig_basis(xmh)) / (2 * h)
            bkd.assert_allclose(jac[i : i + 1, :], fd, atol=1e-6)

    def test_hessian_batch(self, bkd) -> None:
        """Test second derivatives against finite differences."""
        bounds = bkd.array([-np.pi, np.pi])
        nterms = 5
        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        samples = bkd.array([[0.5, 1.0, -0.5]])
        h = 1e-5
        hess = trig_basis.hessian_batch(samples)

        for i in range(samples.shape[1]):
            x = samples[:, i : i + 1]
            xph = x + h
            xmh = x - h
            fd = (trig_basis.jacobian_batch(xph) - trig_basis.jacobian_batch(xmh)) / (
                2 * h
            )
            bkd.assert_allclose(hess[i : i + 1, :], fd, atol=1e-4)
