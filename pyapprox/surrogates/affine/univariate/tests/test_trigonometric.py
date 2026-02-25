"""Tests for TrigonometricPolynomial1D.

Ports the trigonometric polynomial test from legacy
pyapprox/surrogates/affine/tests/test_basis.py.
"""
import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.surrogates.affine.univariate.trigonometric import (
    TrigonometricPolynomial1D,
)
from pyapprox.surrogates.affine.expansions.trigonometric import (
    TrigonometricExpansion,
)


class TestTrigonometricBasis(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_trigonometric_polynomial(self) -> None:
        """Port of legacy test_trigonometric_polynomial.

        Build trig basis with bounds=[-pi, pi], nterms=5.
        Set coefs [1, -4, 0, 0, 6] for f(x) = 1 - 4*cos(x) + 6*sin(2x).
        Verify evaluation matches at 11 test points.
        """
        bkd = self._bkd
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

    def test_trigonometric_expansion_coefficients(self) -> None:
        """Test set/get coefficients roundtrip and multi-QoI evaluation."""
        bkd = self._bkd
        bounds = bkd.array([0.0, 2 * np.pi])
        nterms = 3

        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        trig_exp = TrigonometricExpansion(trig_basis, bkd)

        # Single QoI
        coef1 = bkd.array([2.0, 1.0, 3.0])[:, None]
        trig_exp.set_coefficients(coef1)
        bkd.assert_allclose(trig_exp.get_coefficients(), coef1)
        self.assertEqual(trig_exp.nqoi(), 1)

        # Multi QoI
        coef2 = bkd.array([[1.0, 0.5], [2.0, -1.0], [0.0, 3.0]])
        trig_exp.set_coefficients(coef2)
        bkd.assert_allclose(trig_exp.get_coefficients(), coef2)
        self.assertEqual(trig_exp.nqoi(), 2)

        # Evaluate
        test_samples = bkd.linspace(0, 2 * np.pi, 7)[None, :]
        result = trig_exp(test_samples)
        self.assertEqual(result.shape, (2, 7))

    def test_nterms_must_be_odd(self) -> None:
        """Test that nterms must be odd."""
        bkd = self._bkd
        bounds = bkd.array([-np.pi, np.pi])
        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        with self.assertRaises(ValueError):
            trig_basis.set_nterms(4)

    def test_basis_matrix_shape(self) -> None:
        """Test basis matrix has correct shape."""
        bkd = self._bkd
        bounds = bkd.array([-np.pi, np.pi])
        nterms = 7
        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        samples = bkd.linspace(-np.pi, np.pi, 20)[None, :]
        basis_mat = trig_basis(samples)
        self.assertEqual(basis_mat.shape, (20, nterms))

    def test_jacobian_batch(self) -> None:
        """Test analytical derivatives against finite differences."""
        bkd = self._bkd
        bounds = bkd.array([-np.pi, np.pi])
        nterms = 5
        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        samples = bkd.array([[0.5, 1.0, -0.5]])
        h = 1e-7
        jac = trig_basis.jacobian_batch(samples)

        for i in range(samples.shape[1]):
            x = samples[:, i:i+1]
            xph = x + h
            xmh = x - h
            fd = (trig_basis(xph) - trig_basis(xmh)) / (2 * h)
            bkd.assert_allclose(jac[i:i+1, :], fd, atol=1e-6)

    def test_hessian_batch(self) -> None:
        """Test second derivatives against finite differences."""
        bkd = self._bkd
        bounds = bkd.array([-np.pi, np.pi])
        nterms = 5
        trig_basis = TrigonometricPolynomial1D(bounds, bkd)
        trig_basis.set_nterms(nterms)

        samples = bkd.array([[0.5, 1.0, -0.5]])
        h = 1e-5
        hess = trig_basis.hessian_batch(samples)

        for i in range(samples.shape[1]):
            x = samples[:, i:i+1]
            xph = x + h
            xmh = x - h
            fd = (
                trig_basis.jacobian_batch(xph) -
                trig_basis.jacobian_batch(xmh)
            ) / (2 * h)
            bkd.assert_allclose(hess[i:i+1, :], fd, atol=1e-4)


class TestTrigonometricBasisNumpy(TestTrigonometricBasis[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTrigonometricBasisTorch(TestTrigonometricBasis[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


from pyapprox.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
