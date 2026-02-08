"""Tests for PCE arithmetic operators (+, -, *, **)."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.typing.surrogates.affine.basis import (
    OrthonormalPolynomialBasis,
)
from pyapprox.typing.surrogates.affine.expansions import (
    PolynomialChaosExpansion,
)
from pyapprox.typing.probability import (
    UniformMarginal,
    GaussianMarginal,
)


class TestPCEArithmetic(Generic[Array], unittest.TestCase):
    """Test PCE arithmetic operators."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_pce(self, nvars: int, max_level: int, nqoi: int = 1):
        """Helper to create a Legendre PCE for uniform marginals."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return PolynomialChaosExpansion(basis, bkd, nqoi)

    def _create_pce_pair(self, nvars, level1, level2, nqoi=1):
        """Create two PCEs with different index sets and random coefficients."""
        bkd = self._bkd
        pce1 = self._create_pce(nvars, level1, nqoi)
        pce2 = self._create_pce(nvars, level2, nqoi)
        coef1 = bkd.asarray(
            np.random.randn(pce1.nterms(), nqoi).astype(np.float64)
        )
        coef2 = bkd.asarray(
            np.random.randn(pce2.nterms(), nqoi).astype(np.float64)
        )
        pce1.set_coefficients(coef1)
        pce2.set_coefficients(coef2)
        return pce1, pce2

    def _random_samples(self, nvars, nsamples=101):
        return self._bkd.asarray(
            np.random.uniform(-1, 1, (nvars, nsamples)).astype(np.float64)
        )

    # ------------------------------------------------------------------
    # Addition
    # ------------------------------------------------------------------

    def test_add_same_indices(self):
        """pce1 + pce2 with same index sets."""
        pce1, pce2 = self._create_pce_pair(2, 3, 3)
        pce3 = pce1 + pce2
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) + pce2(samples), rtol=1e-12
        )

    def test_add_different_indices(self):
        """pce1 + pce2 with different index sets."""
        pce1, pce2 = self._create_pce_pair(2, 2, 4)
        pce3 = pce1 + pce2
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) + pce2(samples), rtol=1e-12
        )

    def test_add_1d(self):
        """pce1 + pce2 in 1D."""
        pce1, pce2 = self._create_pce_pair(1, 3, 5)
        pce3 = pce1 + pce2
        samples = self._random_samples(1)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) + pce2(samples), rtol=1e-12
        )

    def test_add_scalar(self):
        """pce + constant."""
        pce1, _ = self._create_pce_pair(2, 3, 3)
        pce3 = pce1 + 5.0
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) + 5.0, rtol=1e-12
        )

    def test_radd_scalar(self):
        """constant + pce."""
        pce1, _ = self._create_pce_pair(2, 3, 3)
        pce3 = 3 + pce1
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), 3.0 + pce1(samples), rtol=1e-12
        )

    def test_add_multi_qoi(self):
        """pce1 + pce2 with nqoi > 1."""
        pce1, pce2 = self._create_pce_pair(2, 3, 3, nqoi=2)
        pce3 = pce1 + pce2
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) + pce2(samples), rtol=1e-12
        )

    # ------------------------------------------------------------------
    # Subtraction
    # ------------------------------------------------------------------

    def test_sub_pce(self):
        """pce1 - pce2."""
        pce1, pce2 = self._create_pce_pair(2, 3, 4)
        pce3 = pce1 - pce2
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) - pce2(samples), rtol=1e-12
        )

    def test_sub_scalar(self):
        """pce - constant."""
        pce1, _ = self._create_pce_pair(2, 3, 3)
        pce3 = pce1 - 7.0
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) - 7.0, rtol=1e-12
        )

    def test_rsub_scalar(self):
        """constant - pce."""
        pce1, _ = self._create_pce_pair(2, 3, 3)
        pce3 = 4 - pce1
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), 4.0 - pce1(samples), rtol=1e-12
        )

    # ------------------------------------------------------------------
    # Multiplication
    # ------------------------------------------------------------------

    def test_mul_1d(self):
        """pce1 * pce2 in 1D."""
        pce1, pce2 = self._create_pce_pair(1, 3, 3, nqoi=2)
        pce3 = pce1 * pce2
        samples = self._random_samples(1)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) * pce2(samples), rtol=1e-11
        )

    def test_mul_2d(self):
        """pce1 * pce2 in 2D."""
        pce1, pce2 = self._create_pce_pair(2, 3, 3, nqoi=2)
        pce3 = pce1 * pce2
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) * pce2(samples), rtol=1e-11
        )

    def test_mul_different_indices(self):
        """pce1 * pce2 with different index sets."""
        pce1, pce2 = self._create_pce_pair(2, 2, 4, nqoi=1)
        pce3 = pce1 * pce2
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) * pce2(samples), rtol=1e-11
        )

    def test_mul_scalar(self):
        """pce * constant."""
        pce1, _ = self._create_pce_pair(2, 3, 3)
        pce3 = pce1 * 3.0
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) * 3.0, rtol=1e-12
        )

    def test_rmul_scalar(self):
        """constant * pce."""
        pce1, _ = self._create_pce_pair(2, 3, 3)
        pce3 = 2 * pce1
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples), 2.0 * pce1(samples), rtol=1e-12
        )

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def test_pow_0(self):
        """pce ** 0 = constant 1."""
        pce1, _ = self._create_pce_pair(2, 3, 3)
        pce0 = pce1 ** 0
        samples = self._random_samples(2)
        expected = self._bkd.ones((1, samples.shape[1]))
        self._bkd.assert_allclose(pce0(samples), expected, rtol=1e-12)

    def test_pow_1(self):
        """pce ** 1 = pce."""
        pce1, _ = self._create_pce_pair(2, 3, 3)
        pce_1 = pce1 ** 1
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce_1(samples), pce1(samples), rtol=1e-12
        )

    def test_pow_2(self):
        """pce ** 2."""
        pce1, _ = self._create_pce_pair(1, 3, 3)
        pce2 = pce1 ** 2
        samples = self._random_samples(1)
        self._bkd.assert_allclose(
            pce2(samples), pce1(samples) ** 2, rtol=1e-12
        )

    def test_pow_3(self):
        """pce ** 3."""
        pce1, _ = self._create_pce_pair(1, 3, 3)
        pce3 = pce1 ** 3
        samples = self._random_samples(1)
        self._bkd.assert_allclose(
            pce3(samples), pce1(samples) ** 3, rtol=1e-10
        )

    # ------------------------------------------------------------------
    # Composite expressions
    # ------------------------------------------------------------------

    def test_composite_linear(self):
        """2 * pce1 - pce2 * 3 + 1."""
        pce1, pce2 = self._create_pce_pair(2, 3, 3, nqoi=2)
        pce3 = 2 * pce1 - pce2 * 3 + 1
        samples = self._random_samples(2)
        self._bkd.assert_allclose(
            pce3(samples),
            2 * pce1(samples) - pce2(samples) * 3 + 1,
            rtol=1e-12,
        )

    def test_composite_mul_add(self):
        """(pce1 + pce2) * pce3."""
        bkd = self._bkd
        pce1 = self._create_pce(1, 2)
        pce2 = self._create_pce(1, 3)
        pce3 = self._create_pce(1, 2)
        for p in [pce1, pce2, pce3]:
            coef = bkd.asarray(
                np.random.randn(p.nterms(), 1).astype(np.float64)
            )
            p.set_coefficients(coef)
        result = (pce1 + pce2) * pce3
        samples = self._random_samples(1)
        self._bkd.assert_allclose(
            result(samples),
            (pce1(samples) + pce2(samples)) * pce3(samples),
            rtol=1e-12,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_incompatible_bases_raise_typeerror(self):
        """Different polynomial types raise TypeError."""
        bkd = self._bkd
        marginals_u = [UniformMarginal(-1.0, 1.0, bkd)]
        marginals_g = [GaussianMarginal(0.0, 1.0, bkd)]
        bases_u = create_bases_1d(marginals_u, bkd)
        bases_g = create_bases_1d(marginals_g, bkd)
        idx = compute_hyperbolic_indices(1, 3, 1.0, bkd)
        basis_u = OrthonormalPolynomialBasis(bases_u, bkd, idx)
        basis_g = OrthonormalPolynomialBasis(bases_g, bkd, idx)
        pce_u = PolynomialChaosExpansion(basis_u, bkd)
        pce_g = PolynomialChaosExpansion(basis_g, bkd)
        coef = bkd.asarray(
            np.random.randn(pce_u.nterms(), 1).astype(np.float64)
        )
        pce_u.set_coefficients(coef)
        pce_g.set_coefficients(coef)
        with self.assertRaises(TypeError):
            pce_u + pce_g
        with self.assertRaises(TypeError):
            pce_u * pce_g

    def test_different_nvars_raise_valueerror(self):
        """Different nvars raise ValueError."""
        pce1, _ = self._create_pce_pair(1, 3, 3)
        pce2, _ = self._create_pce_pair(2, 3, 3)
        with self.assertRaises(ValueError):
            pce1 + pce2

    def test_pow_negative_raises_valueerror(self):
        """Negative power raises ValueError."""
        pce1, _ = self._create_pce_pair(1, 3, 3)
        with self.assertRaises(ValueError):
            pce1 ** (-1)

    def test_pow_noninteger_raises_typeerror(self):
        """Non-integer power raises TypeError."""
        pce1, _ = self._create_pce_pair(1, 3, 3)
        with self.assertRaises(TypeError):
            pce1 ** 2.5

    # ------------------------------------------------------------------
    # Immutability
    # ------------------------------------------------------------------

    def test_add_does_not_modify_inputs(self):
        """pce1 + pce2 leaves originals unchanged."""
        bkd = self._bkd
        pce1, pce2 = self._create_pce_pair(2, 3, 3)
        coef1_before = bkd.copy(pce1.get_coefficients())
        coef2_before = bkd.copy(pce2.get_coefficients())
        idx1_before = bkd.copy(pce1.get_indices())
        idx2_before = bkd.copy(pce2.get_indices())
        _ = pce1 + pce2
        bkd.assert_allclose(pce1.get_coefficients(), coef1_before)
        bkd.assert_allclose(pce2.get_coefficients(), coef2_before)
        bkd.assert_allclose(pce1.get_indices(), idx1_before)
        bkd.assert_allclose(pce2.get_indices(), idx2_before)

    def test_mul_does_not_modify_inputs(self):
        """pce1 * pce2 leaves originals unchanged."""
        bkd = self._bkd
        pce1, pce2 = self._create_pce_pair(1, 3, 3)
        coef1_before = bkd.copy(pce1.get_coefficients())
        coef2_before = bkd.copy(pce2.get_coefficients())
        _ = pce1 * pce2
        bkd.assert_allclose(pce1.get_coefficients(), coef1_before)
        bkd.assert_allclose(pce2.get_coefficients(), coef2_before)


class TestPCEArithmeticNumpy(TestPCEArithmetic[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPCEArithmeticTorch(TestPCEArithmetic[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ------------------------------------------------------------------
# Torch-only autograd tests
# ------------------------------------------------------------------


class TestAutogradPCEArithmetic(unittest.TestCase):
    """Verify autograd graph preservation through PCE arithmetic."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_pce(self, nvars, max_level, nqoi=1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return PolynomialChaosExpansion(basis, bkd, nqoi)

    def test_autograd_add(self):
        """Autograd through pce1 + pce2: gradient of output w.r.t. coefs."""
        bkd = self._bkd
        pce1 = self._create_pce(1, 3)
        pce2 = self._create_pce(1, 3)
        samples = bkd.asarray(
            np.random.uniform(-1, 1, (1, 5)).astype(np.float64)
        )

        coef1 = torch.randn(pce1.nterms(), 1, dtype=torch.float64,
                            requires_grad=True)

        def func(c):
            pce1.set_coefficients(c)
            result = pce1 + pce2
            return result(samples)

        jac = torch.autograd.functional.jacobian(func, coef1)
        # Jacobian should be non-zero (addition is linear in coef1)
        self.assertGreater(torch.abs(jac).max().item(), 0.0)

    def test_autograd_mul(self):
        """Autograd through pce1 * pce2: gradient of output w.r.t. coefs."""
        bkd = self._bkd
        pce1 = self._create_pce(1, 2)
        pce2 = self._create_pce(1, 2)
        coef2 = bkd.asarray(
            np.random.randn(pce2.nterms(), 1).astype(np.float64)
        )
        pce2.set_coefficients(coef2)
        samples = bkd.asarray(
            np.random.uniform(-1, 1, (1, 5)).astype(np.float64)
        )

        coef1 = torch.randn(pce1.nterms(), 1, dtype=torch.float64,
                            requires_grad=True)

        def func(c):
            pce1.set_coefficients(c)
            result = pce1 * pce2
            return result(samples)

        jac = torch.autograd.functional.jacobian(func, coef1)
        self.assertGreater(torch.abs(jac).max().item(), 0.0)
