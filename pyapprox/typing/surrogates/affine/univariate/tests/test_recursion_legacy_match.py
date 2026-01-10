"""Tests verifying typing recursion coefficients match legacy exactly.

These tests ensure the typing module produces identical recursion
coefficients to the legacy implementation for all polynomial families.

Note: For Jacobi with ncoefs=1, the typing module intentionally differs
from legacy. Legacy returns (0, 2) which is incorrect - you need at least
the first row (a0, b0) to evaluate p_0(x) and compute 1-point quadrature.
The typing module correctly returns (1, 2) for ncoefs=1.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

# Legacy imports
from pyapprox.surrogates.univariate.orthonormal_recursions import (
    jacobi_recurrence as legacy_jacobi,
    hermite_recurrence as legacy_hermite,
    laguerre_recurrence as legacy_laguerre,
)

# Typing imports
from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
    jacobi_recurrence as typing_jacobi,
    hermite_recurrence as typing_hermite,
    laguerre_recurrence as typing_laguerre,
)


class TestRecursionLegacyMatchBase(Generic[Array], unittest.TestCase):
    """Base class for testing recursion coefficients match legacy."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_jacobi_legendre_probability_true(self) -> None:
        """Test Jacobi with alpha=beta=0 (Legendre), probability=True.

        Note: ncoefs=1 is excluded because typing intentionally differs
        from legacy (typing returns (1,2), legacy returns (0,2)).
        """
        for ncoefs in [0, 2, 3, 5, 10, 20]:
            legacy = legacy_jacobi(ncoefs, alpha=0.0, beta=0.0, probability=True)
            typing = typing_jacobi(
                ncoefs,
                self._bkd.asarray(0.0),
                self._bkd.asarray(0.0),
                probability=True,
                bkd=self._bkd,
            )
            typing_np = self._bkd.to_numpy(typing)

            self.assertEqual(
                legacy.shape,
                typing_np.shape,
                f"Shape mismatch for ncoefs={ncoefs}",
            )
            if legacy.size > 0:
                self._bkd.assert_allclose(
                    self._bkd.asarray(legacy),
                    typing,
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_jacobi_legendre_probability_false(self) -> None:
        """Test Jacobi with alpha=beta=0 (Legendre), probability=False.

        Note: ncoefs=1 is excluded because typing intentionally differs.
        """
        for ncoefs in [0, 2, 3, 5, 10, 20]:
            legacy = legacy_jacobi(ncoefs, alpha=0.0, beta=0.0, probability=False)
            typing = typing_jacobi(
                ncoefs,
                self._bkd.asarray(0.0),
                self._bkd.asarray(0.0),
                probability=False,
                bkd=self._bkd,
            )
            typing_np = self._bkd.to_numpy(typing)

            self.assertEqual(
                legacy.shape,
                typing_np.shape,
                f"Shape mismatch for ncoefs={ncoefs}",
            )
            if legacy.size > 0:
                self._bkd.assert_allclose(
                    self._bkd.asarray(legacy),
                    typing,
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_jacobi_general_parameters(self) -> None:
        """Test Jacobi with various alpha, beta values."""
        params = [
            (1.0, 2.0),
            (-0.5, -0.5),  # Chebyshev 1st kind
            (0.5, 0.5),    # Chebyshev 2nd kind
            (0.0, 1.0),
            (2.5, 0.3),
        ]
        for alpha, beta in params:
            for ncoefs in [2, 5, 10]:
                for prob in [True, False]:
                    legacy = legacy_jacobi(
                        ncoefs, alpha=alpha, beta=beta, probability=prob
                    )
                    typing = typing_jacobi(
                        ncoefs,
                        self._bkd.asarray(alpha),
                        self._bkd.asarray(beta),
                        probability=prob,
                        bkd=self._bkd,
                    )
                    typing_np = self._bkd.to_numpy(typing)

                    self.assertEqual(
                        legacy.shape,
                        typing_np.shape,
                        f"Shape mismatch for alpha={alpha}, beta={beta}, "
                        f"ncoefs={ncoefs}, prob={prob}",
                    )
                    if legacy.size > 0:
                        self._bkd.assert_allclose(
                            self._bkd.asarray(legacy),
                            typing,
                            rtol=1e-12,
                            atol=1e-14,
                        )

    def test_jacobi_ncoefs_zero_returns_empty(self) -> None:
        """Test that Jacobi returns empty array for ncoefs=0."""
        for prob in [True, False]:
            typing = typing_jacobi(
                0,
                self._bkd.asarray(0.0),
                self._bkd.asarray(0.0),
                probability=prob,
                bkd=self._bkd,
            )
            self.assertEqual(self._bkd.to_numpy(typing).shape, (0, 2))

    def test_jacobi_ncoefs_one_returns_first_row(self) -> None:
        """Test that Jacobi returns (1, 2) for ncoefs=1.

        This is the CORRECT behavior (unlike legacy which returns (0, 2)).
        The first row contains:
        - a[0] = (beta - alpha) / (alpha + beta + 2) = quadrature point
        - b[0] = 1 (probability) or sqrt(integral of weight) (physics)

        For Legendre (alpha=beta=0): a[0]=0, b[0]=1
        """
        for prob in [True, False]:
            typing = typing_jacobi(
                1,
                self._bkd.asarray(0.0),
                self._bkd.asarray(0.0),
                probability=prob,
                bkd=self._bkd,
            )
            typing_np = self._bkd.to_numpy(typing)
            self.assertEqual(typing_np.shape, (1, 2))

            # Check a[0] = 0 for Legendre
            self._bkd.assert_allclose(
                self._bkd.asarray([typing_np[0, 0]]),
                self._bkd.asarray([0.0]),
                rtol=1e-14,
                atol=1e-14,
            )

            if prob:
                # b[0] = 1 for probability measure
                self._bkd.assert_allclose(
                    self._bkd.asarray([typing_np[0, 1]]),
                    self._bkd.asarray([1.0]),
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_jacobi_ncoefs_one_general_params(self) -> None:
        """Test ncoefs=1 for various Jacobi parameters.

        Verifies a[0] = (beta - alpha) / (alpha + beta + 2).
        """
        params = [
            (1.0, 2.0),
            (-0.5, -0.5),  # Chebyshev 1st kind
            (0.5, 0.5),    # Chebyshev 2nd kind
            (0.0, 1.0),
        ]
        for alpha, beta in params:
            typing = typing_jacobi(
                1,
                self._bkd.asarray(alpha),
                self._bkd.asarray(beta),
                probability=True,
                bkd=self._bkd,
            )
            typing_np = self._bkd.to_numpy(typing)
            self.assertEqual(typing_np.shape, (1, 2))

            expected_a0 = (beta - alpha) / (alpha + beta + 2.0)
            self._bkd.assert_allclose(
                self._bkd.asarray([typing_np[0, 0]]),
                self._bkd.asarray([expected_a0]),
                rtol=1e-14,
                atol=1e-14,
            )

    def test_hermite_probability_true(self) -> None:
        """Test Hermite recursion with probability=True."""
        for ncoefs in [1, 2, 3, 5, 10, 20]:
            legacy = legacy_hermite(ncoefs, rho=0.0, probability=True)
            typing = typing_hermite(
                ncoefs,
                self._bkd.asarray(0.0),
                probability=True,
                bkd=self._bkd,
            )
            typing_np = self._bkd.to_numpy(typing)

            self.assertEqual(
                legacy.shape,
                typing_np.shape,
                f"Shape mismatch for ncoefs={ncoefs}",
            )
            if legacy.size > 0:
                self._bkd.assert_allclose(
                    self._bkd.asarray(legacy),
                    typing,
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_hermite_probability_false(self) -> None:
        """Test Hermite recursion with probability=False."""
        for ncoefs in [1, 2, 3, 5, 10, 20]:
            legacy = legacy_hermite(ncoefs, rho=0.0, probability=False)
            typing = typing_hermite(
                ncoefs,
                self._bkd.asarray(0.0),
                probability=False,
                bkd=self._bkd,
            )
            typing_np = self._bkd.to_numpy(typing)

            self.assertEqual(
                legacy.shape,
                typing_np.shape,
                f"Shape mismatch for ncoefs={ncoefs}",
            )
            if legacy.size > 0:
                self._bkd.assert_allclose(
                    self._bkd.asarray(legacy),
                    typing,
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_hermite_general_rho(self) -> None:
        """Test Hermite with various rho values."""
        for rho in [0.0, 0.5, 1.0, 2.0]:
            for ncoefs in [2, 5, 10]:
                for prob in [True, False]:
                    legacy = legacy_hermite(ncoefs, rho=rho, probability=prob)
                    typing = typing_hermite(
                        ncoefs,
                        self._bkd.asarray(rho),
                        probability=prob,
                        bkd=self._bkd,
                    )
                    typing_np = self._bkd.to_numpy(typing)

                    self.assertEqual(
                        legacy.shape,
                        typing_np.shape,
                        f"Shape mismatch for rho={rho}, ncoefs={ncoefs}, prob={prob}",
                    )
                    if legacy.size > 0:
                        self._bkd.assert_allclose(
                            self._bkd.asarray(legacy),
                            typing,
                            rtol=1e-12,
                            atol=1e-14,
                        )

    def test_laguerre_probability_true(self) -> None:
        """Test Laguerre recursion with probability=True."""
        for ncoefs in [1, 2, 3, 5, 10, 20]:
            legacy = legacy_laguerre(ncoefs, rho=0.0, probability=True)
            typing = typing_laguerre(
                ncoefs,
                rho=0.0,
                bkd=self._bkd,
                probability=True,
            )
            typing_np = self._bkd.to_numpy(typing)

            self.assertEqual(
                legacy.shape,
                typing_np.shape,
                f"Shape mismatch for ncoefs={ncoefs}",
            )
            if legacy.size > 0:
                self._bkd.assert_allclose(
                    self._bkd.asarray(legacy),
                    typing,
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_laguerre_probability_false(self) -> None:
        """Test Laguerre recursion with probability=False."""
        for ncoefs in [1, 2, 3, 5, 10, 20]:
            legacy = legacy_laguerre(ncoefs, rho=0.0, probability=False)
            typing = typing_laguerre(
                ncoefs,
                rho=0.0,
                bkd=self._bkd,
                probability=False,
            )
            typing_np = self._bkd.to_numpy(typing)

            self.assertEqual(
                legacy.shape,
                typing_np.shape,
                f"Shape mismatch for ncoefs={ncoefs}",
            )
            if legacy.size > 0:
                self._bkd.assert_allclose(
                    self._bkd.asarray(legacy),
                    typing,
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_laguerre_general_rho(self) -> None:
        """Test Laguerre with various rho values."""
        for rho in [0.0, 0.5, 1.0, 2.5]:
            for ncoefs in [2, 5, 10]:
                for prob in [True, False]:
                    legacy = legacy_laguerre(ncoefs, rho=rho, probability=prob)
                    typing = typing_laguerre(
                        ncoefs,
                        rho=rho,
                        bkd=self._bkd,
                        probability=prob,
                    )
                    typing_np = self._bkd.to_numpy(typing)

                    self.assertEqual(
                        legacy.shape,
                        typing_np.shape,
                        f"Shape mismatch for rho={rho}, ncoefs={ncoefs}, prob={prob}",
                    )
                    if legacy.size > 0:
                        self._bkd.assert_allclose(
                            self._bkd.asarray(legacy),
                            typing,
                            rtol=1e-12,
                            atol=1e-14,
                        )


class TestRecursionLegacyMatchNumpy(TestRecursionLegacyMatchBase[NDArray[Any]]):
    """NumPy backend tests for recursion legacy matching."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRecursionLegacyMatchTorch(TestRecursionLegacyMatchBase[torch.Tensor]):
    """PyTorch backend tests for recursion legacy matching."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
