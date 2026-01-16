"""Tests for Leja weighting strategies."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.variables.univariate.scipy_continuous import (
    ContinuousScipyRandomVariable1D,
)


class TestChristoffelWeighting(Generic[Array], unittest.TestCase):
    """Tests for ChristoffelWeighting."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_weights_shape(self) -> None:
        """Test that weights have correct shape."""
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(self._bkd)
        samples = self._bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self._bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        self.assertEqual(weights.shape, (3, 1))

    def test_weights_positive(self) -> None:
        """Test that weights are positive."""
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(self._bkd)
        samples = self._bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self._bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        self.assertTrue(self._bkd.all_bool(weights > 0))

    def test_jacobian_shape(self) -> None:
        """Test that Jacobian has correct shape."""
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(self._bkd)
        samples = self._bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self._bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        basis_jac = self._bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        jac = weighting.jacobian(samples, basis_values, basis_jac)

        self.assertEqual(jac.shape, (3, 1))


class TestPDFWeighting(Generic[Array], unittest.TestCase):
    """Tests for PDFWeighting."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_weights_shape(self) -> None:
        """Test that PDF weights have correct shape."""
        from pyapprox.typing.surrogates.affine.leja import PDFWeighting

        # Use typing wrapper for scipy distribution
        rv = ContinuousScipyRandomVariable1D(stats.uniform(-1, 2), self._bkd)

        # PDFWeighting expects a callable that returns backend arrays
        def pdf_callable(samples: Array) -> Array:
            return rv.pdf(samples)

        weighting = PDFWeighting(self._bkd, pdf_callable)
        samples = self._bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self._bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        self.assertEqual(weights.shape, (3, 1))

    def test_weights_match_pdf(self) -> None:
        """Test that weights match the PDF values."""
        from pyapprox.typing.surrogates.affine.leja import PDFWeighting

        # Use typing wrapper for scipy distribution
        rv = ContinuousScipyRandomVariable1D(stats.norm(0, 1), self._bkd)

        # PDFWeighting expects a callable that returns backend arrays
        def pdf_callable(samples: Array) -> Array:
            return rv.pdf(samples)

        weighting = PDFWeighting(self._bkd, pdf_callable)
        samples = self._bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = self._bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        # Get expected PDF values using the typed distribution
        expected = rv.pdf(samples[0, :])
        self._bkd.assert_allclose(weights[:, 0], expected, rtol=1e-10)


# NumPy backend tests
class TestChristoffelWeightingNumpy(TestChristoffelWeighting[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPDFWeightingNumpy(TestPDFWeighting[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestChristoffelWeightingTorch(TestChristoffelWeighting[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestPDFWeightingTorch(TestPDFWeighting[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
