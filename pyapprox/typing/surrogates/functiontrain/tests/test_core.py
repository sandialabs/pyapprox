"""Tests for FunctionTrainCore."""

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
from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.probability import UniformMarginal

from pyapprox.typing.surrogates.functiontrain.core import FunctionTrainCore


class TestFunctionTrainCore(Generic[Array], unittest.TestCase):
    """Base class for FunctionTrainCore tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_univariate_expansion(
        self, max_level: int, nqoi: int = 1
    ) -> BasisExpansion[Array]:
        """Create a univariate polynomial expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_core_creation(self) -> None:
        """Test basic core creation."""
        bkd = self._bkd
        # Create 2x3 grid of expansions
        r_left, r_right = 2, 3
        basisexps = []
        for _ in range(r_left):
            row = [self._create_univariate_expansion(2) for _ in range(r_right)]
            basisexps.append(row)

        core = FunctionTrainCore(basisexps, bkd)

        self.assertEqual(core.ranks(), (r_left, r_right))
        self.assertEqual(core.nqoi(), 1)

    def test_core_ranks(self) -> None:
        """Test various rank configurations."""
        bkd = self._bkd

        # (1, 2) - first core shape
        basisexps_1_2 = [[self._create_univariate_expansion(2) for _ in range(2)]]
        core_1_2 = FunctionTrainCore(basisexps_1_2, bkd)
        self.assertEqual(core_1_2.ranks(), (1, 2))

        # (2, 1) - last core shape
        basisexps_2_1 = [
            [self._create_univariate_expansion(2)],
            [self._create_univariate_expansion(2)],
        ]
        core_2_1 = FunctionTrainCore(basisexps_2_1, bkd)
        self.assertEqual(core_2_1.ranks(), (2, 1))

        # (2, 2) - middle core shape
        basisexps_2_2 = [
            [self._create_univariate_expansion(2) for _ in range(2)],
            [self._create_univariate_expansion(2) for _ in range(2)],
        ]
        core_2_2 = FunctionTrainCore(basisexps_2_2, bkd)
        self.assertEqual(core_2_2.ranks(), (2, 2))

    def test_core_call(self) -> None:
        """Test core evaluation."""
        bkd = self._bkd
        # Create (2, 3) core
        basisexps = [
            [self._create_univariate_expansion(2) for _ in range(3)],
            [self._create_univariate_expansion(2) for _ in range(3)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))
        result = core(samples)

        # Shape should be (r_left, r_right, nsamples, nqoi)
        self.assertEqual(result.shape, (2, 3, 5, 1))

    def test_nparams(self) -> None:
        """Test parameter counting."""
        bkd = self._bkd
        max_level = 2
        nterms = max_level + 1  # For univariate: 0, 1, 2

        # (2, 3) core with nqoi=1
        basisexps = [
            [self._create_univariate_expansion(max_level) for _ in range(3)],
            [self._create_univariate_expansion(max_level) for _ in range(3)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        expected = 2 * 3 * nterms * 1  # r_left * r_right * nterms * nqoi
        self.assertEqual(core.nparams(), expected)

    def test_with_params_roundtrip(self) -> None:
        """Test flatten/unflatten roundtrip."""
        bkd = self._bkd
        basisexps = [
            [self._create_univariate_expansion(2) for _ in range(2)],
            [self._create_univariate_expansion(2) for _ in range(2)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        # Set random parameters
        nparams = core.nparams()
        random_params = bkd.asarray(np.random.randn(nparams))

        # Apply and flatten
        new_core = core.with_params(random_params)
        recovered_params = new_core._flatten_params()

        bkd.assert_allclose(recovered_params, random_params, rtol=1e-12)

    def test_with_params_immutability(self) -> None:
        """Test that with_params doesn't modify original."""
        bkd = self._bkd
        basisexps = [
            [self._create_univariate_expansion(2) for _ in range(2)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        # Get original params
        original_params = bkd.copy(core._flatten_params())

        # Create new core with different params
        new_params = bkd.asarray(np.random.randn(core.nparams()))
        _ = core.with_params(new_params)

        # Original should be unchanged
        bkd.assert_allclose(core._flatten_params(), original_params, rtol=1e-12)

    def test_basis_matrix(self) -> None:
        """Test basis matrix extraction."""
        bkd = self._bkd
        basisexps = [
            [self._create_univariate_expansion(2) for _ in range(2)],
            [self._create_univariate_expansion(2) for _ in range(2)],
        ]
        core = FunctionTrainCore(basisexps, bkd)

        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))
        basis_mat = core.basis_matrix(samples, 0, 1)

        # Shape: (nsamples, nterms)
        self.assertEqual(basis_mat.shape[0], 5)
        self.assertEqual(basis_mat.shape[1], 3)  # max_level + 1


class TestFunctionTrainCoreNumpy(TestFunctionTrainCore[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionTrainCoreTorch(TestFunctionTrainCore[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
