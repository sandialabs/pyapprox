"""Tests for PCEFunctionTrain."""

import unittest
from typing import Any, Generic, List

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
from pyapprox.typing.surrogates.functiontrain import (
    FunctionTrain,
    PCEFunctionTrain,
    PCEFunctionTrainCore,
)


class TestPCEFunctionTrain(Generic[Array], unittest.TestCase):
    """Base class for PCEFunctionTrain tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_univariate_pce(
        self, max_level: int, nqoi: int = 1
    ) -> BasisExpansion[Array]:
        """Create a univariate PCE expansion with Legendre basis."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_rank1_ft(
        self, nvars: int, max_level: int
    ) -> FunctionTrain[Array]:
        """Create a rank-1 FunctionTrain with random PCE coefficients."""
        bkd = self._bkd
        nterms = max_level + 1
        cores: List[FunctionTrainCore[Array]] = []

        for _ in range(nvars):
            pce = self._create_univariate_pce(max_level)
            coef = bkd.asarray(np.random.randn(nterms, 1))
            pce = pce.with_params(coef)
            core = FunctionTrainCore([[pce]], bkd)
            cores.append(core)

        return FunctionTrain(cores, bkd, nqoi=1)

    def test_construction_success(self) -> None:
        """Test PCEFunctionTrain construction succeeds for valid FT."""
        ft = self._create_rank1_ft(nvars=3, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        self.assertEqual(pce_ft.nvars(), 3)
        self.assertEqual(pce_ft.nqoi(), 1)
        self.assertEqual(len(pce_ft.pce_cores()), 3)

    def test_construction_rejects_non_functiontrain(self) -> None:
        """Test construction fails for non-FunctionTrain input."""
        with self.assertRaises(TypeError) as ctx:
            PCEFunctionTrain("not a function train")  # type: ignore
        self.assertIn("Expected FunctionTrain", str(ctx.exception))

    def test_construction_rejects_multi_qoi(self) -> None:
        """Test construction fails for nqoi > 1."""
        bkd = self._bkd
        # Create FT with nqoi=2
        pce = self._create_univariate_pce(2, nqoi=2)
        core = FunctionTrainCore([[pce]], bkd)
        ft = FunctionTrain([core], bkd, nqoi=2)

        with self.assertRaises(ValueError) as ctx:
            PCEFunctionTrain(ft)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_ft_accessor(self) -> None:
        """Test ft() returns underlying FunctionTrain."""
        ft = self._create_rank1_ft(nvars=2, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        self.assertIs(pce_ft.ft(), ft)

    def test_pce_cores_type(self) -> None:
        """Test pce_cores returns list of PCEFunctionTrainCore."""
        ft = self._create_rank1_ft(nvars=3, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        cores = pce_ft.pce_cores()
        self.assertEqual(len(cores), 3)
        for core in cores:
            self.assertIsInstance(core, PCEFunctionTrainCore)

    def test_evaluation_delegates_to_ft(self) -> None:
        """Test __call__ delegates to underlying FunctionTrain."""
        bkd = self._bkd
        ft = self._create_rank1_ft(nvars=3, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        samples = bkd.asarray(np.random.uniform(-1, 1, (3, 5)))

        ft_result = ft(samples)
        pce_ft_result = pce_ft(samples)

        bkd.assert_allclose(pce_ft_result, ft_result, rtol=1e-12)

    def test_bkd_accessor(self) -> None:
        """Test bkd() returns correct backend."""
        ft = self._create_rank1_ft(nvars=2, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        self.assertIs(pce_ft.bkd(), self._bkd)

    def test_repr(self) -> None:
        """Test __repr__ provides useful info."""
        ft = self._create_rank1_ft(nvars=3, max_level=2)
        pce_ft = PCEFunctionTrain(ft)

        repr_str = repr(pce_ft)
        self.assertIn("PCEFunctionTrain", repr_str)
        self.assertIn("nvars=3", repr_str)
        self.assertIn("nqoi=1", repr_str)


class TestPCEFunctionTrainNumpy(TestPCEFunctionTrain[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPCEFunctionTrainTorch(TestPCEFunctionTrain[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
