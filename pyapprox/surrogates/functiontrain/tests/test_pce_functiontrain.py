"""Tests for PCEFunctionTrain and create_pce_functiontrain."""

import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability import (
    BetaMarginal,
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import (
    FunctionTrain,
    PCEFunctionTrain,
    PCEFunctionTrainCore,
    create_pce_functiontrain,
)
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


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

    def _create_rank1_ft(self, nvars: int, max_level: int) -> FunctionTrain[Array]:
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


class TestCreatePceFunctiontrain(Generic[Array], unittest.TestCase):
    """Tests for create_pce_functiontrain factory."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_uniform_marginals_construction(self) -> None:
        """Test construction with uniform marginals (Legendre)."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        ft = create_pce_functiontrain(marginals, max_level=5, ranks=[2, 2], bkd=bkd)

        self.assertEqual(ft.nvars(), 3)
        self.assertEqual(ft.nqoi(), 1)
        # All cores should have nterms = max_level + 1 = 6 for each entry
        for core in ft.cores():
            self.assertEqual(core.get_nterms(0, 0), 6)

    def test_mixed_marginals_construction(self) -> None:
        """Test construction with mixed marginals (Legendre, Hermite, Jacobi)."""
        bkd = self._bkd
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),  # Legendre
            GaussianMarginal(0.0, 1.0, bkd),  # Hermite
            BetaMarginal(2.0, 5.0, bkd),  # Jacobi
        ]
        ft = create_pce_functiontrain(marginals, max_level=5, ranks=[2, 2], bkd=bkd)

        self.assertEqual(ft.nvars(), 3)
        self.assertEqual(ft.nqoi(), 1)

    def test_mixed_marginals_evaluation(self) -> None:
        """Test FT with mixed marginals can be evaluated."""
        bkd = self._bkd
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
            BetaMarginal(2.0, 5.0, bkd),
        ]
        ft = create_pce_functiontrain(marginals, max_level=3, ranks=[2, 2], bkd=bkd)

        # Create samples in each variable's domain
        nsamples = 10
        samples = bkd.vstack(
            [
                bkd.asarray(np.random.uniform(-1, 1, nsamples)),  # Uniform [-1, 1]
                bkd.asarray(np.random.randn(nsamples)),  # Gaussian
                bkd.asarray(np.random.beta(2.0, 5.0, nsamples)),  # Beta [0, 1]
            ]
        )

        result = ft(samples)
        self.assertEqual(result.shape, (1, nsamples))

    def test_per_core_max_level(self) -> None:
        """Test per-core polynomial degrees."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        ft = create_pce_functiontrain(
            marginals, max_level=[3, 5, 4], ranks=[2, 2], bkd=bkd
        )

        # Core 0: degree 3 → 4 terms
        # Core 1: degree 5 → 6 terms
        # Core 2: degree 4 → 5 terms
        self.assertEqual(ft.cores()[0].get_nterms(0, 0), 4)
        self.assertEqual(ft.cores()[1].get_nterms(0, 0), 6)
        self.assertEqual(ft.cores()[2].get_nterms(0, 0), 5)

    def test_pce_functiontrain_wrapping(self) -> None:
        """Test FT from mixed marginals can be wrapped in PCEFunctionTrain."""
        bkd = self._bkd
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            GaussianMarginal(0.0, 1.0, bkd),
        ]
        ft = create_pce_functiontrain(marginals, max_level=4, ranks=[2], bkd=bkd)

        # Should not raise
        pce_ft = PCEFunctionTrain(ft)
        self.assertEqual(pce_ft.nvars(), 2)
        self.assertEqual(len(pce_ft.pce_cores()), 2)

    def test_error_empty_marginals(self) -> None:
        """Test error on empty marginals."""
        bkd = self._bkd
        with self.assertRaises(ValueError) as ctx:
            create_pce_functiontrain([], max_level=3, ranks=[], bkd=bkd)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_error_wrong_ranks_length(self) -> None:
        """Test error on wrong ranks length."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        with self.assertRaises(ValueError) as ctx:
            create_pce_functiontrain(marginals, max_level=3, ranks=[2], bkd=bkd)
        self.assertIn("ranks", str(ctx.exception).lower())

    def test_error_wrong_max_level_length(self) -> None:
        """Test error on wrong max_level sequence length."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
        with self.assertRaises(ValueError) as ctx:
            create_pce_functiontrain(marginals, max_level=[3, 4], ranks=[2, 2], bkd=bkd)
        self.assertIn("max_level", str(ctx.exception).lower())

    def test_init_scale_zero(self) -> None:
        """Test zero initialization."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        ft = create_pce_functiontrain(
            marginals, max_level=3, ranks=[2], bkd=bkd, init_scale=0.0
        )

        # With zero init, all coefficients should be zero
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 5)))
        result = ft(samples)
        bkd.assert_allclose(result, bkd.zeros((1, 5)), atol=1e-12)


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


class TestCreatePceFunctiontrainNumpy(TestCreatePceFunctiontrain[NDArray[Any]]):
    """NumPy backend tests for create_pce_functiontrain."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCreatePceFunctiontrainTorch(TestCreatePceFunctiontrain[torch.Tensor]):
    """PyTorch backend tests for create_pce_functiontrain."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
