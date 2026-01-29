"""Tests for FunctionTrainSensitivity."""

import unittest
from itertools import chain, combinations
from typing import Any, Generic, List, Optional, Sequence

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
)
from pyapprox.typing.surrogates.functiontrain.statistics import (
    FunctionTrainMoments,
    FunctionTrainSensitivity,
)


def all_nonempty_subsets(iterable: Sequence[int]) -> chain[tuple[int, ...]]:
    """Generate all non-empty subsets."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


class TestFunctionTrainSensitivity(Generic[Array], unittest.TestCase):
    """Base class for FunctionTrainSensitivity tests."""

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

    def _create_rank1_pce_ft(
        self,
        nvars: int,
        max_level: int,
        coefficients: Optional[List[Array]] = None,
    ) -> PCEFunctionTrain[Array]:
        """Create a rank-1 PCEFunctionTrain."""
        bkd = self._bkd
        nterms = max_level + 1
        cores: List[FunctionTrainCore[Array]] = []

        for k in range(nvars):
            pce = self._create_univariate_pce(max_level)
            if coefficients is not None:
                coef = coefficients[k]
            else:
                coef = bkd.asarray(np.random.randn(nterms, 1))
            pce = pce.with_params(coef)
            core = FunctionTrainCore([[pce]], bkd)
            cores.append(core)

        ft = FunctionTrain(cores, bkd, nqoi=1)
        return PCEFunctionTrain(ft)

    def _create_moments_and_sensitivity(
        self, pce_ft: PCEFunctionTrain[Array]
    ) -> tuple[FunctionTrainMoments[Array], FunctionTrainSensitivity[Array]]:
        """Create moments and sensitivity objects."""
        moments = FunctionTrainMoments(pce_ft)
        sensitivity = FunctionTrainSensitivity(moments)
        return moments, sensitivity

    def test_construction_success(self) -> None:
        """Test FunctionTrainSensitivity construction succeeds."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        moments = FunctionTrainMoments(pce_ft)
        sensitivity = FunctionTrainSensitivity(moments)

        self.assertIs(sensitivity.moments(), moments)
        self.assertIs(sensitivity.bkd(), self._bkd)

    def test_construction_rejects_non_moments(self) -> None:
        """Test construction fails for non-FunctionTrainMoments."""
        with self.assertRaises(TypeError) as ctx:
            FunctionTrainSensitivity("not moments")  # type: ignore
        self.assertIn("Expected FunctionTrainMoments", str(ctx.exception))

    def test_var_idx_bounds_checking(self) -> None:
        """Test variable index bounds checking."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        with self.assertRaises(IndexError):
            sensitivity.main_effect_index(-1)

        with self.assertRaises(IndexError):
            sensitivity.main_effect_index(3)

        with self.assertRaises(IndexError):
            sensitivity.total_effect_index(-1)

        with self.assertRaises(IndexError):
            sensitivity.total_effect_index(3)

    def test_main_effect_matches_sobol_singleton(self) -> None:
        """S_{k} from sobol_index([k]) matches main_effect_index(k)."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        for k in range(3):
            me = sensitivity.main_effect_index(k)
            su = sensitivity.sobol_index([k])
            bkd.assert_allclose(me, su, rtol=1e-10)

    def test_sobol_sum_equals_one(self) -> None:
        """Sum of all S_u over non-empty u equals 1."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        total = bkd.array([0.0])
        for u in all_nonempty_subsets(range(3)):
            total = total + sensitivity.sobol_index(list(u))
        bkd.assert_allclose(total, bkd.array([1.0]), rtol=1e-10)

    def test_total_equals_sum_of_subsets(self) -> None:
        """S_k^T = Σ_{u ∋ k} S_u."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        for k in range(3):
            total_effect = sensitivity.total_effect_index(k)
            sum_subsets = bkd.array([0.0])
            for u in all_nonempty_subsets(range(3)):
                if k in u:
                    sum_subsets = sum_subsets + sensitivity.sobol_index(list(u))
            bkd.assert_allclose(total_effect, sum_subsets, rtol=1e-10)

    def test_total_geq_main(self) -> None:
        """S_k^T >= S_k for all k."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=3)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        main = sensitivity.all_main_effects()
        total = sensitivity.all_total_effects()
        for k in range(3):
            self.assertTrue(
                float(total[k]) >= float(main[k]) - 1e-10,
                f"S_{k}^T = {total[k]} < S_{k} = {main[k]}"
            )

    def test_indices_in_unit_interval(self) -> None:
        """All Sobol indices should be in [0, 1]."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        main = sensitivity.all_main_effects()
        total = sensitivity.all_total_effects()

        for k in range(3):
            self.assertTrue(float(main[k]) >= -1e-10)
            self.assertTrue(float(main[k]) <= 1.0 + 1e-10)
            self.assertTrue(float(total[k]) >= -1e-10)
            self.assertTrue(float(total[k]) <= 1.0 + 1e-10)

    def test_single_variable_all_main_effect(self) -> None:
        """For single variable, S_0 = 1 and S_0^T = 1."""
        bkd = self._bkd
        # f(x) = a0 + a1*P_1(x) with a1 != 0
        coefficients = [bkd.asarray([[1.0], [2.0]])]
        pce_ft = self._create_rank1_pce_ft(nvars=1, max_level=1, coefficients=coefficients)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        bkd.assert_allclose(sensitivity.main_effect_index(0), bkd.array([1.0]), rtol=1e-10)
        bkd.assert_allclose(sensitivity.total_effect_index(0), bkd.array([1.0]), rtol=1e-10)

    def test_constant_variable_zero_effect(self) -> None:
        """Variable with only constant coefficient has zero main effect."""
        bkd = self._bkd
        # f(x, y) = (a0)(b0 + b1*P_1(y))
        # x has no non-constant terms, so S_x = 0
        coefficients = [
            bkd.asarray([[2.0], [0.0]]),  # x: constant only
            bkd.asarray([[1.0], [1.5]]),  # y: has linear term
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=2, max_level=1, coefficients=coefficients)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        # S_x should be 0 (only constant term)
        bkd.assert_allclose(sensitivity.main_effect_index(0), bkd.array([0.0]), atol=1e-12)
        # S_y should be 1 (accounts for all variance)
        bkd.assert_allclose(sensitivity.main_effect_index(1), bkd.array([1.0]), rtol=1e-10)

    def test_additive_function_analytical(self) -> None:
        """Test analytical Sobol indices for additive-like function.

        For rank-1 FT f(x,y) = (a0 + a1*P_1(x))(b0 + b1*P_1(y)):
        - This is NOT additive, it's multiplicative
        - Var[f] = (a0² + a1²)(b0² + b1²) - a0²b0²
        - V_x = a1² * b0²  (main effect of x)
        - V_y = a0² * b1²  (main effect of y)
        - V_{xy} = a1² * b1²  (interaction)
        """
        bkd = self._bkd
        a0, a1 = 2.0, 1.0
        b0, b1 = 3.0, 0.5
        coefficients = [
            bkd.asarray([[a0], [a1]]),
            bkd.asarray([[b0], [b1]]),
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=2, max_level=1, coefficients=coefficients)
        moments, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        # Analytical variance
        total_var = (a0**2 + a1**2) * (b0**2 + b1**2) - (a0 * b0)**2
        bkd.assert_allclose(moments.variance(), bkd.asarray([total_var]), rtol=1e-12)

        # Main effect variances
        V_x = a1**2 * b0**2
        V_y = a0**2 * b1**2
        V_xy = a1**2 * b1**2

        bkd.assert_allclose(
            sensitivity.main_effect_variance(0), bkd.asarray([V_x]), rtol=1e-12
        )
        bkd.assert_allclose(
            sensitivity.main_effect_variance(1), bkd.asarray([V_y]), rtol=1e-12
        )

        # Interaction variance
        bkd.assert_allclose(
            sensitivity.sobol_variance([0, 1]), bkd.asarray([V_xy]), rtol=1e-12
        )

        # Sobol indices
        S_x = V_x / total_var
        S_y = V_y / total_var
        S_xy = V_xy / total_var

        bkd.assert_allclose(sensitivity.main_effect_index(0), bkd.asarray([S_x]), rtol=1e-12)
        bkd.assert_allclose(sensitivity.main_effect_index(1), bkd.asarray([S_y]), rtol=1e-12)
        bkd.assert_allclose(sensitivity.sobol_index([0, 1]), bkd.asarray([S_xy]), rtol=1e-12)

        # Verify sum = 1
        bkd.assert_allclose(
            bkd.asarray([S_x + S_y + S_xy]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_three_variable_analytical(self) -> None:
        """Test with 3 variables - verify sum of all indices = 1."""
        bkd = self._bkd
        coefficients = [
            bkd.asarray([[1.0], [0.5]]),
            bkd.asarray([[2.0], [1.0]]),
            bkd.asarray([[1.5], [0.3]]),
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=1, coefficients=coefficients)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        # Compute all Sobol indices
        S = {}
        for u in all_nonempty_subsets(range(3)):
            S[u] = float(sensitivity.sobol_index(list(u))[0])

        # Sum should equal 1
        total = sum(S.values())
        bkd.assert_allclose(bkd.asarray([total]), bkd.asarray([1.0]), rtol=1e-10)

    def test_all_main_effects_shape(self) -> None:
        """Test all_main_effects returns correct shape."""
        pce_ft = self._create_rank1_pce_ft(nvars=4, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        main = sensitivity.all_main_effects()
        self.assertEqual(main.shape, (4,))

    def test_all_total_effects_shape(self) -> None:
        """Test all_total_effects returns correct shape."""
        pce_ft = self._create_rank1_pce_ft(nvars=4, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        total = sensitivity.all_total_effects()
        self.assertEqual(total.shape, (4,))

    def test_output_shapes(self) -> None:
        """Test individual index methods return correct shapes."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        self.assertEqual(sensitivity.main_effect_variance(0).shape, (1,))
        self.assertEqual(sensitivity.main_effect_index(0).shape, (1,))
        self.assertEqual(sensitivity.total_effect_variance(0).shape, (1,))
        self.assertEqual(sensitivity.total_effect_index(0).shape, (1,))
        self.assertEqual(sensitivity.sobol_variance([0, 1]).shape, (1,))
        self.assertEqual(sensitivity.sobol_index([0, 1]).shape, (1,))


class TestFunctionTrainSensitivityNumpy(TestFunctionTrainSensitivity[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionTrainSensitivityTorch(TestFunctionTrainSensitivity[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
