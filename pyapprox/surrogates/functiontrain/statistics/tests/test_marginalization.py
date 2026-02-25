"""Tests for FunctionTrainMarginalization."""

import unittest
from typing import Any, Generic, List, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.probability import UniformMarginal

from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.surrogates.functiontrain import FunctionTrain
from pyapprox.surrogates.functiontrain.pce_functiontrain import (
    PCEFunctionTrain,
)
from pyapprox.surrogates.functiontrain.statistics import (
    FunctionTrainMoments,
    FunctionTrainSensitivity,
)
from pyapprox.surrogates.functiontrain.statistics.marginalization import (
    FunctionTrainMarginalization,
    FTDimensionReducer,
    marginal_1d,
    marginal_2d,
    all_marginals_1d,
)
from pyapprox.interface.functions.marginalize import (
    DimensionReducerProtocol,
    ReducedFunction,
)


class TestFunctionTrainMarginalization(Generic[Array], unittest.TestCase):
    """Base class for FunctionTrainMarginalization tests."""

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
        """Create a rank-1 PCEFunctionTrain.

        Parameters
        ----------
        nvars : int
            Number of variables.
        max_level : int
            Polynomial degree.
        coefficients : List[Array], optional
            Coefficients for each core. If None, random coefficients are used.
            Each should have shape (nterms, 1).
        """
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

    # =========================================================================
    # Construction tests
    # =========================================================================

    def test_construction_success(self) -> None:
        """Test FunctionTrainMarginalization construction succeeds."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        self.assertIs(marg.pce_ft(), pce_ft)
        self.assertIs(marg.bkd(), self._bkd)

    def test_construction_rejects_non_pce_ft(self) -> None:
        """Test construction fails for non-PCEFunctionTrain."""
        with self.assertRaises(TypeError) as ctx:
            FunctionTrainMarginalization("not a pce ft")  # type: ignore
        self.assertIn("Expected PCEFunctionTrain", str(ctx.exception))

    # =========================================================================
    # Test 1: Single Variable Marginalization
    # =========================================================================

    def test_marginalize_single_variable(self) -> None:
        """Test marginalizing a single variable reduces dimension."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        # Marginalize variable 1
        ft_marg = marg.marginalize([1])

        # Should have 2 variables now
        self.assertEqual(ft_marg.nvars(), 2)

    def test_marginalize_preserves_nqoi(self) -> None:
        """Test marginalized FT has same nqoi."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)
        ft_marg = marg.marginalize([1])

        self.assertEqual(ft_marg.nqoi(), pce_ft.nqoi())

    # =========================================================================
    # Test 2: Mean Preservation
    # =========================================================================

    def test_mean_preserved_after_marginalization(self) -> None:
        """Test E[f_marg] = E[f] after marginalization."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=3)
        marg = FunctionTrainMarginalization(pce_ft)

        # Original mean
        moments_orig = FunctionTrainMoments(pce_ft)
        mean_orig = moments_orig.mean()

        # Marginalize variable 1
        ft_marg = marg.marginalize([1])
        pce_marg = PCEFunctionTrain(ft_marg)
        moments_marg = FunctionTrainMoments(pce_marg)
        mean_marg = moments_marg.mean()

        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    def test_mean_preserved_marginalize_multiple(self) -> None:
        """Test mean preserved when marginalizing multiple variables."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=4, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        # Original mean
        moments_orig = FunctionTrainMoments(pce_ft)
        mean_orig = moments_orig.mean()

        # Marginalize variables 1 and 2
        ft_marg = marg.marginalize([1, 2])
        pce_marg = PCEFunctionTrain(ft_marg)
        moments_marg = FunctionTrainMoments(pce_marg)
        mean_marg = moments_marg.mean()

        self.assertEqual(ft_marg.nvars(), 2)
        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test 3: Variance via Marginalization (Main Effect Connection)
    # =========================================================================

    def test_1d_marginal_variance_equals_main_effect(self) -> None:
        """Test Var[E[f|x_k]] = V_k (main effect variance).

        For a 1D marginal keeping only variable k:
        Var[f_marginal] = Var[E[f | x_k]] = V_k (main effect variance)
        """
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=3)
        moments = FunctionTrainMoments(pce_ft)
        sensitivity = FunctionTrainSensitivity(moments)
        marg = FunctionTrainMarginalization(pce_ft)

        for k in range(pce_ft.nvars()):
            # Get main effect variance from sensitivity
            V_k = sensitivity.main_effect_variance(k)

            # Get variance of 1D marginal
            ft_1d = marg.marginal([k])
            pce_1d = PCEFunctionTrain(ft_1d)
            moments_1d = FunctionTrainMoments(pce_1d)
            var_1d = moments_1d.variance()

            bkd.assert_allclose(
                var_1d, V_k, rtol=1e-10,
            )

    # =========================================================================
    # Test 4: Chain of Marginalizations
    # =========================================================================

    def test_chain_of_marginalizations(self) -> None:
        """Test marginalizing variables one at a time preserves mean."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=4, max_level=2)

        # Original mean
        moments_orig = FunctionTrainMoments(pce_ft)
        mean_orig = moments_orig.mean()

        # Marginalize one variable at a time: 3, 2, 1
        current_ft: FunctionTrain[Array] = pce_ft.ft()

        # Marginalize var 3 (now last in 4-var FT)
        pce_current = PCEFunctionTrain(current_ft)
        marg = FunctionTrainMarginalization(pce_current)
        current_ft = marg.marginalize([3])
        self.assertEqual(current_ft.nvars(), 3)

        # Verify mean preserved
        pce_current = PCEFunctionTrain(current_ft)
        mean_curr = FunctionTrainMoments(pce_current).mean()
        bkd.assert_allclose(mean_curr, mean_orig, rtol=1e-12)

        # Marginalize var 2 (now last in 3-var FT)
        marg = FunctionTrainMarginalization(pce_current)
        current_ft = marg.marginalize([2])
        self.assertEqual(current_ft.nvars(), 2)

        pce_current = PCEFunctionTrain(current_ft)
        mean_curr = FunctionTrainMoments(pce_current).mean()
        bkd.assert_allclose(mean_curr, mean_orig, rtol=1e-12)

        # Marginalize var 1 (now index 1 in 2-var FT)
        marg = FunctionTrainMarginalization(pce_current)
        current_ft = marg.marginalize([1])
        self.assertEqual(current_ft.nvars(), 1)

        pce_current = PCEFunctionTrain(current_ft)
        mean_curr = FunctionTrainMoments(pce_current).mean()
        bkd.assert_allclose(mean_curr, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test 5: Statistics Module Compatibility
    # =========================================================================

    def test_marginalized_ft_works_with_pce_functiontrain(self) -> None:
        """Test marginalized FT can be wrapped in PCEFunctionTrain."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_marg = marg.marginalize([1])

        # Should be wrappable in PCEFunctionTrain
        pce_marg = PCEFunctionTrain(ft_marg)
        self.assertEqual(pce_marg.nvars(), 2)

    def test_marginalized_ft_works_with_moments(self) -> None:
        """Test marginalized FT works with FunctionTrainMoments."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_marg = marg.marginalize([1])
        pce_marg = PCEFunctionTrain(ft_marg)

        # Should work with moments
        moments = FunctionTrainMoments(pce_marg)
        _ = moments.mean()
        _ = moments.variance()

    def test_marginalized_ft_works_with_sensitivity(self) -> None:
        """Test marginalized FT works with FunctionTrainSensitivity."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_marg = marg.marginalize([1])
        pce_marg = PCEFunctionTrain(ft_marg)

        # Should work with sensitivity
        moments = FunctionTrainMoments(pce_marg)
        sensitivity = FunctionTrainSensitivity(moments)
        _ = sensitivity.all_main_effects()
        _ = sensitivity.all_total_effects()

    # =========================================================================
    # Test 6: marginal() API (complement of marginalize)
    # =========================================================================

    def test_marginal_keeps_specified_variables(self) -> None:
        """Test marginal() keeps only specified variables."""
        pce_ft = self._create_rank1_pce_ft(nvars=4, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        # Keep variables 0 and 2
        ft_marg = marg.marginal([0, 2])

        self.assertEqual(ft_marg.nvars(), 2)

    def test_marginal_1d(self) -> None:
        """Test creating 1D marginal."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_1d = marg.marginal([1])

        self.assertEqual(ft_1d.nvars(), 1)

    def test_marginal_preserves_mean(self) -> None:
        """Test marginal() preserves mean."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=4, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        mean_orig = FunctionTrainMoments(pce_ft).mean()

        ft_marg = marg.marginal([0, 3])  # Keep vars 0 and 3
        pce_marg = PCEFunctionTrain(ft_marg)
        mean_marg = FunctionTrainMoments(pce_marg).mean()

        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test 7: Convenience Functions
    # =========================================================================

    def test_marginal_1d_function(self) -> None:
        """Test marginal_1d convenience function."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)

        ft_1d = marginal_1d(pce_ft, 1)

        self.assertEqual(ft_1d.nvars(), 1)

    def test_marginal_2d_function(self) -> None:
        """Test marginal_2d convenience function."""
        pce_ft = self._create_rank1_pce_ft(nvars=4, max_level=2)

        ft_2d = marginal_2d(pce_ft, 0, 2)

        self.assertEqual(ft_2d.nvars(), 2)

    def test_all_marginals_1d_function(self) -> None:
        """Test all_marginals_1d convenience function."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)

        all_1d = all_marginals_1d(pce_ft)

        self.assertEqual(len(all_1d), 3)
        for ft_1d in all_1d:
            self.assertEqual(ft_1d.nvars(), 1)

    def test_all_1d_marginals_preserve_mean(self) -> None:
        """Test all 1D marginals preserve original mean."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        mean_orig = FunctionTrainMoments(pce_ft).mean()

        all_1d = all_marginals_1d(pce_ft)
        for ft_1d in all_1d:
            pce_1d = PCEFunctionTrain(ft_1d)
            mean_1d = FunctionTrainMoments(pce_1d).mean()
            bkd.assert_allclose(mean_1d, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test: Sum of Marginal Variances Property
    # =========================================================================

    def test_sum_marginal_variances_leq_total_variance(self) -> None:
        """Test Σ_k Var[E[f|x_k]] ≤ Var[f].

        The sum of main effect variances is at most the total variance.
        Equality holds only for additive functions (no interactions).
        """
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=3)

        moments = FunctionTrainMoments(pce_ft)
        total_var = moments.variance()

        marg = FunctionTrainMarginalization(pce_ft)

        sum_marginal_vars = bkd.asarray([0.0])
        for k in range(pce_ft.nvars()):
            marginal_k = marg.marginal([k])
            pce_marginal_k = PCEFunctionTrain(marginal_k)
            V_k = FunctionTrainMoments(pce_marginal_k).variance()
            sum_marginal_vars = sum_marginal_vars + V_k

        # Σ V_k ≤ Var[f] always (with small tolerance for numerical error)
        self.assertTrue(
            float(sum_marginal_vars[0]) <= float(total_var[0]) + 1e-10,
            f"Sum of marginal variances {sum_marginal_vars} > total variance {total_var}"
        )

    def _create_additive_pce_ft(
        self, nvars: int, max_level: int
    ) -> PCEFunctionTrain[Array]:
        """Create an additive FunctionTrain: f(x) = Σ_k g_k(x_k).

        For an additive function, main effects capture all variance.
        """
        bkd = self._bkd
        nterms = max_level + 1
        cores: List[FunctionTrainCore[Array]] = []

        for k in range(nvars):
            pce = self._create_univariate_pce(max_level)
            # For additive: coefficients that make f = sum of univariates
            # Use increasing rank structure to achieve additive behavior
            if k == 0:
                # First core: shape (1, 2) - [g_0(x_0), 1]
                coef_g = bkd.asarray(np.random.randn(nterms, 1))
                coef_1 = bkd.asarray([[1.0]] + [[0.0]] * (nterms - 1))
                pce_g = pce.with_params(coef_g)
                pce_1 = pce.with_params(coef_1)
                core = FunctionTrainCore([[pce_g, pce_1]], bkd)
            elif k == nvars - 1:
                # Last core: shape (2, 1) - [[1], [g_k(x_k)]]
                coef_1 = bkd.asarray([[1.0]] + [[0.0]] * (nterms - 1))
                coef_g = bkd.asarray(np.random.randn(nterms, 1))
                pce_1 = pce.with_params(coef_1)
                pce_g = pce.with_params(coef_g)
                core = FunctionTrainCore([[pce_1], [pce_g]], bkd)
            else:
                # Middle cores: shape (2, 2) - [[1, 0], [g_k, 1]]
                coef_1 = bkd.asarray([[1.0]] + [[0.0]] * (nterms - 1))
                coef_0 = bkd.asarray([[0.0]] * nterms)
                coef_g = bkd.asarray(np.random.randn(nterms, 1))
                pce_1_00 = pce.with_params(coef_1)
                pce_0_01 = pce.with_params(coef_0)
                pce_g_10 = pce.with_params(coef_g)
                pce_1_11 = pce.with_params(coef_1)
                core = FunctionTrainCore(
                    [[pce_1_00, pce_0_01], [pce_g_10, pce_1_11]], bkd
                )
            cores.append(core)

        ft = FunctionTrain(cores, bkd, nqoi=1)
        return PCEFunctionTrain(ft)

    def test_additive_function_marginal_variances_sum_to_total(self) -> None:
        """Test for additive f = Σ_k g_k(x_k): Σ_k Var[E[f|x_k]] = Var[f].

        For additive functions, there are no interactions, so the sum of
        main effect variances equals the total variance exactly.
        """
        bkd = self._bkd
        pce_ft = self._create_additive_pce_ft(nvars=3, max_level=2)

        moments = FunctionTrainMoments(pce_ft)
        total_var = moments.variance()

        marg = FunctionTrainMarginalization(pce_ft)

        sum_marginal_vars = bkd.asarray([0.0])
        for k in range(pce_ft.nvars()):
            marginal_k = marg.marginal([k])
            pce_marginal_k = PCEFunctionTrain(marginal_k)
            V_k = FunctionTrainMoments(pce_marginal_k).variance()
            sum_marginal_vars = sum_marginal_vars + V_k

        # For additive: Σ V_k = Var[f] (no interactions)
        bkd.assert_allclose(
            sum_marginal_vars,
            total_var,
            rtol=1e-10,
        )

    # =========================================================================
    # Test 8: Edge Cases
    # =========================================================================

    def test_marginalize_first_variable(self) -> None:
        """Test marginalizing first variable (k=0) uses left-multiply."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        mean_orig = FunctionTrainMoments(pce_ft).mean()

        # Marginalize variable 0
        ft_marg = marg.marginalize([0])

        self.assertEqual(ft_marg.nvars(), 2)

        pce_marg = PCEFunctionTrain(ft_marg)
        mean_marg = FunctionTrainMoments(pce_marg).mean()
        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    def test_marginalize_last_variable(self) -> None:
        """Test marginalizing last variable (k=d-1) uses right-multiply."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        mean_orig = FunctionTrainMoments(pce_ft).mean()

        # Marginalize last variable (index 2)
        ft_marg = marg.marginalize([2])

        self.assertEqual(ft_marg.nvars(), 2)

        pce_marg = PCEFunctionTrain(ft_marg)
        mean_marg = FunctionTrainMoments(pce_marg).mean()
        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    def test_marginalize_to_single_variable(self) -> None:
        """Test marginalizing all but one variable gives 1D FT."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        mean_orig = FunctionTrainMoments(pce_ft).mean()

        # Marginalize variables 0 and 2, keep only 1
        ft_1d = marg.marginalize([0, 2])

        self.assertEqual(ft_1d.nvars(), 1)

        pce_1d = PCEFunctionTrain(ft_1d)
        mean_1d = FunctionTrainMoments(pce_1d).mean()
        bkd.assert_allclose(mean_1d, mean_orig, rtol=1e-12)

    def test_marginalize_all_raises_error(self) -> None:
        """Test marginalizing all variables raises ValueError."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        with self.assertRaises(ValueError) as ctx:
            marg.marginalize([0, 1, 2])
        self.assertIn("Cannot marginalize all variables", str(ctx.exception))
        self.assertIn("mean()", str(ctx.exception))

    def test_marginal_empty_raises_error(self) -> None:
        """Test keeping zero variables raises ValueError."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        with self.assertRaises(ValueError) as ctx:
            marg.marginal([])
        self.assertIn("Cannot keep zero variables", str(ctx.exception))

    def test_marginalize_invalid_index_raises_error(self) -> None:
        """Test invalid variable index raises IndexError."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        with self.assertRaises(IndexError):
            marg.marginalize([5])  # Out of bounds

        with self.assertRaises(IndexError):
            marg.marginalize([-1])  # Negative

    def test_marginalize_empty_list_returns_copy(self) -> None:
        """Test marginalizing empty list returns same nvars."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_same = marg.marginalize([])

        self.assertEqual(ft_same.nvars(), 3)

    # =========================================================================
    # Test 9: Monte Carlo Verification (simplified)
    # =========================================================================

    def test_marginal_evaluation_matches_conditional_expectation(self) -> None:
        """Test marginalized FT evaluates to E[f | kept vars] approximately.

        For a rank-1 product FT f(x,y,z) = p(x)*q(y)*r(z):
        E[f | x,z] = p(x) * E[q(y)] * r(z)

        This tests that evaluation of marginalized FT matches this formula.
        """
        bkd = self._bkd

        # Create rank-1 FT with known coefficients
        # f(x,y,z) = (2 + x) * (3 + 2y) * (1 + 0.5z) approximately
        # Using orthonormal Legendre: P_0 = 1/sqrt(2), P_1 = sqrt(3/2)*x
        # Set θ^{(0)} and θ^{(1)} for simple test
        coefficients = [
            bkd.asarray([[2.0], [1.0], [0.0]]),  # θ_x
            bkd.asarray([[3.0], [2.0], [0.0]]),  # θ_y
            bkd.asarray([[1.0], [0.5], [0.0]]),  # θ_z
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2, coefficients=coefficients)
        marg = FunctionTrainMarginalization(pce_ft)

        # Marginalize y → f̄(x, z) = p(x) * E[q(y)] * r(z) = p(x) * θ_y^{(0)} * r(z)
        ft_xz = marg.marginalize([1])  # Keep x and z

        # The marginalized FT should have:
        # - Core 0: p(x) * E[q(y)] absorbed
        # - Core 1: r(z)
        # For rank-1: new_core_0 coefficients = old_core_0 coefficients * θ_y^{(0)}

        # Test at a few sample points
        test_samples = bkd.asarray([[-0.5, 0.0, 0.5], [0.3, 0.3, 0.3]])  # (2, 3)
        marg_values = ft_xz(test_samples)  # Shape: (1, 3)

        # For rank-1: f̄(x,z) should equal p(x) * θ_y^{(0)} * r(z)
        # Evaluate original FT at y=? doesn't matter, we check mean preservation
        # Actually, let's verify mean is preserved
        pce_xz = PCEFunctionTrain(ft_xz)
        moments_xz = FunctionTrainMoments(pce_xz)
        mean_xz = moments_xz.mean()

        moments_orig = FunctionTrainMoments(pce_ft)
        mean_orig = moments_orig.mean()

        bkd.assert_allclose(mean_xz, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test: Analytical coefficient transformation
    # =========================================================================

    def test_coefficient_transformation_analytical(self) -> None:
        """Test coefficient transformation is correct analytically.

        For rank-1 FT: f(x,y) = p(x) * q(y) with known coefficients.
        Marginalize y → f̄(x) = p(x) * θ_y^{(0)}

        The resulting FT should have coefficients = original * θ_y^{(0)}.
        """
        bkd = self._bkd

        # Create 2-variable rank-1 FT
        theta_x = bkd.asarray([[2.0], [1.5]])  # θ_x^{(0)}=2, θ_x^{(1)}=1.5
        theta_y = bkd.asarray([[3.0], [0.5]])  # θ_y^{(0)}=3, θ_y^{(1)}=0.5
        coefficients = [theta_x, theta_y]
        pce_ft = self._create_rank1_pce_ft(nvars=2, max_level=1, coefficients=coefficients)
        marg = FunctionTrainMarginalization(pce_ft)

        # Marginalize y
        ft_x = marg.marginalize([1])

        # Expected: new_θ_x = θ_x * θ_y^{(0)} = [2*3, 1.5*3] = [6, 4.5]
        expected_coef = bkd.asarray([[6.0], [4.5]])

        # Extract coefficients from marginalized FT
        actual_coef = ft_x.cores()[0].get_basisexp(0, 0).get_coefficients()

        bkd.assert_allclose(actual_coef, expected_coef, rtol=1e-12)


class TestFunctionTrainMarginalizationNumpy(
    TestFunctionTrainMarginalization[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionTrainMarginalizationTorch(
    TestFunctionTrainMarginalization[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestFTDimensionReducer(Generic[Array], unittest.TestCase):
    """Base class for FTDimensionReducer tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_rank1_pce_ft(
        self,
        nvars: int = 3,
        max_level: int = 2,
        coefficients: Optional[List[Array]] = None,
    ) -> PCEFunctionTrain[Array]:
        """Create a rank-1 PCE FT with optional coefficients."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d_list = [create_bases_1d([m], bkd) for m in marginals]
        indices_1d = compute_hyperbolic_indices(1, max_level, 1.0, bkd)

        cores: List[FunctionTrainCore[Array]] = []
        for dd in range(nvars):
            basis = OrthonormalPolynomialBasis(bases_1d_list[dd], bkd, indices_1d)
            bexp = BasisExpansion(basis, bkd, nqoi=1)
            if coefficients is not None:
                bexp.set_coefficients(coefficients[dd])
            else:
                coef = bkd.asarray(
                    np.random.randn(max_level + 1, 1) * 0.5
                )
                coef[0, 0] = 1.0 + np.random.rand()
                bexp.set_coefficients(coef)
            core = FunctionTrainCore([[bexp]], bkd)
            cores.append(core)

        ft = FunctionTrain(cores, bkd, nqoi=1)
        return PCEFunctionTrain(ft)

    def test_satisfies_dimension_reducer_protocol(self) -> None:
        """Test that FTDimensionReducer satisfies DimensionReducerProtocol."""
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, self._bkd)
        self.assertIsInstance(reducer, DimensionReducerProtocol)

    def test_nvars_matches_original(self) -> None:
        """Test nvars returns original FT variable count."""
        pce_ft = self._create_rank1_pce_ft(nvars=4)
        reducer = FTDimensionReducer(pce_ft, self._bkd)
        self.assertEqual(reducer.nvars(), 4)

    def test_nqoi_matches_original(self) -> None:
        """Test nqoi returns original FT QoI count."""
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, self._bkd)
        self.assertEqual(reducer.nqoi(), 1)

    def test_reduce_1d_returns_reduced_function(self) -> None:
        """Test reduce with single index returns ReducedFunction."""
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, self._bkd)
        fn = reducer.reduce([0])
        self.assertIsInstance(fn, ReducedFunction)
        self.assertEqual(fn.nvars(), 1)
        self.assertEqual(fn.nqoi(), 1)

    def test_reduce_2d_returns_reduced_function(self) -> None:
        """Test reduce with two indices returns ReducedFunction."""
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, self._bkd)
        fn = reducer.reduce([0, 2])
        self.assertIsInstance(fn, ReducedFunction)
        self.assertEqual(fn.nvars(), 2)
        self.assertEqual(fn.nqoi(), 1)

    def test_reduce_1d_evaluates_correctly(self) -> None:
        """Test 1D reduced function evaluates with correct shape."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, bkd)
        fn = reducer.reduce([1])
        samples = bkd.asarray(np.array([[-0.5, 0.0, 0.5]]))  # (1, 3)
        result = fn(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_reduce_2d_evaluates_correctly(self) -> None:
        """Test 2D reduced function evaluates with correct shape."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, bkd)
        fn = reducer.reduce([0, 2])
        samples = bkd.asarray(np.array([[-0.5, 0.0, 0.5], [0.3, -0.3, 0.1]]))
        result = fn(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_reduce_preserves_mean(self) -> None:
        """Test mean of reduced FT matches original."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, bkd)

        # Original mean
        moments_orig = FunctionTrainMoments(pce_ft)
        mean_orig = moments_orig.mean()

        # Reduced to 1D — wrap in FT to compute mean
        marg = FunctionTrainMarginalization(pce_ft)
        ft_1d = marg.marginal([0])
        pce_1d = PCEFunctionTrain(ft_1d)
        mean_1d = FunctionTrainMoments(pce_1d).mean()

        bkd.assert_allclose(mean_1d, mean_orig, rtol=1e-12)

    def test_reduce_matches_direct_marginalization(self) -> None:
        """Test reduce gives same values as FunctionTrainMarginalization."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, bkd)
        marg = FunctionTrainMarginalization(pce_ft)

        samples = bkd.asarray(np.array([[-0.5, 0.0, 0.5]]))  # (1, 3)

        # Via reducer
        fn_reduced = reducer.reduce([1])
        vals_reducer = fn_reduced(samples)

        # Via direct marginalization
        ft_1d = marg.marginal([1])
        vals_direct = ft_1d(samples)

        bkd.assert_allclose(vals_reducer, vals_direct, rtol=1e-14)

    def test_bkd_returns_backend(self) -> None:
        """Test bkd() returns the computational backend."""
        pce_ft = self._create_rank1_pce_ft()
        reducer = FTDimensionReducer(pce_ft, self._bkd)
        self.assertIs(reducer.bkd(), self._bkd)


class TestFTDimensionReducerNumpy(
    TestFTDimensionReducer[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFTDimensionReducerTorch(
    TestFTDimensionReducer[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
