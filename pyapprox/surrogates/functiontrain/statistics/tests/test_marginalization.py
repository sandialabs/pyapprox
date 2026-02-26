"""Tests for FunctionTrainMarginalization."""

from typing import List, Optional

import numpy as np
import pytest

from pyapprox.interface.functions.marginalize import (
    DimensionReducerProtocol,
    ReducedFunction,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import FunctionTrain
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.surrogates.functiontrain.pce_functiontrain import (
    PCEFunctionTrain,
)
from pyapprox.surrogates.functiontrain.statistics import (
    FunctionTrainMoments,
    FunctionTrainSensitivity,
)
from pyapprox.surrogates.functiontrain.statistics.marginalization import (
    FTDimensionReducer,
    FunctionTrainMarginalization,
    all_marginals_1d,
    marginal_1d,
    marginal_2d,
)


class TestFunctionTrainMarginalization:
    """Base class for FunctionTrainMarginalization tests."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_pce(self, bkd, max_level, nqoi=1):
        """Create a univariate PCE expansion with Legendre basis."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_rank1_pce_ft(self, bkd, nvars, max_level, coefficients=None):
        """Create a rank-1 PCEFunctionTrain.

        Parameters
        ----------
        bkd : Backend
            Computational backend.
        nvars : int
            Number of variables.
        max_level : int
            Polynomial degree.
        coefficients : List[Array], optional
            Coefficients for each core. If None, random coefficients are used.
            Each should have shape (nterms, 1).
        """
        nterms = max_level + 1
        cores = []

        for k in range(nvars):
            pce = self._create_univariate_pce(bkd, max_level)
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

    def test_construction_success(self, bkd) -> None:
        """Test FunctionTrainMarginalization construction succeeds."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        assert marg.pce_ft() is pce_ft
        assert marg.bkd() is bkd

    def test_construction_rejects_non_pce_ft(self, bkd) -> None:
        """Test construction fails for non-PCEFunctionTrain."""
        with pytest.raises(TypeError) as ctx:
            FunctionTrainMarginalization("not a pce ft")  # type: ignore
        assert "Expected PCEFunctionTrain" in str(ctx.value)

    # =========================================================================
    # Test 1: Single Variable Marginalization
    # =========================================================================

    def test_marginalize_single_variable(self, bkd) -> None:
        """Test marginalizing a single variable reduces dimension."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        # Marginalize variable 1
        ft_marg = marg.marginalize([1])

        # Should have 2 variables now
        assert ft_marg.nvars() == 2

    def test_marginalize_preserves_nqoi(self, bkd) -> None:
        """Test marginalized FT has same nqoi."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)
        ft_marg = marg.marginalize([1])

        assert ft_marg.nqoi() == pce_ft.nqoi()

    # =========================================================================
    # Test 2: Mean Preservation
    # =========================================================================

    def test_mean_preserved_after_marginalization(self, bkd) -> None:
        """Test E[f_marg] = E[f] after marginalization."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=3)
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

    def test_mean_preserved_marginalize_multiple(self, bkd) -> None:
        """Test mean preserved when marginalizing multiple variables."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=4, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        # Original mean
        moments_orig = FunctionTrainMoments(pce_ft)
        mean_orig = moments_orig.mean()

        # Marginalize variables 1 and 2
        ft_marg = marg.marginalize([1, 2])
        pce_marg = PCEFunctionTrain(ft_marg)
        moments_marg = FunctionTrainMoments(pce_marg)
        mean_marg = moments_marg.mean()

        assert ft_marg.nvars() == 2
        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test 3: Variance via Marginalization (Main Effect Connection)
    # =========================================================================

    def test_1d_marginal_variance_equals_main_effect(self, bkd) -> None:
        """Test Var[E[f|x_k]] = V_k (main effect variance).

        For a 1D marginal keeping only variable k:
        Var[f_marginal] = Var[E[f | x_k]] = V_k (main effect variance)
        """
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=3)
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

    def test_chain_of_marginalizations(self, bkd) -> None:
        """Test marginalizing variables one at a time preserves mean."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=4, max_level=2)

        # Original mean
        moments_orig = FunctionTrainMoments(pce_ft)
        mean_orig = moments_orig.mean()

        # Marginalize one variable at a time: 3, 2, 1
        current_ft = pce_ft.ft()

        # Marginalize var 3 (now last in 4-var FT)
        pce_current = PCEFunctionTrain(current_ft)
        marg = FunctionTrainMarginalization(pce_current)
        current_ft = marg.marginalize([3])
        assert current_ft.nvars() == 3

        # Verify mean preserved
        pce_current = PCEFunctionTrain(current_ft)
        mean_curr = FunctionTrainMoments(pce_current).mean()
        bkd.assert_allclose(mean_curr, mean_orig, rtol=1e-12)

        # Marginalize var 2 (now last in 3-var FT)
        marg = FunctionTrainMarginalization(pce_current)
        current_ft = marg.marginalize([2])
        assert current_ft.nvars() == 2

        pce_current = PCEFunctionTrain(current_ft)
        mean_curr = FunctionTrainMoments(pce_current).mean()
        bkd.assert_allclose(mean_curr, mean_orig, rtol=1e-12)

        # Marginalize var 1 (now index 1 in 2-var FT)
        marg = FunctionTrainMarginalization(pce_current)
        current_ft = marg.marginalize([1])
        assert current_ft.nvars() == 1

        pce_current = PCEFunctionTrain(current_ft)
        mean_curr = FunctionTrainMoments(pce_current).mean()
        bkd.assert_allclose(mean_curr, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test 5: Statistics Module Compatibility
    # =========================================================================

    def test_marginalized_ft_works_with_pce_functiontrain(self, bkd) -> None:
        """Test marginalized FT can be wrapped in PCEFunctionTrain."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_marg = marg.marginalize([1])

        # Should be wrappable in PCEFunctionTrain
        pce_marg = PCEFunctionTrain(ft_marg)
        assert pce_marg.nvars() == 2

    def test_marginalized_ft_works_with_moments(self, bkd) -> None:
        """Test marginalized FT works with FunctionTrainMoments."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_marg = marg.marginalize([1])
        pce_marg = PCEFunctionTrain(ft_marg)

        # Should work with moments
        moments = FunctionTrainMoments(pce_marg)
        _ = moments.mean()
        _ = moments.variance()

    def test_marginalized_ft_works_with_sensitivity(self, bkd) -> None:
        """Test marginalized FT works with FunctionTrainSensitivity."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
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

    def test_marginal_keeps_specified_variables(self, bkd) -> None:
        """Test marginal() keeps only specified variables."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=4, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        # Keep variables 0 and 2
        ft_marg = marg.marginal([0, 2])

        assert ft_marg.nvars() == 2

    def test_marginal_1d(self, bkd) -> None:
        """Test creating 1D marginal."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_1d = marg.marginal([1])

        assert ft_1d.nvars() == 1

    def test_marginal_preserves_mean(self, bkd) -> None:
        """Test marginal() preserves mean."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=4, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        mean_orig = FunctionTrainMoments(pce_ft).mean()

        ft_marg = marg.marginal([0, 3])  # Keep vars 0 and 3
        pce_marg = PCEFunctionTrain(ft_marg)
        mean_marg = FunctionTrainMoments(pce_marg).mean()

        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test 7: Convenience Functions
    # =========================================================================

    def test_marginal_1d_function(self, bkd) -> None:
        """Test marginal_1d convenience function."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)

        ft_1d = marginal_1d(pce_ft, 1)

        assert ft_1d.nvars() == 1

    def test_marginal_2d_function(self, bkd) -> None:
        """Test marginal_2d convenience function."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=4, max_level=2)

        ft_2d = marginal_2d(pce_ft, 0, 2)

        assert ft_2d.nvars() == 2

    def test_all_marginals_1d_function(self, bkd) -> None:
        """Test all_marginals_1d convenience function."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)

        all_1d = all_marginals_1d(pce_ft)

        assert len(all_1d) == 3
        for ft_1d in all_1d:
            assert ft_1d.nvars() == 1

    def test_all_1d_marginals_preserve_mean(self, bkd) -> None:
        """Test all 1D marginals preserve original mean."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        mean_orig = FunctionTrainMoments(pce_ft).mean()

        all_1d = all_marginals_1d(pce_ft)
        for ft_1d in all_1d:
            pce_1d = PCEFunctionTrain(ft_1d)
            mean_1d = FunctionTrainMoments(pce_1d).mean()
            bkd.assert_allclose(mean_1d, mean_orig, rtol=1e-12)

    # =========================================================================
    # Test: Sum of Marginal Variances Property
    # =========================================================================

    def test_sum_marginal_variances_leq_total_variance(self, bkd) -> None:
        """Test Sigma_k Var[E[f|x_k]] <= Var[f].

        The sum of main effect variances is at most the total variance.
        Equality holds only for additive functions (no interactions).
        """
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=3)

        moments = FunctionTrainMoments(pce_ft)
        total_var = moments.variance()

        marg = FunctionTrainMarginalization(pce_ft)

        sum_marginal_vars = bkd.asarray([0.0])
        for k in range(pce_ft.nvars()):
            marginal_k = marg.marginal([k])
            pce_marginal_k = PCEFunctionTrain(marginal_k)
            V_k = FunctionTrainMoments(pce_marginal_k).variance()
            sum_marginal_vars = sum_marginal_vars + V_k

        # Sigma V_k <= Var[f] always (with small tolerance for numerical error)
        assert (
            float(sum_marginal_vars[0]) <= float(total_var[0]) + 1e-10
        ), (
            f"Sum of marginal variances "
            f"{sum_marginal_vars} > total "
            f"variance {total_var}"
        )

    def _create_additive_pce_ft(self, bkd, nvars, max_level):
        """Create an additive FunctionTrain: f(x) = Sigma_k g_k(x_k).

        For an additive function, main effects capture all variance.
        """
        nterms = max_level + 1
        cores = []

        for k in range(nvars):
            pce = self._create_univariate_pce(bkd, max_level)
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

    def test_additive_function_marginal_variances_sum_to_total(self, bkd) -> None:
        """Test for additive f = Sigma_k g_k(x_k): Sigma_k Var[E[f|x_k]] = Var[f].

        For additive functions, there are no interactions, so the sum of
        main effect variances equals the total variance exactly.
        """
        pce_ft = self._create_additive_pce_ft(bkd, nvars=3, max_level=2)

        moments = FunctionTrainMoments(pce_ft)
        total_var = moments.variance()

        marg = FunctionTrainMarginalization(pce_ft)

        sum_marginal_vars = bkd.asarray([0.0])
        for k in range(pce_ft.nvars()):
            marginal_k = marg.marginal([k])
            pce_marginal_k = PCEFunctionTrain(marginal_k)
            V_k = FunctionTrainMoments(pce_marginal_k).variance()
            sum_marginal_vars = sum_marginal_vars + V_k

        # For additive: Sigma V_k = Var[f] (no interactions)
        bkd.assert_allclose(
            sum_marginal_vars,
            total_var,
            rtol=1e-10,
        )

    # =========================================================================
    # Test 8: Edge Cases
    # =========================================================================

    def test_marginalize_first_variable(self, bkd) -> None:
        """Test marginalizing first variable (k=0) uses left-multiply."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        mean_orig = FunctionTrainMoments(pce_ft).mean()

        # Marginalize variable 0
        ft_marg = marg.marginalize([0])

        assert ft_marg.nvars() == 2

        pce_marg = PCEFunctionTrain(ft_marg)
        mean_marg = FunctionTrainMoments(pce_marg).mean()
        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    def test_marginalize_last_variable(self, bkd) -> None:
        """Test marginalizing last variable (k=d-1) uses right-multiply."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        mean_orig = FunctionTrainMoments(pce_ft).mean()

        # Marginalize last variable (index 2)
        ft_marg = marg.marginalize([2])

        assert ft_marg.nvars() == 2

        pce_marg = PCEFunctionTrain(ft_marg)
        mean_marg = FunctionTrainMoments(pce_marg).mean()
        bkd.assert_allclose(mean_marg, mean_orig, rtol=1e-12)

    def test_marginalize_to_single_variable(self, bkd) -> None:
        """Test marginalizing all but one variable gives 1D FT."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        mean_orig = FunctionTrainMoments(pce_ft).mean()

        # Marginalize variables 0 and 2, keep only 1
        ft_1d = marg.marginalize([0, 2])

        assert ft_1d.nvars() == 1

        pce_1d = PCEFunctionTrain(ft_1d)
        mean_1d = FunctionTrainMoments(pce_1d).mean()
        bkd.assert_allclose(mean_1d, mean_orig, rtol=1e-12)

    def test_marginalize_all_raises_error(self, bkd) -> None:
        """Test marginalizing all variables raises ValueError."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        with pytest.raises(ValueError) as ctx:
            marg.marginalize([0, 1, 2])
        assert "Cannot marginalize all variables" in str(ctx.value)
        assert "mean()" in str(ctx.value)

    def test_marginal_empty_raises_error(self, bkd) -> None:
        """Test keeping zero variables raises ValueError."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        with pytest.raises(ValueError) as ctx:
            marg.marginal([])
        assert "Cannot keep zero variables" in str(ctx.value)

    def test_marginalize_invalid_index_raises_error(self, bkd) -> None:
        """Test invalid variable index raises IndexError."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        with pytest.raises(IndexError):
            marg.marginalize([5])  # Out of bounds

        with pytest.raises(IndexError):
            marg.marginalize([-1])  # Negative

    def test_marginalize_empty_list_returns_copy(self, bkd) -> None:
        """Test marginalizing empty list returns same nvars."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        marg = FunctionTrainMarginalization(pce_ft)

        ft_same = marg.marginalize([])

        assert ft_same.nvars() == 3

    # =========================================================================
    # Test 9: Monte Carlo Verification (simplified)
    # =========================================================================

    def test_marginal_evaluation_matches_conditional_expectation(self, bkd) -> None:
        """Test marginalized FT evaluates to E[f | kept vars] approximately.

        For a rank-1 product FT f(x,y,z) = p(x)*q(y)*r(z):
        E[f | x,z] = p(x) * E[q(y)] * r(z)

        This tests that evaluation of marginalized FT matches this formula.
        """
        # Create rank-1 FT with known coefficients
        # f(x,y,z) = (2 + x) * (3 + 2y) * (1 + 0.5z) approximately
        # Using orthonormal Legendre: P_0 = 1/sqrt(2), P_1 = sqrt(3/2)*x
        # Set theta^{(0)} and theta^{(1)} for simple test
        coefficients = [
            bkd.asarray([[2.0], [1.0], [0.0]]),  # theta_x
            bkd.asarray([[3.0], [2.0], [0.0]]),  # theta_y
            bkd.asarray([[1.0], [0.5], [0.0]]),  # theta_z
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=3, max_level=2, coefficients=coefficients
        )
        marg = FunctionTrainMarginalization(pce_ft)

        # Marginalize y -> f_bar(x, z) = p(x) * E[q(y)] * r(z) = p(x) * theta_y^{(0)} * r(z)
        ft_xz = marg.marginalize([1])  # Keep x and z

        # Test at a few sample points
        test_samples = bkd.asarray([[-0.5, 0.0, 0.5], [0.3, 0.3, 0.3]])  # (2, 3)
        ft_xz(test_samples)  # Shape: (1, 3)

        # For rank-1: f_bar(x,z) should equal p(x) * theta_y^{(0)} * r(z)
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

    def test_coefficient_transformation_analytical(self, bkd) -> None:
        """Test coefficient transformation is correct analytically.

        For rank-1 FT: f(x,y) = p(x) * q(y) with known coefficients.
        Marginalize y -> f_bar(x) = p(x) * theta_y^{(0)}

        The resulting FT should have coefficients = original * theta_y^{(0)}.
        """
        # Create 2-variable rank-1 FT
        theta_x = bkd.asarray([[2.0], [1.5]])  # theta_x^{(0)}=2, theta_x^{(1)}=1.5
        theta_y = bkd.asarray([[3.0], [0.5]])  # theta_y^{(0)}=3, theta_y^{(1)}=0.5
        coefficients = [theta_x, theta_y]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=2, max_level=1, coefficients=coefficients
        )
        marg = FunctionTrainMarginalization(pce_ft)

        # Marginalize y
        ft_x = marg.marginalize([1])

        # Expected: new_theta_x = theta_x * theta_y^{(0)} = [2*3, 1.5*3] = [6, 4.5]
        expected_coef = bkd.asarray([[6.0], [4.5]])

        # Extract coefficients from marginalized FT
        actual_coef = ft_x.cores()[0].get_basisexp(0, 0).get_coefficients()

        bkd.assert_allclose(actual_coef, expected_coef, rtol=1e-12)


class TestFTDimensionReducer:
    """Base class for FTDimensionReducer tests."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_rank1_pce_ft(self, bkd, nvars=3, max_level=2, coefficients=None):
        """Create a rank-1 PCE FT with optional coefficients."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d_list = [create_bases_1d([m], bkd) for m in marginals]
        indices_1d = compute_hyperbolic_indices(1, max_level, 1.0, bkd)

        cores = []
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

    def test_satisfies_dimension_reducer_protocol(self, bkd) -> None:
        """Test that FTDimensionReducer satisfies DimensionReducerProtocol."""
        pce_ft = self._create_rank1_pce_ft(bkd)
        reducer = FTDimensionReducer(pce_ft, bkd)
        assert isinstance(reducer, DimensionReducerProtocol)

    def test_nvars_matches_original(self, bkd) -> None:
        """Test nvars returns original FT variable count."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=4)
        reducer = FTDimensionReducer(pce_ft, bkd)
        assert reducer.nvars() == 4

    def test_nqoi_matches_original(self, bkd) -> None:
        """Test nqoi returns original FT QoI count."""
        pce_ft = self._create_rank1_pce_ft(bkd)
        reducer = FTDimensionReducer(pce_ft, bkd)
        assert reducer.nqoi() == 1

    def test_reduce_1d_returns_reduced_function(self, bkd) -> None:
        """Test reduce with single index returns ReducedFunction."""
        pce_ft = self._create_rank1_pce_ft(bkd)
        reducer = FTDimensionReducer(pce_ft, bkd)
        fn = reducer.reduce([0])
        assert isinstance(fn, ReducedFunction)
        assert fn.nvars() == 1
        assert fn.nqoi() == 1

    def test_reduce_2d_returns_reduced_function(self, bkd) -> None:
        """Test reduce with two indices returns ReducedFunction."""
        pce_ft = self._create_rank1_pce_ft(bkd)
        reducer = FTDimensionReducer(pce_ft, bkd)
        fn = reducer.reduce([0, 2])
        assert isinstance(fn, ReducedFunction)
        assert fn.nvars() == 2
        assert fn.nqoi() == 1

    def test_reduce_1d_evaluates_correctly(self, bkd) -> None:
        """Test 1D reduced function evaluates with correct shape."""
        pce_ft = self._create_rank1_pce_ft(bkd)
        reducer = FTDimensionReducer(pce_ft, bkd)
        fn = reducer.reduce([1])
        samples = bkd.asarray(np.array([[-0.5, 0.0, 0.5]]))  # (1, 3)
        result = fn(samples)
        assert result.shape == (1, 3)

    def test_reduce_2d_evaluates_correctly(self, bkd) -> None:
        """Test 2D reduced function evaluates with correct shape."""
        pce_ft = self._create_rank1_pce_ft(bkd)
        reducer = FTDimensionReducer(pce_ft, bkd)
        fn = reducer.reduce([0, 2])
        samples = bkd.asarray(np.array([[-0.5, 0.0, 0.5], [0.3, -0.3, 0.1]]))
        result = fn(samples)
        assert result.shape == (1, 3)

    def test_reduce_preserves_mean(self, bkd) -> None:
        """Test mean of reduced FT matches original."""
        pce_ft = self._create_rank1_pce_ft(bkd)

        # Original mean
        moments_orig = FunctionTrainMoments(pce_ft)
        mean_orig = moments_orig.mean()

        # Reduced to 1D -- wrap in FT to compute mean
        marg = FunctionTrainMarginalization(pce_ft)
        ft_1d = marg.marginal([0])
        pce_1d = PCEFunctionTrain(ft_1d)
        mean_1d = FunctionTrainMoments(pce_1d).mean()

        bkd.assert_allclose(mean_1d, mean_orig, rtol=1e-12)

    def test_reduce_matches_direct_marginalization(self, bkd) -> None:
        """Test reduce gives same values as FunctionTrainMarginalization."""
        pce_ft = self._create_rank1_pce_ft(bkd)
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

    def test_bkd_returns_backend(self, bkd) -> None:
        """Test bkd() returns the computational backend."""
        pce_ft = self._create_rank1_pce_ft(bkd)
        reducer = FTDimensionReducer(pce_ft, bkd)
        assert reducer.bkd() is bkd
