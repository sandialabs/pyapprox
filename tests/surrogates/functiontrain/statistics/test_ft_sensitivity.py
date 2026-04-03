"""Tests for FunctionTrainSensitivity."""

import math
from itertools import chain, combinations
from typing import Sequence

import numpy as np
import pytest

from pyapprox.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
    IshigamiSensitivityIndices,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import (
    ALSFitter,
    FunctionTrain,
    PCEFunctionTrain,
    create_uniform_pce_functiontrain,
)
from pyapprox.surrogates.functiontrain.core import FunctionTrainCore
from pyapprox.surrogates.functiontrain.statistics import (
    FunctionTrainMoments,
    FunctionTrainSensitivity,
)
from tests._helpers.markers import slow_test


def all_nonempty_subsets(iterable: Sequence[int]) -> chain[tuple[int, ...]]:
    """Generate all non-empty subsets."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


class TestFunctionTrainSensitivity:
    """Base class for FunctionTrainSensitivity tests."""

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
        """Create a rank-1 PCEFunctionTrain."""
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

    def _create_moments_and_sensitivity(self, pce_ft):
        """Create moments and sensitivity objects."""
        moments = FunctionTrainMoments(pce_ft)
        sensitivity = FunctionTrainSensitivity(moments)
        return moments, sensitivity

    def test_construction_success(self, bkd) -> None:
        """Test FunctionTrainSensitivity construction succeeds."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        moments = FunctionTrainMoments(pce_ft)
        sensitivity = FunctionTrainSensitivity(moments)

        assert sensitivity.moments() is moments
        assert sensitivity.bkd() is bkd

    def test_construction_rejects_non_moments(self, bkd) -> None:
        """Test construction fails for non-FunctionTrainMoments."""
        with pytest.raises(TypeError) as ctx:
            FunctionTrainSensitivity("not moments")  # type: ignore
        assert "Expected FunctionTrainMoments" in str(ctx.value)

    def test_var_idx_bounds_checking(self, bkd) -> None:
        """Test variable index bounds checking."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        with pytest.raises(IndexError):
            sensitivity.main_effect_index(-1)

        with pytest.raises(IndexError):
            sensitivity.main_effect_index(3)

        with pytest.raises(IndexError):
            sensitivity.total_effect_index(-1)

        with pytest.raises(IndexError):
            sensitivity.total_effect_index(3)

    def test_main_effect_matches_sobol_singleton(self, bkd) -> None:
        """S_{k} from sobol_index([k]) matches main_effect_index(k)."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        for k in range(3):
            me = sensitivity.main_effect_index(k)
            su = sensitivity.sobol_index([k])
            bkd.assert_allclose(me, su, rtol=1e-10)

    def test_sobol_sum_equals_one(self, bkd) -> None:
        """Sum of all S_u over non-empty u equals 1."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        total = bkd.array([0.0])
        for u in all_nonempty_subsets(range(3)):
            total = total + sensitivity.sobol_index(list(u))
        bkd.assert_allclose(total, bkd.array([1.0]), rtol=1e-10)

    def test_total_equals_sum_of_subsets(self, bkd) -> None:
        """S_k^T = Sigma_{u containing k} S_u."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        for k in range(3):
            total_effect = sensitivity.total_effect_index(k)
            sum_subsets = bkd.array([0.0])
            for u in all_nonempty_subsets(range(3)):
                if k in u:
                    sum_subsets = sum_subsets + sensitivity.sobol_index(list(u))
            bkd.assert_allclose(total_effect, sum_subsets, rtol=1e-10)

    def test_total_geq_main(self, bkd) -> None:
        """S_k^T >= S_k for all k."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=3)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        main = sensitivity.all_main_effects()
        total = sensitivity.all_total_effects()
        for k in range(3):
            assert (
                float(total[k]) >= float(main[k]) - 1e-10
            ), f"S_{k}^T = {total[k]} < S_{k} = {main[k]}"

    def test_indices_in_unit_interval(self, bkd) -> None:
        """All Sobol indices should be in [0, 1]."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        main = sensitivity.all_main_effects()
        total = sensitivity.all_total_effects()

        for k in range(3):
            assert float(main[k]) >= -1e-10
            assert float(main[k]) <= 1.0 + 1e-10
            assert float(total[k]) >= -1e-10
            assert float(total[k]) <= 1.0 + 1e-10

    def test_single_variable_all_main_effect(self, bkd) -> None:
        """For single variable, S_0 = 1 and S_0^T = 1."""
        # f(x) = a0 + a1*P_1(x) with a1 != 0
        coefficients = [bkd.asarray([[1.0], [2.0]])]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=1, max_level=1, coefficients=coefficients
        )
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        bkd.assert_allclose(
            sensitivity.main_effect_index(0), bkd.array([1.0]), rtol=1e-10
        )
        bkd.assert_allclose(
            sensitivity.total_effect_index(0), bkd.array([1.0]), rtol=1e-10
        )

    def test_constant_variable_zero_effect(self, bkd) -> None:
        """Variable with only constant coefficient has zero main effect."""
        # f(x, y) = (a0)(b0 + b1*P_1(y))
        # x has no non-constant terms, so S_x = 0
        coefficients = [
            bkd.asarray([[2.0], [0.0]]),  # x: constant only
            bkd.asarray([[1.0], [1.5]]),  # y: has linear term
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=2, max_level=1, coefficients=coefficients
        )
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        # S_x should be 0 (only constant term)
        bkd.assert_allclose(
            sensitivity.main_effect_index(0), bkd.array([0.0]), atol=1e-12
        )
        # S_y should be 1 (accounts for all variance)
        bkd.assert_allclose(
            sensitivity.main_effect_index(1), bkd.array([1.0]), rtol=1e-10
        )

    def test_additive_function_analytical(self, bkd) -> None:
        """Test analytical Sobol indices for additive-like function.

        For rank-1 FT f(x,y) = (a0 + a1*P_1(x))(b0 + b1*P_1(y)):
        - This is NOT additive, it's multiplicative
        - Var[f] = (a0^2 + a1^2)(b0^2 + b1^2) - a0^2*b0^2
        - V_x = a1^2 * b0^2  (main effect of x)
        - V_y = a0^2 * b1^2  (main effect of y)
        - V_{xy} = a1^2 * b1^2  (interaction)
        """
        a0, a1 = 2.0, 1.0
        b0, b1 = 3.0, 0.5
        coefficients = [
            bkd.asarray([[a0], [a1]]),
            bkd.asarray([[b0], [b1]]),
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=2, max_level=1, coefficients=coefficients
        )
        moments, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        # Analytical variance
        total_var = (a0**2 + a1**2) * (b0**2 + b1**2) - (a0 * b0) ** 2
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

        bkd.assert_allclose(
            sensitivity.main_effect_index(0), bkd.asarray([S_x]), rtol=1e-12
        )
        bkd.assert_allclose(
            sensitivity.main_effect_index(1), bkd.asarray([S_y]), rtol=1e-12
        )
        bkd.assert_allclose(
            sensitivity.sobol_index([0, 1]), bkd.asarray([S_xy]), rtol=1e-12
        )

        # Verify sum = 1
        bkd.assert_allclose(
            bkd.asarray([S_x + S_y + S_xy]), bkd.asarray([1.0]), rtol=1e-12
        )

    def test_three_variable_analytical(self, bkd) -> None:
        """Test with 3 variables - verify sum of all indices = 1."""
        coefficients = [
            bkd.asarray([[1.0], [0.5]]),
            bkd.asarray([[2.0], [1.0]]),
            bkd.asarray([[1.5], [0.3]]),
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=3, max_level=1, coefficients=coefficients
        )
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        # Compute all Sobol indices
        S = {}
        for u in all_nonempty_subsets(range(3)):
            S[u] = float(sensitivity.sobol_index(list(u))[0])

        # Sum should equal 1
        total = sum(S.values())
        bkd.assert_allclose(bkd.asarray([total]), bkd.asarray([1.0]), rtol=1e-10)

    def test_all_main_effects_shape(self, bkd) -> None:
        """Test all_main_effects returns correct shape."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=4, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        main = sensitivity.all_main_effects()
        assert main.shape == (4,)

    def test_all_total_effects_shape(self, bkd) -> None:
        """Test all_total_effects returns correct shape."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=4, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        total = sensitivity.all_total_effects()
        assert total.shape == (4,)

    def test_output_shapes(self, bkd) -> None:
        """Test individual index methods return correct shapes."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        _, sensitivity = self._create_moments_and_sensitivity(pce_ft)

        assert sensitivity.main_effect_variance(0).shape == (1,)
        assert sensitivity.main_effect_index(0).shape == (1,)
        assert sensitivity.total_effect_variance(0).shape == (1,)
        assert sensitivity.total_effect_index(0).shape == (1,)
        assert sensitivity.sobol_variance([0, 1]).shape == (1,)
        assert sensitivity.sobol_index([0, 1]).shape == (1,)


class TestIshigamiBenchmark:
    """Test FT sensitivity indices against Ishigami function benchmark.

    The Ishigami function is a standard benchmark for sensitivity analysis with
    analytically known Sobol indices. It's defined on [-pi, pi]^3:
    f(x) = sin(x_1) + a*sin^2(x_2) + b*x_3^4*sin(x_1)

    Standard parameters: a=7, b=0.1
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_uniform_pce_template(self, bkd, max_level):
        """Create univariate PCE template on [-pi, pi]."""
        marginals = [UniformMarginal(-math.pi, math.pi, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=1)

    @slow_test
    def test_ishigami_sobol_indices(self, bkd) -> None:
        """Test FT Sobol indices match Ishigami analytical values.

        Fits a rank-3 FT to the Ishigami function and compares computed
        Sobol indices against exact analytical values.

        Targets:
        - FT fit error < 1e-4
        - Main effect error < 1e-4
        - Total effect error < 1e-4
        """
        nvars = 3

        # Standard Ishigami parameters
        a, b = 7.0, 0.1
        ishigami = IshigamiFunction(bkd, a=a, b=b)
        exact_indices = IshigamiSensitivityIndices(bkd, a=a, b=b)

        # Get exact Sobol indices
        S_exact = [float(exact_indices.main_effects()[i, 0]) for i in range(nvars)]
        T_exact = [float(exact_indices.total_effects()[i, 0]) for i in range(nvars)]

        # Create rank-3 FT with degree 12 polynomials
        max_level = 12
        ranks = [3, 3]  # Interior ranks
        pce_template = self._create_uniform_pce_template(bkd, max_level)
        ft_init = create_uniform_pce_functiontrain(pce_template, nvars, ranks, bkd)

        # Generate training samples on [-pi, pi]^3
        nsamples_train = 1500
        train_samples = bkd.asarray(
            np.random.uniform(-math.pi, math.pi, (nvars, nsamples_train))
        )
        train_values = ishigami(train_samples)

        # Fit FT using ALS
        fitter = ALSFitter(bkd, max_sweeps=50, tol=1e-14)
        result = fitter.fit(ft_init, train_samples, train_values)
        fitted_ft = result.surrogate()

        # Verify fit error < 1e-4 on test samples
        nsamples_test = 2000
        test_samples = bkd.asarray(
            np.random.uniform(-math.pi, math.pi, (nvars, nsamples_test))
        )
        test_values = ishigami(test_samples)
        ft_predictions = fitted_ft(test_samples)
        fit_error = bkd.sqrt(bkd.mean((ft_predictions - test_values) ** 2)) / bkd.sqrt(
            bkd.mean(test_values**2)
        )
        assert (
            float(fit_error) < 1e-4
        ), f"FT fit error {float(fit_error):.2e} exceeds 1e-4"

        # Wrap in PCEFunctionTrain and compute sensitivity indices
        pce_ft = PCEFunctionTrain(fitted_ft)
        moments = FunctionTrainMoments(pce_ft)
        sensitivity = FunctionTrainSensitivity(moments)

        # Get computed indices
        S_computed = sensitivity.all_main_effects()
        T_computed = sensitivity.all_total_effects()

        # Verify main effect indices match analytical values
        for i in range(nvars):
            bkd.assert_allclose(
                bkd.asarray([float(S_computed[i])]),
                bkd.asarray([S_exact[i]]),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"S_{i} = {float(S_computed[i]):.4f}, "
                f"expected {S_exact[i]:.4f}",
            )

        # Verify total effect indices match analytical values
        for i in range(nvars):
            bkd.assert_allclose(
                bkd.asarray([float(T_computed[i])]),
                bkd.asarray([T_exact[i]]),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"T_{i} = {float(T_computed[i]):.4f}, "
                f"expected {T_exact[i]:.4f}",
            )

        # Verify interaction index S_13 (x_1-x_3 interaction)
        # S_13 = T_1 - S_1 since x_1 only interacts with x_3
        sobol_exact = exact_indices.sobol_indices()
        S_13_exact = float(sobol_exact[4, 0])  # S_13 is index 4
        S_13_computed = sensitivity.sobol_index([0, 2])
        bkd.assert_allclose(
            S_13_computed,
            bkd.asarray([S_13_exact]),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"S_13 = {float(S_13_computed[0]):.4f}, expected {S_13_exact:.4f}",
        )
