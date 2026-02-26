"""Tests for FunctionTrainMoments."""

from typing import List, Optional

import numpy as np

from pyapprox.benchmarks.functions.genz import GaussianPeakFunction
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
)
from pyapprox.util.test_utils import slow_test

import pytest


class TestFunctionTrainMoments:
    """Base class for FunctionTrainMoments tests."""

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

    def test_construction_success(self, bkd) -> None:
        """Test FunctionTrainMoments construction succeeds."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        moments = FunctionTrainMoments(pce_ft)

        assert moments.pce_ft() is pce_ft
        assert moments.bkd() is bkd

    def test_construction_rejects_non_pce_ft(self, bkd) -> None:
        """Test construction fails for non-PCEFunctionTrain."""
        with pytest.raises(TypeError) as ctx:
            FunctionTrainMoments("not a pce ft")  # type: ignore
        assert "Expected PCEFunctionTrain" in str(ctx.value)

    def test_constant_function_mean(self, bkd) -> None:
        """Test mean of constant function f(x) = c."""
        # Constant function: only theta^{(0)} nonzero
        # For rank-1 FT: f(x) = theta_1^{(0)} * theta_2^{(0)} * theta_3^{(0)}
        c1, c2, c3 = 2.0, 3.0, 0.5
        coefficients = [
            bkd.asarray([[c1], [0.0], [0.0]]),
            bkd.asarray([[c2], [0.0], [0.0]]),
            bkd.asarray([[c3], [0.0], [0.0]]),
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=3, max_level=2, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = c1 * c2 * c3
        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)

    def test_constant_function_variance_is_zero(self, bkd) -> None:
        """Test variance of constant function is zero."""
        coefficients = [
            bkd.asarray([[2.0], [0.0], [0.0]]),
            bkd.asarray([[3.0], [0.0], [0.0]]),
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=2, max_level=2, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        bkd.assert_allclose(moments.variance(), bkd.asarray([0.0]), atol=1e-12)

    def test_variance_non_negative(self, bkd) -> None:
        """Test variance is non-negative."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=3)
        moments = FunctionTrainMoments(pce_ft)

        var = moments.variance()
        assert float(var[0]) >= -1e-14

    def test_std_equals_sqrt_variance(self, bkd) -> None:
        """Test std = sqrt(variance)."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        moments = FunctionTrainMoments(pce_ft)

        expected_std = bkd.sqrt(moments.variance())
        bkd.assert_allclose(moments.std(), expected_std, rtol=1e-12)

    def test_second_moment_geq_mean_squared(self, bkd) -> None:
        """Test E[f^2] >= E[f]^2 (since Var >= 0)."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=3)
        moments = FunctionTrainMoments(pce_ft)

        second_moment = float(moments.second_moment()[0])
        mean_squared = float(moments.mean()[0]) ** 2
        assert second_moment >= mean_squared - 1e-14

    def test_mean_product_of_linear_functions(self, bkd) -> None:
        """Test mean of f(x,y,z) = (a0 + a1*x)(b0 + b1*y)(c0 + c1*z).

        For uniform[-1,1], E[x] = 0, so E[f] = a0 * b0 * c0.
        The orthonormal Legendre P_0 = 1/sqrt(2), P_1 = sqrt(3/2)*x.
        With coefficients in orthonormal basis:
        - theta^{(0)} corresponds to constant term contribution
        - E[f] = theta_1^{(0)} * theta_2^{(0)} * theta_3^{(0)}
        """
        # Use simple coefficients where we know the analytical mean
        # theta^{(0)} = 2, theta^{(1)} = 1 for each core
        # Mean = 2 * 2 * 2 = 8
        coefficients = [
            bkd.asarray([[2.0], [1.0]]),
            bkd.asarray([[2.0], [1.0]]),
            bkd.asarray([[2.0], [1.0]]),
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=3, max_level=1, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = 2.0 * 2.0 * 2.0
        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)

    def test_variance_product_of_linear_functions(self, bkd) -> None:
        """Test variance of f(x,y) = (a0 + a1*P_1(x))(b0 + b1*P_1(y)).

        Using orthonormality E[P_i P_j] = delta_ij:
        E[f] = a0 * b0
        E[f^2] = E[(a0 + a1*P_1)^2] * E[(b0 + b1*P_1)^2]
              = (a0^2 + a1^2) * (b0^2 + b1^2)
        Var[f] = E[f^2] - E[f]^2
        """
        a0, a1 = 3.0, 2.0
        b0, b1 = 1.5, 0.5
        coefficients = [
            bkd.asarray([[a0], [a1]]),
            bkd.asarray([[b0], [b1]]),
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=2, max_level=1, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = a0 * b0
        expected_second_moment = (a0**2 + a1**2) * (b0**2 + b1**2)
        expected_variance = expected_second_moment - expected_mean**2

        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)
        bkd.assert_allclose(
            moments.second_moment(), bkd.asarray([expected_second_moment]), rtol=1e-12
        )
        bkd.assert_allclose(
            moments.variance(), bkd.asarray([expected_variance]), rtol=1e-12
        )

    def test_mean_higher_degree_polynomial(self, bkd) -> None:
        """Test mean with higher degree polynomials.

        f(x) = theta0 + theta1*P_1(x) + theta2*P_2(x) + theta3*P_3(x)
        E[f] = theta0 (only constant term contributes)
        """
        theta = [1.5, 2.0, -0.5, 0.3]
        coefficients = [bkd.asarray([[t] for t in theta])]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=1, max_level=3, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = theta[0]
        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)

    def test_variance_higher_degree_polynomial(self, bkd) -> None:
        """Test variance with higher degree polynomials.

        f(x) = theta0 + theta1*P_1(x) + theta2*P_2(x) + theta3*P_3(x)
        E[f] = theta0
        E[f^2] = theta0^2 + theta1^2 + theta2^2 + theta3^2 (orthonormality)
        Var[f] = theta1^2 + theta2^2 + theta3^2
        """
        theta = [1.5, 2.0, -0.5, 0.3]
        coefficients = [bkd.asarray([[t] for t in theta])]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=1, max_level=3, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        expected_variance = theta[1] ** 2 + theta[2] ** 2 + theta[3] ** 2
        bkd.assert_allclose(
            moments.variance(), bkd.asarray([expected_variance]), rtol=1e-12
        )

    def test_mean_multivariable_higher_degree(self, bkd) -> None:
        """Test mean of product of higher-degree polynomials.

        f(x,y) = (a0 + a1*P_1 + a2*P_2)(b0 + b1*P_1 + b2*P_2)
        E[f] = a0 * b0
        """
        a = [2.0, 1.0, 0.5]
        b = [3.0, -1.0, 0.25]
        coefficients = [
            bkd.asarray([[t] for t in a]),
            bkd.asarray([[t] for t in b]),
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=2, max_level=2, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = a[0] * b[0]
        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)

    def test_caching(self, bkd) -> None:
        """Test mean and second_moment are cached."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=2, max_level=2)
        moments = FunctionTrainMoments(pce_ft)

        mean1 = moments.mean()
        mean2 = moments.mean()
        assert mean1 is mean2

        sm1 = moments.second_moment()
        sm2 = moments.second_moment()
        assert sm1 is sm2

    def test_single_variable(self, bkd) -> None:
        """Test with single variable FunctionTrain."""
        # f(x) = 1 + 2*P_1(x) + 3*P_2(x)
        # where P_i are orthonormal Legendre polynomials
        coefficients = [bkd.asarray([[1.0], [2.0], [3.0]])]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=1, max_level=2, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        # Mean = theta^{(0)} = 1.0
        bkd.assert_allclose(moments.mean(), bkd.asarray([1.0]), rtol=1e-12)

        # Variance = sum_{l>=1} (theta^{(l)})^2 = 2^2 + 3^2 = 13
        bkd.assert_allclose(moments.variance(), bkd.asarray([13.0]), rtol=1e-12)

    def test_product_function_analytical(self, bkd) -> None:
        """Test rank-1 product function with known analytical solution.

        f(x, y) = (a0 + a1*P_1(x)) * (b0 + b1*P_1(y))

        E[f] = a0 * b0  (since E[P_1] = 0)
        E[f^2] = (a0^2 + a1^2) * (b0^2 + b1^2)  (using orthonormality)
        Var[f] = E[f^2] - E[f]^2 = (a0^2 + a1^2)(b0^2 + b1^2) - a0^2*b0^2
        """
        a0, a1 = 2.0, 1.5
        b0, b1 = 3.0, 0.5
        coefficients = [
            bkd.asarray([[a0], [a1]]),
            bkd.asarray([[b0], [b1]]),
        ]
        pce_ft = self._create_rank1_pce_ft(
            bkd, nvars=2, max_level=1, coefficients=coefficients
        )
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = a0 * b0
        expected_second_moment = (a0**2 + a1**2) * (b0**2 + b1**2)
        expected_variance = expected_second_moment - expected_mean**2

        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)
        bkd.assert_allclose(
            moments.second_moment(), bkd.asarray([expected_second_moment]), rtol=1e-12
        )
        bkd.assert_allclose(
            moments.variance(), bkd.asarray([expected_variance]), rtol=1e-12
        )

    def test_output_shapes(self, bkd) -> None:
        """Test output arrays have correct shapes."""
        pce_ft = self._create_rank1_pce_ft(bkd, nvars=3, max_level=2)
        moments = FunctionTrainMoments(pce_ft)

        assert moments.mean().shape == (1,)
        assert moments.second_moment().shape == (1,)
        assert moments.variance().shape == (1,)
        assert moments.std().shape == (1,)

    def _create_uniform_pce_01(self, bkd, max_level):
        """Create univariate PCE on [0, 1] interval."""
        marginals = [UniformMarginal(0.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=1)

    @slow_test
    def test_genz_gaussian_peak_mean_variance(self, bkd) -> None:
        """Test FT mean/variance against analytical for Genz Gaussian Peak.

        Fits a rank-2 FT to the 3D Gaussian Peak function on [0,1]^3 and
        compares FT moments against exact analytical values.

        Targets:
        - FT fit error < 1e-6
        - Mean relative error < 1e-8
        - Variance relative error < 1e-5
        """
        np.random.seed(12345)

        # Setup Genz Gaussian Peak function on [0,1]^3
        # Use moderate c values for smooth but non-trivial function
        nvars = 3
        c_coeffs = [1.0, 1.2, 0.8]  # Width parameters
        w_coeffs = [0.5, 0.5, 0.5]  # Peak centered at [0.5, 0.5, 0.5]
        genz_func = GaussianPeakFunction(bkd, c_coeffs, w_coeffs)

        # Get exact analytical moments
        exact_mean = genz_func.mean()
        exact_var = genz_func.variance()

        # Create rank-2 FT with degree 8 polynomials
        max_level = 8
        ranks = [2, 2]  # Interior ranks for 3 variables
        pce_template = self._create_uniform_pce_01(bkd, max_level)
        ft_init = create_uniform_pce_functiontrain(pce_template, nvars, ranks, bkd)

        # Generate training samples on [0, 1]^nvars
        nsamples_train = 800
        train_samples = bkd.asarray(np.random.uniform(0, 1, (nvars, nsamples_train)))
        train_values = genz_func(train_samples)

        # Fit FT using ALS
        fitter = ALSFitter(bkd, max_sweeps=30, tol=1e-14)
        result = fitter.fit(ft_init, train_samples, train_values)
        fitted_ft = result.surrogate()

        # Verify fit error < 1e-6 on test samples
        nsamples_test = 2000
        test_samples = bkd.asarray(np.random.uniform(0, 1, (nvars, nsamples_test)))
        test_values = genz_func(test_samples)
        ft_predictions = fitted_ft(test_samples)
        fit_error = bkd.sqrt(bkd.mean((ft_predictions - test_values) ** 2)) / bkd.sqrt(
            bkd.mean(test_values**2)
        )
        assert (
            float(fit_error) < 1e-6
        ), f"FT fit error {float(fit_error):.2e} exceeds 1e-6"

        # Wrap in PCEFunctionTrain and compute moments
        pce_ft = PCEFunctionTrain(fitted_ft)
        moments = FunctionTrainMoments(pce_ft)

        # Compare FT moments to exact analytical values
        ft_mean = moments.mean()
        ft_var = moments.variance()

        # Mean error < 1e-8
        mean_error = abs(float(ft_mean[0]) - exact_mean) / abs(exact_mean)
        assert (
            mean_error < 1e-8
        ), f"Mean relative error {mean_error:.2e} exceeds 1e-8"

        # Variance error < 1e-5
        var_error = abs(float(ft_var[0]) - exact_var) / abs(exact_var)
        assert (
            var_error < 1e-5
        ), f"Variance relative error {var_error:.2e} exceeds 1e-5"
