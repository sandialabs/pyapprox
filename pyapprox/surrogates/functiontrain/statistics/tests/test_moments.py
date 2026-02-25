"""Tests for FunctionTrainMoments."""

import unittest
from typing import Any, Generic, List, Optional, Tuple

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
from pyapprox.surrogates.functiontrain import (
    FunctionTrain,
    PCEFunctionTrain,
    ALSFitter,
    create_uniform_pce_functiontrain,
)
from pyapprox.surrogates.functiontrain.statistics import (
    FunctionTrainMoments,
)
from pyapprox.benchmarks.functions.genz import GaussianPeakFunction


class TestFunctionTrainMoments(Generic[Array], unittest.TestCase):
    """Base class for FunctionTrainMoments tests."""

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

    def test_construction_success(self) -> None:
        """Test FunctionTrainMoments construction succeeds."""
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        moments = FunctionTrainMoments(pce_ft)

        self.assertIs(moments.pce_ft(), pce_ft)
        self.assertIs(moments.bkd(), self._bkd)

    def test_construction_rejects_non_pce_ft(self) -> None:
        """Test construction fails for non-PCEFunctionTrain."""
        with self.assertRaises(TypeError) as ctx:
            FunctionTrainMoments("not a pce ft")  # type: ignore
        self.assertIn("Expected PCEFunctionTrain", str(ctx.exception))

    def test_constant_function_mean(self) -> None:
        """Test mean of constant function f(x) = c."""
        bkd = self._bkd
        # Constant function: only θ^{(0)} nonzero
        # For rank-1 FT: f(x) = θ_1^{(0)} * θ_2^{(0)} * θ_3^{(0)}
        c1, c2, c3 = 2.0, 3.0, 0.5
        coefficients = [
            bkd.asarray([[c1], [0.0], [0.0]]),
            bkd.asarray([[c2], [0.0], [0.0]]),
            bkd.asarray([[c3], [0.0], [0.0]]),
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = c1 * c2 * c3
        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)

    def test_constant_function_variance_is_zero(self) -> None:
        """Test variance of constant function is zero."""
        bkd = self._bkd
        coefficients = [
            bkd.asarray([[2.0], [0.0], [0.0]]),
            bkd.asarray([[3.0], [0.0], [0.0]]),
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=2, max_level=2, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        bkd.assert_allclose(moments.variance(), bkd.asarray([0.0]), atol=1e-12)

    def test_variance_non_negative(self) -> None:
        """Test variance is non-negative."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=3)
        moments = FunctionTrainMoments(pce_ft)

        var = moments.variance()
        self.assertTrue(float(var[0]) >= -1e-14)

    def test_std_equals_sqrt_variance(self) -> None:
        """Test std = sqrt(variance)."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        moments = FunctionTrainMoments(pce_ft)

        expected_std = bkd.sqrt(moments.variance())
        bkd.assert_allclose(moments.std(), expected_std, rtol=1e-12)

    def test_second_moment_geq_mean_squared(self) -> None:
        """Test E[f²] >= E[f]² (since Var >= 0)."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=3)
        moments = FunctionTrainMoments(pce_ft)

        second_moment = float(moments.second_moment()[0])
        mean_squared = float(moments.mean()[0]) ** 2
        self.assertTrue(second_moment >= mean_squared - 1e-14)

    def test_mean_product_of_linear_functions(self) -> None:
        """Test mean of f(x,y,z) = (a0 + a1*x)(b0 + b1*y)(c0 + c1*z).

        For uniform[-1,1], E[x] = 0, so E[f] = a0 * b0 * c0.
        The orthonormal Legendre P_0 = 1/sqrt(2), P_1 = sqrt(3/2)*x.
        With coefficients in orthonormal basis:
        - θ^{(0)} corresponds to constant term contribution
        - E[f] = θ_1^{(0)} * θ_2^{(0)} * θ_3^{(0)}
        """
        bkd = self._bkd
        # Use simple coefficients where we know the analytical mean
        # θ^{(0)} = 2, θ^{(1)} = 1 for each core
        # Mean = 2 * 2 * 2 = 8
        coefficients = [
            bkd.asarray([[2.0], [1.0]]),
            bkd.asarray([[2.0], [1.0]]),
            bkd.asarray([[2.0], [1.0]]),
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=1, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = 2.0 * 2.0 * 2.0
        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)

    def test_variance_product_of_linear_functions(self) -> None:
        """Test variance of f(x,y) = (a0 + a1*P_1(x))(b0 + b1*P_1(y)).

        Using orthonormality E[P_i P_j] = δ_ij:
        E[f] = a0 * b0
        E[f²] = E[(a0 + a1*P_1)²] * E[(b0 + b1*P_1)²]
              = (a0² + a1²) * (b0² + b1²)
        Var[f] = E[f²] - E[f]²
        """
        bkd = self._bkd
        a0, a1 = 3.0, 2.0
        b0, b1 = 1.5, 0.5
        coefficients = [
            bkd.asarray([[a0], [a1]]),
            bkd.asarray([[b0], [b1]]),
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=2, max_level=1, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = a0 * b0
        expected_second_moment = (a0**2 + a1**2) * (b0**2 + b1**2)
        expected_variance = expected_second_moment - expected_mean**2

        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)
        bkd.assert_allclose(
            moments.second_moment(), bkd.asarray([expected_second_moment]), rtol=1e-12
        )
        bkd.assert_allclose(moments.variance(), bkd.asarray([expected_variance]), rtol=1e-12)

    def test_mean_higher_degree_polynomial(self) -> None:
        """Test mean with higher degree polynomials.

        f(x) = θ0 + θ1*P_1(x) + θ2*P_2(x) + θ3*P_3(x)
        E[f] = θ0 (only constant term contributes)
        """
        bkd = self._bkd
        theta = [1.5, 2.0, -0.5, 0.3]
        coefficients = [bkd.asarray([[t] for t in theta])]
        pce_ft = self._create_rank1_pce_ft(nvars=1, max_level=3, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = theta[0]
        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)

    def test_variance_higher_degree_polynomial(self) -> None:
        """Test variance with higher degree polynomials.

        f(x) = θ0 + θ1*P_1(x) + θ2*P_2(x) + θ3*P_3(x)
        E[f] = θ0
        E[f²] = θ0² + θ1² + θ2² + θ3² (orthonormality)
        Var[f] = θ1² + θ2² + θ3²
        """
        bkd = self._bkd
        theta = [1.5, 2.0, -0.5, 0.3]
        coefficients = [bkd.asarray([[t] for t in theta])]
        pce_ft = self._create_rank1_pce_ft(nvars=1, max_level=3, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        expected_variance = theta[1]**2 + theta[2]**2 + theta[3]**2
        bkd.assert_allclose(moments.variance(), bkd.asarray([expected_variance]), rtol=1e-12)

    def test_mean_multivariable_higher_degree(self) -> None:
        """Test mean of product of higher-degree polynomials.

        f(x,y) = (a0 + a1*P_1 + a2*P_2)(b0 + b1*P_1 + b2*P_2)
        E[f] = a0 * b0
        """
        bkd = self._bkd
        a = [2.0, 1.0, 0.5]
        b = [3.0, -1.0, 0.25]
        coefficients = [
            bkd.asarray([[t] for t in a]),
            bkd.asarray([[t] for t in b]),
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=2, max_level=2, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = a[0] * b[0]
        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)

    def test_caching(self) -> None:
        """Test mean and second_moment are cached."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=2, max_level=2)
        moments = FunctionTrainMoments(pce_ft)

        mean1 = moments.mean()
        mean2 = moments.mean()
        self.assertIs(mean1, mean2)

        sm1 = moments.second_moment()
        sm2 = moments.second_moment()
        self.assertIs(sm1, sm2)

    def test_single_variable(self) -> None:
        """Test with single variable FunctionTrain."""
        bkd = self._bkd
        # f(x) = 1 + 2*P_1(x) + 3*P_2(x)
        # where P_i are orthonormal Legendre polynomials
        coefficients = [bkd.asarray([[1.0], [2.0], [3.0]])]
        pce_ft = self._create_rank1_pce_ft(nvars=1, max_level=2, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        # Mean = θ^{(0)} = 1.0
        bkd.assert_allclose(moments.mean(), bkd.asarray([1.0]), rtol=1e-12)

        # Variance = sum_{ℓ>=1} (θ^{(ℓ)})² = 2² + 3² = 13
        bkd.assert_allclose(moments.variance(), bkd.asarray([13.0]), rtol=1e-12)

    def test_product_function_analytical(self) -> None:
        """Test rank-1 product function with known analytical solution.

        f(x, y) = (a0 + a1*P_1(x)) * (b0 + b1*P_1(y))

        E[f] = a0 * b0  (since E[P_1] = 0)
        E[f²] = (a0² + a1²) * (b0² + b1²)  (using orthonormality)
        Var[f] = E[f²] - E[f]² = (a0² + a1²)(b0² + b1²) - a0²b0²
        """
        bkd = self._bkd
        a0, a1 = 2.0, 1.5
        b0, b1 = 3.0, 0.5
        coefficients = [
            bkd.asarray([[a0], [a1]]),
            bkd.asarray([[b0], [b1]]),
        ]
        pce_ft = self._create_rank1_pce_ft(nvars=2, max_level=1, coefficients=coefficients)
        moments = FunctionTrainMoments(pce_ft)

        expected_mean = a0 * b0
        expected_second_moment = (a0**2 + a1**2) * (b0**2 + b1**2)
        expected_variance = expected_second_moment - expected_mean**2

        bkd.assert_allclose(moments.mean(), bkd.asarray([expected_mean]), rtol=1e-12)
        bkd.assert_allclose(
            moments.second_moment(), bkd.asarray([expected_second_moment]), rtol=1e-12
        )
        bkd.assert_allclose(moments.variance(), bkd.asarray([expected_variance]), rtol=1e-12)

    def test_output_shapes(self) -> None:
        """Test output arrays have correct shapes."""
        bkd = self._bkd
        pce_ft = self._create_rank1_pce_ft(nvars=3, max_level=2)
        moments = FunctionTrainMoments(pce_ft)

        self.assertEqual(moments.mean().shape, (1,))
        self.assertEqual(moments.second_moment().shape, (1,))
        self.assertEqual(moments.variance().shape, (1,))
        self.assertEqual(moments.std().shape, (1,))

    def _create_uniform_pce_01(
        self, max_level: int
    ) -> BasisExpansion[Array]:
        """Create univariate PCE on [0, 1] interval."""
        bkd = self._bkd
        marginals = [UniformMarginal(0.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=1)

    def test_genz_gaussian_peak_mean_variance(self) -> None:
        """Test FT mean/variance against analytical for Genz Gaussian Peak.

        Fits a rank-2 FT to the 3D Gaussian Peak function on [0,1]³ and
        compares FT moments against exact analytical values.

        Targets:
        - FT fit error < 1e-6
        - Mean relative error < 1e-8
        - Variance relative error < 1e-5
        """
        bkd = self._bkd
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
        pce_template = self._create_uniform_pce_01(max_level)
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
        fit_error = bkd.sqrt(
            bkd.mean((ft_predictions - test_values) ** 2)
        ) / bkd.sqrt(bkd.mean(test_values ** 2))
        self.assertTrue(
            float(fit_error) < 1e-6,
            f"FT fit error {float(fit_error):.2e} exceeds 1e-6"
        )

        # Wrap in PCEFunctionTrain and compute moments
        pce_ft = PCEFunctionTrain(fitted_ft)
        moments = FunctionTrainMoments(pce_ft)

        # Compare FT moments to exact analytical values
        ft_mean = moments.mean()
        ft_var = moments.variance()

        # Mean error < 1e-8
        mean_error = abs(float(ft_mean[0]) - exact_mean) / abs(exact_mean)
        self.assertTrue(
            mean_error < 1e-8,
            f"Mean relative error {mean_error:.2e} exceeds 1e-8"
        )

        # Variance error < 1e-5
        var_error = abs(float(ft_var[0]) - exact_var) / abs(exact_var)
        self.assertTrue(
            var_error < 1e-5,
            f"Variance relative error {var_error:.2e} exceeds 1e-5"
        )


class TestFunctionTrainMomentsNumpy(TestFunctionTrainMoments[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionTrainMomentsTorch(TestFunctionTrainMoments[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
