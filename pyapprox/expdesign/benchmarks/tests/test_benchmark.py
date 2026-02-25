"""
Standalone tests for LinearGaussianOEDBenchmark.

PERMANENT - no legacy imports.

These tests replicate the scenarios from legacy test_bayesoed.py:
1. Design matrix (polynomial basis) correctness
2. Exact EIG computation
3. D-optimal objective = -EIG relationship
4. Gradient checking via DerivativeChecker
5. Data generation consistency
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)

from pyapprox.expdesign.benchmarks import LinearGaussianOEDBenchmark
from pyapprox.expdesign.objective import DOptimalLinearModelObjective


class TestLinearGaussianBenchmarkStandalone(Generic[Array], unittest.TestCase):
    """Standalone tests for LinearGaussianOEDBenchmark.

    Replicates legacy test scenarios from test_bayesoed.py.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 5
        self._degree = 2
        self._min_degree = 0
        self._noise_std = 0.5
        self._prior_std = 0.5

    def _create_benchmark(self) -> LinearGaussianOEDBenchmark[Array]:
        return LinearGaussianOEDBenchmark(
            self._nobs,
            self._degree,
            self._noise_std,
            self._prior_std,
            self._bkd,
            min_degree=self._min_degree,
        )

    # ==========================================================================
    # Design matrix / polynomial basis tests
    # ==========================================================================

    def test_design_matrix_shape(self):
        """Test design matrix has correct shape (nobs, nparams)."""
        benchmark = self._create_benchmark()
        A = benchmark.design_matrix()
        nparams = self._degree - self._min_degree + 1
        self.assertEqual(A.shape, (self._nobs, nparams))

    def test_design_locations_shape(self):
        """Test design locations has correct shape (nobs,)."""
        benchmark = self._create_benchmark()
        x = benchmark.design_locations()
        self.assertEqual(x.shape, (self._nobs,))

    def test_design_locations_in_range(self):
        """Test design locations are in [-1, 1]."""
        benchmark = self._create_benchmark()
        x = self._bkd.to_numpy(benchmark.design_locations())
        self.assertTrue(np.all(x >= -1.0))
        self.assertTrue(np.all(x <= 1.0))

    def test_polynomial_basis_values(self):
        """Test design matrix implements polynomial basis correctly.

        For polynomial basis with min_degree=0, degree=2:
        A[i, j] = x_i^j where j in {0, 1, 2}
        """
        benchmark = self._create_benchmark()
        A = benchmark.design_matrix()
        x = benchmark.design_locations()

        # Check each row matches [x^0, x^1, x^2]
        for i in range(self._nobs):
            xi = float(self._bkd.to_numpy(x)[i])
            expected_row = [xi ** p for p in range(self._min_degree, self._degree + 1)]
            actual_row = self._bkd.to_numpy(A)[i, :]
            self._bkd.assert_allclose(
                self._bkd.asarray(actual_row),
                self._bkd.asarray(expected_row),
                rtol=1e-12,
            )

    def test_nobs_nparams_accessors(self):
        """Test accessors return correct values."""
        benchmark = self._create_benchmark()
        self.assertEqual(benchmark.nobs(), self._nobs)
        nparams = self._degree - self._min_degree + 1
        self.assertEqual(benchmark.nparams(), nparams)

    def test_noise_prior_var_accessors(self):
        """Test noise and prior variance accessors."""
        benchmark = self._create_benchmark()
        self._bkd.assert_allclose(
            self._bkd.asarray([benchmark.noise_var()]),
            self._bkd.asarray([self._noise_std ** 2]),
            rtol=1e-12,
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([benchmark.prior_var()]),
            self._bkd.asarray([self._prior_std ** 2]),
            rtol=1e-12,
        )

    # ==========================================================================
    # Exact EIG tests (replicates test_doptimal_oed_gradients scenario)
    # ==========================================================================

    def test_exact_eig_positive(self):
        """Test exact EIG is positive for informative design."""
        benchmark = self._create_benchmark()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        eig = benchmark.exact_eig(weights)
        self.assertGreater(eig, 0.0)

    def test_exact_eig_finite(self):
        """Test exact EIG is finite for various weights."""
        benchmark = self._create_benchmark()

        # Uniform weights
        uniform_weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        eig_uniform = benchmark.exact_eig(uniform_weights)
        self.assertTrue(np.isfinite(eig_uniform))

        # Random Dirichlet weights
        np.random.seed(123)
        random_weights = self._bkd.asarray(
            np.random.dirichlet(np.ones(self._nobs))[:, None]
        )
        eig_random = benchmark.exact_eig(random_weights)
        self.assertTrue(np.isfinite(eig_random))

        # Concentrated weights
        concentrated_np = np.zeros((self._nobs, 1))
        concentrated_np[0, 0] = 1.0
        concentrated_weights = self._bkd.asarray(concentrated_np)
        eig_concentrated = benchmark.exact_eig(concentrated_weights)
        self.assertTrue(np.isfinite(eig_concentrated))

    def test_d_optimal_is_negative_eig(self):
        """Test D-optimal objective equals negative EIG."""
        benchmark = self._create_benchmark()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        eig = benchmark.exact_eig(weights)
        d_opt = benchmark.d_optimal_objective(weights)

        self._bkd.assert_allclose(
            self._bkd.asarray([d_opt]),
            self._bkd.asarray([-eig]),
            rtol=1e-12,
        )

    # ==========================================================================
    # D-optimal objective relationship tests
    # (replicates test_doptimal_oed_gradients: assert dopt == kl_oed objective)
    # ==========================================================================

    def test_eig_matches_d_optimal_objective_class(self):
        """Test benchmark EIG matches DOptimalLinearModelObjective.

        Replicates legacy test_doptimal_oed_gradients assertion:
        assert bkd.allclose(dopt_objective(x0), kl_oed.objective()(x0), rtol=1e-5)

        For linear Gaussian, D-optimal objective = -EIG.
        """
        benchmark = self._create_benchmark()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        # Benchmark EIG
        eig = benchmark.exact_eig(weights)

        # DOptimalLinearModelObjective
        d_opt_obj = DOptimalLinearModelObjective(
            benchmark.design_matrix(),
            self._bkd.asarray(np.array(benchmark.noise_var())),
            self._bkd.asarray(np.array(benchmark.prior_var())),
            self._bkd,
        )
        d_opt_val = d_opt_obj(weights)

        # D-optimal objective is -1/2 * log(det(Y))
        # EIG is 1/2 * log(det(Y))
        # So d_opt_val = -eig
        d_opt_scalar = float(self._bkd.to_numpy(d_opt_val)[0, 0])
        self._bkd.assert_allclose(
            self._bkd.asarray([d_opt_scalar]),
            self._bkd.asarray([-eig]),
            rtol=1e-10,
        )

    def test_d_optimal_gradient_check(self):
        """Test D-optimal gradients using DerivativeChecker.

        Replicates legacy test_doptimal_oed_gradients gradient checks:
        errors = dopt_objective.check_apply_jacobian(design_weights, disp=False)
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.5
        """
        benchmark = self._create_benchmark()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        d_opt_obj = DOptimalLinearModelObjective(
            benchmark.design_matrix(),
            self._bkd.asarray(np.array(benchmark.noise_var())),
            self._bkd.asarray(np.array(benchmark.prior_var())),
            self._bkd,
        )

        # Wrap for DerivativeChecker
        def value_fun(samples: Array) -> Array:
            nsamples = samples.shape[1]
            results = []
            for ii in range(nsamples):
                w = samples[:, ii : ii + 1]
                val = d_opt_obj(w)
                results.append(val[0, 0])
            return self._bkd.reshape(self._bkd.stack(results), (1, nsamples))

        def jacobian_fun(sample: Array) -> Array:
            return d_opt_obj.jacobian(sample)

        def hvp_fun(sample: Array, vec: Array) -> Array:
            return d_opt_obj.hvp(sample, vec).T

        wrapped = FunctionWithJacobianAndHVPFromCallable(
            nvars=d_opt_obj.nvars(),
            fun=value_fun,
            jacobian=jacobian_fun,
            hvp=hvp_fun,
            bkd=self._bkd,
        )

        checker = DerivativeChecker(wrapped)
        errors = checker.check_derivatives(weights)

        # Jacobian check
        jac_ratio = float(self._bkd.to_numpy(checker.error_ratio(errors[0])))
        self.assertLessEqual(jac_ratio, 1e-6)

        # HVP check
        hvp_ratio = float(self._bkd.to_numpy(checker.error_ratio(errors[1])))
        self.assertLessEqual(hvp_ratio, 1e-6)

    # ==========================================================================
    # Data generation tests
    # ==========================================================================

    def test_generate_data_shapes(self):
        """Test generate_data returns correct shapes."""
        benchmark = self._create_benchmark()
        nsamples = 10

        theta, y = benchmark.generate_data(nsamples)

        self.assertEqual(theta.shape, (benchmark.nparams(), nsamples))
        self.assertEqual(y.shape, (benchmark.nobs(), nsamples))

    def test_generate_noisy_data_shapes(self):
        """Test generate_noisy_data returns correct shapes."""
        benchmark = self._create_benchmark()
        nsamples = 10

        theta, y_clean, y_noisy = benchmark.generate_noisy_data(nsamples)

        self.assertEqual(theta.shape, (benchmark.nparams(), nsamples))
        self.assertEqual(y_clean.shape, (benchmark.nobs(), nsamples))
        self.assertEqual(y_noisy.shape, (benchmark.nobs(), nsamples))

    def test_generate_data_reproducible(self):
        """Test generate_data is reproducible with same seed."""
        benchmark = self._create_benchmark()

        theta1, y1 = benchmark.generate_data(10, seed=42)
        theta2, y2 = benchmark.generate_data(10, seed=42)

        self._bkd.assert_allclose(theta1, theta2, rtol=1e-12)
        self._bkd.assert_allclose(y1, y2, rtol=1e-12)

    def test_forward_model_consistency(self):
        """Test y = A @ theta (forward model relationship)."""
        benchmark = self._create_benchmark()
        nsamples = 5

        theta, y = benchmark.generate_data(nsamples)
        A = benchmark.design_matrix()

        y_expected = self._bkd.dot(A, theta)
        self._bkd.assert_allclose(y, y_expected, rtol=1e-12)

    def test_noisy_data_has_noise(self):
        """Test noisy data differs from clean data."""
        benchmark = self._create_benchmark()
        nsamples = 10

        _, y_clean, y_noisy = benchmark.generate_noisy_data(nsamples)

        # Should not be identical
        diff = self._bkd.to_numpy(y_noisy - y_clean)
        self.assertGreater(np.abs(diff).max(), 0.0)

        # Noise should have reasonable magnitude (related to noise_std)
        noise_empirical_std = np.std(diff)
        self.assertGreater(noise_empirical_std, 0.1 * self._noise_std)
        self.assertLess(noise_empirical_std, 10.0 * self._noise_std)

    # ==========================================================================
    # Edge cases and parameter variations
    # ==========================================================================

    def test_different_degree(self):
        """Test benchmark works with different polynomial degrees."""
        for degree in [1, 3, 4]:
            benchmark = LinearGaussianOEDBenchmark(
                self._nobs, degree, self._noise_std, self._prior_std, self._bkd
            )
            nparams = degree + 1
            self.assertEqual(benchmark.nparams(), nparams)
            self.assertEqual(benchmark.design_matrix().shape, (self._nobs, nparams))

            weights = self._bkd.ones((self._nobs, 1)) / self._nobs
            eig = benchmark.exact_eig(weights)
            self.assertTrue(np.isfinite(eig))

    def test_different_noise_prior_ratio(self):
        """Test benchmark works with various noise/prior ratios."""
        # High SNR (low noise)
        benchmark_high_snr = LinearGaussianOEDBenchmark(
            self._nobs, self._degree, 0.1, 1.0, self._bkd
        )
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        eig_high = benchmark_high_snr.exact_eig(weights)

        # Low SNR (high noise)
        benchmark_low_snr = LinearGaussianOEDBenchmark(
            self._nobs, self._degree, 1.0, 0.1, self._bkd
        )
        eig_low = benchmark_low_snr.exact_eig(weights)

        # Both should be finite and positive
        self.assertTrue(np.isfinite(eig_high))
        self.assertTrue(np.isfinite(eig_low))
        self.assertGreater(eig_high, 0.0)
        self.assertGreater(eig_low, 0.0)

        # High SNR should generally give higher EIG
        self.assertGreater(eig_high, eig_low)


class TestLinearGaussianBenchmarkStandaloneNumpy(
    TestLinearGaussianBenchmarkStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLinearGaussianBenchmarkStandaloneTorch(
    TestLinearGaussianBenchmarkStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
