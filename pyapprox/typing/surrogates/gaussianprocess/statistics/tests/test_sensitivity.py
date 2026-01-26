"""
Tests for GP sensitivity analysis (Sobol indices).

Tests the GaussianProcessSensitivity class for computing main effect and
total effect Sobol indices from fitted GPs.
"""
import unittest
from typing import Generic, Any, List
import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401
from pyapprox.typing.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics import (
    SeparableKernelIntegralCalculator,
    GaussianProcessStatistics,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)


def _create_quadrature_bases(
    marginals: List[Any], nquad_points: int, bkd: Backend[Array]
) -> List[Any]:
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestGaussianProcessSensitivity(Generic[Array], unittest.TestCase):
    """
    Base test class for GaussianProcessSensitivity.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D GP with product kernel
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        self._kernel = k1 * k2

        self._gp = ExactGaussianProcess(
            self._kernel,
            nvars=2,
            bkd=self._bkd,
            nugget=1e-6
        )

        # Training data
        self._n_train = 15
        X_train_np = np.random.rand(2, self._n_train) * 2 - 1  # [-1, 1]^2
        y_train_np = np.sin(np.pi * X_train_np[0, :]) * np.cos(np.pi * X_train_np[1, :])
        y_train_np = y_train_np.reshape(-1, 1)

        self._X_train = self._bkd.array(X_train_np)
        self._y_train = self._bkd.array(y_train_np)

        self._gp.fit(self._X_train, self._y_train)

        # Marginal distributions
        self._marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Create quadrature bases using sparse grid infrastructure
        self._nquad_points = 30
        bases = _create_quadrature_bases(
            self._marginals, self._nquad_points, self._bkd
        )

        # Create calculator and statistics
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, bases, bkd=self._bkd
        )
        self._stats = GaussianProcessStatistics(self._gp, self._calc)
        self._sens = GaussianProcessSensitivity(self._stats)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_nvars(self) -> None:
        """Test nvars returns correct number of variables."""
        self.assertEqual(self._sens.nvars(), 2)

    def test_conditional_variance_scalar(self) -> None:
        """Test conditional_variance returns a scalar."""
        index = self._bkd.asarray([1.0, 0.0])
        V = self._sens.conditional_variance(index)
        self.assertEqual(len(V.shape), 0)

    def test_conditional_variance_nonnegative(self) -> None:
        """Test conditional_variance >= 0."""
        index = self._bkd.asarray([1.0, 0.0])
        V = self._sens.conditional_variance(index)
        self.assertGreaterEqual(float(self._bkd.to_numpy(V)), 0.0)

    def test_conditional_variance_bounded(self) -> None:
        """Test conditional_variance <= E[gamma_f]."""
        index = self._bkd.asarray([1.0, 0.0])
        V = self._sens.conditional_variance(index)
        total_var = self._stats.mean_of_variance()

        V_val = float(self._bkd.to_numpy(V))
        total_val = float(self._bkd.to_numpy(total_var))

        # Allow small numerical tolerance
        self.assertLessEqual(V_val, total_val + 1e-10)

    def test_main_effect_indices_keys(self) -> None:
        """Test main_effect_indices returns correct keys."""
        main_effects = self._sens.main_effect_indices()
        self.assertEqual(set(main_effects.keys()), {0, 1})

    def test_main_effect_indices_nonnegative(self) -> None:
        """Test main effect indices S_i >= 0."""
        main_effects = self._sens.main_effect_indices()
        for i, S_i in main_effects.items():
            S_i_val = float(self._bkd.to_numpy(S_i))
            self.assertGreaterEqual(
                S_i_val, 0.0,
                f"Main effect S_{i} = {S_i_val} should be >= 0"
            )

    def test_main_effect_indices_bounded(self) -> None:
        """Test main effect indices S_i <= 1."""
        main_effects = self._sens.main_effect_indices()
        for i, S_i in main_effects.items():
            S_i_val = float(self._bkd.to_numpy(S_i))
            self.assertLessEqual(
                S_i_val, 1.0 + 1e-10,
                f"Main effect S_{i} = {S_i_val} should be <= 1"
            )

    def test_main_effect_indices_sum_bounded(self) -> None:
        """Test that sum of main effects <= 1."""
        main_effects = self._sens.main_effect_indices()
        total = sum(float(self._bkd.to_numpy(S_i)) for S_i in main_effects.values())
        self.assertLessEqual(
            total, 1.0 + 1e-10,
            f"Sum of main effects {total} should be <= 1"
        )

    def test_total_effect_indices_keys(self) -> None:
        """Test total_effect_indices returns correct keys."""
        total_effects = self._sens.total_effect_indices()
        self.assertEqual(set(total_effects.keys()), {0, 1})

    def test_total_effect_indices_nonnegative(self) -> None:
        """Test total effect indices T_i >= 0."""
        total_effects = self._sens.total_effect_indices()
        for i, T_i in total_effects.items():
            T_i_val = float(self._bkd.to_numpy(T_i))
            self.assertGreaterEqual(
                T_i_val, 0.0,
                f"Total effect T_{i} = {T_i_val} should be >= 0"
            )

    def test_total_effect_indices_bounded(self) -> None:
        """Test total effect indices T_i <= 1."""
        total_effects = self._sens.total_effect_indices()
        for i, T_i in total_effects.items():
            T_i_val = float(self._bkd.to_numpy(T_i))
            self.assertLessEqual(
                T_i_val, 1.0 + 1e-10,
                f"Total effect T_{i} = {T_i_val} should be <= 1"
            )

    def test_total_effect_ge_main_effect(self) -> None:
        """Test T_i >= S_i for all variables."""
        main_effects = self._sens.main_effect_indices()
        total_effects = self._sens.total_effect_indices()

        for i in range(self._sens.nvars()):
            S_i = float(self._bkd.to_numpy(main_effects[i]))
            T_i = float(self._bkd.to_numpy(total_effects[i]))
            self.assertGreaterEqual(
                T_i, S_i - 1e-10,
                f"Total effect T_{i} = {T_i} should be >= main effect S_{i} = {S_i}"
            )

    def test_caching(self) -> None:
        """Test that results are cached."""
        main1 = self._sens.main_effect_indices()
        main2 = self._sens.main_effect_indices()
        # Same object (cached)
        self.assertIs(main1, main2)


class TestKnownFunctions(Generic[Array], unittest.TestCase):
    """
    Test Sobol indices on functions with known analytical indices.

    For additive functions f(x) = f_1(x_1) + f_2(x_2):
    - S_1 = Var[f_1] / (Var[f_1] + Var[f_2])
    - S_2 = Var[f_2] / (Var[f_1] + Var[f_2])
    - Sum of S_i = 1 (no interactions)
    - T_i = S_i (no interactions)
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    @slow_test
    def test_additive_function(self) -> None:
        """
        Test GP trained on additive function f(x) = x_1 + x_2.

        For f = x_1 + x_2 on [-1, 1]^2 with uniform distribution:
        - Var[x_1] = Var[x_2] = 1/3
        - So S_1 = S_2 = 0.5

        The GP approximation should give approximately equal indices.
        """
        # Create GP with shorter length scale for better interpolation
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        kernel = k1 * k2

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=self._bkd, nugget=1e-8)

        # Training data: dense grid
        n_1d = 10
        x1 = np.linspace(-1, 1, n_1d)
        x2 = np.linspace(-1, 1, n_1d)
        X1, X2 = np.meshgrid(x1, x2)
        X_train_np = np.vstack([X1.flatten(), X2.flatten()])

        # Additive function: f = x_1 + x_2
        y_train_np = (X_train_np[0, :] + X_train_np[1, :]).reshape(-1, 1)

        X_train = self._bkd.array(X_train_np)
        y_train = self._bkd.array(y_train_np)
        gp.fit(X_train, y_train)

        # Create quadrature bases
        marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]
        bases = _create_quadrature_bases(marginals, 40, self._bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, bkd=self._bkd)
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        # Get indices
        main_effects = sens.main_effect_indices()
        total_effects = sens.total_effect_indices()

        S_0 = float(self._bkd.to_numpy(main_effects[0]))
        S_1 = float(self._bkd.to_numpy(main_effects[1]))
        T_0 = float(self._bkd.to_numpy(total_effects[0]))
        T_1 = float(self._bkd.to_numpy(total_effects[1]))

        # For additive function, expect S_0 ≈ S_1 ≈ 0.5
        self.assertAlmostEqual(S_0, 0.5, delta=0.1,
            msg=f"Expected S_0 ≈ 0.5, got {S_0}")
        self.assertAlmostEqual(S_1, 0.5, delta=0.1,
            msg=f"Expected S_1 ≈ 0.5, got {S_1}")

        # Sum should be close to 1 (additive = no interactions)
        self.assertAlmostEqual(S_0 + S_1, 1.0, delta=0.1,
            msg=f"Sum of main effects {S_0 + S_1} should ≈ 1")

        # For additive function, T_i ≈ S_i
        self.assertAlmostEqual(T_0, S_0, delta=0.1,
            msg=f"Expected T_0 ≈ S_0, got T_0={T_0}, S_0={S_0}")
        self.assertAlmostEqual(T_1, S_1, delta=0.1,
            msg=f"Expected T_1 ≈ S_1, got T_1={T_1}, S_1={S_1}")

    @slow_test
    def test_single_variable_function(self) -> None:
        """
        Test GP trained on single-variable function f(x) = sin(π x_1).

        For f = sin(π x_1) on [-1, 1]^2:
        - Var[f] depends only on x_1
        - S_1 ≈ 1, S_2 ≈ 0
        - T_1 ≈ 1, T_2 ≈ 0
        """
        # Create GP
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        kernel = k1 * k2

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=self._bkd, nugget=1e-8)

        # Training data: dense grid
        n_1d = 10
        x1 = np.linspace(-1, 1, n_1d)
        x2 = np.linspace(-1, 1, n_1d)
        X1, X2 = np.meshgrid(x1, x2)
        X_train_np = np.vstack([X1.flatten(), X2.flatten()])

        # Single-variable function: f = sin(π x_1)
        y_train_np = np.sin(np.pi * X_train_np[0, :]).reshape(-1, 1)

        X_train = self._bkd.array(X_train_np)
        y_train = self._bkd.array(y_train_np)
        gp.fit(X_train, y_train)

        # Create quadrature bases
        marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]
        bases = _create_quadrature_bases(marginals, 40, self._bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, bkd=self._bkd)
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        # Get indices
        main_effects = sens.main_effect_indices()
        total_effects = sens.total_effect_indices()

        S_0 = float(self._bkd.to_numpy(main_effects[0]))
        S_1 = float(self._bkd.to_numpy(main_effects[1]))
        T_0 = float(self._bkd.to_numpy(total_effects[0]))
        T_1 = float(self._bkd.to_numpy(total_effects[1]))

        # For single-variable function:
        # - S_0 should dominate (close to 1)
        # - S_1 should be small (close to 0)
        self.assertGreater(S_0, 0.8,
            msg=f"Expected S_0 > 0.8, got {S_0}")
        self.assertLess(S_1, 0.2,
            msg=f"Expected S_1 < 0.2, got {S_1}")

        # Total effects should follow same pattern
        self.assertGreater(T_0, 0.8,
            msg=f"Expected T_0 > 0.8, got {T_0}")
        self.assertLess(T_1, 0.2,
            msg=f"Expected T_1 < 0.2, got {T_1}")

    @slow_test
    def test_multiplicative_function(self) -> None:
        """
        Test GP trained on multiplicative function f(x) = x_1 * x_2.

        For f = x_1 * x_2 on [-1, 1]^2 with uniform distribution:
        - This has pure interaction (no main effects)
        - S_1 = S_2 = 0 (no main effects)
        - T_1 = T_2 = 1 (all variance is interaction)
        """
        # Create GP
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        kernel = k1 * k2

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=self._bkd, nugget=1e-8)

        # Training data: dense grid
        n_1d = 10
        x1 = np.linspace(-1, 1, n_1d)
        x2 = np.linspace(-1, 1, n_1d)
        X1, X2 = np.meshgrid(x1, x2)
        X_train_np = np.vstack([X1.flatten(), X2.flatten()])

        # Multiplicative function: f = x_1 * x_2
        y_train_np = (X_train_np[0, :] * X_train_np[1, :]).reshape(-1, 1)

        X_train = self._bkd.array(X_train_np)
        y_train = self._bkd.array(y_train_np)
        gp.fit(X_train, y_train)

        # Create quadrature bases
        marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]
        bases = _create_quadrature_bases(marginals, 40, self._bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, bkd=self._bkd)
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        # Get indices
        main_effects = sens.main_effect_indices()
        total_effects = sens.total_effect_indices()

        S_0 = float(self._bkd.to_numpy(main_effects[0]))
        S_1 = float(self._bkd.to_numpy(main_effects[1]))
        T_0 = float(self._bkd.to_numpy(total_effects[0]))
        T_1 = float(self._bkd.to_numpy(total_effects[1]))

        # For multiplicative function:
        # - Main effects should be small (pure interaction)
        # - Total effects should be equal (symmetric function)
        self.assertLess(S_0 + S_1, 0.3,
            msg=f"Sum of main effects {S_0 + S_1} should be small for multiplicative function")

        # T_0 and T_1 should be similar (symmetric function)
        self.assertAlmostEqual(T_0, T_1, delta=0.2,
            msg=f"Expected T_0 ≈ T_1, got T_0={T_0}, T_1={T_1}")


class TestConditionalPAndU(Generic[Array], unittest.TestCase):
    """
    Test conditional_P and conditional_u consistency.

    Verifies that the conditional quantities satisfy expected relationships.

    Key insight: conditional_variance(p) computes E[Var_{z_p}[E_{z_~p}[f | z_p]]]
    - This is the variance of the conditional mean as conditioned variables change.

    Special cases:
    - index = [0,0,...] (nothing conditioned): E[f] = η is constant, so Var = 0
    - index = [1,1,...] (everything conditioned): E[f|z] = f(z), so Var = E[γ_f]
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D GP
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        kernel = k1 * k2

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=self._bkd, nugget=1e-6)

        n_train = 10
        X_train = self._bkd.array(np.random.rand(2, n_train) * 2 - 1)
        y_train = self._bkd.array(np.random.rand(n_train, 1))
        gp.fit(X_train, y_train)

        marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]
        bases = _create_quadrature_bases(marginals, 30, self._bkd)
        self._calc = SeparableKernelIntegralCalculator(gp, bases, bkd=self._bkd)
        self._stats = GaussianProcessStatistics(gp, self._calc)
        self._sens = GaussianProcessSensitivity(self._stats)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_no_conditioning_gives_zero(self) -> None:
        """
        When nothing is conditioned (index all zeros), conditional mean is
        constant η, so its variance is 0.

        E[γ_f^(∅)] = 0

        Math: P_p = τ τᵀ, u_p = u
        - ζ_p = (τᵀ A⁻¹ y)² = η²
        - v_p² = u - τᵀ A⁻¹ τ = ς²
        - E[γ_f^(∅)] = η² + s² ς² - η² - s² ς² = 0
        """
        # All zeros = nothing conditioned
        index = self._bkd.asarray([0.0, 0.0])
        cond_var = self._sens.conditional_variance(index)

        # Should be zero (conditional mean is constant)
        # Allow small numerical tolerance due to floating point arithmetic
        self._bkd.assert_allclose(
            cond_var, self._bkd.asarray(0.0), atol=1e-6,
            err_msg="Conditional variance with no conditioning should be 0"
        )

    def test_full_conditioning_gives_full_variance(self) -> None:
        """
        When everything is conditioned (index all ones), we get the full variance.

        E[γ_f^(all)] = E[γ_f]

        Math: P_p = P, u_p = 1
        - ζ_p = ζ
        - v_p² = 1 - Tr[P A⁻¹] = v²
        - E[γ_f^(all)] = ζ + s² v² - η² - s² ς² = E[γ_f]
        """
        # All ones = everything conditioned
        index = self._bkd.asarray([1.0, 1.0])
        cond_var = self._sens.conditional_variance(index)
        mean_var = self._stats.mean_of_variance()

        # These should be equal
        self._bkd.assert_allclose(
            cond_var, mean_var, rtol=1e-10,
            err_msg="Conditional variance with full conditioning should equal mean of variance"
        )


# NumPy backend tests
class TestGaussianProcessSensitivityNumpy(
    TestGaussianProcessSensitivity[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKnownFunctionsNumpy(TestKnownFunctions[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConditionalPAndUNumpy(TestConditionalPAndU[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestGaussianProcessSensitivityTorch(
    TestGaussianProcessSensitivity[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestKnownFunctionsTorch(TestKnownFunctions[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestConditionalPAndUTorch(TestConditionalPAndU[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
