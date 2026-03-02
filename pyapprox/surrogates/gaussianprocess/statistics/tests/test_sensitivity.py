"""
Tests for GP sensitivity analysis (Sobol indices).

Tests the GaussianProcessSensitivity class for computing main effect and
total effect Sobol indices from fitted GPs.
"""
import math

import numpy as np

from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.statistics import (
    GaussianProcessStatistics,
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.surrogates.kernels.composition import (
    SeparableProductKernel,
)
from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.util.test_utils import slow_test


def _create_quadrature_bases(
    marginals, nquad_points, bkd,
):
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestGaussianProcessSensitivity:
    """
    Test class for GaussianProcessSensitivity.
    """

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 2D GP with separable product kernel
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(
            kernel, nvars=2, bkd=bkd, nugget=1e-6
        )
        # Skip hyperparameter optimization for these tests
        gp.hyp_list().set_all_inactive()

        # Training data
        n_train = 15
        X_train_np = np.random.rand(2, n_train) * 2 - 1  # [-1, 1]^2
        X_train = bkd.array(X_train_np)
        # Use backend math operations, shape: (nqoi, n_train)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :]) *
            bkd.cos(math.pi * X_train[1, :]),
            (1, -1)
        )

        gp.fit(X_train, y_train)

        # Marginal distributions
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Create quadrature bases using sparse grid infrastructure
        nquad_points = 30
        bases = _create_quadrature_bases(
            marginals, nquad_points, bkd
        )

        # Create calculator and statistics
        calc = SeparableKernelIntegralCalculator(
            gp, bases, marginals, bkd=bkd
        )
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        return sens, stats

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct number of variables."""
        sens, _ = self._setup(bkd)
        assert sens.nvars() == 2

    def test_conditional_variance_scalar(self, bkd) -> None:
        """Test conditional_variance returns a scalar."""
        sens, _ = self._setup(bkd)
        index = bkd.asarray([1.0, 0.0])
        V = sens.conditional_variance(index)
        assert len(V.shape) == 0

    def test_conditional_variance_nonnegative(self, bkd) -> None:
        """Test conditional_variance >= 0."""
        sens, _ = self._setup(bkd)
        index = bkd.asarray([1.0, 0.0])
        V = sens.conditional_variance(index)
        assert float(bkd.to_numpy(V)) >= 0.0

    def test_conditional_variance_bounded(self, bkd) -> None:
        """Test conditional_variance <= E[gamma_f]."""
        sens, stats = self._setup(bkd)
        index = bkd.asarray([1.0, 0.0])
        V = sens.conditional_variance(index)
        total_var = stats.mean_of_variance()

        V_val = float(bkd.to_numpy(V))
        total_val = float(bkd.to_numpy(total_var))

        # Allow small numerical tolerance
        assert V_val <= total_val + 1e-10

    def test_main_effect_indices_keys(self, bkd) -> None:
        """Test main_effect_indices returns correct keys."""
        sens, _ = self._setup(bkd)
        main_effects = sens.main_effect_indices()
        assert set(main_effects.keys()) == {0, 1}

    def test_main_effect_indices_nonnegative(self, bkd) -> None:
        """Test main effect indices S_i >= 0."""
        sens, _ = self._setup(bkd)
        main_effects = sens.main_effect_indices()
        for i, S_i in main_effects.items():
            S_i_val = float(bkd.to_numpy(S_i))
            assert S_i_val >= 0.0, (
                f"Main effect S_{i} = {S_i_val} should be >= 0"
            )

    def test_main_effect_indices_bounded(self, bkd) -> None:
        """Test main effect indices S_i <= 1."""
        sens, _ = self._setup(bkd)
        main_effects = sens.main_effect_indices()
        for i, S_i in main_effects.items():
            S_i_val = float(bkd.to_numpy(S_i))
            assert S_i_val <= 1.0 + 1e-10, (
                f"Main effect S_{i} = {S_i_val} should be <= 1"
            )

    def test_main_effect_indices_sum_bounded(self, bkd) -> None:
        """Test that sum of main effects <= 1."""
        sens, _ = self._setup(bkd)
        main_effects = sens.main_effect_indices()
        total = sum(
            float(bkd.to_numpy(S_i)) for S_i in main_effects.values()
        )
        assert total <= 1.0 + 1e-10, (
            f"Sum of main effects {total} should be <= 1"
        )

    def test_total_effect_indices_keys(self, bkd) -> None:
        """Test total_effect_indices returns correct keys."""
        sens, _ = self._setup(bkd)
        total_effects = sens.total_effect_indices()
        assert set(total_effects.keys()) == {0, 1}

    def test_total_effect_indices_nonnegative(self, bkd) -> None:
        """Test total effect indices T_i >= 0."""
        sens, _ = self._setup(bkd)
        total_effects = sens.total_effect_indices()
        for i, T_i in total_effects.items():
            T_i_val = float(bkd.to_numpy(T_i))
            assert T_i_val >= 0.0, (
                f"Total effect T_{i} = {T_i_val} should be >= 0"
            )

    def test_total_effect_indices_bounded(self, bkd) -> None:
        """Test total effect indices T_i <= 1."""
        sens, _ = self._setup(bkd)
        total_effects = sens.total_effect_indices()
        for i, T_i in total_effects.items():
            T_i_val = float(bkd.to_numpy(T_i))
            assert T_i_val <= 1.0 + 1e-10, (
                f"Total effect T_{i} = {T_i_val} should be <= 1"
            )

    def test_total_effect_ge_main_effect(self, bkd) -> None:
        """Test T_i >= S_i for all variables."""
        sens, _ = self._setup(bkd)
        main_effects = sens.main_effect_indices()
        total_effects = sens.total_effect_indices()

        for i in range(sens.nvars()):
            S_i = float(bkd.to_numpy(main_effects[i]))
            T_i = float(bkd.to_numpy(total_effects[i]))
            assert T_i >= S_i - 1e-10, (
                f"Total effect T_{i} = {T_i} should be >= main effect S_{i} = {S_i}"
            )

    def test_caching(self, bkd) -> None:
        """Test that results are cached."""
        sens, _ = self._setup(bkd)
        main1 = sens.main_effect_indices()
        main2 = sens.main_effect_indices()
        # Same object (cached)
        assert main1 is main2


class TestKnownFunctions:
    """
    Test Sobol indices on functions with known analytical indices.

    For additive functions f(x) = f_1(x_1) + f_2(x_2):
    - S_1 = Var[f_1] / (Var[f_1] + Var[f_2])
    - S_2 = Var[f_2] / (Var[f_1] + Var[f_2])
    - Sum of S_i = 1 (no interactions)
    - T_i = S_i (no interactions)
    """

    @slow_test
    def test_additive_function(self, bkd) -> None:
        """
        Test GP trained on additive function f(x) = x_1 + x_2.

        For f = x_1 + x_2 on [-1, 1]^2 with uniform distribution:
        - Var[x_1] = Var[x_2] = 1/3
        - So S_1 = S_2 = 0.5

        The GP approximation should give approximately equal indices.
        """
        np.random.seed(42)
        # Create GP with shorter length scale for better interpolation
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-8)
        # Skip hyperparameter optimization since SeparableProductKernel
        # doesn't implement jacobian_wrt_params yet
        gp.hyp_list().set_all_inactive()

        # Training data: dense grid
        n_1d = 10
        x1 = bkd.linspace(-1.0, 1.0, n_1d)
        x2 = bkd.linspace(-1.0, 1.0, n_1d)
        X1, X2 = bkd.meshgrid(x1, x2)
        X_train = bkd.vstack([bkd.flatten(X1), bkd.flatten(X2)])

        # Additive function: f = x_1 + x_2, shape: (nqoi, n_train)
        y_train = bkd.reshape(X_train[0, :] + X_train[1, :], (1, -1))
        gp.fit(X_train, y_train)

        # Create quadrature bases
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]
        bases = _create_quadrature_bases(marginals, 40, bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        # Get indices
        main_effects = sens.main_effect_indices()
        total_effects = sens.total_effect_indices()

        S_0 = float(bkd.to_numpy(main_effects[0]))
        S_1 = float(bkd.to_numpy(main_effects[1]))
        T_0 = float(bkd.to_numpy(total_effects[0]))
        T_1 = float(bkd.to_numpy(total_effects[1]))

        # For additive function, expect S_0 approx S_1 approx 0.5
        assert abs(S_0 - 0.5) < 0.1, f"Expected S_0 approx 0.5, got {S_0}"
        assert abs(S_1 - 0.5) < 0.1, f"Expected S_1 approx 0.5, got {S_1}"

        # Sum should be close to 1 (additive = no interactions)
        assert abs(S_0 + S_1 - 1.0) < 0.1, (
            f"Sum of main effects {S_0 + S_1} should approx 1"
        )

        # For additive function, T_i approx S_i
        assert abs(T_0 - S_0) < 0.1, (
            f"Expected T_0 approx S_0, got T_0={T_0}, S_0={S_0}"
        )
        assert abs(T_1 - S_1) < 0.1, (
            f"Expected T_1 approx S_1, got T_1={T_1}, S_1={S_1}"
        )

    @slow_test
    def test_single_variable_function(self, bkd) -> None:
        """
        Test GP trained on single-variable function f(x) = sin(pi x_1).

        For f = sin(pi x_1) on [-1, 1]^2:
        - Var[f] depends only on x_1
        - S_1 approx 1, S_2 approx 0
        - T_1 approx 1, T_2 approx 0
        """
        np.random.seed(42)
        # Create GP
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-8)
        # Skip hyperparameter optimization since SeparableProductKernel
        # doesn't implement jacobian_wrt_params yet
        gp.hyp_list().set_all_inactive()

        # Training data: dense grid
        n_1d = 10
        x1 = bkd.linspace(-1.0, 1.0, n_1d)
        x2 = bkd.linspace(-1.0, 1.0, n_1d)
        X1, X2 = bkd.meshgrid(x1, x2)
        X_train = bkd.vstack([bkd.flatten(X1), bkd.flatten(X2)])

        # Single-variable function: f = sin(pi x_1), shape: (nqoi, n_train)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :]), (1, -1)
        )
        gp.fit(X_train, y_train)

        # Create quadrature bases
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]
        bases = _create_quadrature_bases(marginals, 40, bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        # Get indices
        main_effects = sens.main_effect_indices()
        total_effects = sens.total_effect_indices()

        S_0 = float(bkd.to_numpy(main_effects[0]))
        S_1 = float(bkd.to_numpy(main_effects[1]))
        T_0 = float(bkd.to_numpy(total_effects[0]))
        T_1 = float(bkd.to_numpy(total_effects[1]))

        # For single-variable function:
        # - S_0 should dominate (close to 1)
        # - S_1 should be small (close to 0)
        assert S_0 > 0.8, f"Expected S_0 > 0.8, got {S_0}"
        assert S_1 < 0.2, f"Expected S_1 < 0.2, got {S_1}"

        # Total effects should follow same pattern
        assert T_0 > 0.8, f"Expected T_0 > 0.8, got {T_0}"
        assert T_1 < 0.2, f"Expected T_1 < 0.2, got {T_1}"

    @slow_test
    def test_multiplicative_function(self, bkd) -> None:
        """
        Test GP trained on multiplicative function f(x) = x_1 * x_2.

        For f = x_1 * x_2 on [-1, 1]^2 with uniform distribution:
        - This has pure interaction (no main effects)
        - S_1 = S_2 = 0 (no main effects)
        - T_1 = T_2 = 1 (all variance is interaction)
        """
        np.random.seed(42)
        # Create GP
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-8)
        # Skip hyperparameter optimization since SeparableProductKernel
        # doesn't implement jacobian_wrt_params yet
        gp.hyp_list().set_all_inactive()

        # Training data: dense grid
        n_1d = 10
        x1 = bkd.linspace(-1.0, 1.0, n_1d)
        x2 = bkd.linspace(-1.0, 1.0, n_1d)
        X1, X2 = bkd.meshgrid(x1, x2)
        X_train = bkd.vstack([bkd.flatten(X1), bkd.flatten(X2)])

        # Multiplicative function: f = x_1 * x_2, shape: (nqoi, n_train)
        y_train = bkd.reshape(X_train[0, :] * X_train[1, :], (1, -1))
        gp.fit(X_train, y_train)

        # Create quadrature bases
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]
        bases = _create_quadrature_bases(marginals, 40, bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        # Get indices
        main_effects = sens.main_effect_indices()
        total_effects = sens.total_effect_indices()

        S_0 = float(bkd.to_numpy(main_effects[0]))
        S_1 = float(bkd.to_numpy(main_effects[1]))
        T_0 = float(bkd.to_numpy(total_effects[0]))
        T_1 = float(bkd.to_numpy(total_effects[1]))

        # For multiplicative function:
        # - Main effects should be small (pure interaction)
        # - Total effects should be equal (symmetric function)
        assert S_0 + S_1 < 0.3, (
            f"Sum of main effects {S_0 + S_1} "
            "should be small for multiplicative function"
        )

        # T_0 and T_1 should be similar (symmetric function)
        assert abs(T_0 - T_1) < 0.2, (
            f"Expected T_0 approx T_1, got T_0={T_0}, T_1={T_1}"
        )


class TestIshigamiBenchmark:
    """
    Test GP sensitivity indices against Ishigami function benchmark.

    The Ishigami function is a smooth function with analytically known Sobol
    indices. It's defined on [-pi, pi]^3 with
    f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1).

    We use a=0, b=1 to get cross terms between x1 and x3 while keeping the
    function simple enough for the GP to approximate well.

    CRITICAL: We first verify the GP interpolation error is small (< stats_tol)
    at test points. This is a necessary condition for accurate sensitivity indices.
    """

    def _verify_gp_interpolation(
        self,
        bkd,
        gp,
        test_func,
        X_test,
        stats_tol,
    ):
        """
        Verify GP interpolation error is below stats tolerance.

        Parameters
        ----------
        bkd : Backend
            Backend instance.
        gp : ExactGaussianProcess
            Fitted GP.
        test_func : callable
            True function, takes X of shape (nvars, nsamples).
        X_test : Array
            Test points of shape (nvars, n_test).
        stats_tol : float
            Tolerance for statistics tests.

        Returns
        -------
        max_error : float
            Maximum absolute interpolation error.

        Raises
        ------
        AssertionError
            If max error exceeds stats_tol.
        """
        # Get GP predictions
        y_pred = gp.predict(X_test)  # Shape: (n_test, 1)
        y_pred = bkd.reshape(y_pred, (-1,))

        # Get true values
        y_true = test_func(X_test)  # Shape: (1, n_test) or (n_test,)
        if y_true.ndim == 2:
            y_true = bkd.reshape(y_true, (-1,))

        # Compute max error
        errors = bkd.abs(y_pred - y_true)
        max_error = float(bkd.to_numpy(bkd.max(errors)))

        assert max_error < stats_tol, (
            f"GP interpolation error ({max_error:.2e}) exceeds stats tolerance "
            f"({stats_tol:.2e}). Increase training points or adjust kernel."
        )

        return max_error

    @slow_test
    def test_ishigami_sobol_indices(self, bkd) -> None:
        """
        Test Ishigami function Sobol indices match analytical values.

        Note: We artificially make the function 2D by setting a=0 (removing
        the x2 dependence) so that the GP interpolation error can be made
        sufficiently small with a reasonable number of training points.
        The full 3D Ishigami function (a=7, b=0.1) requires many more points
        to achieve the same accuracy.

        Uses a=0, b=0.01 which gives:
        f(x) = sin(x1) * (1 + 0.01*x3^4)

        This has interaction between x1 and x3, but x2 has no effect.
        """
        np.random.seed(42)
        pi = float(bkd.to_numpy(bkd.asarray(np.pi)))

        # Import utilities
        from pyapprox.benchmarks.functions.algebraic.ishigami import (
            IshigamiFunction,
            IshigamiSensitivityIndices,
        )
        from pyapprox.expdesign.quadrature.sobol import SobolSampler
        from pyapprox.probability.joint.independent import (
            IndependentJoint,
        )

        # Use a=0, b=0.01 for cross terms between x1 and x3
        # x2 has no effect, so we only need 1 point in that dimension
        a, b = 0.0, 0.01
        ishigami = IshigamiFunction(bkd, a=a, b=b)
        indices = IshigamiSensitivityIndices(bkd, a=a, b=b)

        # Analytical Sobol indices
        S_exact = [
            float(bkd.to_numpy(indices.main_effects()[i, 0])) for i in range(3)
        ]
        T_exact = [
            float(bkd.to_numpy(indices.total_effects()[i, 0]))
            for i in range(3)
        ]

        # Tolerance for GP interpolation (relaxed for Sobol sampling)
        interp_tol = 5e-4

        # Create GP with multivariate SE kernel (supports jacobian_wrt_params
        # for hyperparameter optimization, unlike SeparableProductKernel)
        kernel = SquaredExponentialKernel(
            [1.0, 1.0, 1.0], (0.1, 10.0), 3, bkd
        )

        gp = ExactGaussianProcess(kernel, nvars=3, bkd=bkd, nugget=1e-10)

        # Create training data using Sobol sequence on [-pi, pi]^3
        marginals = [UniformMarginal(-pi, pi, bkd) for _ in range(3)]
        dist = IndependentJoint(marginals, bkd)
        sampler = SobolSampler(3, bkd, distribution=dist, seed=42)
        n_train = 700
        X_train, _ = sampler.sample(n_train)

        # Evaluate Ishigami function
        y_train = ishigami(X_train)  # Shape: (nqoi, n_train)
        gp.fit(X_train, y_train)

        # Generate test points
        n_test = 100
        X_test = bkd.array(np.random.rand(3, n_test) * 2 * pi - pi)

        # Verify GP interpolation error is small enough
        self._verify_gp_interpolation(bkd, gp, ishigami, X_test, interp_tol)

        # Create quadrature bases on [-pi, pi]^3
        marginals = [UniformMarginal(-pi, pi, bkd) for _ in range(3)]
        bases = _create_quadrature_bases(marginals, 40, bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        # Get computed indices
        main_effects = sens.main_effect_indices()
        total_effects = sens.total_effect_indices()

        S_values = [float(bkd.to_numpy(main_effects[i])) for i in range(3)]
        T_values = [float(bkd.to_numpy(total_effects[i])) for i in range(3)]

        # Get exact Sobol indices including interactions
        # Order from IshigamiSensitivityIndices: S_1, S_2, S_3, S_12, S_13, S_23, S_123
        sobol_exact = indices.sobol_indices()
        S_13_exact = float(bkd.to_numpy(sobol_exact[4, 0]))  # S_13 interaction

        # Compute S_13 from GP: S_13 = T_1 - S_1 (for this function with a=0)
        # Since only x1 and x3 interact, T_1 = S_1 + S_13
        S_13_computed = T_values[0] - S_values[0]

        # Verify main effect indices match analytical values
        for i in range(3):
            bkd.assert_allclose(
                bkd.asarray([S_values[i]]),
                bkd.asarray([S_exact[i]]),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"S_{i} = {S_values[i]:.4f}, expected {S_exact[i]:.4f}",
            )

        # Verify total effect indices match analytical values
        for i in range(3):
            bkd.assert_allclose(
                bkd.asarray([T_values[i]]),
                bkd.asarray([T_exact[i]]),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"T_{i} = {T_values[i]:.4f}, expected {T_exact[i]:.4f}",
            )

        # Verify S_13 interaction index matches analytical value
        bkd.assert_allclose(
            bkd.asarray([S_13_computed]),
            bkd.asarray([S_13_exact]),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"S_13 = {S_13_computed:.4f}, expected {S_13_exact:.4f}",
        )


class TestConditionalPAndU:
    """
    Test conditional_P and conditional_u consistency.

    Verifies that the conditional quantities satisfy expected relationships.

    Key insight: conditional_variance(p) computes E[Var_{z_p}[E_{z_~p}[f | z_p]]]
    - This is the variance of the conditional mean as conditioned variables change.

    Special cases:
    - index = [0,0,...] (nothing conditioned): E[f] = eta is constant, so Var = 0
    - index = [1,1,...] (everything conditioned): E[f|z] = f(z), so Var = E[gamma_f]
    """

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 2D GP
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-6)
        # Skip hyperparameter optimization since SeparableProductKernel
        # doesn't implement jacobian_wrt_params yet
        gp.hyp_list().set_all_inactive()

        n_train = 10
        X_train = bkd.array(np.random.rand(2, n_train) * 2 - 1)
        y_train = bkd.array(np.random.rand(n_train).reshape(1, -1))
        gp.fit(X_train, y_train)

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]
        bases = _create_quadrature_bases(marginals, 30, bkd)
        calc = SeparableKernelIntegralCalculator(
            gp, bases, marginals, bkd=bkd
        )
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        return sens, stats

    def test_no_conditioning_gives_zero(self, bkd) -> None:
        """
        When nothing is conditioned (index all zeros), conditional mean is
        constant eta, so its variance is 0.

        E[gamma_f^(empty)] = 0

        Math: P_p = tau tau^T, u_p = u
        - zeta_p = (tau^T A^{-1} y)**2 = eta**2
        - v_p**2 = u - tau^T A^{-1} tau = varsigma**2
        - E[gamma_f^(empty)] = eta**2 + s**2 varsigma**2 - eta**2 - s**2 varsigma**2 = 0
        """
        sens, _ = self._setup(bkd)
        # All zeros = nothing conditioned
        index = bkd.asarray([0.0, 0.0])
        cond_var = sens.conditional_variance(index)

        # Should be zero (conditional mean is constant)
        # Allow small numerical tolerance due to floating point arithmetic
        bkd.assert_allclose(
            cond_var,
            bkd.asarray(0.0),
            atol=1e-6,
            err_msg="Conditional variance with no conditioning should be 0",
        )

    def test_full_conditioning_gives_full_variance(self, bkd) -> None:
        """
        When everything is conditioned (index all ones), we get the full variance.

        E[gamma_f^(all)] = E[gamma_f]

        Math: P_p = P, u_p = 1
        - zeta_p = zeta
        - v_p**2 = 1 - Tr[P A^{-1}] = v**2
        - E[gamma_f^(all)] = zeta + s**2 v**2 - eta**2 - s**2 varsigma**2 = E[gamma_f]
        """
        sens, stats = self._setup(bkd)
        # All ones = everything conditioned
        index = bkd.asarray([1.0, 1.0])
        cond_var = sens.conditional_variance(index)
        mean_var = stats.mean_of_variance()

        # These should be equal
        bkd.assert_allclose(
            cond_var,
            mean_var,
            rtol=1e-10,
            err_msg=(
                "Conditional variance with full "
                "conditioning should equal mean of variance"
            ),
        )
