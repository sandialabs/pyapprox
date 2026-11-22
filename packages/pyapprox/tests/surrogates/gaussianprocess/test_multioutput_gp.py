"""
Tests for Gaussian Process with multi-output kernels.

This module tests multi-output GP prediction using IndependentMultiOutputKernel
and LinearCoregionalizationKernel with a single GP handling all outputs.
"""

from typing import List, Tuple

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.gaussianprocess.fitters import (
    MultiOutputGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.gp_loss import (
    GPNegativeLogMarginalLikelihoodLoss,
)
from pyapprox.surrogates.gaussianprocess.multioutput import MultiOutputGP
from pyapprox.surrogates.kernels.base import Kernel
from pyapprox.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.surrogates.kernels.matern import (
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kernels.multioutput import (
    IndependentMultiOutputKernel,
    LinearCoregionalizationKernel,
)
from pyapprox.surrogates.kernels.multioutput.multilevel import MultiLevelKernel
from pyapprox.surrogates.kernels.scalings import (
    PolynomialScalingFunction,
    PolynomialScalingKernel,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slow_test


class TestMultiOutputGPWithIndependentKernel:
    """
    Test multi-output GP using IndependentMultiOutputKernel.

    Tests a single GP that uses a multi-output kernel to handle multiple
    outputs simultaneously, achieving < 1e-6 interpolation accuracy.
    """

    def test_interpolation_accuracy_sufficient_points(self, bkd) -> None:
        """
        Test multi-output GP interpolates training data with < 1e-6 accuracy.

        Uses MultiOutputGP with IndependentMultiOutputKernel to handle
        all outputs simultaneously. Verifies each output correctly interpolates
        its specific function (sin and cos).
        """
        np.random.seed(42)
        nvars = 2
        noutputs = 2
        n_train = 50

        # Create test data
        X_train_np = np.random.randn(nvars, n_train)
        X_train = bkd.array(X_train_np)

        y1 = np.sin(X_train_np[0, :] + X_train_np[1, :])
        y2 = np.cos(X_train_np[0, :] - X_train_np[1, :])

        y_train_stacked_np = np.concatenate([y1, y2])[:, np.newaxis]
        y_train_stacked = bkd.array(y_train_stacked_np)

        y_train_np = np.column_stack([y1, y2])
        y_train_unstacked = bkd.array(y_train_np)

        # Create independent kernels for each output
        kernels = []
        for i in range(noutputs):
            matern = Matern52Kernel([1.0] * nvars, (0.1, 10.0), nvars, bkd)
            constant = PolynomialScalingKernel([1.0], (0.1, 10.0), bkd, nvars=nvars)
            noise = IIDGaussianNoise(1e-14, (1e-16, 1e-12), bkd)
            kernel = constant * matern + noise
            kernels.append(kernel)

        # Create multi-output kernel
        mo_kernel = IndependentMultiOutputKernel(kernels)

        # Create and fit GP (with fixed hyperparameters for interpolation test)
        gp = MultiOutputGP(mo_kernel, nugget=1e-14)
        gp.hyp_list().set_all_inactive()  # Skip optimization for interpolation test
        X_list = [X_train] * noutputs
        result = MultiOutputGPMaximumLikelihoodFitter(bkd).fit(
            gp, X_list, y_train_stacked
        )
        gp = result.surrogate()

        # Predict at training points (should interpolate)
        # predict() returns a list of arrays, each with shape (1, n_train)
        y_pred_list = gp.predict(X_list)

        # Verify each output interpolates its specific function correctly
        X_train_np = bkd.to_numpy(X_train)

        for i in range(noutputs):
            # y_pred_list[i] has shape (1, n_train)
            y_pred_i = y_pred_list[i][0, :]  # (n_train,)
            y_true_i = y_train_unstacked[:, i]  # (n_train,)

            # Check interpolation error
            error_i = bkd.abs(y_pred_i - y_true_i)
            max_error_i = bkd.max(error_i).item()
            assert max_error_i < 1e-6, (
                f"Output {i} error {max_error_i:.2e} exceeds 1e-6"
            )

            # Verify correct function: output 0 = sin, output 1 = cos
            y_pred_i_np = bkd.to_numpy(y_pred_i)
            if i == 0:
                # Output 0: sin(x1 + x2)
                expected = np.sin(X_train_np[0, :] + X_train_np[1, :])
                func_error = np.abs(y_pred_i_np - expected)
                assert np.max(func_error) < 1e-6, "Output 0 does not match sin(x1+x2)"
            elif i == 1:
                # Output 1: cos(x1 - x2)
                expected = np.cos(X_train_np[0, :] - X_train_np[1, :])
                func_error = np.abs(y_pred_i_np - expected)
                assert np.max(func_error) < 1e-6, "Output 1 does not match cos(x1-x2)"

    def test_prediction_at_new_points(self, bkd) -> None:
        """
        Test multi-output GP prediction at new test points.

        Tests both predict() and predict_with_uncertainty() methods.
        Verifies predictions are reasonable approximations of true functions.
        """
        np.random.seed(42)
        nvars = 2
        noutputs = 2
        n_train = 30
        n_test = 10

        # Create training data
        X_train_np = np.random.randn(nvars, n_train)
        X_train = bkd.array(X_train_np)

        y1 = np.sin(X_train_np[0, :] + X_train_np[1, :])
        y2 = np.cos(X_train_np[0, :] - X_train_np[1, :])

        y_train_stacked_np = np.concatenate([y1, y2])[:, np.newaxis]
        y_train_stacked = bkd.array(y_train_stacked_np)

        # Create test data
        X_test_np = np.random.randn(nvars, n_test)
        X_test = bkd.array(X_test_np)

        # True test values for verification
        y_test_0 = np.sin(X_test_np[0, :] + X_test_np[1, :])
        y_test_1 = np.cos(X_test_np[0, :] - X_test_np[1, :])

        # Create multi-output kernel
        kernels = []
        for i in range(noutputs):
            matern = Matern52Kernel([1.0] * nvars, (0.1, 10.0), nvars, bkd)
            kernels.append(matern)

        mo_kernel = IndependentMultiOutputKernel(kernels)

        # Create and fit GP
        gp = MultiOutputGP(mo_kernel, nugget=1e-14)
        X_train_list = [X_train] * noutputs
        result = MultiOutputGPMaximumLikelihoodFitter(bkd).fit(
            gp, X_train_list, y_train_stacked
        )
        gp = result.surrogate()

        # Predict at test points
        X_test_list = [X_test] * noutputs
        # predict() returns a list of arrays, each with shape (1, n_test)
        y_pred_list = gp.predict(X_test_list)

        # Check that we got a list of predictions
        assert len(y_pred_list) == noutputs
        for i in range(noutputs):
            assert y_pred_list[i].shape == (1, n_test)

        # Predictions should be in reasonable range
        for i in range(noutputs):
            y_pred_abs = bkd.abs(y_pred_list[i])
            max_pred = bkd.max(y_pred_abs).item()
            assert max_pred < 10.0

        # Verify predictions are reasonable approximations
        y_pred_0 = bkd.to_numpy(y_pred_list[0][0, :])
        y_pred_1 = bkd.to_numpy(y_pred_list[1][0, :])

        # Should approximate the true functions (not perfect, but reasonable)
        error_0 = np.abs(y_pred_0 - y_test_0)
        error_1 = np.abs(y_pred_1 - y_test_1)

        # With enough training data, predictions should be decent
        assert np.mean(error_0) < 0.5
        assert np.mean(error_1) < 0.5

        # Test predict_with_uncertainty
        # Returns lists of arrays, each with shape (1, n_test)
        y_pred_unc_list, y_std_list = gp.predict_with_uncertainty(X_test_list)

        # Check lengths
        assert len(y_pred_unc_list) == noutputs
        assert len(y_std_list) == noutputs

        # Check shapes
        for i in range(noutputs):
            assert y_pred_unc_list[i].shape == (1, n_test)
            assert y_std_list[i].shape == (1, n_test)

        # Predictions should match
        for i in range(noutputs):
            error_match = bkd.abs(y_pred_list[i] - y_pred_unc_list[i])
            max_match_error = bkd.max(error_match).item()
            assert max_match_error < 1e-10

        # Uncertainties should be positive
        for i in range(noutputs):
            y_std_min = bkd.min(y_std_list[i]).item()
            assert y_std_min >= 0.0

    def test_block_diagonal_structure_preserved(self, bkd) -> None:
        """
        Test that independent kernel maintains block-diagonal structure.

        Uses the kernel directly to verify off-diagonal blocks are zero.
        """
        np.random.seed(42)
        nvars = 2
        noutputs = 2
        n_train = 20

        X_train_np = np.random.randn(nvars, n_train)
        X_train = bkd.array(X_train_np)

        # Create simple kernels
        kernels = []
        for i in range(noutputs):
            matern = Matern52Kernel([1.0] * nvars, (0.1, 10.0), nvars, bkd)
            kernels.append(matern)

        mo_kernel = IndependentMultiOutputKernel(kernels)

        # Build kernel matrix
        X_list = [X_train] * noutputs
        K = mo_kernel(X_list, block_format=False)

        # Check that off-diagonal blocks are zero
        for i in range(noutputs):
            for j in range(noutputs):
                if i != j:
                    row_start = i * n_train
                    row_end = (i + 1) * n_train
                    col_start = j * n_train
                    col_end = (j + 1) * n_train

                    K_ij = K[row_start:row_end, col_start:col_end]
                    K_ij_abs = bkd.abs(K_ij)
                    max_val = bkd.max(K_ij_abs).item()
                    assert max_val < 1e-10, f"Off-diagonal block [{i},{j}] not zero"


class TestMultiOutputGPWithLMCKernel:
    """
    Test multi-output GP using LinearCoregionalizationKernel.

    Tests a single GP with LMC kernel that captures correlations between outputs.
    """

    def test_lmc_interpolation_accuracy(self, bkd) -> None:
        """
        Test LMC kernel achieves < 1e-6 interpolation accuracy.

        LMC kernel models output correlations while maintaining
        high interpolation accuracy. Uses MultiOutputGP.
        """
        np.random.seed(42)
        nvars = 2
        noutputs = 2
        n_train = 40

        X_train_np = np.random.randn(nvars, n_train)
        X_train = bkd.array(X_train_np)

        y1 = np.sin(X_train_np[0, :] + X_train_np[1, :])
        y2 = np.cos(X_train_np[0, :] - X_train_np[1, :])

        y_train_stacked_np = np.concatenate([y1, y2])[:, np.newaxis]
        y_train_stacked = bkd.array(y_train_stacked_np)

        # Create base kernels
        base_kernels = []
        for q in range(2):  # 2 components
            matern = Matern52Kernel([1.0] * nvars, (0.1, 10.0), nvars, bkd)
            constant = PolynomialScalingKernel([1.0], (0.1, 10.0), bkd, nvars=nvars)
            kernel = constant * matern
            base_kernels.append(kernel)

        # Create coregionalization matrices
        B1_np = np.array([[1.0, 0.5], [0.5, 1.0]])  # Positive correlation
        B2_np = np.array([[0.5, 0.0], [0.0, 0.5]])  # Independent component

        coreg_matrices = [
            bkd.array(B1_np),
            bkd.array(B2_np),
        ]

        # Create LMC kernel
        lmc_kernel = LinearCoregionalizationKernel(
            base_kernels, coreg_matrices, noutputs
        )

        # Create and fit GP (with fixed hyperparameters for interpolation test)
        gp = MultiOutputGP(lmc_kernel, nugget=1e-14)
        gp.hyp_list().set_all_inactive()  # Skip optimization for interpolation test
        X_list = [X_train] * noutputs
        result = MultiOutputGPMaximumLikelihoodFitter(bkd).fit(
            gp, X_list, y_train_stacked
        )
        gp = result.surrogate()

        # Predict at training points
        # predict() returns a list of arrays, each with shape (1, n_train)
        y_pred_list = gp.predict(X_list)

        # Check interpolation accuracy for each output
        for i in range(noutputs):
            start = i * n_train
            end = (i + 1) * n_train
            y_true_i = y_train_stacked[start:end, 0]  # (n_train,)
            y_pred_i = y_pred_list[i][0, :]  # (n_train,)

            error_i = bkd.abs(y_pred_i - y_true_i)
            max_error_i = bkd.max(error_i).item()

            assert max_error_i < 1e-6, (
                f"LMC output {i} error {max_error_i:.2e} exceeds 1e-6"
            )

    def test_lmc_captures_correlations(self, bkd) -> None:
        """Test that LMC kernel has non-zero off-diagonal blocks."""
        np.random.seed(42)
        nvars = 2
        noutputs = 2
        n_train = 20

        X_train_np = np.random.randn(nvars, n_train)
        X_train = bkd.array(X_train_np)

        # Create base kernel
        matern = Matern52Kernel([1.0] * nvars, (0.1, 10.0), nvars, bkd)

        # Coregionalization matrix with correlation
        B_np = np.array([[1.0, 0.7], [0.7, 1.0]])
        B = bkd.array(B_np)

        lmc_kernel = LinearCoregionalizationKernel([matern], [B], noutputs)

        # Get kernel blocks
        X_list = [X_train] * noutputs
        K_blocks = lmc_kernel(X_list, block_format=True)

        # Off-diagonal blocks should be non-zero
        K_01 = K_blocks[0][1]
        K_01_abs = bkd.abs(K_01)
        max_val = bkd.max(K_01_abs).item()

        assert max_val > 0.01, "LMC should have non-zero output correlations"


NVARS = 2
NOUTPUTS = 2
NLEVELS = 3


def _stacked_data(bkd: Backend[Array], n_train: int) -> Tuple[Array, Array]:
    np.random.seed(42)
    X_np = np.random.randn(NVARS, n_train)
    X = bkd.array(X_np)
    y1 = np.sin(X_np[0, :] + X_np[1, :])
    y2 = np.cos(X_np[0, :] - X_np[1, :])
    y_stacked = bkd.array(np.concatenate([y1, y2])[:, np.newaxis])
    return X, y_stacked


def _scaled_matern(bkd: Backend[Array], lenscale: float) -> Kernel[Array]:
    matern = Matern52Kernel(bkd.full((NVARS,), lenscale), (0.1, 10.0), NVARS, bkd)
    constant = PolynomialScalingKernel([1.0], (0.1, 10.0), bkd, nvars=NVARS)
    return constant * matern


def _independent_gp(bkd: Backend[Array], lenscale: float) -> MultiOutputGP[Array]:
    kernels = [_scaled_matern(bkd, lenscale) for _ in range(NOUTPUTS)]
    return MultiOutputGP(IndependentMultiOutputKernel(kernels), nugget=1e-6)


def _lmc_gp(bkd: Backend[Array], lenscale: float) -> MultiOutputGP[Array]:
    base_kernels = [_scaled_matern(bkd, lenscale) for _ in range(2)]
    B1 = bkd.array(np.array([[1.0, 0.5], [0.5, 1.0]]))
    B2 = bkd.array(np.array([[0.5, 0.0], [0.0, 0.5]]))
    lmc_kernel = LinearCoregionalizationKernel(base_kernels, [B1, B2], NOUTPUTS)
    return MultiOutputGP(lmc_kernel, nugget=1e-6)


def _assert_loss_gradient_accurate(
    bkd: Backend[Array],
    loss: GPNegativeLogMarginalLikelihoodLoss[Array],
    params: Array,
    tol: float,
) -> None:
    """The loss bundle's jacobian (analytical, or autograd on an autodiff
    backend) converges against finite differences."""
    assert loss.derivatives().jacobian is not None, "loss declares no jacobian"
    checker = DerivativeChecker(loss)
    fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
    errors = checker.check_derivatives(
        params[:, None],
        fd_eps=fd_eps,
        relative=True,
        verbosity=0,
    )
    grad_error = errors[0]
    assert bkd.to_bool(bkd.all(bkd.isfinite(grad_error))), (
        "Gradient errors contain non-finite values"
    )
    min_error = float(bkd.to_numpy(bkd.min(grad_error)))
    assert min_error < tol, f"Min gradient error {min_error} exceeds threshold"
    error_ratio = float(bkd.to_numpy(checker.error_ratio(grad_error)))
    assert error_ratio < tol, f"Error ratio {error_ratio:.2e} suggests poor convergence"


class TestMultiOutputGPOptimization:
    """Multi-output GP hyperparameter optimization, with gradients from an
    autodiff backend."""

    def test_independent_kernel_optimization(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Test optimization with IndependentMultiOutputKernel."""
        bkd = autodiff_bkd
        X_train, y_stacked = _stacked_data(bkd, 30)
        gp = _independent_gp(bkd, 0.3)

        initial_params = bkd.copy(gp.hyp_list().get_active_values())

        X_list = [X_train] * NOUTPUTS
        result = MultiOutputGPMaximumLikelihoodFitter(bkd).fit(gp, X_list, y_stacked)
        gp = result.surrogate()

        final_params = gp.hyp_list().get_active_values()
        assert not bkd.allclose(initial_params, final_params), (
            "Hyperparameters should change during optimization"
        )

    @slow_test
    def test_lmc_kernel_optimization(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Test optimization with LinearCoregionalizationKernel."""
        bkd = autodiff_bkd
        X_train, y_stacked = _stacked_data(bkd, 30)
        gp = _lmc_gp(bkd, 0.3)

        initial_params = bkd.copy(gp.hyp_list().get_active_values())

        X_list = [X_train] * NOUTPUTS
        result = MultiOutputGPMaximumLikelihoodFitter(bkd).fit(gp, X_list, y_stacked)
        gp = result.surrogate()

        final_params = gp.hyp_list().get_active_values()
        assert not bkd.allclose(initial_params, final_params), (
            "Hyperparameters should change during optimization"
        )

    def test_independent_kernel_loss_gradient_accuracy(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Test loss gradient accuracy with DerivativeChecker for independent kernel."""
        bkd = autodiff_bkd
        X_train, y_stacked = _stacked_data(bkd, 20)
        gp = _independent_gp(bkd, 1.0)
        X_list = [X_train] * NOUTPUTS
        gp._fit_internal(X_list, y_stacked)

        loss = GPNegativeLogMarginalLikelihoodLoss(gp, (X_list, y_stacked))
        _assert_loss_gradient_accurate(
            bkd, loss, gp.hyp_list().get_active_values(), 2e-6
        )

    def test_lmc_kernel_loss_gradient_accuracy(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Test loss gradient accuracy with DerivativeChecker for LMC kernel."""
        bkd = autodiff_bkd
        X_train, y_stacked = _stacked_data(bkd, 20)
        gp = _lmc_gp(bkd, 1.0)
        X_list = [X_train] * NOUTPUTS
        gp._fit_internal(X_list, y_stacked)

        loss = GPNegativeLogMarginalLikelihoodLoss(gp, (X_list, y_stacked))
        _assert_loss_gradient_accurate(
            bkd, loss, gp.hyp_list().get_active_values(), 1e-6
        )


def _multilevel_data(bkd: Backend[Array]) -> Tuple[List[Array], List[Array]]:
    """Multi-level data with different X arrays per level."""
    np.random.seed(42)
    # Different number of samples per level (more at cheap levels)
    n_per_level = [25, 15, 8]
    X_list = []
    y_list = []
    for k, n_k in enumerate(n_per_level):
        X_np = np.random.randn(NVARS, n_k)
        # Simulate multi-fidelity: each level is a noisy version of sin
        bias = 0.1 * (NLEVELS - 1 - k)
        y_np = np.sin(X_np[0, :] + X_np[1, :]) + bias
        X_list.append(bkd.array(X_np))
        y_list.append(bkd.array(y_np.reshape(1, -1)))
    return X_list, y_list


def _multilevel_kernel(bkd: Backend[Array]) -> MultiLevelKernel[Array]:
    """A MultiLevelKernel with SE kernels and polynomial scalings."""
    level_kernels = [
        SquaredExponentialKernel(bkd.ones((1,)), (0.05, 5.0), NVARS, bkd)
        for _ in range(NLEVELS)
    ]
    scalings = [
        PolynomialScalingFunction(
            [1.0],
            (-3.0, 3.0),
            bkd,
            nvars=NVARS,
            fixed=False,
        )
        for _ in range(NLEVELS - 1)
    ]
    return MultiLevelKernel(level_kernels, scalings)


class TestMultiOutputGPWithMultiLevelKernelAutodiff:
    """MultiOutputGP with MultiLevelKernel on an autodiff backend.

    Tests autograd-based NLL gradients with different X arrays per level,
    the key scenario where analytical kernel jacobian_wrt_params fails.
    """

    def test_loss_gradient_accuracy_different_X_per_level(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Verify autograd NLL gradients match finite differences."""
        bkd = autodiff_bkd
        X_list, y_list = _multilevel_data(bkd)
        gp = MultiOutputGP(_multilevel_kernel(bkd), nugget=1e-6)
        gp._fit_internal(X_list, y_list)

        loss = GPNegativeLogMarginalLikelihoodLoss(gp, (X_list, y_list))
        _assert_loss_gradient_accurate(
            bkd, loss, gp.hyp_list().get_active_values(), 3e-6
        )

    def test_fit_with_different_X_per_level(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """End-to-end fit() with different X per level optimizes params."""
        bkd = autodiff_bkd
        X_list, y_list = _multilevel_data(bkd)
        gp = MultiOutputGP(_multilevel_kernel(bkd), nugget=1e-6)

        initial_params = bkd.copy(gp.hyp_list().get_active_values())

        result = MultiOutputGPMaximumLikelihoodFitter(bkd).fit(gp, X_list, y_list)
        gp = result.surrogate()

        final_params = gp.hyp_list().get_active_values()
        assert not bkd.allclose(initial_params, final_params), (
            "Hyperparameters should change during optimization"
        )
