"""End-to-end integration tests for kernel operator learning.

Tests actual operator recovery on held-out data, convergence properties,
and NLL gradient correctness — not just training-point interpolation.
"""

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.gaussianprocess.gp_loss import (
    GPNegativeLogMarginalLikelihoodLoss,
)
from pyapprox.surrogates.kerneloperator.encoders.identity import (
    IdentityFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.encoders.pca import (
    PCAFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.fitters.maximum_likelihood_fitter import (
    KernelOperatorMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kerneloperator.regressors.scalar_kernel import (
    ScalarKernelLatentRegressor,
)
from pyapprox.surrogates.kernels.linear import LinearKernel
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.surrogates.kernels.multioutput.independent import (
    IndependentMultiOutputKernel,
)


def _identity_factory(ngrid):
    def factory(data, bkd):
        return IdentityFunctionEncoder(ngrid, bkd)
    return factory


def _pca_factory(ncodes):
    def factory(data, bkd):
        return PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=ncodes)
    return factory


def _sample_smooth_grf(rng, ngrid, N, length_scale=0.1):
    """Sample smooth functions from a Gaussian random field."""
    grid = np.linspace(0, 1, ngrid)
    K = np.exp(-(grid[:, None] - grid[None, :]) ** 2 / (2 * length_scale ** 2))
    L = np.linalg.cholesky(K + 1e-10 * np.eye(ngrid))
    return L @ rng.standard_normal((ngrid, N))


def _integration_operator(u, ngrid):
    """Cumulative trapezoid integration on [0,1]."""
    dx = 1.0 / (ngrid - 1)
    return np.cumsum(u, axis=0) * dx


def _mean_relative_l2_error(pred, true):
    """Mean per-sample relative L2 error."""
    num = np.linalg.norm(pred - true, axis=0)
    den = np.linalg.norm(true, axis=0)
    mask = den > 1e-12
    return (num[mask] / den[mask]).mean()


class TestRecoverIntegrationOperator:
    """Recover a known linear operator to bounded held-out error."""

    def test_pca_matern_held_out_error(self, bkd) -> None:
        rng = np.random.default_rng(0)
        ngrid = 32
        N_train = 150
        N_test = 50
        ncodes = 10

        u_train_np = _sample_smooth_grf(rng, ngrid, N_train)
        v_train_np = _integration_operator(u_train_np, ngrid)
        u_test_np = _sample_smooth_grf(rng, ngrid, N_test)
        v_test_np = _integration_operator(u_test_np, ngrid)

        kernel = Matern52Kernel(
            [1.0] * ncodes, (0.01, 100.0), ncodes, bkd
        )
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_pca_factory(ncodes)],
            [_pca_factory(ncodes)],
            kernel,
            nugget=1e-8,
        )
        result = fitter.fit(
            [bkd.array(u_train_np)], [bkd.array(v_train_np)]
        )
        pred = bkd.to_numpy(
            result.surrogate().predict([bkd.array(u_test_np)])[0]
        )
        rel_err = _mean_relative_l2_error(pred, v_test_np)
        assert rel_err < 0.025, f"Mean relative L2 error {rel_err:.6f} too large"

    def test_identity_encoder_held_out_error(self, bkd) -> None:
        rng = np.random.default_rng(1)
        ngrid = 16
        N_train = 80
        N_test = 20

        u_train_np = _sample_smooth_grf(rng, ngrid, N_train, length_scale=0.2)
        v_train_np = _integration_operator(u_train_np, ngrid)
        u_test_np = _sample_smooth_grf(rng, ngrid, N_test, length_scale=0.2)
        v_test_np = _integration_operator(u_test_np, ngrid)

        kernel = Matern52Kernel(
            [1.0] * ngrid, (0.01, 100.0), ngrid, bkd
        )
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(ngrid)],
            [_identity_factory(ngrid)],
            kernel,
            nugget=1e-8,
        )
        result = fitter.fit(
            [bkd.array(u_train_np)], [bkd.array(v_train_np)]
        )
        pred = bkd.to_numpy(
            result.surrogate().predict([bkd.array(u_test_np)])[0]
        )
        rel_err = _mean_relative_l2_error(pred, v_test_np)
        assert rel_err < 0.012, f"Mean relative L2 error {rel_err:.6f} too large"


class TestLinearOperatorWithLinearKernel:
    """Linear kernel + linear operator -> near-machine-precision recovery."""

    def test_exact_recovery(self, bkd) -> None:
        rng = np.random.default_rng(0)
        ngrid = 12
        N_train = ngrid + 5
        N_test = 20

        A = rng.standard_normal((ngrid, ngrid))
        u_train_np = rng.standard_normal((ngrid, N_train))
        v_train_np = A @ u_train_np
        u_test_np = rng.standard_normal((ngrid, N_test))
        v_test_np = A @ u_test_np

        kernel = LinearKernel(1.0, (0.01, 100.0), ngrid, bkd, fixed=True)
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(ngrid)],
            [_identity_factory(ngrid)],
            kernel,
            nugget=1e-10,
        )
        result = fitter.fit(
            [bkd.array(u_train_np)], [bkd.array(v_train_np)]
        )
        pred = bkd.to_numpy(
            result.surrogate().predict([bkd.array(u_test_np)])[0]
        )
        bkd.assert_allclose(
            bkd.array(pred), bkd.array(v_test_np), atol=1e-8
        )


class TestConvergenceWithN:
    """Test error decreases as training set size grows."""

    def test_error_nonincreasing_with_N(self, bkd) -> None:
        ngrid = 16
        ncodes = 6
        N_test = 30
        N_values = [30, 80, 200]

        rng_test = np.random.default_rng(999)
        u_test_np = _sample_smooth_grf(rng_test, ngrid, N_test)
        v_test_np = _integration_operator(u_test_np, ngrid)

        errors = []
        for N_train in N_values:
            rng = np.random.default_rng(42)
            u_train_np = _sample_smooth_grf(rng, ngrid, N_train)
            v_train_np = _integration_operator(u_train_np, ngrid)

            kernel = Matern52Kernel(
                [1.0] * ncodes, (0.1, 10.0), ncodes, bkd
            )
            kernel.hyp_list().set_all_inactive()
            fitter = KernelOperatorMaximumLikelihoodFitter(
                bkd,
                [_pca_factory(ncodes)],
                [_pca_factory(ncodes)],
                kernel,
                nugget=1e-8,
            )
            result = fitter.fit(
                [bkd.array(u_train_np)], [bkd.array(v_train_np)]
            )
            pred = bkd.to_numpy(
                result.surrogate().predict([bkd.array(u_test_np)])[0]
            )
            errors.append(_mean_relative_l2_error(pred, v_test_np))

        for ii in range(len(errors) - 1):
            assert errors[ii + 1] <= errors[ii], (
                f"Error did not decrease: N={N_values[ii]} err={errors[ii]:.4f}"
                f" -> N={N_values[ii+1]} err={errors[ii+1]:.4f}"
            )


class TestConvergenceWithNCodes:
    """Test error decreases as PCA truncation increases.

    Uses ML optimization so length scales adapt to each code dimension.
    """

    def test_error_nonincreasing_with_ncodes(self, bkd) -> None:
        rng = np.random.default_rng(42)
        ngrid = 32
        N_train = 150
        N_test = 30
        ncodes_values = [3, 6, 12]

        u_train_np = _sample_smooth_grf(rng, ngrid, N_train)
        v_train_np = _integration_operator(u_train_np, ngrid)
        rng_test = np.random.default_rng(999)
        u_test_np = _sample_smooth_grf(rng_test, ngrid, N_test)
        v_test_np = _integration_operator(u_test_np, ngrid)

        errors = []
        for ncodes in ncodes_values:
            kernel = Matern52Kernel(
                [1.0] * ncodes, (0.01, 100.0), ncodes, bkd
            )
            fitter = KernelOperatorMaximumLikelihoodFitter(
                bkd,
                [_pca_factory(ncodes)],
                [_pca_factory(ncodes)],
                kernel,
                nugget=1e-8,
            )
            result = fitter.fit(
                [bkd.array(u_train_np)], [bkd.array(v_train_np)]
            )
            pred = bkd.to_numpy(
                result.surrogate().predict([bkd.array(u_test_np)])[0]
            )
            errors.append(_mean_relative_l2_error(pred, v_test_np))

        for ii in range(len(errors) - 1):
            assert errors[ii + 1] <= errors[ii], (
                f"Error did not decrease: ncodes={ncodes_values[ii]} "
                f"err={errors[ii]:.4f} -> ncodes={ncodes_values[ii+1]} "
                f"err={errors[ii+1]:.4f}"
            )


class TestMultiInputOperator:
    """Two-input operator where both inputs contribute to the output."""

    def test_recovers_additive_operator(self, bkd) -> None:
        """Linear additive operator with linear kernel for exact recovery."""
        rng = np.random.default_rng(0)
        ngrid = 6
        N_train = 40
        N_test = 10
        ncodes_in = ngrid * 2

        u1_train = rng.standard_normal((ngrid, N_train)) * 0.5
        u2_train = rng.standard_normal((ngrid, N_train)) * 0.5
        v_train = u1_train + u2_train

        u1_test = rng.standard_normal((ngrid, N_test)) * 0.5
        u2_test = rng.standard_normal((ngrid, N_test)) * 0.5
        v_test = u1_test + u2_test

        kernel = LinearKernel(1.0, (0.01, 100.0), ncodes_in, bkd, fixed=True)
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(ngrid), _identity_factory(ngrid)],
            [_identity_factory(ngrid)],
            kernel,
            nugget=1e-10,
        )
        result = fitter.fit(
            [bkd.array(u1_train), bkd.array(u2_train)],
            [bkd.array(v_train)],
        )
        pred = bkd.to_numpy(result.surrogate().predict(
            [bkd.array(u1_test), bkd.array(u2_test)]
        )[0])
        rel_err = _mean_relative_l2_error(pred, v_test)
        assert rel_err < 1e-6, f"Rel L2 error {rel_err:.2e} too large"

    def test_predictions_depend_on_both_inputs(self, bkd) -> None:
        """Verify changing u2 changes predictions (input not ignored)."""
        rng = np.random.default_rng(0)
        ngrid = 6
        N_train = 40
        ncodes_in = ngrid * 2

        u1_train = rng.standard_normal((ngrid, N_train)) * 0.5
        u2_train = rng.standard_normal((ngrid, N_train)) * 0.5
        v_train = u1_train + u2_train

        kernel = LinearKernel(1.0, (0.01, 100.0), ncodes_in, bkd, fixed=True)
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(ngrid), _identity_factory(ngrid)],
            [_identity_factory(ngrid)],
            kernel,
            nugget=1e-10,
        )
        result = fitter.fit(
            [bkd.array(u1_train), bkd.array(u2_train)],
            [bkd.array(v_train)],
        )
        surr = result.surrogate()

        N_probe = 5
        u1_fixed = bkd.array(rng.standard_normal((ngrid, N_probe)) * 0.5)
        u2_a = bkd.array(rng.standard_normal((ngrid, N_probe)) * 0.5)
        u2_b = bkd.array(rng.standard_normal((ngrid, N_probe)) * 0.5)
        pred_a = surr.predict([u1_fixed, u2_a])[0]
        pred_b = surr.predict([u1_fixed, u2_b])[0]
        diff = float(bkd.to_numpy(bkd.sum((pred_a - pred_b) ** 2)))
        assert diff > 0.01, "Predictions don't depend on second input"


class TestNLLGradientAccuracy:
    """NLL gradient matches finite differences via DerivativeChecker."""

    def test_scalar_kernel_nll_gradient(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        rng = np.random.default_rng(0)
        ncodes = 5
        N = 15

        U = bkd.array(rng.standard_normal((ncodes, N)))
        V = bkd.array(rng.standard_normal((ncodes, N)))

        kernel = Matern52Kernel(
            [1.0] * ncodes, (0.1, 10.0), ncodes, bkd
        )
        reg = ScalarKernelLatentRegressor(
            kernel, ncodes, ncodes, bkd, nugget=1e-8
        )
        reg.fit_internal(U, V)

        loss = GPNegativeLogMarginalLikelihoodLoss(reg._gp, (U, V))
        assert hasattr(loss, "jacobian"), (
            "Loss should have jacobian (Matern52 has jacobian_wrt_params)"
        )

        checker = DerivativeChecker(loss)
        params = reg.hyp_list().get_active_values()
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True, verbosity=0
        )

        error_ratio = float(checker.error_ratio(errors[0]))
        assert error_ratio < 1e-6, (
            f"NLL gradient error ratio {error_ratio:.2e} too large"
        )


class TestPredictStdCalibration:
    """Calibration check: most errors within predicted uncertainty.

    Uses identity encoder to avoid PCA covariance approximation issues,
    testing the scalar-kernel path where code-space std is exact.
    """

    def test_coverage_on_held_out_identity_encoder(self, bkd) -> None:
        rng = np.random.default_rng(0)
        ngrid = 10
        N_train = 60
        N_test = 30

        u_train_np = _sample_smooth_grf(rng, ngrid, N_train, length_scale=0.2)
        v_train_np = _integration_operator(u_train_np, ngrid)
        u_test_np = _sample_smooth_grf(rng, ngrid, N_test, length_scale=0.2)
        v_test_np = _integration_operator(u_test_np, ngrid)

        kernel = Matern52Kernel(
            [1.0] * ngrid, (0.01, 100.0), ngrid, bkd
        )
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(ngrid)],
            [_identity_factory(ngrid)],
            kernel,
            nugget=1e-8,
        )
        result = fitter.fit(
            [bkd.array(u_train_np)], [bkd.array(v_train_np)]
        )
        surr = result.surrogate()

        pred = bkd.to_numpy(surr.predict([bkd.array(u_test_np)])[0])
        std = bkd.to_numpy(surr.predict_std([bkd.array(u_test_np)])[0])

        abs_err = np.abs(pred - v_test_np)
        within_1std = (abs_err < 1.0 * std).mean()
        within_2std = (abs_err < 2.0 * std).mean()
        assert within_1std > 0.60, (
            f"Only {within_1std:.1%} within 1 std (expected > 60%)"
        )
        assert within_2std > 0.90, (
            f"Only {within_2std:.1%} within 2 std (expected > 90%)"
        )

    def test_pca_std_nonnegative(self, bkd) -> None:
        """PCA path should produce non-negative stds after variance fix."""
        rng = np.random.default_rng(0)
        ngrid = 20
        N_train = 60
        N_test = 15
        ncodes = 8

        u_train_np = _sample_smooth_grf(rng, ngrid, N_train)
        v_train_np = _integration_operator(u_train_np, ngrid)
        u_test_np = _sample_smooth_grf(rng, ngrid, N_test)

        kernel = Matern52Kernel(
            [1.0] * ncodes, (0.01, 100.0), ncodes, bkd
        )
        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_pca_factory(ncodes)],
            [_pca_factory(ncodes)],
            kernel,
            nugget=1e-8,
        )
        result = fitter.fit(
            [bkd.array(u_train_np)], [bkd.array(v_train_np)]
        )
        std = result.surrogate().predict_std([bkd.array(u_test_np)])[0]
        assert bkd.all_bool(std >= 0)


class TestMultiOutputWithIndependentKernel:
    """Multi-output path with real operator structure."""

    def test_independent_kernel_recovers_operator(self, bkd) -> None:
        rng = np.random.default_rng(0)
        ngrid = 8
        N_train = 80
        N_test = 20

        u_train_np = _sample_smooth_grf(rng, ngrid, N_train, length_scale=0.2)
        v_train_np = _integration_operator(u_train_np, ngrid)

        u_test_np = _sample_smooth_grf(rng, ngrid, N_test, length_scale=0.2)
        v_test_np = _integration_operator(u_test_np, ngrid)

        kernels = [
            Matern52Kernel([1.0] * ngrid, (0.1, 10.0), ngrid, bkd)
            for _ in range(ngrid)
        ]
        mo_kernel = IndependentMultiOutputKernel(kernels)

        fitter = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_identity_factory(ngrid)],
            [_identity_factory(ngrid)],
            mo_kernel,
            nugget=1e-8,
        )
        result = fitter.fit(
            [bkd.array(u_train_np)], [bkd.array(v_train_np)]
        )
        pred = bkd.to_numpy(
            result.surrogate().predict([bkd.array(u_test_np)])[0]
        )
        rel_err = _mean_relative_l2_error(pred, v_test_np)
        assert rel_err < 0.015, (
            f"IndependentKernel rel L2 error {rel_err:.6f} too large"
        )


class TestMLOptimizationImprovesHeldOutError:
    """ML optimization should improve held-out prediction, not just NLL."""

    def test_optimized_beats_fixed_on_held_out(self, bkd) -> None:
        rng = np.random.default_rng(42)
        ngrid = 20
        N_train = 60
        N_test = 20
        ncodes = 5

        u_train_np = _sample_smooth_grf(rng, ngrid, N_train)
        v_train_np = _integration_operator(u_train_np, ngrid)
        rng_test = np.random.default_rng(999)
        u_test_np = _sample_smooth_grf(rng_test, ngrid, N_test)
        v_test_np = _integration_operator(u_test_np, ngrid)

        kernel_fixed = Matern52Kernel(
            [1.0] * ncodes, (0.01, 100.0), ncodes, bkd
        )
        kernel_fixed.hyp_list().set_all_inactive()
        fitter_fixed = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_pca_factory(ncodes)],
            [_pca_factory(ncodes)],
            kernel_fixed,
            nugget=1e-8,
        )
        result_fixed = fitter_fixed.fit(
            [bkd.array(u_train_np)], [bkd.array(v_train_np)]
        )
        pred_fixed = bkd.to_numpy(
            result_fixed.surrogate().predict([bkd.array(u_test_np)])[0]
        )
        err_fixed = _mean_relative_l2_error(pred_fixed, v_test_np)

        kernel_opt = Matern52Kernel(
            [1.0] * ncodes, (0.01, 100.0), ncodes, bkd
        )
        fitter_opt = KernelOperatorMaximumLikelihoodFitter(
            bkd,
            [_pca_factory(ncodes)],
            [_pca_factory(ncodes)],
            kernel_opt,
            nugget=1e-8,
        )
        result_opt = fitter_opt.fit(
            [bkd.array(u_train_np)], [bkd.array(v_train_np)]
        )
        pred_opt = bkd.to_numpy(
            result_opt.surrogate().predict([bkd.array(u_test_np)])[0]
        )
        err_opt = _mean_relative_l2_error(pred_opt, v_test_np)

        assert err_opt < err_fixed * 0.5, (
            f"Optimized error {err_opt:.4f} not substantially better than "
            f"fixed {err_fixed:.4f}"
        )
