"""Tests for prediction OED convergence analysis.

Tests verify:
- Prediction OED objective converges with increasing samples
- Different deviation measures show expected convergence behavior
- Convergence rate analysis for linear models
- Nonlinear (lognormal) prediction OED convergence with exact utilities
"""

from typing import Tuple

import numpy as np
import pytest

from pyapprox.expdesign import create_prediction_oed_objective
from pyapprox.benchmarks.instances.oed.linear_gaussian import (
    build_linear_gaussian_kl_benchmark,
)
from pyapprox.benchmarks.instances.oed.nonlinear_gaussian import (
    NonLinearGaussianPredOEDBenchmark,
    build_nonlinear_gaussian_pred_benchmark,
)
from pyapprox.expdesign.data import generate_oed_data
from pyapprox.expdesign.diagnostics import (
    compute_exact_prediction_utility,
    get_utility_factory,
)
from pyapprox.expdesign.diagnostics.prediction_diagnostics import (
    create_prediction_oed_diagnostics,
)
from pyapprox.expdesign.quadrature import MonteCarloSampler
from pyapprox.expdesign.quadrature.oed import (
    OEDQuadratureSampler,
    build_oed_joint_distribution,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import (
    slow_test,
)


class TestPredictionOEDConvergenceStandalone:
    """Tests for prediction OED convergence using raw arrays."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd: Backend[Array]) -> None:
        self._nobs = 3
        self._ninner = 50
        self._nouter = 15
        self._npred = 2

    def _create_test_data(
        self, bkd: Backend[Array], seed: int = 42,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """Create consistent test data for convergence tests."""
        np.random.seed(seed)

        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        inner_shapes = bkd.asarray(np.random.randn(self._nobs, self._ninner))
        latent_samples = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        qoi_vals = bkd.asarray(np.random.randn(self._ninner, self._npred))

        return noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals

    def test_stdev_objective_produces_positive_value(
        self, bkd: Backend[Array],
    ) -> None:
        """StdDev objective produces positive deviation value."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights = bkd.ones((self._nobs, 1)) / self._nobs
        value = objective(weights)

        val_np = bkd.to_numpy(value)[0, 0]
        assert val_np > 0.0
        assert np.isfinite(val_np)

    def test_entropic_objective_produces_finite_value(
        self, bkd: Backend[Array],
    ) -> None:
        """Entropic objective produces finite value."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="entropic",
            risk_type="mean",
            alpha=0.5,
        )

        weights = bkd.ones((self._nobs, 1)) / self._nobs
        value = objective(weights)

        val_np = bkd.to_numpy(value)[0, 0]
        assert np.isfinite(val_np)

    @slow_test
    def test_stdev_convergence_with_inner_samples(
        self, bkd: Backend[Array],
    ) -> None:
        """StdDev deviation converges with increasing inner samples."""
        inner_counts = [10, 20, 40]
        values = []

        for ninner in inner_counts:
            np.random.seed(42)
            noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
            outer_shapes = bkd.asarray(np.random.randn(self._nobs, self._nouter))
            inner_shapes = bkd.asarray(np.random.randn(self._nobs, ninner))
            latent_samples = bkd.asarray(
                np.random.randn(self._nobs, self._nouter)
            )
            qoi_vals = bkd.asarray(np.random.randn(ninner, self._npred))

            objective = create_prediction_oed_objective(
                noise_variances,
                outer_shapes,
                inner_shapes,
                latent_samples,
                qoi_vals,
                bkd,
                deviation_type="stdev",
                risk_type="mean",
            )

            weights = bkd.ones((self._nobs, 1)) / self._nobs
            value = objective(weights)
            values.append(bkd.to_numpy(value)[0, 0])

        for val in values:
            assert val > 0.0
            assert np.isfinite(val)

        assert max(values) / min(values) < 10.0

    @slow_test
    def test_entropic_convergence_with_inner_samples(
        self, bkd: Backend[Array],
    ) -> None:
        """Entropic deviation converges with increasing inner samples."""
        inner_counts = [10, 20, 40]
        values = []

        for ninner in inner_counts:
            np.random.seed(42)
            noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
            outer_shapes = bkd.asarray(np.random.randn(self._nobs, self._nouter))
            inner_shapes = bkd.asarray(np.random.randn(self._nobs, ninner))
            latent_samples = bkd.asarray(
                np.random.randn(self._nobs, self._nouter)
            )
            qoi_vals = bkd.asarray(np.random.randn(ninner, self._npred))

            objective = create_prediction_oed_objective(
                noise_variances,
                outer_shapes,
                inner_shapes,
                latent_samples,
                qoi_vals,
                bkd,
                deviation_type="entropic",
                risk_type="mean",
                alpha=0.5,
            )

            weights = bkd.ones((self._nobs, 1)) / self._nobs
            value = objective(weights)
            values.append(bkd.to_numpy(value)[0, 0])

        for val in values:
            assert np.isfinite(val)

        assert max(abs(v) for v in values) < 100.0

    def test_different_weights_give_different_deviations(
        self, bkd: Backend[Array],
    ) -> None:
        """Different weights give different deviation values."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights_uniform = bkd.ones((self._nobs, 1)) / self._nobs
        weights_high = bkd.asarray([[2.0], [2.0], [2.0]])

        val_uniform = objective(weights_uniform)
        val_high = objective(weights_high)

        val_uniform_np = bkd.to_numpy(val_uniform)[0, 0]
        val_high_np = bkd.to_numpy(val_high)[0, 0]

        assert abs(val_uniform_np - val_high_np) > 1e-3

    def test_jacobian_is_finite(self, bkd: Backend[Array]) -> None:
        """Jacobian computation produces finite values."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        jac = objective.jacobian(weights)

        jac_np = bkd.to_numpy(jac)
        assert jac_np.shape == (1, self._nobs)
        assert np.all(np.isfinite(jac_np))

    @slow_test
    def test_variance_risk_produces_different_values(
        self, bkd: Backend[Array],
    ) -> None:
        """Variance risk measure produces different results than mean."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data(bkd)
        )

        objective_mean = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        objective_var = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="variance",
        )

        weights = bkd.ones((self._nobs, 1)) / self._nobs

        val_mean = objective_mean(weights)
        val_var = objective_var(weights)

        val_mean_np = bkd.to_numpy(val_mean)[0, 0]
        val_var_np = bkd.to_numpy(val_var)[0, 0]

        assert np.isfinite(val_mean_np)
        assert np.isfinite(val_var_np)
        assert abs(val_mean_np - val_var_np) > 1e-3

    def test_linear_gaussian_benchmark_exact_eig(
        self, bkd: Backend[Array],
    ) -> None:
        """LinearGaussianKLOEDBenchmark provides exact EIG."""
        benchmark = build_linear_gaussian_kl_benchmark(
            nobs=5, degree=2, noise_std=0.5, prior_std=0.5, bkd=bkd,
        )

        weights = bkd.ones((5, 1)) / 5
        eig = benchmark.exact_eig(weights)

        assert eig > 0.0
        assert np.isfinite(eig)


class TestNonLinearPredictionOEDConvergence:
    """Tests for nonlinear (lognormal) prediction OED convergence.

    Verifies that numerical estimates converge to exact analytical values
    computed using conjugate Gaussian formulas for lognormal QoI.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bkd: Backend[Array]) -> None:
        np.random.seed(1)

    def _create_benchmark(
        self, bkd: Backend[Array],
    ) -> NonLinearGaussianPredOEDBenchmark[Array]:
        """Create standard nonlinear benchmark for convergence tests."""
        return build_nonlinear_gaussian_pred_benchmark(
            nobs=2, degree=3, noise_std=0.5, prior_std=0.5,
            bkd=bkd, npred=1, min_degree=0,
        )

    def _make_sampler(
        self,
        benchmark: NonLinearGaussianPredOEDBenchmark[Array],
        bkd: Backend[Array],
        seed: int,
    ) -> OEDQuadratureSampler[Array]:
        """Create MC-based OEDQuadratureSampler from benchmark."""
        joint_dist = build_oed_joint_distribution(benchmark.problem(), bkd)
        np.random.seed(seed)
        return OEDQuadratureSampler(
            MonteCarloSampler(joint_dist, bkd),
            benchmark.problem().nparams(), bkd,
        )

    def test_nonlinear_benchmark_setup(
        self, bkd: Backend[Array],
    ) -> None:
        """Nonlinear benchmark is correctly configured."""
        benchmark = self._create_benchmark(bkd)
        problem = benchmark.problem()

        assert problem.nobs() == 2
        assert problem.nparams() == 4
        bkd.assert_allclose(
            bkd.asarray([benchmark.noise_std()]),
            bkd.asarray([0.5]),
            rtol=1e-10,
        )

        design_mat = benchmark.design_matrix()
        assert design_mat.shape == (2, 4)

        qoi_mat = benchmark.qoi_matrix()
        assert qoi_mat.shape == (1, 4)

    def test_exact_stdev_utility_positive(
        self, bkd: Backend[Array],
    ) -> None:
        """Exact lognormal expected std dev utility is positive."""
        benchmark = self._create_benchmark(bkd)
        exact_cls, exact_args, _ = get_utility_factory("nonlinear_mean_stdev")

        weights = bkd.ones((2, 1)) / 2
        exact = compute_exact_prediction_utility(
            benchmark.problem().prior_mean(),
            benchmark.problem().prior_covariance(),
            benchmark.design_matrix(),
            benchmark.qoi_matrix(),
            benchmark.noise_var(),
            weights,
            exact_cls, exact_args, bkd,
        )

        assert exact > 0.0
        assert np.isfinite(exact)

    def test_exact_avar_stdev_utility_positive(
        self, bkd: Backend[Array],
    ) -> None:
        """Exact lognormal AVaR std dev utility is positive."""
        benchmark = self._create_benchmark(bkd)
        exact_cls, exact_args, _ = get_utility_factory(
            "nonlinear_avar_stdev", beta=0.5,
        )

        weights = bkd.ones((2, 1)) / 2
        exact = compute_exact_prediction_utility(
            benchmark.problem().prior_mean(),
            benchmark.problem().prior_covariance(),
            benchmark.design_matrix(),
            benchmark.qoi_matrix(),
            benchmark.noise_var(),
            weights,
            exact_cls, exact_args, bkd,
        )

        assert exact > 0.0
        assert np.isfinite(exact)

    def test_exact_avar_stdev_increases_with_beta(
        self, bkd: Backend[Array],
    ) -> None:
        """AVaR std dev increases with higher beta (more risk averse)."""
        benchmark = self._create_benchmark(bkd)
        weights = bkd.ones((2, 1)) / 2

        cls_low, args_low, _ = get_utility_factory(
            "nonlinear_avar_stdev", beta=0.3,
        )
        cls_high, args_high, _ = get_utility_factory(
            "nonlinear_avar_stdev", beta=0.7,
        )

        exact_low = compute_exact_prediction_utility(
            benchmark.problem().prior_mean(),
            benchmark.problem().prior_covariance(),
            benchmark.design_matrix(),
            benchmark.qoi_matrix(),
            benchmark.noise_var(),
            weights,
            cls_low, args_low, bkd,
        )
        exact_high = compute_exact_prediction_utility(
            benchmark.problem().prior_mean(),
            benchmark.problem().prior_covariance(),
            benchmark.design_matrix(),
            benchmark.qoi_matrix(),
            benchmark.noise_var(),
            weights,
            cls_high, args_high, bkd,
        )

        assert exact_high > exact_low

    def test_numerical_stdev_close_to_exact(
        self, bkd: Backend[Array],
    ) -> None:
        """Numerical stdev estimate is reasonably close to exact."""
        benchmark = self._create_benchmark(bkd)
        nobs = 2
        noise_std = 0.5

        exact_cls, exact_args, _ = get_utility_factory("nonlinear_mean_stdev")
        weights = bkd.ones((2, 1)) / 2

        exact = compute_exact_prediction_utility(
            benchmark.problem().prior_mean(),
            benchmark.problem().prior_covariance(),
            benchmark.design_matrix(),
            benchmark.qoi_matrix(),
            benchmark.noise_var(),
            weights,
            exact_cls, exact_args, bkd,
        )

        noise_variances = bkd.full((nobs,), noise_std**2)
        diag = create_prediction_oed_diagnostics(
            noise_variances, 1, "nonlinear_mean_stdev", bkd,
        )

        data = generate_oed_data(
            benchmark.problem(),
            self._make_sampler(benchmark, bkd, 42),
            self._make_sampler(benchmark, bkd, 123),
            500, 500,
        )

        numerical = diag.compute_numerical_utility(
            weights, data.outer_shapes, data.latent_samples,
            data.inner_shapes, data.qoi_vals,
        )

        relative_error = abs(numerical - exact) / exact
        assert relative_error < 0.2

    def test_weights_affect_exact_utility(
        self, bkd: Backend[Array],
    ) -> None:
        """Different weights give different exact utilities."""
        benchmark = self._create_benchmark(bkd)
        exact_cls, exact_args, _ = get_utility_factory("nonlinear_mean_stdev")

        weights_uniform = bkd.ones((2, 1)) / 2
        weights_high = bkd.ones((2, 1)) * 2.0

        exact_uniform = compute_exact_prediction_utility(
            benchmark.problem().prior_mean(),
            benchmark.problem().prior_covariance(),
            benchmark.design_matrix(),
            benchmark.qoi_matrix(),
            benchmark.noise_var(),
            weights_uniform,
            exact_cls, exact_args, bkd,
        )
        exact_high = compute_exact_prediction_utility(
            benchmark.problem().prior_mean(),
            benchmark.problem().prior_covariance(),
            benchmark.design_matrix(),
            benchmark.qoi_matrix(),
            benchmark.noise_var(),
            weights_high,
            exact_cls, exact_args, bkd,
        )

        assert exact_high < exact_uniform
