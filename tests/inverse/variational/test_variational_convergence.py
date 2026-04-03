"""Tests for VI convergence diagnostics.

Tests cover:
- ConvergenceCheckProtocol satisfaction
- ImportanceWeightedCheck with known-answer Gaussian
- VIConvergenceMonitor iteration tracking and check scheduling
- Scipy callback integration
- ROL StatusTest integration (when pyrol available)
"""

import math

import numpy as np
import pytest

from pyapprox.inverse.variational.convergence import (
    VIConvergenceMonitor,
    make_scipy_convergence_callback,
)
from pyapprox.inverse.variational.convergence_protocols import (
    ConvergenceCheckProtocol,
    ConvergenceCheckResult,
)
from pyapprox.inverse.variational.elbo import make_single_problem_elbo
from pyapprox.inverse.variational.importance_diagnostics import (
    ImportanceWeightedCheck,
    ImportanceWeightedMetrics,
    _logsumexp,
    make_importance_check_from_elbo,
)
from pyapprox.probability.conditional.gaussian import ConditionalGaussian
from pyapprox.probability.conditional.joint import (
    ConditionalIndependentJoint,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.protocols import Backend
from tests._helpers.markers import slow_test


def _make_degree0_expansion(bkd: Backend, coeff: float = 0.0) -> BasisExpansion:
    """Create a degree-0 BasisExpansion (constant function)."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1, 0, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=1)
    exp.set_coefficients(bkd.asarray([[coeff]]))
    return exp


def _make_cond_gaussian(
    bkd: Backend, mean: float = 0.0, log_stdev: float = 0.0
) -> ConditionalGaussian:
    """Create a ConditionalGaussian with constant mean and log_stdev."""
    mean_func = _make_degree0_expansion(bkd, mean)
    log_stdev_func = _make_degree0_expansion(bkd, log_stdev)
    return ConditionalGaussian(mean_func, log_stdev_func, bkd)


def _make_gaussian_vi_setup(bkd, obs_value=2.0, noise_var=0.5, nsamples=100):
    """Create a simple 1D Gaussian VI problem.

    True model: y = z + noise, z ~ N(0, 1), noise ~ N(0, noise_var).
    Observation: y = obs_value.
    Exact posterior: N(posterior_mean, posterior_var).
    """
    prior_mean = 0.0
    prior_var = 1.0
    # Exact posterior: precision-weighted mean
    posterior_var = 1.0 / (1.0 / prior_var + 1.0 / noise_var)
    posterior_mean = posterior_var * (
        prior_mean / prior_var + obs_value / noise_var
    )

    log_std = math.log(math.sqrt(posterior_var))
    cond = _make_cond_gaussian(bkd, mean=posterior_mean, log_stdev=log_std)
    prior = GaussianMarginal(prior_mean, math.sqrt(prior_var), bkd)

    obs = bkd.asarray([[obs_value]])

    def log_lik_fn(z):
        diff = z - obs
        return -0.5 / noise_var * diff**2 - 0.5 * math.log(2 * math.pi * noise_var)

    def wrapped_log_lik(z, labels):
        return log_lik_fn(z)

    def log_prior_fn(z):
        return prior.logpdf(z)

    np.random.seed(42)
    base_samples = bkd.asarray(np.random.normal(0, 1, (1, nsamples)))
    weights = bkd.full((1, nsamples), 1.0 / nsamples)

    elbo = make_single_problem_elbo(
        cond, log_lik_fn, prior, base_samples, weights, bkd,
    )

    return {
        "cond": cond,
        "prior": prior,
        "elbo": elbo,
        "log_lik_fn": wrapped_log_lik,
        "log_prior_fn": log_prior_fn,
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "nlabel_dims": cond.nvars(),
    }


# --- Protocol tests ---


class TestConvergenceCheckProtocol:
    def test_importance_weighted_satisfies_protocol(self, bkd) -> None:
        setup = _make_gaussian_vi_setup(bkd)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
        )
        assert isinstance(check, ConvergenceCheckProtocol)

    def test_check_result_fields(self, bkd) -> None:
        result = ConvergenceCheckResult(
            should_stop=True,
            approximation_quality=0.8,
            detail={"ess": 100.0},
            check_type="test",
        )
        assert result.should_stop is True
        assert result.approximation_quality == 0.8
        assert result.detail["ess"] == 100.0
        assert result.check_type == "test"


# --- Logsumexp tests ---


class TestLogsumexp:
    def test_basic_values(self, bkd) -> None:
        x = bkd.asarray([1.0, 2.0, 3.0])
        result = _logsumexp(bkd, x)
        expected = math.log(math.exp(1) + math.exp(2) + math.exp(3))
        bkd.assert_allclose(
            bkd.reshape(result, (1,)),
            bkd.asarray([expected]),
            rtol=1e-10,
        )

    def test_large_values_stable(self, bkd) -> None:
        """logsumexp should handle large values without overflow."""
        x = bkd.asarray([1000.0, 1001.0, 1002.0])
        result = _logsumexp(bkd, x)
        expected = 1002.0 + math.log(
            math.exp(-2) + math.exp(-1) + 1.0
        )
        bkd.assert_allclose(
            bkd.reshape(result, (1,)),
            bkd.asarray([expected]),
            rtol=1e-10,
        )

    def test_negative_values(self, bkd) -> None:
        x = bkd.asarray([-1000.0, -999.0])
        result = _logsumexp(bkd, x)
        expected = -999.0 + math.log(math.exp(-1) + 1.0)
        bkd.assert_allclose(
            bkd.reshape(result, (1,)),
            bkd.asarray([expected]),
            rtol=1e-10,
        )


# --- ImportanceWeightedCheck tests ---


class TestImportanceWeightedCheck:
    def test_compute_log_weights_shape(self, bkd) -> None:
        setup = _make_gaussian_vi_setup(bkd, nsamples=50)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=30,
        )
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(123)
        log_w = check.compute_log_weights(params)
        assert log_w.shape == (30,)

    def test_compute_metrics_returns_dataclass(self, bkd) -> None:
        setup = _make_gaussian_vi_setup(bkd, nsamples=50)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=50,
        )
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        metrics = check.compute_metrics(params)
        assert isinstance(metrics, ImportanceWeightedMetrics)
        assert metrics.n_samples == 50
        assert metrics.ess_ratio >= 0.0
        assert metrics.ess_ratio <= 1.0 + 1e-12

    def test_evidence_bound_ge_elbo(self, bkd) -> None:
        """Evidence bound should be >= ELBO estimate (Jensen's inequality)."""
        setup = _make_gaussian_vi_setup(bkd, nsamples=200)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=500,
        )
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        metrics = check.compute_metrics(params)
        # evidence_bound >= elbo_estimate (up to Monte Carlo noise)
        assert metrics.evidence_gap >= -0.5  # allow small MC noise

    def test_check_returns_result(self, bkd) -> None:
        setup = _make_gaussian_vi_setup(bkd, nsamples=50)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=50,
        )
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        result = check.check(params, 0.01)
        assert isinstance(result, ConvergenceCheckResult)
        assert result.check_type == "importance_weighted"
        assert 0.0 <= result.approximation_quality <= 1.0

    def test_check_does_not_stop_with_large_improvement(self, bkd) -> None:
        """With large recent improvement, should not stop."""
        setup = _make_gaussian_vi_setup(bkd, nsamples=50)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=50,
            gap_ratio_threshold=10.0,
        )
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        result = check.check(params, 1e6)
        assert result.should_stop is False

    def test_check_stops_with_tiny_improvement(self, bkd) -> None:
        """With negligible improvement vs gap, should stop."""
        setup = _make_gaussian_vi_setup(bkd, nsamples=50)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=200,
            gap_ratio_threshold=0.001,
        )
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        result = check.check(params, 1e-15)
        assert result.should_stop is True

    def test_check_zero_improvement_does_not_stop(self, bkd) -> None:
        """Zero improvement should not trigger stop (division protection)."""
        setup = _make_gaussian_vi_setup(bkd, nsamples=50)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=50,
        )
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        result = check.check(params, 0.0)
        assert result.should_stop is False

    @slow_test
    def test_good_fit_has_high_ess(self, bkd) -> None:
        """A well-fit variational distribution should have high ESS ratio."""
        setup = _make_gaussian_vi_setup(bkd, nsamples=200)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=1000,
        )
        # Use the exact posterior parameters
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        metrics = check.compute_metrics(params)
        # Exact posterior should have high ESS
        assert metrics.ess_ratio > 0.3

    @slow_test
    def test_bad_fit_has_low_ess(self, bkd) -> None:
        """A poorly-fit variational distribution should have low ESS ratio."""
        setup = _make_gaussian_vi_setup(bkd, nsamples=200)
        # Create a bad variational dist (wrong mean and variance)
        bad_cond = _make_cond_gaussian(bkd, mean=10.0, log_stdev=2.0)
        check = ImportanceWeightedCheck(
            bad_cond,
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            bad_cond.nvars(),
            None,
            bkd,
            n_diagnostic_samples=1000,
        )
        params = bkd.asarray(
            bad_cond.hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        metrics = check.compute_metrics(params)
        assert metrics.ess_ratio < 0.3


# --- Factory tests ---


class TestMakeImportanceCheckFromElbo:
    def test_factory_creates_check(self, bkd) -> None:
        setup = _make_gaussian_vi_setup(bkd)
        check = make_importance_check_from_elbo(
            setup["elbo"],
            setup["log_prior_fn"],
            n_diagnostic_samples=50,
        )
        assert isinstance(check, ImportanceWeightedCheck)
        assert isinstance(check, ConvergenceCheckProtocol)

    def test_factory_check_runs(self, bkd) -> None:
        setup = _make_gaussian_vi_setup(bkd)
        check = make_importance_check_from_elbo(
            setup["elbo"],
            setup["log_prior_fn"],
            n_diagnostic_samples=50,
        )
        params = bkd.asarray(
            setup["cond"].hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        result = check.check(params, 0.01)
        assert isinstance(result, ConvergenceCheckResult)


# --- VIConvergenceMonitor tests ---


class _DummyCheck:
    """Deterministic convergence check for testing the monitor."""

    def __init__(self, stop_after: int = 5) -> None:
        self._call_count = 0
        self._stop_after = stop_after

    def check(
        self, params: object, recent_elbo_improvement: float,
    ) -> ConvergenceCheckResult:
        self._call_count += 1
        should_stop = self._call_count >= self._stop_after
        return ConvergenceCheckResult(
            should_stop=should_stop,
            approximation_quality=0.5,
            check_type="dummy",
        )


class TestVIConvergenceMonitor:
    def test_initial_state(self, numpy_bkd) -> None:
        check = _DummyCheck()
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=10)
        assert monitor.triggered is False
        assert monitor.last_result is None
        assert len(monitor.check_history) == 0
        assert len(monitor.elbo_history) == 0

    def test_record_iteration(self, numpy_bkd) -> None:
        check = _DummyCheck()
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=10)
        params = numpy_bkd.zeros((2, 1))
        monitor.record_iteration(params, 1.0)
        monitor.record_iteration(params, 2.0)
        assert len(monitor.elbo_history) == 2
        assert monitor.elbo_history[0] == 1.0
        assert monitor.elbo_history[1] == 2.0

    def test_should_check_respects_min_iterations(self, numpy_bkd) -> None:
        check = _DummyCheck()
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=10)
        params = numpy_bkd.zeros((2, 1))
        for i in range(10):
            monitor.record_iteration(params, float(i))
            if i < 9:  # iteration count is 1-indexed internally
                assert not monitor.should_check()

    def test_should_check_respects_check_every(self, numpy_bkd) -> None:
        check = _DummyCheck()
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=5)
        params = numpy_bkd.zeros((2, 1))
        check_iterations = []
        for i in range(30):
            monitor.record_iteration(params, float(i))
            if monitor.should_check():
                check_iterations.append(i + 1)  # 1-indexed
        # Should check at iterations 5, 10, 15, 20, 25, 30
        assert check_iterations == [5, 10, 15, 20, 25, 30]

    def test_recent_improvement(self, numpy_bkd) -> None:
        check = _DummyCheck()
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=5)
        params = numpy_bkd.zeros((2, 1))
        # Record 10 iterations with ELBO values 0..9
        for i in range(10):
            monitor.record_iteration(params, float(i))
        # lookback = min(5, 9) = 5, so improvement = 9 - 4 = 5
        assert monitor.recent_improvement() == 5.0

    def test_trigger_stops(self, numpy_bkd) -> None:
        check = _DummyCheck(stop_after=1)  # stop on first check
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=5)
        params = numpy_bkd.zeros((2, 1))
        for i in range(5):
            monitor.record_iteration(params, float(i))
        assert monitor.should_check()
        result = monitor.run_check(params)
        assert result.should_stop is True
        assert monitor.triggered is True

    def test_no_trigger_until_stop(self, numpy_bkd) -> None:
        check = _DummyCheck(stop_after=3)
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=5)
        params = numpy_bkd.zeros((2, 1))
        for i in range(10):
            monitor.record_iteration(params, float(i))
            if monitor.should_check():
                monitor.run_check(params)
        # First two checks should not trigger, third should
        assert len(monitor.check_history) == 2
        # First check at iteration 5, second at 10
        assert monitor.check_history[0].should_stop is False
        assert monitor.check_history[1].should_stop is False
        assert monitor.triggered is False

        # Continue to trigger at iteration 15
        for i in range(10, 15):
            monitor.record_iteration(params, float(i))
            if monitor.should_check():
                monitor.run_check(params)
        assert monitor.triggered is True


# --- Scipy callback tests ---


class TestScipyCallback:
    def test_callback_creation(self, numpy_bkd) -> None:
        check = _DummyCheck()
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=5)
        callback = make_scipy_convergence_callback(monitor, numpy_bkd)
        assert callable(callback)

    def test_callback_records_iterations(self, numpy_bkd) -> None:
        check = _DummyCheck()
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=5)
        callback = make_scipy_convergence_callback(monitor, numpy_bkd)

        # Simulate scipy calling callback
        class _FakeState:
            fun = -1.5  # negative ELBO

        x = np.array([0.1, 0.2])
        callback(x, _FakeState())
        assert len(monitor.elbo_history) == 1
        assert monitor.elbo_history[0] == 1.5  # ELBO = -(-1.5)

    def test_callback_returns_false_normally(self, numpy_bkd) -> None:
        check = _DummyCheck(stop_after=100)
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=5)
        callback = make_scipy_convergence_callback(monitor, numpy_bkd)

        class _FakeState:
            fun = -1.0

        x = np.array([0.1, 0.2])
        result = callback(x, _FakeState())
        assert result is False

    def test_callback_returns_true_when_triggered(self, numpy_bkd) -> None:
        check = _DummyCheck(stop_after=1)
        monitor = VIConvergenceMonitor(check, check_every=5, min_iterations=5)
        callback = make_scipy_convergence_callback(monitor, numpy_bkd)

        class _FakeState:
            fun = -1.0

        x = np.array([0.1, 0.2])
        # Run 5 iterations to pass min_iterations
        for _ in range(4):
            callback(x, _FakeState())
        assert not monitor.triggered
        # 5th iteration triggers check
        result = callback(x, _FakeState())
        assert result is True
        assert monitor.triggered is True


# --- Scipy optimizer integration ---


class TestScipyOptimizerWithCallback:
    @slow_test
    def test_early_stopping_with_callback(self, bkd) -> None:
        """Test that ScipyTrustConstrOptimizer stops early with callback."""
        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )

        setup = _make_gaussian_vi_setup(bkd, nsamples=200)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=100,
            gap_ratio_threshold=0.0001,
        )
        monitor = VIConvergenceMonitor(
            check, check_every=5, min_iterations=5,
        )
        callback = make_scipy_convergence_callback(monitor, bkd)

        optimizer = ScipyTrustConstrOptimizer(
            maxiter=500, callback=callback,
        )
        optimizer.bind(setup["elbo"], setup["elbo"].bounds())
        init_guess = bkd.zeros((setup["elbo"].nvars(), 1))
        result = optimizer.minimize(init_guess)
        # Should have completed (possibly early)
        assert result.optima() is not None
        # Monitor should have recorded some iterations
        assert len(monitor.elbo_history) > 0


# --- ROL integration tests ---

_HAS_PYROL = True
try:
    import pyrol  # noqa: F401
except ImportError:
    _HAS_PYROL = False


@pytest.mark.skipif(not _HAS_PYROL, reason="pyrol not installed")
class TestROLConvergenceStatusTest:
    @slow_test
    def test_rol_with_status_test(self, bkd) -> None:
        """Test that ROLOptimizer accepts and uses a status_test."""
        from pyapprox.inverse.variational.convergence import (
            make_rol_convergence_status_test,
        )
        from pyapprox.optimization.minimize.rol.rol_optimizer import (
            ROLOptimizer,
        )

        setup = _make_gaussian_vi_setup(bkd, nsamples=200)
        check = ImportanceWeightedCheck(
            setup["cond"],
            setup["log_lik_fn"],
            setup["log_prior_fn"],
            setup["nlabel_dims"],
            None,
            bkd,
            n_diagnostic_samples=100,
            gap_ratio_threshold=0.0001,
        )
        monitor = VIConvergenceMonitor(
            check, check_every=5, min_iterations=5,
        )
        status_test = make_rol_convergence_status_test(monitor, bkd)

        optimizer = ROLOptimizer(
            verbosity=0,
            status_test=status_test,
        )
        optimizer.bind(setup["elbo"], setup["elbo"].bounds())
        init_guess = bkd.zeros((setup["elbo"].nvars(), 1))
        result = optimizer.minimize(init_guess)
        assert result.optima() is not None
        assert len(monitor.elbo_history) > 0


# --- Multi-QoI joint distribution test ---


class TestImportanceWeightedCheckJoint:
    def test_joint_distribution_check(self, bkd) -> None:
        """Test with ConditionalIndependentJoint (2D)."""
        cond1 = _make_cond_gaussian(bkd, mean=1.0, log_stdev=math.log(0.5))
        cond2 = _make_cond_gaussian(bkd, mean=-1.0, log_stdev=math.log(0.3))
        joint_cond = ConditionalIndependentJoint([cond1, cond2], bkd)

        prior = IndependentJoint(
            [GaussianMarginal(0.0, 1.0, bkd), GaussianMarginal(0.0, 1.0, bkd)],
            bkd,
        )

        def log_lik_fn(z, labels):
            return -0.5 * bkd.sum(z**2, axis=0, keepdims=True)

        def log_prior_fn(z):
            return prior.logpdf(z)

        check = ImportanceWeightedCheck(
            joint_cond,
            log_lik_fn,
            log_prior_fn,
            joint_cond.nvars(),
            None,
            bkd,
            n_diagnostic_samples=100,
        )
        params = bkd.asarray(
            joint_cond.hyp_list().get_active_values()
        )[:, None]
        np.random.seed(42)
        metrics = check.compute_metrics(params)
        assert isinstance(metrics, ImportanceWeightedMetrics)
        assert metrics.n_samples == 100
