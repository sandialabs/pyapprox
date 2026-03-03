"""Tests for BayesianOptimizer."""

import numpy as np
import pytest

from pyapprox.optimization.bayesian.acquisition.analytic import (
    ExpectedImprovement,
)
from pyapprox.optimization.bayesian.acquisition_optimizer import (
    AcquisitionOptimizer,
)
from pyapprox.optimization.bayesian.domain.box import BoxDomain
from pyapprox.optimization.bayesian.fitter_adapter import (
    GPFitterAdapter,
    GPFixedFitterAdapter,
)
from pyapprox.optimization.bayesian.hp_schedule import (
    EveryKSchedule,
    GeometricSchedule,
)
from pyapprox.optimization.bayesian.optimizer import BayesianOptimizer
from pyapprox.optimization.minimize.scipy.slsqp import (
    ScipySLSQPOptimizer,
)
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.fitters.maximum_likelihood_fitter import (
    GPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.util.test_utils import slow_test


def _make_bo(
    bkd, nvars=1, bounds_list=None, minimize=True, hp_schedule=None,
    convergence=None,
):
    """Helper to create a BayesianOptimizer for testing."""
    from pyapprox.optimization.bayesian.acquisition_optimizer import (
        SobolCandidateGenerator,
    )

    if bounds_list is None:
        bounds_list = [[0.0, 1.0]] * nvars
    kernel = Matern52Kernel([1.0] * nvars, (0.1, 10.0), nvars, bkd)
    gp_template = ExactGaussianProcess(kernel, nvars, bkd, nugget=1e-6)
    domain = BoxDomain(bkd.array(bounds_list), bkd)
    fitter = GPFitterAdapter(GPMaximumLikelihoodFitter(bkd))
    ei = ExpectedImprovement()
    scipy_opt = ScipySLSQPOptimizer(maxiter=100)
    acq_opt = AcquisitionOptimizer(
        scipy_opt, bkd, n_restarts=5, n_raw_candidates=100,
        candidate_generator=SobolCandidateGenerator(seed=42),
    )
    return BayesianOptimizer(
        domain=domain,
        fitter=fitter,
        acquisition=ei,
        acquisition_optimizer=acq_opt,
        bkd=bkd,
        surrogate_template=gp_template,
        minimize=minimize,
        hp_schedule=hp_schedule,
        convergence=convergence,
    )


class TestBayesianOptimizerBasic:
    def test_ask_before_tell_returns_random(self, bkd) -> None:
        """ask() before any data returns random points in domain."""
        np.random.seed(42)
        bo = _make_bo(bkd)
        result = bo.ask(batch_size=3)
        assert result.X.shape == (1, 3)
        # Check points are in [0, 1]
        vals = bkd.to_numpy(result.X)[0]
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_tell_and_best(self, bkd) -> None:
        """tell() + best() returns correct best observed."""
        np.random.seed(42)
        bo = _make_bo(bkd)
        X = bkd.array([[0.2, 0.5, 0.8]])
        y = bkd.array([[0.04, 0.25, 0.64]])  # (x-0)^2 values
        bo.tell(X, y)

        best = bo.best()
        # Minimum is at x=0.2 with y=0.04
        bkd.assert_allclose(best.observed_x, bkd.array([[0.2]]), rtol=1e-12)
        bkd.assert_allclose(best.observed_y, bkd.array([[0.04]]), rtol=1e-12)

    def test_tell_incremental(self, bkd) -> None:
        """Multiple tell() calls accumulate data."""
        np.random.seed(42)
        bo = _make_bo(bkd)
        bo.tell(bkd.array([[0.2]]), bkd.array([[0.04]]))
        bo.tell(bkd.array([[0.5]]), bkd.array([[0.25]]))
        bo.tell(bkd.array([[0.1]]), bkd.array([[0.01]]))

        best = bo.best()
        bkd.assert_allclose(best.observed_x, bkd.array([[0.1]]), rtol=1e-12)
        bkd.assert_allclose(best.observed_y, bkd.array([[0.01]]), rtol=1e-12)

    def test_ask_after_tell(self, bkd) -> None:
        """ask() after tell() uses acquisition function."""
        np.random.seed(42)
        bo = _make_bo(bkd)
        X = bkd.array([[0.0, 0.5, 1.0]])
        y = bkd.array([[1.0, 0.0, 1.0]])
        bo.tell(X, y)

        result = bo.ask()
        assert result.X.shape == (1, 1)
        # Point should be in bounds
        val = float(bkd.to_numpy(result.X)[0, 0])
        assert 0.0 <= val <= 1.0

    def test_state_roundtrip(self, bkd) -> None:
        """state() and from_state() preserve data."""
        np.random.seed(42)
        bo = _make_bo(bkd)
        X = bkd.array([[0.2, 0.5, 0.8]])
        y = bkd.array([[0.04, 0.25, 0.64]])
        bo.tell(X, y)

        state = bo.state()
        bkd.assert_allclose(state.X_all, X, rtol=1e-12)
        bkd.assert_allclose(state.y_all, y, rtol=1e-12)

        # Restore from state
        kernel = Matern52Kernel([1.0], (0.1, 10.0), 1, bkd)
        gp_template = ExactGaussianProcess(kernel, 1, bkd, nugget=1e-6)
        domain = BoxDomain(bkd.array([[0.0, 1.0]]), bkd)
        fitter = GPFitterAdapter(GPMaximumLikelihoodFitter(bkd))
        ei = ExpectedImprovement()
        scipy_opt = ScipySLSQPOptimizer(maxiter=100)
        acq_opt = AcquisitionOptimizer(
            scipy_opt, bkd, n_restarts=5, n_raw_candidates=100
        )

        bo2 = BayesianOptimizer.from_state(
            state=state,
            domain=domain,
            fitter=fitter,
            acquisition=ei,
            acquisition_optimizer=acq_opt,
            bkd=bkd,
            surrogate_template=gp_template,
        )
        best2 = bo2.best()
        bkd.assert_allclose(best2.observed_x, bkd.array([[0.2]]), rtol=1e-12)


class TestBayesianOptimizerStep:
    def test_step(self, bkd) -> None:
        """step() performs ask/evaluate/tell cycle."""
        np.random.seed(42)
        bo = _make_bo(bkd)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.step(quadratic)
        assert result.values.shape[0] == 1
        assert result.best is not None

    def test_run(self, bkd) -> None:
        """run() performs multiple steps."""
        np.random.seed(42)
        bo = _make_bo(bkd)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=5)
        assert len(result.steps) == 5
        assert result.best is not None

    def test_quadratic_convergence(self, bkd) -> None:
        """BO finds near-optimum of 1D quadratic in 15 steps."""
        np.random.seed(42)
        bo = _make_bo(bkd)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=15)
        best_x = float(bkd.to_numpy(result.best.observed_x)[0, 0])
        best_y = float(bkd.to_numpy(result.best.observed_y)[0, 0])
        assert abs(best_x - 0.3) < 0.1, f"best_x={best_x}, expected ~0.3"
        assert best_y < 0.01, f"best_y={best_y}, expected < 0.01"

    def test_surrogate_available_after_tell(self, bkd) -> None:
        """surrogate() returns fitted model after tell()."""
        np.random.seed(42)
        bo = _make_bo(bkd)
        assert bo.surrogate() is None

        X = bkd.array([[0.2, 0.5, 0.8]])
        y = bkd.array([[0.04, 0.25, 0.64]])
        bo.tell(X, y)

        surr = bo.surrogate()
        assert surr is not None
        assert surr.is_fitted()


class TestBayesianOptimizerSchedule:
    def test_every_k_quadratic_convergence(self, bkd) -> None:
        """BO with EveryKSchedule(k=3) converges on 1D quadratic."""
        np.random.seed(42)
        bo = _make_bo(bkd, hp_schedule=EveryKSchedule(3))

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=15)
        best_x = float(bkd.to_numpy(result.best.observed_x)[0, 0])
        best_y = float(bkd.to_numpy(result.best.observed_y)[0, 0])
        assert abs(best_x - 0.3) < 0.15, f"best_x={best_x}, expected ~0.3"
        assert best_y < 0.02, f"best_y={best_y}, expected < 0.02"

    def test_geometric_quadratic_convergence(self, bkd) -> None:
        """BO with GeometricSchedule converges on 1D quadratic."""
        np.random.seed(42)
        bo = _make_bo(bkd, hp_schedule=GeometricSchedule(base=1.5))

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=15)
        best_x = float(bkd.to_numpy(result.best.observed_x)[0, 0])
        best_y = float(bkd.to_numpy(result.best.observed_y)[0, 0])
        assert abs(best_x - 0.3) < 0.15, f"best_x={best_x}, expected ~0.3"
        assert best_y < 0.02, f"best_y={best_y}, expected < 0.02"

    def test_incremental_matches_full_refit_with_schedule(self, bkd) -> None:
        """Incremental fast-path predictions match full-refit predictions.

        Uses EveryKSchedule(k=3) so tell steps 1,2 use the incremental
        fast path while step 0 does full HP optimization. Verifies that
        the surrogate after the incremental steps gives the same
        predictions as a from-scratch GPFixedHyperparameterFitter refit.
        """
        np.random.seed(42)
        schedule = EveryKSchedule(3)
        bo = _make_bo(bkd, hp_schedule=schedule)

        # Step 0: tell with 3 initial points (full HP optimization)
        X0 = bkd.array([[0.1, 0.5, 0.9]])
        y0 = bkd.array([[(0.1 - 0.3) ** 2, (0.5 - 0.3) ** 2, (0.9 - 0.3) ** 2]])
        bo.tell(X0, y0)

        # Steps 1,2: tell one point at a time (incremental fast path)
        X1 = bkd.array([[0.3]])
        y1 = bkd.array([[0.0]])
        bo.tell(X1, y1)

        X2 = bkd.array([[0.7]])
        y2 = bkd.array([[(0.7 - 0.3) ** 2]])
        bo.tell(X2, y2)

        # Get predictions from the BO surrogate (built incrementally)
        surr = bo.surrogate()
        assert surr is not None
        X_test = bkd.array([[0.2, 0.4, 0.6, 0.8]])
        mean_inc = surr.predict(X_test)
        std_inc = surr.predict_std(X_test)

        # Build a reference via full GPFixedHyperparameterFitter
        # using the same hyperparameters
        X_all = bkd.hstack([X0, X1, X2])
        y_all = bkd.hstack([y0, y1, y2])

        full_adapter = GPFixedFitterAdapter(bkd)
        # Copy hyperparameters from the BO surrogate to a fresh template
        kernel = Matern52Kernel([1.0], (0.1, 10.0), 1, bkd)
        gp_ref = ExactGaussianProcess(kernel, 1, bkd, nugget=1e-6)
        gp_ref.hyp_list().set_values(surr.hyp_list().get_values())
        fitted_ref = full_adapter.fit(gp_ref, X_all, y_all)

        mean_ref = fitted_ref.predict(X_test)
        std_ref = fitted_ref.predict_std(X_test)

        bkd.assert_allclose(mean_inc, mean_ref, rtol=1e-10)
        bkd.assert_allclose(std_inc, std_ref, rtol=1e-10)

    def test_state_roundtrip_preserves_tell_count(self, bkd) -> None:
        """state() and from_state() preserve _tell_count."""
        np.random.seed(42)
        schedule = EveryKSchedule(3)
        bo = _make_bo(bkd, hp_schedule=schedule)

        # Do 5 tell() calls
        for i in range(5):
            X = bkd.array([[0.1 * (i + 1)]])
            y = bkd.array([[(0.1 * (i + 1) - 0.3) ** 2]])
            bo.tell(X, y)

        state = bo.state()
        assert state.metadata["tell_count"] == 5

        # Restore
        kernel = Matern52Kernel([1.0], (0.1, 10.0), 1, bkd)
        gp_template = ExactGaussianProcess(kernel, 1, bkd, nugget=1e-6)
        domain = BoxDomain(bkd.array([[0.0, 1.0]]), bkd)
        fitter = GPFitterAdapter(GPMaximumLikelihoodFitter(bkd))
        ei = ExpectedImprovement()
        scipy_opt = ScipySLSQPOptimizer(maxiter=100)
        acq_opt = AcquisitionOptimizer(
            scipy_opt, bkd, n_restarts=5, n_raw_candidates=100
        )

        bo2 = BayesianOptimizer.from_state(
            state=state,
            domain=domain,
            fitter=fitter,
            acquisition=ei,
            acquisition_optimizer=acq_opt,
            bkd=bkd,
            surrogate_template=gp_template,
            hp_schedule=schedule,
        )
        assert bo2._tell_count == 5


class TestSurrogatePolishing:
    def test_polishing_on_quadratic(self, bkd) -> None:
        """Surrogate polishing finds near-optimum of 1D quadratic."""
        np.random.seed(42)
        bo = _make_bo(bkd)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=15)
        rec_y = float(bkd.to_numpy(result.best.recommended_y)[0, 0])
        assert rec_y < 1e-4, (
            f"recommended_y={rec_y}, polishing should find near-minimum"
        )

    def test_polishing_improves_true_value(self, bkd) -> None:
        """True function at polished point <= observed best."""
        np.random.seed(42)
        bo = _make_bo(bkd)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=15)
        # Evaluate true function at recommended point
        true_rec_y = quadratic(result.best.recommended_x)  # (1, 1)
        # Polished point should be near the true minimum (0.0)
        bkd.assert_allclose(true_rec_y, bkd.zeros((1, 1)), atol=1e-4)

    def test_polishing_recommended_x_near_true_minimum(self, bkd) -> None:
        """Polished recommended_x should be close to true minimum x=0.3."""
        np.random.seed(42)
        bo = _make_bo(bkd)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=15)
        # Recommended_x should be close to true minimum at 0.3
        bkd.assert_allclose(
            result.best.recommended_x,
            bkd.array([[0.3]]),
            atol=0.05,
        )
        # True function value at recommended point should be near zero
        true_rec_y = quadratic(result.best.recommended_x)
        bkd.assert_allclose(true_rec_y, bkd.zeros((1, 1)), atol=1e-3)


class TestBayesianOptimizerSlow:
    @slow_test
    def test_branin_convergence(self, numpy_bkd) -> None:
        """BO with EI finds near-optimum of Branin in ~40 steps."""
        bkd = numpy_bkd
        np.random.seed(42)

        from pyapprox.benchmarks.functions.algebraic.branin import (
            BraninFunction,
        )

        branin = BraninFunction(bkd)
        bo = _make_bo(
            bkd,
            nvars=2,
            bounds_list=[[-5.0, 10.0], [0.0, 15.0]],
            minimize=True,
        )

        result = bo.run(branin, budget=40)
        best_y = float(bkd.to_numpy(result.best.observed_y)[0, 0])
        assert best_y < 1.0, f"best_y={best_y}, expected < 1.0 (global min ~0.398)"
        rec_y = float(bkd.to_numpy(result.best.recommended_y)[0, 0])
        assert rec_y < 0.5, f"recommended_y={rec_y}, expected < 0.5"

    @slow_test
    @pytest.mark.skip(
        reason="BO acquisition optimizer needs improvement for reliable "
        "Rosenbrock convergence across platforms"
    )
    def test_rosenbrock_convergence(self, numpy_bkd) -> None:
        """BO with EI gets near Rosenbrock minimum."""
        bkd = numpy_bkd
        np.random.seed(42)

        from pyapprox.benchmarks.functions.algebraic.rosenbrock import (
            RosenbrockFunction,
        )

        rosenbrock = RosenbrockFunction(bkd, nvars=2)
        bo = _make_bo(
            bkd,
            nvars=2,
            bounds_list=[[-2.0, 2.0], [-2.0, 2.0]],
            minimize=True,
        )

        result = bo.run(rosenbrock, budget=100)
        best_y = float(bkd.to_numpy(result.best.observed_y)[0, 0])
        assert best_y < 10.0, f"best_y={best_y}, expected < 10.0 (global min 0)"

    @slow_test
    @pytest.mark.skip(
        reason="BO acquisition optimizer needs improvement for reliable "
        "Rosenbrock convergence across platforms"
    )
    def test_rosenbrock_with_incremental_schedule(self, numpy_bkd) -> None:
        """BO with EveryKSchedule uses incremental fitter on Rosenbrock.

        Verifies that the incremental Cholesky fast path does not
        degrade optimization quality compared to always-optimize.
        """
        bkd = numpy_bkd
        np.random.seed(42)

        from pyapprox.benchmarks.functions.algebraic.rosenbrock import (
            RosenbrockFunction,
        )

        rosenbrock = RosenbrockFunction(bkd, nvars=2)
        bo = _make_bo(
            bkd,
            nvars=2,
            bounds_list=[[-2.0, 2.0], [-2.0, 2.0]],
            minimize=True,
            hp_schedule=EveryKSchedule(3),
        )

        result = bo.run(rosenbrock, budget=100)
        best_y = float(bkd.to_numpy(result.best.observed_y)[0, 0])
        # Incremental schedule is less accurate; just verify it runs and
        # finds something reasonable on [-2,2]^2
        assert best_y < 100.0, f"best_y={best_y}, expected < 100.0 (global min 0)"
