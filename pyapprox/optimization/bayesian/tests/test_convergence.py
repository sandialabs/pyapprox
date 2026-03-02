"""Tests for convergence criteria."""

import numpy as np

from pyapprox.optimization.bayesian.acquisition.analytic import (
    ExpectedImprovement,
)
from pyapprox.optimization.bayesian.acquisition_optimizer import (
    AcquisitionOptimizer,
)
from pyapprox.optimization.bayesian.convergence import (
    AcquisitionToleranceCriterion,
    DistanceToleranceCriterion,
    ValueToleranceCriterion,
)
from pyapprox.optimization.bayesian.domain.box import BoxDomain
from pyapprox.optimization.bayesian.fitter_adapter import GPFitterAdapter
from pyapprox.optimization.bayesian.optimizer import BayesianOptimizer
from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.fitters.maximum_likelihood_fitter import (
    GPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


def _make_bo(bkd, convergence=None, budget_hint=20):
    """Helper to create a BO for convergence testing."""
    kernel = Matern52Kernel([1.0], (0.1, 10.0), 1, bkd)
    gp_template = ExactGaussianProcess(kernel, 1, bkd, nugget=1e-6)
    domain = BoxDomain(bkd.array([[0.0, 1.0]]), bkd)
    fitter = GPFitterAdapter(GPMaximumLikelihoodFitter(bkd))
    ei = ExpectedImprovement()
    scipy_opt = ScipySLSQPOptimizer(maxiter=100)
    acq_opt = AcquisitionOptimizer(
        scipy_opt, bkd, n_restarts=5, n_raw_candidates=100
    )
    return BayesianOptimizer(
        domain=domain,
        fitter=fitter,
        acquisition=ei,
        acquisition_optimizer=acq_opt,
        bkd=bkd,
        surrogate_template=gp_template,
        minimize=True,
        convergence=convergence,
    )


class TestValueToleranceCriterion:
    def test_stops_early(self, bkd) -> None:
        """run() stops before budget when value tolerance is met."""
        np.random.seed(42)
        criterion = ValueToleranceCriterion(atol=1e-3, patience=2)
        bo = _make_bo(bkd, convergence=criterion)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=30)
        # Should converge and stop early
        assert len(result.steps) < 30, (
            f"Expected early stop, got {len(result.steps)} steps"
        )

    def test_patience_respected(self, numpy_bkd) -> None:
        """Convergence requires patience consecutive steps."""
        criterion = ValueToleranceCriterion(atol=1e-3, patience=3)
        # Count should only increment on consecutive below-threshold steps
        assert criterion._count == 0


class TestAcquisitionToleranceCriterion:
    def test_stops_early(self, bkd) -> None:
        """run() stops when max acquisition value is below tolerance."""
        np.random.seed(42)
        # Use a generous tolerance so it triggers
        criterion = AcquisitionToleranceCriterion(atol=1e-2)
        bo = _make_bo(bkd, convergence=criterion)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=30)
        # Should stop early once EI becomes small
        assert len(result.steps) < 30, (
            f"Expected early stop, got {len(result.steps)} steps"
        )


class TestDistanceToleranceCriterion:
    def test_stops_early(self, bkd) -> None:
        """run() stops when recommended_x stops moving."""
        np.random.seed(42)
        criterion = DistanceToleranceCriterion(atol=1e-3, patience=2)
        bo = _make_bo(bkd, convergence=criterion)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=30)
        assert len(result.steps) < 30, (
            f"Expected early stop, got {len(result.steps)} steps"
        )


class TestNoConvergence:
    def test_none_convergence_uses_full_budget(self, bkd) -> None:
        """No criterion runs to full budget."""
        np.random.seed(42)
        bo = _make_bo(bkd, convergence=None)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=5)
        assert len(result.steps) == 5

    def test_no_convergence_uses_full_budget(self, bkd) -> None:
        """Criterion that never fires runs to full budget."""
        np.random.seed(42)
        # Extremely tight tolerance that won't be met in 5 steps
        criterion = ValueToleranceCriterion(atol=1e-20, patience=100)
        bo = _make_bo(bkd, convergence=criterion)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        result = bo.run(quadratic, budget=5)
        assert len(result.steps) == 5


class TestHasConvergedAskTell:
    def test_has_converged_false_without_criterion(self, bkd) -> None:
        """has_converged() returns False when no criterion set."""
        np.random.seed(42)
        bo = _make_bo(bkd, convergence=None)
        X = bkd.array([[0.2, 0.5, 0.8]])
        y = bkd.array([[0.04, 0.25, 0.64]])
        bo.tell(X, y)
        assert bo.has_converged() is False

    def test_has_converged_false_before_tell(self, bkd) -> None:
        """has_converged() returns False before any data."""
        criterion = ValueToleranceCriterion(atol=1e-3, patience=1)
        bo = _make_bo(bkd, convergence=criterion)
        assert bo.has_converged() is False

    def test_has_converged_in_ask_tell_workflow(self, bkd) -> None:
        """Manual ask/tell loop detects convergence via has_converged()."""
        np.random.seed(42)
        criterion = ValueToleranceCriterion(atol=1e-3, patience=2)
        bo = _make_bo(bkd, convergence=criterion)

        def quadratic(X):
            return (X[0:1, :] - 0.3) ** 2

        n_evals = 0
        max_evals = 30
        while n_evals < max_evals and not bo.has_converged():
            x = bo.ask()
            y = quadratic(x.X)
            bo.tell(x.X, y)
            n_evals += 1

        assert n_evals < max_evals, (
            f"Expected convergence before {max_evals} evals, got {n_evals}"
        )
        assert bo.has_converged()
