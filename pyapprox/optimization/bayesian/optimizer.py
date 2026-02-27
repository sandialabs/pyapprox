"""Bayesian Optimization orchestrator with ask/tell/step/run workflows."""

import copy
import warnings
from typing import Callable, Generic, List, Optional

import numpy as np

from pyapprox.optimization.bayesian.batch.greedy import KrigingBeliever
from pyapprox.optimization.bayesian.convergence import (
    ConvergenceCriterionProtocol,
)
from pyapprox.optimization.bayesian.fitter_adapter import GPIncrementalFitterAdapter
from pyapprox.optimization.bayesian.hp_schedule import (
    AlwaysOptimizeSchedule,
    HPRefitScheduleProtocol,
)
from pyapprox.optimization.bayesian.protocols import (
    AcquisitionContext,
    AcquisitionFunctionProtocol,
    AcquisitionOptimizerProtocol,
    BODomainProtocol,
    BatchStrategyProtocol,
    SurrogateFitterProtocol,
    SurrogateProtocol,
)
from pyapprox.optimization.bayesian.state import (
    AskResult,
    BestResult,
    BOState,
    ConvergenceContext,
    RunResult,
    StepResult,
)
from pyapprox.util.backends.protocols import Array, Backend


class BayesianOptimizer(Generic[Array]):
    """Bayesian Optimization orchestrator.

    Provides ask/tell/step/run workflows for optimizing expensive
    black-box functions using a Gaussian process surrogate.

    Parameters
    ----------
    domain : BODomainProtocol[Array]
        Search domain.
    fitter : SurrogateFitterProtocol[Array]
        Surrogate fitter (e.g., GPFitterAdapter).
    acquisition : AcquisitionFunctionProtocol[Array]
        Acquisition function (e.g., ExpectedImprovement).
    acquisition_optimizer : AcquisitionOptimizerProtocol[Array]
        Optimizer for acquisition maximization.
    bkd : Backend[Array]
        Computational backend.
    surrogate_template : SurrogateProtocol[Array]
        Unfitted surrogate template (cloned for each refit).
    batch_strategy : BatchStrategyProtocol[Array]
        Strategy for selecting batch points. Default KrigingBeliever.
    minimize : bool
        If True, minimize the objective. Default True.
    hp_schedule : HPRefitScheduleProtocol
        Schedule controlling when full HP optimization runs vs
        cheap Cholesky-only refits. Default AlwaysOptimizeSchedule.
    convergence : Optional[ConvergenceCriterionProtocol[Array]]
        Optional convergence criterion for early stopping in run()
        and has_converged() in ask/tell workflows.
    """

    def __init__(
        self,
        domain: BODomainProtocol[Array],
        fitter: SurrogateFitterProtocol[Array],
        acquisition: AcquisitionFunctionProtocol[Array],
        acquisition_optimizer: AcquisitionOptimizerProtocol[Array],
        bkd: Backend[Array],
        surrogate_template: SurrogateProtocol[Array],
        batch_strategy: Optional[BatchStrategyProtocol[Array]] = None,
        minimize: bool = True,
        hp_schedule: Optional[HPRefitScheduleProtocol] = None,
        convergence: Optional[ConvergenceCriterionProtocol[Array]] = None,
    ) -> None:
        self._domain = domain
        self._fitter = fitter
        self._acquisition = acquisition
        self._acq_optimizer = acquisition_optimizer
        self._bkd = bkd
        self._surrogate_template = surrogate_template
        self._batch_strategy: BatchStrategyProtocol[Array] = (
            batch_strategy if batch_strategy is not None else KrigingBeliever()
        )
        self._minimize = minimize
        self._hp_schedule: HPRefitScheduleProtocol = (
            hp_schedule if hp_schedule is not None else AlwaysOptimizeSchedule()
        )
        self._convergence = convergence
        self._fixed_fitter: SurrogateFitterProtocol[Array] = (
            GPIncrementalFitterAdapter(bkd)
        )

        # Mutable state
        self._X_all: Optional[Array] = None
        self._y_all: Optional[Array] = None
        self._surrogate: Optional[SurrogateProtocol[Array]] = None
        self._tell_count: int = 0
        self._prev_best: Optional[BestResult[Array]] = None
        self._last_max_acq_value: Optional[float] = None

    def tell(self, X: Array, y: Array) -> None:
        """Add observed data.

        Warm-starts GP hyperparameters from the previous fit when available.

        Parameters
        ----------
        X : Array
            Observed inputs, shape (nvars, n).
        y : Array
            Observed outputs, shape (nqoi, n).
        """
        # Save previous best for convergence tracking
        if self._surrogate is not None and self._surrogate.is_fitted():
            self._prev_best = self.best()

        if self._X_all is None:
            self._X_all = X
            self._y_all = y
        else:
            self._X_all = self._bkd.hstack([self._X_all, X])
            self._y_all = self._bkd.hstack([self._y_all, y])  # type: ignore[list-item]

        # Warm-start: set previous hyperparameters on template
        self._warm_start_template()

        # Refit surrogate: full HP optimization or Cholesky-only
        if self._hp_schedule.should_optimize(self._tell_count):
            self._surrogate = self._fitter.fit(
                self._surrogate_template, self._X_all, self._y_all  # type: ignore[arg-type]
            )
            # Seed incremental cache with post-optimization GP
            if hasattr(self._fixed_fitter, 'set_prev_surrogate'):
                self._fixed_fitter.set_prev_surrogate(self._surrogate)  # type: ignore[union-attr]
        else:
            self._surrogate = self._fixed_fitter.fit(
                self._surrogate_template, self._X_all, self._y_all  # type: ignore[arg-type]
            )
        self._tell_count += 1

    def ask(self, batch_size: int = 1) -> AskResult[Array]:
        """Select next points to evaluate.

        If no data has been observed yet, returns random points
        from the domain.

        Parameters
        ----------
        batch_size : int
            Number of points to select. Default 1.

        Returns
        -------
        AskResult[Array]
            Points to evaluate.
        """
        if self._surrogate is None or not self._surrogate.is_fitted():
            # No data yet — random sampling
            self._last_max_acq_value = None
            X = self._random_sample(batch_size)
            return AskResult(X=X)

        if batch_size == 1:
            ctx = self._build_context()
            X, acq_val = self._acq_optimizer.maximize_with_value(
                self._acquisition, ctx, self._domain
            )
            self._last_max_acq_value = acq_val
        else:
            X = self._batch_strategy.select_batch(
                batch_size,
                self._acquisition,
                self._build_context,
                self._acq_optimizer,
                self._domain,
            )
            self._last_max_acq_value = None

        return AskResult(X=X)

    def step(
        self,
        function: Callable[[Array], Array],
        batch_size: int = 1,
    ) -> StepResult[Array]:
        """Perform one ask/evaluate/tell cycle.

        Parameters
        ----------
        function : Callable[[Array], Array]
            Function to evaluate. Takes (nvars, n) returns (nqoi, n).
        batch_size : int
            Number of points per step. Default 1.

        Returns
        -------
        StepResult[Array]
            Result of this step.
        """
        ask_result = self.ask(batch_size)
        values = function(ask_result.X)
        self.tell(ask_result.X, values)
        best = self.best()

        return StepResult(
            ask_result=ask_result,
            values=values,
            constraint_values=None,
            best=best,
        )

    def run(
        self,
        function: Callable[[Array], Array],
        budget: int,
        batch_size: int = 1,
    ) -> RunResult[Array]:
        """Run optimization for a fixed budget of evaluations.

        Stops early if a convergence criterion is set and satisfied.

        Parameters
        ----------
        function : Callable[[Array], Array]
            Function to evaluate.
        budget : int
            Total number of function evaluations.
        batch_size : int
            Number of points per step. Default 1.

        Returns
        -------
        RunResult[Array]
            Combined results from all steps.
        """
        steps: List[StepResult[Array]] = []
        n_evaluated = 0

        while n_evaluated < budget:
            current_batch = min(batch_size, budget - n_evaluated)
            step_result = self.step(function, current_batch)
            steps.append(step_result)
            n_evaluated += current_batch
            if self.has_converged():
                break

        return RunResult(steps=steps, best=self.best())

    def best(self) -> BestResult[Array]:
        """Return the best observed and recommended results.

        When a fitted surrogate is available, the recommended point is
        found by optimizing the surrogate mean (polishing), which
        typically finds a point closer to the true optimum than any
        observed point.

        If the polished point has a worse surrogate prediction than the
        best observed point, the recommendation falls back to the best
        observed point and a warning is issued. This guards against
        the GP posterior mean being unreliable in under-explored regions.

        Returns
        -------
        BestResult[Array]
            Best observed and surrogate-recommended points.

        Raises
        ------
        RuntimeError
            If no data has been observed.
        """
        if self._X_all is None or self._y_all is None:
            raise RuntimeError("No data observed yet. Call tell() first.")

        observed_x, observed_y = self._find_best_observed()

        # Polish: optimize surrogate mean to find recommended point
        if self._surrogate is not None and self._surrogate.is_fitted():
            polished_x = self._acq_optimizer.optimize_surrogate(
                self._surrogate, self._domain, self._bkd, self._minimize
            )
            polished_y = self._surrogate.predict(polished_x)
            observed_surr_y = self._surrogate.predict(observed_x)

            # Clamp: only use polished point if surrogate says it's
            # at least as good as the best observed point. Use a small
            # relative tolerance to avoid false triggers from
            # floating-point noise.
            polished_val = float(
                self._bkd.to_numpy(polished_y[0:1, 0:1]).flat[0]
            )
            observed_surr_val = float(
                self._bkd.to_numpy(observed_surr_y[0:1, 0:1]).flat[0]
            )
            tol = 1e-8 * (1.0 + abs(observed_surr_val))

            if self._minimize:
                polishing_improved = polished_val <= observed_surr_val + tol
            else:
                polishing_improved = polished_val >= observed_surr_val - tol

            if polishing_improved:
                recommended_x = polished_x
                recommended_y = polished_y
                recommended_std = self._surrogate.predict_std(polished_x)
            else:
                warnings.warn(
                    "Surrogate polishing returned a point worse than the "
                    "best observed point (surrogate prediction "
                    f"{polished_val:.4e} vs {observed_surr_val:.4e}). "
                    "Falling back to the best observed point. This "
                    "typically occurs when the GP posterior mean is "
                    "unreliable in under-explored regions of the domain. "
                    "Consider adding more initial samples, reducing the "
                    "domain bounds, or increasing the evaluation budget.",
                    stacklevel=2,
                )
                recommended_x = observed_x
                recommended_y = observed_surr_y
                recommended_std = self._surrogate.predict_std(observed_x)
        else:
            recommended_x = observed_x
            recommended_y = observed_y
            recommended_std = self._bkd.zeros_like(observed_y)

        return BestResult(
            observed_x=observed_x,
            observed_y=observed_y,
            recommended_x=recommended_x,
            recommended_y=recommended_y,
            recommended_std=recommended_std,
        )

    def has_converged(self) -> bool:
        """Check if convergence criterion is satisfied.

        For use in ask/tell workflows and inside run(). Returns False
        if no criterion is set or if no data has been observed yet.

        Returns
        -------
        bool
            True if converged.
        """
        if self._convergence is None:
            return False
        if self._surrogate is None or not self._surrogate.is_fitted():
            return False

        best = self.best()
        ctx = ConvergenceContext(
            step_index=self._tell_count,
            best=best,
            prev_best=self._prev_best,
            max_acquisition_value=self._last_max_acq_value,
        )
        return self._convergence.has_converged(ctx)

    def state(self) -> BOState[Array]:
        """Return serializable optimizer state.

        Returns
        -------
        BOState[Array]
            Snapshot of current state.
        """
        if self._X_all is None or self._y_all is None:
            raise RuntimeError("No data observed yet.")

        metadata: dict = {"tell_count": self._tell_count}
        if self._surrogate is not None and hasattr(self._surrogate, "hyp_list"):
            hyp_list = self._surrogate.hyp_list()  # type: ignore[union-attr]
            metadata["hyperparameters"] = self._bkd.to_numpy(
                hyp_list.get_values()
            ).tolist()

        return BOState(
            X_all=self._X_all,
            y_all=self._y_all,
            metadata=metadata,
        )

    @classmethod
    def from_state(
        cls,
        state: BOState[Array],
        domain: BODomainProtocol[Array],
        fitter: SurrogateFitterProtocol[Array],
        acquisition: AcquisitionFunctionProtocol[Array],
        acquisition_optimizer: AcquisitionOptimizerProtocol[Array],
        bkd: Backend[Array],
        surrogate_template: SurrogateProtocol[Array],
        batch_strategy: Optional[BatchStrategyProtocol[Array]] = None,
        minimize: bool = True,
        hp_schedule: Optional[HPRefitScheduleProtocol] = None,
        convergence: Optional[ConvergenceCriterionProtocol[Array]] = None,
    ) -> "BayesianOptimizer[Array]":
        """Restore optimizer from saved state.

        Parameters
        ----------
        state : BOState[Array]
            Previously saved state.
        (remaining params same as __init__)

        Returns
        -------
        BayesianOptimizer[Array]
            Restored optimizer.
        """
        # Warm-start hyperparameters if available
        if "hyperparameters" in state.metadata:
            hyp_values = state.metadata["hyperparameters"]
            if hasattr(surrogate_template, "hyp_list"):
                surrogate_template.hyp_list().set_values(  # type: ignore[union-attr]
                    bkd.array(hyp_values)
                )

        opt = cls(
            domain=domain,
            fitter=fitter,
            acquisition=acquisition,
            acquisition_optimizer=acquisition_optimizer,
            bkd=bkd,
            surrogate_template=surrogate_template,
            batch_strategy=batch_strategy,
            minimize=minimize,
            hp_schedule=hp_schedule,
            convergence=convergence,
        )
        opt.tell(state.X_all, state.y_all)

        # Restore tell_count from metadata (tell() above incremented by 1,
        # but the saved count reflects the full history)
        if "tell_count" in state.metadata:
            opt._tell_count = state.metadata["tell_count"]

        return opt

    def surrogate(self) -> Optional[SurrogateProtocol[Array]]:
        """Return the current fitted surrogate, or None if not yet fitted."""
        return self._surrogate

    def _build_context(
        self, pending_X: Optional[Array] = None
    ) -> AcquisitionContext[Array]:
        """Build acquisition context, optionally with fantasized points.

        Parameters
        ----------
        pending_X : Optional[Array]
            Pending points to fantasize, shape (nvars, n_pending).

        Returns
        -------
        AcquisitionContext[Array]
            Context for acquisition function evaluation.
        """
        assert self._surrogate is not None
        assert self._X_all is not None
        assert self._y_all is not None

        if pending_X is not None:
            # Fantasize: predict mean at pending points, refit GP
            fantasy_y = self._surrogate.predict(pending_X)
            X_aug = self._bkd.hstack([self._X_all, pending_X])
            y_aug = self._bkd.hstack([self._y_all, fantasy_y])
            surrogate = self._fitter.fit(
                self._surrogate_template, X_aug, y_aug
            )
        else:
            surrogate = self._surrogate

        best_value = self._compute_best_value()

        return AcquisitionContext(
            surrogate=surrogate,
            best_value=best_value,
            bkd=self._bkd,
            pending_X=pending_X,
            minimize=self._minimize,
        )

    def _compute_best_value(self) -> Array:
        """Compute best observed value (raw, not negated).

        For minimization: best_value = min(y).
        For maximization: best_value = max(y).

        Returns
        -------
        Array
            Best value, shape (1,).
        """
        assert self._y_all is not None
        y = self._y_all[0]  # First QoI, shape (n,)

        if self._minimize:
            best_idx = self._bkd.argmin(y)
        else:
            best_idx = self._bkd.argmax(y)

        return self._bkd.reshape(y[best_idx], (1,))

    def _find_best_observed(self):
        """Find best observed point.

        Returns
        -------
        Tuple[Array, Array]
            (best_x, best_y) with shapes (nvars, 1) and (nqoi, 1).
        """
        assert self._X_all is not None
        assert self._y_all is not None

        y = self._y_all[0]  # First QoI
        if self._minimize:
            best_idx_arr = self._bkd.argmin(y)
        else:
            best_idx_arr = self._bkd.argmax(y)

        best_idx = int(self._bkd.to_numpy(best_idx_arr).item())
        best_x = self._X_all[:, best_idx : best_idx + 1]
        best_y = self._y_all[:, best_idx : best_idx + 1]
        return best_x, best_y

    def _warm_start_template(self) -> None:
        """Set previous hyperparameters on surrogate template for warm-starting.

        If a surrogate has been fitted previously, copies its optimized
        hyperparameters to the template so the next fit starts from a
        better initial point.
        """
        if self._surrogate is None:
            return
        if not hasattr(self._surrogate, "hyp_list"):
            return
        if not hasattr(self._surrogate_template, "hyp_list"):
            return

        prev_values = self._surrogate.hyp_list().get_values()  # type: ignore[union-attr]
        self._surrogate_template.hyp_list().set_values(prev_values)  # type: ignore[union-attr]

    def _random_sample(self, n: int) -> Array:
        """Generate initial samples within domain bounds.

        Uses the acquisition optimizer's candidate generator for
        consistent space-filling behavior.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        Array
            Samples, shape (nvars, n).
        """
        return self._acq_optimizer.generate_candidates(
            n, self._domain.bounds(), self._bkd
        )
