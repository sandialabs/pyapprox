"""State and result dataclasses for Bayesian optimization.

Provides serializable data containers for ask/tell workflow results,
best-so-far tracking, and optimizer state snapshots.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional

from pyapprox.util.backends.protocols import Array


@dataclass
class AskResult(Generic[Array]):
    """Result from BayesianOptimizer.ask().

    Attributes
    ----------
    X : Array
        Points to evaluate, shape (nvars, batch_size).
    fidelity : Optional[Array]
        Fidelity parameters (None for single-fidelity).
    """

    X: Array
    fidelity: Optional[Array] = None


@dataclass
class BestResult(Generic[Array]):
    """Best observed and recommended points.

    Attributes
    ----------
    observed_x : Array
        Best observed input, shape (nvars, 1).
    observed_y : Array
        Best observed output, shape (nqoi, 1).
    recommended_x : Array
        Recommended input (surrogate prediction), shape (nvars, 1).
    recommended_y : Array
        Recommended output (surrogate prediction), shape (nqoi, 1).
    recommended_std : Array
        Uncertainty at recommended point, shape (nqoi, 1).
    """

    observed_x: Array
    observed_y: Array
    recommended_x: Array
    recommended_y: Array
    recommended_std: Array


@dataclass
class StepResult(Generic[Array]):
    """Result from a single BayesianOptimizer.step().

    Attributes
    ----------
    ask_result : AskResult[Array]
        Points that were evaluated.
    values : Array
        Observed function values, shape (nqoi, batch_size).
    constraint_values : Optional[Array]
        Constraint values if applicable.
    best : BestResult[Array]
        Best result after this step.
    """

    ask_result: AskResult[Array]
    values: Array
    constraint_values: Optional[Array]
    best: BestResult[Array]


@dataclass
class RunResult(Generic[Array]):
    """Result from BayesianOptimizer.run().

    Attributes
    ----------
    steps : List[StepResult[Array]]
        Results from each step.
    best : BestResult[Array]
        Best result across all steps.
    """

    steps: List[StepResult[Array]]
    best: BestResult[Array]


@dataclass
class ConvergenceContext(Generic[Array]):
    """Context passed to convergence criteria.

    Attributes
    ----------
    step_index : int
        Tell count (0-based).
    best : BestResult[Array]
        Current best (with polished recommended_x).
    prev_best : Optional[BestResult[Array]]
        Previous best (None on first step).
    max_acquisition_value : Optional[float]
        Maximum acquisition value at the chosen point.
    step_result : Optional[StepResult[Array]]
        Only available inside run().
    """

    step_index: int
    best: BestResult[Array]
    prev_best: Optional[BestResult[Array]]
    max_acquisition_value: Optional[float]
    step_result: Optional[StepResult[Array]] = None


@dataclass
class BOState(Generic[Array]):
    """Serializable snapshot of BayesianOptimizer state.

    Attributes
    ----------
    X_all : Array
        All observed inputs, shape (nvars, n_total).
    y_all : Array
        All observed outputs, shape (nqoi, n_total).
    metadata : Dict
        Additional state (e.g., hyperparameter values for warm-start).
    """

    X_all: Array
    y_all: Array
    metadata: Dict[str, Any] = field(default_factory=dict)
