"""Protocols for Bayesian optimization components.

Defines interfaces for surrogates, acquisition functions, domains,
fitters, and batch selection strategies.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SurrogateProtocol(Protocol, Generic[Array]):
    """Protocol for surrogate models used in Bayesian optimization.

    ExactGaussianProcess satisfies this protocol as-is.
    """

    def predict(self, X: Array) -> Array:
        """Predict posterior mean at X.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n).

        Returns
        -------
        Array
            Posterior mean, shape (nqoi, n).
        """
        ...

    def predict_std(self, X: Array) -> Array:
        """Predict posterior standard deviation at X.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n).

        Returns
        -------
        Array
            Posterior std, shape (nqoi, n).
        """
        ...

    def predict_covariance(self, X: Array) -> Array:
        """Predict posterior covariance at X.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n).

        Returns
        -------
        Array
            Posterior covariance, shape (n, n).
        """
        ...

    def is_fitted(self) -> bool:
        """Return True if the surrogate has been fitted to data."""
        ...


@runtime_checkable
class SurrogateFitterProtocol(Protocol, Generic[Array]):
    """Protocol for fitting surrogates to data."""

    def fit(
        self,
        surrogate: SurrogateProtocol[Array],
        X: Array,
        y: Array,
    ) -> SurrogateProtocol[Array]:
        """Fit surrogate to data and return fitted surrogate.

        Parameters
        ----------
        surrogate : SurrogateProtocol[Array]
            Surrogate model template (may be unfitted).
        X : Array
            Training inputs, shape (nvars, n_train).
        y : Array
            Training outputs, shape (nqoi, n_train).

        Returns
        -------
        SurrogateProtocol[Array]
            Fitted surrogate (may be a new instance).
        """
        ...


@dataclass
class AcquisitionContext(Generic[Array]):
    """Context passed to acquisition functions.

    Attributes
    ----------
    surrogate : SurrogateProtocol[Array]
        Fitted surrogate model.
    best_value : Array
        Best observed value, shape (1,). Raw value:
        min(y_observed) if minimizing, max(y_observed) if maximizing.
    bkd : Backend[Array]
        Computational backend.
    pending_X : Optional[Array]
        Pending evaluation points, shape (nvars, n_pending) or None.
    minimize : bool
        True if minimizing the objective.
    """

    surrogate: SurrogateProtocol[Array]
    best_value: Array
    bkd: Backend[Array]
    pending_X: Optional[Array]
    minimize: bool


@runtime_checkable
class AcquisitionFunctionProtocol(Protocol, Generic[Array]):
    """Protocol for acquisition functions.

    Acquisition functions evaluate candidate points and return values
    where higher is always better (even for minimization problems).
    """

    def evaluate(self, X: Array, ctx: AcquisitionContext[Array]) -> Array:
        """Evaluate acquisition function at candidate points.

        Parameters
        ----------
        X : Array
            Candidate points, shape (nvars, n).
        ctx : AcquisitionContext[Array]
            Context with surrogate, best value, etc.

        Returns
        -------
        Array
            Acquisition values, shape (n,). Higher is better.
        """
        ...


@runtime_checkable
class BODomainProtocol(Protocol, Generic[Array]):
    """Protocol for optimization domains."""

    def bounds(self) -> Array:
        """Return variable bounds, shape (nvars, 2).

        Each row is [lower, upper] for one variable.
        """
        ...

    def nvars(self) -> int:
        """Return number of variables."""
        ...

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        ...

    def contains(self, X: Array) -> Array:
        """Check if points are in the domain.

        Parameters
        ----------
        X : Array
            Points to check, shape (nvars, n).

        Returns
        -------
        Array
            Boolean array, shape (n,). True if point is in domain.
        """
        ...


@runtime_checkable
class BatchStrategyProtocol(Protocol, Generic[Array]):
    """Protocol for batch point selection strategies."""

    def select_batch(
        self,
        batch_size: int,
        acquisition: AcquisitionFunctionProtocol[Array],
        context_factory: Callable[[Optional[Array]], AcquisitionContext[Array]],
        acquisition_optimizer: "AcquisitionOptimizerProtocol[Array]",
        domain: BODomainProtocol[Array],
    ) -> Array:
        """Select a batch of points for evaluation.

        Parameters
        ----------
        batch_size : int
            Number of points to select.
        acquisition : AcquisitionFunctionProtocol[Array]
            Acquisition function to maximize.
        context_factory : Callable
            Creates AcquisitionContext given pending_X (or None).
        acquisition_optimizer : AcquisitionOptimizerProtocol[Array]
            Optimizer for acquisition function maximization.
        domain : BODomainProtocol[Array]
            Search domain.

        Returns
        -------
        Array
            Selected batch points, shape (nvars, batch_size).
        """
        ...


@runtime_checkable
class AcquisitionOptimizerProtocol(Protocol, Generic[Array]):
    """Protocol for acquisition function optimizers."""

    def maximize(
        self,
        acquisition: AcquisitionFunctionProtocol[Array],
        ctx: AcquisitionContext[Array],
        domain: BODomainProtocol[Array],
    ) -> Array:
        """Find the point that maximizes the acquisition function.

        Parameters
        ----------
        acquisition : AcquisitionFunctionProtocol[Array]
            Acquisition function to maximize.
        ctx : AcquisitionContext[Array]
            Acquisition context.
        domain : BODomainProtocol[Array]
            Search domain.

        Returns
        -------
        Array
            Best point found, shape (nvars, 1).
        """
        ...
