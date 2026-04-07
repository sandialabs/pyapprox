"""Result classes for GP fitting operations.

All attributes accessed via methods per CLAUDE.md conventions.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class PredictiveGPSurrogateProtocol(Protocol, Generic[Array]):
    """Minimal protocol for fitted GP surrogates that can predict."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def __call__(self, X: Array) -> Array:
        """Predict posterior mean at X."""
        ...

    def predict(self, X: Array) -> Array:
        """Predict posterior mean at X."""
        ...

    def predict_std(self, X: Array) -> Array:
        """Predict posterior standard deviation at X."""
        ...


S = TypeVar("S", bound=PredictiveGPSurrogateProtocol)  # type: ignore[type-arg]


class GPFitResult(Generic[Array, S]):
    """Result from GP fitting without hyperparameter optimization.

    Returned by GPDirectFitter (fixed hyperparameters, Cholesky + alpha only).

    All attributes are accessed via methods per CLAUDE.md convention.

    Parameters
    ----------
    surrogate : S
        The fitted GP instance.
    neg_log_marginal_likelihood : Array
        Scalar negative log marginal likelihood at the fitted state.
    """

    def __init__(
        self,
        surrogate: S,
        neg_log_marginal_likelihood: Array,
    ):
        self._surrogate = surrogate
        self._nll = neg_log_marginal_likelihood

    def surrogate(self) -> S:
        """Return the fitted GP surrogate."""
        return self._surrogate

    def neg_log_marginal_likelihood(self) -> Array:
        """Return the negative log marginal likelihood at fitted state."""
        return self._nll

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()

    def __call__(self, X: Array) -> Array:
        """Predict posterior mean at X (delegates to surrogate).

        Parameters
        ----------
        X : Array
            Input locations.

        Returns
        -------
        Array
            Posterior mean predictions.
        """
        result: Array = self._surrogate(X)
        return result

    def predict(self, X: Array) -> Array:
        """Predict posterior mean at X (delegates to surrogate).

        Parameters
        ----------
        X : Array
            Input locations.

        Returns
        -------
        Array
            Posterior mean predictions.
        """
        return self._surrogate.predict(X)

    def predict_std(self, X: Array) -> Array:
        """Predict posterior standard deviation at X (delegates to surrogate).

        Parameters
        ----------
        X : Array
            Input locations.

        Returns
        -------
        Array
            Posterior standard deviation.
        """
        return self._surrogate.predict_std(X)

    def __repr__(self) -> str:
        return (
            f"GPFitResult(nll={self._nll:.4f}, "
            f"surrogate={type(self._surrogate).__name__})"
        )


class GPOptimizedFitResult(Generic[Array, S]):
    """Result from GP fitting with hyperparameter optimization.

    Returned by GPHyperparameterFitter.

    All attributes are accessed via methods per CLAUDE.md convention.

    Parameters
    ----------
    surrogate : S
        The fitted GP with optimized hyperparameters.
    neg_log_marginal_likelihood : Array
        Scalar NLL at the optimal hyperparameters.
    initial_hyperparameters : Array
        Hyperparameter values before optimization.
    optimized_hyperparameters : Array
        Hyperparameter values after optimization.
    optimization_result : object or None
        The raw result from the optimizer, or None if no optimization
        was performed (all hyperparameters inactive).
    """

    def __init__(
        self,
        surrogate: S,
        neg_log_marginal_likelihood: Array,
        initial_hyperparameters: Array,
        optimized_hyperparameters: Array,
        optimization_result: Optional[object],
    ):
        self._surrogate = surrogate
        self._nll = neg_log_marginal_likelihood
        self._initial_hyps = initial_hyperparameters
        self._optimized_hyps = optimized_hyperparameters
        self._opt_result = optimization_result

    def surrogate(self) -> S:
        """Return the fitted GP surrogate with optimized hyperparameters."""
        return self._surrogate

    def neg_log_marginal_likelihood(self) -> Array:
        """Return the NLL at optimal hyperparameters."""
        return self._nll

    def initial_hyperparameters(self) -> Array:
        """Return hyperparameter values before optimization."""
        return self._initial_hyps

    def optimized_hyperparameters(self) -> Array:
        """Return hyperparameter values after optimization."""
        return self._optimized_hyps

    def optimization_result(self) -> Optional[object]:
        """Return the raw optimizer result, or None if no optimization."""
        return self._opt_result

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()

    def __call__(self, X: Array) -> Array:
        """Predict posterior mean at X (delegates to surrogate).

        Parameters
        ----------
        X : Array
            Input locations.

        Returns
        -------
        Array
            Posterior mean predictions.
        """
        result: Array = self._surrogate(X)
        return result

    def predict(self, X: Array) -> Array:
        """Predict posterior mean at X (delegates to surrogate).

        Parameters
        ----------
        X : Array
            Input locations.

        Returns
        -------
        Array
            Posterior mean predictions.
        """
        return self._surrogate.predict(X)

    def predict_std(self, X: Array) -> Array:
        """Predict posterior standard deviation at X (delegates to surrogate).

        Parameters
        ----------
        X : Array
            Input locations.

        Returns
        -------
        Array
            Posterior standard deviation.
        """
        return self._surrogate.predict_std(X)

    def __repr__(self) -> str:
        opt_str = "optimized" if self._opt_result is not None else "no optimization"
        return (
            f"GPOptimizedFitResult(nll={self._nll:.4f}, "
            f"{opt_str}, "
            f"surrogate={type(self._surrogate).__name__})"
        )
