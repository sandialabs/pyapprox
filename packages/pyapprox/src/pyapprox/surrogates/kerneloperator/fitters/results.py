"""Result classes for kernel operator fitting operations."""

from __future__ import annotations

from typing import Generic, Optional

from pyapprox.surrogates.kerneloperator.surrogate import (
    KernelOperatorSurrogate,
)
from pyapprox.util.backends.protocols import Array, Backend


class KernelOperatorFitResult(Generic[Array]):
    """Result from kernel operator fitting without hyperparameter optimization.

    Parameters
    ----------
    surrogate : KernelOperatorSurrogate[Array]
        The fitted surrogate.
    neg_log_marginal_likelihood : Array
        Scalar NLL at the fitted state.
    """

    def __init__(
        self,
        surrogate: KernelOperatorSurrogate[Array],
        neg_log_marginal_likelihood: Array,
    ) -> None:
        self._surrogate = surrogate
        self._nll = neg_log_marginal_likelihood

    def surrogate(self) -> KernelOperatorSurrogate[Array]:
        """Return the fitted surrogate."""
        return self._surrogate

    def neg_log_marginal_likelihood(self) -> Array:
        """Return the negative log marginal likelihood at fitted state."""
        return self._nll

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()


class KernelOperatorOptimizedFitResult(Generic[Array]):
    """Result from kernel operator fitting with hyperparameter optimization.

    Parameters
    ----------
    surrogate : KernelOperatorSurrogate[Array]
        The fitted surrogate with optimized hyperparameters.
    neg_log_marginal_likelihood : Array
        Scalar NLL at optimal hyperparameters.
    initial_hyperparameters : Array
        Hyperparameter values before optimization.
    optimized_hyperparameters : Array
        Hyperparameter values after optimization.
    optimization_result : object or None
        Raw optimizer result, or None if no optimization was performed.
    """

    def __init__(
        self,
        surrogate: KernelOperatorSurrogate[Array],
        neg_log_marginal_likelihood: Array,
        initial_hyperparameters: Array,
        optimized_hyperparameters: Array,
        optimization_result: Optional[object],
    ) -> None:
        self._surrogate = surrogate
        self._nll = neg_log_marginal_likelihood
        self._initial_hyps = initial_hyperparameters
        self._optimized_hyps = optimized_hyperparameters
        self._opt_result = optimization_result

    def surrogate(self) -> KernelOperatorSurrogate[Array]:
        """Return the fitted surrogate with optimized hyperparameters."""
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
