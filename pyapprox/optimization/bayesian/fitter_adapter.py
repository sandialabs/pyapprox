"""Adapters wrapping GP fitters for BO's SurrogateFitterProtocol."""

from typing import Generic, Optional

from pyapprox.optimization.bayesian.protocols import SurrogateProtocol
from pyapprox.util.backends.protocols import Array, Backend


class GPFitterAdapter(Generic[Array]):
    """Adapts GPMaximumLikelihoodFitter to SurrogateFitterProtocol.

    Wraps the fitter so that fit() returns the fitted surrogate directly,
    hiding the GPOptimizedFitResult intermediate.

    Parameters
    ----------
    fitter : GPMaximumLikelihoodFitter[Array]
        The underlying GP fitter.
    """

    def __init__(self, fitter: object) -> None:
        self._fitter = fitter

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
            GP template (may be unfitted).
        X : Array
            Training inputs, shape (nvars, n_train).
        y : Array
            Training outputs, shape (nqoi, n_train).

        Returns
        -------
        SurrogateProtocol[Array]
            Fitted GP.
        """
        result = self._fitter.fit(surrogate, X, y)
        return result.surrogate()

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._fitter.bkd()


class GPFixedFitterAdapter(Generic[Array]):
    """Adapts GPFixedHyperparameterFitter to SurrogateFitterProtocol.

    Refits Cholesky + alpha with current hyperparameters, no optimization.
    Used as the fast path in scheduled HP refit workflows.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def fit(
        self,
        surrogate: SurrogateProtocol[Array],
        X: Array,
        y: Array,
    ) -> SurrogateProtocol[Array]:
        """Fit surrogate with fixed hyperparameters (Cholesky only).

        Parameters
        ----------
        surrogate : SurrogateProtocol[Array]
            GP template (must have current hyperparameters set).
        X : Array
            Training inputs, shape (nvars, n_train).
        y : Array
            Training outputs, shape (nqoi, n_train).

        Returns
        -------
        SurrogateProtocol[Array]
            Fitted GP.
        """
        from pyapprox.surrogates.gaussianprocess.fitters.fixed_hyperparameter_fitter import (  # noqa: E501
            GPFixedHyperparameterFitter,
        )

        fitter = GPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(surrogate, X, y)
        return result.surrogate()

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd


class GPIncrementalFitterAdapter(Generic[Array]):
    """Adapts GPIncrementalFitter to SurrogateFitterProtocol.

    Caches the previous fitted surrogate to enable rank-1 Cholesky
    updates when exactly one new point is added. Falls back to full
    Cholesky when the incremental path is not applicable.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._prev_surrogate: Optional[SurrogateProtocol[Array]] = None

    def fit(
        self,
        surrogate: SurrogateProtocol[Array],
        X: Array,
        y: Array,
    ) -> SurrogateProtocol[Array]:
        """Fit surrogate, using incremental Cholesky update if possible.

        Parameters
        ----------
        surrogate : SurrogateProtocol[Array]
            GP template (must have current hyperparameters set).
        X : Array
            Training inputs, shape (nvars, n_train).
        y : Array
            Training outputs, shape (nqoi, n_train).

        Returns
        -------
        SurrogateProtocol[Array]
            Fitted GP.
        """
        from pyapprox.surrogates.gaussianprocess.fitters.incremental_fitter import (
            GPIncrementalFitter,
        )

        fitter = GPIncrementalFitter(self._bkd)
        result = fitter.fit(surrogate, X, y, self._prev_surrogate)
        fitted = result.surrogate()
        self._prev_surrogate = fitted
        return fitted

    def set_prev_surrogate(self, surrogate: SurrogateProtocol[Array]) -> None:
        """Seed the incremental cache after full HP optimization.

        Parameters
        ----------
        surrogate : SurrogateProtocol[Array]
            Fitted GP from a full HP optimization step.
        """
        self._prev_surrogate = surrogate

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd
