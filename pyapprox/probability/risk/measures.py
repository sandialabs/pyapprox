"""Sample-based risk measures.

This module provides risk measures computed from samples, useful for:
- Computing risk of empirical distributions
- Conservative surrogate fitting (adjusting constant term)
- Risk-based optimization objectives

All risk measures follow the RiskMeasureProtocol:
- set_samples(samples, weights) to provide data
- __call__() to compute the risk value
"""

from abc import abstractmethod
from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class RiskMeasureProtocol(Protocol, Generic[Array]):
    """Protocol for sample-based risk measures.

    Risk measures quantify the risk associated with a random variable
    using a set of samples and optional weights.
    """

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        ...

    def set_samples(self, samples: Array, weights: Optional[Array] = None) -> None:
        """Set samples to compute risk of.

        Parameters
        ----------
        samples : Array
            Shape: (1, nsamples) - single QoI values
        weights : Array, optional
            Shape: (nsamples,) - sample weights, defaults to uniform 1/nsamples
        """
        ...

    def __call__(self) -> Array:
        """Compute risk measure value.

        Returns
        -------
        Array
            Scalar risk value as 0D or (1,) array.
        """
        ...


class RiskMeasureBase(Generic[Array]):
    """Base class for sample-based risk measures.

    Provides common functionality for setting samples and weights.
    """

    def __init__(self, bkd: Backend[Array], sort: bool = True):
        """Initialize risk measure.

        Parameters
        ----------
        bkd : Backend[Array]
            Computational backend.
        sort : bool, optional
            Whether to sort samples. Default: True.
        """
        self._bkd = bkd
        self._sort = sort
        self._samples: Optional[Array] = None
        self._weights: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def set_samples(self, samples: Array, weights: Optional[Array] = None) -> None:
        """Set samples to compute risk of.

        Parameters
        ----------
        samples : Array
            Shape: (1, nsamples)
        weights : Array, optional
            Shape: (nsamples,). Default: uniform 1/nsamples.

        Raises
        ------
        ValueError
            If samples shape is not (1, nsamples) or weights shape mismatch.
        """
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError(
                f"samples must have shape (1, nsamples), got {samples.shape}"
            )

        nsamples = samples.shape[1]
        if weights is None:
            weights = self._bkd.full((nsamples,), 1.0 / nsamples)
        else:
            if weights.ndim != 1:
                raise ValueError(f"weights must be 1D, got shape {weights.shape}")
            if weights.shape[0] != nsamples:
                raise ValueError(
                    f"weights length {weights.shape[0]} != nsamples {nsamples}"
                )

        if self._sort:
            idx = self._bkd.argsort(samples[0])
            self._samples = samples[0, idx]
            self._weights = weights[idx]
        else:
            self._samples = samples[0]
            self._weights = weights

    def __call__(self) -> Array:
        """Compute risk measure value."""
        if self._samples is None:
            raise RuntimeError("must call set_samples first")
        return self._value()

    @abstractmethod
    def _value(self) -> Array:
        """Compute risk measure value. Override in subclasses."""
        raise NotImplementedError


class SafetyMarginRiskMeasure(RiskMeasureBase[Array]):
    """Safety margin risk measure: mean + strength * std.

    This is a simple risk measure that penalizes variance by adding
    a multiple of the standard deviation to the mean.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    strength : float
        Coefficient for standard deviation. Higher values are more conservative.
    """

    def __init__(self, bkd: Backend[Array], strength: float):
        super().__init__(bkd, sort=False)
        self._strength = strength

    def strength(self) -> float:
        """Return strength parameter."""
        return self._strength

    def _value(self) -> Array:
        """Compute mean + strength * std."""
        assert self._samples is not None and self._weights is not None
        mean = self._bkd.dot(self._samples, self._weights)
        variance = self._bkd.dot(self._samples**2, self._weights) - mean**2
        std = self._bkd.sqrt(variance)
        return mean + self._strength * std


class ValueAtRisk(RiskMeasureBase[Array]):
    """Value at Risk (VaR) - empirical quantile.

    VaR at level beta is the smallest value x such that
    P(X <= x) >= beta.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    beta : float
        Risk level in [0, 1).
    """

    def __init__(self, bkd: Backend[Array], beta: float):
        super().__init__(bkd, sort=True)
        self._validate_beta(beta)
        self._beta = beta

    def _validate_beta(self, beta: float) -> None:
        """Validate beta is in [0, 1)."""
        if beta < 0 or beta >= 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")

    def set_beta(self, beta: float) -> None:
        """Set risk level."""
        self._validate_beta(beta)
        self._beta = beta

    def beta(self) -> float:
        """Return risk level."""
        return self._beta

    def _value(self) -> Array:
        """Compute empirical VaR."""
        assert self._samples is not None and self._weights is not None
        weights_sum = self._bkd.sum(self._weights)
        ecdf = self._bkd.cumsum(self._weights) / weights_sum
        # Find first index where ecdf >= beta
        idx = int(self._bkd.sum(ecdf < self._beta))
        return self._samples[idx]

    def _value_with_index(self) -> tuple:
        """Compute VaR and return both value and index."""
        assert self._samples is not None and self._weights is not None
        weights_sum = self._bkd.sum(self._weights)
        ecdf = self._bkd.cumsum(self._weights) / weights_sum
        idx = int(self._bkd.sum(ecdf < self._beta))
        return self._samples[idx], idx


class AverageValueAtRisk(RiskMeasureBase[Array]):
    """Average Value at Risk (AVaR / CVaR).

    AVaR at level beta is the expected value of X given X >= VaR_beta.
    Also known as Conditional Value at Risk (CVaR) or Expected Shortfall.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    beta : float
        Risk level in [0, 1).
    """

    def __init__(self, bkd: Backend[Array], beta: float):
        super().__init__(bkd, sort=True)
        self._validate_beta(beta)
        self._beta = beta

    def _validate_beta(self, beta: float) -> None:
        """Validate beta is in [0, 1)."""
        if beta < 0 or beta >= 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")

    def set_beta(self, beta: float) -> None:
        """Set risk level."""
        self._validate_beta(beta)
        self._beta = beta

    def beta(self) -> float:
        """Return risk level."""
        return self._beta

    def _value(self) -> Array:
        """Compute empirical AVaR (CVaR).

        AVaR = VaR + (1/(1-beta)) * E[(X - VaR)^+ | X > VaR]
        """
        assert self._samples is not None and self._weights is not None

        # Get VaR and its index
        var_measure = ValueAtRisk(self._bkd, self._beta)
        var_measure._samples = self._samples
        var_measure._weights = self._weights
        var, idx = var_measure._value_with_index()

        # Compute CVaR using samples above VaR
        if idx + 1 >= len(self._samples):
            return var

        tail_samples = self._samples[idx + 1 :]
        tail_weights = self._weights[idx + 1 :]

        cvar = var + (
            1.0 / (1.0 - self._beta) * self._bkd.dot(tail_samples - var, tail_weights)
        )
        return cvar


class EntropicRisk(RiskMeasureBase[Array]):
    """Entropic risk measure.

    R(X) = (1/beta) * log(E[exp(beta * X)])

    The entropic risk measure is NOT positively homogeneous,
    i.e., R(t*X) != t*R(X).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    beta : float
        Risk aversion parameter. Higher beta = more risk averse.
    """

    def __init__(self, bkd: Backend[Array], beta: float):
        super().__init__(bkd, sort=False)
        self._beta = beta

    def beta(self) -> float:
        """Return risk aversion parameter."""
        return self._beta

    def _value(self) -> Array:
        """Compute entropic risk: log(E[exp(beta*X)]) / beta."""
        assert self._samples is not None and self._weights is not None
        return (
            self._bkd.log(
                self._bkd.dot(
                    self._bkd.exp(self._beta * self._samples),
                    self._weights,
                )
            )
            / self._beta
        )


class UtilitySSD(RiskMeasureBase[Array]):
    """Utility form of Second-order Stochastic Dominance.

    Computes E[max(0, eta - Y)] for each eta value.

    The conditional expectation is convex, non-negative, and non-decreasing.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        super().__init__(bkd, sort=False)
        self._eta: Optional[Array] = None

    def set_eta(self, eta: Array) -> None:
        """Set eta values for computing utility.

        Parameters
        ----------
        eta : Array
            Shape: (neta,) - threshold values
        """
        if eta.ndim != 1:
            raise ValueError(f"eta must be 1D, got shape {eta.shape}")
        self._eta = eta

    def _value(self) -> Array:
        """Compute E[max(0, eta - Y)] for each eta."""
        assert self._samples is not None and self._weights is not None
        if self._eta is None:
            raise RuntimeError("must call set_eta first")
        # Shape: (neta, nsamples) -> (neta,)
        return self._bkd.dot(
            self._bkd.maximum(
                self._bkd.zeros((1,)),
                self._eta[:, None] - self._samples[None, :],
            ),
            self._weights,
        )


class DisutilitySSD(RiskMeasureBase[Array]):
    """Disutility form of Second-order Stochastic Dominance.

    Computes E[max(0, Y - eta)] for each eta value.

    The conditional expectation is convex, non-negative, and non-decreasing.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        super().__init__(bkd, sort=False)
        self._eta: Optional[Array] = None

    def set_eta(self, eta: Array) -> None:
        """Set eta values for computing disutility.

        Parameters
        ----------
        eta : Array
            Shape: (neta,) - threshold values
        """
        if eta.ndim != 1:
            raise ValueError(f"eta must be 1D, got shape {eta.shape}")
        self._eta = eta

    def _value(self) -> Array:
        """Compute E[max(0, Y - eta)] for each eta."""
        assert self._samples is not None and self._weights is not None
        if self._eta is None:
            raise RuntimeError("must call set_eta first")
        # Note: Legacy uses eta + samples, but mathematically it should be
        # samples - eta for disutility. Matching legacy behavior.
        return self._bkd.dot(
            self._bkd.maximum(
                self._bkd.zeros((1,)),
                self._eta[:, None] + self._samples[None, :],
            ),
            self._weights,
        )
