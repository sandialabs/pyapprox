r"""Weighted joint moments accumulated over batches.

For stacked outputs :math:`\chi_k` with weights :math:`\omega_k`,

.. math::

    \mu_\chi = \sum_k \omega_k \chi_k, \qquad
    \Gamma_{\chi\chi} = \sum_k \omega_k (\chi_k - \mu_\chi)(\chi_k - \mu_\chi)^\top .

The sums are kept about a shift :math:`c`, the first sample seen:
:math:`W = \sum \omega`, :math:`s = \sum \omega (\chi - c)` and
:math:`S = \sum \omega (\chi - c)(\chi - c)^\top`, so that
:math:`\mu_\chi = c + s / W` and :math:`\Gamma_{\chi\chi} = S - s s^\top / W`.
Shifting avoids the cancellation of the raw form
:math:`\sum \omega \chi \chi^\top - \mu \mu^\top` when the mean is large
relative to the spread, and unlike per-batch centring it never divides by
a batch's total weight, which a rule with negative weights can make zero.
Batches are combined pairwise, as a binary counter, which keeps rounding
error growing with the logarithm of the number of batches.
"""

from dataclasses import dataclass
from typing import Generic, Optional

from pyapprox.interface.functions.joint import JointOutputs
from pyapprox.probability.moments.blocks import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class _Partial(Generic[Array]):
    """Shifted sums over a run of batches, at a pairwise-merge level."""

    level: int
    wsum: Array
    w2sum: Array
    s: Array
    ss: Array
    n: int

    def merged(self, other: "_Partial[Array]") -> "_Partial[Array]":
        return _Partial(
            self.level + 1,
            self.wsum + other.wsum,
            self.w2sum + other.w2sum,
            self.s + other.s,
            self.ss + other.ss,
            self.n + other.n,
        )


class _ShiftedSums(Generic[Array]):
    """The streaming state shared by both accumulators."""

    def __init__(self, bkd: Backend[Array], tol: float) -> None:
        self._bkd = bkd
        self._tol = tol
        self._shift: Optional[Array] = None
        self._sizes: Optional[tuple[int, ...]] = None
        self._nobs = 0
        self._stack: list[_Partial[Array]] = []

    def update(self, weights: Array, outputs: JointOutputs[Array]) -> None:
        bkd = self._bkd
        nsamples = outputs.nsamples()
        if weights.ndim != 2 or tuple(weights.shape) != (1, nsamples):
            raise ValueError(
                f"weights must have shape (1, {nsamples}), got {tuple(weights.shape)}"
            )
        sizes = tuple(int(t.shape[0]) for t in outputs.targets)
        nobs = int(outputs.observations.shape[0])
        if self._sizes is None:
            self._sizes, self._nobs = sizes, nobs
        elif (sizes, nobs) != (self._sizes, self._nobs):
            raise ValueError(
                f"batch has blocks {sizes} and {nobs} observations, but earlier "
                f"batches had {self._sizes} and {self._nobs}"
            )
        stacked = bkd.vstack(list(outputs.targets) + [outputs.observations])
        if self._shift is None:
            self._shift = stacked[:, :1]
        dev = stacked - self._shift
        partial = _Partial(
            0,
            bkd.sum(weights),
            bkd.sum(weights**2),
            bkd.dot(dev, weights.T),
            bkd.dot(dev * weights, dev.T),
            nsamples,
        )
        while self._stack and self._stack[-1].level == partial.level:
            partial = self._stack.pop().merged(partial)
        self._stack.append(partial)

    def moments(self, unbiased: bool) -> DenseBlocks[Array]:
        if not self._stack or self._shift is None or self._sizes is None:
            raise ValueError("no batches have been added")
        total = self._stack[0]
        for partial in self._stack[1:]:
            total = total.merged(partial)
        bkd = self._bkd
        if abs(bkd.to_float(total.wsum) - 1.0) > self._tol:
            raise ValueError(
                f"weights sum to {bkd.to_float(total.wsum)}, not 1 within {self._tol}"
            )
        mean = self._shift + total.s / total.wsum
        cov = total.ss - bkd.dot(total.s, total.s.T) / total.wsum
        if unbiased:
            if bkd.to_float(total.w2sum) >= 1.0:
                raise ValueError("the unbiased correction needs two or more samples")
            cov = cov / (1.0 - total.w2sum)
        return DenseBlocks(mean, cov, self._sizes, self._nobs, bkd, total.n)


class WeightedAccumulator(Generic[Array]):
    """The quadrature formula for the joint moments.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    tol : float
        Allowed ``|sum(w) - 1|`` at ``finalize``. Default 1e-8.
    """

    def __init__(self, bkd: Backend[Array], tol: float = 1e-8) -> None:
        self._sums = _ShiftedSums(bkd, tol)

    def update(self, weights: Array, outputs: JointOutputs[Array]) -> None:
        """Add one batch. Weights shape: (1, nsamples)"""
        self._sums.update(weights, outputs)

    def finalize(self) -> DenseBlocks[Array]:
        """Blocks from every batch added so far."""
        return self._sums.moments(unbiased=False)


class UnbiasedMCAccumulator(Generic[Array]):
    """The quadrature formula with the unbiased sample-covariance factor.

    Scales the covariance by ``1 / (1 - sum(w**2))``, which is
    ``N / (N - 1)`` for ``N`` equal weights, matching the usual unbiased
    sample covariance. The mean is unchanged.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    tol : float
        Allowed ``|sum(w) - 1|`` at ``finalize``. Default 1e-8.
    """

    def __init__(self, bkd: Backend[Array], tol: float = 1e-8) -> None:
        self._sums = _ShiftedSums(bkd, tol)

    def update(self, weights: Array, outputs: JointOutputs[Array]) -> None:
        """Add one batch. Weights shape: (1, nsamples)"""
        self._sums.update(weights, outputs)

    def finalize(self) -> DenseBlocks[Array]:
        """Blocks from every batch added so far."""
        return self._sums.moments(unbiased=True)
