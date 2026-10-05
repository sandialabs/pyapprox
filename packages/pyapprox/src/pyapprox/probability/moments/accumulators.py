r"""Weighted joint moments accumulated over batches.

For stacked outputs :math:`\chi_k` with weights :math:`\omega_k`,

.. math::

    \mu_\chi = \sum_k \omega_k \chi_k, \qquad
    \Gamma_{\chi\chi} = \sum_k \omega_k (\chi_k - \mu_\chi)(\chi_k - \mu_\chi)^\top .

Each batch keeps its sums about its own center
:math:`c = \sum |\omega| \chi / \sum |\omega|`:
:math:`W = \sum \omega`, :math:`s = \sum \omega (\chi - c)` and
:math:`S = \sum \omega (\chi - c)(\chi - c)^\top`, so that
:math:`\mu_\chi = c + s / W` and :math:`\Gamma_{\chi\chi} = S - s s^\top / W`.
For positive weights :math:`c` is the weighted mean, so :math:`s = 0` and
one batch is as accurate as the two-pass formula; the raw form
:math:`\sum \omega \chi \chi^\top - \mu \mu^\top`, or a center far from the
mean, cancels catastrophically instead. The absolute weights keep the
center defined when negative weights make :math:`W` small.

Two batches merge by moving both to a common center :math:`c'` with the
exact identities :math:`s' = s + W d` and
:math:`S' = S + d s^\top + s d^\top + W d d^\top`, :math:`d = c - c'`, and
adding. Nothing divides by a batch's total weight, which a rule with
negative weights can make zero. Batches are combined pairwise, as a binary
counter, so rounding error grows with the logarithm of the number of
batches.
"""

from dataclasses import dataclass
from typing import Generic, Optional

from pyapprox.interface.functions.joint import JointOutputs
from pyapprox.probability.moments.blocks import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class _Partial(Generic[Array]):
    """Sums about ``center`` over a run of batches, at a merge level."""

    level: int
    center: Array
    abswsum: Array
    wsum: Array
    w2sum: Array
    s: Array
    ss: Array
    n: int

    def _moved(self, center: Array) -> tuple[Array, Array]:
        """``s`` and ``ss`` re-expressed about ``center``, exactly."""
        d = self.center - center
        ds = d @ self.s.T
        return (
            self.s + self.wsum * d,
            self.ss + ds + ds.T + self.wsum * (d @ d.T),
        )

    def merged(self, other: "_Partial[Array]") -> "_Partial[Array]":
        abswsum = self.abswsum + other.abswsum
        center = (self.abswsum * self.center + other.abswsum * other.center) / abswsum
        s1, ss1 = self._moved(center)
        s2, ss2 = other._moved(center)
        return _Partial(
            self.level + 1,
            center,
            abswsum,
            self.wsum + other.wsum,
            self.w2sum + other.w2sum,
            s1 + s2,
            ss1 + ss2,
            self.n + other.n,
        )


class _ShiftedSums(Generic[Array]):
    """The streaming state shared by both accumulators."""

    def __init__(self, bkd: Backend[Array], tol: float) -> None:
        self._bkd = bkd
        self._tol = tol
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
        absw = bkd.abs(weights)
        abswsum = bkd.sum(absw)
        if bkd.to_float(abswsum) == 0.0:
            raise ValueError("every weight in the batch is zero")
        center = bkd.dot(stacked, absw.T) / abswsum
        dev = stacked - center
        partial = _Partial(
            0,
            center,
            abswsum,
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
        if not self._stack or self._sizes is None:
            raise ValueError("no batches have been added")
        total = self._stack[0]
        for partial in self._stack[1:]:
            total = total.merged(partial)
        bkd = self._bkd
        if abs(bkd.to_float(total.wsum) - 1.0) > self._tol:
            raise ValueError(
                f"weights sum to {bkd.to_float(total.wsum)}, not 1 within {self._tol}"
            )
        mean = total.center + total.s / total.wsum
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
    """The unbiased sample covariance of equally weighted Monte Carlo draws.

    Scales the covariance by ``N / (N - 1)``, computed as
    ``1 / (1 - sum(w**2))`` with ``N`` equal weights ``1/N``. The mean is
    unchanged.

    **Only for independent, equally weighted random draws.** The factor
    corrects the bias from measuring spread about the sample mean rather
    than the true mean, and is exact only under independence. It does not
    apply to:

    - deterministic rules (Gauss, sparse grids), which need no correction:
      for the two-point Gauss-Hermite rule the variance would double;
    - quasi-Monte Carlo, whose points are not random, or randomized
      quasi-Monte Carlo, whose negatively correlated points make the
      factor over-correct;
    - importance sampling, whose weights are random and correlated with
      the points, so the factor is only approximately unbiased.

    Unequal weights are rejected. Equal weights cannot distinguish Monte
    Carlo from quasi-Monte Carlo, so that is the caller's responsibility.
    Use ``WeightedAccumulator`` in every other case.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    tol : float
        Allowed ``|sum(w) - 1|`` at ``finalize``. Default 1e-8.
    rtol : float
        Allowed relative difference between any weight and the first one
        seen. Default 1e-12.
    """

    def __init__(
        self, bkd: Backend[Array], tol: float = 1e-8, rtol: float = 1e-12
    ) -> None:
        self._sums = _ShiftedSums(bkd, tol)
        self._bkd = bkd
        self._rtol = rtol
        self._first_weight: Optional[float] = None

    def update(self, weights: Array, outputs: JointOutputs[Array]) -> None:
        """Add one batch of equally weighted draws. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If any weight differs from the first weight seen.
        """
        bkd = self._bkd
        if self._first_weight is None and weights.ndim == 2 and weights.shape[1] > 0:
            self._first_weight = bkd.to_float(weights[0, 0])
        if self._first_weight is not None:
            spread = bkd.to_float(bkd.max(bkd.abs(weights - self._first_weight)))
            if spread > self._rtol * abs(self._first_weight):
                raise ValueError(
                    "UnbiasedMCAccumulator needs equal weights; got weights "
                    f"differing from {self._first_weight} by up to {spread}. "
                    "Use WeightedAccumulator for unequal weights."
                )
        self._sums.update(weights, outputs)

    def finalize(self) -> DenseBlocks[Array]:
        """Blocks from every batch added so far."""
        return self._sums.moments(unbiased=True)
