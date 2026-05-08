"""Quadrature strategy — maps tolerance to quadrature level.

Uses a ``ParameterizedQuadratureRuleProtocol`` to generate samples and
weights at a level determined by ``tol``. An LRU cache avoids redundant
rule evaluations.
"""

import math
from collections import OrderedDict
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.protocols.quadrature import (
    ParameterizedQuadratureRuleProtocol,
)


class QuadratureStrategy(Generic[Array]):
    """Strategy that maps ROL tolerance to a quadrature refinement level.

    The level is computed as::

        level = max(min_level, min(max_level, ceil(-log(tol) / log(rate))))

    The three most recent ``(level, samples, weights)`` tuples are cached
    to avoid redundant rule evaluations when ROL alternates tolerances.

    Parameters
    ----------
    rule : ParameterizedQuadratureRuleProtocol[Array]
        Quadrature rule with ``__call__(level)`` returning
        ``(samples, weights)``.
    bkd : Backend[Array]
        Computational backend.
    min_level : int, optional
        Minimum quadrature level. Default ``1``.
    max_level : int, optional
        Maximum quadrature level. Default ``20``.
    rate : float, optional
        Base for the log mapping. Default ``2.0``.
    """

    def __init__(
        self,
        rule: ParameterizedQuadratureRuleProtocol[Array],
        bkd: Backend[Array],
        min_level: int = 1,
        max_level: int = 20,
        rate: float = 2.0,
    ) -> None:
        if min_level < 0:
            raise ValueError(
                f"min_level must be non-negative, got {min_level}"
            )
        if max_level < min_level:
            raise ValueError(
                f"max_level ({max_level}) must be >= min_level ({min_level})"
            )
        if rate <= 1.0:
            raise ValueError(f"rate must be > 1.0, got {rate}")
        self._rule = rule
        self._bkd = bkd
        self._min_level = min_level
        self._max_level = max_level
        self._rate = rate
        self._log_rate = math.log(rate)
        # LRU cache: level -> (samples, weights), max 3 entries
        self._cache: OrderedDict[int, Tuple[Array, Array]] = OrderedDict()
        self._cache_maxsize = 3

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of random variables."""
        return self._rule.nvars()

    def _level_for_tol(self, tol: float) -> int:
        """Compute quadrature level for a given tolerance."""
        if tol <= 0:
            return self._max_level
        raw = math.ceil(-math.log(tol) / self._log_rate)
        return max(self._min_level, min(self._max_level, raw))

    def _get_cached(self, level: int) -> Tuple[Array, Array]:
        """Get samples/weights from cache or compute and cache them."""
        if level in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(level)
            return self._cache[level]
        # Compute new entry
        samples, weights = self._rule(level)
        # Evict oldest if at capacity
        if len(self._cache) >= self._cache_maxsize:
            self._cache.popitem(last=False)
        self._cache[level] = (samples, weights)
        return samples, weights

    def samples_and_weights(self, tol: float) -> Tuple[Array, Array]:
        """Return quadrature samples/weights for the given tolerance.

        Parameters
        ----------
        tol : float
            Accuracy tolerance from ROL. Smaller means higher level.

        Returns
        -------
        Tuple[Array, Array]
            ``(samples, weights)`` with shapes
            ``(n_random_vars, n_points)`` and ``(n_points,)``.
        """
        level = self._level_for_tol(tol)
        return self._get_cached(level)
