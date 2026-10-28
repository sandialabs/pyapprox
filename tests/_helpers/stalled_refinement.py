"""Reproduce a stalled quadrature-scored refinement, for tests only.

A quadrature-scored indicator raises ``QuadratureRefinementStalledError``
for a candidate whose new points carry no quadrature weight, because
scoring it zero stops its dimension refining for the rest of the run.
Tests that need that stalled run, to compare against it, wrap the
indicator in ``IgnoreZeroWeight``. The library deliberately offers no
such wrapper; ``ZeroWeightFallback`` is the one to use.
"""

import warnings
from typing import Generic

from pyapprox.surrogates.sparsegrids.candidate_info import (
    AdaptiveGridViewProtocol,
    Candidate,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    ErrorIndicatorProtocol,
    QuadratureRefinementStalled,
    QuadratureRefinementStalledError,
)
from pyapprox.util.backends.protocols import Array


class IgnoreZeroWeight(Generic[Array]):
    """Score candidates with weightless new points as zero; tests only.

    Returns ``inner``'s score, or 0.0 when ``inner`` raises
    ``QuadratureRefinementStalledError``. Warns once, with
    ``QuadratureRefinementStalled``.
    """

    def __init__(self, inner: ErrorIndicatorProtocol[Array]) -> None:
        if not isinstance(inner, ErrorIndicatorProtocol):
            raise TypeError(
                f"inner must satisfy ErrorIndicatorProtocol, got {type(inner).__name__}"
            )
        self._inner = inner
        self._warned = False

    def __call__(
        self, candidate: Candidate[Array], grid: AdaptiveGridViewProtocol[Array]
    ) -> float:
        try:
            return self._inner(candidate, grid)
        except QuadratureRefinementStalledError:
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"{type(self._inner).__name__} stalled; scoring weightless "
                    "candidates as zero.",
                    QuadratureRefinementStalled,
                    stacklevel=2,
                )
            return 0.0
