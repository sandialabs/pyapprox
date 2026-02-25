"""Leja and Fekete point selection for well-conditioned basis expansions.

This module provides methods for selecting sample points that produce
well-conditioned polynomial interpolation matrices.

Two types of point selection are provided:

**Leja Sequences** - Incremental/nested point selection:

1. **Optimization-based univariate Leja** (leja.univariate):
   - Uses gradient-based optimization to find each new point
   - Points are added one at a time, each maximizing a weighted residual
   - Uses optimizers from typing.optimization.minimize (default: trust-constr)

2. **LU-based multivariate Leja** (leja.multivariate.LejaSampler):
   - Uses pivoted LU factorization for incremental selection from candidates
   - Points selected sequentially via partial pivoting

**Fekete Points** - One-shot point selection:

- **QR-based Fekete** (leja.multivariate.FeketeSampler):
   - Uses pivoted QR factorization for one-shot selection
   - Selects all points at once to approximately maximize the
     determinant of the interpolation matrix

Weighting strategies:
- ChristoffelWeighting: Weight by inverse Christoffel function (default)
- PDFWeighting: Weight by probability density function
- CompositeWeighting: Combine multiple weightings
"""

from .multivariate import (
    FeketeSampler,
    LejaSampler,
    WeightedLejaSampler,
)
from .protocols import (
    FeketeSamplerProtocol,
    LejaSamplerProtocol,
    # Sequence protocols
    LejaSequence1DProtocol,
    # Weighting protocols
    LejaWeightingProtocol,
    LejaWeightingWithJacobianProtocol,
)
from .univariate import (
    LejaObjective,
    LejaSequence1D,
    ScipyTrustConstrMinimizer,
    TwoPointLejaObjective,
)
from .weighting import (
    ChristoffelWeighting,
    CompositeWeighting,
    PDFWeighting,
)

__all__ = [
    # Weighting protocols
    "LejaWeightingProtocol",
    "LejaWeightingWithJacobianProtocol",
    # Sequence protocols
    "LejaSequence1DProtocol",
    "LejaSamplerProtocol",
    "FeketeSamplerProtocol",
    # Weighting implementations
    "ChristoffelWeighting",
    "PDFWeighting",
    "CompositeWeighting",
    # Univariate implementations
    "ScipyTrustConstrMinimizer",
    "LejaObjective",
    "TwoPointLejaObjective",
    "LejaSequence1D",
    # Multivariate implementations
    "LejaSampler",
    "FeketeSampler",
    "WeightedLejaSampler",
]
