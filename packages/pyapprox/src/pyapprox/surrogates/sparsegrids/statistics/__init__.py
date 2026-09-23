"""Post-processing applied to fitted sparse grids.

Surrogates and subspaces evaluate; quantities derived from them live
here, following the pattern of ``functiontrain/statistics``.
"""

from pyapprox.surrogates.sparsegrids.statistics.cache import (
    SubspaceCache,
    box_sum,
)
from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    subspace_mean,
    subspace_raw_moment,
    subspace_variance,
)

__all__ = [
    "SubspaceCache",
    "box_sum",
    "subspace_mean",
    "subspace_raw_moment",
    "subspace_variance",
]
