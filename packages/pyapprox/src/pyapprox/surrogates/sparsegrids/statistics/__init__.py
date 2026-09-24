"""Post-processing applied to fitted sparse grids.

Surrogates and subspaces evaluate; quantities derived from them live
here, following the pattern of ``functiontrain/statistics``.
"""

from pyapprox.surrogates.sparsegrids.statistics.cache import (
    SubspaceCache,
    box_sum,
)
from pyapprox.surrogates.sparsegrids.statistics.cross_moments import (
    CrossMomentMoments,
)
from pyapprox.surrogates.sparsegrids.statistics.hierarchical_moments import (
    HierarchicalMoments,
)
from pyapprox.surrogates.sparsegrids.statistics.moments import (
    PCEMoments,
    QuadratureMoments,
)
from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    subspace_mean,
    subspace_raw_moment,
    subspace_variance,
)

__all__ = [
    "CrossMomentMoments",
    "HierarchicalMoments",
    "PCEMoments",
    "QuadratureMoments",
    "SubspaceCache",
    "box_sum",
    "subspace_mean",
    "subspace_raw_moment",
    "subspace_variance",
]
