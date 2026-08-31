"""Dimension reduction: linear subspaces and polynomial manifolds.

Reduction maps a high-dimensional state to a small set of coordinates and
back.  A linear reduction decodes as :math:`g(z) = V z`; a polynomial
manifold adds a nonlinear correction :math:`g(z) = V z + W h(z)`, where the
feature map :math:`h` supplies the monomials and is the only piece that
distinguishes a quadratic manifold from a cubic one.

Key Protocols
-------------
- FeatureMap: reduced coordinates -> nonlinear features
- DifferentiableFeatureMap: a feature map that also supplies its Jacobian

Key Classes
-----------
- MonomialFeatureMap: monomials of selected total degrees
- SparseMonomialFeatureMap: monomials over an arbitrary multi-index set
- ManifoldScorer: scores trial bases and fits the correction weights

Utilities
---------
- center_and_decompose: center snapshots and take their thin SVD
"""

from pyapprox.surrogates.reduction.feature_maps import (
    DifferentiableFeatureMap,
    FeatureMap,
    MonomialFeatureMap,
    SparseMonomialFeatureMap,
    build_feature_map,
)
from pyapprox.surrogates.reduction.manifold_scoring import (
    ManifoldScorer,
    center_and_decompose,
)

__all__ = [
    "DifferentiableFeatureMap",
    "FeatureMap",
    "ManifoldScorer",
    "MonomialFeatureMap",
    "SparseMonomialFeatureMap",
    "build_feature_map",
    "center_and_decompose",
]
