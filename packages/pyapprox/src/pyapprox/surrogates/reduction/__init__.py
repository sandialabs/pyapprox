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
- DecoderProtocol and its refinements: what a consumer of a decoder can
  rely on, from decoding alone up to a closed-form tangent

Key Classes
-----------
- MonomialFeatureMap: monomials of selected total degrees
- SparseMonomialFeatureMap: monomials over an arbitrary multi-index set
- MonomialManifoldEncoder: linear encoder, polynomial-correction decoder
- ManifoldScorer: scores trial bases and fits the correction weights

Snapshots too large to hold
---------------------------
The correction weights W are (nstates, p), so at a large ambient
dimension they are the object that does not fit. The fit streams
readily: the matrix it inverts is (p, p) and contracts over snapshots,
so each row of W depends only on the same row of the data.

- CenteredSource: subtracts the mean as each block is read, so a
  centered copy of the data is never held
- encode_from_source: V^T S, accumulated over row blocks
- fit_weights_from_source: W a row block at a time, into a sink
- select_gamma_from_source: held-out gamma selection with the gamma loop
  inside the block loop, so a grid costs no ambient array per gamma

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
from pyapprox.surrogates.reduction.manifold_streaming import (
    CenteredSource,
    encode_from_source,
    fit_weights_from_source,
    select_gamma_from_source,
)
from pyapprox.surrogates.reduction.monomial_manifold import (
    MonomialManifoldEncoder,
    build_monomial_manifold_encoder,
)
from pyapprox.surrogates.reduction.protocols import (
    DecoderProtocol,
    LinearDecoderProtocol,
    ManifoldDecoderProtocol,
    SelfJacobianDecoderProtocol,
    is_linear_decoder,
    is_manifold_decoder,
    is_self_jacobian_decoder,
)

__all__ = [
    "DecoderProtocol",
    "DifferentiableFeatureMap",
    "FeatureMap",
    "LinearDecoderProtocol",
    "ManifoldDecoderProtocol",
    "ManifoldScorer",
    "CenteredSource",
    "encode_from_source",
    "fit_weights_from_source",
    "select_gamma_from_source",
    "MonomialFeatureMap",
    "MonomialManifoldEncoder",
    "SelfJacobianDecoderProtocol",
    "SparseMonomialFeatureMap",
    "build_feature_map",
    "build_monomial_manifold_encoder",
    "center_and_decompose",
    "is_linear_decoder",
    "is_manifold_decoder",
    "is_self_jacobian_decoder",
]
