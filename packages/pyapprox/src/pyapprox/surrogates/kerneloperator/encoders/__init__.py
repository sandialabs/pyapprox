"""Function encoders for kernel operator learning.

For a PCA encoder use ``pyapprox.surrogates.kle.fit_kle_encoder``, which
returns a ``KLEEncoder`` satisfying the same protocol.

It lives beside the expansions because a PCA basis and a data-driven KLE
are the same basis, mean and spectrum traversed in opposite directions.
The expansion goes coefficients to field, ``f = mean + sum_i
sqrt(lambda_i) phi_i z_i``, where ``z`` is standardized and the
``sqrt(lambda)`` scaling is what gives the result its covariance. The
encoder goes field to coordinates and back, ``z = V^T M (f - mean)`` and
``f = V z + mean``, where ``z`` is a coordinate rather than a
standardized variable and the basis is therefore used unweighted.

Keeping them together is what lets an encoder be persisted by
``save_kle``, given a metric, or built with a chosen eigensolver.
"""

from pyapprox.surrogates.kerneloperator.encoders.identity import (
    IdentityFunctionEncoder,
)

__all__ = [
    "IdentityFunctionEncoder",
]
