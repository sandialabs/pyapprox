"""Coordinate transforms for spectral collocation methods."""

from pyapprox.typing.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)
from pyapprox.typing.pde.collocation.mesh.transforms.polar import (
    PolarTransform,
)
from pyapprox.typing.pde.collocation.mesh.transforms.spherical import (
    SphericalTransform,
)
from pyapprox.typing.pde.collocation.mesh.transforms.chained import (
    ChainedTransform,
)

__all__ = [
    # Affine transforms
    "AffineTransform1D",
    "AffineTransform2D",
    "AffineTransform3D",
    # Curvilinear transforms
    "PolarTransform",
    "SphericalTransform",
    # Composition
    "ChainedTransform",
]
