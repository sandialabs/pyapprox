"""Coordinate transforms for spectral collocation methods."""

from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)
from pyapprox.pde.collocation.mesh.transforms.polar import (
    PolarTransform,
)
from pyapprox.pde.collocation.mesh.transforms.spherical import (
    SphericalTransform,
)
from pyapprox.pde.collocation.mesh.transforms.elliptical import (
    EllipticalTransform,
)
from pyapprox.pde.collocation.mesh.transforms.chained import (
    ChainedTransform,
)
from pyapprox.pde.collocation.mesh.transforms.sympy_transform import (
    SympyTransform2D,
    SympyTransform3D,
)

__all__ = [
    # Affine transforms
    "AffineTransform1D",
    "AffineTransform2D",
    "AffineTransform3D",
    # Curvilinear transforms
    "PolarTransform",
    "SphericalTransform",
    "EllipticalTransform",
    # User-defined transforms
    "SympyTransform2D",
    "SympyTransform3D",
    # Composition
    "ChainedTransform",
]
