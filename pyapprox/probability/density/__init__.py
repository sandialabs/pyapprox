"""Density estimation via quadrature projection.

Estimate the density f_Y(y) of a scalar QoI Y = g(xi) by projecting onto a
basis in y-space using weighted evaluations. The framework is general: it
accepts any (y_values, weights) pairs from arbitrary quadrature rules.
"""

from pyapprox.probability.density._fitters import (
    DensityFitterProtocol,
    KDEFitter,
    LinearDensityFitter,
)
from pyapprox.probability.density.kernel_density_basis import (
    KernelDensityBasis,
)
from pyapprox.probability.density.piecewise_density_basis import (
    PiecewiseDensityBasis,
)
from pyapprox.probability.density.projection import (
    ISEOptimizingFitter,
    ProjectionDensityFitter,
)
from pyapprox.probability.density.protocols import (
    DensityBasisProtocol,
)
from pyapprox.probability.density.pushforward import (
    PushforwardDensity,
)

__all__ = [
    "DensityBasisProtocol",
    "DensityFitterProtocol",
    "PiecewiseDensityBasis",
    "KernelDensityBasis",
    "LinearDensityFitter",
    "KDEFitter",
    "ProjectionDensityFitter",
    "ISEOptimizingFitter",
    "PushforwardDensity",
]
