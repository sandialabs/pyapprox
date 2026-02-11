"""Density estimation via quadrature projection.

Estimate the density f_Y(y) of a scalar QoI Y = g(xi) by projecting onto a
basis in y-space using weighted evaluations. The framework is general: it
accepts any (y_values, weights) pairs from arbitrary quadrature rules.
"""

from pyapprox.typing.probability.density.protocols import (
    DensityBasisProtocol,
)
from pyapprox.typing.probability.density.piecewise_density_basis import (
    PiecewiseDensityBasis,
)
from pyapprox.typing.probability.density.kernel_density_basis import (
    KernelDensityBasis,
)
from pyapprox.typing.probability.density.projection import (
    ProjectionDensityFitter,
    ISEOptimizingFitter,
)
from pyapprox.typing.probability.density.pushforward import (
    PushforwardDensity,
)
from pyapprox.typing.probability.density._fitters import (
    DensityFitterProtocol,
    LinearDensityFitter,
    KDEFitter,
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
