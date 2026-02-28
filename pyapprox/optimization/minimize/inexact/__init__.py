"""Inexact gradient support for ROL trust-region optimization.

Provides strategies that map ROL's adaptive tolerance to sample counts
or quadrature levels, enabling cost-efficient gradient evaluation in
early optimization iterations.
"""

from pyapprox.optimization.minimize.inexact.fixed import (
    FixedSampleStrategy,
)
from pyapprox.optimization.minimize.inexact.monte_carlo import (
    MonteCarloSAAStrategy,
)
from pyapprox.optimization.minimize.inexact.protocols import (
    InexactDifferentiable,
    InexactEvaluable,
    InexactGradientStrategyProtocol,
)
from pyapprox.optimization.minimize.inexact.qmc import (
    QMCSAAStrategy,
)
from pyapprox.optimization.minimize.inexact.quadrature import (
    QuadratureStrategy,
)

__all__ = [
    "FixedSampleStrategy",
    "InexactDifferentiable",
    "InexactEvaluable",
    "InexactGradientStrategyProtocol",
    "MonteCarloSAAStrategy",
    "QMCSAAStrategy",
    "QuadratureStrategy",
]
