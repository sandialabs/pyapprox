"""Genz integration test functions.

These functions are widely used for benchmarking multidimensional
integration routines. Four of the six functions implement
FunctionWithJacobianAndHVPProtocol (the differentiable ones).
"""

from pyapprox.typing.benchmarks.functions.genz.oscillatory import (
    OscillatoryFunction,
)
from pyapprox.typing.benchmarks.functions.genz.product_peak import (
    ProductPeakFunction,
)
from pyapprox.typing.benchmarks.functions.genz.corner_peak import (
    CornerPeakFunction,
)
from pyapprox.typing.benchmarks.functions.genz.gaussian_peak import (
    GaussianPeakFunction,
)

__all__ = [
    "OscillatoryFunction",
    "ProductPeakFunction",
    "CornerPeakFunction",
    "GaussianPeakFunction",
]
