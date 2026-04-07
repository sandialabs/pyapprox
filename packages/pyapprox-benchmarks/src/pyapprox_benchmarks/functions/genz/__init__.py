"""Genz integration test functions.

These functions are widely used for benchmarking multidimensional
integration routines. Four of the six functions implement
FunctionWithJacobianAndHVPProtocol (the differentiable ones).
"""

from pyapprox_benchmarks.functions.genz.corner_peak import (
    CornerPeakFunction,
)
from pyapprox_benchmarks.functions.genz.gaussian_peak import (
    GaussianPeakFunction,
)
from pyapprox_benchmarks.functions.genz.oscillatory import (
    OscillatoryFunction,
)
from pyapprox_benchmarks.functions.genz.product_peak import (
    ProductPeakFunction,
)

__all__ = [
    "OscillatoryFunction",
    "ProductPeakFunction",
    "CornerPeakFunction",
    "GaussianPeakFunction",
]

# TODO: Need to port over from legacy linalg branch
# the other two genz functions just with FunctionProtocol

# Need to add tests of Genz functions, speicifically
# DerivativeChecker for jacobian and hvp.
