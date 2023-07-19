"""The :mod:`pyapprox.variables` module provides tools for creating and
transforming multivariate random variables.
"""

from pyapprox.variables.joint import (
    IndependentMarginalsVariable, GaussCopulaVariable,
    combine_uncertain_and_bounded_design_variables, JointVariable
)
from pyapprox.variables.sampling import print_statistics
from pyapprox.variables.transforms import (
    AffineTransform, NatafTransform,
    RosenblattTransform, ConfigureVariableTransformation
)

__all__ = ["IndependentMarginalsVariable", "GaussCopulaVariable",
           "print_statistics", "JointVariable",
           "AffineTransform", "NatafTransform",
           "RosenblattTransform", "ConfigureVariableTransformation",
           "combine_uncertain_and_bounded_design_variables"]
