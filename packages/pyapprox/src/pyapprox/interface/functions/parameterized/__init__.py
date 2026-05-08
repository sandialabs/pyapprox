"""
Parameterized functions for PyApprox typing module.

This module provides protocols and wrappers for functions with tunable parameters.
Parameterized functions are useful for optimization, calibration, and inverse problems.

Key Classes
-----------
- ParameterizedFunctionProtocol: Protocol for parameterized functions
- ParameterizedFunctionWithJacobianProtocol: With Jacobian support
- ParameterizedFunctionWithJacobianAndHVPProtocol: With Jacobian and HVP support
- FunctionOfParameters: Wrapper for parameterized functions
- FunctionOfParametersWithJacobian: With Jacobian wrapper
- FunctionOfParametersWithJacobianAndHVP: With Jacobian and HVP wrapper

Key Functions
-------------
- convert_to_function_of_parameters: Convert parameterized function to standard function

Examples
--------
>>> from pyapprox.interface.functions.parameterized import (
...     convert_to_function_of_parameters
... )
>>> param_func = ...  # Some parameterized function
>>> func = convert_to_function_of_parameters(param_func, fixed_param)
"""

from .factory import convert_to_function_of_parameters
from .protocols import (
    ParameterizedFunctionProtocol,
    ParameterizedFunctionWithJacobianAndHVPProtocol,
    ParameterizedFunctionWithJacobianProtocol,
)
from .validation import validate_parameterized_function
from .wrappers import (
    FunctionOfParameters,
    FunctionOfParametersWithJacobian,
    FunctionOfParametersWithJacobianAndHVP,
)

__all__ = [
    "ParameterizedFunctionProtocol",
    "ParameterizedFunctionWithJacobianProtocol",
    "ParameterizedFunctionWithJacobianAndHVPProtocol",
    "FunctionOfParameters",
    "FunctionOfParametersWithJacobian",
    "FunctionOfParametersWithJacobianAndHVP",
    "convert_to_function_of_parameters",
    "validate_parameterized_function",
]
