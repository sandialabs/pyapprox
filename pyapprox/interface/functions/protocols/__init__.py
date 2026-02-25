"""
Function protocols for PyApprox typing module.

This module defines protocols for function objects with varying levels of
derivative support. Protocols enable duck typing with runtime type checking.

Key Protocols
-------------
- FunctionProtocol: Basic function evaluation
- FunctionWithJacobianProtocol: Function with Jacobian support
- FunctionWithJVPProtocol: Function with Jacobian-vector product
- FunctionWithJacobianAndHVPProtocol: Function with Jacobian and Hessian-vector product
- FunctionWithJVPAndHVPProtocol: Function with JVP and HVP
- FunctionWithJacobianAndWHVPProtocol: Function with Jacobian and weighted HVP

Examples
--------
>>> from pyapprox.interface.functions.protocols import FunctionProtocol
>>> def use_function(f: FunctionProtocol):
...     return f(samples)
"""

from .function import FunctionProtocol
from .jacobian import (
    FunctionWithJacobianProtocol,
    FunctionWithJVPProtocol,
)
from .hessian import (
    FunctionWithJacobianAndHVPProtocol,
    FunctionWithJVPAndHVPProtocol,
    FunctionWithJacobianAndWHVPProtocol,
)

__all__ = [
    "FunctionProtocol",
    "FunctionWithJacobianProtocol",
    "FunctionWithJVPProtocol",
    "FunctionWithJacobianAndHVPProtocol",
    "FunctionWithJVPAndHVPProtocol",
    "FunctionWithJacobianAndWHVPProtocol",
]
