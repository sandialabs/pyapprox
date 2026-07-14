"""
Function protocols for PyApprox typing module.

Protocols enable duck typing with runtime type checking. Derivative
capability travels in the ``Derivatives`` bundle, exposed through the
``derivatives()`` accessor required by ``ObjectiveProtocol`` and
``NonlinearConstraintProtocol``.

Key Protocols
-------------
- FunctionProtocol: Basic function evaluation (value-only base shape)
- ObjectiveProtocol: FunctionProtocol plus a ``derivatives()`` bundle
- NonlinearConstraintProtocol: ObjectiveProtocol shape plus bounds

Examples
--------
>>> from pyapprox.interface.functions.protocols import FunctionProtocol
>>> def use_function(f: FunctionProtocol):
...     return f(samples)
"""

from .constraint import NonlinearConstraintProtocol
from .function import FunctionProtocol
from .objective import (
    Function,
    ObjectiveProtocol,
)

__all__ = [
    "FunctionProtocol",
    "ObjectiveProtocol",
    "NonlinearConstraintProtocol",
    "Function",
]
