"""Algebraic test functions.

These functions implement FunctionWithJacobianAndHVPProtocol directly.
"""

from pyapprox.typing.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
)
from pyapprox.typing.benchmarks.functions.algebraic.rosenbrock import (
    RosenbrockFunction,
)

__all__ = [
    "IshigamiFunction",
    "RosenbrockFunction",
]
