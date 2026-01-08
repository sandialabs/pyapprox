"""Algebraic test functions.

These functions implement FunctionWithJacobianAndHVPProtocol directly.
"""

from pyapprox.typing.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
)
from pyapprox.typing.benchmarks.functions.algebraic.rosenbrock import (
    RosenbrockFunction,
)
from pyapprox.typing.benchmarks.functions.algebraic.sobol_g import (
    SobolGFunction,
    sobol_g_indices,
)
from pyapprox.typing.benchmarks.functions.algebraic.branin import (
    BraninFunction,
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
)

__all__ = [
    "IshigamiFunction",
    "RosenbrockFunction",
    "SobolGFunction",
    "sobol_g_indices",
    "BraninFunction",
    "BRANIN_GLOBAL_MINIMUM",
    "BRANIN_MINIMIZERS",
]
