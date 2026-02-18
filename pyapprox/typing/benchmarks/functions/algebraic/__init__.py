"""Algebraic test functions.

These functions implement FunctionWithJacobianAndHVPProtocol directly.
"""

from pyapprox.typing.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
    IshigamiSensitivityIndices,
)
from pyapprox.typing.benchmarks.functions.algebraic.rosenbrock import (
    RosenbrockFunction,
)
from pyapprox.typing.benchmarks.functions.algebraic.sobol_g import (
    SobolGFunction,
    SobolGSensitivityIndices,
)
from pyapprox.typing.benchmarks.functions.algebraic.branin import (
    BraninFunction,
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
)
from pyapprox.typing.benchmarks.functions.algebraic.cantilever_beam import (
    CantileverBeam1DAnalytical,
)
from pyapprox.typing.benchmarks.functions.algebraic.cantilever_beam_2d import (
    CantileverBeam2DAnalytical,
)

__all__ = [
    "IshigamiFunction",
    "IshigamiSensitivityIndices",
    "RosenbrockFunction",
    "SobolGFunction",
    "SobolGSensitivityIndices",
    "BraninFunction",
    "BRANIN_GLOBAL_MINIMUM",
    "BRANIN_MINIMIZERS",
    "CantileverBeam1DAnalytical",
    "CantileverBeam2DAnalytical",
]
