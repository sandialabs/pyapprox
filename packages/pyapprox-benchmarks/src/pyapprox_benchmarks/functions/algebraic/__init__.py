"""Algebraic test functions.

These functions implement FunctionWithJacobianAndHVPProtocol directly.
"""

from pyapprox_benchmarks.functions.algebraic.quadratic import (
    DiagonalQuadraticFunction,
    QuadraticFunction,
    add,
    square,
)
from pyapprox_benchmarks.functions.algebraic.branin import (
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
    BraninFunction,
)
from pyapprox_benchmarks.functions.algebraic.cantilever_beam import (
    CantileverBeam1DAnalytical,
    HomogeneousBeam1DAnalytical,
)
from pyapprox_benchmarks.functions.algebraic.cantilever_beam_2d import (
    CantileverBeam2DAnalytical,
)
from pyapprox_benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
    IshigamiSensitivityIndices,
)
from pyapprox_benchmarks.functions.algebraic.rosenbrock import (
    RosenbrockFunction,
)
from pyapprox_benchmarks.functions.algebraic.sobol_g import (
    SobolGFunction,
    SobolGSensitivityIndices,
)

__all__ = [
    "IshigamiFunction",
    "IshigamiSensitivityIndices",
    "RosenbrockFunction",
    "SobolGFunction",
    "SobolGSensitivityIndices",
    "BraninFunction",
    "BRANIN_GLOBAL_MINIMUM",  # TODO: Does this need to be in __init__
    "BRANIN_MINIMIZERS",  # TODO: Does this need to be in __init__
    "CantileverBeam1DAnalytical",
    "CantileverBeam2DAnalytical",
    "HomogeneousBeam1DAnalytical",
    "QuadraticFunction",
    "DiagonalQuadraticFunction",
    "square",
    "add",
]
