"""The :mod:`pyapprox.benchmarks` module implements numerous benchmarks from
the modeling literature.
"""

from pyapprox.benchmarks.algebraic import (
    IshigamiBenchmark,
    OakleyBenchmark,
    SobolGBenchmark,
    RosenbrockUnconstrainedOptimizationBenchmark,
    RosenbrockConstrainedOptimizationBenchmark,
    CantileverBeamDeterminsticOptimizationBenchmark,
    CantileverBeamUncertainOptimizationBenchmark,
    PistonBenchmark,
    WingWeightBenchmark,
    EvtushenkoConstrainedOptimizationBenchmark,
)
from pyapprox.benchmarks.genz import GenzBenchmark
from pyapprox.benchmarks.multifidelity_benchmarks import (
    PolynomialModelEnsemble,
    TunableModelEnsemble,
    ShortColumnModelEnsemble,
    MultiOutputModelEnsemble,
    PSDMultiOutputModelEnsemble,
    MultiLevelCosineBenchmark,
)

from pyapprox.benchmarks.ode import (
    ChemicalReactionBenchmark,
    LotkaVolterraBenchmark,
    CoupledSpringsBenchmark,
    HastingsEcologyBenchmark,
)

from pyapprox.benchmarks.pde import (
    PyApproxPaperAdvectionDiffusionKLEInversionBenchmark,
    TransientViscousBurgers1DOperatorBenchmark,
    SteadyDarcy2DOperatorBenchmark,
    ParameterizedNonlinearSystemOfEquationsBenchmark,
)

__all__ = [
    "IshigamiBenchmark",
    "OakleyBenchmark",
    "SobolGBenchmark",
    "RosenbrockUnconstrainedOptimizationBenchmark",
    "RosenbrockConstrainedOptimizationBenchmark",
    "CantileverBeamDeterminsticOptimizationBenchmark",
    "CantileverBeamUncertainOptimizationBenchmark",
    "PistonBenchmark",
    "WingWeightBenchmark",
    "EvtushenkoConstrainedOptimizationBenchmark",
    "GenzBenchmark",
    "PolynomialModelEnsemble",
    "TunableModelEnsemble",
    "ShortColumnModelEnsemble",
    "MultiOutputModelEnsemble",
    "PSDMultiOutputModelEnsemble",
    "MultiLevelCosineBenchmark",
    "ChemicalReactionBenchmark",
    "LotkaVolterraBenchmark",
    "CoupledSpringsBenchmark",
    "HastingsEcologyBenchmark",
    "PyApproxPaperAdvectionDiffusionKLEInversionBenchmark",
    "TransientViscousBurgers1DOperatorBenchmark",
    "SteadyDarcy2DOperatorBenchmark",
    "ParameterizedNonlinearSystemOfEquationsBenchmark",
]
