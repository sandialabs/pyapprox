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
    PolynomialModelEnsembleBenchmark,
    TunableModelEnsembleBenchmark,
    ShortColumnModelEnsembleBenchmark,
    MultiOutputModelEnsembleBenchmark,
    PSDMultiOutputModelEnsembleBenchmark,
    MultiLevelCosineBenchmark,
)

from pyapprox.benchmarks.ode import (
    ChemicalReactionBenchmark,
    LotkaVolterraBenchmark,
    LotkaVolterraOEDBenchmark,
    CoupledSpringsBenchmark,
    HastingsEcologyBenchmark,
)

from pyapprox.benchmarks.pde import (
    PyApproxPaperAdvectionDiffusionKLEInversionBenchmark,
    TransientViscousBurgers1DOperatorBenchmark,
    SteadyDarcy2DOperatorBenchmark,
    NonlinearSystemOfEquationsBenchmark,
)

__all__ = [
    # Sensitivity Analysis
    "IshigamiBenchmark",
    "OakleyBenchmark",
    "SobolGBenchmark",
    # Optimization
    "RosenbrockUnconstrainedOptimizationBenchmark",
    "RosenbrockConstrainedOptimizationBenchmark",
    "CantileverBeamDeterminsticOptimizationBenchmark",
    "CantileverBeamUncertainOptimizationBenchmark",
    "EvtushenkoConstrainedOptimizationBenchmark",
    # Quadrature
    "GenzBenchmark",
    # Multifidelity Estimation
    "PolynomialModelEnsembleBenchmark",
    "TunableModelEnsembleBenchmark",
    "ShortColumnModelEnsembleBenchmark",
    "MultiOutputModelEnsembleBenchmark",
    "PSDMultiOutputModelEnsembleBenchmark",
    "MultiLevelCosineBenchmark",
    "LotkaVolterraOEDBenchmark",
    # Inference
    "PyApproxPaperAdvectionDiffusionKLEInversionBenchmark",
    # Operator
    "TransientViscousBurgers1DOperatorBenchmark",
    "SteadyDarcy2DOperatorBenchmark",
    # MISC
    "LotkaVolterraBenchmark",
    "ChemicalReactionBenchmark",
    "CoupledSpringsBenchmark",
    "HastingsEcologyBenchmark",
    "PistonBenchmark",
    "WingWeightBenchmark",
    "NonlinearSystemOfEquationsBenchmark",
]
