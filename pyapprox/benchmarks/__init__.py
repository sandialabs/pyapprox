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

from pyapprox.benchmarks.ode import (
    ChemicalReactionBenchmark,
    LotkaVolterraBenchmark,
    CoupledSpringsBenchmark,
    HastingsEcologyBenchmark,
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
    "ChemicalReactionBenchmark",
    "LotkaVolterraBenchmark",
    "CoupledSpringsBenchmark",
    "HastingsEcologyBenchmark",
]
