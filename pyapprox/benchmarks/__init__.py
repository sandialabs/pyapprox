"""The :mod:`pyapprox.benchmarks` module implements numerous benchmarks from
the modeling literature.
"""

from pyapprox.benchmarks.algebraic import (
    IshigamiBenchmark,
    OakleyBenchmark,
    SobolGBenchmark,
    RosenbrockBenchmark,
    CantileverBeamDeterminsticOptimizationBenchmark,
    CantileverBeamUncertainOptimizationBenchmark,
    PistonBenchmark,
    WingWeightBenchmark,
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
    "RosenbrockBenchmark",
    "CantileverBeamDeterminsticOptimizationBenchmark",
    "CantileverBeamUncertainOptimizationBenchmark",
    "PistonBenchmark",
    "WingWeightBenchmark",
    "GenzBenchmark",
    "ChemicalReactionBenchmark",
    "LotkaVolterraBenchmark",
    "CoupledSpringsBenchmark",
    "HastingsEcologyBenchmark",
]
