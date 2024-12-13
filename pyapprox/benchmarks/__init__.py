"""The :mod:`pyapprox.benchmarks` module implements numerous benchmarks from
the modeling literature.
"""

from pyapprox.benchmarks.sensitivity_benchmarks import (
    IshigamiBenchmark,
    OakleyBenchmark,
    SobolGBenchmark,
)

from pyapprox.benchmarks.surrogate_benchmarks import (
    RosenbrockBenchmark,
    CantileverBeamDeterminsticOptimizationBenchmark,
    CantileverBeamUncertainOptimizationBenchmark,
)

__all__ = [
    "IshigamiBenchmark",
    "OakleyBenchmark",
    "SobolGBenchmark",
    "RosenbrockBenchmark",
    "CantileverBeamDeterminsticOptimizationBenchmark",
    "CantileverBeamUncertainOptimizationBenchmark",
]
