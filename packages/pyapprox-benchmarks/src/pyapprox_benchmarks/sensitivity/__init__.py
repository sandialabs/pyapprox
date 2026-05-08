"""Sensitivity analysis benchmarks."""

from pyapprox_benchmarks.sensitivity.ishigami import IshigamiBenchmark
from pyapprox_benchmarks.sensitivity.sobol_g import SobolGBenchmark

__all__ = [
    "IshigamiBenchmark",
    "SobolGBenchmark",
]
