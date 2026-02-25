"""Benchmark test functions.

Functions in this module implement existing protocols from
pyapprox.interface.functions.protocols directly.
"""

from pyapprox.benchmarks.functions.ode import (
    ODEBenchmark,
    ODETimeConfig,
)

__all__ = [
    "ODEBenchmark",
    "ODETimeConfig",
]
