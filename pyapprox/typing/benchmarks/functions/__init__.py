"""Benchmark test functions.

Functions in this module implement existing protocols from
pyapprox.typing.interface.functions.protocols directly.
"""

from pyapprox.typing.benchmarks.functions.ode import (
    ODEBenchmark,
    ODETimeConfig,
)

__all__ = [
    "ODEBenchmark",
    "ODETimeConfig",
]
