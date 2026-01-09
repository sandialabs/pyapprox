"""Legacy benchmark wrappers.

This module provides adapters for wrapping legacy benchmarks from
pyapprox.benchmarks without modifying the original code.
"""

from pyapprox.typing.benchmarks.legacy.adapter import (
    LegacyFunctionAdapter,
    LegacyFunctionWithJacobianAdapter,
)
from pyapprox.typing.benchmarks.legacy.wrappers import (
    wrap_legacy_ishigami,
    wrap_legacy_genz,
)

__all__ = [
    "LegacyFunctionAdapter",
    "LegacyFunctionWithJacobianAdapter",
    "wrap_legacy_ishigami",
    "wrap_legacy_genz",
]
