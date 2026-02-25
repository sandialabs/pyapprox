"""Wrappers for adding functionality to functions.

This module provides wrappers for:
- Work tracking (evaluation counts and wall times)
- Finite difference derivatives
"""

from pyapprox.interface.wrappers.work_tracker import (
    WorkTracker,
    TrackedModel,
)
from pyapprox.interface.wrappers.finite_difference import (
    FiniteDifferenceWrapper,
)

__all__ = [
    "WorkTracker",
    "TrackedModel",
    "FiniteDifferenceWrapper",
]
