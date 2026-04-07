"""Wrappers for adding functionality to functions.

This module provides wrappers for:
- Work tracking (evaluation counts and wall times)
- Finite difference derivatives
"""

from pyapprox.interface.wrappers.finite_difference import (
    FiniteDifferenceWrapper,
)
from pyapprox.interface.wrappers.work_tracker import (
    TrackedModel,
    WorkTracker,
)

__all__ = [
    "WorkTracker",
    "TrackedModel",
    "FiniteDifferenceWrapper",
]
