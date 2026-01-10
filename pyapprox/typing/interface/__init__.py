"""Interface module for model wrappers and function utilities.

This module provides:

Wrappers:
- WorkTracker: Track model evaluation counts and wall times
- TrackedModel: Transparent wrapper that records to WorkTracker
- FiniteDifferenceWrapper: Add finite difference jacobian/hessian to functions

UMBridge:
- UMBridgeModel: HTTP client for UMBridge models
- UMBRIDGE_AVAILABLE: Whether umbridge package is installed

Parallel (from parallel submodule):
- ParallelConfig, make_parallel, etc. (see parallel submodule)

Functions (from functions submodule):
- FunctionProtocol, FunctionFromCallable, etc. (see functions submodule)
"""

from pyapprox.typing.interface.wrappers import (
    WorkTracker,
    TrackedModel,
    FiniteDifferenceWrapper,
)
from pyapprox.typing.interface.umbridge import (
    UMBridgeModel,
    UMBRIDGE_AVAILABLE,
)

__all__ = [
    # Wrappers
    "WorkTracker",
    "TrackedModel",
    "FiniteDifferenceWrapper",
    # UMBridge
    "UMBridgeModel",
    "UMBRIDGE_AVAILABLE",
]
