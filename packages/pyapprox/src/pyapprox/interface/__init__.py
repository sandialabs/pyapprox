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

from pyapprox.interface.wrappers import (
    FiniteDifferenceWrapper,
    TrackedModel,
    WorkTracker,
)

__all__ = [
    # Wrappers
    "WorkTracker",
    "TrackedModel",
    "FiniteDifferenceWrapper",
    # UMBridge (lazy import to avoid ~300ms umbridge load time)
    "UMBridgeModel",
    "UMBRIDGE_AVAILABLE",
]


def __getattr__(name: str) -> object:
    """Lazy import for UMBridge symbols.

    UMBridge is deferred because importing umbridge pulls in aiohttp and
    other HTTP dependencies (~300ms), which penalizes users who never use
    the UMBridge client.  Other symbols in this module (WorkTracker, etc.)
    are lightweight and imported eagerly.
    """
    if name in ("UMBridgeModel", "UMBRIDGE_AVAILABLE"):
        from pyapprox.interface.umbridge import (
            UMBRIDGE_AVAILABLE as _UMBRIDGE_AVAILABLE,
        )
        from pyapprox.interface.umbridge import (
            UMBridgeModel as _UMBridgeModel,
        )

        globals()["UMBridgeModel"] = _UMBridgeModel
        globals()["UMBRIDGE_AVAILABLE"] = _UMBRIDGE_AVAILABLE
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
