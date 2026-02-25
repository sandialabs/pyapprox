"""UMBridge model interface.

This module provides a client for UMBridge (Unified Model Bridge) models,
which allows calling external models via HTTP.
"""

from pyapprox.interface.umbridge.client import (
    UMBRIDGE_AVAILABLE,
    UMBridgeModel,
)

__all__ = [
    "UMBridgeModel",
    "UMBRIDGE_AVAILABLE",
]
