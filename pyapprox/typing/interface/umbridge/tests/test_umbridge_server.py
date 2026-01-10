"""Test UMBridge server for unit tests.

This module provides a simple UMBridge model for testing purposes.
Run this file directly to start the server:
    python test_umbridge_server.py
"""

from typing import Any, Dict, List

try:
    import umbridge
    import numpy as np

    UMBRIDGE_AVAILABLE = True
except ImportError:
    UMBRIDGE_AVAILABLE = False


if UMBRIDGE_AVAILABLE:

    class QuadraticModel(umbridge.Model):  # type: ignore[misc]
        """Simple quadratic test model: f(x) = sum(x_i^2)."""

        def __init__(self) -> None:
            super().__init__("quadratic")

        def get_input_sizes(self, config: Dict[str, Any]) -> List[int]:
            return [config.get("nvars", 2)]

        def get_output_sizes(self, config: Dict[str, Any]) -> List[int]:
            return [1]

        def __call__(
            self, parameters: List[List[float]], config: Dict[str, Any]
        ) -> List[List[float]]:
            x = np.asarray(parameters[0])
            result = float(np.sum(x**2))
            return [[result]]

        def supports_evaluate(self) -> bool:
            return True

        def gradient(
            self,
            out_wrt: int,
            in_wrt: int,
            parameters: List[List[float]],
            sens: List[float],
            config: Dict[str, Any],
        ) -> List[float]:
            x = np.asarray(parameters[0])
            # Gradient of sum(x_i^2) is 2*x
            grad: List[float] = (2 * x * sens[0]).tolist()
            return grad

        def supports_gradient(self) -> bool:
            return True

    class LinearModel(umbridge.Model):  # type: ignore[misc]
        """Simple linear test model: f(x) = sum(x_i)."""

        def __init__(self) -> None:
            super().__init__("linear")

        def get_input_sizes(self, config: Dict[str, Any]) -> List[int]:
            return [config.get("nvars", 2)]

        def get_output_sizes(self, config: Dict[str, Any]) -> List[int]:
            return [1]

        def __call__(
            self, parameters: List[List[float]], config: Dict[str, Any]
        ) -> List[List[float]]:
            x = np.asarray(parameters[0])
            result = float(np.sum(x))
            return [[result]]

        def supports_evaluate(self) -> bool:
            return True


if __name__ == "__main__":
    if not UMBRIDGE_AVAILABLE:
        print("umbridge not available, cannot start server")
        exit(1)

    models = [QuadraticModel(), LinearModel()]
    print("Starting UMBridge test server on port 4242...")
    umbridge.serve_models(models, 4242)
