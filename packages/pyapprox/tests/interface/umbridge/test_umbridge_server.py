"""Test UMBridge server for unit tests.

Provides models spanning the capability combinations the client must
handle: full first- and second-order support, evaluation only, and a
vector-valued output. Run this file directly to start the server::

    python test_umbridge_server.py [port]

The port is an argument because the test suite runs in parallel and a
fixed port would make concurrent runs collide.
"""

import sys
from typing import Any, Dict, List

try:
    import numpy as np
    import umbridge

    UMBRIDGE_AVAILABLE = True
except ImportError:
    UMBRIDGE_AVAILABLE = False


if UMBRIDGE_AVAILABLE:

    class QuadraticModel(umbridge.Model):  # type: ignore[misc]
        """f(x) = sum(x_i^2): gradient, apply_jacobian and apply_hessian.

        Every derivative is exact and known in closed form, so the
        client's decoding can be checked against analytic values rather
        than against a finite-difference approximation.
        """

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

        def apply_jacobian(
            self,
            out_wrt: int,
            in_wrt: int,
            parameters: List[List[float]],
            vec: List[float],
            config: Dict[str, Any],
        ) -> List[float]:
            x = np.asarray(parameters[0])
            v = np.asarray(vec)
            # J v = 2 x . v, a single output
            return [float(np.dot(2 * x, v))]

        def supports_apply_jacobian(self) -> bool:
            return True

        def apply_hessian(
            self,
            out_wrt: int,
            in_wrt1: int,
            in_wrt2: int,
            parameters: List[List[float]],
            sens: List[float],
            vec: List[float],
            config: Dict[str, Any],
        ) -> List[float]:
            v = np.asarray(vec)
            # Hessian of sum(x_i^2) is 2I, so H v = 2 v, scaled by the
            # adjoint seed on the single output.
            hv: List[float] = (2 * v * sens[0]).tolist()
            return hv

        def supports_apply_hessian(self) -> bool:
            return True

    class LinearModel(umbridge.Model):  # type: ignore[misc]
        """f(x) = sum(x_i): evaluation only, no derivative capability."""

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

    class ShortHessianModel(umbridge.Model):  # type: ignore[misc]
        """A model whose ApplyHessian returns an output-sized vector.

        H v has nvars entries, but a model written against the Python
        reference server's length check returns nqoi of them instead.
        This model does exactly that, so the client's own check on the
        reply length has something to catch.

        Declaring nqoi == 1 keeps the client's scalar-output
        precondition satisfied, and declaring nvars == 1 keeps the
        Python server's reply-length check satisfied, so the malformed
        reply reaches the client instead of being rejected en route.
        The client is then pointed at a larger nvars through config to
        make the received length wrong.
        """

        def __init__(self) -> None:
            super().__init__("short_hessian")

        def get_input_sizes(self, config: Dict[str, Any]) -> List[int]:
            return [config.get("nvars", 1)]

        def get_output_sizes(self, config: Dict[str, Any]) -> List[int]:
            return [1]

        def __call__(
            self, parameters: List[List[float]], config: Dict[str, Any]
        ) -> List[List[float]]:
            x = np.asarray(parameters[0])
            return [[float(np.sum(x**2))]]

        def supports_evaluate(self) -> bool:
            return True

        def apply_hessian(
            self,
            out_wrt: int,
            in_wrt1: int,
            in_wrt2: int,
            parameters: List[List[float]],
            sens: List[float],
            vec: List[float],
            config: Dict[str, Any],
        ) -> List[float]:
            # One entry regardless of nvars: the malformed reply.
            return [2.0 * vec[0] * sens[0]]

        def supports_apply_hessian(self) -> bool:
            return True

    class VectorModel(umbridge.Model):  # type: ignore[misc]
        """f(x) = [sum(x_i^2), sum(x_i)]: two outputs.

        Exercises the paths that a single-output model cannot reach: the
        client assembling a (nqoi, nsamples) array, and building a
        jacobian one output row at a time.
        """

        def __init__(self) -> None:
            super().__init__("vector")

        def get_input_sizes(self, config: Dict[str, Any]) -> List[int]:
            return [config.get("nvars", 2)]

        def get_output_sizes(self, config: Dict[str, Any]) -> List[int]:
            return [2]

        def __call__(
            self, parameters: List[List[float]], config: Dict[str, Any]
        ) -> List[List[float]]:
            x = np.asarray(parameters[0])
            return [[float(np.sum(x**2)), float(np.sum(x))]]

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
            # Row 0 is d(sum x^2)/dx = 2x; row 1 is d(sum x)/dx = 1
            rows = np.stack([2 * x, np.ones_like(x)])
            grad: List[float] = (sens @ rows).tolist()
            return grad

        def supports_gradient(self) -> bool:
            return True


if __name__ == "__main__":
    if not UMBRIDGE_AVAILABLE:
        print("umbridge not available, cannot start server")
        exit(1)

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4242
    models = [
        QuadraticModel(),
        LinearModel(),
        ShortHessianModel(),
        VectorModel(),
    ]
    print(f"Starting UMBridge test server on port {port}...")
    umbridge.serve_models(models, port)
