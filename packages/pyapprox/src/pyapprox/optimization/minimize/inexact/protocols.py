"""Protocols for inexact gradient evaluation.

Defines structural protocols for:
- InexactGradientStrategyProtocol: maps tolerance to (samples, weights)
- InexactEvaluable: tolerance-dependent function evaluation
- InexactDifferentiable: tolerance-dependent jacobian evaluation
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class InexactGradientStrategyProtocol(Protocol, Generic[Array]):
    """Strategy for mapping ROL tolerance to quadrature samples/weights.

    Contract for ``tol`` semantics:

    - ``tol > 0``: return samples/weights sufficient to approximate the
      integral to accuracy ~``tol``.
    - ``tol <= 0``: return **all available** samples/weights (maximum
      fidelity).
    - Strategies must enforce internal caps (``max_samples``,
      ``max_level``) to prevent runaway computation when ROL passes
      near-zero tolerances.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of random variables."""
        ...

    def samples_and_weights(self, tol: float) -> Tuple[Array, Array]:
        """Return samples and weights for the given tolerance.

        Parameters
        ----------
        tol : float
            Accuracy tolerance from ROL. Smaller means more samples.

        Returns
        -------
        Tuple[Array, Array]
            ``(samples, weights)`` with shapes
            ``(n_random_vars, n_points)`` and ``(n_points,)``.
        """
        ...


@runtime_checkable
class InexactEvaluable(Protocol, Generic[Array]):
    """Protocol for objects supporting tolerance-dependent evaluation."""

    def inexact_value(self, sample: Array, tol: float) -> Array:
        """Evaluate with tolerance-dependent accuracy.

        Parameters
        ----------
        sample : Array
            Input sample. Shape ``(nvars, 1)``.
        tol : float
            Accuracy tolerance from ROL.

        Returns
        -------
        Array
            Function value. Shape ``(nqoi, 1)``.
        """
        ...


@runtime_checkable
class InexactDifferentiable(Protocol, Generic[Array]):
    """Protocol for objects supporting tolerance-dependent jacobian."""

    def inexact_jacobian(self, sample: Array, tol: float) -> Array:
        """Compute jacobian with tolerance-dependent accuracy.

        Parameters
        ----------
        sample : Array
            Input sample. Shape ``(nvars, 1)``.
        tol : float
            Accuracy tolerance from ROL.

        Returns
        -------
        Array
            Jacobian. Shape ``(nqoi, nvars)``.
        """
        ...
