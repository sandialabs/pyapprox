"""Convergence diagnostic protocols for variational inference.

Provides a protocol for convergence checks and a result dataclass.
Implementations can assess whether further VI optimization iterations
will meaningfully improve the posterior approximation.
"""

from dataclasses import dataclass, field
from typing import Dict, Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array


@dataclass
class ConvergenceCheckResult:
    """Result from a convergence diagnostic check.

    Attributes
    ----------
    should_stop : bool
        Whether the optimizer should stop early.
    approximation_quality : float
        Quality metric in [0, 1] where 1 is perfect.
    detail : dict
        Implementation-specific diagnostic details.
    check_type : str
        Name of the check that produced this result.
    """

    should_stop: bool
    approximation_quality: float
    detail: Dict[str, float] = field(default_factory=dict)
    check_type: str = ""


@runtime_checkable
class ConvergenceCheckProtocol(Protocol, Generic[Array]):
    """Protocol for VI convergence diagnostic checks.

    Implementations assess whether further optimization iterations will
    meaningfully improve the variational approximation. The ``check``
    method is called periodically during optimization.
    """

    def check(
        self,
        params: Array,
        recent_elbo_improvement: float,
    ) -> ConvergenceCheckResult:
        """Run the convergence diagnostic.

        Parameters
        ----------
        params : Array
            Current variational parameters, shape ``(nvars, 1)``.
        recent_elbo_improvement : float
            Recent ELBO improvement (positive = improving).

        Returns
        -------
        ConvergenceCheckResult
            Diagnostic result with stop recommendation.
        """
        ...
