"""Sensitivity mixin for forward sensitivity analysis."""

from abc import ABC, abstractmethod
from typing import Generic

from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import ActionOperatorProtocol, FormatProtocol


class SensitivityMixin(ABC, Generic[Array]):
    """Mixin providing sensitivity protocol methods.

    Declares is_explicit, has_prev_state_hessian, and off_diag_operator
    as abstract methods that concrete steppers must implement.
    """

    @abstractmethod
    def is_explicit(self) -> bool:
        """Return True if the time stepping scheme is explicit."""
        ...

    @abstractmethod
    def has_prev_state_hessian(self) -> bool:
        """Return True if R_{n+1} depends on f(y_n)."""
        ...

    @abstractmethod
    def off_diag_operator(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> ActionOperatorProtocol[Array, FormatProtocol]:
        r""":math:`B_n = dR_n/dy_{n-1}` as an operator; the forward
        sensitivity sweep applies it, the adjoint sweep its transpose."""
        ...
