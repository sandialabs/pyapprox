"""Sensitivity mixin for forward sensitivity analysis."""

from abc import ABC, abstractmethod
from typing import Generic

from pyapprox.util.backends.protocols import Array


class SensitivityMixin(ABC, Generic[Array]):
    """Mixin providing sensitivity protocol methods.

    Declares is_explicit, has_prev_state_hessian, and
    sensitivity_off_diag_jacobian as abstract methods that concrete
    steppers must implement.
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
    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        """Compute dR_n/dy_{n-1} for forward sensitivity propagation."""
        ...
