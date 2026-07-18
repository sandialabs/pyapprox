"""HVP mixin for Hessian-vector product computation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


class HVPMixin(ABC, Generic[Array]):
    """Mixin providing HVP methods for second-order adjoint computation.

    Provides four same-step HVP methods and three cross-step prev_*
    methods. All steppers implement prev_* (BE returns zero, FE/Heun
    delegate to state_*_hvp).

    Requires the underlying ODE residual to support HVP methods
    (ODEResidualWithHVPProtocol). Enforced by the concrete stepper's
    __init__ accepting the narrower type.
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]
        _residual: ODEResidualProtocol[Array]

    @property
    def _hvp_residual(self) -> ODEResidualWithHVPProtocol[Array]:
        """Narrow the ODE residual to HVP capable, or raise.

        Capability narrowing is lazy: constructing an HVP-tier stepper
        over a residual without second-order derivatives is legal until
        the first HVP method call reaches this accessor.
        """
        residual = self._residual
        if not isinstance(residual, ODEResidualWithHVPProtocol):
            raise TypeError(
                f"{type(self).__name__} HVP methods require an ODE "
                "residual with the four HVP contractions; got "
                f"{type(residual).__name__} (construct the residual with "
                "a parameterization providing second-order derivatives)"
            )
        return residual

    # native_residual deliberately NOT overridden here: general access
    # narrows only to the adjoint tier (AdjointMixin), so gradient-only
    # flows on an HVP-class stepper never demand second-order
    # capability. HVP-internal calls use _hvp_residual, which narrows
    # (and raises actionably) at the first genuine HVP call.

    # -- Same-step HVP methods --

    @abstractmethod
    def state_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R/dy^2)w contracted with adjoint."""
        ...

    @abstractmethod
    def state_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R/dy dp)v contracted with adjoint."""
        ...

    @abstractmethod
    def param_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R/dp dy)w contracted with adjoint."""
        ...

    @abstractmethod
    def param_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R/dp^2)v contracted with adjoint."""
        ...

    # -- Cross-step HVP methods --

    @abstractmethod
    def prev_state_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dy_n^2)w contracted with adjoint."""
        ...

    @abstractmethod
    def prev_state_param_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dy_n dp)v contracted with adjoint."""
        ...

    @abstractmethod
    def prev_param_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dp dy_n)w contracted with adjoint."""
        ...

    def state_prev_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec_prev: Array,
    ) -> Array:
        """Compute adj^T * d^2R_n/(dy_n dy_{n-1}) * w_{n-1}.

        Nonzero only when y_n and y_{n-1} enter f through a shared
        evaluation point (e.g. implicit midpoint). Default returns zero.
        """
        return self._bkd.zeros(wvec_prev.shape)

    def prev_state_curr_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec_curr_of_next: Array,
    ) -> Array:
        """Compute adj^T * d^2R_{n+1}/(dy_n dy_{n+1}) * w_{n+1}.

        By Schwarz symmetry same bilinear form as state_prev_state_hvp
        but evaluated at R_{n+1}'s midpoint. Default returns zero.
        """
        return self._bkd.zeros(wvec_curr_of_next.shape)

