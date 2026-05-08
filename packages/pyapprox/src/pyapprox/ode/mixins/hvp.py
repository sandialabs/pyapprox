"""HVP mixins for Hessian-vector product computation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, cast

from pyapprox.ode.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
)
from pyapprox.util.backends.protocols import Array


class HVPMixin(ABC, Generic[Array]):
    """Mixin providing four HVP methods for second-order adjoint computation.

    Requires the underlying ODE residual to support HVP methods
    (ODEResidualWithHVPProtocol). Enforced by the concrete stepper's
    __init__ accepting the narrower type.
    """

    if TYPE_CHECKING:
        _residual: ODEResidualProtocol[Array]

    @property
    def _hvp_residual(self) -> ODEResidualWithHVPProtocol[Array]:
        """Typed access to the ODE residual as HVP capable.

        Safe because concrete stepper __init__ enforces this type.
        """
        return cast(ODEResidualWithHVPProtocol[Array], self._residual)

    @property
    def native_residual(self) -> ODEResidualWithHVPProtocol[Array]:
        """Get the underlying ODE residual (narrowed to HVP capable)."""
        return self._hvp_residual

    @abstractmethod
    def state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R/dy^2)w contracted with adjoint."""
        ...

    @abstractmethod
    def state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R/dy dp)v contracted with adjoint."""
        ...

    @abstractmethod
    def param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R/dp dy)w contracted with adjoint."""
        ...

    @abstractmethod
    def param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R/dp^2)v contracted with adjoint."""
        ...


class PrevStepHVPMixin(ABC, Generic[Array]):
    """Mixin for cross-step HVP methods (Crank-Nicolson only).

    Provides prev_state_state_hvp, prev_state_param_hvp, prev_param_state_hvp
    for schemes where R_{n+1} depends on f(y_n).
    """

    if TYPE_CHECKING:
        _residual: ODEResidualProtocol[Array]

    @property
    def _hvp_residual(self) -> ODEResidualWithHVPProtocol[Array]:
        """Typed access to the ODE residual as HVP capable."""
        return cast(ODEResidualWithHVPProtocol[Array], self._residual)

    @abstractmethod
    def prev_state_state_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dy_n^2)w contracted with adjoint."""
        ...

    @abstractmethod
    def prev_state_param_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dy_n dp)v contracted with adjoint."""
        ...

    @abstractmethod
    def prev_param_state_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dp dy_n)w contracted with adjoint."""
        ...
