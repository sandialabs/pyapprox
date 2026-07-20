"""ForcingParameterization: binds a FieldMap to forcing term."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.fields import ConstantInTimeField
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class _CollocationForcingPhysicsProtocol(Protocol, Generic[Array]):
    """Physics members this parameterization calls (interim; the typed
    facades of the parameterization redesign replace it).

    ``set_forcing`` is declared against the concrete field callable this
    parameterization passes (keeps ``Array`` in an invariant position;
    real physics accept any ``Callable[[float], Array]``, which is
    structurally wider).
    """

    def npts(self) -> int: ...

    def set_forcing(self, forcing: ConstantInTimeField[Array]) -> None: ...


class ForcingParameterization(Generic[Array]):
    """Parameterization that maps parameters to forcing term.

    The physics is bound at construction: one instance serves one
    physics (ensembles construct one parameterization per physics).

    Parameters
    ----------
    physics : _CollocationForcingPhysicsProtocol
        Collocation physics with ``set_forcing`` and ``npts`` members.
    field_map : FieldMapProtocol
        Maps parameter vector to forcing field.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: _CollocationForcingPhysicsProtocol[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, _CollocationForcingPhysicsProtocol):
            raise TypeError(
                f"physics must provide set_forcing/npts, "
                f"got {type(physics).__name__}"
            )
        if not isinstance(field_map, FieldMapProtocol):
            raise TypeError(
                f"field_map must satisfy FieldMapProtocol, "
                f"got {type(field_map).__name__}"
            )
        self._physics = physics
        self._field_map = field_map
        self._bkd = bkd
        self._derivs: ParamDerivatives[Array] = ParamDerivatives.first_order(
            self.param_jacobian,
            self.initial_param_jacobian,
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def physics(self) -> _CollocationForcingPhysicsProtocol[Array]:
        """Return the bound physics instance."""
        return self._physics

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return self._derivs

    def nparams(self) -> int:
        return self._field_map.nvars()

    def apply(self, params_1d: Array) -> None:
        """Apply parameterization: set forcing field on physics."""
        field = self._field_map(params_1d)
        self._physics.set_forcing(ConstantInTimeField(field))

    def param_jacobian(
        self,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Compute d(residual)/d(params). Shape: (npts, nparams).

        Forcing enters residual linearly: d(residual)/d(f) = I,
        so d(residual)/d(params) = fm_jac.
        """
        return self._field_map.jacobian(params_1d)  # (npts, nparams)

    def initial_param_jacobian(self, params_1d: Array) -> Array:
        """Return d(initial_state)/d(params). Shape: (nstates, nparams)."""
        npts = self._physics.npts()
        return self._bkd.zeros((npts, self.nparams()))
