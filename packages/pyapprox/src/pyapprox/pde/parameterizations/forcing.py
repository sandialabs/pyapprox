"""ForcingParameterization: binds a FieldMap to forcing term."""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
    FieldMapWithHVPProtocol,
    field_map_has_hvp,
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
        # Second order when the field map has a usable hvp; narrowed
        # ONCE here so the HVP methods keep a typed reference. Forcing
        # enters the residual additively (dR/dF = I), so both mixed
        # contractions are identically zero and all parameter curvature
        # is the field map's.
        self._hvp_field_map: Optional[FieldMapWithHVPProtocol[Array]] = None
        if field_map_has_hvp(field_map) and isinstance(
            field_map, FieldMapWithHVPProtocol
        ):
            self._hvp_field_map = field_map
            self._derivs: ParamDerivatives[Array] = (
                ParamDerivatives.second_order(
                    self.param_jacobian,
                    self.initial_param_jacobian,
                    self._param_param_hvp,
                    self._state_param_hvp,
                    self._param_state_hvp,
                )
            )
        else:
            self._derivs = ParamDerivatives.first_order(
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

    def _require_hvp_field_map(self) -> FieldMapWithHVPProtocol[Array]:
        field_map = self._hvp_field_map
        if field_map is None:
            raise RuntimeError(
                "HVP methods are unavailable; check param_derivatives() "
                "before calling"
            )
        return field_map

    def _param_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2R/dp^2) v = field-map hvp with weights adj.

        dR/dF = I, so the adjoint pulls back unchanged.
        Shape: (nparams,).
        """
        return self._require_hvp_field_map().hvp(params_1d, adj_state, vvec)

    def _state_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2R/du dp) v = 0 (forcing is state-independent).

        Shape: (npts,).
        """
        return self._bkd.zeros((self._physics.npts(),))

    def _param_state_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """adj^T (d^2R/dp du) w = 0 (forcing is state-independent).

        Shape: (nparams,).
        """
        return self._bkd.zeros((self.nparams(),))
