"""ReactionParameterization: binds a FieldMap to reaction coefficient."""

from typing import Callable, Generic, Optional

from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ReactionParameterization(Generic[Array]):
    """Parameterization that maps parameters to reaction coefficient.

    Parameters
    ----------
    field_map : FieldMapProtocol
        Maps parameter vector to reaction field.
    bkd : Backend
        Computational backend.
    time_modulation : Callable[[float], float], optional
        Time modulation factor. Defaults to constant 1.0.
    """

    def __init__(
        self,
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
        time_modulation: Optional[Callable[[float], float]] = None,
    ) -> None:
        if not isinstance(field_map, FieldMapProtocol):
            raise TypeError(
                f"field_map must satisfy FieldMapProtocol, "
                f"got {type(field_map).__name__}"
            )
        self._field_map = field_map
        self._bkd = bkd
        self._time_mod = (
            time_modulation if time_modulation is not None else lambda t: 1.0
        )

        # Dynamic binding: param_jacobian only if field_map has jacobian
        if hasattr(self._field_map, "jacobian"):
            self.param_jacobian = self._param_jacobian
            self.initial_param_jacobian = self._initial_param_jacobian

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nparams(self) -> int:
        return self._field_map.nvars()

    def apply(self, physics: object, params_1d: Array) -> None:
        """Apply parameterization: set reaction field on physics."""
        field = self._field_map(params_1d)
        physics.set_reaction(lambda t, _f=field: _f * self._time_mod(t))

    def _param_jacobian(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Compute d(residual)/d(params) via chain rule. Shape: (npts, nparams)."""
        fm_jac = self._field_map.jacobian(params_1d)  # (npts, nparams)
        mod = self._time_mod(time)
        dr_dr = physics.residual_reaction_sensitivity(state, time)  # (npts,)
        npts = state.shape[0]
        nparams = self.nparams()
        scaled_jac = fm_jac * mod
        result = self._bkd.zeros((npts, nparams))
        result = self._bkd.copy(result)
        for i in range(nparams):
            col = dr_dr * scaled_jac[:, i]
            for j in range(npts):
                result[j, i] = col[j]
        return result

    def _initial_param_jacobian(self, physics: object, params_1d: Array) -> Array:
        """Return d(initial_state)/d(params). Shape: (nstates, nparams)."""
        npts = physics.npts()
        return self._bkd.zeros((npts, self.nparams()))
