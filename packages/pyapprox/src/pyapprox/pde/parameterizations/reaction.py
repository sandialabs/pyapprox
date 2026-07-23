"""ReactionParameterization: binds a FieldMap to reaction coefficient."""

from typing import Callable, Generic, Optional, Protocol, runtime_checkable

from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
    FieldMapWithHVPProtocol,
    field_map_has_hvp,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.fields import ConstantInTimeField
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class _CollocationReactionPhysicsProtocol(Protocol, Generic[Array]):
    """Physics members this parameterization calls (interim; the typed
    facades of the parameterization redesign replace it)."""

    def npts(self) -> int: ...

    def set_reaction(self, reaction: Callable[[float], Array]) -> None: ...

    def residual_reaction_sensitivity(
        self, state: Array, time: float
    ) -> Array: ...


class ReactionParameterization(Generic[Array]):
    """Parameterization that maps parameters to reaction coefficient.

    The physics is bound at construction: one instance serves one
    physics (ensembles construct one parameterization per physics).

    Parameters
    ----------
    physics : _CollocationReactionPhysicsProtocol
        Collocation physics with ``set_reaction``,
        ``residual_reaction_sensitivity``, and ``npts`` members.
    field_map : FieldMapProtocol
        Maps parameter vector to reaction field.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: _CollocationReactionPhysicsProtocol[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, _CollocationReactionPhysicsProtocol):
            raise TypeError(
                f"physics must provide set_reaction/"
                f"residual_reaction_sensitivity/npts, "
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
        # ONCE here so the HVP methods keep a typed reference. The
        # reaction enters as r(x) * u (residual_reaction_sensitivity's
        # "pointwise = state" contract), so the mixed tensor
        # d^2R_j/du_i dr_k = delta_jk delta_ij is pointwise-diagonal
        # and all contractions reduce to elementwise products.
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

    def physics(self) -> _CollocationReactionPhysicsProtocol[Array]:
        """Return the bound physics instance."""
        return self._physics

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return self._derivs

    def nparams(self) -> int:
        return self._field_map.nvars()

    def apply(self, params_1d: Array) -> None:
        """Apply parameterization: set reaction field on physics."""
        field = self._field_map(params_1d)
        self._physics.set_reaction(ConstantInTimeField(field))

    def param_jacobian(
        self,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Compute d(residual)/d(params) via chain rule. Shape: (npts, nparams)."""
        fm_jac = self._field_map.jacobian(params_1d)  # (npts, nparams)
        dr_dr = self._physics.residual_reaction_sensitivity(
            state, time
        )  # (npts,)
        npts = state.shape[0]
        nparams = self.nparams()
        result = self._bkd.zeros((npts, nparams))
        result = self._bkd.copy(result)
        for i in range(nparams):
            col = dr_dr * fm_jac[:, i]
            for j in range(npts):
                result[j, i] = col[j]
        return result

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
        """adj^T (d^2R/dp^2) v via the field map's curvature.

        R is affine in the reaction field with dR/dr = diag(u), so the
        field-space weights are u * adj. Shape: (nparams,).
        """
        weights = (
            self._physics.residual_reaction_sensitivity(state, time)
            * adj_state
        )
        return self._require_hvp_field_map().hvp(params_1d, weights, vvec)

    def _state_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2R/du dp) v = adj * (d(r)/dp v), state-shaped.

        Pointwise-diagonal mixed tensor (see class docstring comment).
        Shape: (npts,).
        """
        delta_r = self._field_map.jacobian(params_1d) @ vvec
        return adj_state * delta_r

    def _param_state_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """adj^T (d^2R/dp du) w = fm_jac^T (adj * w), param-shaped.

        Transpose contraction of ``_state_param_hvp``.
        Shape: (nparams,).
        """
        return self._field_map.jacobian(params_1d).T @ (adj_state * wvec)
