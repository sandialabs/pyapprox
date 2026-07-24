"""Private field-parameterization derivative engine (framework-internal).

One class, ``_FieldParameterizationTerm``, holds ALL the derivative
calculus for a parameterized coefficient field exactly once: the chain
rule through the field map, the linearity identities, and the map
curvature. Physics classes expose typed field-derivative assemblies
(``residual_<field>_jacobian`` etc. — weak-form self-knowledge, no
parameter concepts); per-physics facades wire those bound methods into
terms; users never see this module.

Vocabulary: with residual term :math:`R(u, g)` for field DOFs ``g`` and
field map :math:`g = G(p)`,

- ``field_jacobian`` :math:`S(u, t) = \\partial R/\\partial g`
  (shape ``(nstates, nfield)``, sparse at the assembly seam),
- ``field_state_jacobian``
  :math:`A(\\delta g, u, t) = \\partial [S(u) \\delta g]/\\partial u`
  (shape ``(nstates, nstates)``),

and the engine produces the RAW (no Dirichlet) ``ParamDerivatives``
bundle callables:

.. math::

    dR/dp &= S(u) G'(p) \\\\
    \\lambda^T (d^2R/dp^2) v &= \\text{field\\_map.hvp}(p, S(u)^T
        \\lambda, v) + G'^T \\, [\\lambda^T \\partial^2 R/\\partial g^2]
        (G' v) \\\\
    \\lambda^T (d^2R/du \\, dp) v &= A(G' v, u)^T \\lambda \\\\
    \\lambda^T (d^2R/dp \\, du) w &= G'^T S(w)^T \\lambda
        \\quad (\\text{linearity identity})

Second-derivative slots are REQUIRED and three-valued (``Zero()``,
``FromLinearity()``, or a callable) so every term's dependency
structure is stated explicitly at the construction site.
"""

from typing import Callable, Generic, Optional, Union

import numpy as np
from scipy.sparse import spmatrix

from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
    FieldMapWithHVPProtocol,
    field_map_has_hvp,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Backend


class Zero:
    """Sentinel: the derivative is certified identically zero.

    A STRUCTURAL identity of the weak form — "this derivative is zero
    for all states, all field values, and all p" — decided by
    inspecting the weak-form term, never by inspecting current values.
    The engine skips assembly and accumulation entirely (no zero
    arrays, no axpy). Do NOT use it for a zero-VALUED coefficient or a
    structurally absent term.
    """


class FromLinearity:
    """Sentinel: derive the mixed contraction from first-order data.

    Valid ONLY when the field-carrying term is linear in the state
    with :math:`S(0) = 0`: then the mixed tensor is state-independent
    and its two contractions follow from ``field_jacobian`` evaluated
    at direction vectors (param-shaped side) and from
    ``field_state_jacobian`` (state-shaped side).
    """


# Signatures use the pde 1D-array convention; ``time`` is threaded
# everywhere (coefficients are constant in time today — D9.6).
FieldJacobianFn = Callable[[Array, float], Union[spmatrix, Array]]
FieldStateJacobianFn = Callable[[Array, Array, float], Union[spmatrix, Array]]
FieldShapedHVPFn = Callable[[Array, float, Array, Array], Array]
StateShapedHVPFn = Callable[[Array, float, Array, Array], Array]

_FieldStateSlot = Union[Zero, FromLinearity, FieldShapedHVPFn[Array]]
_StateFieldSlot = Union[Zero, FromLinearity, StateShapedHVPFn[Array]]
_FieldFieldSlot = Union[Zero, FieldShapedHVPFn[Array]]


class _FieldParameterizationTerm(Generic[Array]):
    """One parameterized coefficient field's derivative calculus.

    Framework-internal: constructed by facades from the physics's bound
    typed field-derivative methods. Produces the RAW ``ParamDerivatives``
    bundle (D9.7: the BC-enforcing wrapper/adapters own all Dirichlet
    handling).

    Parameters
    ----------
    setter : Callable[[Array], None]
        Sets the field DOFs on the physics (invalidating its caches).
    field_jacobian : FieldJacobianFn
        :math:`S(u, t) = dR/d(\\text{field DOFs})`.
        Shape: (nstates, nfield).
    field_state_hvp : Zero | FromLinearity | callable
        Param-shaped mixed contraction
        :math:`\\lambda^T (\\partial^2 R/\\partial g \\partial u) w`
        in FIELD space: callable(state, time, adj, wvec) -> (nfield,).
    state_field_hvp : Zero | FromLinearity | callable
        State-shaped mixed contraction
        :math:`[\\partial^2 R/\\partial u \\partial g \\, \\delta g]^T
        \\lambda`: callable(state, time, adj, delta_field) ->
        (nstates,). ``FromLinearity`` requires ``field_state_jacobian``.
    field_field_hvp : Zero | callable
        Field-shaped curvature contraction
        :math:`[\\lambda^T \\partial^2 R/\\partial g^2](\\delta g)`:
        callable(state, time, adj, delta_field) -> (nfield,).
        (``FromLinearity`` is meaningless here: linearity in g makes
        this exactly ``Zero()``.)
    field_map : FieldMapProtocol
        Map from parameters to field DOFs. A usable ``hvp`` makes the
        bundle second order.
    bkd : Backend
        Computational backend.
    nstates : int
        Number of state DOFs (initial_param_jacobian shape).
    nfield_dofs : int
        Expected field length; ``apply`` validates against it. Differs
        from ``nstates`` for blocked fields (e.g. velocity).
    field_state_jacobian : FieldStateJacobianFn, optional
        :math:`A(\\delta g, u, t)`. Required when ``state_field_hvp``
        is ``FromLinearity()``.
    require_positive : bool, default False
        If True, ``apply`` raises when the mapped field is not strictly
        positive everywhere.
    """

    def __init__(
        self,
        setter: Callable[[Array], None],
        field_jacobian: FieldJacobianFn[Array],
        field_state_hvp: _FieldStateSlot[Array],
        state_field_hvp: _StateFieldSlot[Array],
        field_field_hvp: _FieldFieldSlot[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
        nstates: int,
        nfield_dofs: int,
        field_state_jacobian: Optional[FieldStateJacobianFn[Array]] = None,
        require_positive: bool = False,
    ) -> None:
        if not isinstance(field_map, FieldMapProtocol):
            raise TypeError(
                "field_map must satisfy FieldMapProtocol, got "
                f"{type(field_map).__name__}"
            )
        for name, slot in (
            ("field_state_hvp", field_state_hvp),
            ("state_field_hvp", state_field_hvp),
        ):
            if not isinstance(slot, (Zero, FromLinearity)) and not callable(
                slot
            ):
                raise TypeError(
                    f"{name} must be Zero(), FromLinearity(), or a "
                    f"callable, got {type(slot).__name__}"
                )
        if isinstance(field_field_hvp, FromLinearity):
            raise TypeError(
                "field_field_hvp cannot be FromLinearity(): linearity "
                "in the field makes it exactly Zero()"
            )
        if not isinstance(field_field_hvp, Zero) and not callable(
            field_field_hvp
        ):
            raise TypeError(
                "field_field_hvp must be Zero() or a callable, got "
                f"{type(field_field_hvp).__name__}"
            )
        if (
            isinstance(state_field_hvp, FromLinearity)
            and field_state_jacobian is None
        ):
            raise TypeError(
                "state_field_hvp=FromLinearity() requires "
                "field_state_jacobian (the mixed assembly A(delta_g, u)) "
                "— the state-shaped contraction cannot be derived from "
                "field_jacobian alone without symmetry assumptions"
            )
        self._setter = setter
        self._field_jacobian = field_jacobian
        self._field_state_hvp = field_state_hvp
        self._state_field_hvp = state_field_hvp
        self._field_field_hvp = field_field_hvp
        self._field_map = field_map
        self._bkd = bkd
        self._nstates = nstates
        self._nfield_dofs = nfield_dofs
        self._field_state_jacobian = field_state_jacobian
        self._require_positive = require_positive

        self._hvp_field_map: Optional[FieldMapWithHVPProtocol[Array]] = None
        if field_map_has_hvp(field_map) and isinstance(
            field_map, FieldMapWithHVPProtocol
        ):
            self._hvp_field_map = field_map
            self._derivs: ParamDerivatives[Array] = (
                ParamDerivatives.second_order(
                    self.param_jacobian,
                    self.initial_param_jacobian,
                    self.param_param_hvp,
                    self.state_param_hvp,
                    self.param_state_hvp,
                )
            )
        else:
            self._derivs = ParamDerivatives.first_order(
                self.param_jacobian,
                self.initial_param_jacobian,
            )

    # -- named constructors (sugar over the one class, never subclasses)

    @staticmethod
    def linear_field_state(
        setter: Callable[[Array], None],
        field_jacobian: FieldJacobianFn[Array],
        field_state_jacobian: FieldStateJacobianFn[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
        nstates: int,
        nfield_dofs: int,
        require_positive: bool = False,
    ) -> "_FieldParameterizationTerm[Array]":
        """Term linear in the field AND the state (e.g. kappa grad u,
        r*u): slots (FromLinearity, FromLinearity, Zero)."""
        return _FieldParameterizationTerm(
            setter,
            field_jacobian,
            FromLinearity(),
            FromLinearity(),
            Zero(),
            field_map,
            bkd,
            nstates,
            nfield_dofs,
            field_state_jacobian=field_state_jacobian,
            require_positive=require_positive,
        )

    @staticmethod
    def state_independent(
        setter: Callable[[Array], None],
        field_jacobian: FieldJacobianFn[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
        nstates: int,
        nfield_dofs: int,
        require_positive: bool = False,
    ) -> "_FieldParameterizationTerm[Array]":
        """Term depending on the field only (e.g. forcing): slots
        (Zero, Zero, Zero)."""
        return _FieldParameterizationTerm(
            setter,
            field_jacobian,
            Zero(),
            Zero(),
            Zero(),
            field_map,
            bkd,
            nstates,
            nfield_dofs,
            require_positive=require_positive,
        )

    # -- parameterization surface (facades delegate here)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._field_map.nvars()

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the derivative capability bundle."""
        return self._derivs

    def apply(self, params_1d: Array) -> None:
        """Map parameters to field DOFs and set them on the physics."""
        field = self._field_map(params_1d)
        if field.shape[0] != self._nfield_dofs:
            raise ValueError(
                f"field map produced {field.shape[0]} DOFs but the "
                f"physics field has {self._nfield_dofs}"
            )
        if self._require_positive:
            min_val = self._bkd.to_float(self._bkd.min(field))
            if min_val <= 0.0:
                raise ValueError(
                    "field must be positive at all DOFs; found min "
                    f"value {min_val:.2e}"
                )
        self._setter(field)

    # -- derivative calculus (exists exactly once, here)

    def _field_jacobian_np(self, params_1d: Array) -> np.ndarray:
        """d(field)/d(params) as numpy. Shape: (nfield, nparams)."""
        return np.asarray(
            self._bkd.to_numpy(self._field_map.jacobian(params_1d))
        )

    def _field_direction(self, params_1d: Array, vvec: Array) -> Array:
        """delta_g = G'(p) v. Shape: (nfield,)."""
        return self._bkd.asarray(
            self._field_jacobian_np(params_1d) @ self._bkd.to_numpy(vvec)
        )

    def param_jacobian(
        self, state: Array, time: float, params_1d: Array
    ) -> Array:
        """dR/dp = S(u) @ G'(p). Shape: (nstates, nparams)."""
        return self._bkd.asarray(
            self._field_jacobian(state, time)
            @ self._field_jacobian_np(params_1d)
        )

    def initial_param_jacobian(self, params_1d: Array) -> Array:
        """d(u_0)/dp = 0 (coefficient fields do not set the IC)."""
        return self._bkd.zeros((self._nstates, self.nparams()))

    def param_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2R/dp^2) v: map curvature + field curvature.

        Shape: (nparams,).
        """
        field_map = self._require_hvp_field_map()
        weights = self._bkd.asarray(
            self._field_jacobian(state, time).T
            @ self._bkd.to_numpy(adj_state)
        )
        out = field_map.hvp(params_1d, weights, vvec)
        ff = self._field_field_hvp
        if not isinstance(ff, Zero):
            delta = self._field_direction(params_1d, vvec)
            out = out + self._bkd.asarray(
                self._field_jacobian_np(params_1d).T
                @ self._bkd.to_numpy(ff(state, time, adj_state, delta))
            )
        return out

    def state_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2R/du dp) v = A(G'v, u)^T adj, state-shaped.

        Shape: (nstates,).
        """
        slot = self._state_field_hvp
        if isinstance(slot, Zero):
            return self._bkd.zeros((state.shape[0],))
        delta = self._field_direction(params_1d, vvec)
        if isinstance(slot, FromLinearity):
            mixed = self._require_field_state_jacobian()
            return self._bkd.asarray(
                mixed(delta, state, time).T @ self._bkd.to_numpy(adj_state)
            )
        return slot(state, time, adj_state, delta)

    def param_state_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """adj^T (d^2R/dp du) w = G'^T S(w)^T adj, param-shaped.

        The linearity identity: for terms linear in the state with
        S(0) = 0, the mixed tensor contracted with w equals
        ``field_jacobian`` evaluated at w. Shape: (nparams,).
        """
        slot = self._field_state_hvp
        if isinstance(slot, Zero):
            return self._bkd.zeros((self.nparams(),))
        if isinstance(slot, FromLinearity):
            weights = self._field_jacobian(
                wvec, time
            ).T @ self._bkd.to_numpy(adj_state)
        else:
            weights = self._bkd.to_numpy(
                slot(state, time, adj_state, wvec)
            )
        result: Array = self._bkd.asarray(
            self._field_jacobian_np(params_1d).T @ np.asarray(weights)
        )
        return result

    def _require_hvp_field_map(self) -> FieldMapWithHVPProtocol[Array]:
        field_map = self._hvp_field_map
        if field_map is None:
            raise RuntimeError(
                "HVP methods are unavailable; check param_derivatives() "
                "before calling"
            )
        return field_map

    def _require_field_state_jacobian(self) -> FieldStateJacobianFn[Array]:
        mixed = self._field_state_jacobian
        if mixed is None:
            raise RuntimeError(
                "field_state_jacobian is required for the "
                "FromLinearity state-shaped contraction"
            )
        return mixed
