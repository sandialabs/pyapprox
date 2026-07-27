"""Private field-parameterization derivative engine (framework-internal).

One class, ``_FieldParameterizationTerm``, holds ALL the derivative
calculus for a parameterized coefficient field exactly once: the chain
rule through the field map, the linearity identities, and the map
curvature. Physics classes expose typed field-derivative assemblies
(``residual_<field>_jacobian`` etc. — weak-form self-knowledge, no
parameter concepts); per-physics facades wire those bound methods into
terms; users never see this module. Third-party facades follow the
contract in ``docs/conventions/pde_solver_extension.md``.

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

from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

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
# everywhere (coefficients are constant in time today).
FieldJacobianFn = Callable[[Array, float], Union[spmatrix, Array]]
FieldStateJacobianFn = Callable[[Array, Array, float], Union[spmatrix, Array]]
FieldShapedHVPFn = Callable[[Array, float, Array, Array], Array]
StateShapedHVPFn = Callable[[Array, float, Array, Array], Array]
# (state, time, bc_indices, normals) -> (n_bc, nfield): the boundary
# rows' normal-flux derivative w.r.t. the coefficient field (only BCs
# whose normal operator has coefficient dependence supply one).
BCFluxFieldJacobianFn = Callable[
    [Array, float, Array, Array], Union[spmatrix, Array]
]

_FieldStateSlot = Union[Zero, FromLinearity, FieldShapedHVPFn[Array]]
_StateFieldSlot = Union[Zero, FromLinearity, StateShapedHVPFn[Array]]
_FieldFieldSlot = Union[Zero, FieldShapedHVPFn[Array]]

PhysicsT = TypeVar("PhysicsT")


# Slot adapters: parameterizations must be picklable, so facades wire
# physics methods through these module-level classes (a bound method
# pickles by object-reference + name; a lambda does not).


class ToNumpySetter(Generic[Array]):
    """Picklable setter adapter: backend values -> numpy -> bound setter."""

    def __init__(
        self,
        set_fn: Callable[[np.ndarray], None],
        bkd: Backend[Array],
    ) -> None:
        self._set_fn = set_fn
        self._bkd = bkd

    def __call__(self, values: Array) -> None:
        self._set_fn(np.asarray(self._bkd.to_numpy(values)))


class StateJacobianAdapter(Generic[Array]):
    """Adapts ``residual_<field>_jacobian(state)`` to ``(state, time)``."""

    def __init__(
        self, fn: Callable[[Array], Union[spmatrix, Array]]
    ) -> None:
        self._fn = fn

    def __call__(
        self, state: Array, time: float
    ) -> Union[spmatrix, Array]:
        return self._fn(state)


class ConstantJacobianAdapter(Generic[Array]):
    """Adapts a no-argument assembly (state-independent field jacobian,
    e.g. ``residual_forcing_jacobian()``) to ``(state, time)``."""

    def __init__(
        self, fn: Callable[[], Union[spmatrix, Array]]
    ) -> None:
        self._fn = fn

    def __call__(
        self, state: Array, time: float
    ) -> Union[spmatrix, Array]:
        return self._fn()


class FieldStateJacobianAdapter(Generic[Array]):
    """Adapts ``residual_<field>_state_jacobian(delta, state)`` to
    ``(delta, state, time)``."""

    def __init__(
        self, fn: Callable[[Array, Array], Union[spmatrix, Array]]
    ) -> None:
        self._fn = fn

    def __call__(
        self, delta: Array, state: Array, time: float
    ) -> Union[spmatrix, Array]:
        return self._fn(delta, state)


class MixedHVPAdapter(Generic[Array]):
    """Adapts a bound ``(state, adj, vec)`` mixed contraction to the
    ``(state, time, adj, vec)`` slot signature."""

    def __init__(self, fn: Callable[[Array, Array, Array], Array]) -> None:
        self._fn = fn

    def __call__(
        self, state: Array, time: float, adj_state: Array, vec: Array
    ) -> Array:
        return self._fn(state, adj_state, vec)


class _FieldParameterizationTerm(Generic[Array, PhysicsT]):
    """One parameterized coefficient field's derivative calculus.

    Framework-internal: constructed by facades from the physics's bound
    typed field-derivative methods. Produces the RAW ``ParamDerivatives``
    bundle (the BC-enforcing wrapper/adapters own all Dirichlet
    handling).

    Parameters
    ----------
    setter : Callable[[Array], None]
        Sets the field DOFs on the physics (invalidating its caches).
    physics : PhysicsT
        The physics instance the bound assemblies come from. Stored
        only for ``ParameterizationProtocol``'s ``physics()`` identity
        accessor (consumers validate they drive the same instance);
        the engine itself never dereferences it.
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
    owned_coefficients : Tuple[str, ...]
        Identifiers of the coefficient fields the setter writes;
        ``CompositeParameterization`` rejects overlapping parts.
    field_state_jacobian : FieldStateJacobianFn, optional
        :math:`A(\\delta g, u, t)`. Required when ``state_field_hvp``
        is ``FromLinearity()``.
    require_positive : bool, default False
        If True, ``apply`` raises when the mapped field is not strictly
        positive everywhere.
    bc_flux_field_jacobian : BCFluxFieldJacobianFn, optional
        :math:`B(u, t) = \\partial(\\text{flux} \\cdot n)/\\partial g`
        at boundary points, ``(state, time, bc_indices, normals) ->
        (n_bc, nfield)``. When set, the bundle carries
        ``bc_flux_param_sensitivity`` (the chain rule through the
        field map). Only coefficient-dependent flux BCs need it;
        galerkin facades never set it (natural BCs are weak there).
    """

    def __init__(
        self,
        setter: Callable[[Array], None],
        physics: PhysicsT,
        field_jacobian: FieldJacobianFn[Array],
        field_state_hvp: _FieldStateSlot[Array],
        state_field_hvp: _StateFieldSlot[Array],
        field_field_hvp: _FieldFieldSlot[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
        nstates: int,
        nfield_dofs: int,
        owned_coefficients: Tuple[str, ...],
        field_state_jacobian: Optional[FieldStateJacobianFn[Array]] = None,
        require_positive: bool = False,
        bc_flux_field_jacobian: Optional[
            BCFluxFieldJacobianFn[Array]
        ] = None,
    ) -> None:
        if not owned_coefficients:
            raise ValueError(
                "owned_coefficients must name at least one coefficient "
                "field (CompositeParameterization uses the names to "
                "reject overlapping parts)"
            )
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
        if bc_flux_field_jacobian is not None and not callable(
            bc_flux_field_jacobian
        ):
            raise TypeError(
                "bc_flux_field_jacobian must be a callable "
                "(state, time, bc_indices, normals) -> (n_bc, nfield) "
                f"or None, got {type(bc_flux_field_jacobian).__name__}"
            )
        self._setter = setter
        self._physics = physics
        self._field_jacobian = field_jacobian
        self._field_state_hvp = field_state_hvp
        self._state_field_hvp = state_field_hvp
        self._field_field_hvp = field_field_hvp
        self._field_map = field_map
        self._bkd = bkd
        self._nstates = nstates
        self._nfield_dofs = nfield_dofs
        self._owned_coefficients = tuple(owned_coefficients)
        self._field_state_jacobian = field_state_jacobian
        self._require_positive = require_positive
        self._bc_flux_field_jacobian = bc_flux_field_jacobian

        bc_flux_fn = (
            self.bc_flux_param_sensitivity
            if bc_flux_field_jacobian is not None
            else None
        )
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
                    bc_flux_param_sensitivity=bc_flux_fn,
                )
            )
        else:
            self._derivs = ParamDerivatives.first_order(
                self.param_jacobian,
                self.initial_param_jacobian,
                bc_flux_param_sensitivity=bc_flux_fn,
            )

    # -- named constructors (sugar over the one class, never subclasses)

    @staticmethod
    def linear_field_state(
        setter: Callable[[Array], None],
        physics: PhysicsT,
        field_jacobian: FieldJacobianFn[Array],
        field_state_jacobian: FieldStateJacobianFn[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
        nstates: int,
        nfield_dofs: int,
        owned_coefficients: Tuple[str, ...],
        require_positive: bool = False,
        bc_flux_field_jacobian: Optional[
            BCFluxFieldJacobianFn[Array]
        ] = None,
    ) -> "_FieldParameterizationTerm[Array, PhysicsT]":
        """Term linear in the field AND the state (e.g. kappa grad u,
        r*u): slots (FromLinearity, FromLinearity, Zero)."""
        return _FieldParameterizationTerm(
            setter,
            physics,
            field_jacobian,
            FromLinearity(),
            FromLinearity(),
            Zero(),
            field_map,
            bkd,
            nstates,
            nfield_dofs,
            owned_coefficients,
            field_state_jacobian=field_state_jacobian,
            require_positive=require_positive,
            bc_flux_field_jacobian=bc_flux_field_jacobian,
        )

    @staticmethod
    def state_independent(
        setter: Callable[[Array], None],
        physics: PhysicsT,
        field_jacobian: FieldJacobianFn[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
        nstates: int,
        nfield_dofs: int,
        owned_coefficients: Tuple[str, ...],
        require_positive: bool = False,
    ) -> "_FieldParameterizationTerm[Array, PhysicsT]":
        """Term depending on the field only (e.g. forcing): slots
        (Zero, Zero, Zero)."""
        return _FieldParameterizationTerm(
            setter,
            physics,
            field_jacobian,
            Zero(),
            Zero(),
            Zero(),
            field_map,
            bkd,
            nstates,
            nfield_dofs,
            owned_coefficients,
            require_positive=require_positive,
        )

    # -- parameterization surface (facades delegate here)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def physics(self) -> PhysicsT:
        """Return the bound physics instance (identity token only)."""
        return self._physics

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._field_map.nvars()

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the coefficient fields ``apply`` writes."""
        return self._owned_coefficients

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

    def _field_direction(self, params_1d: Array, vvec: Array) -> Array:
        """delta_g = G'(p) v. Shape: (nfield,)."""
        return self._field_map.jacobian(params_1d) @ vvec

    def _assembly_apply(
        self,
        mat: Union[spmatrix, Array],
        operand: Array,
        transpose: bool = False,
    ) -> Array:
        """``mat @ operand`` (or ``mat.T @ operand``) respecting spaces.

        The chain rule is backend-generic; numpy is a property of the
        sparse operand, not of the engine. Scipy sparse assemblies can
        only multiply numpy operands, so that branch crosses to numpy
        and ingests the (inherently dense) product back — the engine's
        only torch-to-numpy seam; the sparse matrix itself is never
        densified. Dense assemblies stay in backend space end-to-end:
        no silent device/dtype round trip, autograd preserved.
        """
        if transpose:
            mat = mat.T
        if isinstance(mat, spmatrix):
            return self._bkd.asarray(mat @ self._bkd.to_numpy(operand))
        return mat @ operand

    def param_jacobian(
        self, state: Array, time: float, params_1d: Array
    ) -> Array:
        """dR/dp = S(u) @ G'(p). Shape: (nstates, nparams)."""
        return self._assembly_apply(
            self._field_jacobian(state, time),
            self._field_map.jacobian(params_1d),
        )

    def initial_param_jacobian(self, params_1d: Array) -> Array:
        """d(u_0)/dp = 0 (coefficient fields do not set the IC)."""
        return self._bkd.zeros((self._nstates, self.nparams()))

    def bc_flux_param_sensitivity(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """d(flux·n)/dp = B(u, t) @ G'(p). Shape: (n_bc, nparams).

        Signature pinned to the ``bc_flux_param_sensitivity`` bundle
        field consumed by the BC time-residual wrapper and the steady
        adapter (they supply ``bc_indices``/``normals`` at call time).
        """
        fn = self._bc_flux_field_jacobian
        if fn is None:
            raise RuntimeError(
                "bc_flux_param_sensitivity is unavailable; check "
                "param_derivatives() before calling"
            )
        return self._assembly_apply(
            fn(state, time, bc_indices, normals),
            self._field_map.jacobian(params_1d),
        )

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
        weights = self._assembly_apply(
            self._field_jacobian(state, time), adj_state, transpose=True
        )
        out = field_map.hvp(params_1d, weights, vvec)
        ff = self._field_field_hvp
        if not isinstance(ff, Zero):
            delta = self._field_direction(params_1d, vvec)
            out = out + self._field_map.jacobian(params_1d).T @ ff(
                state, time, adj_state, delta
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
            return self._assembly_apply(
                mixed(delta, state, time), adj_state, transpose=True
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
            weights = self._assembly_apply(
                self._field_jacobian(wvec, time), adj_state, transpose=True
            )
        else:
            weights = slot(state, time, adj_state, wvec)
        result: Array = self._field_map.jacobian(params_1d).T @ weights
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
