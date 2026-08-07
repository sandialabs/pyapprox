"""Typed facade for parameterizing collocation ADR coefficient fields.

Mirrors the galerkin ADR facade: explicit typed kwargs state which
coefficients are parameterized; each non-None field map constructs a
``_FieldParameterizationTerm`` wired to the physics's full-matrix
field-derivative assemblies with the correct derivative slots. All
derivative arithmetic lives in the engine — this module is
construction wiring only.

Divergence from galerkin (deliberate): collocation coefficient fields
are raw nodal arrays set through ``set_<coef>`` callables (the field
array IS the DOF vector; the physics caches nothing), so setters wrap
the mapped array in a picklable ``ConstantInTimeField`` instead of
writing into a nodal-field object. The diffusion term additionally
carries the boundary normal-flux assembly so coefficient-dependent
flux BCs (flux Neumann/Robin with parameterized :math:`D`) get their
parameter-Jacobian rows corrected.
"""

from typing import Callable, Generic, List, Optional, Tuple, Union

from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.parameterizations.composite import (
    CompositeParameterization,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.field_term import (
    ConstantJacobianAdapter,
    FieldStateJacobianAdapter,
    StateJacobianAdapter,
    _FieldParameterizationTerm,
)
from pyapprox.pde.parameterizations.fields import ConstantInTimeField
from pyapprox.util.backends.protocols import Array, Backend

_ADRTerm = _FieldParameterizationTerm[
    Array, AdvectionDiffusionReaction[Array]
]


class _ConstantInTimeFieldSetter(Generic[Array]):
    """Picklable setter adapter for collocation coefficient fields.

    Wraps the mapped field array in a ``ConstantInTimeField`` and hands
    it to the physics's bound ``set_<coef>`` method (bound methods
    pickle by object reference + name; a lambda would not).
    """

    def __init__(
        self, set_fn: Callable[[Callable[[float], Array]], None]
    ) -> None:
        self._set_fn = set_fn

    def writes_params(self) -> bool:
        return False

    def __call__(self, values: Array) -> None:
        self._set_fn(ConstantInTimeField(values))


class CollocationAdvectionDiffusionParameterization(Generic[Array]):
    """Parameterize collocation ADR coefficient fields through field maps.

    Parameters occupy contiguous slices in kwarg order:
    diffusion, reaction, forcing.

    Parameters
    ----------
    physics : AdvectionDiffusionReaction
        The bound collocation physics.
    diffusion_map : FieldMapProtocol, optional
        Map onto the diffusion field (positivity enforced on apply;
        carries the boundary normal-flux assembly for
        coefficient-dependent flux BCs).
    reaction_map : FieldMapProtocol, optional
        Map onto the reaction coefficient field.
    forcing_map : FieldMapProtocol, optional
        Map onto the forcing field.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: AdvectionDiffusionReaction[Array],
        *,
        diffusion_map: Optional[FieldMapProtocol[Array]] = None,
        reaction_map: Optional[FieldMapProtocol[Array]] = None,
        forcing_map: Optional[FieldMapProtocol[Array]] = None,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, AdvectionDiffusionReaction):
            raise TypeError(
                "physics must be a collocation AdvectionDiffusionReaction, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = bkd

        terms: List[_ADRTerm[Array]] = []
        if diffusion_map is not None:
            terms.append(self._diffusion_term(diffusion_map))
        if reaction_map is not None:
            terms.append(self._reaction_term(reaction_map))
        if forcing_map is not None:
            terms.append(self._forcing_term(forcing_map))
        if not terms:
            raise TypeError(
                "at least one field map must be supplied; nothing to "
                "parameterize"
            )
        self._inner: Union[
            _ADRTerm[Array], CompositeParameterization[Array]
        ]
        if len(terms) == 1:
            self._inner = terms[0]
        else:
            self._inner = CompositeParameterization(list(terms), bkd)

    # -- per-field term builders (construction wiring only) --

    def _diffusion_term(
        self, field_map: FieldMapProtocol[Array]
    ) -> _ADRTerm[Array]:
        physics, bkd = self._physics, self._bkd
        return _FieldParameterizationTerm.linear_field_state(
            setter=_ConstantInTimeFieldSetter(physics.set_diffusion),
            physics=physics,
            field_jacobian=StateJacobianAdapter(
                physics.residual_diffusion_jacobian
            ),
            field_state_jacobian=FieldStateJacobianAdapter(
                physics.residual_diffusion_state_jacobian
            ),
            field_map=field_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=physics.npts(),
            owned_coefficients=("diffusion",),
            bc_flux_field_jacobian=physics.boundary_flux_diffusion_jacobian,
        )

    def _reaction_term(
        self, field_map: FieldMapProtocol[Array]
    ) -> _ADRTerm[Array]:
        physics, bkd = self._physics, self._bkd
        return _FieldParameterizationTerm.linear_field_state(
            setter=_ConstantInTimeFieldSetter(physics.set_reaction),
            physics=physics,
            field_jacobian=StateJacobianAdapter(
                physics.residual_reaction_jacobian
            ),
            field_state_jacobian=FieldStateJacobianAdapter(
                physics.residual_reaction_state_jacobian
            ),
            field_map=field_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=physics.npts(),
            owned_coefficients=("reaction",),
        )

    def _forcing_term(
        self, field_map: FieldMapProtocol[Array]
    ) -> _ADRTerm[Array]:
        physics, bkd = self._physics, self._bkd
        return _FieldParameterizationTerm.state_independent(
            setter=_ConstantInTimeFieldSetter(physics.set_forcing),
            physics=physics,
            field_jacobian=ConstantJacobianAdapter[Array](
                physics.residual_forcing_jacobian
            ),
            field_map=field_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=physics.npts(),
            owned_coefficients=("forcing",),
        )

    # -- parameterization surface --

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the total number of parameters."""
        return self._inner.nparams()

    def physics(self) -> AdvectionDiffusionReaction[Array]:
        """Return the bound physics instance."""
        return self._physics

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the parameterized coefficient fields."""
        return self._inner.owned_coefficients()

    def apply(self, params_1d: Array) -> None:
        """Map parameters onto all parameterized coefficient fields."""
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the composed derivative capability bundle."""
        return self._inner.param_derivatives()
