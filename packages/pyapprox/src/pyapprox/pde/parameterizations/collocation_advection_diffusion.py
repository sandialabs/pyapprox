"""Typed facade for parameterizing collocation ADR coefficient fields.

Mirrors the galerkin ADR facade: explicit typed kwargs state which
coefficients are parameterized; each non-None field map constructs a
``_FieldParameterizationTerm`` wired to the physics's full-matrix
field-derivative assemblies with the correct derivative slots. All
derivative arithmetic lives in the engine — this module is
construction wiring only.

As on the galerkin side, the physics is built with the coefficient
objects and the parameterization writes into them: each parameterized
coefficient must be a ``NodalField`` (its values are the DOF vector),
and the mapped values go to its ``set_values``. The physics knows
nothing of the parameterization. The diffusion term additionally
carries the boundary normal-flux assembly so coefficient-dependent
flux BCs (flux Neumann/Robin with parameterized :math:`D`) get their
parameter-Jacobian rows corrected.
"""

from typing import Generic, List, Optional, Tuple, Union

from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import (
    NodalCoefficientProtocol,
    NodalField,
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
from pyapprox.util.backends.protocols import Array, Backend

_ADRTerm = _FieldParameterizationTerm[
    Array, AdvectionDiffusionReaction[Array]
]


class _NodalFieldSetter(Generic[Array]):
    """Picklable setter writing mapped values into a ``NodalField``."""

    def __init__(self, field: NodalField[Array]) -> None:
        self._field = field

    def writes_params(self) -> bool:
        return False

    def __call__(self, values: Array) -> None:
        self._field.set_values(values)


def _require_field(
    coefficient: Optional[NodalCoefficientProtocol[Array]], map_name: str
) -> NodalField[Array]:
    """The physics' coefficient a map writes, which must be a NodalField.

    Raises
    ------
    TypeError
        If the coefficient is absent or not a ``NodalField``.
    """
    if not isinstance(coefficient, NodalField):
        raise TypeError(
            f"{map_name} needs the physics built with a NodalField for that "
            f"coefficient, got {type(coefficient).__name__}"
        )
    return coefficient


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
            setter=_NodalFieldSetter(
                _require_field(physics.diffusion_function(), "diffusion_map")
            ),
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
            setter=_NodalFieldSetter(
                _require_field(physics.reaction_function(), "reaction_map")
            ),
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
            setter=_NodalFieldSetter(
                _require_field(physics.forcing_function(), "forcing_map")
            ),
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

    def targets(self) -> Tuple[object, ...]:
        """Return the objects whose coefficients ``apply`` writes."""
        return (self._physics,)

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the parameterized coefficient fields."""
        return self._inner.owned_coefficients()

    def apply(self, params_1d: Array) -> None:
        """Map parameters onto all parameterized coefficient fields."""
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the composed derivative capability bundle."""
        return self._inner.param_derivatives()
