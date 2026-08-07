"""Typed facade for parameterizing galerkin ADR coefficient fields.

The sole user-facing API: explicit typed kwargs state which
coefficients are parameterized; each non-None field map constructs a
``_FieldParameterizationTerm`` wired to the physics's typed
field-derivative assemblies with the correct derivative slots. All
derivative arithmetic lives in the engine — this module is
construction wiring only.
"""

from typing import Generic, List, Optional, Tuple, Union

import numpy as np

from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldLinearReaction,
    NodalFieldVelocity,
    TimeModulatedFieldProtocol,
)
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.galerkin.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.parameterizations.composite import (
    CompositeParameterization,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.field_term import (
    ConstantJacobianAdapter,
    FieldStateJacobianAdapter,
    ParamSetter,
    StateJacobianAdapter,
    ToNumpySetter,
    _FieldParameterizationTerm,
)
from pyapprox.util.backends.protocols import Array, Backend

_ADRTerm = _FieldParameterizationTerm[
    Array, AdvectionDiffusionReaction[Array]
]


class _FromField:
    """Sentinel: take the parameter-to-field map from the coefficient.

    A separable coefficient (see ``TimeModulatedFieldProtocol``) already
    holds its spatial modes, so the map from mode coefficients to field
    DOFs is fixed the moment the physics is built. There is no map for
    the caller to supply --- but a slot is still only parameterized when
    the caller asks, so the kwarg remains and this is what they pass.

    Passing an actual map for such a coefficient RAISES rather than
    being ignored: two sets of modes could disagree, and the failure
    would be invisible --- the forward solve using one set and the
    gradient the other, each self-consistent, with finite differences
    agreeing because they perturb the same wrong forward field.
    """

    def __repr__(self) -> str:
        return "FROM_FIELD"


FROM_FIELD = _FromField()
"""Opt a slot in when the coefficient supplies its own spatial modes."""


def _require_field(
    physics_field: object, expected: type, kwarg: str
) -> object:
    """Eager validation: the facade kwarg must target the physics's
    differentiable (nodal-field) representation."""
    if not isinstance(physics_field, expected):
        raise TypeError(
            f"{kwarg} was supplied but the physics's coefficient is "
            f"{type(physics_field).__name__}; construct the physics "
            f"with a {expected.__name__} (the differentiable "
            "representation) to parameterize it"
        )
    return physics_field


class AdvectionDiffusionParameterization(Generic[Array]):
    """Parameterize galerkin ADR coefficient fields through field maps.

    Parameters occupy contiguous slices in kwarg order:
    diffusivity, forcing, reaction, velocity.

    Parameters
    ----------
    physics : AdvectionDiffusionReaction
        The bound physics. Each parameterized coefficient must be its
        nodal-field representation (``NodalFieldDiffusion`` etc.) —
        validated eagerly.
    diffusivity_map : FieldMapProtocol, optional
        Map onto the diffusivity DOFs (positivity enforced on apply).
    forcing_map : FieldMapProtocol, optional
        Map onto the forcing DOFs.
    reaction_map : FieldMapProtocol, optional
        Map onto the linear-reaction coefficient DOFs.
    velocity_map : FieldMapProtocol, optional
        Map onto the velocity DOFs (vector-basis ordering).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: AdvectionDiffusionReaction[Array],
        *,
        diffusivity_map: Optional[FieldMapProtocol[Array]] = None,
        forcing_map: Optional[FieldMapProtocol[Array]] = None,
        reaction_map: Optional[
            Union[FieldMapProtocol[Array], _FromField]
        ] = None,
        velocity_map: Optional[FieldMapProtocol[Array]] = None,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, AdvectionDiffusionReaction):
            raise TypeError(
                "physics must be an AdvectionDiffusionReaction, got "
                f"{type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = bkd

        terms: List[_ADRTerm[Array]] = []
        if diffusivity_map is not None:
            terms.append(self._diffusivity_term(diffusivity_map))
        if forcing_map is not None:
            terms.append(self._forcing_term(forcing_map))
        if reaction_map is not None:
            terms.append(self._reaction_term(reaction_map))
        if velocity_map is not None:
            terms.append(self._velocity_term(velocity_map))
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
            self._inner = CompositeParameterization(
                list(terms), bkd
            )

    # -- per-field term builders (construction wiring only) --

    def _diffusivity_term(
        self, field_map: FieldMapProtocol[Array]
    ) -> _ADRTerm[Array]:
        physics, bkd = self._physics, self._bkd
        diffusion = _require_field(
            physics.diffusion_function(),
            NodalFieldDiffusion,
            "diffusivity_map",
        )
        assert isinstance(diffusion, NodalFieldDiffusion)
        return _FieldParameterizationTerm.linear_field_state(
            setter=ToNumpySetter(diffusion.set_dofs, bkd),
            physics=physics,
            field_jacobian=StateJacobianAdapter(
                physics.residual_diffusivity_jacobian
            ),
            field_state_jacobian=FieldStateJacobianAdapter(
                physics.residual_diffusivity_state_jacobian
            ),
            field_map=field_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=diffusion.ndofs(),
            owned_coefficients=("diffusivity",),
        )

    def _forcing_term(
        self, field_map: FieldMapProtocol[Array]
    ) -> _ADRTerm[Array]:
        physics, bkd = self._physics, self._bkd
        forcing = _require_field(
            physics.forcing_function(), NodalFieldForcing, "forcing_map"
        )
        assert isinstance(forcing, NodalFieldForcing)
        return _FieldParameterizationTerm.state_independent(
            setter=ToNumpySetter(forcing.set_dofs, bkd),
            physics=physics,
            field_jacobian=ConstantJacobianAdapter[Array](
                physics.residual_forcing_jacobian
            ),
            field_map=field_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=forcing.ndofs(),
            owned_coefficients=("forcing",),
        )

    def _reaction_term(
        self, field_map: Union[FieldMapProtocol[Array], _FromField]
    ) -> _ADRTerm[Array]:
        physics, bkd = self._physics, self._bkd
        modulated = physics.reaction_function()
        if isinstance(modulated, TimeModulatedFieldProtocol):
            if not isinstance(field_map, _FromField):
                raise TypeError(
                    "the physics's reaction is a separable "
                    f"{type(modulated).__name__}, which already fixes "
                    "its spatial modes; pass reaction_map=FROM_FIELD to "
                    "parameterize its mode coefficients. A second map "
                    "could disagree with the field's own modes, and the "
                    "forward solve and the gradient would then describe "
                    "different controls"
                )
            return self._modulated_reaction_term(modulated)
        if isinstance(field_map, _FromField):
            # Without this the term's own guard reports only
            # "field_map must satisfy FieldMapProtocol, got _FromField",
            # which does not say what the caller got wrong.
            raise TypeError(
                "reaction_map=FROM_FIELD takes the spatial modes from "
                "the coefficient, but the physics's reaction is "
                f"{type(modulated).__name__}, which has none; supply a "
                "field map, or build the physics with a separable "
                "reaction"
            )
        reaction = _require_field(
            physics.reaction_function(),
            NodalFieldLinearReaction,
            "reaction_map",
        )
        assert isinstance(reaction, NodalFieldLinearReaction)
        return _FieldParameterizationTerm.linear_field_state(
            setter=ToNumpySetter(reaction.set_dofs, bkd),
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
            nfield_dofs=reaction.ndofs(),
            owned_coefficients=("reaction",),
        )

    def _modulated_reaction_term(
        self, reaction: TimeModulatedFieldProtocol
    ) -> _ADRTerm[Array]:
        """Parameterize a reaction that is separable in space and time.

        The parameters are the MODE COEFFICIENTS, so the setter writes
        those rather than nodal DOFs and ``nfield_dofs`` counts modes.

        The field map is DERIVED from the field's own modes rather than
        supplied alongside them. Two copies of the modes could disagree,
        and the failure would be invisible: the forward solve would use
        one set and the gradient another, each self-consistent, with
        finite differences agreeing because they perturb the same wrong
        forward field. Deriving makes that impossible rather than
        merely discouraged.

        The caller still passes ``reaction_map`` to opt this slot in,
        but it must be the sentinel :data:`FROM_FIELD`: the physics
        fixed the spatial modes when it was built, so an actual map
        could only contradict them. Refusing is the point --- silently
        discarding a map the caller supplied would leave them believing
        it was used.
        """
        physics, bkd = self._physics, self._bkd
        modes = np.asarray(reaction.spatial_modes())
        derived_map = BasisExpansion(
            bkd,
            0.0,
            [bkd.asarray(modes[:, k]) for k in range(modes.shape[1])],
        )
        return _FieldParameterizationTerm.linear_field_state(
            setter=ParamSetter(reaction.set_coefficients, bkd),
            physics=physics,
            field_jacobian=StateJacobianAdapter(
                physics.residual_reaction_jacobian
            ),
            field_state_jacobian=FieldStateJacobianAdapter(
                physics.residual_reaction_state_jacobian
            ),
            field_map=derived_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=reaction.nmodes(),
            owned_coefficients=("reaction",),
            time_modulation=reaction.modulation(),
        )

    def _velocity_term(
        self, field_map: FieldMapProtocol[Array]
    ) -> _ADRTerm[Array]:
        physics, bkd = self._physics, self._bkd
        velocity = _require_field(
            physics.velocity_function(),
            NodalFieldVelocity,
            "velocity_map",
        )
        assert isinstance(velocity, NodalFieldVelocity)
        return _FieldParameterizationTerm.linear_field_state(
            setter=ToNumpySetter(velocity.set_dofs, bkd),
            physics=physics,
            field_jacobian=StateJacobianAdapter(
                physics.residual_velocity_jacobian
            ),
            field_state_jacobian=FieldStateJacobianAdapter(
                physics.residual_velocity_state_jacobian
            ),
            field_map=field_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=velocity.ndofs(),
            owned_coefficients=("velocity",),
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
