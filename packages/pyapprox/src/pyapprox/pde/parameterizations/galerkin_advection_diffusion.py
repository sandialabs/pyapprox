"""Typed facade for parameterizing galerkin ADR coefficient fields.

The sole user-facing API: explicit typed kwargs state which
coefficients are parameterized; each non-None field map constructs a
``_FieldParameterizationTerm`` wired to the physics's typed
field-derivative assemblies with the correct derivative slots. All
derivative arithmetic lives in the engine — this module is
construction wiring only.
"""

from typing import Generic, List, Optional, Union

from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldLinearReaction,
    NodalFieldVelocity,
)
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
    StateJacobianAdapter,
    ToNumpySetter,
    _FieldParameterizationTerm,
)
from pyapprox.util.backends.protocols import Array, Backend

_ADRTerm = _FieldParameterizationTerm[
    Array, AdvectionDiffusionReaction[Array]
]


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
        reaction_map: Optional[FieldMapProtocol[Array]] = None,
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
            require_positive=True,
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
        )

    def _reaction_term(
        self, field_map: FieldMapProtocol[Array]
    ) -> _ADRTerm[Array]:
        physics, bkd = self._physics, self._bkd
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

    def apply(self, params_1d: Array) -> None:
        """Map parameters onto all parameterized coefficient fields."""
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the composed derivative capability bundle."""
        return self._inner.param_derivatives()
