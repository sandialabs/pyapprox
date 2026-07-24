"""Lame (E, nu) parameterizations for galerkin composite elasticity.

``create_galerkin_lame_parameterization`` wires the shared
``_FieldParameterizationTerm`` engine to ``CompositeLinearElasticity``'s
typed Lame assemblies (``residual_lame_jacobian`` /
``residual_lame_state_jacobian``) through ``ENuToLameFieldMap``,
yielding the full second-order derivative bundle.
``CompositeHyperelasticityPhysics`` has no typed Lame assemblies yet, so
it gets an apply-only parameterization with an empty bundle.
"""

from typing import Generic, Union

from pyapprox.pde.field_maps.lame import ENuToLameFieldMap
from pyapprox.pde.galerkin.physics.composite_hyperelasticity import (
    CompositeHyperelasticityPhysics,
)
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.field_term import (
    _FieldParameterizationTerm,
)
from pyapprox.util.backends.protocols import Array, Backend

_LameTerm = _FieldParameterizationTerm[
    Array, CompositeLinearElasticity[Array]
]


class LameApplyOnlyParameterization(Generic[Array]):
    """Apply-only (E, nu) parameterization for hyperelastic composites.

    Maps ``[E_1, nu_1, ...]`` through ``ENuToLameFieldMap`` onto the
    physics's per-material Lame values. The physics has no typed Lame
    residual assemblies, so ``param_derivatives()`` is the empty bundle.

    Parameters
    ----------
    physics : CompositeHyperelasticityPhysics
        The bound physics.
    field_map : ENuToLameFieldMap
        Per-material (E, nu) to Lame-value map.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: CompositeHyperelasticityPhysics[Array],
        field_map: ENuToLameFieldMap[Array],
        bkd: Backend[Array],
    ) -> None:
        self._physics = physics
        self._field_map = field_map
        self._bkd = bkd
        self._derivs: ParamDerivatives[Array] = ParamDerivatives.none()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def physics(self) -> CompositeHyperelasticityPhysics[Array]:
        """Return the bound physics instance."""
        return self._physics

    def nparams(self) -> int:
        """Return the number of parameters (2 per material)."""
        return self._field_map.nvars()

    def apply(self, params_1d: Array) -> None:
        """Map (E, nu) parameters onto the per-material Lame values."""
        self._physics.set_lame_material_values(
            self._bkd.to_numpy(self._field_map(params_1d))
        )

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the empty derivative capability bundle."""
        return self._derivs


def create_galerkin_lame_parameterization(
    physics: Union[
        CompositeLinearElasticity[Array],
        CompositeHyperelasticityPhysics[Array],
    ],
    bkd: Backend[Array],
) -> Union[_LameTerm[Array], LameApplyOnlyParameterization[Array]]:
    """Create an (E, nu) parameterization for a composite elasticity physics.

    The parameter vector is ``[E_1, nu_1, E_2, nu_2, ...]`` of length
    ``2 * nmaterials`` in ``material_names()`` order.

    Parameters
    ----------
    physics : CompositeLinearElasticity | CompositeHyperelasticityPhysics
        Physics to bind. ``CompositeLinearElasticity`` yields the
        engine-backed term with the full second-order bundle;
        ``CompositeHyperelasticityPhysics`` yields an apply-only
        parameterization (no typed Lame assemblies yet).
    bkd : Backend
        Computational backend.

    Returns
    -------
    _FieldParameterizationTerm | LameApplyOnlyParameterization
    """
    if isinstance(physics, CompositeLinearElasticity):
        field_map: ENuToLameFieldMap[Array] = ENuToLameFieldMap(
            physics.nmaterials(), bkd
        )
        return _FieldParameterizationTerm.linear_field_state(
            setter=lambda field: physics.set_lame_material_values(
                bkd.to_numpy(field)
            ),
            physics=physics,
            field_jacobian=lambda state, time: (
                physics.residual_lame_jacobian(state)
            ),
            field_state_jacobian=lambda delta, state, time: (
                physics.residual_lame_state_jacobian(delta, state)
            ),
            field_map=field_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=2 * physics.nmaterials(),
        )
    if isinstance(physics, CompositeHyperelasticityPhysics):
        return LameApplyOnlyParameterization(
            physics, ENuToLameFieldMap(physics.nmaterials(), bkd), bkd
        )
    raise TypeError(
        "physics must be a CompositeLinearElasticity or "
        f"CompositeHyperelasticityPhysics, got {type(physics).__name__}"
    )
