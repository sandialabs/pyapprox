"""Lame (E, nu) parameterizations for galerkin composite elasticity.

``create_galerkin_lame_parameterization`` wires the shared
``_FieldParameterizationTerm`` engine to ``CompositeLinearElasticity``'s
typed Lame assemblies (``residual_lame_jacobian`` /
``residual_lame_state_jacobian``) through ``ENuToLameFieldMap``,
yielding the full second-order derivative bundle.
``CompositeHyperelasticityPhysics`` has no typed Lame assemblies yet, so
it gets an apply-only parameterization with an empty bundle.
``LameFieldApplyOnlyParameterization`` takes the Lame fields themselves
as parameters (per element or per quadrature point), for callers whose
own map produces them; it too is apply-only.
"""

from typing import Generic, Tuple, Union

import numpy as np

from pyapprox.pde.field_maps.lame import ENuToLameFieldMap
from pyapprox.pde.galerkin.physics.composite_hyperelasticity import (
    CompositeHyperelasticityPhysics,
)
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.field_term import (
    FieldStateJacobianAdapter,
    StateJacobianAdapter,
    ToNumpySetter,
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

    def targets(self) -> Tuple[object, ...]:
        """Return the objects whose coefficients ``apply`` writes."""
        return (self._physics,)

    def nparams(self) -> int:
        """Return the number of parameters (2 per material)."""
        return self._field_map.nvars()

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the parameterized coefficient fields."""
        return ("mu", "lamda")

    def apply(self, params_1d: Array) -> None:
        """Map (E, nu) parameters onto the per-material Lame values."""
        self._physics.set_lame_material_values(
            self._bkd.to_numpy(self._field_map(params_1d))
        )

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the empty derivative capability bundle."""
        return self._derivs


class LameFieldApplyOnlyParameterization(Generic[Array]):
    r"""Apply-only parameterization by the Lame fields themselves.

    The parameters are the values of :math:`\lambda` then :math:`\mu` on
    a fixed grid of shape ``field_shape`` (``(nelems,)`` per element, or
    ``(nelems, nquad)`` per quadrature point), flattened in C order and
    concatenated: ``[lam.ravel(), mu.ravel()]``. ``apply`` hands them to
    the physics' ``set_lame_parameters``. A caller mapping its own inputs
    (a random field, a mixture rule) to Lame values applies them through
    a :class:`~pyapprox.pde.parameterizations.parameter_state.ParameterState`
    over this parameterization, so solves that key on the parameters'
    version see the change. No derivatives: ``param_derivatives()`` is
    the empty bundle.

    Parameters
    ----------
    physics : CompositeLinearElasticity | CompositeHyperelasticityPhysics
        The bound physics.
    field_shape : tuple of int
        The shape of each Lame field.
    bkd : Backend
        Computational backend.

    Raises
    ------
    ValueError
        If ``field_shape`` is empty or has a non-positive entry.
    """

    def __init__(
        self,
        physics: Union[
            CompositeLinearElasticity[Array],
            CompositeHyperelasticityPhysics[Array],
        ],
        field_shape: Tuple[int, ...],
        bkd: Backend[Array],
    ) -> None:
        if len(field_shape) == 0 or any(n <= 0 for n in field_shape):
            raise ValueError(
                f"field_shape must be non-empty and positive, got {field_shape}"
            )
        self._physics = physics
        self._field_shape = tuple(field_shape)
        self._nfield = int(np.prod(field_shape))
        self._bkd = bkd
        self._derivs: ParamDerivatives[Array] = ParamDerivatives.none()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def targets(self) -> Tuple[object, ...]:
        """Return the objects whose coefficients ``apply`` writes."""
        return (self._physics,)

    def nparams(self) -> int:
        """Return the number of parameters, two fields' worth."""
        return 2 * self._nfield

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the parameterized coefficient fields."""
        return ("mu", "lamda")

    def apply(self, params_1d: Array) -> None:
        """Set the physics' Lame fields to ``params_1d``, shape
        ``(nparams,)``.

        Raises
        ------
        ValueError
            If ``params_1d`` is not of shape ``(nparams,)``.
        """
        if tuple(params_1d.shape) != (self.nparams(),):
            raise ValueError(
                f"params_1d must have shape ({self.nparams()},), got "
                f"{tuple(params_1d.shape)}"
            )
        values = self._bkd.to_numpy(params_1d)
        self._physics.set_lame_parameters(
            np.reshape(values[: self._nfield], self._field_shape),
            np.reshape(values[self._nfield :], self._field_shape),
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
            setter=ToNumpySetter(physics.set_lame_material_values, bkd),
            physics=physics,
            field_jacobian=StateJacobianAdapter(
                physics.residual_lame_jacobian
            ),
            field_state_jacobian=FieldStateJacobianAdapter(
                physics.residual_lame_state_jacobian
            ),
            field_map=field_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=2 * physics.nmaterials(),
            owned_coefficients=("mu", "lamda"),
        )
    if isinstance(physics, CompositeHyperelasticityPhysics):
        return LameApplyOnlyParameterization(
            physics, ENuToLameFieldMap(physics.nmaterials(), bkd), bkd
        )
    raise TypeError(
        "physics must be a CompositeLinearElasticity or "
        f"CompositeHyperelasticityPhysics, got {type(physics).__name__}"
    )
