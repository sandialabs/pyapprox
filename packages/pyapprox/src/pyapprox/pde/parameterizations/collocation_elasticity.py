"""Typed facade for parameterizing collocation linear elasticity.

One stacked-lame engine term: with fixed Poisson ratio both Lame
fields are rigid rescalings of one Young's modulus field, so they are
NOT independent composite parts — the tie lives inside a single
``linear_field_state`` term whose field is the stacked
:math:`[\\mu; \\lambda]` vector produced by ``FixedPoissonRatioLameMap``.
The chain rule then sums the contributions automatically:
:math:`\\partial R/\\partial p = [S_\\mu | S_\\lambda]\\,
[c_\\mu E'; c_\\lambda E']
= c_\\mu S_\\mu E' + c_\\lambda S_\\lambda E'`.

All derivative arithmetic lives in the engine — this module is
construction wiring only. The engine's second-order tier switches on
whenever the field map has a usable HVP, which the stacked map
delegates to the E-map.
"""

from typing import Callable, Generic, Tuple

from pyapprox.pde.collocation.physics.coefficients import (
    NodalCoefficientProtocol,
    NodalField,
)
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.field_maps.lame import FixedPoissonRatioLameMap
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.field_term import (
    _FieldParameterizationTerm,
)
from pyapprox.util.backends.protocols import Array, Backend

_ElasticityTerm = _FieldParameterizationTerm[
    Array, LinearElasticityPhysics[Array]
]


class _StackedLameSetter(Generic[Array]):
    """Picklable setter splitting a stacked [mu; lambda] field into the
    physics' two Lamé ``NodalField``s.

    The physics validates positivity of mu and non-negativity of lambda
    when it reads them, which for the fixed-nu map is equivalent to
    positivity of the underlying Young's modulus field.
    """

    def __init__(
        self, mu: NodalField[Array], lamda: NodalField[Array], npts: int
    ) -> None:
        self._mu = mu
        self._lamda = lamda
        self._npts = npts

    def writes_params(self) -> bool:
        return False

    def __call__(self, values: Array) -> None:
        self._mu.set_values(values[: self._npts])
        self._lamda.set_values(values[self._npts :])


def _require_field(
    coefficient: NodalCoefficientProtocol[Array], name: str
) -> NodalField[Array]:
    """The coefficient as the ``NodalField`` a parameterization rewrites.

    Raises
    ------
    TypeError
        If the physics was not built with a ``NodalField`` for it.
    """
    if not isinstance(coefficient, NodalField):
        raise TypeError(
            f"parameterizing {name} requires the physics to hold it as a "
            f"NodalField, got {type(coefficient).__name__}"
        )
    return coefficient


class _StackedLameJacobianAdapter(Generic[Array]):
    """Picklable :math:`S = [S_\\mu | S_\\lambda]` assembly adapter."""

    def __init__(
        self,
        mu_fn: Callable[[Array], Array],
        lamda_fn: Callable[[Array], Array],
        bkd: Backend[Array],
    ) -> None:
        self._mu_fn = mu_fn
        self._lamda_fn = lamda_fn
        self._bkd = bkd

    def __call__(self, state: Array, time: float) -> Array:
        return self._bkd.concatenate(
            [self._mu_fn(state), self._lamda_fn(state)], axis=1
        )


class _StackedLameStateJacobianAdapter(Generic[Array]):
    """Picklable mixed assembly :math:`A(\\delta) = A_\\mu(\\delta_\\mu)
    + A_\\lambda(\\delta_\\lambda)` for a stacked field direction."""

    def __init__(
        self,
        mu_fn: Callable[[Array, Array], Array],
        lamda_fn: Callable[[Array, Array], Array],
        npts: int,
    ) -> None:
        self._mu_fn = mu_fn
        self._lamda_fn = lamda_fn
        self._npts = npts

    def __call__(self, delta: Array, state: Array, time: float) -> Array:
        return self._mu_fn(delta[: self._npts], state) + self._lamda_fn(
            delta[self._npts :], state
        )


class CollocationElasticityParameterization(Generic[Array]):
    """Parameterize the Young's modulus field of collocation elasticity.

    Parameters
    ----------
    physics : LinearElasticityPhysics
        The bound collocation physics.
    youngs_modulus_map : FieldMapProtocol
        Map onto the Young's modulus field E(x) (positivity enforced
        when the physics reads its Lamé fields). The physics must hold
        both Lamé parameters as ``NodalField``s; a constant passed to it
        is one.

    Raises
    ------
    TypeError
        If ``physics`` is not a collocation ``LinearElasticityPhysics``,
        or holds a Lamé parameter that is not a ``NodalField``.
    poisson_ratio : float
        Fixed Poisson ratio, -1 < nu < 0.5.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: LinearElasticityPhysics[Array],
        *,
        youngs_modulus_map: FieldMapProtocol[Array],
        poisson_ratio: float,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, LinearElasticityPhysics):
            raise TypeError(
                "physics must be a collocation LinearElasticityPhysics, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = bkd
        npts = physics.npts()
        stacked_map = FixedPoissonRatioLameMap(
            youngs_modulus_map, poisson_ratio, npts, bkd
        )
        self._inner: _ElasticityTerm[Array] = (
            _FieldParameterizationTerm.linear_field_state(
                setter=_StackedLameSetter[Array](
                    _require_field(physics.mu_function(), "mu"),
                    _require_field(physics.lamda_function(), "lamda"),
                    npts,
                ),
                physics=physics,
                field_jacobian=_StackedLameJacobianAdapter(
                    physics.residual_mu_jacobian,
                    physics.residual_lamda_jacobian,
                    bkd,
                ),
                field_state_jacobian=_StackedLameStateJacobianAdapter(
                    physics.residual_mu_state_jacobian,
                    physics.residual_lamda_state_jacobian,
                    npts,
                ),
                field_map=stacked_map,
                bkd=bkd,
                nstates=physics.nstates(),
                nfield_dofs=2 * npts,
                owned_coefficients=("mu", "lamda"),
                bc_flux_field_jacobian=(
                    physics.boundary_traction_lame_jacobian
                ),
            )
        )

    # -- parameterization surface --

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the total number of parameters."""
        return self._inner.nparams()

    def physics(self) -> LinearElasticityPhysics[Array]:
        """Return the bound physics instance."""
        return self._physics

    def targets(self) -> Tuple[object, ...]:
        """Return the objects whose coefficients ``apply`` writes."""
        return (self._physics,)

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the parameterized coefficient fields."""
        return self._inner.owned_coefficients()

    def apply(self, params_1d: Array) -> None:
        """Map parameters onto the stacked Lame coefficient fields."""
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the derivative capability bundle."""
        return self._inner.param_derivatives()
