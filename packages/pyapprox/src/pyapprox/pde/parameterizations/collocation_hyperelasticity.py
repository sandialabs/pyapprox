"""Typed facade for parameterizing collocation hyperelasticity.

One stacked-lame engine term at FIRST order: the PK1 stress
sensitivities are state-nonlinear, so the linearity identities the
engine's second-order tier relies on do not apply. ``_WithoutHVP``
strips the stacked map's HVP so the engine builds an honest
first-order bundle, and the mixed-contraction slots hold raising
callables so a direct second-order call fails loudly.

Extension point (second order): stress-model tangent assemblies
:math:`\\partial^2 P/\\partial F \\partial \\mu` and
:math:`\\partial^2 P/\\partial F \\partial \\lambda` would supply the
genuine mixed contractions as callable slots (the quasilinear-physics
pattern); with those in place the ``_WithoutHVP`` wrapper and the
raising slots are removed and the engine's second-order tier applies
unchanged.
"""

from typing import Generic, Tuple

from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.field_maps.lame import FixedPoissonRatioLameMap
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.parameterizations.collocation_elasticity import (
    _StackedLameJacobianAdapter,
    _StackedLameSetter,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.field_term import (
    _FieldParameterizationTerm,
)
from pyapprox.util.backends.protocols import Array, Backend

_HyperelasticTerm = _FieldParameterizationTerm[
    Array, HyperelasticityPhysics[Array]
]


class _WithoutHVP(Generic[Array]):
    """Picklable field-map view without the adjoint-weighted HVP.

    Exposes only the ``FieldMapProtocol`` surface of the wrapped map,
    so ``field_map_has_hvp`` reports False and the engine builds a
    first-order bundle even when the wrapped map could provide
    curvature.
    """

    def __init__(self, inner_map: FieldMapProtocol[Array]) -> None:
        if not isinstance(inner_map, FieldMapProtocol):
            raise TypeError(
                "inner_map must satisfy FieldMapProtocol, got "
                f"{type(inner_map).__name__}"
            )
        self._inner = inner_map

    def nvars(self) -> int:
        return self._inner.nvars()

    def __call__(self, params_1d: Array) -> Array:
        return self._inner(params_1d)

    def jacobian(self, params_1d: Array) -> Array:
        return self._inner.jacobian(params_1d)


class _SecondOrderUnavailable(Generic[Array]):
    """Raising slot: the mixed contractions need stress-model tangent
    assemblies that do not exist yet; failing loudly here prevents a
    silently-wrong second-order result on direct engine calls."""

    def __call__(
        self, state: Array, time: float, adj_state: Array, vec: Array
    ) -> Array:
        raise NotImplementedError(
            "second-order contractions are unavailable for the "
            "hyperelastic parameterization: they require stress-model "
            "tangent assemblies (d^2 P / dF dmu, d^2 P / dF dlambda)"
        )


class CollocationHyperelasticityParameterization(Generic[Array]):
    """Parameterize the Young's modulus field of collocation
    hyperelasticity (first order).

    Parameters
    ----------
    physics : HyperelasticityPhysics
        The bound collocation physics (1D or 2D).
    youngs_modulus_map : FieldMapProtocol
        Map onto the Young's modulus field E(x) (positivity enforced
        through the physics's Lame setters on apply).
    poisson_ratio : float
        Fixed Poisson ratio, -1 < nu < 0.5.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: HyperelasticityPhysics[Array],
        *,
        youngs_modulus_map: FieldMapProtocol[Array],
        poisson_ratio: float,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, HyperelasticityPhysics):
            raise TypeError(
                "physics must be a collocation HyperelasticityPhysics, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = bkd
        npts = physics.npts()
        stacked_map = _WithoutHVP[Array](
            FixedPoissonRatioLameMap(
                youngs_modulus_map, poisson_ratio, npts, bkd
            )
        )
        self._inner: _HyperelasticTerm[Array] = _FieldParameterizationTerm(
            setter=_StackedLameSetter[Array](
                physics.set_mu, physics.set_lamda, npts
            ),
            physics=physics,
            field_jacobian=_StackedLameJacobianAdapter(
                physics.residual_mu_jacobian,
                physics.residual_lamda_jacobian,
                bkd,
            ),
            field_state_hvp=_SecondOrderUnavailable[Array](),
            state_field_hvp=_SecondOrderUnavailable[Array](),
            field_field_hvp=_SecondOrderUnavailable[Array](),
            field_map=stacked_map,
            bkd=bkd,
            nstates=physics.nstates(),
            nfield_dofs=2 * npts,
            owned_coefficients=("mu", "lamda"),
            bc_flux_field_jacobian=(
                physics.boundary_traction_lame_jacobian
            ),
        )

    # -- parameterization surface --

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the total number of parameters."""
        return self._inner.nparams()

    def physics(self) -> HyperelasticityPhysics[Array]:
        """Return the bound physics instance."""
        return self._physics

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the parameterized coefficient fields."""
        return self._inner.owned_coefficients()

    def apply(self, params_1d: Array) -> None:
        """Map parameters onto the stacked Lame coefficient fields."""
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the derivative capability bundle (first order)."""
        return self._inner.param_derivatives()
