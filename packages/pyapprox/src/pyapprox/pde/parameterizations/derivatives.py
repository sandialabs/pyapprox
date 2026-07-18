"""Frozen bundle of optional parameterization derivative capabilities.

The pde parameterization family's optional derivatives are
``(physics, state, time, params, ...)`` callables — they cannot be curried
into the optimizer :class:`~pyapprox.interface.functions.derivatives.
Derivatives` bundle (physics/state/time vary per call), so per the
family-bundle rule in ``docs/OPTIONAL_METHODS_CONVENTION.md`` the family
gets its own frozen bundle, exposed through the
``param_derivatives()`` accessor on ``ParameterizationProtocol``.

Absence of a capability is ``None``, never a missing attribute. No
finite-difference fallback exists here or in producers: what to do about
an absent capability is the consumer's decision.

Field signatures (``physics`` is typed ``object`` until per-module
physics protocols land in a later phase of the refactor):

- ``param_jacobian``: ``(physics, state, time, params_1d)
  -> (nstates, nparams)`` — d(residual)/d(params), RAW (no Dirichlet
  handling; BC wrappers own all constraint corrections).
- ``initial_param_jacobian``: ``(physics, params_1d)
  -> (nstates, nparams)`` — d(initial_state)/d(params).
- ``param_param_hvp``: ``(physics, state, time, params_1d, adj_state,
  vvec (nparams,)) -> (nparams,)`` — lambda^T (d^2R/dp^2) v.
- ``state_param_hvp``: ``(physics, state, time, params_1d, adj_state,
  vvec (nparams,)) -> (nstates,)`` — lambda^T (d^2R/dy dp) v.
- ``param_state_hvp``: ``(physics, state, time, params_1d, adj_state,
  wvec (nstates,)) -> (nparams,)`` — lambda^T (d^2R/dp dy) w.
- ``bc_flux_param_sensitivity``: ``(physics, state, time, params_1d,
  bc_indices, normals) -> (n_bc, nparams)`` — d(flux·n)/d(params) at
  boundary nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from pyapprox.util.backends.protocols import Array, ArrayProtocol

A = TypeVar("A", bound=ArrayProtocol)

ParamJacobianFn = Callable[[object, Array, float, Array], Array]
InitialParamJacobianFn = Callable[[object, Array], Array]
ParamHVPFn = Callable[[object, Array, float, Array, Array, Array], Array]
BCFluxParamSensitivityFn = Callable[
    [object, Array, float, Array, Array, Array], Array
]

_FIELD_NAMES = (
    "param_jacobian",
    "initial_param_jacobian",
    "param_param_hvp",
    "state_param_hvp",
    "param_state_hvp",
    "bc_flux_param_sensitivity",
)


@dataclass(frozen=True)
class ParamDerivatives(Generic[Array]):
    """Bundle of optional parameterization derivative capabilities.

    Fields hold callables (typically bound methods). Producers with
    unconditional capability keep their real public methods — the bundle
    references them. Bundles are frozen; when capability changes (e.g.
    ``CompositeParameterization.append``) the producer rebuilds its
    bundle — never mutates one.
    """

    param_jacobian: Optional[ParamJacobianFn[Array]] = None
    initial_param_jacobian: Optional[InitialParamJacobianFn[Array]] = None
    param_param_hvp: Optional[ParamHVPFn[Array]] = None
    state_param_hvp: Optional[ParamHVPFn[Array]] = None
    param_state_hvp: Optional[ParamHVPFn[Array]] = None
    bc_flux_param_sensitivity: Optional[
        BCFluxParamSensitivityFn[Array]
    ] = None

    def __post_init__(self) -> None:
        for name in _FIELD_NAMES:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"ParamDerivatives field '{name}' must be callable or "
                    f"None; got {type(value).__name__}. Did you pass a "
                    "computed value instead of the function itself?"
                )

    @staticmethod
    def none() -> "ParamDerivatives[A]":
        """Bundle with no capabilities."""
        return ParamDerivatives()

    @staticmethod
    def first_order(
        param_jacobian: ParamJacobianFn[A],
        initial_param_jacobian: InitialParamJacobianFn[A],
        *,
        bc_flux_param_sensitivity: Optional[
            BCFluxParamSensitivityFn[A]
        ] = None,
    ) -> "ParamDerivatives[A]":
        """First-order capability: both parameter jacobians required."""
        if param_jacobian is None or initial_param_jacobian is None:
            raise TypeError(
                "first_order requires param_jacobian and "
                "initial_param_jacobian callables"
            )
        return ParamDerivatives(
            param_jacobian=param_jacobian,
            initial_param_jacobian=initial_param_jacobian,
            bc_flux_param_sensitivity=bc_flux_param_sensitivity,
        )

    @staticmethod
    def second_order(
        param_jacobian: ParamJacobianFn[A],
        initial_param_jacobian: InitialParamJacobianFn[A],
        param_param_hvp: ParamHVPFn[A],
        state_param_hvp: ParamHVPFn[A],
        param_state_hvp: ParamHVPFn[A],
        *,
        bc_flux_param_sensitivity: Optional[
            BCFluxParamSensitivityFn[A]
        ] = None,
    ) -> "ParamDerivatives[A]":
        """Second-order capability: jacobians AND all three HVPs required."""
        if (
            param_jacobian is None
            or initial_param_jacobian is None
            or param_param_hvp is None
            or state_param_hvp is None
            or param_state_hvp is None
        ):
            raise TypeError(
                "second_order requires param_jacobian, "
                "initial_param_jacobian, and all three HVP callables; use "
                "first_order if the HVPs are unavailable"
            )
        return ParamDerivatives(
            param_jacobian=param_jacobian,
            initial_param_jacobian=initial_param_jacobian,
            param_param_hvp=param_param_hvp,
            state_param_hvp=state_param_hvp,
            param_state_hvp=param_state_hvp,
            bc_flux_param_sensitivity=bc_flux_param_sensitivity,
        )
