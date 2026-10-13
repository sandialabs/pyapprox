"""The raw residual of a parameterized Galerkin system.

``F(u, p, t)`` is the spatial residual of a composed system, interior
plus natural boundary terms, BEFORE its essential rows are replaced by
the constraints. Almost every consumer differentiates the constrained
residual (``SteadyView``, ``constrain_param_derivatives``); quantities
that read the essential rows themselves, such as the flux through an
essential boundary, need the raw one.

The parameter enters only through the parameterization, which writes
the coefficients of objects the operator owns. This object applies
``p`` itself before every evaluation, so a consumer never relies on
``apply`` having been called, and it checks once, at construction, that
the parameterization targets this operator.

Its derivatives come in a bundle of their own
(``ParameterizedResidualDerivatives``) because they take ``p``
explicitly: the parameterization's
``ParamDerivatives`` assume ``apply`` was called, and the operator's
state curvature takes no ``p`` at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from pyapprox.ode.protocols.spatial_operator import SpatialOperatorProtocol
from pyapprox.ode.state_derivatives import StateStateHVPFn
from pyapprox.pde.parameterizations.binding import require_owned_targets
from pyapprox.pde.parameterizations.derivatives import (
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend

A = TypeVar("A", bound=ArrayProtocol)

_HVP_FIELDS = (
    "state_state_hvp",
    "state_param_hvp",
    "param_state_hvp",
    "param_param_hvp",
)


@dataclass(frozen=True)
class ParameterizedResidualDerivatives(Generic[Array]):
    """Optional derivatives of the raw residual ``F(u, p, t)``.

    Every callable applies ``p`` before evaluating. Signatures (``state``
    ``(nstates,)``, ``params`` ``(nparams,)``, ``adj`` ``(nstates,)``):

    - ``param_jacobian``: ``(state, time, params) -> (nstates, nparams)``;
    - ``state_state_hvp``: ``(state, time, params, adj, wvec (nstates,))
      -> (nstates,)``, ``adj^T (d^2 F/du^2) w``;
    - ``state_param_hvp``: ``(state, time, params, adj, vvec (nparams,))
      -> (nstates,)``;
    - ``param_state_hvp``: ``(state, time, params, adj, wvec) ->
      (nparams,)``;
    - ``param_param_hvp``: ``(state, time, params, adj, vvec) ->
      (nparams,)``.

    The four contractions come together or not at all (the second-order
    adjoint needs every one), and they require ``param_jacobian``.
    """

    param_jacobian: Optional[ParamJacobianFn[Array]] = None
    state_state_hvp: Optional[ParamHVPFn[Array]] = None
    state_param_hvp: Optional[ParamHVPFn[Array]] = None
    param_state_hvp: Optional[ParamHVPFn[Array]] = None
    param_param_hvp: Optional[ParamHVPFn[Array]] = None

    def __post_init__(self) -> None:
        for name in ("param_jacobian",) + _HVP_FIELDS:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"ParameterizedResidualDerivatives field '{name}' must "
                    f"be callable or None; got {type(value).__name__}"
                )
        present = [getattr(self, name) is not None for name in _HVP_FIELDS]
        if any(present) and not all(present):
            raise TypeError(
                "the four HVP fields must be given together or not at all"
            )
        if all(present) and self.param_jacobian is None:
            raise TypeError("the HVPs require param_jacobian")

    @staticmethod
    def none() -> "ParameterizedResidualDerivatives[A]":
        """No parameter derivatives."""
        return ParameterizedResidualDerivatives()

    @staticmethod
    def first_order(
        param_jacobian: ParamJacobianFn[A],
    ) -> "ParameterizedResidualDerivatives[A]":
        """The parameter Jacobian only."""
        return ParameterizedResidualDerivatives(param_jacobian=param_jacobian)

    @staticmethod
    def second_order(
        param_jacobian: ParamJacobianFn[A],
        state_state_hvp: ParamHVPFn[A],
        state_param_hvp: ParamHVPFn[A],
        param_state_hvp: ParamHVPFn[A],
        param_param_hvp: ParamHVPFn[A],
    ) -> "ParameterizedResidualDerivatives[A]":
        """The parameter Jacobian and all four contractions."""
        return ParameterizedResidualDerivatives(
            param_jacobian=param_jacobian,
            state_state_hvp=state_state_hvp,
            state_param_hvp=state_param_hvp,
            param_state_hvp=param_state_hvp,
            param_param_hvp=param_param_hvp,
        )


class GalerkinParameterizedResidual(Generic[Array]):
    """``F(u, p, t)`` of a parameterized system, before row replacement.

    Parameters
    ----------
    spatial_operator : SpatialOperatorProtocol
        The system's composed operator (``system.spatial_operator()``),
        interior plus natural boundary terms, without constraints.
    parameterization : ParameterizationProtocol
        Writes the coefficients of objects the operator owns.

    Raises
    ------
    TypeError
        If either argument does not satisfy its protocol.
    ValueError
        If the parameterization writes an object the operator does not
        hold.
    """

    def __init__(
        self,
        spatial_operator: SpatialOperatorProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        if not isinstance(spatial_operator, SpatialOperatorProtocol):
            raise TypeError(
                "spatial_operator must satisfy SpatialOperatorProtocol, "
                f"got {type(spatial_operator).__name__}"
            )
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                "parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        require_owned_targets(parameterization, spatial_operator)
        self._operator = spatial_operator
        self._parameterization = parameterization
        # Capability is fixed at construction: the narrowed callables are
        # captured once, and None when absent.
        param_derivs = parameterization.param_derivatives()
        self._param_jacobian_fn: Optional[ParamJacobianFn[Array]] = (
            param_derivs.param_jacobian
        )
        self._param_param_fn: Optional[ParamHVPFn[Array]] = None
        self._state_param_fn: Optional[ParamHVPFn[Array]] = None
        self._param_state_fn: Optional[ParamHVPFn[Array]] = None
        self._state_state_fn: Optional[StateStateHVPFn[Array]] = None
        self._derivs: ParameterizedResidualDerivatives[Array] = (
            ParameterizedResidualDerivatives.none()
        )
        if param_derivs.param_jacobian is None:
            return
        state_state = spatial_operator.state_derivatives().state_state_hvp
        param_param = param_derivs.param_param_hvp
        state_param = param_derivs.state_param_hvp
        param_state = param_derivs.param_state_hvp
        if (
            state_state is None
            or param_param is None
            or state_param is None
            or param_state is None
        ):
            self._derivs = ParameterizedResidualDerivatives.first_order(
                self._param_jacobian
            )
            return
        self._state_state_fn = state_state
        self._param_param_fn = param_param
        self._state_param_fn = state_param
        self._param_state_fn = param_state
        self._derivs = ParameterizedResidualDerivatives.second_order(
            self._param_jacobian,
            self._state_state_hvp,
            self._state_param_hvp,
            self._param_state_hvp,
            self._param_param_hvp,
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._operator.bkd()

    def nstates(self) -> int:
        """Return the number of states."""
        return self._operator.nstates()

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._parameterization.nparams()

    def spatial_operator(self) -> SpatialOperatorProtocol[Array]:
        """Return the operator whose residual this is."""
        return self._operator

    def parameterization(self) -> ParameterizationProtocol[Array]:
        """Return the parameterization applied before each evaluation."""
        return self._parameterization

    def derivatives(self) -> ParameterizedResidualDerivatives[Array]:
        """Return the optional parameter derivatives."""
        return self._derivs

    def residual(self, state: Array, time: float, params: Array) -> Array:
        """Compute ``F(u, p, t)``. Shape: (nstates,)."""
        self._parameterization.apply(params)
        return self._operator.spatial_residual(state, time)

    def state_jacobian(
        self, state: Array, time: float, params: Array
    ) -> Array:
        """Compute ``dF/du``. Shape: (nstates, nstates); may be sparse."""
        self._parameterization.apply(params)
        return self._operator.spatial_jacobian(state, time)

    @staticmethod
    def _unavailable(name: str) -> RuntimeError:
        return RuntimeError(
            f"{name} is unavailable; check derivatives() before calling"
        )

    def _param_jacobian(
        self, state: Array, time: float, params: Array
    ) -> Array:
        fn = self._param_jacobian_fn
        if fn is None:
            raise self._unavailable("param_jacobian")
        self._parameterization.apply(params)
        return fn(state, time, params)

    def _state_state_hvp(
        self, state: Array, time: float, params: Array, adj: Array,
        wvec: Array,
    ) -> Array:
        fn = self._state_state_fn
        if fn is None:
            raise self._unavailable("state_state_hvp")
        self._parameterization.apply(params)
        return fn(state, adj, wvec, time)

    def _state_param_hvp(
        self, state: Array, time: float, params: Array, adj: Array,
        vvec: Array,
    ) -> Array:
        fn = self._state_param_fn
        if fn is None:
            raise self._unavailable("state_param_hvp")
        self._parameterization.apply(params)
        return fn(state, time, params, adj, vvec)

    def _param_state_hvp(
        self, state: Array, time: float, params: Array, adj: Array,
        wvec: Array,
    ) -> Array:
        fn = self._param_state_fn
        if fn is None:
            raise self._unavailable("param_state_hvp")
        self._parameterization.apply(params)
        return fn(state, time, params, adj, wvec)

    def _param_param_hvp(
        self, state: Array, time: float, params: Array, adj: Array,
        vvec: Array,
    ) -> Array:
        fn = self._param_param_fn
        if fn is None:
            raise self._unavailable("param_param_hvp")
        self._parameterization.apply(params)
        return fn(state, time, params, adj, vvec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"operator={type(self._operator).__name__}, "
            f"parameterization={type(self._parameterization).__name__}, "
            f"nstates={self.nstates()}, nparams={self.nparams()})"
        )
