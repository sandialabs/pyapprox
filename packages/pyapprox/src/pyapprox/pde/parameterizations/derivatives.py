"""Frozen bundle of optional parameterization derivative capabilities.

The pde parameterization family's optional derivatives are
``(state, time, params, ...)`` callables closed over a physics instance
bound at producer construction — they cannot be curried into the
optimizer :class:`~pyapprox.interface.functions.derivatives.
Derivatives` bundle (state/time vary per call), so per the
family-bundle rule in ``docs/conventions/optional_methods.md`` the family
gets its own frozen bundle, exposed through the
``param_derivatives()`` accessor on ``ParameterizationProtocol``.

Absence of a capability is ``None``, never a missing attribute. No
finite-difference fallback exists here or in producers: what to do about
an absent capability is the consumer's decision.

Field signatures (the producer holds its physics; one producer
instance binds ONE physics instance — ensembles construct one producer
per physics):

- ``param_jacobian``: ``(state, time, params_1d) ->``
  :class:`ParamJacobianProtocol` — d(residual)/d(params) at that point,
  applied to blocks rather than returned as a matrix, so a producer
  whose Jacobian is a product (a sparse assembly times a field map's
  Jacobian) never forms it. RAW (no Dirichlet handling; BC wrappers own
  all constraint corrections). A producer with the matrix wraps its
  function with :func:`dense_param_jacobian`.
- ``initial_param_jacobian``: ``(params_1d) -> (nstates, nparams)`` —
  d(initial_state)/d(params).
- ``initial_param_hvp``: ``(params_1d, weight (nstates,),
  vvec (nparams,)) -> (nparams,)`` — sum_i weight_i
  (d^2 initial_state_i/d params^2) v. Required for second order: zero
  must be declared, never assumed.
- ``param_param_hvp``: ``(state, time, params_1d, adj_state,
  vvec (nparams,)) -> (nparams,)`` — lambda^T (d^2R/dp^2) v.
- ``state_param_hvp``: ``(state, time, params_1d, adj_state,
  vvec (nparams,)) -> (nstates,)`` — lambda^T (d^2R/dy dp) v.
- ``param_state_hvp``: ``(state, time, params_1d, adj_state,
  wvec (nstates,)) -> (nparams,)`` — lambda^T (d^2R/dp dy) w.
- ``bc_flux_param_sensitivity``: ``(state, time, params_1d,
  bc_indices, normals) -> (n_bc, nparams)`` — d(flux·n)/d(params) at
  boundary nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array, ArrayProtocol

A = TypeVar("A", bound=ArrayProtocol)


@runtime_checkable
class ParamJacobianProtocol(Protocol[Array]):
    r""":math:`\partial F/\partial p` at one ``(state, time, params)``.

    Blocks are backend arrays with one column per vector. The actions are
    what a sweep reads; :meth:`matrix` is for a consumer that needs the
    entries (a projection onto a basis, a dense model), and costs a
    matrix-free implementation ``nparams`` actions or one product.
    """

    def apply(self, vecs: Array) -> Array:
        r""":math:`(\partial F/\partial p) V`. Shape: (nparams, ncols) ->
        (nstates, ncols)."""
        ...

    def apply_transpose(self, vecs: Array) -> Array:
        r""":math:`(\partial F/\partial p)^\top W`. Shape:
        (nstates, ncols) -> (nparams, ncols)."""
        ...

    def matrix(self) -> Array:
        """The entries. Shape: (nstates, nparams)."""
        ...


ParamJacobianMatrixFn = Callable[[Array, float, Array], Array]
"""``(state, time, params_1d) -> (nstates, nparams)``: a parameter
Jacobian as a matrix."""
ParamJacobianFn = Callable[[Array, float, Array], ParamJacobianProtocol[Array]]
InitialParamJacobianFn = Callable[[Array], Array]
InitialParamHVPFn = Callable[[Array, Array, Array], Array]
ParamHVPFn = Callable[[Array, float, Array, Array, Array], Array]
BCFluxParamSensitivityFn = Callable[
    [Array, float, Array, Array, Array], Array
]

class DenseParamJacobian(Generic[Array]):
    """A parameter Jacobian stored as its matrix.

    Satisfies :class:`ParamJacobianProtocol`; the actions are backend
    products, so a torch graph through the matrix or the blocks is kept.

    Parameters
    ----------
    matrix : Array
        Shape: (nstates, nparams).
    """

    def __init__(self, matrix: Array) -> None:
        if matrix.ndim != 2:
            raise ValueError(
                "matrix must have shape (nstates, nparams), got "
                f"{tuple(matrix.shape)}"
            )
        self._matrix = matrix

    def apply(self, vecs: Array) -> Array:
        """``matrix @ vecs``. Shape: (nparams, ncols) -> (nstates, ncols)."""
        result: Array = self._matrix @ vecs
        return result

    def apply_transpose(self, vecs: Array) -> Array:
        """``matrix.T @ vecs``. Shape: (nstates, ncols) -> (nparams, ncols)."""
        result: Array = self._matrix.T @ vecs
        return result

    def matrix(self) -> Array:
        """The stored matrix. Shape: (nstates, nparams)."""
        return self._matrix

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={tuple(self._matrix.shape)})"


class _DenseParamJacobianFn(Generic[Array]):
    """A function returning the matrix, as one returning the Jacobian."""

    def __init__(self, matrix: ParamJacobianMatrixFn[Array]) -> None:
        self._matrix = matrix

    def __call__(
        self, state: Array, time: float, params_1d: Array
    ) -> DenseParamJacobian[Array]:
        return DenseParamJacobian(self._matrix(state, time, params_1d))


def dense_param_jacobian(
    matrix: ParamJacobianMatrixFn[Array],
) -> ParamJacobianFn[Array]:
    """Declare a parameter Jacobian given as a matrix of the state, time
    and parameters.

    Parameters
    ----------
    matrix : callable
        ``(state, time, params_1d) -> (nstates, nparams)``; typically a
        public bound method, which the producer keeps.
    """
    if not callable(matrix):
        raise TypeError(
            f"matrix must be callable, got {type(matrix).__name__}"
        )
    return _DenseParamJacobianFn(matrix)


_FIELD_NAMES = (
    "param_jacobian",
    "initial_param_jacobian",
    "initial_param_hvp",
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
    initial_param_hvp: Optional[InitialParamHVPFn[Array]] = None
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
        initial_param_hvp: InitialParamHVPFn[A],
        *,
        bc_flux_param_sensitivity: Optional[
            BCFluxParamSensitivityFn[A]
        ] = None,
    ) -> "ParamDerivatives[A]":
        """Second-order capability: jacobians, the three HVPs and the
        initial-state curvature all required."""
        if (
            param_jacobian is None
            or initial_param_jacobian is None
            or param_param_hvp is None
            or state_param_hvp is None
            or param_state_hvp is None
            or initial_param_hvp is None
        ):
            raise TypeError(
                "second_order requires param_jacobian, "
                "initial_param_jacobian, all three HVP callables and "
                "initial_param_hvp; use first_order if they are unavailable"
            )
        return ParamDerivatives(
            param_jacobian=param_jacobian,
            initial_param_jacobian=initial_param_jacobian,
            initial_param_hvp=initial_param_hvp,
            param_param_hvp=param_param_hvp,
            state_param_hvp=state_param_hvp,
            param_state_hvp=param_state_hvp,
            bc_flux_param_sensitivity=bc_flux_param_sensitivity,
        )
