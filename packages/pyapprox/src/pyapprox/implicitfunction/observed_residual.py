r"""A linear observation of a residual, as a term of a functional of one
state.

A residual :math:`R(u; p)` vanishes at a solved state. Another residual,
evaluated at that state, need not: the residual of the same equations
before some of their rows were replaced (essential boundary rows pinned
to their data, say) is, on those rows, the physical residual the
replacement left out, a boundary flux or a reaction force. A linear
observation :math:`H` of it,

.. math::

    Q(u) = H R(u; p),

is a state term (:class:`~pyapprox.implicitfunction.functional_terms.
StateTermProtocol`) of ``H.nobs()`` entries, :class:`ObservedResidual`.
Its derivatives are the residual's composed with :math:`H`:

.. math::

    Q_u = H R_u, \qquad Q_p = H R_p, \qquad
    (\omega^\top Q)'' = (\lambda^\top R)'' \text{ at } \lambda = H^\top
    \omega,

so it reaches the state, the residual and the parameters only through the
residual's bundle and the observation, and is generic in the stack. It is
the steady analog of :class:`~pyapprox.ode.step_residual_terms.
StepResidualTerm`.

:func:`pinned_rows_residual` builds it from a solved pinned residual,
whose unpinned :math:`F` and pinned rows it reads, so the term cannot
read another parameter state, time or space than the solve.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.implicitfunction.functional_terms import (
    LinearizationProtocol,
    StateTermDerivatives,
)
from pyapprox.implicitfunction.observation_terms import (
    RowWeights,
    StateObservationProtocol,
)
from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.residual_derivatives import (
    DerivativeOperator,
    ResidualJacobian,
    ResidualParamFn,
    ResidualSecondDerivativeFn,
    is_stated_zero,
)
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearizationPoint,
    VectorSpaceProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


class _Observed(Generic[V, A]):
    r"""A residual derivative :math:`D` seen through :math:`H`: forward
    :math:`H D x`; transposed :math:`D^\top H^\top \omega`."""

    def __init__(
        self,
        observation: StateObservationProtocol[V, A],
        operator: DerivativeOperator[V],
    ) -> None:
        self._observation, self._operator = observation, operator

    def apply(self, x: V) -> A:
        return self._observation.apply(self._operator.apply(x))

    def apply_transpose(self, weights: A) -> V:
        return self._operator.apply_transpose(
            self._observation.apply_transpose(weights)
        )


class ObservedResidual(Generic[V, A]):
    r""":math:`Q(u) = H R(u; p)`, a state term of ``observation.nobs()``
    entries (see the module docstring).

    Its derivatives follow the residual's bundle, read once here: first
    derivatives when the bundle has the state and parameter Jacobians,
    second derivatives when it also has its second-order fields. A
    residual whose bundle has no parameter Jacobian gives a term with no
    derivatives: a missing field means unavailable, not zero, so the term
    never claims :math:`Q_p = 0`.

    The term reads the parameters through the residual, and nothing here
    can tell whether those are the solved problem's: build :math:`R` from
    the solved residual itself (a pinned residual's ``unpinned()``), so
    both read one parameter state at one time.

    Parameters
    ----------
    residual : ResidualProtocol
        :math:`R`.
    observation : StateObservationProtocol
        :math:`H`, on the residual's values.
    space : VectorSpaceProtocol
        The space of :math:`R`'s states and values, which :math:`H` must
        observe.

    Raises
    ------
    TypeError
        If ``residual``, ``observation`` or ``space`` does not satisfy its
        protocol.
    SpaceMismatchError
        If ``observation`` observes another space than ``space``.
    """

    def __init__(
        self,
        residual: ResidualProtocol[V, FormatProtocol],
        observation: StateObservationProtocol[V, A],
        space: VectorSpaceProtocol[V],
    ) -> None:
        if not isinstance(residual, ResidualProtocol):
            raise TypeError(
                f"residual must satisfy ResidualProtocol, got {type(residual).__name__}"
            )
        if not isinstance(observation, StateObservationProtocol):
            raise TypeError(
                "observation must satisfy StateObservationProtocol, got "
                f"{type(observation).__name__}"
            )
        if not isinstance(space, VectorSpaceProtocol):
            raise TypeError(
                f"space must satisfy VectorSpaceProtocol, got {type(space).__name__}"
            )
        space.descriptor().require_same(observation.domain(), "the observation")
        self._residual, self._observation = residual, observation
        bundle = residual.derivatives()
        self._jacobian: Optional[ResidualJacobian[V, FormatProtocol]] = bundle.jacobian
        self._param: Optional[ResidualParamFn[V]] = bundle.param
        self._state_state: Optional[ResidualSecondDerivativeFn[V]] = bundle.state_state
        self._state_param: Optional[ResidualSecondDerivativeFn[V]] = bundle.state_param
        self._param_param: Optional[ResidualSecondDerivativeFn[V]] = bundle.param_param
        self._derivatives: StateTermDerivatives[V, A]
        if self._jacobian is None or self._param is None:
            self._derivatives = StateTermDerivatives.none()
        elif self._state_state is None:
            self._derivatives = StateTermDerivatives.first_order(
                self._wrt_state, self._wrt_param
            )
        else:
            self._derivatives = StateTermDerivatives.second_order(
                self._wrt_state,
                self._wrt_param,
                self._curvature_state_state,
                self._curvature_state_param,
                self._curvature_param_state,
                self._curvature_param_param,
            )

    def nqoi(self) -> int:
        return self._observation.nobs()

    def value(self, state: V) -> A:
        return self._observation.apply(self._residual(state))

    def derivatives(self) -> StateTermDerivatives[V, A]:
        return self._derivatives

    # ------------------------------------------------------------------
    # First derivatives: H composed with the residual's Jacobians
    # ------------------------------------------------------------------

    def _wrt_state(self, state: V) -> LinearizationProtocol[V, A]:
        jacobian = self._jacobian
        if jacobian is None:
            raise RuntimeError(
                "the state derivative is unavailable; check derivatives() "
                "before calling"
            )
        # A point no other linearization shares: this Jacobian is applied
        # here and never set up with a solver, so nothing can reuse it.
        point = LinearizationPoint(self._residual.parameters_version(), object())
        return _Observed(self._observation, jacobian(state, point))

    def _wrt_param(self, state: V) -> Optional[LinearizationProtocol[V, A]]:
        param = self._param
        if param is None:
            raise RuntimeError(
                "the parameter derivative is unavailable; check "
                "derivatives() before calling"
            )
        return _Observed(self._observation, param(state))

    # ------------------------------------------------------------------
    # Second derivatives: the residual's at lambda = H^T omega
    # ------------------------------------------------------------------

    def _block(
        self,
        field: Optional[ResidualSecondDerivativeFn[V]],
        state: V,
        weights: A,
    ) -> Optional[DerivativeOperator[V]]:
        """The residual's second-derivative block at the adjoint
        :math:`H^\\top \\omega`; None for a block declared zero."""
        if field is None:
            raise RuntimeError(
                "the second derivatives are unavailable; check "
                "derivatives() before calling"
            )
        if is_stated_zero(field):
            return None
        return field(state, self._observation.apply_transpose(weights))

    def _curvature_state_state(self, state: V, weights: A, w: V) -> Optional[V]:
        block = self._block(self._state_state, state, weights)
        return None if block is None else block.apply(w)

    def _curvature_state_param(self, state: V, weights: A, v: V) -> Optional[V]:
        block = self._block(self._state_param, state, weights)
        return None if block is None else block.apply(v)

    def _curvature_param_state(self, state: V, weights: A, w: V) -> Optional[V]:
        block = self._block(self._state_param, state, weights)
        return None if block is None else block.apply_transpose(w)

    def _curvature_param_param(self, state: V, weights: A, v: V) -> Optional[V]:
        block = self._block(self._param_param, state, weights)
        return None if block is None else block.apply(v)


def pinned_rows_residual(
    bkd: Backend[Array],
    pinned: PinnedResidual[Array, F],
    weights: Array,
) -> ObservedResidual[Array, Array]:
    r""":math:`Q = W^\top F(u)` on a pinned residual's pinned rows: what the
    pinning left out (an essential boundary's flux or reaction), weighted.

    Built from the solved residual itself, so :math:`F`, the parameters it
    reads, the time it is frozen at and the space are the solve's by
    construction; the weights are refused off the pinned rows
    (:class:`~pyapprox.implicitfunction.observation_terms.RowWeights`).
    States and weights are backend arrays.

    Parameters
    ----------
    bkd : Backend
        The backend.
    pinned : PinnedResidual
        The solved residual.
    weights : Array
        :math:`W`, shape ``(nstates, nobs)``, vanishing off the pinned rows.

    Raises
    ------
    TypeError
        If ``pinned`` is not a :class:`PinnedResidual`.
    ValueError
        If ``weights`` has the wrong shape or a nonzero weight off the
        pinned rows.
    """
    if not isinstance(pinned, PinnedResidual):
        raise TypeError(f"pinned must be a PinnedResidual, got {type(pinned).__name__}")
    return ObservedResidual(
        pinned.unpinned(),
        RowWeights(bkd, pinned.pin().rows(), weights),
        pinned.space(),
    )
