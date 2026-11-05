r"""A functional of a stepper's own step residual, read at the solved
trajectory.

A step residual :math:`R_n(u_n;\, y_n, \dots;\, p)` vanishes at a solved
step. Built on another system, the one the solved system was derived
from before its rows were replaced (the unconstrained system: raw
:math:`F`, no pinning), and evaluated at the solved trajectory's own
unknowns, it does not: on the replaced rows it is the physical residual
the replacement left out, an essential boundary's flux. So

.. math::

    Q = \sum_n \Omega(\tilde R_n), \qquad \tilde R_n = R_n^{\text{unc}}(u_n;
    y_n, \dots;\, p),

with :math:`\Omega` a linear observation of the residual (weights on the
boundary rows of the block producing :math:`y_{n+1}`, :class:`LastBlock`),
is a step term (:class:`StepResidualTerm`), exact for the method that
took the steps, since it reads the stages and unknowns the method used.

Its derivatives are the unconstrained stepper's: :math:`\Omega\, \partial
\tilde R_n` from its ``diag``, ``off_diag`` and ``param`` (transposed
through :math:`\Omega^\top`), and its second-order blocks at the step
adjoint :math:`\mu = \Omega^\top \omega`, contracted as the sweeps
contract a step's (:class:`~pyapprox.ode.stepper_derivatives.
StepSecondOrder`). The unconstrained stepper must have the solved
stepper's unknown (the same method) and read the same parameters, set
with the solved system's: the sweeps check a trajectory against the
solved stepper's parameter version, not this one's.
"""

from typing import Dict, Generic, Optional, TypeVar

from pyapprox.ode.observation_terms import ObservationOperatorProtocol
from pyapprox.ode.protocols import DifferentiableStepperProtocol
from pyapprox.ode.stepper_derivatives import (
    StepOperator,
    StepperDerivatives,
    StepSecondOrder,
    StepWindow,
)
from pyapprox.ode.trajectory_terms import (
    ChainedLinearization,
    ChainPart,
    LinearizationProtocol,
    StepTermDerivatives,
)
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import BlockInnerProductSpace, BlockVector

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)


class LastBlock(Generic[V, A]):
    r"""A state observation :math:`H` read on the last block of a step's
    residual (the block producing :math:`y_{n+1}`): :math:`\Omega(R) = H
    R_{\text{last}}`.

    Parameters
    ----------
    observation : ObservationOperatorProtocol
        :math:`H`, on a single block.
    residual_space : BlockInnerProductSpace
        The step residual's space.

    Raises
    ------
    TypeError
        If ``observation`` does not satisfy
        :class:`ObservationOperatorProtocol`.
    """

    def __init__(
        self,
        observation: ObservationOperatorProtocol[V, A],
        residual_space: BlockInnerProductSpace[V, A],
    ) -> None:
        if not isinstance(observation, ObservationOperatorProtocol):
            raise TypeError(
                "observation must satisfy ObservationOperatorProtocol, got "
                f"{type(observation).__name__}"
            )
        self._observation, self._space = observation, residual_space

    def nobs(self) -> int:
        return self._observation.nobs()

    def apply(self, state: BlockVector[V]) -> A:
        return self._observation.apply(state.blocks[-1])

    def apply_transpose(self, values: A) -> BlockVector[V]:
        blocks = list(self._space.zeros().blocks)
        blocks[-1] = self._observation.apply_transpose(values)
        return BlockVector(tuple(blocks))


def _one(vector: V) -> BlockVector[V]:
    return BlockVector((vector,))


def _first(vector: BlockVector[V]) -> V:
    return vector.blocks[0]


def _same(vector: BlockVector[V]) -> BlockVector[V]:
    return vector


class StepResidualTerm(Generic[V, A]):
    r""":math:`\Omega(\tilde R_n)` at each step, :math:`\tilde R_n` the
    unconstrained stepper's residual at the solved trajectory's window, a
    step term of ``observation.nobs()`` entries (see the module
    docstring).

    Parameters
    ----------
    stepper : DifferentiableStepperProtocol
        The unconstrained stepper: the solved stepper's method on the
        system before its rows were replaced, reading the same
        parameters.
    observation : ObservationOperatorProtocol
        :math:`\Omega`, on the step residual (a block vector of the
        stepper's residual space), e.g. :class:`LastBlock`.

    Raises
    ------
    TypeError
        If ``stepper`` or ``observation`` does not satisfy its protocol.
    """

    def __init__(
        self,
        stepper: DifferentiableStepperProtocol[V, A],
        observation: ObservationOperatorProtocol[BlockVector[V], A],
    ) -> None:
        if not isinstance(stepper, DifferentiableStepperProtocol):
            raise TypeError(
                "stepper must satisfy DifferentiableStepperProtocol, got "
                f"{type(stepper).__name__}"
            )
        if not isinstance(observation, ObservationOperatorProtocol):
            raise TypeError(
                "observation must satisfy ObservationOperatorProtocol, got "
                f"{type(observation).__name__}"
            )
        self._stepper, self._observation = stepper, observation
        self._derivatives: StepTermDerivatives[V, A] = (
            StepTermDerivatives.second_order(
                self._wrt_unknown,
                self._wrt_lag,
                self._wrt_param,
                self._curvature_wrt_unknown,
                self._curvature_wrt_lag,
                self._curvature_wrt_param,
            )
            if stepper.derivatives().state_state is not None
            else StepTermDerivatives.first_order(
                self._wrt_unknown, self._wrt_lag, self._wrt_param
            )
        )

    def nqoi(self) -> int:
        return self._observation.nobs()

    def value(self, step: int, window: StepWindow[V]) -> A:
        return self._observation.apply(self._stepper.step_residual(window))

    def derivatives(self) -> StepTermDerivatives[V, A]:
        return self._derivatives

    def _bundle(self) -> StepperDerivatives[V]:
        return self._stepper.derivatives()

    # ------------------------------------------------------------------
    # First derivatives: Omega composed with the step's operators
    # ------------------------------------------------------------------

    def _observed(
        self, operator: StepOperator[V]
    ) -> ChainedLinearization[BlockVector[V], V, A]:
        return ChainedLinearization(
            [ChainPart(self._observation, operator, None)],
            _same,
            _same,
            self._add_unknown,
        )

    def _add_unknown(
        self, x: Optional[BlockVector[V]], y: BlockVector[V]
    ) -> BlockVector[V]:
        space = self._stepper.unknown_space()
        return space.copy(y) if x is None else space.axpby(1.0, x, 1.0, y)

    def _add_state(self, x: Optional[V], y: V) -> V:
        space = self._stepper.state_space().component(0)
        return space.copy(y) if x is None else space.axpby(1.0, x, 1.0, y)

    def _add_param(self, x: Optional[V], y: V) -> V:
        params = self._stepper.param_space()
        if params is None:
            raise ValueError("the stepper has no parameter space")
        space = params.component(0)
        return space.copy(y) if x is None else space.axpby(1.0, x, 1.0, y)

    def _wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[BlockVector[V], A]]:
        return self._observed(self._stepper.diag(window))

    def _wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[LinearizationProtocol[V, A]]:
        field = self._bundle().off_diag
        if field is None:
            raise ValueError("the stepper's bundle has no off_diag")
        return ChainedLinearization(
            [ChainPart(self._observation, field(window, lag), None)],
            _one,
            _first,
            self._add_state,
        )

    def _wrt_param(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[V, A]]:
        field = self._bundle().param
        if field is None:
            return None
        return ChainedLinearization(
            [ChainPart(self._observation, field(window), None)],
            _one,
            _first,
            self._add_param,
        )

    # ------------------------------------------------------------------
    # Curvatures: the step's second-order blocks at mu = Omega^T omega
    # ------------------------------------------------------------------

    def _second(self) -> StepSecondOrder[V]:
        return StepSecondOrder.of(self._bundle(), type(self._stepper).__name__)

    def _lag_tangents(
        self, window: StepWindow[V], tangents: StepWindow[V]
    ) -> Dict[int, BlockVector[V]]:
        """:math:`dX_b` by lag, as many lags as the window read."""
        lag_tangents = {0: tangents.unknown}
        for lag in range(1, len(window.past_states) + 1):
            lag_tangents[lag] = _one(tangents.past_state(lag))
        return lag_tangents

    def _curvature_wrt_unknown(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[BlockVector[V]]:
        return self._second().at_lag(
            window,
            self._observation.apply_transpose(weights),
            0,
            self._lag_tangents(window, tangents),
            _one(direction),
            self._stepper.unknown_space(),
        )

    def _curvature_wrt_lag(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        lag: int,
        direction: V,
        weights: A,
    ) -> Optional[V]:
        total = self._second().at_lag(
            window,
            self._observation.apply_transpose(weights),
            lag,
            self._lag_tangents(window, tangents),
            _one(direction),
            self._stepper.state_space(),
        )
        return None if total is None else _first(total)

    def _curvature_wrt_param(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[V]:
        params = self._stepper.param_space()
        if params is None:
            raise ValueError("the stepper has no parameter space")
        return self._second().wrt_param(
            window,
            self._observation.apply_transpose(weights),
            self._lag_tangents(window, tangents),
            _one(direction),
            params.component(0),
        )
