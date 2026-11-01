r"""Protocol for a stepper whose step equation the sweeps differentiate.

The step's unknown :math:`u_n` and residual :math:`R_n` are block
vectors of the stepper's unknown and residual spaces (see
:mod:`pyapprox.ode.stepper_derivatives`); a stored state and the
parameters are one-block vectors of their own spaces. The required
members are the step residual, its diagonal block
:math:`\partial R_n/\partial u_n` with that block's solver handle, and
the readout of the new state from :math:`u_n`; the optional derivatives
come in a bundle.

The spaces are inner-product spaces, so generic code (the sweeps, the
derivative checker) measures and compares step vectors on any stack;
``A`` is the type of their reductions, small dense arrays.
"""

from typing import Optional, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.ode.stepper_derivatives import (
    StepOperator,
    StepperDerivatives,
    StepWindow,
)
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    BlockInnerProductSpace,
    BlockVector,
    SolverHandleProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)


@runtime_checkable
class StepSessionProtocol(Protocol[V]):
    """The steps of one solve.

    A session may reuse work between its steps (a factorization of a stage
    operator that is provably the same), and only between them: it is
    opened for one solve and dropped with it, so nothing it keeps outlives
    the parameters it was opened at.
    """

    def step(
        self, past_states: Sequence[V], times: Sequence[float]
    ) -> BlockVector[V]:
        """Solve one step, as :meth:`ForwardStepperProtocol.step`."""
        ...


@runtime_checkable
class ForwardStepperProtocol(Protocol[V]):
    """What an integrator needs of a stepper to march forward."""

    def depth(self) -> int:
        """The number of past stored states a step reads at most."""
        ...

    def session(self) -> StepSessionProtocol[V]:
        """A new session for one solve's steps, reusing nothing from any
        other."""
        ...

    def step(
        self, past_states: Sequence[V], times: Sequence[float]
    ) -> BlockVector[V]:
        """Solve one step from the stored states it reads, most recent
        first (single columns of the state space): the stepper's depth of
        them, or all there are at a startup step. ``times`` has one entry
        per lag, :math:`(t_{n+1}, t_n, t_{n-1}, \\dots)`, as in
        :class:`~pyapprox.ode.stepper_derivatives.StepWindow`, so steps of
        different sizes are described exactly. Returns the step's
        unknown. A session of this one step: nothing is reused from or
        for another call."""
        ...

    def readout(self) -> StepOperator[V]:
        """The map from a step's unknown to the new stored state."""
        ...

    def parameters_version(self) -> int:
        """The parameter version steps are taken at."""
        ...


@runtime_checkable
class DifferentiableStepperProtocol(Protocol[V, A]):
    """A stepper exposing its step equation and its derivatives."""

    def depth(self) -> int:
        """The number of past stored states a step reads (its largest
        lag)."""
        ...

    def unknown_space(self) -> BlockInnerProductSpace[V, A]:
        """The space of a step's unknown :math:`u_n`."""
        ...

    def residual_space(self) -> BlockInnerProductSpace[V, A]:
        """The space of a step's residual :math:`R_n`."""
        ...

    def state_space(self) -> BlockInnerProductSpace[V, A]:
        """The stored states, as one-block vectors (a window holds each
        as that block's single-column vector)."""
        ...

    def param_space(self) -> Optional[BlockInnerProductSpace[V, A]]:
        """The parameters, as one-block vectors; ``None`` without them."""
        ...

    def readout(self) -> StepOperator[V]:
        """The map from :math:`u_n` to the new stored state."""
        ...

    def step_residual(self, window: StepWindow[V]) -> BlockVector[V]:
        """:math:`R_n` at the window's unknown, past states and the
        current parameters."""
        ...

    def diag(self, window: StepWindow[V]) -> StepOperator[V]:
        r""":math:`\partial R_n/\partial u_n`, block lower triangular."""
        ...

    def diag_handle(
        self, window: StepWindow[V]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """A solver handle for :meth:`diag`; its ``transpose()`` serves
        the adjoint sweep."""
        ...

    def derivatives(self) -> StepperDerivatives[V]:
        """The optional derivatives, valid for the parameters set when
        fetched."""
        ...

    def parameters_version(self) -> int:
        """The current parameter version; a window recorded at another
        version is not differentiated at the current parameters."""
        ...


@runtime_checkable
class StepperProtocol(
    ForwardStepperProtocol[V], DifferentiableStepperProtocol[V, A], Protocol[V, A]
):
    """A stepper the integrator marches and the sweeps differentiate:
    what a stepping method builds."""
