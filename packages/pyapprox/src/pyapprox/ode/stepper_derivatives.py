r"""The derivatives of a time step's equation, as operators.

Step :math:`n` has one unknown :math:`u_n`, a block vector (one block per
stage value, plus the new state when the method is not stiffly
accurate), and one residual :math:`R_n(u_n;\, y_n, \dots, y_{n+1-k};\, p)`
with a block per unknown block. A stepper solves :math:`R_n = 0` and
reads the new stored state out of :math:`u_n`. The sweeps differentiate
:math:`R_n` with respect to

- **lag 0**, the step's own unknown :math:`u_n`: the required
  ``diag(window)``, block lower triangular, solved stage by stage;
- **lag** :math:`k \ge 1`, the stored state :math:`y_{n+1-k}`:
  ``off_diag(window, k)``;
- the parameters :math:`p`: ``param(window)``;

and contract its second derivatives with a step adjoint :math:`\mu_n`,
a block vector of the residual space. Every operator acts on block
vectors: a stored state and the parameters are one-block vectors of
their own spaces, so all step operators share one vector type.

The second derivatives come as three operator-valued fields, as in the
residual bundle: ``state_state(window, mu, a, b)``, the lag-:math:`b` to
lag-:math:`a` block of :math:`\mu^\top \partial^2 R/\partial x_a
\partial x_b`; ``state_param(window, mu, a)``, parameters to lag
:math:`a`, whose transpose is the parameter-state term; and
``param_param(window, mu)``. The bundle declares which lag pairs and
which lags have nonzero second derivatives, so the sweeps skip the rest
without computing them.

A bundle is valid for the parameters set when it was fetched: after
``set_param`` a consumer fetches the stepper's ``derivatives()`` again.

Everything here is generic over the stack's vector type: windows hold
stack vectors, and only the stepper converts them to the residual's own
form, where it calls the residual.
"""

from dataclasses import dataclass
from typing import Callable, FrozenSet, Generic, Mapping, Optional, Tuple, TypeVar

from pyapprox.ode.step_span import StepSpan
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    BlockVector,
    FormatProtocol,
    VectorSpaceProtocol,
)

V = TypeVar("V")
"""The stack's vector type."""
W = TypeVar("W")

StepOperator = ActionOperatorProtocol[BlockVector[V], FormatProtocol]
"""An operator between spaces of a step: block vectors, any format."""


def check_step_span(span: StepSpan, nstates: int) -> None:
    """Check a step's span against the stored states it reads.

    Parameters
    ----------
    span : StepSpan
        The step's sizes and times, one size per state read.
    nstates : int
        The number of stored states the step reads.

    Raises
    ------
    ValueError
        If the span does not have one size per state read.
    """
    if len(span.sizes) != nstates:
        raise ValueError(
            f"a step reading {nstates} stored states needs a span with "
            f"{nstates} sizes, got {len(span.sizes)}"
        )


@dataclass(frozen=True)
class StepWindow(Generic[V]):
    """Where a step's derivatives are evaluated.

    States are stack vectors: single-column vectors of the state space,
    converted to the residual's own form only where the residual is
    called (see :mod:`pyapprox.ode.residual_form`).

    Parameters
    ----------
    span : StepSpan
        The step's size and the time of every lag, exactly: lag 0 the
        step's end and lag :math:`l \\ge 1` the time of the stored state
        :math:`y_{n+1-l}` (see :mod:`pyapprox.ode.step_span`).
    past_states : tuple of V
        The stored states the step reads, most recent first:
        :math:`(y_n, y_{n-1}, \\dots)`, each a single-column vector of the
        state space.
    unknown : BlockVector
        The converged unknown :math:`u_n`, a single-column vector of the
        stepper's unknown space.
    parameters_version : int, default 0
        The residual's parameter version the step was taken at.

    Raises
    ------
    ValueError
        If the span does not have one size per past state.
    """

    span: StepSpan
    past_states: Tuple[V, ...]
    unknown: BlockVector[V]
    parameters_version: int = 0

    def __post_init__(self) -> None:
        check_step_span(self.span, len(self.past_states))

    def time(self) -> float:
        """:math:`t_n`, the start of the step."""
        return self.span.time()

    def deltat(self) -> float:
        """:math:`h_n`, the step's size, as given."""
        return self.span.deltat()

    def lag_time(self, lag: int) -> float:
        """:math:`t_{n+1-\\mathrm{lag}}`: the step's end at lag 0, the
        time of the stored state of that lag otherwise."""
        return self.span.lag_time(lag)

    def past_state(self, lag: int) -> V:
        """:math:`y_{n+1-\\mathrm{lag}}`, for ``lag`` at least 1.

        Raises
        ------
        ValueError
            If ``lag`` is not between 1 and the number of past states.
        """
        if not 1 <= lag <= len(self.past_states):
            raise ValueError(
                f"lag must be between 1 and {len(self.past_states)}, got {lag}"
            )
        return self.past_states[lag - 1]


OffDiagFn = Callable[[StepWindow[V], int], StepOperator[V]]
"""``(window, lag) ->`` :math:`\\partial R_n/\\partial y_{n+1-\\mathrm{lag}}`."""
StepParamFn = Callable[[StepWindow[V]], StepOperator[V]]
"""``window ->`` :math:`\\partial R_n/\\partial p`."""
StateStateFn = Callable[[StepWindow[V], BlockVector[V], int, int], StepOperator[V]]
"""``(window, mu, a, b) ->`` lag-:math:`b` to lag-:math:`a` block of
:math:`\\mu^\\top \\partial^2 R_n`."""
StateParamFn = Callable[[StepWindow[V], BlockVector[V], int], StepOperator[V]]
"""``(window, mu, a) ->`` parameters to lag :math:`a` of
:math:`\\mu^\\top \\partial^2 R_n`."""
ParamParamFn = Callable[[StepWindow[V], BlockVector[V]], StepOperator[V]]
"""``(window, mu) ->`` :math:`\\mu^\\top \\partial^2 R_n/\\partial p^2`."""

_FIELDS = ("off_diag", "param", "state_state", "state_param", "param_param")
_SECOND_ORDER = ("state_state", "state_param", "param_param")


@dataclass(frozen=True)
class StepperDerivatives(Generic[V]):
    """The optional derivatives of a step's equation.

    Build with the named constructors. Invariants: ``param`` requires
    ``off_diag``; the three second-order fields come together and require
    ``param``; ``state_pairs`` and ``param_lags`` are given exactly with
    the second-order fields, ``state_pairs`` closed under swapping its
    pairs (the :math:`(a, b)` and :math:`(b, a)` blocks are transposes)
    and every lag nonnegative.

    Parameters
    ----------
    off_diag, param, state_state, state_param, param_param : callable, optional
        See :mod:`pyapprox.ode.stepper_derivatives`.
    state_pairs : frozenset of (int, int)
        The lag pairs :math:`(a, b)` whose ``state_state`` block may be
        nonzero; every other block is zero and is never requested.
    param_lags : frozenset of int
        The lags :math:`a` whose ``state_param`` block may be nonzero.
    """

    off_diag: Optional[OffDiagFn[V]] = None
    param: Optional[StepParamFn[V]] = None
    state_state: Optional[StateStateFn[V]] = None
    state_param: Optional[StateParamFn[V]] = None
    param_param: Optional[ParamParamFn[V]] = None
    state_pairs: FrozenSet[Tuple[int, int]] = frozenset()
    param_lags: FrozenSet[int] = frozenset()

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"StepperDerivatives field '{name}' must be callable or "
                    f"None; got {type(value).__name__}"
                )
        if self.param is not None and self.off_diag is None:
            raise ValueError("param requires off_diag")
        present = [getattr(self, name) is not None for name in _SECOND_ORDER]
        if any(present) and not all(present):
            raise ValueError(
                "the three second-order fields come together: "
                + ", ".join(_SECOND_ORDER)
            )
        second_order = all(present)
        if second_order and self.param is None:
            raise ValueError("second order requires param")
        if not second_order and (self.state_pairs or self.param_lags):
            raise ValueError("state_pairs and param_lags come with second order")
        for a, b in self.state_pairs:
            if a < 0 or b < 0:
                raise ValueError(f"lags must be nonnegative, got ({a}, {b})")
            if (b, a) not in self.state_pairs:
                raise ValueError(
                    f"state_pairs has ({a}, {b}) but not ({b}, {a}); the two "
                    "blocks are transposes of each other"
                )
        if any(lag < 0 for lag in self.param_lags):
            raise ValueError(f"lags must be nonnegative, got {sorted(self.param_lags)}")

    @staticmethod
    def none() -> "StepperDerivatives[W]":
        """No derivatives: the stepper can step forward only."""
        return StepperDerivatives()

    @staticmethod
    def state_only(off_diag: OffDiagFn[W]) -> "StepperDerivatives[W]":
        """The coupling to past states and no parameters: tangent and
        adjoint sweeps of the state, no parameter gradients."""
        return StepperDerivatives(off_diag=off_diag)

    @staticmethod
    def first_order(
        off_diag: OffDiagFn[W], param: StepParamFn[W]
    ) -> "StepperDerivatives[W]":
        """The coupling to past states and the parameters: gradients."""
        return StepperDerivatives(off_diag=off_diag, param=param)

    @staticmethod
    def second_order(
        off_diag: OffDiagFn[W],
        param: StepParamFn[W],
        state_state: StateStateFn[W],
        state_param: StateParamFn[W],
        param_param: ParamParamFn[W],
        state_pairs: FrozenSet[Tuple[int, int]],
        param_lags: FrozenSet[int],
    ) -> "StepperDerivatives[W]":
        """First order plus the second derivatives: Hessian-vector
        products. ``state_pairs`` and ``param_lags`` declare which second
        derivative blocks may be nonzero."""
        return StepperDerivatives(
            off_diag=off_diag,
            param=param,
            state_state=state_state,
            state_param=state_param,
            param_param=param_param,
            state_pairs=state_pairs,
            param_lags=param_lags,
        )


@dataclass(frozen=True)
class StepSecondOrder(Generic[V]):
    r"""A step bundle's second-order fields, all present, and its declared
    blocks, with their contractions along a tangent: what a Hessian-vector
    product reads of a step residual :math:`R_n` weighted by an adjoint
    :math:`\mu` (the sweeps' step adjoint, or a functional's
    :math:`\Omega^\top \omega` on a step residual).

    Build with :meth:`of`.
    """

    state_state: StateStateFn[V]
    state_param: StateParamFn[V]
    param_param: ParamParamFn[V]
    state_pairs: FrozenSet[Tuple[int, int]]
    param_lags: FrozenSet[int]

    @staticmethod
    def of(derivs: StepperDerivatives[W], owner: str) -> "StepSecondOrder[W]":
        """The second order of ``derivs``.

        Raises
        ------
        TypeError
            If the bundle has none; ``owner`` names it in the message.
        """
        if (
            derivs.state_state is None
            or derivs.state_param is None
            or derivs.param_param is None
        ):
            raise TypeError(
                f"{owner}'s derivatives() has no second derivatives, so it has "
                "no Hessian-vector products"
            )
        return StepSecondOrder(
            derivs.state_state,
            derivs.state_param,
            derivs.param_param,
            derivs.state_pairs,
            derivs.param_lags,
        )

    def at_lag(
        self,
        window: StepWindow[V],
        mu: BlockVector[V],
        lag: int,
        lag_tangents: Mapping[int, BlockVector[V]],
        direction: BlockVector[V],
        space: VectorSpaceProtocol[BlockVector[V]],
    ) -> Optional[BlockVector[V]]:
        r""":math:`\sum_b (\mu^\top R_n)_{ab}\, dX_b + (\mu^\top
        R_n)_{ap}\, v` for :math:`a` = ``lag``, over the declared blocks the
        window has (a startup step reads fewer lags, so ``lag_tangents``
        holds those it has); a new vector, or None when none is declared.

        Parameters
        ----------
        window : StepWindow
            The step.
        mu : BlockVector
            The adjoint, a single column of the residual space.
        lag : int
            :math:`a`.
        lag_tangents : mapping
            :math:`dX_b` by lag, each a block vector of its lag's space.
        direction : BlockVector
            :math:`v`, a one-block parameter vector.
        space : VectorSpaceProtocol
            Lag ``lag``'s space, for the sum.
        """
        total: Optional[BlockVector[V]] = None
        for a, other in sorted(self.state_pairs):
            if a != lag or other not in lag_tangents:
                continue
            term = self.state_state(window, mu, lag, other).apply(lag_tangents[other])
            total = term if total is None else space.axpby(
                1.0, term, 1.0, total, out=total
            )
        if lag in self.param_lags:
            term = self.state_param(window, mu, lag).apply(direction)
            total = term if total is None else space.axpby(
                1.0, term, 1.0, total, out=total
            )
        return total

    def wrt_param(
        self,
        window: StepWindow[V],
        mu: BlockVector[V],
        lag_tangents: Mapping[int, BlockVector[V]],
        direction: BlockVector[V],
        space: VectorSpaceProtocol[V],
    ) -> V:
        r""":math:`\sum_a (\mu^\top R_n)_{pa}\, dX_a + (\mu^\top R_n)_{pp}\,
        v`, the mixed blocks transposed, over the declared lags the window
        has; a new vector.

        Parameters
        ----------
        window, mu, lag_tangents, direction
            As for :meth:`at_lag`.
        space : VectorSpaceProtocol
            The parameters' (one block's) space, for the sum.
        """
        total = self.param_param(window, mu).apply(direction).blocks[0]
        for lag in sorted(self.param_lags):
            if lag not in lag_tangents:
                continue
            mixed = self.state_param(window, mu, lag)
            term = mixed.apply_transpose(lag_tangents[lag]).blocks[0]
            total = space.axpby(1.0, term, 1.0, total, out=total)
        return total
