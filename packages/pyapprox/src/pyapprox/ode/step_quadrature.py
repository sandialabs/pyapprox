r"""A stepper's own quadrature in time: its method applied to
:math:`\dot z = q(y, t)`.

A functional integrating over time,
:math:`Q = \int_{t_0}^{t_N} q(y, t)\, dt`, is the final value of
:math:`z` for the appended equation :math:`\dot z = q(y, t)`,
:math:`z(t_0) = 0`. Integrating it with the stepper's own method makes
the quadrature consistent with the step by construction, so it is as
accurate as the method is for the state, and needs no rule chosen apart
from the method. Since :math:`q` does not depend on :math:`z`, each step
is an explicit linear recursion,

.. math::

    \alpha_0 z_{n+1} + \sum_{l \ge 1} \alpha_l z_{n+1-l}
    = \sum_{\text{nodes}} w\, q(y_{\text{node}}, t_{\text{node}}),

whose nodes are states the step already has: blocks of its unknown or
stored states it read. A :class:`StepRule` holds one step's
:math:`\alpha` and nodes.

- A Runge-Kutta method has :math:`\alpha = (1, -1)` and nodes
  :math:`\Delta t\, b_i` at the stages, :math:`t_n + c_i \Delta t`: for
  backward Euler, Crank-Nicolson and implicit midpoint the right
  rectangle, the trapezoid and the midpoint rule.
- A linear multistep method has its own :math:`\alpha` and nodes
  :math:`\Delta t\, \beta_l` at the lags. For Adams methods,
  :math:`\alpha = (1, -1, 0, \dots)`, this is a sum of per-step terms;
  for a backward differentiation formula the recursion weighs each
  :math:`q` by the recursion's response up to the end, a quadrature
  over the whole trajectory.

With :math:`q = f` the recursion is the step's own for :math:`M y`, so
:math:`z_N = M (y_N - y_0)` for every method (:math:`\sum_l \alpha_l = 0`
by consistency): the identity that checks a rule against its stepper.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, TypeVar, runtime_checkable

from pyapprox.ode.multistep_methods import MultistepFormulaProtocol
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import check_step_span
from pyapprox.ode.tableau import ButcherTableau
from pyapprox.util.backends.protocols import ArrayProtocol

R = TypeVar("R", bound=ArrayProtocol)


@dataclass(frozen=True)
class QuadratureNode:
    r"""One node of a step's rule.

    Parameters
    ----------
    weight : float
        The node's weight: :math:`\Delta t\, b_i` or :math:`\Delta t\,
        \beta_l`.
    time : float
        The node's time.
    lag : int
        Where the node's state lives: 0 for a block of the step's unknown,
        :math:`l \ge 1` for the stored state :math:`y_{n+1-l}`.
    block : int
        The unknown's block when ``lag`` is 0; 0 otherwise.
    """

    weight: float
    time: float
    lag: int
    block: int


@dataclass(frozen=True)
class StepRule:
    r"""One step of the recursion :math:`\alpha_0 z_{n+1} + \sum_{l \ge 1}
    \alpha_l z_{n+1-l} = \sum w\, q(\text{node})`.

    Parameters
    ----------
    alpha : tuple of float
        :math:`(\alpha_0, \dots, \alpha_k)`, one per lag of the step;
        :math:`\alpha_0 \ne 0`.
    nodes : tuple of QuadratureNode
        The right-hand side's nodes, each of nonzero weight.

    Raises
    ------
    ValueError
        If ``alpha`` reads no earlier value or :math:`\alpha_0 = 0`, or a
        node's lag is beyond ``alpha``'s.
    """

    alpha: Tuple[float, ...]
    nodes: Tuple[QuadratureNode, ...]

    def __post_init__(self) -> None:
        if len(self.alpha) < 2 or self.alpha[0] == 0.0:
            raise ValueError(
                f"alpha must have alpha_0 != 0 and at least one lag, got {self.alpha}"
            )
        for node in self.nodes:
            if not 0 <= node.lag < len(self.alpha):
                raise ValueError(
                    f"a node at lag {node.lag} is beyond the step's "
                    f"{len(self.alpha) - 1} lags"
                )

    def advance(self, past: Sequence[R], increment: R) -> R:
        r""":math:`z_{n+1} = (\text{increment} - \sum_{l \ge 1} \alpha_l
        z_{n+1-l}) / \alpha_0`.

        Parameters
        ----------
        past : sequence
            :math:`(z_n, z_{n-1}, \dots)`, most recent first, one per lag
            :math:`l \ge 1`.
        increment : array
            :math:`\sum w\, q(\text{node})`.

        Raises
        ------
        ValueError
            If ``past`` does not have one value per lag.
        """
        if len(past) != len(self.alpha) - 1:
            raise ValueError(
                f"the step reads {len(self.alpha) - 1} earlier values, got {len(past)}"
            )
        total = increment
        for alpha, value in zip(self.alpha[1:], past):
            if alpha != 0.0:
                total = total - alpha * value
        return total / self.alpha[0]


@runtime_checkable
class StepQuadratureProtocol(Protocol):
    r"""A stepper's method applied to :math:`\dot z = q`, step by step."""

    def rule(self, span: StepSpan) -> StepRule:
        """The recursion of the step at ``span``."""
        ...


class TableauQuadrature:
    r"""A Runge-Kutta tableau applied to :math:`\dot z = q`:
    :math:`z_{n+1} = z_n + \Delta t \sum_i b_i\, q(Y_i, t_n + c_i \Delta
    t)`.

    Stage :math:`i` is block :math:`i - o` of the step's unknown, where
    :math:`o` is 1 when the first stage is trivial and 0 otherwise; a
    trivial first stage is :math:`y_n` itself, the stored state at lag 1.
    Stages of zero weight are no nodes.

    Parameters
    ----------
    tableau : ButcherTableau
        The method.
    """

    def __init__(self, tableau: ButcherTableau) -> None:
        self._tableau = tableau
        offset = 1 if tableau.has_trivial_first_stage() else 0
        # (b_i, c_i, lag, block) per stage of nonzero weight.
        self._stages = tuple(
            (b, c, 1, 0) if i < offset else (b, c, 0, i - offset)
            for i, (b, c) in enumerate(zip(tableau.b, tableau.c))
            if b != 0.0
        )

    def rule(self, span: StepSpan) -> StepRule:
        """The recursion of the step at ``span``.

        Raises
        ------
        ValueError
            If ``span`` does not have one size: a one-step method reads
            one state.
        """
        check_step_span(span, 1)
        time, deltat = span.time(), span.deltat()
        return StepRule(
            (1.0, -1.0),
            tuple(
                QuadratureNode(deltat * b, time + c * deltat, lag, block)
                for b, c, lag, block in self._stages
            ),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tableau.name})"


class MultistepQuadrature:
    r"""A linear multistep method applied to :math:`\dot z = q`: the
    step's formula, :math:`\sum_l \alpha_l z_{n+1-l} = \Delta t \sum_l
    \beta_l\, q(y_{n+1-l}, t_{n+1-l})`, with the coefficients the stepper
    takes for the same span (a startup step's included).

    Lag 0 is the step's unknown, the new state; lag :math:`l \ge 1` the
    stored state :math:`y_{n+1-l}`. Lags of zero :math:`\beta_l` are no
    nodes.

    Parameters
    ----------
    method : MultistepFormulaProtocol
        The method.
    """

    def __init__(self, method: MultistepFormulaProtocol) -> None:
        self._method = method

    def rule(self, span: StepSpan) -> StepRule:
        """The recursion of the step at ``span``.

        Raises
        ------
        ValueError
            If the method has no formula for the span.
        """
        formula = self._method.coefficients_at(span)
        deltat = span.deltat()
        return StepRule(
            formula.alpha,
            tuple(
                QuadratureNode(deltat * beta, span.lag_time(lag), lag, 0)
                for lag, beta in enumerate(formula.beta)
                if beta != 0.0
            ),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._method.name})"
