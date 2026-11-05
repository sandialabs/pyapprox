r"""A stepping method's own value inside a step, with its derivatives.

A method's continuous extension gives :math:`y(t)` for :math:`t` in step
:math:`n`'s span from that step's window (its unknown, so its stages, and
the stored states it read), consistently with the integration: a
Runge-Kutta method's weight polynomials :math:`b_i(\theta)`
(:meth:`~pyapprox.ode.tableau.ButcherTableau.weights_at`), a multistep
method's interpolant through its lags. It ends at the step's new state.

Its derivatives are those of a function of the step's arguments (the
unknown, the past states, the parameters), in the form a step term's
bundle takes them (:class:`~pyapprox.ode.trajectory_terms.
StepTermDerivatives`), so a term reading :math:`y(t)` (an observation at a
time that is not a stored time) is a step term whose derivatives follow
from these: first derivatives as step operators into the state, second
derivatives weighted by a state-space weight :math:`\lambda`, each the
derivative of the matching :math:`(\partial y(t)/\partial\,\cdot)^\top
\lambda` along a tangent of all the step's arguments together.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.stepper_derivatives import StepOperator, StepWindow
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import BlockInnerProductSpace, BlockVector

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)

ExtensionLinearizationFn = Callable[[StepWindow[V], float], StepOperator[V]]
"""``(window, time) ->`` :math:`\\partial y(t)` in one argument (the
unknown, or the parameters), a step operator into the state as a
one-block vector."""

ExtensionLagFn = Callable[[StepWindow[V], float, int], StepOperator[V]]
"""``(window, time, lag) ->`` :math:`\\partial y(t)/\\partial
y_{n+1-\\mathrm{lag}}`, state to state."""

ExtensionUnknownCurvatureFn = Callable[
    [StepWindow[V], float, StepWindow[V], V, V], Optional[BlockVector[V]]
]
"""``(window, time, tangents, direction, weights) ->`` the derivative of
:math:`(\\partial y(t)/\\partial u_n)^\\top \\lambda` along the tangent;
None for zero."""

ExtensionLagCurvatureFn = Callable[
    [StepWindow[V], float, StepWindow[V], int, V, V], Optional[V]
]
"""``(window, time, tangents, lag, direction, weights) ->`` the derivative
of :math:`(\\partial y(t)/\\partial y_{n+1-\\mathrm{lag}})^\\top \\lambda`
along the tangent; None for zero."""

ExtensionParamCurvatureFn = Callable[
    [StepWindow[V], float, StepWindow[V], V, V], Optional[V]
]
"""``(window, time, tangents, direction, weights) ->`` the derivative of
:math:`(\\partial y(t)/\\partial p)^\\top \\lambda` along the tangent;
None for zero."""


@dataclass(frozen=True)
class ContinuousExtensionDerivatives(Generic[V]):
    r"""A continuous extension's optional derivatives (see the module
    docstring). Build with :meth:`none`, :meth:`first_order` or
    :meth:`second_order`.

    Parameters
    ----------
    wrt_unknown, wrt_lag : callable, optional
        :math:`\partial y(t)/\partial u_n` and
        :math:`\partial y(t)/\partial y_{n+1-\mathrm{lag}}`.
    wrt_param : callable, optional
        :math:`\partial y(t)/\partial p`; None without parameters.
    curvature_wrt_unknown, curvature_wrt_lag, curvature_wrt_param :
    callable, optional
        The second derivatives, weighted by :math:`\lambda`.

    Raises
    ------
    ValueError
        If only one of ``wrt_unknown`` and ``wrt_lag`` is given, the
        parameter derivative or curvatures without them, or some but not
        all three curvatures.
    """

    wrt_unknown: Optional[ExtensionLinearizationFn[V]] = None
    wrt_lag: Optional[ExtensionLagFn[V]] = None
    wrt_param: Optional[ExtensionLinearizationFn[V]] = None
    curvature_wrt_unknown: Optional[ExtensionUnknownCurvatureFn[V]] = None
    curvature_wrt_lag: Optional[ExtensionLagCurvatureFn[V]] = None
    curvature_wrt_param: Optional[ExtensionParamCurvatureFn[V]] = None

    def __post_init__(self) -> None:
        first = (self.wrt_unknown is None, self.wrt_lag is None)
        if first[0] != first[1]:
            raise ValueError("wrt_unknown and wrt_lag come together")
        curvatures = [
            field is not None
            for field in (
                self.curvature_wrt_unknown,
                self.curvature_wrt_lag,
                self.curvature_wrt_param,
            )
        ]
        if any(curvatures) and not all(curvatures):
            raise ValueError("the curvatures come all three or none")
        if self.wrt_unknown is None and (self.wrt_param is not None or any(curvatures)):
            raise ValueError("the other derivatives need wrt_unknown and wrt_lag")

    @staticmethod
    def none() -> "ContinuousExtensionDerivatives[V]":
        """No derivatives: the extension is only evaluated."""
        return ContinuousExtensionDerivatives()

    @staticmethod
    def first_order(
        wrt_unknown: ExtensionLinearizationFn[V],
        wrt_lag: ExtensionLagFn[V],
        wrt_param: Optional[ExtensionLinearizationFn[V]],
    ) -> "ContinuousExtensionDerivatives[V]":
        """The first derivatives."""
        return ContinuousExtensionDerivatives(wrt_unknown, wrt_lag, wrt_param)

    @staticmethod
    def second_order(
        wrt_unknown: ExtensionLinearizationFn[V],
        wrt_lag: ExtensionLagFn[V],
        wrt_param: Optional[ExtensionLinearizationFn[V]],
        curvature_wrt_unknown: ExtensionUnknownCurvatureFn[V],
        curvature_wrt_lag: ExtensionLagCurvatureFn[V],
        curvature_wrt_param: ExtensionParamCurvatureFn[V],
    ) -> "ContinuousExtensionDerivatives[V]":
        """The first derivatives and the three curvatures."""
        return ContinuousExtensionDerivatives(
            wrt_unknown,
            wrt_lag,
            wrt_param,
            curvature_wrt_unknown,
            curvature_wrt_lag,
            curvature_wrt_param,
        )


@runtime_checkable
class ContinuousExtensionProtocol(Protocol[V, A]):
    """A stepping method's value inside a step, from the step's window;
    its derivatives act on the stepper's spaces, which it gives."""

    def value(self, window: StepWindow[V], time: float) -> V:
        r""":math:`y(t)`, a single column of the state space, for ``time``
        in the window's span :math:`[t_n, t_{n+1}]`."""
        ...

    def derivatives(self) -> ContinuousExtensionDerivatives[V]:
        """The optional derivatives, valid for the parameters set when
        fetched."""
        ...

    def unknown_space(self) -> BlockInnerProductSpace[V, A]:
        """The space of a step's unknown."""
        ...

    def state_space(self) -> BlockInnerProductSpace[V, A]:
        """The stored states (and :math:`y(t)`), as one-block vectors."""
        ...

    def param_space(self) -> Optional[BlockInnerProductSpace[V, A]]:
        """The parameters as one-block vectors; None without them."""
        ...


def fraction_of_step(window: StepWindow[V], time: float) -> float:
    r""":math:`\theta = (t - t_n)/h_n`, the fraction of the window's step
    at ``time``, clipped to :math:`[0, 1]` within rounding of the span's
    ends.

    Raises
    ------
    ValueError
        If ``time`` lies outside the step by more than rounding.
    """
    start, deltat = window.time(), window.deltat()
    theta = (time - start) / deltat
    slack = 1e-12
    if not -slack <= theta <= 1.0 + slack:
        raise ValueError(
            f"time {time} is outside the step [{start}, {window.lag_time(0)}]"
        )
    return min(max(theta, 0.0), 1.0)
