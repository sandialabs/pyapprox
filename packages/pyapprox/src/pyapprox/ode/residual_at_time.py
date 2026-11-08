r"""One member of a right-hand side's family as a time-free residual.

A right-hand side :math:`f(y, t; p)` is a family of residuals, one per
time. :class:`RightHandSideAtTime` is the member at one time :math:`t^*`,
:math:`R(u; p) = f(u, t^*; p)`, as
:class:`~pyapprox.implicitfunction.residual.ResidualProtocol` reads it:
its value and bundle take and return single-column vectors of the state
space, whatever form the right-hand side itself takes (the form converts
once, here).
"""

from typing import Generic, TypeVar

from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    precompose_states,
)
from pyapprox.ode.mass_form_system import VersionedRightHandSideProtocol
from pyapprox.ode.residual_form import ResidualFormProtocol
from pyapprox.util.linalg.operators import FormatProtocol

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


class RightHandSideAtTime(Generic[V, F]):
    r""":math:`R(u; p) = f(u, t^*; p)`: a right-hand side at one time.

    Its bundle is :meth:`~pyapprox.ode.residual_derivatives.
    ODEResidualDerivatives.at_time` of :math:`f`'s, with states and
    adjoints converted to :math:`f`'s form before :math:`f`'s callables
    see them; its parameter version is :math:`f`'s.

    Parameters
    ----------
    rhs : VersionedRightHandSideProtocol
        :math:`f`, reporting its parameter version.
    time : float
        :math:`t^*`.
    form : ResidualFormProtocol
        Converts the space's vectors to :math:`f`'s state form and back.

    Raises
    ------
    TypeError
        If ``rhs`` does not satisfy ``VersionedRightHandSideProtocol`` or
        ``form`` does not satisfy ``ResidualFormProtocol``.
    """

    def __init__(
        self,
        rhs: VersionedRightHandSideProtocol[V, F],
        time: float,
        form: ResidualFormProtocol[V, V],
    ) -> None:
        if not isinstance(rhs, VersionedRightHandSideProtocol):
            raise TypeError(
                "rhs must satisfy VersionedRightHandSideProtocol, got "
                f"{type(rhs).__name__}"
            )
        if not isinstance(form, ResidualFormProtocol):
            raise TypeError(
                f"form must satisfy ResidualFormProtocol, got {type(form).__name__}"
            )
        self._rhs, self._time, self._form = rhs, time, form
        # Which fields are present is fixed for the right-hand side's life.
        self._derivatives = precompose_states(
            rhs.derivatives().at_time(time), form.to_residual
        )

    def time(self) -> float:
        r""":math:`t^*`."""
        return self._time

    def __call__(self, state: V) -> V:
        r""":math:`f(u, t^*)` as a single column."""
        return self._form.from_residual(
            self._rhs(self._form.to_residual(state), self._time)
        )

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._derivatives

    def parameters_version(self) -> int:
        return self._rhs.parameters_version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._rhs!r}, time={self._time})"
