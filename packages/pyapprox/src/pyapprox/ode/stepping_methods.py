r"""Stepping methods: what a model is given to step its own system.

A model builds the system it steps (:class:`MassFormSystem`), so it
cannot be handed a built stepper; it is handed a method, which binds to
that system with :meth:`SteppingMethodProtocol.stepper`. A method carries
everything about the scheme that is not the system: its coefficients and,
for an implicit method, the nonlinear solver its stages use. An explicit
method takes no solver, so pairing one with solver settings is refused
rather than ignored.

New methods satisfy :class:`SteppingMethodProtocol` and are passed in;
nothing here enumerates them.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.linear_multistep import LinearMultistepStepper
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.multistep_methods import MultistepFormulaProtocol
from pyapprox.ode.pinned_rows import RowPinningProtocol
from pyapprox.ode.protocols.differentiable_stepper import StepperProtocol
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.tableau import ButcherTableau
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import FormatProtocol, ReusePolicyProtocol
from pyapprox.util.rootfinding.nonlinear import Newton, NonlinearSolverProtocol

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


@runtime_checkable
class SteppingMethodProtocol(Protocol[V, A]):
    """A time-stepping method, bound to a system on request."""

    def name(self) -> str:
        """The method's name."""
        ...

    def stepper(
        self,
        system: MassFormSystem[V, A, F],
        row_pinning: Optional[RowPinningProtocol[V]] = None,
    ) -> StepperProtocol[V, A]:
        """A stepper of this method for ``system``.

        Parameters
        ----------
        system : MassFormSystem
            What to step.
        row_pinning : RowPinningProtocol, optional
            Rows pinned to boundary data inside the step's solves (see
            :mod:`pyapprox.ode.pinned_rows`); None for none.

        Returns
        -------
        StepperProtocol
            A new stepper; each call builds its own.
        """
        ...


def _checked_solver(
    name: str,
    explicit: bool,
    nonlinear_solver: Optional[NonlinearSolverProtocol[V, A]],
) -> Optional[NonlinearSolverProtocol[V, A]]:
    """The solver an implicit method's stages use: the given one, or
    Newton at its defaults; None for an explicit method.

    Raises
    ------
    TypeError
        If an explicit method is given a solver, or the solver does not
        satisfy :class:`NonlinearSolverProtocol`.
    """
    if nonlinear_solver is None:
        return None if explicit else Newton()
    if explicit:
        raise TypeError(f"{name} is explicit, so it takes no nonlinear solver")
    if not isinstance(nonlinear_solver, NonlinearSolverProtocol):
        raise TypeError(
            "nonlinear_solver must satisfy NonlinearSolverProtocol, got "
            f"{type(nonlinear_solver).__name__}"
        )
    return nonlinear_solver


class RungeKutta(Generic[V, A]):
    """A Runge-Kutta method: a tableau and, when it has implicit stages,
    their nonlinear solver.

    Parameters
    ----------
    tableau : ButcherTableau
        The coefficients.
    nonlinear_solver : NonlinearSolverProtocol, optional
        Solves the implicit stages. Defaults to :class:`Newton` at its
        defaults for a tableau with implicit stages.
    reuse : ReusePolicyProtocol, optional
        How a solve keeps stage setups between steps; the stepper's
        default when None (see :class:`RungeKuttaStepper`).

    Raises
    ------
    TypeError
        If ``tableau`` is not a :class:`ButcherTableau`, an explicit
        tableau is given a solver, or the solver does not satisfy its
        protocol.

    Examples
    --------
    >>> method = RungeKutta(backward_euler(), Newton(atol=1e-12, rtol=0.0))
    >>> stepper = method.stepper(dense_mass_form_system(model))
    """

    def __init__(
        self,
        tableau: ButcherTableau,
        nonlinear_solver: Optional[NonlinearSolverProtocol[V, A]] = None,
        reuse: Optional[ReusePolicyProtocol] = None,
    ) -> None:
        if not isinstance(tableau, ButcherTableau):
            raise TypeError(
                f"tableau must be a ButcherTableau, got {type(tableau).__name__}"
            )
        self._tableau = tableau
        self._solver = _checked_solver(
            tableau.name, tableau.is_explicit(), nonlinear_solver
        )
        self._reuse = reuse

    def name(self) -> str:
        """The tableau's name."""
        return self._tableau.name

    def tableau(self) -> ButcherTableau:
        """The coefficients."""
        return self._tableau

    def stepper(
        self,
        system: MassFormSystem[V, A, F],
        row_pinning: Optional[RowPinningProtocol[V]] = None,
    ) -> RungeKuttaStepper[V, A, F]:
        """A :class:`RungeKuttaStepper` of this tableau for ``system``."""
        return RungeKuttaStepper(
            self._tableau, system, self._solver, row_pinning, self._reuse
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tableau.name})"


class LinearMultistep(Generic[V, A]):
    """A linear multistep method: its coefficients and, when implicit,
    the nonlinear solver of each step.

    Parameters
    ----------
    method : MultistepFormulaProtocol
        Gives the formula of each step from its times.
    nonlinear_solver : NonlinearSolverProtocol, optional
        Solves each step of an implicit method. Defaults to
        :class:`Newton` at its defaults for an implicit method.
    reuse : ReusePolicyProtocol, optional
        How a solve keeps step setups between steps; the stepper's
        default when None (see :class:`LinearMultistepStepper`).

    Raises
    ------
    TypeError
        If ``method`` does not satisfy :class:`MultistepFormulaProtocol`,
        an explicit method is given a solver, or the solver does not
        satisfy its protocol.
    """

    def __init__(
        self,
        method: MultistepFormulaProtocol,
        nonlinear_solver: Optional[NonlinearSolverProtocol[V, A]] = None,
        reuse: Optional[ReusePolicyProtocol] = None,
    ) -> None:
        if not isinstance(method, MultistepFormulaProtocol):
            raise TypeError(
                "method must satisfy MultistepFormulaProtocol, got "
                f"{type(method).__name__}"
            )
        self._method = method
        self._solver = _checked_solver(
            method.name, method.is_explicit(), nonlinear_solver
        )
        self._reuse = reuse

    def name(self) -> str:
        """The method's name."""
        return self._method.name

    def coefficients(self) -> MultistepFormulaProtocol:
        """The method's formulas."""
        return self._method

    def stepper(
        self,
        system: MassFormSystem[V, A, F],
        row_pinning: Optional[RowPinningProtocol[V]] = None,
    ) -> LinearMultistepStepper[V, A, F]:
        """A :class:`LinearMultistepStepper` of this method for
        ``system``."""
        return LinearMultistepStepper(
            self._method, system, self._solver, row_pinning, self._reuse
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._method.name})"
