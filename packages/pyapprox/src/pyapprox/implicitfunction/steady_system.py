r"""The steady problem :math:`R(u; p) = 0`, solved on any stack.

A :class:`SteadySystem` is a residual (:class:`~pyapprox.implicitfunction.
residual.ResidualProtocol`), its state space and a setup strategy
(:class:`~pyapprox.implicitfunction.setups.SetupStrategyProtocol`), as a
nonlinear solver reads them
(:class:`~pyapprox.util.rootfinding.nonlinear.NonlinearSystemProtocol`):
the value is :math:`R`, a Newton step solves with :math:`R_u`.

Every handle for :math:`R_u` is a request to the strategy, which decides
whether to set it up or return one it kept; the system acts the same under
every strategy. A Newton iterate is requested without a position, so its
setup is never found again; a solved point is its own position, so a
strategy keeping setups at solutions answers every sensitivity solve at
the point with one setup.

:meth:`SteadySystem.solve` returns a :class:`SteadyPoint`: the state, the
parameter version it was solved at, how the solve ended, and handles for
:math:`R_u` there, which sensitivities solve with.
"""

from typing import Generic, Hashable, Optional, TypeVar

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.residual_derivatives import ResidualDerivatives
from pyapprox.implicitfunction.setups import (
    ResidualJacobianBuilder,
    SetupStrategyProtocol,
    SteadyRequest,
)
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    InnerProductSpaceProtocol,
    SolverHandleProtocol,
)
from pyapprox.util.rootfinding.nonlinear import (
    NonlinearSolveInfo,
    NonlinearSolverProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


class SteadyPoint(Generic[V, A]):
    """A solve of a :class:`SteadySystem`: where it ended, at which
    parameters, and :math:`R_u` there. The point is the position its
    state is requested at.

    Build with :meth:`SteadySystem.solve`.
    """

    def __init__(
        self,
        system: "SteadySystem[V, A, F]",
        state: V,
        parameters_version: int,
        info: NonlinearSolveInfo[A],
    ) -> None:
        self._system = system
        self._state, self._version, self._info = state, parameters_version, info

    def state(self) -> V:
        """The last iterate; the solution when :meth:`converged`."""
        return self._state

    def parameters_version(self) -> int:
        """The residual's parameter version the solve ran at."""
        return self._version

    def info(self) -> NonlinearSolveInfo[A]:
        """How the solve ended."""
        return self._info

    def converged(self) -> bool:
        """Whether the state solves :math:`R(u; p) = 0`."""
        return self._info.converged

    def solution(self) -> V:
        r"""The state, as a solution of the current residual.

        Raises
        ------
        ValueError
            If the solve did not converge, or the residual's parameters
            have changed since: the state is not a solution of the current
            residual.
        """
        if not self.converged():
            raise ValueError(
                f"the steady solve did not converge ({self._info.message}); "
                "there is no solution to linearize at"
            )
        current = self._system.residual_version()
        if current != self._version:
            raise ValueError(
                f"the residual's parameters changed (version {self._version} "
                f"-> {current}) after this solve; solve again"
            )
        return self._state

    def jacobian_handle(self) -> SolverHandleProtocol[V]:
        r"""A handle for :math:`R_u` at the solution, requested from the
        system's strategy with this point as the position.

        Raises
        ------
        ValueError
            As :meth:`solution`.
        """
        return self._system.jacobian_handle(self.solution(), position=self)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(converged={self.converged()}, "
            f"parameters_version={self._version})"
        )


class SteadySystem(Generic[V, A, F]):
    r""":math:`R(u; p) = 0` as a nonlinear solver reads it.

    Satisfies :class:`~pyapprox.util.rootfinding.nonlinear.
    NonlinearSystemProtocol`.

    Parameters
    ----------
    residual : ResidualProtocol
        :math:`R`, with a Jacobian in format ``F``.
    space : InnerProductSpaceProtocol
        The state space; states are its single-column vectors.
    setups : SetupStrategyProtocol
        Sets up :math:`R_u` or returns a kept handle, with its linear
        solver, which pairs with ``F``. Whoever changes the residual's
        parameters calls its ``clear``.

    Raises
    ------
    TypeError
        If ``residual`` does not satisfy :class:`ResidualProtocol` or its
        bundle has no Jacobian, or ``space`` is not an inner-product space.
    """

    def __init__(
        self,
        residual: ResidualProtocol[V, F],
        space: InnerProductSpaceProtocol[V, A],
        setups: SetupStrategyProtocol[SteadyRequest[V, F], V, F],
    ) -> None:
        if not isinstance(residual, ResidualProtocol):
            raise TypeError(
                f"residual must satisfy ResidualProtocol, got {type(residual).__name__}"
            )
        if not isinstance(space, InnerProductSpaceProtocol):
            raise TypeError(
                "space must satisfy InnerProductSpaceProtocol, got "
                f"{type(space).__name__}"
            )
        jacobian = residual.derivatives().jacobian
        if jacobian is None:
            raise TypeError(
                f"{type(residual).__name__}'s derivatives bundle has no "
                "jacobian; a steady solve needs one"
            )
        self._residual, self._space = residual, space
        # Which fields are present is fixed for the residual's life.
        self._affine = jacobian.affine()
        self._setups = setups
        self._builder = ResidualJacobianBuilder(residual)

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._space

    def residual(self, x: V) -> V:
        """:math:`R(x)`."""
        return self._residual(x)

    def is_affine(self) -> bool:
        """Whether :math:`R_u` is constant, so one exact Newton step
        solves."""
        return self._affine

    def residual_version(self) -> int:
        """The residual's current parameter version."""
        return self._residual.parameters_version()

    def residual_derivatives(self) -> ResidualDerivatives[V, F]:
        """The residual's derivatives bundle."""
        return self._residual.derivatives()

    def jacobian_handle(
        self, x: V, position: Optional[Hashable] = None
    ) -> SolverHandleProtocol[V]:
        r"""A handle for :math:`R_u` at ``x``, from the strategy.

        Parameters
        ----------
        x : V
            A single column of the state space; read only while the
            handle is formed.
        position : hashable, optional
            Names ``x`` when it is a solution (a :class:`SteadyPoint`);
            None for a Newton iterate.
        """
        request = SteadyRequest(self._residual, x, position)
        return self._setups.handle(request, self._builder)

    def solve(
        self, nonlinear: NonlinearSolverProtocol[V, A], initial_guess: V
    ) -> SteadyPoint[V, A]:
        """Solve from ``initial_guess`` with ``nonlinear``. Never raises on
        failure: check :meth:`SteadyPoint.converged`."""
        version = self.residual_version()
        state, info = nonlinear.solve(self, initial_guess)
        return SteadyPoint(self, state, version, info)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._residual!r}, {self._setups!r})"
