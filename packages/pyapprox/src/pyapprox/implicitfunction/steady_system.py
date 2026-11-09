r"""The steady problem :math:`R(u; p) = 0`, solved on any stack.

A :class:`SteadySystem` is a residual (:class:`~pyapprox.implicitfunction.
residual.ResidualProtocol`), its state space and a linear solver, as a
nonlinear solver reads them
(:class:`~pyapprox.util.rootfinding.nonlinear.NonlinearSystemProtocol`):
the value is :math:`R`, a Newton step solves with :math:`R_u`.

Solver setups are labeled so they are reused exactly when the operator
is the same. Labels hold identity tokens and typed values, never names,
so labels of different systems cannot be equal even in a shared cache:

- A constant :math:`R_u` (an affine :math:`R`) is labeled by the
  system's own identity and the parameter version, so one setup serves
  every Newton step, the solved point and repeated solves until the
  parameters change.
- A :math:`R_u` that depends on the state is labeled with a fresh
  identity at each Newton iterate, provisionally, and at the solved
  point, so no setup is reused for another state.

:meth:`SteadySystem.solve` returns a :class:`SteadyPoint`: the state, the
parameter version it was solved at, how the solve ended, and a handle for
:math:`R_u` at the state, set up on first request, which sensitivities
solve with.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    InnerProductSpaceProtocol,
    JacobianSnapshotProtocol,
    KeepWithinSolve,
    KeyedSolverProtocol,
    LinearizationPoint,
    LinearSolverProtocol,
    ReuseKey,
    ReusePolicyProtocol,
    SolverCapabilities,
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
    parameters, and :math:`R_u` there.

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
        self._handle: Optional[SolverHandleProtocol[V]] = None

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
        r"""A handle for :math:`R_u` at the state, set up on first request
        and kept; for a constant :math:`R_u`, the setup the solve used.

        Raises
        ------
        ValueError
            As :meth:`solution`.
        """
        self.solution()
        if self._handle is None:
            self._handle = self._system.solution_handle(self._state, self._version)
        return self._handle

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
    solver : LinearSolverProtocol
        Solves with :math:`R_u`; pairs with ``F``.
    reuse : ReusePolicyProtocol, optional
        How setups are kept; by default one is (``KeepWithinSolve(1)``),
        enough for a constant :math:`R_u`.

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
        solver: LinearSolverProtocol[V, F],
        reuse: Optional[ReusePolicyProtocol] = None,
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
        self._jacobian: ResidualJacobian[V, F] = jacobian
        policy = KeepWithinSolve(1) if reuse is None else reuse
        self._solver: KeyedSolverProtocol[V, F] = policy.solver(solver, 1)
        # Names this system's constant R_u; equal only to itself.
        self._identity = object()

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._space

    def residual(self, x: V) -> V:
        """:math:`R(x)`."""
        return self._residual(x)

    def is_affine(self) -> bool:
        """Whether :math:`R_u` is constant, so one exact Newton step
        solves."""
        return self._jacobian.affine()

    def residual_version(self) -> int:
        """The residual's current parameter version."""
        return self._residual.parameters_version()

    def residual_derivatives(self) -> ResidualDerivatives[V, F]:
        """The residual's derivatives bundle."""
        return self._residual.derivatives()

    def solver_capabilities(self) -> SolverCapabilities:
        """What the linear solver can do: whether it solves transposed."""
        return self._solver.capabilities()

    def jacobian_handle(self, x: V) -> SolverHandleProtocol[V]:
        r"""A handle for :math:`R_u` at the Newton iterate ``x``."""
        version = self.residual_version()
        if self.is_affine():
            return self._constant_handle(x, version)
        point = LinearizationPoint(version, object(), provisional=True)
        return self._solver.setup(self._jacobian(x, point))

    def solution_handle(
        self, state: V, parameters_version: int
    ) -> SolverHandleProtocol[V]:
        r"""A handle for :math:`R_u` at a solved ``state``, not provisional;
        the constant one's kept setup when :math:`R_u` is constant."""
        if self.is_affine():
            return self._constant_handle(state, parameters_version)
        point = LinearizationPoint(parameters_version, object())
        return self._solver.setup(self._jacobian(state, point))

    def _constant_handle(self, x: V, version: int) -> SolverHandleProtocol[V]:
        label = LinearizationPoint(version, self._identity)
        descriptor = self._space.descriptor()
        return self._solver.setup_keyed(
            ReuseKey(label, descriptor, descriptor), _Labeled(self._jacobian, x, label)
        )

    def solve(
        self, nonlinear: NonlinearSolverProtocol[V, A], initial_guess: V
    ) -> SteadyPoint[V, A]:
        """Solve from ``initial_guess`` with ``nonlinear``. Never raises on
        failure: check :meth:`SteadyPoint.converged`."""
        version = self.residual_version()
        state, info = nonlinear.solve(self, initial_guess)
        return SteadyPoint(self, state, version, info)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._residual!r})"


class _Labeled(Generic[V, F]):
    """Builds :math:`R_u` at a state under a fixed label, on demand."""

    def __init__(
        self, jacobian: ResidualJacobian[V, F], state: V, label: LinearizationPoint
    ) -> None:
        self._jacobian, self._state, self._label = jacobian, state, label

    def __call__(self) -> JacobianSnapshotProtocol[V, F]:
        return self._jacobian(self._state, self._label)
