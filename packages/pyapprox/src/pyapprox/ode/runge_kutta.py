r"""One Runge-Kutta stepper for every lower-triangular tableau: explicit
and diagonally implicit methods alike.

For :math:`M \dot y = f(y, t; p)` and a tableau :math:`(A, b, c)`, step
:math:`n` solves for the stage values :math:`Y_i` in mass form,

.. math::

    M (Y_i - y_n) - \Delta t \sum_{j \le i} a_{ij}\, f(Y_j, t_n + c_j \Delta t) = 0,

one stage after another (:math:`A` is lower triangular), and, unless the
method is stiffly accurate (then :math:`y_{n+1} = Y_s`), the final block

.. math::

    M (y_{n+1} - y_n) - \Delta t \sum_i b_i\, f(Y_i, t_n + c_i \Delta t) = 0.

The step's unknown :math:`u_n` is the block vector of the stage values
that are unknowns (a trivial first stage, a zero row of :math:`A`, is
:math:`y_n` itself) followed by :math:`y_{n+1}` when there is a final
block; its residual :math:`R_n` has one block per equation above. An
implicit stage is solved by the injected nonlinear solver with the stage
operator :math:`M - \Delta t\, a_{ii} J(Y_i)`; an explicit stage and the
final block are solved with :math:`M`. No mass is ever inverted inside
an implicit stage, so a singular mass is fine there.

The stepper reads only the tableau's coefficients: a new method is a new
tableau. States are stack vectors, converted to the residual's form only
where :math:`f` and :math:`J` are called.
"""

from itertools import count
from typing import Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.stepper_derivatives import StepOperator, StepWindow
from pyapprox.ode.tableau import ButcherTableau
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    BlockInnerProductSpace,
    BlockOperator,
    BlockVector,
    FormatProtocol,
    IdentityOperator,
    InnerProductSpaceProtocol,
    LinearizationPoint,
    LinearMapProtocol,
    ScaledSum,
    SolveFailedError,
    SolverHandleProtocol,
)
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


@runtime_checkable
class _VersionedProtocol(Protocol):
    """A residual that reports its parameter version."""

    def parameters_version(self) -> int: ...


class _StageSystem(Generic[V, A, F]):
    r"""Stage :math:`i`'s equation for the nonlinear solver,

    .. math::

        G(Y) = M (Y - y_n) - \Delta t\, a_{ii} f(Y, t_i) - \Delta t\, k = 0,

    with :math:`k = \sum_{j<i} a_{ij} f(Y_j)` the known stages'
    contribution, and Jacobian :math:`M - \Delta t\, a_{ii} J(Y, t_i)`.

    Parameters
    ----------
    stepper : RungeKuttaStepper
        Supplies the system, :math:`f`, :math:`J` and the labels.
    state : V
        :math:`y_n`, a single column of the state space.
    known : V
        :math:`k`, a single column of the state space.
    time : float
        The stage time :math:`t_i`.
    scale : float
        :math:`\Delta t\, a_{ii}`.
    deltat : float
        The step size.
    """

    def __init__(
        self,
        stepper: "RungeKuttaStepper[V, A, F]",
        state: V,
        known: V,
        time: float,
        scale: float,
        deltat: float,
    ) -> None:
        self._stepper = stepper
        self._state, self._known = state, known
        self._time, self._scale, self._deltat = time, scale, deltat

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._stepper.system().space

    def residual(self, x: V) -> V:
        space, mass = self.space(), self._stepper.system().mass
        # M (Y - y_n), then subtract dt a_ii f(Y) and dt k.
        value = mass.apply(space.axpby(1.0, x, -1.0, self._state))
        value = space.axpby(
            -self._scale, self._stepper.rhs(x, self._time), 1.0, value, out=value
        )
        return space.axpby(-self._deltat, self._known, 1.0, value, out=value)

    def jacobian_handle(self, x: V) -> SolverHandleProtocol[V]:
        return self._stepper.stage_handle(x, self._time, self._scale)

    def is_linear(self) -> bool:
        return False


class RungeKuttaStepper(Generic[V, A, F]):
    r"""A Runge-Kutta stepper in mass form for any lower-triangular tableau.

    Parameters
    ----------
    tableau : ButcherTableau
        The method.
    system : MassFormSystem
        :math:`f`, :math:`M`, :math:`J`, the state space, the linear
        solver and the residual form.
    nonlinear_solver : NonlinearSolverProtocol, optional
        Solves the implicit stages; required exactly when the tableau has
        one.

    Raises
    ------
    TypeError
        If the tableau has an implicit stage and no nonlinear solver is
        given.
    """

    def __init__(
        self,
        tableau: ButcherTableau,
        system: MassFormSystem[V, A, F],
        nonlinear_solver: Optional[NonlinearSolverProtocol[V, A]] = None,
    ) -> None:
        if not tableau.is_explicit() and nonlinear_solver is None:
            raise TypeError(
                f"{tableau.name} has implicit stages, so it needs a nonlinear solver"
            )
        self._tableau = tableau
        self._system = system
        self._nonlinear_solver = nonlinear_solver
        nstages = tableau.nstages()
        first = 1 if tableau.has_trivial_first_stage() else 0
        # The stages that are unknowns, in order; a trivial first stage is
        # y_n itself.
        self._unknown_stages = list(range(first, nstages))
        self._final_block = not tableau.is_stiffly_accurate()
        nblocks = len(self._unknown_stages) + (1 if self._final_block else 0)
        space = system.space
        self._unknown_space = BlockInnerProductSpace([space] * nblocks, "unknown")
        self._residual_space = BlockInnerProductSpace([space] * nblocks, "residual")
        self._state_space = BlockInnerProductSpace([space], "state")
        self._mass_handle: Optional[SolverHandleProtocol[V]] = None
        self._points = count()

    def tableau(self) -> ButcherTableau:
        """The method."""
        return self._tableau

    def system(self) -> MassFormSystem[V, A, F]:
        """What the stepper steps."""
        return self._system

    def depth(self) -> int:
        """A one-step method reads one past state."""
        return 1

    def unknown_space(self) -> BlockInnerProductSpace[V, A]:
        """The space of :math:`u_n`: one block per unknown stage, then the
        new state when there is a final block."""
        return self._unknown_space

    def residual_space(self) -> BlockInnerProductSpace[V, A]:
        """The space of :math:`R_n`, block for block like the unknown."""
        return self._residual_space

    def state_space(self) -> BlockInnerProductSpace[V, A]:
        """The stored states, as one-block vectors."""
        return self._state_space

    def readout(self) -> StepOperator[V]:
        """The map from :math:`u_n` to :math:`y_{n+1}`: its last block,
        the last stage or the final block's state."""
        nblocks = self._unknown_space.nblocks()
        row: List[Optional[LinearMapProtocol[V]]] = [None] * nblocks
        row[-1] = IdentityOperator(self._system.space)
        return BlockOperator([row], self._unknown_space, self._state_space)

    # ------------------------------------------------------------------
    # f, J and the solves
    # ------------------------------------------------------------------

    def rhs(self, state: V, time: float) -> V:
        """:math:`f(y, t)` for a single column ``state`` of the state
        space, converted to and from the residual's form."""
        form = self._system.form
        return form.from_residual(self._system.rhs(form.to_residual(state), time))

    def _point(self) -> LinearizationPoint:
        """A new label for a linearization at a Newton iterate: never
        reused, since an iterate is not a converged point."""
        rhs = self._system.rhs
        version = rhs.parameters_version() if isinstance(rhs, _VersionedProtocol) else 0
        return LinearizationPoint(
            version, ("stage", next(self._points)), provisional=True
        )

    def stage_operator(
        self, state: V, time: float, scale: float
    ) -> ActionOperatorProtocol[V, F]:
        r""":math:`M - s\, J(y, t)` with :math:`s = \Delta t\, a_{ii}`.

        Parameters
        ----------
        state : V
            A single column of the state space.
        time : float
            The stage time.
        scale : float
            :math:`s`.
        """
        system = self._system
        jacobian = system.jacobian(system.form.to_residual(state), time, self._point())
        return ScaledSum([(1.0, system.mass), (-scale, jacobian)], system.space)

    def stage_handle(
        self, state: V, time: float, scale: float
    ) -> SolverHandleProtocol[V]:
        """A solver handle for :meth:`stage_operator`."""
        return self._system.solver.setup(self.stage_operator(state, time, scale))

    def _mass_solve(self, rhs: V) -> V:
        """:math:`M^{-1}` ``rhs``, with the mass set up once."""
        if self._mass_handle is None:
            self._mass_handle = self._system.solver.setup(self._system.mass)
        solution, info = self._mass_handle.solve(rhs)
        if not info.converged:
            raise SolveFailedError(f"{self._tableau.name}: mass solve: {info.message}")
        return solution

    # ------------------------------------------------------------------
    # The forward step
    # ------------------------------------------------------------------

    def step(self, state: V, time: float, deltat: float) -> BlockVector[V]:
        r"""Solve step :math:`n` from :math:`y_n` = ``state``.

        Parameters
        ----------
        state : V
            :math:`y_n`, a single column of the state space.
        time : float
            :math:`t_n`.
        deltat : float
            The step size.

        Returns
        -------
        BlockVector
            The converged unknown :math:`u_n`; :meth:`readout` (or its last
            block) gives :math:`y_{n+1}`.

        Raises
        ------
        SolveFailedError
            If a stage's nonlinear solve or a mass solve fails.
        """
        tableau, space = self._tableau, self._system.space
        stages: List[V] = []
        slopes: List[V] = []
        for i in range(tableau.nstages()):
            stage_time = time + tableau.c[i] * deltat
            # k = sum_{j<i} a_ij f(Y_j), the known stages' contribution.
            known = space.zeros()
            for j in range(i):
                known = space.axpby(tableau.a[i][j], slopes[j], 1.0, known, out=known)
            if i == 0 and tableau.has_trivial_first_stage():
                stage = space.copy(state)
            elif tableau.is_explicit_stage(i):
                # M (Y - y_n) = dt k.
                increment = self._mass_solve(space.scale(deltat, known))
                stage = space.axpby(1.0, state, 1.0, increment)
            else:
                stage = self._implicit_stage(i, state, known, stage_time, deltat)
            stages.append(stage)
            slopes.append(self.rhs(stage, stage_time))
        blocks = [stages[i] for i in self._unknown_stages]
        if self._final_block:
            # M (y_{n+1} - y_n) = dt sum_i b_i f(Y_i).
            weighted = space.zeros()
            for weight, slope in zip(tableau.b, slopes):
                weighted = space.axpby(weight, slope, 1.0, weighted, out=weighted)
            increment = self._mass_solve(space.scale(deltat, weighted))
            blocks.append(space.axpby(1.0, state, 1.0, increment))
        return BlockVector(tuple(blocks))

    def _implicit_stage(
        self, i: int, state: V, known: V, stage_time: float, deltat: float
    ) -> V:
        """Solve implicit stage ``i`` from the guess :math:`y_n`."""
        solver = self._nonlinear_solver
        if solver is None:
            raise TypeError(f"{self._tableau.name} needs a nonlinear solver")
        scale = deltat * self._tableau.a[i][i]
        system = _StageSystem(self, state, known, stage_time, scale, deltat)
        stage, info = solver.solve(system, state)
        if not info.converged:
            raise SolveFailedError(
                f"{self._tableau.name}: stage {i} at t = {stage_time}: {info.message}"
            )
        return stage

    # ------------------------------------------------------------------
    # The step residual
    # ------------------------------------------------------------------

    def stage_values(self, window: StepWindow[V]) -> List[V]:
        """Every stage value at ``window``: :math:`y_n` for a trivial
        first stage, otherwise the unknown's blocks."""
        stages: List[V] = []
        if self._tableau.has_trivial_first_stage():
            stages.append(window.past_state(1))
        stages += list(window.unknown.blocks[: len(self._unknown_stages)])
        return stages

    def step_residual(self, window: StepWindow[V]) -> BlockVector[V]:
        r""":math:`R_n` at the window's unknown and past state, at the
        current parameters (see the module docstring)."""
        tableau, space = self._tableau, self._system.space
        mass = self._system.mass
        state = window.past_state(1)
        stages = self.stage_values(window)
        slopes = [
            self.rhs(stage, window.time + tableau.c[j] * window.deltat)
            for j, stage in enumerate(stages)
        ]

        def block(value: V, weights: tuple[float, ...]) -> V:
            # M (value - y_n) - dt sum_j weights_j f(Y_j)
            result = mass.apply(space.axpby(1.0, value, -1.0, state))
            for weight, slope in zip(weights, slopes):
                if weight != 0.0:
                    result = space.axpby(
                        -window.deltat * weight, slope, 1.0, result, out=result
                    )
            return result

        blocks = [block(stages[i], tableau.a[i]) for i in self._unknown_stages]
        if self._final_block:
            blocks.append(block(window.unknown.blocks[-1], tableau.b))
        return BlockVector(tuple(blocks))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._tableau.name})"
