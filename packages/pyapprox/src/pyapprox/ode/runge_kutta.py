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

from functools import partial
from itertools import count
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.pinned_rows import BlockPinProtocol, RowPinningProtocol
from pyapprox.ode.step_quadrature import TableauQuadrature
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import (
    StepOperator,
    StepperDerivatives,
    StepWindow,
    check_step_span,
)
from pyapprox.ode.tableau import ButcherTableau
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    BlockInnerProductSpace,
    BlockLowerTriangularHandle,
    BlockOperator,
    BlockVector,
    FormatProtocol,
    IdentityOperator,
    InnerProductSpaceProtocol,
    KeepWithinSolve,
    KeyedSolverProtocol,
    LinearizationPoint,
    LinearizedOperator,
    LinearMapProtocol,
    ReuseKey,
    ReusePolicyProtocol,
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


def _unversioned() -> int:
    """The version of a residual without parameters: one for its life."""
    return 0


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
    block : int
        The stage's residual block.
    pin : BlockPinProtocol, optional
        How the stage's rows are pinned, at its time; None for none.
    solver : KeyedSolverProtocol
        The session's solver, which sets up the stage operator.
    """

    def __init__(
        self,
        stepper: "RungeKuttaStepper[V, A, F]",
        state: V,
        known: V,
        time: float,
        scale: float,
        deltat: float,
        block: int,
        pin: Optional[BlockPinProtocol[V]],
        solver: KeyedSolverProtocol[V, F],
    ) -> None:
        self._stepper = stepper
        self._state, self._known = state, known
        self._time, self._scale, self._deltat = time, scale, deltat
        self._block, self._pin, self._solver = block, pin, solver

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._stepper.system().space

    def residual(self, x: V) -> V:
        space, mass = self.space(), self._stepper.system().mass
        # M (Y - y_n), then subtract dt a_ii f(Y) and dt k.
        value = mass.apply(space.axpby(1.0, x, -1.0, self._state))
        value = space.axpby(
            -self._scale, self._stepper.rhs(x, self._time), 1.0, value, out=value
        )
        value = space.axpby(-self._deltat, self._known, 1.0, value, out=value)
        if self._pin is None:
            return value
        return self._pin.residual(value, x, self._time, space)

    def jacobian_handle(self, x: V) -> SolverHandleProtocol[V]:
        return self._stepper.stage_handle(
            x, self._time, self._scale, self._block, self._pin, self._solver
        )

    def is_affine(self) -> bool:
        """Whether the stage equation is affine in the stage value: the
        system's :math:`f` is, and pinned rows (identity rows with data)
        keep it so."""
        return self._stepper.system().is_affine()


class _RungeKuttaSession(Generic[V, A, F]):
    """One solve's steps of a :class:`RungeKuttaStepper`.

    The implicit stages are set up through one keyed solver, made by the
    stepper's reuse policy for the session only: under the default policy
    it keeps a handle per distinct stage operator of a step, so a stage
    operator the system labels for reuse (see
    :meth:`MassFormSystem.operator_label`) is formed and set up once per
    label, and a Jacobian constant in time at a fixed step size is
    factored once per solve.
    """

    def __init__(self, stepper: "RungeKuttaStepper[V, A, F]") -> None:
        self._stepper = stepper
        self._solver: KeyedSolverProtocol[V, F] = stepper.reuse().solver(
            stepper.system().solver, stepper.distinct_stage_operators()
            + stepper._distinct_mass_blocks()
        )

    def step(self, past_states: Sequence[V], span: StepSpan) -> BlockVector[V]:
        """Solve one step (see :meth:`RungeKuttaStepper.step`)."""
        return self._stepper._step(past_states, span, self._solver)

    def diag_handle(
        self, window: StepWindow[V]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """The step's diagonal handle (see
        :meth:`RungeKuttaStepper.diag_handle`), its blocks set up through
        this session."""
        return self._stepper._diag_handle(window, self._solver)


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
    row_pinning : RowPinningProtocol, optional
        Which residual blocks have rows pinned to boundary data inside
        their solves, and how (see :mod:`pyapprox.ode.pinned_rows`); the
        blocks are the unknown stages in order, then the final block when
        there is one, each pinned at its own time. None for no pinning.
    reuse : ReusePolicyProtocol, optional
        How a solve keeps stage setups between steps: by default a handle
        per distinct stage operator of a step (:class:`KeepWithinSolve`);
        :class:`NoReuse` keeps none, to save memory.

    Raises
    ------
    TypeError
        If the tableau has an implicit stage and no nonlinear solver is
        given, or ``row_pinning`` or ``reuse`` does not satisfy its
        protocol.
    """

    def __init__(
        self,
        tableau: ButcherTableau,
        system: MassFormSystem[V, A, F],
        nonlinear_solver: Optional[NonlinearSolverProtocol[V, A]] = None,
        row_pinning: Optional[RowPinningProtocol[V]] = None,
        reuse: Optional[ReusePolicyProtocol] = None,
    ) -> None:
        if not tableau.is_explicit() and nonlinear_solver is None:
            raise TypeError(
                f"{tableau.name} has implicit stages, so it needs a nonlinear solver"
            )
        if row_pinning is not None and not isinstance(row_pinning, RowPinningProtocol):
            raise TypeError(
                "row_pinning must satisfy RowPinningProtocol, got "
                f"{type(row_pinning).__name__}"
            )
        if reuse is not None and not isinstance(reuse, ReusePolicyProtocol):
            raise TypeError(
                f"reuse must satisfy ReusePolicyProtocol, got {type(reuse).__name__}"
            )
        self._reuse: ReusePolicyProtocol = KeepWithinSolve() if reuse is None else reuse
        # Whether the residual reports a version is fixed for its life, so
        # decided once rather than by a protocol check on every call.
        rhs = system.rhs
        self._version: Callable[[], int] = (
            rhs.parameters_version
            if isinstance(rhs, _VersionedProtocol)
            else _unversioned
        )
        self._tableau = tableau
        self._quadrature = TableauQuadrature(tableau)
        self._system = system
        self._nonlinear_solver = nonlinear_solver
        nstages = tableau.nstages()
        first = 1 if tableau.has_trivial_first_stage() else 0
        # The stages that are unknowns, in order; a trivial first stage is
        # y_n itself.
        self._unknown_stages = list(range(first, nstages))
        self._final_block = not tableau.is_stiffly_accurate()
        # Whether the forward step reads stage i's slope: a later stage
        # through a_ji, or the final block through b_i.
        self._slope_read = [
            any(tableau.a[j][i] != 0.0 for j in range(i + 1, nstages))
            or (self._final_block and tableau.b[i] != 0.0)
            for i in range(nstages)
        ]
        nblocks = len(self._unknown_stages) + (1 if self._final_block else 0)
        space = system.space
        self._unknown_space = BlockInnerProductSpace([space] * nblocks, "unknown")
        self._residual_space = BlockInnerProductSpace([space] * nblocks, "residual")
        self._state_space = BlockInnerProductSpace([space], "state")
        self._mass_handle: Optional[SolverHandleProtocol[V]] = None
        self._row_pinning = row_pinning
        # Each pinned block's row-replaced mass, for its explicit solve,
        # set up once.
        self._pinned_mass_handles: Dict[int, SolverHandleProtocol[V]] = {}
        self._points = count()
        # The stage operators a step forms: one per diagonal coefficient,
        # and per pinned block, since its rows are replaced.
        offset = self._first_stage_offset()
        self._distinct = len(
            {
                (
                    tableau.a[i][i],
                    None if self._block_pin(i - offset) is None else i - offset,
                )
                for i in self._unknown_stages
                if not tableau.is_explicit_stage(i)
            }
        )

    def tableau(self) -> ButcherTableau:
        """The method."""
        return self._tableau

    def reuse(self) -> ReusePolicyProtocol:
        """How a solve keeps stage setups between steps."""
        return self._reuse

    def distinct_stage_operators(self) -> int:
        r"""The number of different stage operators :math:`M - \Delta t\,
        a_{ii} J` one step forms at a fixed step size: one per diagonal
        coefficient, and one per pinned block, whose rows are replaced.
        A session keeps this many handles by default, and one per
        :meth:`_distinct_mass_blocks` more for the sweeps."""
        return self._distinct

    def _distinct_mass_blocks(self) -> int:
        """The diagonal blocks of :meth:`diag` that are :math:`M` alone (an
        explicit stage, the final block), counted once per pinning: a
        sweep's session keeps a handle for each."""
        stages = len(self._unknown_stages)
        kinds = set()
        for r in range(self._unknown_space.nblocks()):
            if r < stages and not self._tableau.is_explicit_stage(
                self._unknown_stages[r]
            ):
                continue
            kinds.add(None if self._block_pin(r) is None else r)
        return len(kinds)

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

    def parameters_version(self) -> int:
        """The residual's parameter version; 0 for a residual without
        parameters, which has one version for its whole life."""
        return self._version()

    def time_quadrature(self) -> TableauQuadrature:
        r"""The tableau applied to :math:`\dot z = q(y, t)`, step by step
        (see :mod:`pyapprox.ode.step_quadrature`), its nodes located as
        :meth:`stage_values` reads the stages."""
        return self._quadrature

    def _point(self) -> LinearizationPoint:
        """A new label for a linearization at a Newton iterate: never
        reused, since an iterate is not a converged point."""
        return LinearizationPoint(
            self.parameters_version(), ("stage", next(self._points)), provisional=True
        )

    def _stage_operator(
        self,
        state: V,
        time: float,
        scale: float,
        pin: Optional[BlockPinProtocol[V]],
        point: LinearizationPoint,
    ) -> ActionOperatorProtocol[V, F]:
        r""":math:`M - s\, J(y, t)`, :math:`J` labeled ``point``, with the
        rows ``pin`` pins replaced."""
        system = self._system
        jacobian = system.jacobian(system.form.to_residual(state), time, point)
        operator = ScaledSum([(1.0, system.mass), (-scale, jacobian)], system.space)
        return operator if pin is None else pin.own(operator)

    def _labeled_stage_operator(
        self,
        state: V,
        time: float,
        scale: float,
        pin: Optional[BlockPinProtocol[V]],
        label: LinearizationPoint,
    ) -> LinearizedOperator[V, F]:
        """The stage operator with :math:`J` labeled ``label``, wrapped
        under ``label`` so a caching solver keys it."""
        return LinearizedOperator(
            self._stage_operator(state, time, scale, pin, label), label
        )

    def stage_handle(
        self,
        state: V,
        time: float,
        scale: float,
        block: int,
        pin: Optional[BlockPinProtocol[V]],
        solver: KeyedSolverProtocol[V, F],
    ) -> SolverHandleProtocol[V]:
        r"""A handle for the stage operator :math:`M - s\, J(y, t)` with
        :math:`s = \Delta t\, a_{ii}`, the rows of residual block ``block``
        replaced as ``pin`` says.

        Set up by ``solver`` under the system's reuse label when it gives
        one: equal stage operators of a session then share one setup, and
        one whose handle is kept is not formed again. A :math:`J` that sees
        the state gives no label, and each Newton iterate is set up afresh.

        Parameters
        ----------
        state : V
            A single column of the state space.
        time : float
            The stage time.
        scale : float
            :math:`s`.
        block : int
            The stage's residual block.
        pin : BlockPinProtocol, optional
            How the block's rows are pinned; None for none.
        solver : KeyedSolverProtocol
            The session's solver.
        """
        system = self._system
        key = ("stage", scale, None if pin is None else block)
        label = system.operator_label(self.parameters_version(), time, key)
        if label is None:
            return solver.setup(
                self._stage_operator(state, time, scale, pin, self._point())
            )
        descriptor = system.space.descriptor()
        return solver.setup_keyed(
            ReuseKey(label, descriptor, descriptor),
            partial(self._labeled_stage_operator, state, time, scale, pin, label),
        )

    def _mass_solve(self, rhs: V) -> V:
        """:math:`M^{-1}` ``rhs``, with the mass set up once."""
        if self._mass_handle is None:
            self._mass_handle = self._system.solver.setup(self._system.mass)
        solution, info = self._mass_handle.solve(rhs)
        if not info.converged:
            raise SolveFailedError(f"{self._tableau.name}: mass solve: {info.message}")
        return solution

    def _block_pin(self, block: int) -> Optional[BlockPinProtocol[V]]:
        """How residual block ``block`` is pinned, or None."""
        if self._row_pinning is None:
            return None
        return self._row_pinning.at(block, self._unknown_space.nblocks())

    def _explicit_solve(self, block: int, state: V, increment: V, time: float) -> V:
        r"""Block ``block``'s value :math:`Y` from :math:`M (Y - y_n) = `
        ``increment``. Pinned at ``time``, it solves :math:`(P M + I - P)\,
        Y = P (M y_n + \mathrm{increment}) + (I - P)\, g(t)` instead: one
        solve either way."""
        space, pin = self._system.space, self._block_pin(block)
        if pin is None:
            return space.axpby(1.0, state, 1.0, self._mass_solve(increment))
        handle = self._pinned_mass_handles.get(block)
        if handle is None:
            handle = self._system.solver.setup(pin.own(self._system.mass))
            self._pinned_mass_handles[block] = handle
        rhs = space.axpby(1.0, self._system.mass.apply(state), 1.0, increment)
        solution, info = handle.solve(pin.right_hand_side(rhs, time))
        if not info.converged:
            raise SolveFailedError(
                f"{self._tableau.name}: pinned mass solve of block {block}: "
                f"{info.message}"
            )
        return solution

    # ------------------------------------------------------------------
    # The forward step
    # ------------------------------------------------------------------

    def session(self) -> _RungeKuttaSession[V, A, F]:
        """A new session for one solve's steps, sharing stage setups
        between them and with no other session."""
        return _RungeKuttaSession(self)

    def step(self, past_states: Sequence[V], span: StepSpan) -> BlockVector[V]:
        r"""Solve step :math:`n` from :math:`y_n`, as a session of this one
        step (its stages may share a setup; nothing is kept after it).

        Parameters
        ----------
        past_states : sequence of V
            :math:`(y_n,)`: a one-step method reads one state, a single
            column of the state space.
        span : StepSpan
            The step from :math:`t_n` of size :math:`h_n`, one size.

        Returns
        -------
        BlockVector
            The converged unknown :math:`u_n`; :meth:`readout` (or its last
            block) gives :math:`y_{n+1}`.

        Raises
        ------
        ValueError
            If ``past_states`` is not one state, or ``span`` does not have
            one size.
        SolveFailedError
            If a stage's nonlinear solve or a mass solve fails.
        """
        return self.session().step(past_states, span)

    def _step(
        self,
        past_states: Sequence[V],
        span: StepSpan,
        solver: KeyedSolverProtocol[V, F],
    ) -> BlockVector[V]:
        """:meth:`step`, its implicit stages set up by ``solver``."""
        if len(past_states) != 1:
            raise ValueError(
                f"a one-step method reads one state, got {len(past_states)}"
            )
        check_step_span(span, 1)
        new_time, time, deltat = span.end(), span.time(), span.deltat()
        state = past_states[0]
        tableau, space = self._tableau, self._system.space
        offset = self._first_stage_offset()
        stages: List[V] = []
        slopes: List[V] = []
        for i in range(tableau.nstages()):
            stage_time = time + tableau.c[i] * deltat
            # k = sum_{j<i} a_ij f(Y_j), the known stages' contribution.
            known = space.zeros()
            for j in range(i):
                if tableau.a[i][j] != 0.0:
                    known = space.axpby(
                        tableau.a[i][j], slopes[j], 1.0, known, out=known
                    )
            if i == 0 and tableau.has_trivial_first_stage():
                stage = space.copy(state)
            elif tableau.is_explicit_stage(i):
                # M (Y - y_n) = dt k; stage i is residual block i - offset.
                stage = self._explicit_solve(
                    i - offset, state, space.scale(deltat, known), stage_time
                )
            else:
                stage = self._implicit_stage(
                    i, state, known, stage_time, deltat, solver
                )
            stages.append(stage)
            # f(Y_i) only when a later stage or the final block reads it;
            # a stiffly accurate method's last slope is never used.
            slopes.append(
                self.rhs(stage, stage_time) if self._slope_read[i] else space.zeros()
            )
        blocks = [stages[i] for i in self._unknown_stages]
        if self._final_block:
            # M (y_{n+1} - y_n) = dt sum_i b_i f(Y_i).
            weighted = space.zeros()
            for weight, slope in zip(tableau.b, slopes):
                if weight != 0.0:
                    weighted = space.axpby(weight, slope, 1.0, weighted, out=weighted)
            blocks.append(
                self._explicit_solve(
                    len(blocks), state, space.scale(deltat, weighted), new_time
                )
            )
        return BlockVector(tuple(blocks))

    def _implicit_stage(
        self,
        i: int,
        state: V,
        known: V,
        stage_time: float,
        deltat: float,
        linear_solver: KeyedSolverProtocol[V, F],
    ) -> V:
        """Solve implicit stage ``i`` from the guess :math:`y_n`, its rows
        pinned in its Newton system when its block is, its stage operator
        set up by ``linear_solver``."""
        solver = self._nonlinear_solver
        if solver is None:
            raise TypeError(f"{self._tableau.name} needs a nonlinear solver")
        scale = deltat * self._tableau.a[i][i]
        block = i - self._first_stage_offset()
        system = _StageSystem(
            self,
            state,
            known,
            stage_time,
            scale,
            deltat,
            block,
            self._block_pin(block),
            linear_solver,
        )
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
            self.rhs(stage, window.time() + tableau.c[j] * window.deltat())
            for j, stage in enumerate(stages)
        ]

        def block(value: V, weights: tuple[float, ...]) -> V:
            # M (value - y_n) - dt sum_j weights_j f(Y_j)
            result = mass.apply(space.axpby(1.0, value, -1.0, state))
            for weight, slope in zip(weights, slopes):
                if weight != 0.0:
                    result = space.axpby(
                        -window.deltat() * weight, slope, 1.0, result, out=result
                    )
            return result

        blocks = [block(stages[i], tableau.a[i]) for i in self._unknown_stages]
        if self._final_block:
            blocks.append(block(window.unknown.blocks[-1], tableau.b))
        for r in range(len(blocks)):
            pin = self._block_pin(r)
            if pin is not None:
                # A pinned block's rows read Y_R - g_R at its time.
                blocks[r] = pin.residual(
                    blocks[r],
                    window.unknown.blocks[r],
                    self._block_time(window, r),
                    space,
                )
        return BlockVector(tuple(blocks))

    def _block_time(self, window: StepWindow[V], block: int) -> float:
        """Residual block ``block``'s time: its stage's, or
        :math:`t_{n+1}` for the final block."""
        if block < len(self._unknown_stages):
            return self._stage_time(window, self._unknown_stages[block])
        return window.lag_time(0)

    # ------------------------------------------------------------------
    # The step's derivatives
    #
    # Row r of R_n is a stage row (stage i = unknown_stages[r], weights
    # a[i]) or the final row (weights b); every row is
    # M (value - y_n) - dt sum_j w_j f(Y_j). Column k of the unknown is
    # stage unknown_stages[k], or y_{n+1} for the final block. Stage j of
    # the tableau is the unknown's column j - first when it is an
    # unknown; a trivial first stage is y_n, the lag-1 argument.
    # ------------------------------------------------------------------

    def _rows(self) -> List[tuple[float, ...]]:
        """Each residual row's weights on the stages' f values."""
        rows = [self._tableau.a[i] for i in self._unknown_stages]
        if self._final_block:
            rows.append(self._tableau.b)
        return rows

    def _stage_time(self, window: StepWindow[V], j: int) -> float:
        return window.time() + self._tableau.c[j] * window.deltat()

    def _converged_point(self, window: StepWindow[V]) -> LinearizationPoint:
        """A label for a linearization at a converged step."""
        return LinearizationPoint(
            window.parameters_version, ("window", next(self._points))
        )

    def _jacobian_at(
        self, window: StepWindow[V], stage: V, j: int
    ) -> ActionOperatorProtocol[V, F]:
        """:math:`J(Y_j, t_j)`."""
        system = self._system
        return system.jacobian(
            system.form.to_residual(stage),
            self._stage_time(window, j),
            self._converged_point(window),
        )

    def _first_stage_offset(self) -> int:
        """The tableau index of the unknown's first stage block."""
        return 1 if self._tableau.has_trivial_first_stage() else 0

    def _diag_blocks(
        self, window: StepWindow[V]
    ) -> List[List[Optional[ActionOperatorProtocol[V, F]]]]:
        """The grid of :math:`\\partial R_n/\\partial u_n`."""
        system, deltat = self._system, window.deltat()
        stages = self.stage_values(window)
        offset = self._first_stage_offset()
        nblocks = self._unknown_space.nblocks()
        jacobians = {
            k: self._jacobian_at(window, stages[k + offset], k + offset)
            for k in range(len(self._unknown_stages))
        }
        grid: List[List[Optional[ActionOperatorProtocol[V, F]]]] = []
        for r, row_weights in enumerate(self._rows()):
            row: List[Optional[ActionOperatorProtocol[V, F]]] = [None] * nblocks
            for k, jacobian in jacobians.items():
                # Block (r, k) = delta_rk M - dt w_k J_k: row r holds
                # M (its own unknown - y_n), so only its own column k = r
                # carries M; w_k is row r's coefficient on stage k's f.
                weight = row_weights[k + offset]
                own_block = k == r
                if weight == 0.0 and not own_block:
                    continue
                terms: List[tuple[float, ActionOperatorProtocol[V, F]]] = []
                if own_block:
                    terms.append((1.0, system.mass))
                if weight != 0.0:
                    terms.append((-deltat * weight, jacobian))
                row[k] = ScaledSum(terms, system.space)
            if self._final_block and r == nblocks - 1:
                # The final block's own column: M.
                row[-1] = system.mass
            grid.append(row)
        return self._pin_rows(grid)

    def _pin_rows(
        self, grid: List[List[Optional[ActionOperatorProtocol[V, F]]]]
    ) -> List[List[Optional[ActionOperatorProtocol[V, F]]]]:
        """``diag``'s grid with each pinned block row edited: its own block
        row-replaced, its other blocks with the rows zeroed."""
        for r in range(len(grid)):
            pin = self._block_pin(r)
            if pin is None:
                continue
            grid[r] = [
                None
                if block is None
                else pin.own(block)
                if k == r
                else pin.coupled(block)
                for k, block in enumerate(grid[r])
            ]
        return grid

    def diag(self, window: StepWindow[V]) -> StepOperator[V]:
        r""":math:`\partial R_n/\partial u_n`, block lower triangular: the
        stage rows' :math:`\delta_{ik} M - \Delta t\, a_{ik} J_k`, the final
        row's :math:`-\Delta t\, b_k J_k` and :math:`M`.

        Parameters
        ----------
        window : StepWindow
            The step's converged unknown, past state, time and size.
        """
        return BlockOperator(
            self._diag_blocks(window), self._unknown_space, self._residual_space
        )

    def diag_handle(
        self, window: StepWindow[V]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """A solver handle for :meth:`diag`, as a session of this one
        window: forward substitution with each diagonal block set up by
        the system's solver."""
        return self.session().diag_handle(window)

    def _diag_label(
        self, window: StepWindow[V], r: int
    ) -> Optional[LinearizationPoint]:
        """The reuse label of diagonal block ``r``, or None: a stage block
        :math:`M - \\Delta t\\, a_{rr} J` is labeled exactly as the forward
        step labels it (so a sweep's blocks repeat as the steps' do); a
        block that is :math:`M` alone (an explicit stage, the final block)
        by the mass, fixed for the system's life."""
        pinned = None if self._block_pin(r) is None else r
        version = window.parameters_version
        if r < len(self._unknown_stages):
            i = self._unknown_stages[r]
            diagonal = self._tableau.a[i][i]
            if diagonal != 0.0:
                key = ("stage", window.deltat() * diagonal, pinned)
                time = self._stage_time(window, i)
                return self._system.operator_label(version, time, key)
        return LinearizationPoint(version, ("mass", pinned))

    def _diag_handle(
        self, window: StepWindow[V], solver: KeyedSolverProtocol[V, F]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """:meth:`diag_handle`, each diagonal block set up by ``solver``
        under its reuse label when it has one."""
        grid = self._diag_blocks(window)
        handles = []
        for r, row in enumerate(grid):
            block = row[r]
            if block is None:
                raise ValueError(f"{self._tableau.name}: diagonal block {r} is zero")
            label = self._diag_label(window, r)
            labeled = block if label is None else LinearizedOperator(block, label)
            handles.append(solver.setup(labeled))
        operator = BlockOperator(grid, self._unknown_space, self._residual_space)
        return BlockLowerTriangularHandle(operator, handles)

    def _off_diag(self, window: StepWindow[V], lag: int) -> StepOperator[V]:
        r""":math:`\partial R_n/\partial y_n`: :math:`-M` in every row, and
        :math:`-\Delta t\, w_1 J(y_n)` from a trivial first stage.

        Parameters
        ----------
        window : StepWindow
            The step.
        lag : int
            Must be 1: a one-step method reads only :math:`y_n`.
        """
        if lag != 1:
            raise ValueError(f"a one-step method has lag 1 only, got {lag}")
        system, deltat = self._system, window.deltat()
        trivial = self._tableau.has_trivial_first_stage()
        jacobian = (
            self._jacobian_at(window, window.past_state(1), 0) if trivial else None
        )
        grid: List[List[Optional[LinearMapProtocol[V]]]] = []
        for r, weights in enumerate(self._rows()):
            terms: List[tuple[float, ActionOperatorProtocol[V, F]]] = [
                (-1.0, system.mass)
            ]
            if jacobian is not None and weights[0] != 0.0:
                terms.append((-deltat * weights[0], jacobian))
            block = ScaledSum(terms, system.space)
            pin = self._block_pin(r)
            # A pinned block's rows do not read y_n.
            grid.append([block if pin is None else pin.coupled(block)])
        return BlockOperator(grid, self._state_space, self._residual_space)

    def _param_block_space(self) -> BlockInnerProductSpace[V, A]:
        """The parameters as a one-block space, from the residual's bundle.

        Raises
        ------
        TypeError
            If the bundle's parameter space has no inner product.
        """
        space = self._system.rhs.derivatives().param_space
        if not isinstance(space, InnerProductSpaceProtocol):
            raise TypeError(
                "the residual's parameter space must be an inner-product space, "
                f"got {type(space).__name__}"
            )
        return BlockInnerProductSpace([space], "param")

    def param_space(self) -> Optional[BlockInnerProductSpace[V, A]]:
        """The parameters as one-block vectors; ``None`` when the residual
        has no parameter derivatives."""
        if self._system.rhs.derivatives().param is None:
            return None
        return self._param_block_space()

    def _param(self, window: StepWindow[V]) -> StepOperator[V]:
        r""":math:`\partial R_n/\partial p`: row weights :math:`w`, each row
        :math:`-\Delta t \sum_j w_j f_p(Y_j)` from the residual's ``param``.

        Parameters
        ----------
        window : StepWindow
            The step.
        """
        bundle = self._system.rhs.derivatives()
        if bundle.param is None:
            raise ValueError("the residual has no parameter derivatives")
        form, space, deltat = self._system.form, self._system.space, window.deltat()
        stages = self.stage_values(window)
        param_ops = [
            bundle.param(form.to_residual(stage), self._stage_time(window, j))
            for j, stage in enumerate(stages)
        ]
        params = self._param_block_space()
        grid: List[List[Optional[LinearMapProtocol[V]]]] = []
        for r, weights in enumerate(self._rows()):
            terms = [
                (-deltat * weight, op)
                for weight, op in zip(weights, param_ops)
                if weight != 0.0
            ]
            block = ScaledSum(terms, space, params.component(0))
            pin = self._block_pin(r)
            # A pinned block's rows do not depend on the parameters.
            grid.append([block if pin is None else pin.coupled(block)])
        return BlockOperator(grid, params, self._residual_space)

    def _stage_adjoints(self, window: StepWindow[V], adj: BlockVector[V]) -> List[V]:
        r"""The stage adjoints :math:`\lambda_j = -\Delta t \sum_r w_{rj}\,
        \mu_r`: :math:`\mu^\top R_n` depends on stage :math:`j` only through
        :math:`\lambda_j^\top f(Y_j)`.

        Parameters
        ----------
        window : StepWindow
            The step.
        adj : BlockVector
            :math:`\mu`, a single column of the residual space.
        """
        space, deltat = self._system.space, window.deltat()
        rows = self._rows()
        # A pinned block's rows carry no f, so no curvature.
        blocks = []
        for r, mu in enumerate(adj.blocks):
            pin = self._block_pin(r)
            blocks.append(mu if pin is None else pin.adjoint(mu, space))
        adjoints: List[V] = []
        for j in range(self._tableau.nstages()):
            total = space.zeros()
            for weights, mu in zip(rows, blocks):
                if weights[j] != 0.0:
                    total = space.axpby(-deltat * weights[j], mu, 1.0, total, out=total)
            adjoints.append(total)
        return adjoints

    def _stage_terms(
        self, window: StepWindow[V], adj: BlockVector[V]
    ) -> List[tuple[V, float, V]]:
        """For each stage, its value, time and adjoint in the residual's
        form, ready for the residual bundle's second derivatives."""
        to_residual = self._system.form.to_residual
        return [
            (to_residual(stage), self._stage_time(window, j), to_residual(adjoint))
            for j, (stage, adjoint) in enumerate(
                zip(self.stage_values(window), self._stage_adjoints(window, adj))
            )
        ]

    def _state_state(
        self, window: StepWindow[V], adj: BlockVector[V], a: int, b: int
    ) -> StepOperator[V]:
        r"""The :math:`(a, b)` lag block of :math:`\mu^\top \partial^2 R_n`:
        for :math:`(0, 0)` block diagonal over the unknown stages,
        :math:`\lambda_k^\top f_{yy}(Y_k)`; for :math:`(1, 1)` the trivial
        first stage's :math:`\lambda_1^\top f_{yy}(y_n)`.

        Parameters
        ----------
        window : StepWindow
            The step.
        adj : BlockVector
            :math:`\mu`.
        a, b : int
            The lags; only declared pairs are asked for.
        """
        field = self._system.rhs.derivatives().state_state
        if field is None:
            raise ValueError("the residual has no second derivatives")
        terms = self._stage_terms(window, adj)
        offset = self._first_stage_offset()
        if (a, b) == (0, 0):
            nblocks = self._unknown_space.nblocks()
            grid: List[List[Optional[LinearMapProtocol[V]]]] = [
                [None] * nblocks for _ in range(nblocks)
            ]
            for k in range(len(self._unknown_stages)):
                grid[k][k] = field(*terms[k + offset])
            return BlockOperator(grid, self._unknown_space, self._unknown_space)
        if (a, b) == (1, 1) and offset == 1:
            return BlockOperator(
                [[field(*terms[0])]], self._state_space, self._state_space
            )
        raise ValueError(f"the ({a}, {b}) block is not declared nonzero")

    def _state_param(
        self, window: StepWindow[V], adj: BlockVector[V], a: int
    ) -> StepOperator[V]:
        r"""The parameters-to-lag-:math:`a` block of
        :math:`\mu^\top \partial^2 R_n`: :math:`\lambda_j^\top f_{yp}(Y_j)`
        for the unknown stages (lag 0) or the trivial first stage (lag 1).

        Parameters
        ----------
        window : StepWindow
            The step.
        adj : BlockVector
            :math:`\mu`.
        a : int
            The lag; only declared lags are asked for.
        """
        field = self._system.rhs.derivatives().state_param
        if field is None:
            raise ValueError("the residual has no second derivatives")
        terms = self._stage_terms(window, adj)
        offset = self._first_stage_offset()
        params = self._param_block_space()
        if a == 0:
            grid: List[List[Optional[LinearMapProtocol[V]]]] = [
                [None] for _ in range(self._unknown_space.nblocks())
            ]
            for k in range(len(self._unknown_stages)):
                grid[k][0] = field(*terms[k + offset])
            return BlockOperator(grid, params, self._unknown_space)
        if a == 1 and offset == 1:
            return BlockOperator([[field(*terms[0])]], params, self._state_space)
        raise ValueError(f"lag {a} is not declared nonzero")

    def _param_param(
        self, window: StepWindow[V], adj: BlockVector[V]
    ) -> StepOperator[V]:
        r""":math:`\mu^\top \partial^2 R_n/\partial p^2 = \sum_j
        \lambda_j^\top f_{pp}(Y_j)`.

        Parameters
        ----------
        window : StepWindow
            The step.
        adj : BlockVector
            :math:`\mu`.
        """
        field = self._system.rhs.derivatives().param_param
        if field is None:
            raise ValueError("the residual has no second derivatives")
        params = self._param_block_space()
        total = ScaledSum(
            [(1.0, field(*term)) for term in self._stage_terms(window, adj)],
            params.component(0),
        )
        return BlockOperator([[total]], params, params)

    def derivatives(self) -> StepperDerivatives[V]:
        """The step's derivatives, as far as the residual's bundle goes:
        the coupling to :math:`y_n` always, the parameters with the
        bundle's ``param``, the second derivatives with its second order.
        Valid for the parameters set when fetched."""
        bundle = self._system.rhs.derivatives()
        if bundle.param is None:
            return StepperDerivatives.state_only(self._off_diag)
        if bundle.state_state is None:
            return StepperDerivatives.first_order(self._off_diag, self._param)
        state_pairs, param_lags = self._declared_second_order()
        return StepperDerivatives.second_order(
            self._off_diag,
            self._param,
            self._state_state,
            self._state_param,
            self._param_param,
            state_pairs,
            param_lags,
        )

    def _declared_second_order(
        self,
    ) -> tuple[frozenset[tuple[int, int]], frozenset[int]]:
        r"""Which second-derivative blocks may be nonzero, so the sweeps
        skip the rest.

        Lag 0 is the step's unknown (the stages), lag 1 is :math:`y_n`.
        :math:`\mu^\top R_n` depends on stage :math:`j` only through
        :math:`\lambda_j^\top f(Y_j)`, and each :math:`f(Y_j)` reads one
        argument only, so:

        - the stage-stage block :math:`(0, 0)` and the stages' mixed
          parameter block (lag 0) are always present;
        - a trivial first stage evaluates :math:`f` at :math:`y_n` itself,
          adding the :math:`(1, 1)` block and the mixed block of lag 1;
          otherwise :math:`y_n` enters only through :math:`-M y_n`, which
          is linear;
        - :math:`(0, 1)` and :math:`(1, 0)` are never present: no
          :math:`f(Y_j)` reads both a stage and :math:`y_n`.

        Returns
        -------
        tuple
            ``(state_pairs, param_lags)``.
        """
        if self._tableau.has_trivial_first_stage():
            return frozenset({(0, 0), (1, 1)}), frozenset({0, 1})
        return frozenset({(0, 0)}), frozenset({0})

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._tableau.name})"
