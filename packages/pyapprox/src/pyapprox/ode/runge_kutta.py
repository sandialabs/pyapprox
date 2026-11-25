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

from typing import (
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from pyapprox.ode.continuous_extension import (
    ContinuousExtensionDerivatives,
    fraction_of_step,
)
from pyapprox.ode.mass_form_system import MassFormSystem, parameters_version_reader
from pyapprox.ode.pinned_rows import (
    BlockPinProtocol,
    RowPinningProtocol,
    StepBlock,
    data_curvature_term,
    data_jacobian_term,
)
from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.ode.setups import (
    MassShiftBuilderProtocol,
    MassShiftRequest,
    MassShiftStrategy,
    StatePosition,
    stepper_setups,
)
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
    LinearizationPoint,
    LinearMapProtocol,
    Product,
    ScaledSum,
    SolveFailedError,
    SolverHandleProtocol,
)
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)
G = TypeVar("G", bound=FormatProtocol)


class _StageSystem(Generic[V, A, F]):
    r"""Stage :math:`i`'s equation for the nonlinear solver,

    .. math::

        G(Y) = M (Y - y_n) - \Delta t\, a_{ii} f(Y, t_i) - \Delta t\, k = 0,

    with :math:`k = \sum_{j<i} a_{ij} f(Y_j)` the known stages'
    contribution, and Jacobian :math:`M - \Delta t\, a_{ii} J(Y, t_i)`.

    Parameters
    ----------
    stepper : RungeKuttaStepper
        Supplies the system, :math:`f`, and the stage operator's handle.
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
        pin: Optional[BlockPinProtocol[V, F]],
    ) -> None:
        self._stepper = stepper
        self._state, self._known = state, known
        self._time, self._scale, self._deltat = time, scale, deltat
        self._block, self._pin = block, pin

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
        # An iterate, not a converged stage: it has no position.
        return self._stepper._shift_handle(self._scale, self._pin, self._time, x, None)

    def is_affine(self) -> bool:
        """Whether the stage equation is affine in the stage value: the
        system's :math:`f` is, and pinned rows (identity rows with data)
        keep it so."""
        return self._stepper.system().is_affine()


class _TableauExtension(Generic[V, A, F]):
    r"""A tableau stepper's continuous extension (see
    :meth:`RungeKuttaStepper.continuous_extension`).

    :math:`y(t)` at :math:`\theta = (t - t_n)/h` is one more row of the
    step equation, the final block with the weights :math:`b(\theta)`:
    :math:`E\, y(t) = P (M y_n + h \sum_j b_j(\theta) f(Y_j)) + (I - P)
    g(t)`, :math:`E = P M + I - P`, pinned as the block producing
    :math:`y_{n+1}` is, at ``time`` (:math:`P = I` unpinned). Its
    derivatives are the row's, solved with :math:`E`:
    :math:`\partial y(t) = E^{-1} P (M\, \partial y_n + h \sum_j
    b_j(\theta)\, (J_j\, \partial Y_j + f_{p,j}\, \partial p))`, and with
    :math:`x = P E^{-\top} \lambda` the stage weights :math:`\lambda_j = h
    b_j(\theta)\, x` contract the residual's second derivatives, as the
    step's stage adjoints do.
    """

    def __init__(self, stepper: "RungeKuttaStepper[V, A, F]") -> None:
        self._stepper = stepper

    def _weights(self, window: StepWindow[V], time: float) -> tuple[float, ...]:
        return self._stepper.tableau().weights_at(fraction_of_step(window, time))

    def _last(self) -> int:
        """The block producing :math:`y_{n+1}`, whose pin the extension
        takes."""
        return self._stepper.unknown_space().nblocks() - 1

    def unknown_space(self) -> BlockInnerProductSpace[V, A]:
        return self._stepper.unknown_space()

    def state_space(self) -> BlockInnerProductSpace[V, A]:
        return self._stepper.state_space()

    def param_space(self) -> Optional[BlockInnerProductSpace[V, A]]:
        return self._stepper.param_space()

    def value(self, window: StepWindow[V], time: float) -> V:
        """:math:`y(t)` for ``time`` in the window's span."""
        stepper = self._stepper
        space = stepper.system().space
        weighted = space.zeros()
        for j, (weight, stage) in enumerate(
            zip(self._weights(window, time), stepper.stage_values(window))
        ):
            if weight != 0.0:
                slope = stepper.rhs(stage, stepper._stage_time(window, j))
                weighted = space.axpby(weight, slope, 1.0, weighted, out=weighted)
        increment = space.scale(window.deltat(), weighted)
        return stepper._explicit_solve(
            self._last(), window.past_state(1), increment, time
        )

    def derivatives(self) -> ContinuousExtensionDerivatives[V]:
        """As far as the residual's bundle goes: the unknown and the past
        state always, the parameters with its ``param``, the curvatures
        with its second order."""
        bundle = self._stepper.system().rhs.derivatives()
        if bundle.param is None:
            return ContinuousExtensionDerivatives.first_order(
                self._wrt_unknown, self._wrt_lag, None
            )
        if bundle.state_state is None:
            return ContinuousExtensionDerivatives.first_order(
                self._wrt_unknown, self._wrt_lag, self._wrt_param
            )
        return ContinuousExtensionDerivatives.second_order(
            self._wrt_unknown,
            self._wrt_lag,
            self._wrt_param,
            self._curvature_wrt_unknown,
            self._curvature_wrt_lag,
            self._curvature_wrt_param,
        )

    def _handle(self, window: StepWindow[V], time: float) -> SolverHandleProtocol[V]:
        """:math:`E`'s handle, from the stepper's setup strategy: the
        request of the block producing :math:`y_{n+1}`."""
        return self._stepper._mass_block_handle(
            self._last(), time, window.past_state(1)
        )

    def _solved(
        self,
        op: Optional[ActionOperatorProtocol[V, G]],
        window: StepWindow[V],
        time: float,
    ) -> Optional[LinearMapProtocol[V]]:
        r""":math:`E^{-1} P\, \mathrm{op}`, or None for a zero ``op``."""
        if op is None:
            return None
        pin = self._stepper._block_pin(self._last())
        coupled = op if pin is None else pin.coupled(op)
        return Product([self._handle(window, time), coupled])

    def _wrt_unknown(self, window: StepWindow[V], time: float) -> StepOperator[V]:
        stepper = self._stepper
        system, deltat = stepper.system(), window.deltat()
        weights = self._weights(window, time)
        stages = stepper.stage_values(window)
        offset = stepper._first_stage_offset()
        unknown = stepper.unknown_space()
        row: List[Optional[LinearMapProtocol[V]]] = [None] * unknown.nblocks()
        for k in range(len(stepper._unknown_stages)):
            weight = weights[k + offset]
            if weight == 0.0:
                continue
            jacobian = stepper._jacobian_at(window, stages[k + offset], k + offset)
            row[k] = self._solved(
                ScaledSum([(deltat * weight, jacobian)], system.space), window, time
            )
        return BlockOperator([row], unknown, stepper.state_space())

    def _wrt_lag(self, window: StepWindow[V], time: float, lag: int) -> StepOperator[V]:
        if lag != 1:
            raise ValueError(f"a one-step method has lag 1 only, got {lag}")
        stepper = self._stepper
        system = stepper.system()
        terms: List[tuple[float, ActionOperatorProtocol[V, F]]] = [(1.0, system.mass)]
        weight = self._weights(window, time)[0]
        if stepper.tableau().has_trivial_first_stage() and weight != 0.0:
            jacobian = stepper._jacobian_at(window, window.past_state(1), 0)
            terms.append((window.deltat() * weight, jacobian))
        block = self._solved(ScaledSum(terms, system.space), window, time)
        states = stepper.state_space()
        return BlockOperator([[block]], states, states)

    def _wrt_param(self, window: StepWindow[V], time: float) -> StepOperator[V]:
        stepper = self._stepper
        bundle = stepper.system().rhs.derivatives()
        if bundle.param is None:
            raise ValueError("the residual has no parameter derivatives")
        form, space = stepper.system().form, stepper.system().space
        params = stepper._param_block_space()
        terms = [
            (
                window.deltat() * weight,
                bundle.param(form.to_residual(stage), stepper._stage_time(window, j)),
            )
            for j, (weight, stage) in enumerate(
                zip(self._weights(window, time), stepper.stage_values(window))
            )
            if weight != 0.0
        ]
        pin = stepper._block_pin(self._last())
        data = (
            None
            if pin is None
            else data_jacobian_term(pin, time, space, params.component(0))
        )
        if data is None:
            block = (
                self._solved(ScaledSum(terms, space, params.component(0)), window, time)
                if terms
                else None
            )
            return BlockOperator([[block]], params, stepper.state_space())
        # The pinned rows hold g(t; p): E^{-1} (P op + (I - P) g_p).
        pinned: List[tuple[float, DerivativeOperator[V]]] = [(1.0, data)]
        if terms and pin is not None:
            pinned.append(
                (1.0, pin.coupled(ScaledSum(terms, space, params.component(0))))
            )
        factors: List[LinearMapProtocol[V]] = [
            self._handle(window, time),
            ScaledSum(pinned, space, params.component(0)),
        ]
        solved: LinearMapProtocol[V] = Product(factors)
        return BlockOperator([[solved]], params, stepper.state_space())

    def _stage_weights(
        self, window: StepWindow[V], time: float, weights: V
    ) -> List[Optional[tuple[V, float, V]]]:
        r"""For each stage read, its value, time and weight
        :math:`\lambda_j = h b_j(\theta)\, P E^{-\top} \lambda` in the
        residual's form; None for a stage with a zero weight."""
        stepper = self._stepper
        space, to_residual = stepper.system().space, stepper.system().form.to_residual
        x = self._handle(window, time).apply_transpose(weights)
        pin = stepper._block_pin(self._last())
        if pin is not None:
            # Pinned rows carry no f, so no curvature.
            x = pin.adjoint(x, space)
        terms: List[Optional[tuple[V, float, V]]] = []
        for j, (weight, stage) in enumerate(
            zip(self._weights(window, time), stepper.stage_values(window))
        ):
            if weight == 0.0:
                terms.append(None)
                continue
            adjoint = space.scale(window.deltat() * weight, x)
            terms.append(
                (
                    to_residual(stage),
                    stepper._stage_time(window, j),
                    to_residual(adjoint),
                )
            )
        return terms

    def _curvature_wrt_unknown(
        self,
        window: StepWindow[V],
        time: float,
        tangents: StepWindow[V],
        direction: V,
        weights: V,
    ) -> Optional[BlockVector[V]]:
        stepper = self._stepper
        bundle = stepper.system().rhs.derivatives()
        state_state, state_param = bundle.state_state, bundle.state_param
        if state_state is None or state_param is None:
            raise ValueError("the residual has no second derivatives")
        space, offset = stepper.system().space, stepper._first_stage_offset()
        terms = self._stage_weights(window, time, weights)
        tangent_stages = stepper.stage_values(tangents)
        blocks = [space.zeros() for _ in range(stepper.unknown_space().nblocks())]
        any_read = False
        for k in range(len(stepper._unknown_stages)):
            term = terms[k + offset]
            if term is None:
                continue
            any_read = True
            value = state_state(*term).apply(tangent_stages[k + offset])
            blocks[k] = space.axpby(
                1.0, state_param(*term).apply(direction), 1.0, value
            )
        return BlockVector(tuple(blocks)) if any_read else None

    def _curvature_wrt_lag(
        self,
        window: StepWindow[V],
        time: float,
        tangents: StepWindow[V],
        lag: int,
        direction: V,
        weights: V,
    ) -> Optional[V]:
        if lag != 1:
            raise ValueError(f"a one-step method has lag 1 only, got {lag}")
        stepper = self._stepper
        if not stepper.tableau().has_trivial_first_stage():
            # y_n enters only through M y_n, which is linear.
            return None
        term = self._stage_weights(window, time, weights)[0]
        if term is None:
            return None
        bundle = stepper.system().rhs.derivatives()
        state_state, state_param = bundle.state_state, bundle.state_param
        if state_state is None or state_param is None:
            raise ValueError("the residual has no second derivatives")
        value = state_state(*term).apply(tangents.past_state(1))
        return stepper.system().space.axpby(
            1.0, state_param(*term).apply(direction), 1.0, value
        )

    def _curvature_wrt_param(
        self,
        window: StepWindow[V],
        time: float,
        tangents: StepWindow[V],
        direction: V,
        weights: V,
    ) -> Optional[V]:
        stepper = self._stepper
        bundle = stepper.system().rhs.derivatives()
        if bundle.state_param is None or bundle.param_param is None:
            raise ValueError("the residual has no second derivatives")
        params = stepper._param_block_space().component(0)
        tangent_stages = stepper.stage_values(tangents)
        total: Optional[V] = None
        for term, tangent in zip(
            self._stage_weights(window, time, weights), tangent_stages
        ):
            if term is None:
                continue
            # (lambda_j^T f)_py dY_j + (lambda_j^T f)_pp v.
            value = bundle.state_param(*term).apply_transpose(tangent)
            value = params.axpby(
                1.0, bundle.param_param(*term).apply(direction), 1.0, value
            )
            total = value if total is None else params.axpby(1.0, value, 1.0, total)
        # The pinned rows hold g(t; p): (E^{-T} lambda)_R^T g_pp v.
        pin = stepper._block_pin(self._last())
        if pin is None:
            return total
        x = self._handle(window, time).apply_transpose(weights)
        data = data_curvature_term(pin, x, time, stepper.system().space, params)
        if data is None:
            return total
        value = data.apply(direction)
        return value if total is None else params.axpby(1.0, value, 1.0, total)


class RungeKuttaStepper(Generic[V, A, F]):
    r"""A Runge-Kutta stepper in mass form for any lower-triangular tableau.

    Parameters
    ----------
    tableau : ButcherTableau
        The method.
    system : MassFormSystem
        :math:`f`, :math:`M`, :math:`J`, the state space, the linear
        solver and the residual form.
    setups : SetupStrategyProtocol
        Sets up each operator the stepper solves with
        (:class:`~pyapprox.ode.setups.MassShiftRequest`) or returns a kept
        handle; its solver pairs with ``F``. Whoever changes the
        parameters calls its ``clear``.
    nonlinear_solver : NonlinearSolverProtocol, optional
        Solves the implicit stages; required exactly when the tableau has
        one.
    row_pinning : RowPinningProtocol, optional
        Which residual blocks have rows pinned to boundary data inside
        their solves, and how (see :mod:`pyapprox.ode.pinned_rows`); the
        blocks are the unknown stages in order, then the final block when
        there is one, each pinned at its own time. None for no pinning.
    builder : OperatorBuilderProtocol, optional
        Forms the operators ``setups`` sets up; by default
        :class:`~pyapprox.ode.setups.MassShiftBuilder` of ``system``.

    Raises
    ------
    TypeError
        If the tableau has an implicit stage and no nonlinear solver is
        given or the residual's bundle has no Jacobian, or ``setups``,
        ``row_pinning`` or ``builder`` does not satisfy its protocol.
    """

    def __init__(
        self,
        tableau: ButcherTableau,
        system: MassFormSystem[V, A, F],
        setups: MassShiftStrategy[V, A, F],
        nonlinear_solver: Optional[NonlinearSolverProtocol[V, A]] = None,
        row_pinning: Optional[RowPinningProtocol[V, F]] = None,
        builder: Optional[MassShiftBuilderProtocol[V, A, F]] = None,
    ) -> None:
        self._setups, self._builder = stepper_setups(system, setups, builder)
        if not tableau.is_explicit() and nonlinear_solver is None:
            raise TypeError(
                f"{tableau.name} has implicit stages, so it needs a nonlinear solver"
            )
        if not tableau.is_explicit() and system.rhs.derivatives().jacobian is None:
            raise TypeError(
                f"{tableau.name} has implicit stages, which solve with the "
                f"Jacobian; {type(system.rhs).__name__}'s derivatives bundle "
                "has none"
            )
        if row_pinning is not None and not isinstance(row_pinning, RowPinningProtocol):
            raise TypeError(
                "row_pinning must satisfy RowPinningProtocol, got "
                f"{type(row_pinning).__name__}"
            )
        # Whether the residual reports a version is fixed for its life, so
        # decided once rather than by a protocol check on every call.
        self._version = parameters_version_reader(system.rhs)
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
        # stage_map's spaces: the unknown's blocks then y_n, and the stages.
        self._stage_inputs_space = BlockInnerProductSpace(
            [space] * (nblocks + 1), "unknown and state"
        )
        self._stages_space = BlockInnerProductSpace([space] * nstages, "stages")
        self._row_pinning = row_pinning

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

    def parameters_version(self) -> int:
        """The residual's parameter version; 0 for a residual without
        parameters, which has one version for its whole life."""
        return self._version()

    def time_quadrature(self) -> TableauQuadrature:
        r"""The tableau applied to :math:`\dot z = q(y, t)`, step by step
        (see :mod:`pyapprox.ode.step_quadrature`), its nodes located as
        :meth:`stage_values` reads the stages."""
        return self._quadrature

    def continuous_extension(self) -> _TableauExtension[V, A, F]:
        r"""The method's value inside a step, :math:`M (y(t) - y_n) = h
        \sum_j b_j(\theta) f(Y_j)` with the tableau's weight polynomials
        (:meth:`ButcherTableau.weights_at`), from the step's stages, pinned
        as the block producing :math:`y_{n+1}` is, at its own time; with
        its derivatives (see :mod:`pyapprox.ode.continuous_extension`)."""
        return _TableauExtension(self)

    def _shift_handle(
        self,
        scale: float,
        pin: Optional[BlockPinProtocol[V, F]],
        time: float,
        state: V,
        position: Optional[StatePosition],
    ) -> SolverHandleProtocol[V]:
        r"""A handle for the stage operator :math:`M - s\, J(y, t)`, its
        rows replaced as ``pin`` says, from the setup strategy: the forward
        stages and the sweeps' stage blocks ask with the same :math:`s`
        and time, so a kept setup serves both.

        Parameters
        ----------
        scale : float
            :math:`s = \Delta t\, a_{ii}`.
        pin : BlockPinProtocol, optional
            How the block's rows are pinned; None for none.
        time : float
            The stage time.
        state : V
            A single column of the state space.
        position : StatePosition, optional
            Names ``state`` when it is a converged stage of a stored
            step; None for a Newton iterate.
        """
        request = MassShiftRequest(self._system, 1.0, scale, pin, time, state, position)
        return self._setups.handle(request, self._builder)

    def _block_pin(self, block: int) -> Optional[BlockPinProtocol[V, F]]:
        """How residual block ``block`` is pinned, or None."""
        if self._row_pinning is None:
            return None
        return self._row_pinning.at(StepBlock(block, self._unknown_space.nblocks()))

    def _mass_block_handle(
        self, block: int, time: float, state: V
    ) -> SolverHandleProtocol[V]:
        r"""A handle for block ``block``'s operator when its own block is
        :math:`M` alone: :math:`M`, or :math:`P M + I - P` when the block
        is pinned, from the setup strategy. The block's ``time`` and
        ``state`` complete the request; with :math:`s = 0` neither enters
        the operator or its key."""
        return self._shift_handle(0.0, self._block_pin(block), time, state, None)

    def _explicit_solve(self, block: int, state: V, increment: V, time: float) -> V:
        r"""Block ``block``'s value :math:`Y` from :math:`M (Y - y_n) = `
        ``increment``. Pinned at ``time``, it solves :math:`(P M + I - P)\,
        Y = P (M y_n + \mathrm{increment}) + (I - P)\, g(t)` instead: one
        solve either way."""
        space, pin = self._system.space, self._block_pin(block)
        handle = self._mass_block_handle(block, time, state)
        if pin is None:
            solution, info = handle.solve(increment)
            if not info.converged:
                raise SolveFailedError(
                    f"{self._tableau.name}: mass solve: {info.message}"
                )
            return space.axpby(1.0, state, 1.0, solution)
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

    def step(self, past_states: Sequence[V], span: StepSpan) -> BlockVector[V]:
        r"""Solve step :math:`n` from :math:`y_n`, every stage operator
        and mass solve set up through the setup strategy, which decides
        what is kept.

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
                stage = self._implicit_stage(i, state, known, stage_time, deltat)
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
    ) -> V:
        """Solve implicit stage ``i`` from the guess :math:`y_n`, its rows
        pinned in its Newton system when its block is, its stage operator
        set up through the strategy."""
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
        first stage, otherwise the unknown's blocks; :meth:`stage_map`
        applied to the window's unknown and :math:`y_n`."""
        read = BlockVector(tuple(window.unknown.blocks) + (window.past_state(1),))
        return list(self.stage_map(window).apply(read).blocks)

    def stage_map(self, window: StepWindow[V]) -> StepOperator[V]:
        r"""The linear map :math:`(u_n, y_n) \mapsto (Y_1, \dots, Y_s)`
        from the unknown's blocks followed by :math:`y_n` to the stage
        values, with its transpose: an identity block per stage, at
        :math:`y_n` for a trivial first stage and at the stage's unknown
        block otherwise. The final block, which no stage reads, is a zero
        column. The one statement of how stages sit in the unknown."""
        space = self._system.space
        nblocks = self._unknown_space.nblocks()
        rows: List[List[Optional[LinearMapProtocol[V]]]] = []
        for i in range(self._tableau.nstages()):
            row: List[Optional[LinearMapProtocol[V]]] = [None] * (nblocks + 1)
            if i in self._unknown_stages:
                row[self._unknown_stages.index(i)] = IdentityOperator(space)
            else:
                row[nblocks] = IdentityOperator(space)
            rows.append(row)
        return BlockOperator(rows, self._stage_inputs_space, self._stages_space)

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
        """A label for a linearization at a converged step, equal to no
        other: these Jacobians are applied, never set up, so nothing keys
        on it."""
        return LinearizationPoint(window.parameters_version, object())

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
        return self._pin_rows(window, grid)

    def _pin_rows(
        self,
        window: StepWindow[V],
        grid: List[List[Optional[ActionOperatorProtocol[V, F]]]],
    ) -> List[List[Optional[ActionOperatorProtocol[V, F]]]]:
        """``diag``'s grid with each pinned block row edited: its own block
        row-replaced at the block's time, its other blocks with the rows
        zeroed."""
        for r in range(len(grid)):
            pin = self._block_pin(r)
            if pin is None:
                continue
            time = self._block_time(window, r)
            grid[r] = [
                None
                if block is None
                else pin.own(block, time)
                if k == r
                else pin.coupled(block)
                for k, block in enumerate(grid[r])
            ]
        return grid

    def diag(self, window: StepWindow[V]) -> BlockOperator[V]:
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
        r"""A solver handle for :meth:`diag`: forward substitution, each
        diagonal block's handle from the setup strategy. A stage block
        :math:`M - \Delta t\, a_{ii} J(Y_i)` is asked with the forward
        stage's :math:`s` and time and named by the window's origin and
        the block (a window built by hand names nothing), so a kept
        forward or earlier sweep's setup serves it; a block that is
        :math:`M` alone is the forward solve's mass request."""
        grid = self._diag_blocks(window)
        stages = self.stage_values(window)
        handles: List[SolverHandleProtocol[V]] = []
        for r, row in enumerate(grid):
            block = row[r]
            if block is None:
                raise ValueError(f"{self._tableau.name}: diagonal block {r} is zero")
            if r < len(self._unknown_stages):
                i = self._unknown_stages[r]
                diagonal = self._tableau.a[i][i]
                if diagonal != 0.0:
                    origin = window.origin
                    handles.append(
                        self._shift_handle(
                            window.deltat() * diagonal,
                            self._block_pin(r),
                            self._stage_time(window, i),
                            stages[i],
                            None if origin is None else StatePosition(origin, r),
                        )
                    )
                    continue
            handles.append(
                self._mass_block_handle(
                    r, self._block_time(window, r), window.past_state(1)
                )
            )
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
            if pin is None:
                grid.append([block])
                continue
            # A pinned block's rows read Y_R - g_R(t; p): their parameter
            # Jacobian is the data's, negated.
            coupled = pin.coupled(block)
            data = data_jacobian_term(
                pin, self._block_time(window, r), space, params.component(0)
            )
            grid.append(
                [
                    coupled
                    if data is None
                    else ScaledSum(
                        [(1.0, coupled), (-1.0, data)], space, params.component(0)
                    )
                ]
            )
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
        \lambda_j^\top f_{pp}(Y_j)`, less each pinned block's
        :math:`\sum_{i \in R} \mu_{r,i}\, \partial^2 g_i/\partial p^2` at
        its time.

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
        terms: List[tuple[float, DerivativeOperator[V]]] = [
            (1.0, field(*term)) for term in self._stage_terms(window, adj)
        ]
        for r, mu in enumerate(adj.blocks):
            pin = self._block_pin(r)
            data = (
                None
                if pin is None
                else data_curvature_term(
                    pin,
                    mu,
                    self._block_time(window, r),
                    self._system.space,
                    params.component(0),
                )
            )
            if data is not None:
                terms.append((-1.0, data))
        total = ScaledSum(terms, params.component(0))
        return BlockOperator([[total]], params, params)

    def derivatives(self) -> StepperDerivatives[V]:
        """The step's derivatives, as far as the residual's bundle goes:
        the coupling to :math:`y_n` always, the parameters with the
        bundle's ``param``, the second derivatives with its second order;
        none without the bundle's Jacobian. Valid for the parameters set
        when fetched."""
        bundle = self._system.rhs.derivatives()
        if bundle.jacobian is None:
            return StepperDerivatives.none()
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
