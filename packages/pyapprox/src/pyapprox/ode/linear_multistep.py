r"""One stepper for every linear multistep method, explicit or implicit.

For :math:`M \dot y = f(y, t; p)` and a method with coefficients
:math:`(\alpha_l, \beta_l)` (see :mod:`pyapprox.ode.multistep_methods`),
step :math:`n` solves for the new state :math:`u_n = y_{n+1}` in mass
form,

.. math::

    R_n = \sum_{l=0}^{k} \alpha_l\, M y_{n+1-l}
    - \Delta t \sum_{l=0}^{k} \beta_l\, f(y_{n+1-l}, t_{n+1-l}) = 0,

lag 0 being the unknown and lag :math:`l \ge 1` the stored state
:math:`y_{n+1-l}`. An implicit formula (:math:`\beta_0 \ne 0`) is solved
by the injected nonlinear solver with the operator :math:`\alpha_0 M -
\Delta t\, \beta_0 J(u)`; an explicit one with :math:`M`. A step is
given the stored states it reads; a startup step, with fewer of them
than the method's depth, takes the method's startup formula for that
many.

The step's derivatives follow from the coefficients alone:

- ``diag`` :math:`= \alpha_0 M - \Delta t\, \beta_0 J(u)`;
- ``off_diag`` at lag :math:`l` :math:`= \alpha_l M - \Delta t\,
  \beta_l J(y_{n+1-l})`;
- ``param`` :math:`= -\Delta t \sum_l \beta_l f_p(y_{n+1-l})`;
- :math:`\mu^\top R_n` depends on the state at lag :math:`l` only through
  :math:`-\Delta t\, \beta_l\, \mu^\top f(y_{n+1-l})`, so the second
  derivatives are the residual's, weighted by :math:`\lambda_l = -\Delta
  t\, \beta_l\, \mu`: the :math:`(l, l)` blocks and the mixed block of
  lag :math:`l` for every lag with :math:`\beta_l \ne 0`, and no block
  coupling two lags.

A step is given one time per lag, :math:`(t_{n+1}, t_n, \dots)`, and
asks the method for the formula at those times, which has exactly one
coefficient pair per lag; :math:`f` at lag :math:`l` is evaluated at
:math:`t_{n+1-l}` and :math:`\Delta t = t_{n+1} - t_n`. The coefficients
depend only on the times, so the derivatives above keep their structure
whatever the step sizes. States are stack vectors, converted to the
residual's form only where :math:`f` and its derivatives are called.
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

from pyapprox.ode.continuous_extension import (
    ContinuousExtensionDerivatives,
    fraction_of_step,
)
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.multistep_methods import (
    MultistepCoefficients,
    MultistepContinuousCoefficients,
    MultistepFormulaProtocol,
)
from pyapprox.ode.pinned_rows import BlockPinProtocol, RowPinningProtocol, StepBlock
from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.ode.step_quadrature import MultistepQuadrature
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import (
    StepOperator,
    StepperDerivatives,
    StepWindow,
    check_step_span,
)
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
    Product,
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
G = TypeVar("G", bound=FormatProtocol)


@runtime_checkable
class _VersionedProtocol(Protocol):
    """A residual that reports its parameter version."""

    def parameters_version(self) -> int: ...


def _unversioned() -> int:
    """The version of a residual without parameters: one for its life."""
    return 0


class _NewStateSystem(Generic[V, A, F]):
    r"""An implicit step's equation for the nonlinear solver,

    .. math::

        G(Y) = \alpha_0 M Y - \Delta t\, \beta_0 f(Y, t_{n+1}) + k = 0,

    with :math:`k = \sum_{l \ge 1} (\alpha_l M y_{n+1-l} - \Delta t\,
    \beta_l f(y_{n+1-l}))` the stored states' contribution, and Jacobian
    :math:`\alpha_0 M - \Delta t\, \beta_0 J(Y, t_{n+1})`.

    Parameters
    ----------
    stepper : LinearMultistepStepper
        Supplies the system, :math:`f`, :math:`J` and the labels.
    known : V
        :math:`k`, a single column of the state space.
    time : float
        :math:`t_{n+1}`.
    formula : MultistepCoefficients
        The step's coefficients.
    deltat : float
        The step size.
    solver : KeyedSolverProtocol
        The session's solver, which sets up the step's operator.
    """

    def __init__(
        self,
        stepper: "LinearMultistepStepper[V, A, F]",
        known: V,
        time: float,
        formula: MultistepCoefficients,
        deltat: float,
        solver: KeyedSolverProtocol[V, F],
    ) -> None:
        self._stepper, self._known, self._time = stepper, known, time
        self._alpha0 = formula.alpha[0]
        self._scale = deltat * formula.beta[0]
        self._solver = solver

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._stepper.system().space

    def residual(self, x: V) -> V:
        space, mass = self.space(), self._stepper.system().mass
        # alpha_0 M Y - dt beta_0 f(Y) + k
        value = space.scale(self._alpha0, mass.apply(x))
        value = space.axpby(
            -self._scale, self._stepper.rhs(x, self._time), 1.0, value, out=value
        )
        value = space.axpby(1.0, self._known, 1.0, value, out=value)
        pin = self._stepper.pin()
        # Pinned rows read Y_R - g_R(t_{n+1}) instead.
        return value if pin is None else pin.residual(value, x, self._time, space)

    def jacobian_handle(self, x: V) -> SolverHandleProtocol[V]:
        return self._stepper.new_state_handle(
            x, self._time, self._alpha0, self._scale, self._solver
        )

    def is_affine(self) -> bool:
        """Whether the step equation is affine in the new state: the
        system's :math:`f` is, and pinned rows (identity rows with data)
        keep it so."""
        return self._stepper.system().is_affine()


class _MultistepSession(Generic[V, A, F]):
    """One solve's steps of a :class:`LinearMultistepStepper`.

    Implicit steps are set up through one keyed solver, made by the
    stepper's reuse policy for the session only: under the default policy
    a step operator the system labels for reuse (see
    :meth:`MassFormSystem.operator_label`) is formed and set up once per
    label, keeping one handle, since a step forms one operator and a
    startup formula's is not used again.
    """

    def __init__(self, stepper: "LinearMultistepStepper[V, A, F]") -> None:
        self._stepper = stepper
        self._solver: KeyedSolverProtocol[V, F] = stepper.reuse().solver(
            stepper.system().solver, 1
        )

    def step(self, past_states: Sequence[V], span: StepSpan) -> BlockVector[V]:
        """Solve one step (see :meth:`LinearMultistepStepper.step`)."""
        return self._stepper._step(past_states, span, self._solver)

    def diag_handle(
        self, window: StepWindow[V]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """The step's diagonal handle (see
        :meth:`LinearMultistepStepper.diag_handle`), set up through this
        session."""
        return self._stepper._diag_handle(window, self._solver)


class _MultistepExtension(Generic[V, A, F]):
    r"""A multistep stepper's continuous extension (see
    :meth:`LinearMultistepStepper.continuous_extension`).

    With the method's coefficients :math:`(c_l, d_l)` at :math:`\theta`
    (:class:`~pyapprox.ode.multistep_methods.
    MultistepContinuousCoefficients`), :math:`y(t)` solves one more row
    pinned as the new state's is, at ``time``: :math:`E\, y(t) = P (M
    \sum_l c_l y_{n+1-l} + h \sum_l d_l f_{n+1-l}) + (I - P) g(t)`,
    :math:`E = P M + I - P`. Its derivatives are the row's, solved with
    :math:`E`: :math:`E^{-1} P (c_l M + h d_l J_l)` at lag :math:`l` and
    :math:`E^{-1} P h \sum_l d_l f_{p,l}`, and the lag weights
    :math:`\lambda_l = h d_l\, P E^{-\top} \lambda` contract the
    residual's second derivatives, as the step's do.
    """

    def __init__(self, stepper: "LinearMultistepStepper[V, A, F]") -> None:
        self._stepper = stepper

    def unknown_space(self) -> BlockInnerProductSpace[V, A]:
        return self._stepper.unknown_space()

    def state_space(self) -> BlockInnerProductSpace[V, A]:
        return self._stepper.state_space()

    def param_space(self) -> Optional[BlockInnerProductSpace[V, A]]:
        return self._stepper.param_space()

    def _coefficients(
        self, window: StepWindow[V], time: float
    ) -> MultistepContinuousCoefficients:
        method = self._stepper.method()
        coefficients = method.continuous_coefficients_at(
            window.span, fraction_of_step(window, time)
        )
        if coefficients is None:
            raise TypeError(f"{method.name} has no continuous extension")
        if len(coefficients.states) != window.span.nlags():
            raise ValueError(
                f"{method.name} gave {len(coefficients.states)} coefficients for "
                f"a step with {window.span.nlags()} lags"
            )
        return coefficients

    def _handle(self) -> SolverHandleProtocol[V]:
        """:math:`E^{-1}`: the mass, pinned as the new state is."""
        return self._stepper._mass_handle_for(1.0)

    def value(self, window: StepWindow[V], time: float) -> V:
        """:math:`y(t)` for ``time`` in the window's span."""
        stepper = self._stepper
        space, mass = stepper.system().space, stepper.system().mass
        coefficients = self._coefficients(window, time)
        rhs = space.zeros()
        for lag, state in enumerate(stepper._lag_states(window)):
            c, d = coefficients.states[lag], coefficients.slopes[lag]
            if c != 0.0:
                rhs = space.axpby(c, mass.apply(state), 1.0, rhs, out=rhs)
            if d != 0.0:
                slope = stepper.rhs(state, window.lag_time(lag))
                rhs = space.axpby(window.deltat() * d, slope, 1.0, rhs, out=rhs)
        pin = stepper.pin()
        if pin is not None:
            rhs = pin.right_hand_side(rhs, time)
        solution, info = self._handle().solve(rhs)
        if not info.converged:
            raise SolveFailedError(
                f"{stepper.method().name}: continuous extension's mass solve: "
                f"{info.message}"
            )
        return solution

    def derivatives(self) -> ContinuousExtensionDerivatives[V]:
        """As far as the residual's bundle goes: every lag always, the
        parameters with its ``param``, the curvatures with its second
        order."""
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

    def _solved(
        self, op: ActionOperatorProtocol[V, G]
    ) -> LinearMapProtocol[V]:
        r""":math:`E^{-1} P\, \mathrm{op}`."""
        pin = self._stepper.pin()
        coupled = op if pin is None else pin.coupled(op)
        return Product([self._handle(), coupled])

    def _lag_block(
        self, window: StepWindow[V], time: float, lag: int
    ) -> Optional[LinearMapProtocol[V]]:
        r""":math:`E^{-1} P (c_l M + h d_l J_l)`, or None when both are
        zero."""
        stepper = self._stepper
        coefficients = self._coefficients(window, time)
        c, d = coefficients.states[lag], coefficients.slopes[lag]
        terms: List[tuple[float, ActionOperatorProtocol[V, F]]] = []
        if c != 0.0:
            terms.append((c, stepper.system().mass))
        if d != 0.0:
            terms.append(
                (window.deltat() * d, stepper._converged_jacobian(window, lag))
            )
        if not terms:
            return None
        return self._solved(ScaledSum(terms, stepper.system().space))

    def _wrt_unknown(self, window: StepWindow[V], time: float) -> StepOperator[V]:
        stepper = self._stepper
        return BlockOperator(
            [[self._lag_block(window, time, 0)]],
            stepper.unknown_space(),
            stepper.state_space(),
        )

    def _wrt_lag(
        self, window: StepWindow[V], time: float, lag: int
    ) -> StepOperator[V]:
        if not 1 <= lag <= len(window.past_states):
            raise ValueError(
                f"lag must be between 1 and {len(window.past_states)}, got {lag}"
            )
        states = self._stepper.state_space()
        return BlockOperator([[self._lag_block(window, time, lag)]], states, states)

    def _wrt_param(self, window: StepWindow[V], time: float) -> StepOperator[V]:
        stepper = self._stepper
        bundle = stepper.system().rhs.derivatives()
        if bundle.param is None:
            raise ValueError("the residual has no parameter derivatives")
        form, space = stepper.system().form, stepper.system().space
        coefficients = self._coefficients(window, time)
        params = stepper._param_block_space()
        terms = [
            (
                window.deltat() * coefficients.slopes[lag],
                bundle.param(form.to_residual(state), window.lag_time(lag)),
            )
            for lag, state in enumerate(stepper._lag_states(window))
            if coefficients.slopes[lag] != 0.0
        ]
        pin = stepper.pin()
        data = None if pin is None else pin.data_param(time, space)
        if data is None:
            block = (
                self._solved(ScaledSum(terms, space, params.component(0)))
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
            self._handle(),
            ScaledSum(pinned, space, params.component(0)),
        ]
        solved: LinearMapProtocol[V] = Product(factors)
        return BlockOperator([[solved]], params, stepper.state_space())

    def _lag_weights(
        self, window: StepWindow[V], time: float, weights: V
    ) -> List[Optional[tuple[V, float, V]]]:
        r"""For each lag, its state, time and weight :math:`\lambda_l = h
        d_l\, P E^{-\top} \lambda` in the residual's form; None for a lag
        with :math:`d_l = 0`."""
        stepper = self._stepper
        space, to_residual = stepper.system().space, stepper.system().form.to_residual
        coefficients = self._coefficients(window, time)
        x = self._handle().apply_transpose(weights)
        pin = stepper.pin()
        if pin is not None:
            # Pinned rows carry no f, so no curvature.
            x = pin.adjoint(x, space)
        terms: List[Optional[tuple[V, float, V]]] = []
        for lag, state in enumerate(stepper._lag_states(window)):
            d = coefficients.slopes[lag]
            if d == 0.0:
                terms.append(None)
                continue
            adjoint = space.scale(window.deltat() * d, x)
            terms.append(
                (to_residual(state), window.lag_time(lag), to_residual(adjoint))
            )
        return terms

    def _lag_curvature(
        self,
        window: StepWindow[V],
        time: float,
        lag: int,
        tangent: V,
        direction: V,
        weights: V,
    ) -> Optional[V]:
        r""":math:`(\lambda_l^\top f)_{yy}\, w_l + (\lambda_l^\top f)_{yp}\,
        v`, or None when :math:`d_l = 0`."""
        stepper = self._stepper
        term = self._lag_weights(window, time, weights)[lag]
        if term is None:
            return None
        bundle = stepper.system().rhs.derivatives()
        state_state, state_param = bundle.state_state, bundle.state_param
        if state_state is None or state_param is None:
            raise ValueError("the residual has no second derivatives")
        value = state_state(*term).apply(tangent)
        return stepper.system().space.axpby(
            1.0, state_param(*term).apply(direction), 1.0, value
        )

    def _curvature_wrt_unknown(
        self,
        window: StepWindow[V],
        time: float,
        tangents: StepWindow[V],
        direction: V,
        weights: V,
    ) -> Optional[BlockVector[V]]:
        value = self._lag_curvature(
            window, time, 0, tangents.unknown.blocks[0], direction, weights
        )
        return None if value is None else BlockVector((value,))

    def _curvature_wrt_lag(
        self,
        window: StepWindow[V],
        time: float,
        tangents: StepWindow[V],
        lag: int,
        direction: V,
        weights: V,
    ) -> Optional[V]:
        return self._lag_curvature(
            window, time, lag, tangents.past_state(lag), direction, weights
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
        tangent_states = stepper._lag_states(tangents)
        total: Optional[V] = None
        for term, tangent in zip(
            self._lag_weights(window, time, weights), tangent_states
        ):
            if term is None:
                continue
            # (lambda_l^T f)_py w_l + (lambda_l^T f)_pp v.
            value = bundle.state_param(*term).apply_transpose(tangent)
            value = params.axpby(
                1.0, bundle.param_param(*term).apply(direction), 1.0, value
            )
            total = value if total is None else params.axpby(1.0, value, 1.0, total)
        # The pinned rows hold g(t; p): (E^{-T} lambda)_R^T g_pp v.
        pin = stepper.pin()
        if pin is None:
            return total
        x = self._handle().apply_transpose(weights)
        data = pin.data_curvature(x, time, stepper.system().space)
        if data is None:
            return total
        value = data.apply(direction)
        return value if total is None else params.axpby(1.0, value, 1.0, total)


class LinearMultistepStepper(Generic[V, A, F]):
    r"""A linear multistep stepper in mass form, with the formula of each
    step taken from the method at that step's times.

    Parameters
    ----------
    method : MultistepFormulaProtocol
        Gives the formula of each step from its times.
    system : MassFormSystem
        :math:`f`, :math:`M`, :math:`J`, the state space, the linear
        solver and the residual form.
    nonlinear_solver : NonlinearSolverProtocol, optional
        Solves implicit steps; required exactly when the method is
        implicit.
    row_pinning : RowPinningProtocol, optional
        Whether the step's one residual block, the one producing
        :math:`y_{n+1}`, has rows pinned to boundary data inside its solve,
        and how (see :mod:`pyapprox.ode.pinned_rows`). None for no
        pinning.
    reuse : ReusePolicyProtocol, optional
        How a solve keeps step setups between steps: by default the
        current one (:class:`KeepWithinSolve`); :class:`NoReuse` keeps
        none, to save memory.

    Raises
    ------
    TypeError
        If the method does not satisfy its protocol, is implicit and no
        nonlinear solver is given, or ``row_pinning`` or ``reuse`` does not
        satisfy its protocol.
    """

    def __init__(
        self,
        method: MultistepFormulaProtocol,
        system: MassFormSystem[V, A, F],
        nonlinear_solver: Optional[NonlinearSolverProtocol[V, A]] = None,
        row_pinning: Optional[RowPinningProtocol[V]] = None,
        reuse: Optional[ReusePolicyProtocol] = None,
    ) -> None:
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
        if not isinstance(method, MultistepFormulaProtocol):
            raise TypeError(
                "method must satisfy MultistepFormulaProtocol, got "
                f"{type(method).__name__}"
            )
        if not method.is_explicit() and nonlinear_solver is None:
            raise TypeError(
                f"{method.name} is implicit, so it needs a nonlinear solver"
            )
        if row_pinning is not None and not isinstance(row_pinning, RowPinningProtocol):
            raise TypeError(
                "row_pinning must satisfy RowPinningProtocol, got "
                f"{type(row_pinning).__name__}"
            )
        self._method = method
        self._quadrature = MultistepQuadrature(method)
        self._system = system
        self._nonlinear_solver = nonlinear_solver
        space = system.space
        self._unknown_space = BlockInnerProductSpace([space], "unknown")
        self._residual_space = BlockInnerProductSpace([space], "residual")
        self._state_space = BlockInnerProductSpace([space], "state")
        self._mass_handle: Optional[SolverHandleProtocol[V]] = None
        # The one block, of one, is the step end.
        self._pin = None if row_pinning is None else row_pinning.at(StepBlock(0, 1))
        # Explicit steps' row-replaced alpha_0 M, set up once per alpha_0.
        self._pinned_mass_handles: Dict[float, SolverHandleProtocol[V]] = {}
        self._points = count()

    def method(self) -> MultistepFormulaProtocol:
        """The method."""
        return self._method

    def reuse(self) -> ReusePolicyProtocol:
        """How a solve keeps step setups between steps."""
        return self._reuse

    def pin(self) -> Optional[BlockPinProtocol[V]]:
        """How the step's block is pinned, or None."""
        return self._pin

    def system(self) -> MassFormSystem[V, A, F]:
        """What the stepper steps."""
        return self._system

    def depth(self) -> int:
        """The number of stored states a full step reads."""
        return self._method.depth()

    def unknown_space(self) -> BlockInnerProductSpace[V, A]:
        """The space of :math:`u_n = y_{n+1}`, one block."""
        return self._unknown_space

    def residual_space(self) -> BlockInnerProductSpace[V, A]:
        """The space of :math:`R_n`, one block."""
        return self._residual_space

    def state_space(self) -> BlockInnerProductSpace[V, A]:
        """The stored states, as one-block vectors."""
        return self._state_space

    def readout(self) -> StepOperator[V]:
        """The map from :math:`u_n` to :math:`y_{n+1}`: the identity."""
        return BlockOperator(
            [[IdentityOperator(self._system.space)]],
            self._unknown_space,
            self._state_space,
        )

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

    def time_quadrature(self) -> MultistepQuadrature:
        r"""The method applied to :math:`\dot z = q(y, t)`, step by step
        (see :mod:`pyapprox.ode.step_quadrature`)."""
        return self._quadrature

    def continuous_extension(self) -> _MultistepExtension[V, A, F]:
        r"""The method's value inside a step, from its own polynomial
        (:meth:`MultistepFormulaProtocol.continuous_coefficients_at`): the
        interpolant of :math:`y` for BDF, the partial integral of the
        slopes' interpolant for Adams; pinned as the new state is, at its
        own time; with its derivatives (see
        :mod:`pyapprox.ode.continuous_extension`). A method with no
        extension (a fixed table) is refused when the extension is read."""
        return _MultistepExtension(self)

    def _point(self, provisional: bool, version: int) -> LinearizationPoint:
        """A new label for a linearization: provisional at a Newton
        iterate, never reused."""
        return LinearizationPoint(
            version, ("multistep", next(self._points)), provisional=provisional
        )

    def _jacobian(
        self, state: V, time: float, point: LinearizationPoint
    ) -> ActionOperatorProtocol[V, F]:
        """:math:`J(y, t)` at a single column ``state``."""
        system = self._system
        return system.jacobian(system.form.to_residual(state), time, point)

    def _new_state_operator(
        self,
        state: V,
        time: float,
        alpha0: float,
        scale: float,
        point: LinearizationPoint,
    ) -> ActionOperatorProtocol[V, F]:
        r""":math:`\alpha_0 M - s\, J(y, t)`, :math:`J` labeled ``point``,
        with the step-end rows replaced when they are pinned."""
        system = self._system
        jacobian = self._jacobian(state, time, point)
        operator = ScaledSum([(alpha0, system.mass), (-scale, jacobian)], system.space)
        return operator if self._pin is None else self._pin.own(operator)

    def _labeled_new_state_operator(
        self,
        state: V,
        time: float,
        alpha0: float,
        scale: float,
        label: LinearizationPoint,
    ) -> LinearizedOperator[V, F]:
        """The step operator with :math:`J` labeled ``label``, wrapped
        under ``label`` so a caching solver keys it."""
        return LinearizedOperator(
            self._new_state_operator(state, time, alpha0, scale, label), label
        )

    def new_state_handle(
        self,
        state: V,
        time: float,
        alpha0: float,
        scale: float,
        solver: KeyedSolverProtocol[V, F],
    ) -> SolverHandleProtocol[V]:
        r"""A handle for the step operator :math:`\alpha_0 M - s\, J(y, t)`
        with :math:`s = \Delta t\, \beta_0`, its step-end rows replaced when
        they are pinned.

        Set up by ``solver`` under the system's reuse label when it gives
        one: equal step operators of a session then share one setup, and
        one whose handle is kept is not formed again. A :math:`J` that sees
        the state gives no label, and each Newton iterate is set up afresh.

        Parameters
        ----------
        state : V
            A single column of the state space.
        time : float
            :math:`t_{n+1}`.
        alpha0 : float
            :math:`\alpha_0`.
        scale : float
            :math:`s`.
        solver : KeyedSolverProtocol
            The session's solver.
        """
        system = self._system
        version = self.parameters_version()
        label = system.operator_label(version, time, ("new state", alpha0, scale))
        if label is None:
            return solver.setup(
                self._new_state_operator(
                    state, time, alpha0, scale, self._point(True, version)
                )
            )
        descriptor = system.space.descriptor()
        return solver.setup_keyed(
            ReuseKey(label, descriptor, descriptor),
            partial(
                self._labeled_new_state_operator, state, time, alpha0, scale, label
            ),
        )

    def _mass_handle_for(self, alpha0: float) -> SolverHandleProtocol[V]:
        r"""A handle for the new-state block of an explicit step: :math:`M`
        (the right-hand side then carries :math:`1/\alpha_0`), or
        :math:`P \alpha_0 M + I - P` with the step-end rows pinned, each set
        up once (per :math:`\alpha_0` when pinned)."""
        pin = self._pin
        if pin is None:
            if self._mass_handle is None:
                self._mass_handle = self._system.solver.setup(self._system.mass)
            return self._mass_handle
        handle = self._pinned_mass_handles.get(alpha0)
        if handle is None:
            scaled = ScaledSum([(alpha0, self._system.mass)], self._system.space)
            handle = self._system.solver.setup(pin.own(scaled))
            self._pinned_mass_handles[alpha0] = handle
        return handle

    def _mass_solve(self, rhs: V) -> V:
        """:math:`M^{-1}` ``rhs``, with the mass set up once."""
        if self._pin is not None:
            raise AssertionError("only called without step-end rows")
        solution, info = self._mass_handle_for(1.0).solve(rhs)
        if not info.converged:
            raise SolveFailedError(f"{self._method.name}: mass solve: {info.message}")
        return solution

    def _pinned_mass_solve(self, known: V, alpha0: float, time: float) -> V:
        r"""An explicit step with the step-end rows pinned:
        :math:`(P \alpha_0 M + I - P)\, y_{n+1} = -P k + (I - P)\, g(t)`,
        one solve as without them, its operator set up once per
        :math:`\alpha_0`."""
        pin = self._pin
        if pin is None:
            raise AssertionError("only called with step-end rows")
        handle = self._mass_handle_for(alpha0)
        rhs = self._system.space.scale(-1.0, known)
        solution, info = handle.solve(pin.right_hand_side(rhs, time))
        if not info.converged:
            raise SolveFailedError(
                f"{self._method.name}: pinned mass solve: {info.message}"
            )
        return solution

    # ------------------------------------------------------------------
    # The forward step
    # ------------------------------------------------------------------

    def _formula(self, span: StepSpan) -> MultistepCoefficients:
        """The method's formula for a step over ``span``, checked to have
        exactly one coefficient pair per lag."""
        formula = self._method.coefficients_at(span)
        if len(formula.alpha) != span.nlags():
            raise ValueError(
                f"{self._method.name} gave {len(formula.alpha)} coefficients for "
                f"a step with {span.nlags()} lags"
            )
        return formula

    def _known(
        self,
        past_states: Sequence[V],
        span: StepSpan,
        formula: MultistepCoefficients,
    ) -> V:
        r""":math:`k = \sum_{l \ge 1} (\alpha_l M y_{n+1-l} - \Delta t\,
        \beta_l f(y_{n+1-l}, t_{n+1-l}))`, with :math:`\Delta t = h_n`, the
        span's own size."""
        space, mass = self._system.space, self._system.mass
        deltat = span.deltat()
        known = space.zeros()
        for lag, state in enumerate(past_states, start=1):
            alpha, beta = formula.alpha[lag], formula.beta[lag]
            if alpha != 0.0:
                known = space.axpby(alpha, mass.apply(state), 1.0, known, out=known)
            if beta != 0.0:
                slope = self.rhs(state, span.lag_time(lag))
                known = space.axpby(-deltat * beta, slope, 1.0, known, out=known)
        return known

    def session(self) -> _MultistepSession[V, A, F]:
        """A new session for one solve's steps, sharing step setups
        between them and with no other session."""
        return _MultistepSession(self)

    def step(self, past_states: Sequence[V], span: StepSpan) -> BlockVector[V]:
        r"""Solve step :math:`n` for :math:`y_{n+1}`, as a session of this
        one step (nothing is kept after it).

        Parameters
        ----------
        past_states : sequence of V
            :math:`(y_n, y_{n-1}, \dots)`, single columns of the state
            space: the method's depth of them, or fewer at a startup
            step.
        span : StepSpan
            The step's size and the times of its lags, one size per state
            read.

        Returns
        -------
        BlockVector
            The converged unknown :math:`u_n = y_{n+1}`.

        Raises
        ------
        ValueError
            If the number of states is not between 1 and the depth, the
            span does not match them, or the method's coefficients do not
            cover these step sizes.
        SolveFailedError
            If the nonlinear solve or a mass solve fails.
        """
        return self.session().step(past_states, span)

    def _step(
        self,
        past_states: Sequence[V],
        span: StepSpan,
        linear_solver: KeyedSolverProtocol[V, F],
    ) -> BlockVector[V]:
        """:meth:`step`, an implicit step's operator set up by
        ``linear_solver``."""
        check_step_span(span, len(past_states))
        formula = self._formula(span)
        space = self._system.space
        known = self._known(past_states, span, formula)
        new_time, deltat = span.end(), span.deltat()
        if formula.is_explicit():
            # alpha_0 M y_{n+1} = -k
            if self._pin is None:
                new_state = self._mass_solve(
                    space.scale(-1.0 / formula.alpha[0], known)
                )
            else:
                new_state = self._pinned_mass_solve(known, formula.alpha[0], new_time)
            return BlockVector((new_state,))
        solver = self._nonlinear_solver
        if solver is None:
            raise TypeError(f"{self._method.name} needs a nonlinear solver")
        system = _NewStateSystem(self, known, new_time, formula, deltat, linear_solver)
        new_state, info = solver.solve(system, past_states[0])
        if not info.converged:
            raise SolveFailedError(
                f"{self._method.name}: step to t = {new_time}: {info.message}"
            )
        return BlockVector((new_state,))

    # ------------------------------------------------------------------
    # The step residual and its derivatives
    # ------------------------------------------------------------------

    def _lag_states(self, window: StepWindow[V]) -> List[V]:
        """The states by lag: the unknown at 0, the stored states at
        :math:`l \\ge 1`."""
        return [window.unknown.blocks[0], *window.past_states]

    def _window_formula(self, window: StepWindow[V]) -> MultistepCoefficients:
        return self._formula(window.span)

    def step_residual(self, window: StepWindow[V]) -> BlockVector[V]:
        r""":math:`R_n` at the window's unknown and stored states, at the
        current parameters (see the module docstring)."""
        formula = self._window_formula(window)
        space, mass = self._system.space, self._system.mass
        unknown = window.unknown.blocks[0]
        residual = space.scale(formula.alpha[0], mass.apply(unknown))
        if formula.beta[0] != 0.0:
            slope = self.rhs(unknown, window.lag_time(0))
            residual = space.axpby(
                -window.deltat() * formula.beta[0], slope, 1.0, residual, out=residual
            )
        known = self._known(window.past_states, window.span, formula)
        residual = space.axpby(1.0, known, 1.0, residual, out=residual)
        if self._pin is not None:
            # Pinned rows read y_R - g_R(t_{n+1}).
            residual = self._pin.residual(residual, unknown, window.lag_time(0), space)
        return BlockVector((residual,))

    def _converged_jacobian(
        self, window: StepWindow[V], lag: int
    ) -> ActionOperatorProtocol[V, F]:
        """:math:`J` at the window's state of lag ``lag``."""
        state = self._lag_states(window)[lag]
        return self._jacobian(
            state,
            window.lag_time(lag),
            self._point(False, window.parameters_version),
        )

    def _lag_block(
        self, window: StepWindow[V], lag: int
    ) -> Optional[ActionOperatorProtocol[V, F]]:
        r""":math:`\alpha_l M - \Delta t\, \beta_l J(y_{n+1-l})`, or None
        when both coefficients are zero; with step-end rows, lag 0's block
        row-replaced and the others' rows zeroed."""
        formula = self._window_formula(window)
        alpha, beta = formula.alpha[lag], formula.beta[lag]
        terms: List[tuple[float, ActionOperatorProtocol[V, F]]] = []
        if alpha != 0.0:
            terms.append((alpha, self._system.mass))
        if beta != 0.0:
            jacobian = self._converged_jacobian(window, lag)
            terms.append((-window.deltat() * beta, jacobian))
        if not terms:
            return None
        block = ScaledSum(terms, self._system.space)
        pin = self._pin
        if pin is None:
            return block
        return pin.own(block) if lag == 0 else pin.coupled(block)

    def diag(self, window: StepWindow[V]) -> StepOperator[V]:
        r""":math:`\partial R_n/\partial u_n = \alpha_0 M - \Delta t\,
        \beta_0 J(u_n)`.

        Parameters
        ----------
        window : StepWindow
            The step's converged unknown, stored states, time and size.
        """
        return BlockOperator(
            [[self._lag_block(window, 0)]], self._unknown_space, self._residual_space
        )

    def diag_handle(
        self, window: StepWindow[V]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """A solver handle for :meth:`diag`, as a session of this one
        window, set up by the system's solver."""
        return self.session().diag_handle(window)

    def _diag_label(self, window: StepWindow[V]) -> Optional[LinearizationPoint]:
        r"""The reuse label of the diagonal block, or None: an implicit
        step's :math:`\alpha_0 M - \Delta t\, \beta_0 J` labeled exactly as
        the forward step labels it, an explicit step's :math:`\alpha_0 M`
        by the mass and :math:`\alpha_0`."""
        formula = self._window_formula(window)
        alpha0, beta0 = formula.alpha[0], formula.beta[0]
        version = window.parameters_version
        if beta0 == 0.0:
            return LinearizationPoint(version, ("mass", alpha0))
        key = ("new state", alpha0, window.deltat() * beta0)
        return self._system.operator_label(version, window.lag_time(0), key)

    def _diag_handle(
        self, window: StepWindow[V], solver: KeyedSolverProtocol[V, F]
    ) -> SolverHandleProtocol[BlockVector[V]]:
        """:meth:`diag_handle`, the block set up by ``solver`` under its
        reuse label when it has one."""
        # alpha_0 != 0, so the diagonal block is never zero.
        block = self._lag_block(window, 0)
        if block is None:
            raise AssertionError("alpha_0 is nonzero")
        label = self._diag_label(window)
        labeled = block if label is None else LinearizedOperator(block, label)
        operator = BlockOperator([[block]], self._unknown_space, self._residual_space)
        return BlockLowerTriangularHandle(operator, [solver.setup(labeled)])

    def _off_diag(self, window: StepWindow[V], lag: int) -> StepOperator[V]:
        r""":math:`\partial R_n/\partial y_{n+1-l} = \alpha_l M - \Delta t\,
        \beta_l J(y_{n+1-l})`.

        Parameters
        ----------
        window : StepWindow
            The step.
        lag : int
            Between 1 and the number of states the window read.
        """
        if not 1 <= lag <= len(window.past_states):
            raise ValueError(
                f"lag must be between 1 and {len(window.past_states)}, got {lag}"
            )
        return BlockOperator(
            [[self._lag_block(window, lag)]], self._state_space, self._residual_space
        )

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
        r""":math:`\partial R_n/\partial p = -\Delta t \sum_l \beta_l
        f_p(y_{n+1-l})`.

        Parameters
        ----------
        window : StepWindow
            The step.
        """
        bundle = self._system.rhs.derivatives()
        if bundle.param is None:
            raise ValueError("the residual has no parameter derivatives")
        formula, form = self._window_formula(window), self._system.form
        terms = [
            (
                -window.deltat() * formula.beta[lag],
                bundle.param(
                    form.to_residual(state),
                    window.lag_time(lag),
                ),
            )
            for lag, state in enumerate(self._lag_states(window))
            if formula.beta[lag] != 0.0
        ]
        params = self._param_block_space()
        space = self._system.space
        block = ScaledSum(terms, space, params.component(0))
        if self._pin is None:
            return BlockOperator([[block]], params, self._residual_space)
        # The step-end rows read y_R - g_R(t; p): their parameter Jacobian
        # is the data's, negated.
        coupled = self._pin.coupled(block)
        data = self._pin.data_param(window.lag_time(0), space)
        pinned = (
            coupled
            if data is None
            else ScaledSum([(1.0, coupled), (-1.0, data)], space, params.component(0))
        )
        return BlockOperator([[pinned]], params, self._residual_space)

    def _weighted(
        self, window: StepWindow[V], adj: BlockVector[V], lag: int
    ) -> Optional[tuple[V, float, V]]:
        r"""The residual-form state, time and adjoint weight
        :math:`\lambda_l = -\Delta t\, \beta_l \mu` of lag ``lag``, ready
        for the residual's second derivatives; None when
        :math:`\beta_l = 0` at this window."""
        formula = self._window_formula(window)
        beta = formula.beta[lag]
        if beta == 0.0:
            return None
        space, to_residual = self._system.space, self._system.form.to_residual
        mu = adj.blocks[0]
        if self._pin is not None:
            # Pinned rows carry no f, so no curvature.
            mu = self._pin.adjoint(mu, space)
        weight = space.scale(-window.deltat() * beta, mu)
        return (
            to_residual(self._lag_states(window)[lag]),
            window.lag_time(lag),
            to_residual(weight),
        )

    def _lag_space(self, lag: int) -> BlockInnerProductSpace[V, A]:
        return self._unknown_space if lag == 0 else self._state_space

    def _state_state(
        self, window: StepWindow[V], adj: BlockVector[V], a: int, b: int
    ) -> StepOperator[V]:
        r"""The :math:`(a, b)` lag block of :math:`\mu^\top \partial^2
        R_n`: :math:`\lambda_a^\top f_{yy}(y_{n+1-a})` when :math:`a = b`,
        zero at a window whose :math:`\beta_a` is zero.

        Raises
        ------
        ValueError
            If :math:`a \ne b` (never declared).
        """
        if a != b:
            raise ValueError(f"the ({a}, {b}) block is not declared nonzero")
        field = self._system.rhs.derivatives().state_state
        if field is None:
            raise ValueError("the residual has no second derivatives")
        weighted = self._weighted(window, adj, a)
        block = None if weighted is None else field(*weighted)
        return BlockOperator([[block]], self._lag_space(b), self._lag_space(a))

    def _state_param(
        self, window: StepWindow[V], adj: BlockVector[V], a: int
    ) -> StepOperator[V]:
        r"""The parameters-to-lag-:math:`a` block of :math:`\mu^\top
        \partial^2 R_n`: :math:`\lambda_a^\top f_{yp}(y_{n+1-a})`."""
        field = self._system.rhs.derivatives().state_param
        if field is None:
            raise ValueError("the residual has no second derivatives")
        weighted = self._weighted(window, adj, a)
        block = None if weighted is None else field(*weighted)
        return BlockOperator([[block]], self._param_block_space(), self._lag_space(a))

    def _param_param(
        self, window: StepWindow[V], adj: BlockVector[V]
    ) -> StepOperator[V]:
        r""":math:`\mu^\top \partial^2 R_n/\partial p^2 = \sum_l
        \lambda_l^\top f_{pp}(y_{n+1-l})`, less the pinned rows'
        :math:`\sum_{i \in R} \mu_i\, \partial^2 g_i/\partial p^2` at
        :math:`t_{n+1}`."""
        field = self._system.rhs.derivatives().param_param
        if field is None:
            raise ValueError("the residual has no second derivatives")
        params = self._param_block_space()
        terms: List[tuple[float, DerivativeOperator[V]]] = [
            (1.0, field(*weighted))
            for lag in range(len(window.past_states) + 1)
            if (weighted := self._weighted(window, adj, lag)) is not None
        ]
        data = (
            None
            if self._pin is None
            else self._pin.data_curvature(
                adj.blocks[0], window.lag_time(0), self._system.space
            )
        )
        if data is not None:
            terms.append((-1.0, data))
        return BlockOperator([[ScaledSum(terms, params.component(0))]], params, params)

    def derivatives(self) -> StepperDerivatives[V]:
        """The step's derivatives, as far as the residual's bundle goes:
        the coupling to the stored states always, the parameters with the
        bundle's ``param``, the second derivatives with its second order.
        Valid for the parameters set when fetched."""
        bundle = self._system.rhs.derivatives()
        if bundle.param is None:
            return StepperDerivatives.state_only(self._off_diag)
        if bundle.state_state is None:
            return StepperDerivatives.first_order(self._off_diag, self._param)
        # Only lags passed through f carry second derivatives, each its own
        # (l, l) and mixed blocks.
        curved = self._method.curved_lags()
        return StepperDerivatives.second_order(
            self._off_diag,
            self._param,
            self._state_state,
            self._state_param,
            self._param_param,
            frozenset((lag, lag) for lag in curved),
            curved,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._method.name})"
