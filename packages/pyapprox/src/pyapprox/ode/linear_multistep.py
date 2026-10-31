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

from itertools import count
from typing import (
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
from pyapprox.ode.multistep_methods import (
    MultistepCoefficients,
    MultistepFormulaProtocol,
)
from pyapprox.ode.pinned_rows import BlockPinProtocol, RowPinningProtocol
from pyapprox.ode.stepper_derivatives import (
    StepOperator,
    StepperDerivatives,
    StepWindow,
    check_step_times,
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
    LinearizationPoint,
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
    """

    def __init__(
        self,
        stepper: "LinearMultistepStepper[V, A, F]",
        known: V,
        time: float,
        formula: MultistepCoefficients,
        deltat: float,
    ) -> None:
        self._stepper, self._known, self._time = stepper, known, time
        self._alpha0 = formula.alpha[0]
        self._scale = deltat * formula.beta[0]

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
        stepper = self._stepper
        operator = stepper.new_state_operator(x, self._time, self._alpha0, self._scale)
        pin = stepper.pin()
        return stepper.system().solver.setup(
            operator if pin is None else pin.own(operator)
        )

    def is_affine(self) -> bool:
        return False


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

    Raises
    ------
    TypeError
        If the method does not satisfy its protocol, is implicit and no
        nonlinear solver is given, or ``row_pinning`` does not satisfy its
        protocol.
    """

    def __init__(
        self,
        method: MultistepFormulaProtocol,
        system: MassFormSystem[V, A, F],
        nonlinear_solver: Optional[NonlinearSolverProtocol[V, A]] = None,
        row_pinning: Optional[RowPinningProtocol[V]] = None,
    ) -> None:
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
        self._system = system
        self._nonlinear_solver = nonlinear_solver
        space = system.space
        self._unknown_space = BlockInnerProductSpace([space], "unknown")
        self._residual_space = BlockInnerProductSpace([space], "residual")
        self._state_space = BlockInnerProductSpace([space], "state")
        self._mass_handle: Optional[SolverHandleProtocol[V]] = None
        # The one block, of one, is the step end.
        self._pin = None if row_pinning is None else row_pinning.at(0, 1)
        # Explicit steps' row-replaced alpha_0 M, set up once per alpha_0.
        self._pinned_mass_handles: Dict[float, SolverHandleProtocol[V]] = {}
        self._points = count()

    def method(self) -> MultistepFormulaProtocol:
        """The method."""
        return self._method

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
        rhs = self._system.rhs
        return rhs.parameters_version() if isinstance(rhs, _VersionedProtocol) else 0

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

    def new_state_operator(
        self, state: V, time: float, alpha0: float, scale: float
    ) -> ActionOperatorProtocol[V, F]:
        r""":math:`\alpha_0 M - s\, J(y, t)` with :math:`s = \Delta t\,
        \beta_0`, at a Newton iterate.

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
        """
        system = self._system
        jacobian = self._jacobian(
            state, time, self._point(True, self.parameters_version())
        )
        return ScaledSum([(alpha0, system.mass), (-scale, jacobian)], system.space)

    def _mass_solve(self, rhs: V) -> V:
        """:math:`M^{-1}` ``rhs``, with the mass set up once."""
        if self._mass_handle is None:
            self._mass_handle = self._system.solver.setup(self._system.mass)
        solution, info = self._mass_handle.solve(rhs)
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
        handle = self._pinned_mass_handles.get(alpha0)
        if handle is None:
            scaled = ScaledSum([(alpha0, self._system.mass)], self._system.space)
            handle = self._system.solver.setup(pin.own(scaled))
            self._pinned_mass_handles[alpha0] = handle
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

    def _formula(self, times: Sequence[float]) -> MultistepCoefficients:
        """The method's formula for a step with these times, checked to
        have exactly one coefficient pair per lag."""
        formula = self._method.coefficients_at(times)
        if len(formula.alpha) != len(times):
            raise ValueError(
                f"{self._method.name} gave {len(formula.alpha)} coefficients for "
                f"a step with {len(times)} lags"
            )
        return formula

    def _known(
        self,
        past_states: Sequence[V],
        times: Sequence[float],
        formula: MultistepCoefficients,
    ) -> V:
        r""":math:`k = \sum_{l \ge 1} (\alpha_l M y_{n+1-l} - \Delta t\,
        \beta_l f(y_{n+1-l}, t_{n+1-l}))`, with :math:`\Delta t = t_{n+1} -
        t_n`."""
        space, mass = self._system.space, self._system.mass
        deltat = times[0] - times[1]
        known = space.zeros()
        for lag, state in enumerate(past_states, start=1):
            alpha, beta = formula.alpha[lag], formula.beta[lag]
            if alpha != 0.0:
                known = space.axpby(alpha, mass.apply(state), 1.0, known, out=known)
            if beta != 0.0:
                slope = self.rhs(state, times[lag])
                known = space.axpby(-deltat * beta, slope, 1.0, known, out=known)
        return known

    def step(
        self, past_states: Sequence[V], times: Sequence[float]
    ) -> BlockVector[V]:
        r"""Solve step :math:`n` for :math:`y_{n+1}`.

        Parameters
        ----------
        past_states : sequence of V
            :math:`(y_n, y_{n-1}, \dots)`, single columns of the state
            space: the method's depth of them, or fewer at a startup
            step.
        times : sequence of float
            :math:`(t_{n+1}, t_n, t_{n-1}, \dots)`, one per lag.

        Returns
        -------
        BlockVector
            The converged unknown :math:`u_n = y_{n+1}`.

        Raises
        ------
        ValueError
            If the number of states is not between 1 and the depth, the
            times do not match them, or the method's coefficients do not
            cover these step sizes.
        SolveFailedError
            If the nonlinear solve or a mass solve fails.
        """
        check_step_times(times, len(past_states))
        formula = self._formula(times)
        space = self._system.space
        known = self._known(past_states, times, formula)
        new_time, deltat = times[0], times[0] - times[1]
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
        system = _NewStateSystem(self, known, new_time, formula, deltat)
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
        return self._formula(window.times)

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
        known = self._known(window.past_states, window.times, formula)
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
        """A solver handle for :meth:`diag`, set up by the system's
        solver."""
        # alpha_0 != 0, so the diagonal block is never zero.
        block = self._lag_block(window, 0)
        if block is None:
            raise AssertionError("alpha_0 is nonzero")
        operator = BlockOperator([[block]], self._unknown_space, self._residual_space)
        return BlockLowerTriangularHandle(operator, [self._system.solver.setup(block)])

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
        block = ScaledSum(terms, self._system.space, params.component(0))
        # The step-end rows do not depend on the parameters.
        pinned = block if self._pin is None else self._pin.coupled(block)
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
        \lambda_l^\top f_{pp}(y_{n+1-l})`."""
        field = self._system.rhs.derivatives().param_param
        if field is None:
            raise ValueError("the residual has no second derivatives")
        params = self._param_block_space()
        terms = [
            (1.0, field(*weighted))
            for lag in range(len(window.past_states) + 1)
            if (weighted := self._weighted(window, adj, lag)) is not None
        ]
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
