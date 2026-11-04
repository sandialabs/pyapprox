r"""Finite-difference check of a functional's terms, on any stack.

The terms of :mod:`pyapprox.ode.trajectory_terms` carry their
derivatives in bundles the sweeps read; this checks each bundle against
the term's own value, so a term's value and derivatives cannot drift
apart. Each check varies one argument where the term is read and
compares a declared derivative with finite differences of a quantity:

- a parameter term :math:`r(p)`: ``wrt_param``, the value in the
  parameters; ``param_param``, :math:`(\partial r/\partial p)^\top
  \omega` in the parameters;
- a state term :math:`g(y)` at one stored state: ``wrt_state`` and
  ``wrt_param``, the value in the state and in the parameters;
  ``state_state`` and ``state_param``, :math:`(\partial g/\partial
  y)^\top \omega` in the state and in the parameters; ``param_state``
  and ``param_param``, :math:`(\partial g/\partial p)^\top \omega` in
  the state and in the parameters;
- a step term :math:`c_n` at one step's window: ``wrt_unknown``,
  ``wrt_lag[l]`` and ``wrt_param``, the value in the unknown, in
  :math:`y_{n+1-l}` and in the parameters; ``curvature_wrt_unknown``,
  ``curvature_wrt_lag[l]`` and ``curvature_wrt_param``, the matching
  transposed derivative applied to :math:`\omega`, in all of the step's
  arguments at once,

with :math:`\omega` the weights of the term's values. A first derivative
returning None, or a parameter derivative left out of the bundle, is
compared with zero, so a dependence the bundle does not declare fails. A
step term's curvatures are derivatives along a tangent of all its
arguments together (the sweeps' form), so they are checked along one
joint direction. Every linearization also gets a dot test against the
weights, the only check of its transpose no finite difference reaches.

Each finite-difference check keeps its error curve, judged by the
report's :meth:`~pyapprox.ode.residual_derivatives_check.
BundleCheckReport.failures` with the caller's tolerances.

Everything goes through the stepper's spaces, so the checker works on
any stack. The caller supplies random vectors (``sample``) and, for
parameter derivatives, how to set the parameters the terms read
(``set_param``).
"""

from dataclasses import replace
from typing import Callable, Dict, Generic, Optional, Sequence, TypeVar

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    FDCurve,
)
from pyapprox.ode._space_checks import (
    DEFAULT_FD_STEPS,
    SpaceSampler,
    space_dot_test_error,
    space_fd_check,
)
from pyapprox.ode.protocols import DifferentiableStepperProtocol
from pyapprox.ode.residual_derivatives_check import BundleCheckReport
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_store import TrajectoryStoreProtocol
from pyapprox.ode.trajectory_terms import (
    LinearizationProtocol,
    ParameterTermProtocol,
    StateCurvatureFn,
    StateTermDerivatives,
    StateTermProtocol,
    StepTermProtocol,
    TrajectoryFunctional,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.backend_dense import BackendVectorSpace, backend_dtype_name
from pyapprox.util.linalg.operators import (
    BlockInnerProductSpace,
    BlockVector,
    SpaceDescriptor,
)

V = TypeVar("V")
X = TypeVar("X")
W = TypeVar("W")
U = TypeVar("U")
A = TypeVar("A", bound=ArrayProtocol)
T = TypeVar("T")


def _one(vector: W) -> BlockVector[W]:
    return BlockVector((vector,))


def _present(field: Optional[T], name: str) -> T:
    if field is None:
        raise ValueError(
            f"a bundle fetched after set_param lacks '{name}', which the first "
            "fetch had; which fields are present must not depend on the "
            "parameter values"
        )
    return field


class _Results:
    """The finite-difference curves and dot-test errors gathered for one
    report, each under a prefixed name."""

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix
        self.fd_checks: Dict[str, FDCurve] = {}
        self.transpose_errors: Dict[str, float] = {}

    def name(self, name: str) -> str:
        return f"{self._prefix}{name}"

    def report(self) -> BundleCheckReport:
        return BundleCheckReport(self.fd_checks, self.transpose_errors)

    def merge_into(self, other: "_Results") -> None:
        other.fd_checks.update(self.fd_checks)
        other.transpose_errors.update(self.transpose_errors)


class TrajectoryTermChecker(Generic[V, A]):
    r"""Check the derivatives of a functional's terms by finite
    differences.

    Parameters
    ----------
    stepper : DifferentiableStepperProtocol
        The stepper of the trajectories the terms read: its unknown, state
        and parameter spaces are the terms' arguments.
    bkd : Backend
        The backend of the terms' values and weights.
    sample : callable
        ``(descriptor, seed) ->`` a random single-column vector of the
        component space ``descriptor`` describes.
    set_param : callable, optional
        ``p ->`` sets the parameters every term reads, from a single
        column of the stepper's parameter space; required to check
        parameter derivatives. The check leaves the parameters at the
        point checked.
    fd_steps : sequence of float, optional
        The finite-difference steps; default ``1e-1`` down to ``1e-13``.
    central : bool, default False
        Central rather than forward differences.
    seed : int, default 0
        First seed passed to ``sample``.

    Raises
    ------
    TypeError
        If ``stepper`` does not satisfy
        :class:`DifferentiableStepperProtocol`.
    """

    def __init__(
        self,
        stepper: DifferentiableStepperProtocol[V, A],
        bkd: Backend[A],
        sample: Callable[[SpaceDescriptor, int], V],
        set_param: Optional[Callable[[V], None]] = None,
        fd_steps: Sequence[float] = DEFAULT_FD_STEPS,
        central: bool = False,
        seed: int = 0,
    ) -> None:
        if not isinstance(stepper, DifferentiableStepperProtocol):
            raise TypeError(
                "stepper must satisfy DifferentiableStepperProtocol, got "
                f"{type(stepper).__name__}"
            )
        self._stepper = stepper
        self._bkd = bkd
        self._sample = sample
        self._set_param = set_param
        self._fd_steps = tuple(fd_steps)
        self._central = central
        self._seed = seed
        self._sampler = SpaceSampler(sample, seed)

    # ------------------------------------------------------------------
    # Public checks
    # ------------------------------------------------------------------

    def check(
        self,
        functional: TrajectoryFunctional[V, A],
        store: TrajectoryStoreProtocol[V],
        weights: Optional[A] = None,
        param: Optional[V] = None,
    ) -> BundleCheckReport:
        """Check every term of ``functional`` everywhere it reads the
        trajectory in ``store``: each parameter term once, each state term
        at every stored state it reads, each step term at every step's
        window. Names are prefixed by the term and the place, e.g.
        ``"state_terms[0]@10.wrt_state"``.

        Parameters
        ----------
        functional : TrajectoryFunctional
            The functional.
        store : TrajectoryStoreProtocol
            The recorded trajectory, at ``param``.
        weights : A, optional
            The weights of the functional's values, as
            :meth:`TrajectoryFunctional.weights` takes them.
        param : V, optional
            The parameters; required when a term has derivatives that need
            them. The terms are left at these.
        """
        omega = functional.weights(weights)
        results = _Results()
        self._sampler = SpaceSampler(self._sample, self._seed)
        try:
            for i, parameter_term in enumerate(functional.parameter_terms):
                self._parameter_term(
                    parameter_term, omega, param, _Results(f"parameter_terms[{i}].")
                ).merge_into(results)
            ntimes = store.ntimes()
            for i, state_term in enumerate(functional.state_terms):
                for index in range(ntimes):
                    if not state_term.reads(index, index == ntimes - 1):
                        continue
                    self._state_term(
                        state_term,
                        index,
                        store.time(index),
                        store.state(index),
                        omega,
                        param,
                        _Results(f"state_terms[{i}]@{index}."),
                    ).merge_into(results)
            depth = self._stepper.depth()
            for i, step_term in enumerate(functional.step_terms):
                for step in range(store.nsteps()):
                    window = store.window(step, min(depth, step + 1))
                    self._step_term(
                        step_term,
                        step,
                        window,
                        omega,
                        param,
                        _Results(f"step_terms[{i}]@{step}."),
                    ).merge_into(results)
        finally:
            self._at(param)
        return results.report()

    def check_parameter_term(
        self,
        term: ParameterTermProtocol[V, A],
        weights: A,
        param: Optional[V] = None,
    ) -> BundleCheckReport:
        """Check one parameter term at ``param``.

        Raises
        ------
        TypeError
            If ``term`` does not satisfy :class:`ParameterTermProtocol`.
        ValueError
            If the term has derivatives and ``param`` or ``set_param`` is
            missing.
        """
        if not isinstance(term, ParameterTermProtocol):
            raise TypeError(
                "term must satisfy ParameterTermProtocol, got "
                f"{type(term).__name__}"
            )
        self._sampler = SpaceSampler(self._sample, self._seed)
        try:
            return self._parameter_term(term, weights, param, _Results()).report()
        finally:
            self._at(param)

    def check_state_term(
        self,
        term: StateTermProtocol[V, A],
        index: int,
        time: float,
        state: V,
        weights: A,
        param: Optional[V] = None,
    ) -> BundleCheckReport:
        """Check one state term at one stored state.

        Raises
        ------
        TypeError
            If ``term`` does not satisfy :class:`StateTermProtocol`.
        ValueError
            If the term has second derivatives and ``param`` or
            ``set_param`` is missing.
        """
        if not isinstance(term, StateTermProtocol):
            raise TypeError(
                f"term must satisfy StateTermProtocol, got {type(term).__name__}"
            )
        self._sampler = SpaceSampler(self._sample, self._seed)
        try:
            return self._state_term(
                term, index, time, state, weights, param, _Results()
            ).report()
        finally:
            self._at(param)

    def check_step_term(
        self,
        term: StepTermProtocol[V, A],
        step: int,
        window: StepWindow[V],
        weights: A,
        param: Optional[V] = None,
    ) -> BundleCheckReport:
        """Check one step term at one step's window.

        Raises
        ------
        TypeError
            If ``term`` does not satisfy :class:`StepTermProtocol`.
        ValueError
            If the term has curvatures and ``param`` or ``set_param`` is
            missing.
        """
        if not isinstance(term, StepTermProtocol):
            raise TypeError(
                f"term must satisfy StepTermProtocol, got {type(term).__name__}"
            )
        self._sampler = SpaceSampler(self._sample, self._seed)
        try:
            return self._step_term(
                term, step, window, weights, param, _Results()
            ).report()
        finally:
            self._at(param)

    # ------------------------------------------------------------------
    # Spaces, parameters and the two kinds of check
    # ------------------------------------------------------------------

    def _values(self, nqoi: int) -> BlockInnerProductSpace[A, A]:
        """The values of a term of ``nqoi`` entries, as one-block vectors."""
        descriptor = SpaceDescriptor("value", nqoi, backend_dtype_name(self._bkd))
        return BlockInnerProductSpace(
            [BackendVectorSpace(descriptor, self._bkd)], "values"
        )

    def _param_space(self) -> BlockInnerProductSpace[V, A]:
        space = self._stepper.param_space()
        if space is None:
            raise ValueError("the term has parameter derivatives but no param_space")
        return space

    def _at(self, param: Optional[V]) -> None:
        """Set ``param`` when given: every quantity sets the parameters it
        is evaluated at, since a term reads those currently set."""
        if param is not None and self._set_param is not None:
            self._set_param(param)

    def _require_param(self, param: Optional[V], what: str) -> V:
        if param is None or self._set_param is None:
            raise ValueError(f"the term has {what}; pass param and set_param")
        return param

    def _fd(
        self,
        results: _Results,
        name: str,
        point: BlockVector[V],
        space: BlockInnerProductSpace[V, A],
        output: BlockInnerProductSpace[U, A],
        quantity: Callable[[BlockVector[V]], BlockVector[U]],
        action: Callable[[BlockVector[V], BlockVector[V]], BlockVector[U]],
    ) -> None:
        results.fd_checks[results.name(name)] = space_fd_check(
            point,
            space,
            output,
            quantity,
            action,
            self._sampler.random(space),
            self._fd_steps,
            self._central,
        )

    def _dot(
        self,
        results: _Results,
        name: str,
        linearization: LinearizationProtocol[X, A],
        wrap: Callable[[X], BlockVector[V]],
        unwrap: Callable[[BlockVector[V]], X],
        domain: BlockInnerProductSpace[V, A],
        weights: A,
    ) -> None:
        """The dot test of one linearization against the weights."""
        values = self._values(int(weights.shape[0]))

        def apply(x: BlockVector[V]) -> BlockVector[A]:
            return _one(linearization.apply(unwrap(x)))

        def apply_transpose(w: BlockVector[A]) -> BlockVector[V]:
            return wrap(linearization.apply_transpose(w.blocks[0]))

        results.transpose_errors[results.name(name)] = space_dot_test_error(
            apply,
            apply_transpose,
            self._sampler.random(domain),
            _one(weights),
            domain,
            values,
        )

    # ------------------------------------------------------------------
    # Parameter terms
    # ------------------------------------------------------------------

    def _parameter_term(
        self,
        term: ParameterTermProtocol[V, A],
        weights: A,
        param: Optional[V],
        results: _Results,
    ) -> "_Results":
        derivs = term.derivatives()
        if derivs.wrt_param is None:
            return results
        p0 = self._require_param(param, "a parameter derivative")
        params, values = self._param_space(), self._values(term.nqoi())

        def wrt_param(p: V) -> LinearizationProtocol[V, A]:
            self._at(p)
            return _present(term.derivatives().wrt_param, "wrt_param")()

        def value(x: BlockVector[V]) -> BlockVector[A]:
            self._at(x.blocks[0])
            return _one(term.value())

        def first(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[A]:
            return _one(wrt_param(x.blocks[0]).apply(d.blocks[0]))

        self._fd(results, "wrt_param", _one(p0), params, values, value, first)
        self._dot(
            results, "wrt_param", wrt_param(p0), _one, _first_block, params, weights
        )
        if derivs.param_param is None:
            return results

        def gradient(x: BlockVector[V]) -> BlockVector[V]:
            return _one(wrt_param(x.blocks[0]).apply_transpose(weights))

        def second(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[V]:
            self._at(x.blocks[0])
            field = _present(term.derivatives().param_param, "param_param")
            result = field(weights, d.blocks[0])
            return params.zeros() if result is None else _one(result)

        self._fd(results, "param_param", _one(p0), params, params, gradient, second)
        return results

    # ------------------------------------------------------------------
    # State terms
    # ------------------------------------------------------------------

    def _state_term(
        self,
        term: StateTermProtocol[V, A],
        index: int,
        time: float,
        state: V,
        weights: A,
        param: Optional[V],
        results: _Results,
    ) -> "_Results":
        derivs = term.derivatives()
        if derivs.wrt_state is None:
            return results
        second_order = derivs.state_state is not None
        if second_order:
            self._require_param(param, "second derivatives")
        if param is not None and self._set_param is None:
            raise ValueError("param was given without set_param")
        states, values = self._stepper.state_space(), self._values(term.nqoi())

        def wrt_state(y: V, p: Optional[V]) -> LinearizationProtocol[V, A]:
            self._at(p)
            field = _present(term.derivatives().wrt_state, "wrt_state")
            return field(index, time, y)

        def wrt_param(y: V, p: Optional[V]) -> Optional[LinearizationProtocol[V, A]]:
            self._at(p)
            field = term.derivatives().wrt_param
            return None if field is None else field(index, time, y)

        def value_in_state(x: BlockVector[V]) -> BlockVector[A]:
            self._at(param)
            return _one(term.value(index, time, x.blocks[0]))

        def first_in_state(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[A]:
            return _one(wrt_state(x.blocks[0], param).apply(d.blocks[0]))

        self._fd(
            results,
            "wrt_state",
            _one(state),
            states,
            values,
            value_in_state,
            first_in_state,
        )
        self._dot(
            results,
            "wrt_state",
            wrt_state(state, param),
            _one,
            _first_block,
            states,
            weights,
        )
        if param is None:
            return results
        params = self._param_space()

        def value_in_param(x: BlockVector[V]) -> BlockVector[A]:
            self._at(x.blocks[0])
            return _one(term.value(index, time, state))

        def first_in_param(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[A]:
            linearization = wrt_param(state, x.blocks[0])
            if linearization is None:
                return values.zeros()
            return _one(linearization.apply(d.blocks[0]))

        self._fd(
            results,
            "wrt_param",
            _one(param),
            params,
            values,
            value_in_param,
            first_in_param,
        )
        at_param = wrt_param(state, param)
        if at_param is not None:
            self._dot(
                results, "wrt_param", at_param, _one, _first_block, params, weights
            )
        if not second_order:
            return results

        def state_gradient(y: V, p: V) -> BlockVector[V]:
            return _one(wrt_state(y, p).apply_transpose(weights))

        def param_gradient(y: V, p: V) -> BlockVector[V]:
            linearization = wrt_param(y, p)
            if linearization is None:
                return params.zeros()
            return _one(linearization.apply_transpose(weights))

        p0 = param

        def second(
            name: str,
            select: Callable[
                [StateTermDerivatives[V, A]], Optional[StateCurvatureFn[V, A]]
            ],
            in_state: bool,
            gradient: Callable[[V, V], BlockVector[V]],
            output: BlockInnerProductSpace[V, A],
        ) -> None:
            """One second-derivative block: ``gradient`` varied in the
            state or in the parameters, against the selected field."""

            def block(y: V, p: V, x: V) -> BlockVector[V]:
                self._at(p)
                result = _present(select(term.derivatives()), name)(
                    index, time, y, weights, x
                )
                return output.zeros() if result is None else _one(result)

            if in_state:

                def quantity_y(x: BlockVector[V]) -> BlockVector[V]:
                    return gradient(x.blocks[0], p0)

                def action_y(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[V]:
                    return block(x.blocks[0], p0, d.blocks[0])

                self._fd(
                    results, name, _one(state), states, output, quantity_y, action_y
                )
                return

            def quantity_p(x: BlockVector[V]) -> BlockVector[V]:
                return gradient(state, x.blocks[0])

            def action_p(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[V]:
                return block(state, x.blocks[0], d.blocks[0])

            self._fd(results, name, _one(p0), params, output, quantity_p, action_p)

        second("state_state", _state_state, True, state_gradient, states)
        second("state_param", _state_param, False, state_gradient, states)
        second("param_state", _param_state, True, param_gradient, params)
        second("param_param", _param_param, False, param_gradient, params)
        return results

    # ------------------------------------------------------------------
    # Step terms
    # ------------------------------------------------------------------

    def _step_term(
        self,
        term: StepTermProtocol[V, A],
        step: int,
        window: StepWindow[V],
        weights: A,
        param: Optional[V],
        results: _Results,
    ) -> "_Results":
        derivs = term.derivatives()
        if derivs.wrt_unknown is None:
            return results
        second_order = derivs.curvature_wrt_unknown is not None
        if second_order:
            self._require_param(param, "curvatures")
        if param is not None and self._set_param is None:
            raise ValueError("param was given without set_param")
        unknowns, values = self._stepper.unknown_space(), self._values(term.nqoi())
        nlags = len(window.past_states)

        # First derivatives: the value in the unknown, each lag, and the
        # parameters, the others held at the window and ``param``.
        def value_unknown(x: BlockVector[V]) -> BlockVector[A]:
            self._at(param)
            return _one(term.value(step, replace(window, unknown=x)))

        def first_unknown(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[A]:
            self._at(param)
            field = _present(term.derivatives().wrt_unknown, "wrt_unknown")
            lin = field(step, replace(window, unknown=x))
            return values.zeros() if lin is None else _one(lin.apply(d))

        self._fd(
            results,
            "wrt_unknown",
            window.unknown,
            unknowns,
            values,
            value_unknown,
            first_unknown,
        )
        self._at(param)
        at_unknown = _present(term.derivatives().wrt_unknown, "wrt_unknown")(
            step, window
        )
        if at_unknown is not None:
            self._dot(
                results,
                "wrt_unknown",
                at_unknown,
                _same,
                _same,
                unknowns,
                weights,
            )
        for lag in range(1, nlags + 1):
            self._lag_first(results, term, step, window, lag, weights, param)
        if param is not None:
            self._param_first(results, term, step, window, weights, param)
        if second_order:
            self._curvatures(
                results,
                term,
                step,
                window,
                weights,
                self._require_param(param, "curvatures"),
            )
        return results

    def _lag_first(
        self,
        results: _Results,
        term: StepTermProtocol[V, A],
        step: int,
        window: StepWindow[V],
        lag: int,
        weights: A,
        param: Optional[V],
    ) -> None:
        """``wrt_lag[lag]``: the value in :math:`y_{n+1-l}`, and its dot
        test."""
        states, values = self._stepper.state_space(), self._values(term.nqoi())

        def placed(y: V) -> StepWindow[V]:
            past = list(window.past_states)
            past[lag - 1] = y
            return replace(window, past_states=tuple(past))

        def lag_linearization(y: V) -> Optional[LinearizationProtocol[V, A]]:
            self._at(param)
            field = _present(term.derivatives().wrt_lag, "wrt_lag")
            return field(step, placed(y), lag)

        def value(x: BlockVector[V]) -> BlockVector[A]:
            self._at(param)
            return _one(term.value(step, placed(x.blocks[0])))

        def first(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[A]:
            lin = lag_linearization(x.blocks[0])
            return values.zeros() if lin is None else _one(lin.apply(d.blocks[0]))

        name = f"wrt_lag[{lag}]"
        point = _one(window.past_state(lag))
        self._fd(results, name, point, states, values, value, first)
        at_window = lag_linearization(window.past_state(lag))
        if at_window is not None:
            self._dot(results, name, at_window, _one, _first_block, states, weights)

    def _param_first(
        self,
        results: _Results,
        term: StepTermProtocol[V, A],
        step: int,
        window: StepWindow[V],
        weights: A,
        param: V,
    ) -> None:
        """``wrt_param``: the value in the parameters, and its dot test."""
        params, values = self._param_space(), self._values(term.nqoi())

        def param_linearization(p: V) -> Optional[LinearizationProtocol[V, A]]:
            self._at(p)
            return _present(term.derivatives().wrt_param, "wrt_param")(step, window)

        def value(x: BlockVector[V]) -> BlockVector[A]:
            self._at(x.blocks[0])
            return _one(term.value(step, window))

        def first(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[A]:
            lin = param_linearization(x.blocks[0])
            return values.zeros() if lin is None else _one(lin.apply(d.blocks[0]))

        self._fd(results, "wrt_param", _one(param), params, values, value, first)
        at_param = param_linearization(param)
        if at_param is not None:
            self._dot(
                results, "wrt_param", at_param, _one, _first_block, params, weights
            )

    def _curvatures(
        self,
        results: _Results,
        term: StepTermProtocol[V, A],
        step: int,
        window: StepWindow[V],
        weights: A,
        param: V,
    ) -> None:
        r"""The three curvatures, each the derivative of a transposed first
        derivative along one joint direction of the unknown, the lags and
        the parameters: the direction's unknown and lag blocks are the
        tangents window, its last block the parameter direction."""
        unknowns, states = self._stepper.unknown_space(), self._stepper.state_space()
        params = self._param_space()
        nunknown, nlags = unknowns.nblocks(), len(window.past_states)
        joint: BlockInnerProductSpace[V, A] = BlockInnerProductSpace(
            [unknowns.component(i) for i in range(nunknown)]
            + [states.component(0)] * nlags
            + [params.component(0)],
            "step arguments",
        )
        point = BlockVector(
            window.unknown.blocks + tuple(window.past_states) + (param,)
        )

        def placed(x: BlockVector[V]) -> StepWindow[V]:
            """The window at ``x``, its parameters set."""
            self._at(x.blocks[-1])
            return replace(
                window,
                unknown=BlockVector(x.blocks[:nunknown]),
                past_states=tuple(x.blocks[nunknown : nunknown + nlags]),
            )

        def tangents(d: BlockVector[V]) -> StepWindow[V]:
            return replace(
                window,
                unknown=BlockVector(d.blocks[:nunknown]),
                past_states=tuple(d.blocks[nunknown : nunknown + nlags]),
            )

        def unknown_gradient(x: BlockVector[V]) -> BlockVector[V]:
            at = placed(x)
            field = _present(term.derivatives().wrt_unknown, "wrt_unknown")
            lin = field(step, at)
            return unknowns.zeros() if lin is None else lin.apply_transpose(weights)

        def unknown_curvature(
            x: BlockVector[V], d: BlockVector[V]
        ) -> BlockVector[V]:
            at = placed(x)
            field = _present(
                term.derivatives().curvature_wrt_unknown, "curvature_wrt_unknown"
            )
            result = field(step, at, tangents(d), d.blocks[-1], weights)
            return unknowns.zeros() if result is None else result

        self._fd(
            results,
            "curvature_wrt_unknown",
            point,
            joint,
            unknowns,
            unknown_gradient,
            unknown_curvature,
        )
        for lag in range(1, nlags + 1):
            self._fd(
                results,
                f"curvature_wrt_lag[{lag}]",
                point,
                joint,
                states,
                *self._lag_curvature(term, step, lag, weights, placed, tangents),
            )

        def param_gradient(x: BlockVector[V]) -> BlockVector[V]:
            at = placed(x)
            lin = _present(term.derivatives().wrt_param, "wrt_param")(step, at)
            return params.zeros() if lin is None else _one(lin.apply_transpose(weights))

        def param_curvature(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[V]:
            at = placed(x)
            field = _present(
                term.derivatives().curvature_wrt_param, "curvature_wrt_param"
            )
            result = field(step, at, tangents(d), d.blocks[-1], weights)
            return params.zeros() if result is None else _one(result)

        self._fd(
            results,
            "curvature_wrt_param",
            point,
            joint,
            params,
            param_gradient,
            param_curvature,
        )

    def _lag_curvature(
        self,
        term: StepTermProtocol[V, A],
        step: int,
        lag: int,
        weights: A,
        placed: Callable[[BlockVector[V]], StepWindow[V]],
        tangents: Callable[[BlockVector[V]], StepWindow[V]],
    ) -> tuple[
        Callable[[BlockVector[V]], BlockVector[V]],
        Callable[[BlockVector[V], BlockVector[V]], BlockVector[V]],
    ]:
        """The quantity and action of ``curvature_wrt_lag[lag]``."""
        states = self._stepper.state_space()

        def gradient(x: BlockVector[V]) -> BlockVector[V]:
            at = placed(x)
            lin = _present(term.derivatives().wrt_lag, "wrt_lag")(step, at, lag)
            return states.zeros() if lin is None else _one(lin.apply_transpose(weights))

        def curvature(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[V]:
            at = placed(x)
            field = _present(term.derivatives().curvature_wrt_lag, "curvature_wrt_lag")
            result = field(step, at, tangents(d), lag, d.blocks[-1], weights)
            return states.zeros() if result is None else _one(result)

        return gradient, curvature


def _state_state(
    bundle: StateTermDerivatives[V, A],
) -> Optional[StateCurvatureFn[V, A]]:
    return bundle.state_state


def _state_param(
    bundle: StateTermDerivatives[V, A],
) -> Optional[StateCurvatureFn[V, A]]:
    return bundle.state_param


def _param_state(
    bundle: StateTermDerivatives[V, A],
) -> Optional[StateCurvatureFn[V, A]]:
    return bundle.param_state


def _param_param(
    bundle: StateTermDerivatives[V, A],
) -> Optional[StateCurvatureFn[V, A]]:
    return bundle.param_param


def _first_block(x: BlockVector[W]) -> W:
    return x.blocks[0]


def _same(x: BlockVector[W]) -> BlockVector[W]:
    return x
