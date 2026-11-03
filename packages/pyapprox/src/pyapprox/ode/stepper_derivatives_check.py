r"""Finite-difference check of a stepper's step-equation derivatives, on
any stack.

The stepper counterpart of
:class:`~pyapprox.ode.residual_derivatives_check.ODEResidualBundleChecker`.
A step's residual :math:`R_n(u_n;\, y_n, \dots, y_{n+1-k};\, p)` has three
kinds of argument: the step's own unknown :math:`u_n` (lag 0), the past
stored states (lag :math:`k \ge 1`) and the parameters. Each check varies
one argument at a :class:`~pyapprox.ode.stepper_derivatives.StepWindow`
and compares a declared operator's action with finite differences of a
quantity:

==========================  ===============================  ==============
term                        quantity differentiated          in argument
==========================  ===============================  ==============
``diag``                    :math:`R_n`                      lag 0
``off_diag[k]``             :math:`R_n`                      lag :math:`k`
``param``                   :math:`R_n`                      parameters
``state_state[a,b]``        :math:`D_a R_n^\top \mu`         lag :math:`b`
``state_param[a]``          :math:`D_a R_n^\top \mu`         parameters
``param_state[a]``          :math:`D_p R_n^\top \mu`         lag :math:`a`
``param_param``             :math:`D_p R_n^\top \mu`         parameters
==========================  ===============================  ==============

where :math:`D_a` is the derivative in the lag-:math:`a` argument,
:math:`D_p` the one in the parameters and :math:`\mu` the step adjoint.
Every pair of lags is checked; a pair the bundle does not declare
nonzero is compared with zero, so a missing declaration fails.
``param_state[a]`` checks the transpose of the declared
``state_param[a]``, so the two mixed derivatives must agree.

Every operator also gets a dot test, the readout included, and the
diagonal block's solver handle is checked to invert the diagonal block,
transposed too.

Everything goes through the stepper's spaces (``axpby``, ``norm``,
``dot``), so the checker works on any stack. The caller supplies random
vectors (``sample``) and, for parameter derivatives, how to set the
parameters from a parameter vector (``set_param``): neither is something
a vector space provides.
"""

from dataclasses import dataclass, replace
from typing import Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    FDCurve,
)
from pyapprox.ode._space_checks import (
    DEFAULT_FD_STEPS,
    SpaceSampler,
    scalar,
    space_dot_test_error,
    space_fd_check,
)
from pyapprox.ode.protocols import DifferentiableStepperProtocol
from pyapprox.ode.residual_derivatives_check import BundleCheckReport
from pyapprox.ode.stepper_derivatives import (
    StepOperator,
    StepperDerivatives,
    StepWindow,
)
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    BlockInnerProductSpace,
    BlockVector,
    SpaceDescriptor,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)

_Point = Tuple[StepWindow[V], StepperDerivatives[V]]
"""A window and the bundle fetched for the parameters it was taken at."""

_OperatorAt = Callable[[StepWindow[V], StepperDerivatives[V]], StepOperator[V]]
"""Picks one operator from a stepper at a window and its bundle."""


@dataclass(frozen=True)
class _Argument(Generic[V, A]):
    """One argument of the step residual that a check varies.

    Attributes
    ----------
    point : BlockVector
        The argument's value at the window being checked.
    space : BlockInnerProductSpace
        Its space; operators taking this argument act on it.
    place : callable
        ``x -> (window, bundle)``: sets the argument to ``x`` (in a copy
        of the window, or through ``set_param``) and returns the window
        and the bundle fetched for it.
    """

    point: BlockVector[V]
    space: BlockInnerProductSpace[V, A]
    place: Callable[[BlockVector[V]], _Point[V]]


@dataclass(frozen=True)
class _Term(Generic[V, A]):
    """One finite-difference check.

    Attributes
    ----------
    name : str
        The name in the report.
    arg : _Argument
        The argument varied.
    output : BlockInnerProductSpace
        The space of the quantity.
    quantity : callable
        ``x -> q(x)``, the quantity at the argument value ``x``.
    action : callable
        ``(x, d) ->`` the declared derivative of ``q`` at ``x`` applied to
        the direction ``d``.
    """

    name: str
    arg: _Argument[V, A]
    output: BlockInnerProductSpace[V, A]
    quantity: Callable[[BlockVector[V]], BlockVector[V]]
    action: Callable[[BlockVector[V], BlockVector[V]], BlockVector[V]]


class StepperBundleChecker(Generic[V, A]):
    r"""Check a stepper's step-equation derivatives by finite differences.

    Parameters
    ----------
    stepper : DifferentiableStepperProtocol
        The stepper.
    sample : callable
        ``(descriptor, seed) ->`` a random single-column vector of the
        component space ``descriptor`` describes; a block vector's random
        directions are drawn component by component.
    set_param : callable, optional
        ``p ->`` sets the residual's parameters from a single column of
        the stepper's parameter space; required when the bundle has
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
        self._sample = sample
        self._set_param = set_param
        self._fd_steps = tuple(fd_steps)
        self._central = central
        self._seed = seed

    def check(
        self,
        window: StepWindow[V],
        adj: Optional[BlockVector[V]] = None,
        param: Optional[V] = None,
    ) -> BundleCheckReport:
        """Check every term the stepper declares at one window.

        Parameters
        ----------
        window : StepWindow
            The step: its converged unknown, past states, time and size.
        adj : BlockVector, optional
            The step adjoint :math:`\\mu` the second derivatives contract
            with, a single column of the residual space; required when the
            bundle has second derivatives.
        param : V, optional
            The parameters, a single column of the stepper's parameter
            space's component; required when the bundle has parameter
            derivatives. The residual is left at these.

        Returns
        -------
        BundleCheckReport
            The finite-difference error of every term and the dot-test
            and handle errors of every operator.

        Raises
        ------
        ValueError
            If ``param``, ``set_param`` or ``adj`` is missing where the
            bundle needs it.
        """
        self._sampler = SpaceSampler(self._sample, self._seed)
        derivs = self._fetch(param)
        if derivs.param is not None and (param is None or self._set_param is None):
            raise ValueError(
                "the bundle has parameter derivatives; pass param and set_param"
            )
        if derivs.state_state is not None and adj is None:
            raise ValueError("the bundle has second derivatives; pass adj")
        # Terms that vary the parameters set them as they go. The finally
        # block restores the parameters checked at even if a term raises,
        # so the caller's residual is never left at a perturbed value.
        try:
            terms = self._first_order_terms(derivs, window, param)
            if adj is not None and param is not None:
                terms += self._second_order_terms(derivs, window, adj, param)
            fd_checks = {term.name: self._fd_check(term) for term in terms}
            dot_errors = self._transpose_errors(self._fetch(param), window, adj)
        finally:
            self._fetch(param)
        return BundleCheckReport(fd_checks, dot_errors)

    # ------------------------------------------------------------------
    # Random vectors, parameters and the arguments of the step residual
    # ------------------------------------------------------------------

    def _random(self, space: BlockInnerProductSpace[V, A]) -> BlockVector[V]:
        """A random single column of ``space``, component by component."""
        return self._sampler.random(space)

    def _fetch(self, param: Optional[V]) -> StepperDerivatives[V]:
        """Set ``param`` (when given) and fetch the stepper's bundle: a
        bundle is valid only for the parameters set when it was fetched.

        Parameters
        ----------
        param : V, optional
            The parameters to set first.

        Returns
        -------
        StepperDerivatives
            The bundle at those parameters.
        """
        if param is not None and self._set_param is not None:
            self._set_param(param)
        return self._stepper.derivatives()

    def _lag_space(self, lag: int) -> BlockInnerProductSpace[V, A]:
        """The space of the lag-``lag`` argument."""
        if lag == 0:
            return self._stepper.unknown_space()
        return self._stepper.state_space()

    def _param_space(self) -> BlockInnerProductSpace[V, A]:
        space = self._stepper.param_space()
        if space is None:
            raise ValueError("the stepper has parameter derivatives but no param_space")
        return space

    def _lag_argument(
        self, window: StepWindow[V], lag: int, param: Optional[V]
    ) -> _Argument[V, A]:
        """The lag-``lag`` argument: the unknown for 0, a past state (as a
        one-block vector) otherwise. Moving it copies the window; the
        parameters stay at ``param``.

        Parameters
        ----------
        window : StepWindow
            The window being checked.
        lag : int
            0 for the unknown, :math:`k \\ge 1` for :math:`y_{n+1-k}`.
        param : V, optional
            The parameters, held fixed.

        Returns
        -------
        _Argument
            The argument, its value and its space.
        """
        if lag == 0:

            def place_unknown(x: BlockVector[V]) -> _Point[V]:
                return replace(window, unknown=x), self._fetch(param)

            return _Argument(window.unknown, self._lag_space(0), place_unknown)

        def place_past(x: BlockVector[V]) -> _Point[V]:
            past = list(window.past_states)
            past[lag - 1] = x.blocks[0]
            return replace(window, past_states=tuple(past)), self._fetch(param)

        return _Argument(
            BlockVector((window.past_state(lag),)), self._lag_space(lag), place_past
        )

    def _param_argument(self, window: StepWindow[V], param: V) -> _Argument[V, A]:
        """The parameters as an argument (a one-block vector): moving them
        calls ``set_param``; the window stays as it is."""

        def place(p: BlockVector[V]) -> _Point[V]:
            return window, self._fetch(p.blocks[0])

        return _Argument(BlockVector((param,)), self._param_space(), place)

    def _first_derivative(self, lag: int) -> _OperatorAt[V]:
        """The derivative of :math:`R_n` in the lag-``lag`` argument:
        ``diag`` for 0, ``off_diag(lag)`` otherwise."""

        def operator(
            window: StepWindow[V], derivs: StepperDerivatives[V]
        ) -> StepOperator[V]:
            if lag == 0:
                return self._stepper.diag(window)
            if derivs.off_diag is None:
                raise ValueError("the bundle has no off_diag")
            return derivs.off_diag(window, lag)

        return operator

    @staticmethod
    def _param_derivative(
        window: StepWindow[V], derivs: StepperDerivatives[V]
    ) -> StepOperator[V]:
        """The derivative of :math:`R_n` in the parameters."""
        if derivs.param is None:
            raise ValueError("a bundle fetched after set_param lacks 'param'")
        return derivs.param(window)

    # ------------------------------------------------------------------
    # The terms
    # ------------------------------------------------------------------

    def _first_order_term(
        self, name: str, arg: _Argument[V, A], derivative: _OperatorAt[V]
    ) -> _Term[V, A]:
        """:math:`R_n` as a function of ``arg``, against ``derivative``,
        an operator from ``arg``'s space to the residual space."""

        def quantity(x: BlockVector[V]) -> BlockVector[V]:
            window, _ = arg.place(x)
            return self._stepper.step_residual(window)

        def action(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[V]:
            window, derivs = arg.place(x)
            return derivative(window, derivs).apply(d)

        return _Term(name, arg, self._stepper.residual_space(), quantity, action)

    def _second_order_term(
        self,
        name: str,
        arg: _Argument[V, A],
        first: Tuple[_OperatorAt[V], BlockInnerProductSpace[V, A]],
        declared: Optional[Tuple[_OperatorAt[V], bool]],
        adj: BlockVector[V],
    ) -> _Term[V, A]:
        r"""A first derivative transposed onto the step adjoint, as a
        function of ``arg``, against a declared second-derivative block.

        Parameters
        ----------
        name : str
            The term's name in the report.
        arg : _Argument
            The argument varied.
        first : (callable, BlockInnerProductSpace)
            ``(window, bundle) ->`` the first derivative :math:`D` whose
            :math:`D^\top \mu` is differentiated, and :math:`D`'s domain
            (where :math:`D^\top \mu` lives).
        declared : (callable, bool) or None
            ``(window, bundle) ->`` the declared block, and whether to
            apply its transpose (``param_state`` checks ``state_param``'s
            transpose); ``None`` for a block declared zero.
        adj : BlockVector
            :math:`\mu`, a single column of the residual space.
        """
        first_at, output = first

        def quantity(x: BlockVector[V]) -> BlockVector[V]:
            window, derivs = arg.place(x)
            return first_at(window, derivs).apply_transpose(adj)

        if declared is None:

            def zero(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[V]:
                return output.zeros()

            return _Term(name, arg, output, quantity, zero)

        block_at, transposed = declared

        def action(x: BlockVector[V], d: BlockVector[V]) -> BlockVector[V]:
            window, derivs = arg.place(x)
            block = block_at(window, derivs)
            return block.apply_transpose(d) if transposed else block.apply(d)

        return _Term(name, arg, output, quantity, action)

    def _first_order_terms(
        self,
        derivs: StepperDerivatives[V],
        window: StepWindow[V],
        param: Optional[V],
    ) -> List[_Term[V, A]]:
        """``diag``, ``off_diag[k]`` for every lag and, with parameters,
        ``param``."""
        # The lags the window read: a multistep startup step reads fewer
        # states than the stepper's depth.
        nlags = len(window.past_states) if derivs.off_diag is not None else 0
        terms = [
            self._first_order_term(
                "diag" if lag == 0 else f"off_diag[{lag}]",
                self._lag_argument(window, lag, param),
                self._first_derivative(lag),
            )
            for lag in range(nlags + 1)
        ]
        if derivs.param is not None and param is not None:
            terms.append(
                self._first_order_term(
                    "param", self._param_argument(window, param), self._param_derivative
                )
            )
        return terms

    def _second_order_terms(
        self,
        derivs: StepperDerivatives[V],
        window: StepWindow[V],
        adj: BlockVector[V],
        param: V,
    ) -> List[_Term[V, A]]:
        """Every second-derivative block, declared or not (see the table
        in the module docstring)."""
        lags = range(len(window.past_states) + 1)
        params = self._param_argument(window, param)
        lag_args = {lag: self._lag_argument(window, lag, param) for lag in lags}

        def state_state(a: int, b: int) -> _OperatorAt[V]:
            def block(w: StepWindow[V], d: StepperDerivatives[V]) -> StepOperator[V]:
                if d.state_state is None:
                    raise ValueError("the bundle lacks state_state")
                return d.state_state(w, adj, a, b)

            return block

        def state_param(a: int) -> _OperatorAt[V]:
            def block(w: StepWindow[V], d: StepperDerivatives[V]) -> StepOperator[V]:
                if d.state_param is None:
                    raise ValueError("the bundle lacks state_param")
                return d.state_param(w, adj, a)

            return block

        def param_param(w: StepWindow[V], d: StepperDerivatives[V]) -> StepOperator[V]:
            if d.param_param is None:
                raise ValueError("the bundle lacks param_param")
            return d.param_param(w, adj)

        terms: List[_Term[V, A]] = []
        for a in lags:
            first_a = (self._first_derivative(a), lag_args[a].space)
            # D_a R^T mu in every lag b: the (a, b) block of the state
            # Hessian, or zero when the pair is not declared.
            for b in lags:
                pair = (
                    (state_state(a, b), False) if (a, b) in derivs.state_pairs else None
                )
                terms.append(
                    self._second_order_term(
                        f"state_state[{a},{b}]", lag_args[b], first_a, pair, adj
                    )
                )
            declared = a in derivs.param_lags
            # D_a R^T mu in the parameters: the state_param block of lag a.
            terms.append(
                self._second_order_term(
                    f"state_param[{a}]",
                    params,
                    first_a,
                    (state_param(a), False) if declared else None,
                    adj,
                )
            )
            # D_p R^T mu in lag a: the same block, transposed.
            terms.append(
                self._second_order_term(
                    f"param_state[{a}]",
                    lag_args[a],
                    (self._param_derivative, params.space),
                    (state_param(a), True) if declared else None,
                    adj,
                )
            )
        # D_p R^T mu in the parameters.
        terms.append(
            self._second_order_term(
                "param_param",
                params,
                (self._param_derivative, params.space),
                (param_param, False),
                adj,
            )
        )
        return terms

    def _fd_check(self, term: _Term[V, A]) -> FDCurve:
        """The finite-difference errors of ``term`` over the steps, along
        one random unit direction, with their predicted rounding: relative
        to the action's norm, or absolute where the action is zero (a
        stated zero)."""
        arg = term.arg
        return space_fd_check(
            arg.point,
            arg.space,
            term.output,
            term.quantity,
            term.action,
            self._random(arg.space),
            self._fd_steps,
            self._central,
        )

    # ------------------------------------------------------------------
    # Dot tests and the diagonal handle
    # ------------------------------------------------------------------

    def _dot_test(
        self,
        op: StepOperator[V],
        domain: BlockInnerProductSpace[V, A],
        range_: BlockInnerProductSpace[V, A],
    ) -> float:
        r"""The relative difference of :math:`\langle w, A v\rangle` and
        :math:`\langle A^\top w, v\rangle` for random ``v`` and ``w``."""
        v, w = self._random(domain), self._random(range_)
        return space_dot_test_error(
            op.apply, op.apply_transpose, v, w, domain, range_
        )

    def _symmetry_error(
        self, op: StepOperator[V], space: BlockInnerProductSpace[V, A]
    ) -> float:
        """The relative difference of :math:`A v` and :math:`A^\\top v`."""
        v = self._random(space)
        av, atv = op.apply(v), op.apply_transpose(v)
        scale = max(scalar(space.norm(av)), scalar(space.norm(atv)))
        difference = scalar(space.norm(space.axpby(1.0, av, -1.0, atv)))
        return 0.0 if scale == 0.0 else difference / scale

    def _transpose_errors(
        self,
        derivs: StepperDerivatives[V],
        window: StepWindow[V],
        adj: Optional[BlockVector[V]],
    ) -> Dict[str, float]:
        """Dot-test errors of every operator at the window, symmetry
        errors of those declaring symmetry, and the handle errors."""
        stepper = self._stepper
        unknown, residual = stepper.unknown_space(), stepper.residual_space()
        state = stepper.state_space()
        # (name, operator, (domain space, range space))
        operators: List[
            Tuple[
                str,
                StepOperator[V],
                Tuple[BlockInnerProductSpace[V, A], BlockInnerProductSpace[V, A]],
            ]
        ] = [
            ("diag", stepper.diag(window), (unknown, residual)),
            ("readout", stepper.readout(), (unknown, state)),
        ]
        if derivs.off_diag is not None:
            for lag in range(1, len(window.past_states) + 1):
                operators.append(
                    (
                        f"off_diag[{lag}]",
                        derivs.off_diag(window, lag),
                        (state, residual),
                    )
                )
        params = stepper.param_space()
        if derivs.param is not None and params is not None:
            operators.append(("param", derivs.param(window), (params, residual)))
        if (
            adj is not None
            and params is not None
            and derivs.state_state is not None
            and derivs.state_param is not None
            and derivs.param_param is not None
        ):
            # The declared blocks among the lags the window read.
            read = len(window.past_states)
            for a, b in sorted(derivs.state_pairs):
                if a > read or b > read:
                    continue
                operators.append(
                    (
                        f"state_state[{a},{b}]",
                        derivs.state_state(window, adj, a, b),
                        (self._lag_space(b), self._lag_space(a)),
                    )
                )
            for a in sorted(derivs.param_lags):
                if a > read:
                    continue
                operators.append(
                    (
                        f"state_param[{a}]",
                        derivs.state_param(window, adj, a),
                        (params, self._lag_space(a)),
                    )
                )
            operators.append(
                ("param_param", derivs.param_param(window, adj), (params, params))
            )
        errors: Dict[str, float] = {}
        for name, op, (domain, range_) in operators:
            errors[name] = self._dot_test(op, domain, range_)
            if op.properties().symmetric:
                errors[f"{name} (symmetric)"] = self._symmetry_error(op, domain)
        errors.update(self._handle_errors(window))
        return errors

    def _handle_errors(self, window: StepWindow[V]) -> Dict[str, float]:
        """How far the diagonal handle is from inverting ``diag``, and its
        transpose from inverting ``diag``'s transpose: relative errors
        :math:`\\|H(A x) - x\\| / \\|x\\|`."""
        stepper = self._stepper
        unknown = stepper.unknown_space()
        diag = stepper.diag(window)
        handle = stepper.diag_handle(window)
        x = self._random(unknown)
        size = scalar(unknown.norm(x))

        def relative_error(recovered: BlockVector[V]) -> float:
            return scalar(unknown.norm(unknown.axpby(1.0, recovered, -1.0, x))) / size

        # Solve with the handle after applying the operator: x again.
        errors = {"diag_handle": relative_error(handle.apply(diag.apply(x)))}
        transposed = handle.transpose()
        if transposed is not None:
            recovered = transposed.apply(diag.apply_transpose(x))
            errors["diag_handle (transpose)"] = relative_error(recovered)
        return errors
