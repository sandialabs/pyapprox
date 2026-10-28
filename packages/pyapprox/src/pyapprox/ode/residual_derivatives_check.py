r"""Finite-difference check of an ODE residual's derivatives bundle.

The residual counterpart of
:class:`~pyapprox.interface.functions.derivative_checks.derivative_checker.DerivativeChecker`:
it checks whatever the bundle from ``derivatives()`` declares, so a
residual is checked in one call however its derivatives were written
(dense matrices, actions, assembled operators).

Each term is the derivative of a quantity built from lower-order terms,
compared with its finite differences along one direction:

- ``jacobian``: :math:`f(y)` in the state;
- ``param``: :math:`f(p)` in the parameters;
- ``state_state``: :math:`J(y)^\top\lambda` in the state;
- ``state_param``: :math:`J(p)^\top\lambda` in the parameters, and its
  transpose, ``param_state``, as :math:`(\partial f/\partial p)(y)^\top
  \lambda` in the state, so the two mixed partials must agree;
- ``param_param``: :math:`(\partial f/\partial p)(p)^\top\lambda` in the
  parameters;
- ``initial_state``: the caller's :math:`y_0(p)` in the parameters, and
  ``initial_state_curvature`` with weight :math:`\lambda` as
  :math:`(\partial y_0/\partial p)(p)^\top\lambda`. The residual does not
  own its initial state, so these two are checked only when the caller
  supplies :math:`y_0(p)`.

Moving the parameters goes through ``set_param``, after which the bundle
is fetched again, as the bundle's contract requires. A term declared
``STATED_ZERO`` is checked like any other: its zero operator is compared
with the finite differences in absolute terms, so a false declaration
fails. Every operator also gets a dot test,
:math:`\langle w, A v\rangle = \langle A^\top w, v\rangle`, the only
check of a transpose no finite difference reaches (the Jacobian's, the
parameter Jacobian's), and an operator declaring itself symmetric is
checked to be.
"""

from dataclasses import dataclass
from itertools import count
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np

from pyapprox.interface.functions.derivative_checks.base import JVPChecker
from pyapprox.ode.residual_derivatives import (
    DerivativeOperator,
    ODEResidualDerivatives,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import FormatProtocol, LinearizationPoint

T = TypeVar("T")
_Bundle = ODEResidualDerivatives[Array, FormatProtocol]
_Term = tuple[str, Array, Callable[[Array], Array], Callable[[Array, Array], Array]]


@runtime_checkable
class DifferentiableODEResidualProtocol(Protocol, Generic[Array]):
    """A residual :math:`f(y, t)` with a derivatives bundle."""

    def bkd(self) -> Backend[Array]: ...

    def __call__(self, state: Array, time: float) -> Array: ...

    def derivatives(self) -> ODEResidualDerivatives[Array, FormatProtocol]: ...


@runtime_checkable
class ParameterizedODEResidualProtocol(Protocol, Generic[Array]):
    """A residual :math:`f(y, t; p)` with a derivatives bundle whose
    parameters can be set."""

    def bkd(self) -> Backend[Array]: ...

    def __call__(self, state: Array, time: float) -> Array: ...

    def derivatives(self) -> ODEResidualDerivatives[Array, FormatProtocol]: ...

    def set_param(self, param: Array) -> None: ...


@dataclass(frozen=True)
class ODEResidualBundleReport:
    """Result of :meth:`ODEResidualBundleChecker.check`.

    Attributes
    ----------
    fd_errors : Mapping[str, float]
        For each term checked, the smallest finite-difference error over
        the steps: relative to the operator's action, or absolute where
        the action is zero (a stated zero).
    transpose_errors : Mapping[str, float]
        For each operator, the relative dot-test error; for an operator
        declaring itself symmetric, also ``"<name> (symmetric)"``, the
        relative difference between its action and its transpose's.
    """

    fd_errors: Mapping[str, float]
    transpose_errors: Mapping[str, float]

    def failures(self, fd_tol: float = 1e-6, transpose_tol: float = 1e-12) -> List[str]:
        """The names of the checks whose error exceeds its tolerance."""
        return [
            name for name, error in self.fd_errors.items() if not error <= fd_tol
        ] + [
            name
            for name, error in self.transpose_errors.items()
            if not error <= transpose_tol
        ]


def _present(field: Optional[T], name: str) -> T:
    if field is None:
        raise ValueError(
            f"a bundle fetched after set_param lacks '{name}', which the first "
            "fetch had; which fields are present must not depend on the "
            "parameter values"
        )
    return field


class _TermView(Generic[Array]):
    """One term as the finite-difference engine reads it: a quantity of a
    1-D input and the derivative's action on a direction block."""

    def __init__(
        self,
        bkd: Backend[Array],
        sizes: tuple[int, int],
        value: Callable[[Array], Array],
        action: Callable[[Array, Array], Array],
    ) -> None:
        self._bkd = bkd
        self._nvars, self._nqoi = sizes
        self._value, self._action = value, action

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        return self._bkd.stack(
            [self._value(samples[:, k]) for k in range(samples.shape[1])], axis=1
        )

    def jvp(self, sample: Array, vec: Array) -> Array:
        return self._action(sample[:, 0], vec)


class ODEResidualBundleChecker(Generic[Array]):
    r"""Check an ODE residual's derivatives bundle by finite differences.

    Parameters
    ----------
    residual : DifferentiableODEResidualProtocol
        The residual. One whose bundle has parameter derivatives must
        also satisfy :class:`ParameterizedODEResidualProtocol`; the check
        moves its parameters and leaves them at the point checked.
    initial_state : callable, optional
        :math:`p \mapsto y_0(p)`, 1-D to 1-D, for checking the bundle's
        ``initial_state`` and ``initial_state_curvature``; without it
        those two get only their dot tests.
    fd_eps : Array, optional
        The finite-difference steps; default ``1e-1`` down to ``1e-13``.
    central : bool, default False
        Central rather than forward differences.
    seed : int, default 0
        Seed of the directions and dot-test vectors.

    Raises
    ------
    TypeError
        If ``residual`` does not satisfy
        :class:`DifferentiableODEResidualProtocol`.
    """

    def __init__(
        self,
        residual: DifferentiableODEResidualProtocol[Array],
        initial_state: Optional[Callable[[Array], Array]] = None,
        fd_eps: Optional[Array] = None,
        central: bool = False,
        seed: int = 0,
    ) -> None:
        if not isinstance(residual, DifferentiableODEResidualProtocol):
            raise TypeError(
                "residual must satisfy DifferentiableODEResidualProtocol, got "
                f"{type(residual).__name__}"
            )
        self._residual = residual
        self._bkd = residual.bkd()
        self._initial_state = initial_state
        self._fd_eps = fd_eps
        self._central = central
        self._seed = seed
        self._version = 0
        self._points = count()

    def check(
        self,
        state: Array,
        time: float,
        adj: Optional[Array] = None,
        param: Optional[Array] = None,
    ) -> ODEResidualBundleReport:
        """Check every term the bundle declares at one point.

        Parameters
        ----------
        state : Array
            The state. Shape: (nstates,)
        time : float
            The time.
        adj : Array, optional
            The adjoint weighting the second derivatives, also the weight
            of the initial-state curvature; required when the bundle has
            second derivatives. Shape: (nstates,)
        param : Array, optional
            The parameters; required when the bundle has parameter
            derivatives. The residual is left at these. Shape: (nparams,)

        Raises
        ------
        ValueError
            If the bundle has parameter derivatives and ``param`` is
            missing, or second derivatives and ``adj`` is missing.
        """
        bundle = self._fetch(param)
        if bundle.param is not None and param is None:
            raise ValueError("the bundle has parameter derivatives; pass param")
        if bundle.state_state is not None and adj is None:
            raise ValueError("the bundle has second derivatives; pass adj")
        weight = self._bkd.zeros(state.shape) if adj is None else adj
        rng = np.random.default_rng(self._seed)
        try:
            fd_errors = {
                name: self._fd_error(point, value, action, rng)
                for name, point, value, action in self._terms(
                    bundle, state, time, weight, param
                )
            }
            transpose_errors = self._transpose_errors(
                self._fetch(param), state, time, weight, rng
            )
        finally:
            self._fetch(param)
        return ODEResidualBundleReport(fd_errors, transpose_errors)

    def _fetch(self, param: Optional[Array]) -> _Bundle[Array]:
        """The bundle, after setting ``param`` if given."""
        if param is not None:
            if not isinstance(self._residual, ParameterizedODEResidualProtocol):
                raise TypeError(
                    "the bundle has parameter derivatives, so the residual must "
                    "satisfy ParameterizedODEResidualProtocol (set_param), got "
                    f"{type(self._residual).__name__}"
                )
            self._residual.set_param(param)
            self._version += 1
        return self._residual.derivatives()

    def _jacobian(
        self, bundle: _Bundle[Array], state: Array, time: float
    ) -> DerivativeOperator[Array]:
        point = LinearizationPoint(self._version, next(self._points))
        return _present(bundle.jacobian, "jacobian")(state, time, point)

    def _terms(
        self,
        bundle: _Bundle[Array],
        state: Array,
        time: float,
        adj: Array,
        param: Optional[Array],
    ) -> List[_Term[Array]]:
        """Each declared term: its name, the point it is checked at, the
        quantity it differentiates and its action. Every evaluation
        fetches the bundle at the parameters it needs: the varied ones for
        a term in the parameters, ``param`` for a term in the state."""
        f, lam = self._residual, self._bkd.reshape(adj, (adj.shape[0], 1))
        terms: List[_Term[Array]] = []

        def base() -> _Bundle[Array]:
            return self._fetch(param)

        if bundle.jacobian is not None:

            def residual(y: Array) -> Array:
                base()
                return f(y, time)

            def jacobian(y: Array, v: Array) -> Array:
                return self._jacobian(base(), y, time).apply(v)

            terms.append(("jacobian", state, residual, jacobian))
        if param is None or bundle.param is None:
            return terms

        def residual_of_param(p: Array) -> Array:
            self._fetch(p)
            return f(state, time)

        def param_action(p: Array, v: Array) -> Array:
            return _present(self._fetch(p).param, "param")(state, time).apply(v)

        terms.append(("param", param, residual_of_param, param_action))
        y0 = self._initial_state
        if y0 is not None:

            def initial_state(p: Array, v: Array) -> Array:
                field = _present(self._fetch(p).initial_state, "initial_state")
                return field().apply(v)

            terms.append(("initial_state", param, y0, initial_state))
        if bundle.state_state is None:
            return terms

        def adjoint_jacobian(y: Array) -> Array:
            return self._jacobian(base(), y, time).apply_transpose(lam)[:, 0]

        def state_state(y: Array, w: Array) -> Array:
            field = _present(base().state_state, "state_state")
            return field(y, time, adj).apply(w)

        def adjoint_jacobian_of_param(p: Array) -> Array:
            return self._jacobian(self._fetch(p), state, time).apply_transpose(lam)[
                :, 0
            ]

        def state_param(p: Array, v: Array) -> Array:
            field = _present(self._fetch(p).state_param, "state_param")
            return field(state, time, adj).apply(v)

        def adjoint_param(y: Array) -> Array:
            field = _present(base().param, "param")
            return field(y, time).apply_transpose(lam)[:, 0]

        def param_state(y: Array, w: Array) -> Array:
            field = _present(base().state_param, "state_param")
            return field(y, time, adj).apply_transpose(w)

        def adjoint_param_of_param(p: Array) -> Array:
            field = _present(self._fetch(p).param, "param")
            return field(state, time).apply_transpose(lam)[:, 0]

        def param_param(p: Array, v: Array) -> Array:
            field = _present(self._fetch(p).param_param, "param_param")
            return field(state, time, adj).apply(v)

        terms += [
            ("state_state", state, adjoint_jacobian, state_state),
            ("state_param", param, adjoint_jacobian_of_param, state_param),
            ("param_state", state, adjoint_param, param_state),
            ("param_param", param, adjoint_param_of_param, param_param),
        ]
        if y0 is not None:

            def adjoint_initial_state(p: Array) -> Array:
                field = _present(self._fetch(p).initial_state, "initial_state")
                return field().apply_transpose(lam)[:, 0]

            def curvature(p: Array, v: Array) -> Array:
                field = _present(
                    self._fetch(p).initial_state_curvature, "initial_state_curvature"
                )
                return field(adj).apply(v)

            terms.append(
                ("initial_state_curvature", param, adjoint_initial_state, curvature)
            )
        return terms

    def _fd_error(
        self,
        point: Array,
        value: Callable[[Array], Array],
        action: Callable[[Array, Array], Array],
        rng: np.random.Generator,
    ) -> float:
        """Smallest finite-difference error of one term along a random
        direction. The quantity and the action are both evaluated at the
        point first, so a term varying the parameters starts there."""
        bkd = self._bkd
        nvars = point.shape[0]
        nqoi = value(point).shape[0]
        direction = bkd.asarray(rng.normal(size=(nvars, 1)))
        checker = JVPChecker(
            _TermView(bkd, (nvars, nqoi), value, action),
            fd_eps=self._fd_eps,
            direction=direction / bkd.norm(direction),
            central=self._central,
        )
        return bkd.to_float(bkd.min(checker.check(bkd.reshape(point, (nvars, 1)))))

    def _transpose_errors(
        self,
        bundle: _Bundle[Array],
        state: Array,
        time: float,
        adj: Array,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        operators: List[tuple[str, DerivativeOperator[Array]]] = []
        if bundle.jacobian is not None:
            operators.append(("jacobian", self._jacobian(bundle, state, time)))
        if bundle.param is not None and bundle.initial_state is not None:
            operators += [
                ("param", bundle.param(state, time)),
                ("initial_state", bundle.initial_state()),
            ]
        if (
            bundle.state_state is not None
            and bundle.state_param is not None
            and bundle.param_param is not None
            and bundle.initial_state_curvature is not None
        ):
            operators += [
                ("state_state", bundle.state_state(state, time, adj)),
                ("state_param", bundle.state_param(state, time, adj)),
                ("param_param", bundle.param_param(state, time, adj)),
                ("initial_state_curvature", bundle.initial_state_curvature(adj)),
            ]
        errors: Dict[str, float] = {}
        for name, op in operators:
            errors[name] = self._dot_test(op, rng)
            if op.properties().symmetric:
                errors[f"{name} (symmetric)"] = self._symmetry_error(op, rng)
        return errors

    def _random_column(self, size: int, rng: np.random.Generator) -> Array:
        return self._bkd.asarray(rng.normal(size=(size, 1)))

    def _dot_test(
        self, op: DerivativeOperator[Array], rng: np.random.Generator
    ) -> float:
        bkd = self._bkd
        v = self._random_column(op.domain().size, rng)
        w = self._random_column(op.range().size, rng)
        av, atw = op.apply(v), op.apply_transpose(w)
        scale = max(
            bkd.to_float(bkd.norm(w) * bkd.norm(av)),
            bkd.to_float(bkd.norm(v) * bkd.norm(atw)),
        )
        difference = bkd.to_float(bkd.sum(w * av) - bkd.sum(atw * v))
        return 0.0 if scale == 0.0 else abs(difference) / scale

    def _symmetry_error(
        self, op: DerivativeOperator[Array], rng: np.random.Generator
    ) -> float:
        bkd = self._bkd
        v = self._random_column(op.domain().size, rng)
        av, atv = op.apply(v), op.apply_transpose(v)
        scale = max(bkd.to_float(bkd.norm(av)), bkd.to_float(bkd.norm(atv)))
        difference = bkd.to_float(bkd.norm(av - atv))
        return 0.0 if scale == 0.0 else difference / scale
