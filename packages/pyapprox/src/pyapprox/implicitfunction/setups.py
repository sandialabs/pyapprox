r"""Linear-solver setups as an injected strategy.

A consumer that solves with an operator (a Newton step, a sensitivity
solve) describes the operator in a **request** and hands it, with a
**builder** that can form the operator, to a **strategy**. The strategy
either sets the operator up with its linear solver or returns a handle it
kept. The consumer makes the same calls under every strategy; only the
number of setups differs, so how setups are reused is decided by the
object injected, never by the consumer.

A request's :meth:`~SetupRequestProtocol.key` is what a strategy keeps:
two requests have equal keys exactly when they describe the same
operator. The operator's dependence on the state is read from the source's
derivatives bundle when the request is made, never passed by the consumer.
Where the operator depends on the state, the key names the state by its
**position**, as a trajectory store names its states (a store and a step,
say, or a solved steady point), never by the state object: a store that
recomputes states from checkpoints hands back new objects at the same
position. A state with no position (an iterate of a solve) gives a key
equal to no other. The key never holds the state, so kept setups keep no
states alive; the request holds it only for the builder, which reads it
while :meth:`SetupStrategyProtocol.handle` runs.

Kept setups are valid for one parameter value. A request carries no
parameters: whoever changes them (``set_param``, or a coefficient's own
setter) calls :meth:`SetupStrategyProtocol.clear`, as whoever runs a solve
supplies a new trajectory store. A strategy that keeps nothing,
:class:`RecomputeAlways`, cannot go stale.

This module holds the family-free protocols and key, the steady family
(:class:`SteadyRequest`, :class:`ResidualJacobianBuilder`: :math:`R_u` of
:math:`R(u; p) = 0`) and the strategies, which serve every family.
"""

from typing import (
    Dict,
    Generic,
    Hashable,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    LinearizationPoint,
    LinearSolverProtocol,
    SolverHandleProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)
F_co = TypeVar("F_co", bound=FormatProtocol, covariant=True)
F_contra = TypeVar("F_contra", bound=FormatProtocol, contravariant=True)


class SetupKey:
    """What determines an operator: parts compared by identity (its
    source, a pin) and parts compared by value (coefficients, a time, a
    position). Holds no state.

    Parameters
    ----------
    identities : tuple
        Objects compared with ``is``.
    values : tuple
        Hashable values compared with ``==``.
    """

    __slots__ = ("_identities", "_values")

    def __init__(
        self, identities: Tuple[object, ...], values: Tuple[Hashable, ...]
    ) -> None:
        self._identities = identities
        self._values = values

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SetupKey):
            return NotImplemented
        return (
            len(other._identities) == len(self._identities)
            and all(a is b for a, b in zip(other._identities, self._identities))
            and other._values == self._values
        )

    def __hash__(self) -> int:
        return hash((tuple(id(part) for part in self._identities), self._values))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(values={self._values!r})"


class _Unpositioned:
    """Stands for a state with no position in a key: equal only to
    itself, so the key matches no other request's."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "unpositioned"


def state_key(position: Optional[Hashable]) -> Hashable:
    """The key part naming a state: its position, or, for a state with
    none, a part equal to no other."""
    return _Unpositioned() if position is None else position


@runtime_checkable
class SetupRequestProtocol(Protocol):
    """A description of one operator and the point it is evaluated at.

    Each family adds the fields its builders read, among them the source
    the operator is built from, typed as that family's source.
    """

    def key(self) -> SetupKey:
        """Equal exactly for requests describing the same operator; holds
        no state."""
        ...

    def sees_state(self) -> bool:
        """Whether the operator depends on the evaluation state."""
        ...

    def at_solution(self) -> bool:
        """Whether the state is a solution named by its position, rather
        than an iterate of a solve."""
        ...


Q = TypeVar("Q", bound=SetupRequestProtocol)
Q_contra = TypeVar("Q_contra", bound=SetupRequestProtocol, contravariant=True)


@runtime_checkable
class OperatorBuilderProtocol(Protocol[Q_contra, V, F_co]):
    """Forms the operator a request describes; a function of the request
    alone, keeping nothing between calls."""

    def build(self, request: Q_contra) -> ActionOperatorProtocol[V, F_co]: ...


@runtime_checkable
class SetupStrategyProtocol(Protocol[Q, V, F_contra]):
    """What a consumer of one request family holds: decides, per request,
    whether to set up or return a kept handle."""

    def handle(
        self, request: Q, builder: OperatorBuilderProtocol[Q, V, F_contra]
    ) -> SolverHandleProtocol[V]:
        """A handle for the operator ``request`` describes; ``builder``
        forms it when a setup is needed."""
        ...

    def clear(self) -> None:
        """Drop every kept handle; call when the parameters change."""
        ...


@runtime_checkable
class AnyRequestStrategyProtocol(Protocol[V, F_contra]):
    """A strategy serving every request family; what a strategy composed
    of others holds."""

    def handle(
        self, request: Q, builder: OperatorBuilderProtocol[Q, V, F_contra]
    ) -> SolverHandleProtocol[V]: ...

    def clear(self) -> None: ...


class SteadyRequest(Generic[V, F]):
    r""":math:`R_u` of a residual at a state.

    Parameters
    ----------
    source : ResidualProtocol
        The residual :math:`R`; compared by identity.
    state : V
        Where :math:`R_u` is evaluated, a single column of the state
        space; read by the builder only.
    position : hashable or None
        Names ``state`` when it solves :math:`R(u; p) = 0` (the solved
        point, say); ``None`` for a Newton iterate. Compared by value.

    Raises
    ------
    TypeError
        If the residual's bundle has no Jacobian.
    """

    __slots__ = ("_source", "_state", "_position", "_sees_state", "_key")

    def __init__(
        self,
        source: ResidualProtocol[V, F],
        state: V,
        position: Optional[Hashable],
    ) -> None:
        jacobian = source.derivatives().jacobian
        if jacobian is None:
            raise TypeError(
                f"{type(source).__name__}'s derivatives bundle has no jacobian; "
                "there is no operator to request"
            )
        self._source = source
        self._state = state
        self._position = position
        self._sees_state = not jacobian.affine()
        where = state_key(position) if self._sees_state else None
        self._key = SetupKey((source,), (where,))

    def source(self) -> ResidualProtocol[V, F]:
        """The residual; compared by identity."""
        return self._source

    def state(self) -> V:
        return self._state

    def position(self) -> Optional[Hashable]:
        return self._position

    def at_solution(self) -> bool:
        return self._position is not None

    def sees_state(self) -> bool:
        return self._sees_state

    def key(self) -> SetupKey:
        return self._key

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._source!r}, "
            f"sees_state={self._sees_state}, position={self._position!r})"
        )


class ResidualJacobianBuilder(Generic[V, F]):
    r"""Forms :math:`R_u` at a request's state from the residual's own
    Jacobian.

    Raises
    ------
    ValueError
        From :meth:`build`, for a request of another residual.
    """

    def __init__(self, residual: ResidualProtocol[V, F]) -> None:
        self._residual = residual

    def build(self, request: SteadyRequest[V, F]) -> ActionOperatorProtocol[V, F]:
        if request.source() is not self._residual:
            raise ValueError(
                f"a request of {request.source()!r} given to the builder of "
                f"{self._residual!r}"
            )
        jacobian = self._residual.derivatives().jacobian
        if jacobian is None:
            raise AssertionError("a request is made only of a residual with one")
        # The label is still part of the Jacobian's call; nothing reads it.
        version = self._residual.parameters_version()
        if request.sees_state():
            label = LinearizationPoint(
                version, object(), provisional=not request.at_solution()
            )
        else:
            label = LinearizationPoint(version)
        return jacobian(request.state(), label)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._residual!r})"


class RecomputeAlways(Generic[V, F]):
    """Sets up every request; keeps nothing, so it cannot go stale."""

    def __init__(self, solver: LinearSolverProtocol[V, F]) -> None:
        self._solver = solver

    def handle(
        self, request: Q, builder: OperatorBuilderProtocol[Q, V, F]
    ) -> SolverHandleProtocol[V]:
        return self._solver.setup(builder.build(request))

    def clear(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._solver!r})"


class KeepStateFree(Generic[V, F]):
    """Keeps every setup whose operator does not depend on the state
    (constant or time-only Jacobians, mass-only operators) until
    :meth:`clear`; sets up the rest afresh."""

    def __init__(self, solver: LinearSolverProtocol[V, F]) -> None:
        self._solver = solver
        self._kept: Dict[SetupKey, SolverHandleProtocol[V]] = {}

    def handle(
        self, request: Q, builder: OperatorBuilderProtocol[Q, V, F]
    ) -> SolverHandleProtocol[V]:
        if request.sees_state():
            return self._solver.setup(builder.build(request))
        key = request.key()
        kept = self._kept.get(key)
        if kept is None:
            kept = self._solver.setup(builder.build(request))
            self._kept[key] = kept
        return kept

    def clear(self) -> None:
        self._kept = {}

    def nkept(self) -> int:
        """How many handles are kept."""
        return len(self._kept)

    def __getstate__(self) -> Dict[str, object]:
        # Kept handles are valid only in this process's run.
        return {"_solver": self._solver, "_kept": {}}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._solver!r}, nkept={self.nkept()})"


class KeepAll(Generic[V, F]):
    """Keeps every state-free setup and every setup at a solution (a
    converged stage, step or steady state, named by its position) until
    :meth:`clear`; a setup at an iterate of a solve is used once.

    Holds one setup per solution requested, but never the solution itself:
    keys name states by position.
    """

    def __init__(self, solver: LinearSolverProtocol[V, F]) -> None:
        self._solver = solver
        self._kept: Dict[SetupKey, SolverHandleProtocol[V]] = {}

    def handle(
        self, request: Q, builder: OperatorBuilderProtocol[Q, V, F]
    ) -> SolverHandleProtocol[V]:
        key = request.key()
        kept = self._kept.get(key)
        if kept is not None:
            return kept
        handle = self._solver.setup(builder.build(request))
        if request.at_solution() or not request.sees_state():
            self._kept[key] = handle
        return handle

    def clear(self) -> None:
        self._kept = {}

    def nkept(self) -> int:
        """How many handles are kept."""
        return len(self._kept)

    def __getstate__(self) -> Dict[str, object]:
        # Kept handles are valid only in this process's run.
        return {"_solver": self._solver, "_kept": {}}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._solver!r}, nkept={self.nkept()})"
