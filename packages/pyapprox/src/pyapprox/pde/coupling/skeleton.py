r"""The skeleton residual and its Schur-complement Jacobian.

The coupled problem is stated on the skeleton, the one space every
subdomain's ports read from:

.. math::

    S(\lambda) = \sum_i E_i^\top r_i(E_i \lambda) = 0,

with :math:`r_i` subdomain :math:`i`'s local response and :math:`E_i` its
port transfer. Its Jacobian, the Schur complement (Steklov–Poincaré
operator) :math:`J = \sum_i E_i^\top (\partial r_i/\partial \lambda_i)
E_i`, is formed by an injected :class:`SchurBuilderProtocol` from the
responses' trace directional derivatives, never by asking what a
response is: :class:`MatrixFreeSchur` applies it (one derivative per
subdomain per product, for a Krylov solver) and :class:`DenseSchur`
assembles it (``nport`` derivatives per subdomain, for a direct solver).
The builder fixes the Jacobian's format, so pairing it with a solver of
another format is a type error. :class:`SkeletonResidual` is a
:class:`~pyapprox.implicitfunction.residual.ResidualProtocol`, so it is
solved by :class:`~pyapprox.implicitfunction.steady_system.SteadySystem`
with any nonlinear solver and setup strategy.

**Three kinds of reuse, each injected.** A subdomain's own solver setups
are its business (its own setup strategy); the skeleton's linear-solver
setups are the injected strategy of the system solving it; and the
response *points* (each subdomain linearized at one :math:`\lambda`,
holding its interior solution) are kept by an injected
:class:`PointStoreProtocol`. Evaluating :math:`S(\lambda)` produces the
points and the Jacobian at the same :math:`\lambda` needs them, so a
store keeping them saves an interior solve per subdomain per product. A
store recognizes a kept iterate by value (one reduction), never by
position or identity: a Newton iterate has no position, and a solver may
overwrite it in place. The residual acts the same under every store; only
the number of linearizations differs. Whoever changes what the responses
depend on calls the store's ``clear``.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.pde.coupling.ports import PortTransferProtocol
from pyapprox.pde.coupling.responses import (
    LocalResponseProtocol,
    ResponsePointProtocol,
    TimeWindow,
    TraceJVP,
)
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearizationPoint,
    LinearizedOperator,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
    VectorSpaceProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)
F_co = TypeVar("F_co", bound=FormatProtocol, covariant=True)
T = TypeVar("T")


@dataclass(frozen=True)
class CoupledSubdomain(Generic[V, Array]):
    """A subdomain's response and the transfer to its ports."""

    response: LocalResponseProtocol[Array]
    transfer: PortTransferProtocol[V, Array]


@runtime_checkable
class SubdomainExecutorProtocol(Protocol):
    """Runs per-subdomain work over the subdomains a rank holds."""

    def map(self, task: Callable[[int], T], subdomains: Sequence[int]) -> list[T]:
        """``task(i)`` for each ``i`` in ``subdomains``, returned in that
        order whatever order they ran in, so sums over the results do not
        depend on scheduling."""
        ...


class SerialExecutor:
    """Runs each subdomain's work in turn, in order."""

    def map(self, task: Callable[[int], T], subdomains: Sequence[int]) -> list[T]:
        return [task(i) for i in subdomains]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@runtime_checkable
class PointBuilderProtocol(Protocol[V, Array]):
    """What a point store asks of the residual it serves."""

    def linearize_all(self, state: V) -> Sequence[ResponsePointProtocol[Array]]:
        """Every subdomain's response point at ``state``."""
        ...

    def same_state(self, a: V, b: V) -> bool:
        """Whether ``a`` and ``b`` are equal, by one reduction."""
        ...

    def copy_state(self, state: V) -> V:
        """A copy of ``state`` no caller can overwrite."""
        ...


@runtime_checkable
class PointStoreProtocol(Protocol[V, Array]):
    """Decides, per request, whether to linearize or return kept points."""

    def points(
        self, state: V, builder: PointBuilderProtocol[V, Array]
    ) -> Sequence[ResponsePointProtocol[Array]]:
        """Every subdomain's point at ``state``."""
        ...

    def clear(self) -> None:
        """Drop every kept point."""
        ...


class RecomputePoints(Generic[V, Array]):
    """Linearizes every subdomain on every request; keeps nothing."""

    def points(
        self, state: V, builder: PointBuilderProtocol[V, Array]
    ) -> Sequence[ResponsePointProtocol[Array]]:
        return builder.linearize_all(state)

    def clear(self) -> None:
        return None


class KeepCurrentPoints(Generic[V, Array]):
    """Keeps the points of the last state requested, one per subdomain,
    until another state is requested or :meth:`clear`."""

    def __init__(self) -> None:
        self._state: Optional[V] = None
        self._points: Sequence[ResponsePointProtocol[Array]] = ()

    def points(
        self, state: V, builder: PointBuilderProtocol[V, Array]
    ) -> Sequence[ResponsePointProtocol[Array]]:
        if self._state is not None and builder.same_state(self._state, state):
            return self._points
        self._points = builder.linearize_all(state)
        self._state = builder.copy_state(state)
        return self._points

    def clear(self) -> None:
        self._state, self._points = None, ()


@runtime_checkable
class SchurBuilderProtocol(Protocol[V, Array, F_co]):
    """Forms the skeleton Jacobian at one set of response points, in a
    format of its choice; the caller pairs it with a solver for that
    format."""

    def build(
        self,
        space: VectorSpaceProtocol[V],
        subdomains: Sequence[CoupledSubdomain[V, Array]],
        points: Sequence[ResponsePointProtocol[Array]],
        executor: SubdomainExecutorProtocol,
    ) -> ActionOperatorProtocol[V, F_co]: ...


def _tangents(points: Sequence[ResponsePointProtocol[Array]]) -> list[TraceJVP[Array]]:
    tangents = []
    for i, point in enumerate(points):
        tangent = point.derivatives().jvp_trace
        if tangent is None:
            raise TypeError(
                f"subdomain {i}'s response point has no jvp_trace; the "
                "skeleton Jacobian needs one from every response"
            )
        tangents.append(tangent)
    return tangents


class MatrixFreeSchur(Generic[V, Array]):
    """Builds :class:`SchurOperator`: applied, never stored. Pairs with
    any solver that only applies its operator (Krylov)."""

    def build(
        self,
        space: VectorSpaceProtocol[V],
        subdomains: Sequence[CoupledSubdomain[V, Array]],
        points: Sequence[ResponsePointProtocol[Array]],
        executor: SubdomainExecutorProtocol,
    ) -> "SchurOperator[V, Array]":
        return SchurOperator(space, subdomains, points, executor)


class DenseSchur(Generic[Array]):
    r"""Assembles the skeleton Jacobian densely on a skeleton of backend
    arrays: :math:`J = \sum_i \tilde E_i^\top S_i \tilde E_i`, with
    :math:`S_i` the trace directional derivatives applied to the port's
    unit vectors and :math:`\tilde E_i` the transfer applied to the
    skeleton's. Pairs with a dense direct solver. Costs ``nport`` trace
    directional derivatives per subdomain per build.

    Parameters
    ----------
    bkd
        The skeleton's backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def build(
        self,
        space: VectorSpaceProtocol[Array],
        subdomains: Sequence[CoupledSubdomain[Array, Array]],
        points: Sequence[ResponsePointProtocol[Array]],
        executor: SubdomainExecutorProtocol,
    ) -> BackendDenseOperator[Array]:
        bkd, descriptor = self._bkd, space.descriptor()
        identity = bkd.eye(descriptor.size)
        tangents = _tangents(points)

        def block(i: int) -> Array:
            # S_i E_i: the tangent of each column of E_i.
            return tangents[i](subdomains[i].transfer.scatter(identity))

        blocks = executor.map(block, range(len(subdomains)))
        matrix = bkd.zeros((descriptor.size, descriptor.size))
        for sd, local in zip(subdomains, blocks):
            matrix = sd.transfer.gather_add(local, matrix)
        return BackendDenseOperator(matrix, descriptor, bkd)


class SchurOperator(Generic[V, Array]):
    r""":math:`J = \sum_i E_i^\top (\partial r_i/\partial \lambda_i) E_i` at
    one set of response points, applied matrix-free.

    Raises
    ------
    TypeError
        If a point's bundle has no ``jvp_trace``.
    """

    def __init__(
        self,
        space: VectorSpaceProtocol[V],
        subdomains: Sequence[CoupledSubdomain[V, Array]],
        points: Sequence[ResponsePointProtocol[Array]],
        executor: SubdomainExecutorProtocol,
    ) -> None:
        self._space = space
        self._transfers = [sd.transfer for sd in subdomains]
        self._tangents = _tangents(points)
        self._executor = executor

    def domain(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        def local(i: int) -> Array:
            return self._tangents[i](self._transfers[i].scatter(x))

        blocks = self._executor.map(local, range(len(self._transfers)))
        total = self._space.zeros(self._space.ncols(x))
        for transfer, block in zip(self._transfers, blocks):
            total = transfer.gather_add(block, total)
        return total

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        raise TypeError(
            "the responses' bundles carry no vjp_trace, so the Schur "
            "operator has no transpose action"
        )


class SkeletonResidual(Generic[V, A, Array, F]):
    r""":math:`S(\lambda) = \sum_i E_i^\top r_i(E_i \lambda)`.

    Satisfies :class:`~pyapprox.implicitfunction.residual.ResidualProtocol`
    with a Jacobian in the format its Schur builder forms. Without
    parameters: every response must have none, and the parameter version
    is 0.

    Parameters
    ----------
    space
        The skeleton's space; states are its single-column vectors.
    subdomains
        Each subdomain's response and port transfer: every subdomain in a
        serial run, a rank's own (:func:`~pyapprox.pde.coupling.topology.
        local_subdomains`) in a distributed one, where the skeleton
        stack's scatter-add sums across ranks.
    schur
        Forms the Jacobian: :class:`MatrixFreeSchur` for a Krylov solver,
        :class:`DenseSchur` for a dense direct one. The setup strategy of
        the system solving the residual pairs a solver with its format.
    points
        Keeps response points between requests. Default:
        :class:`KeepCurrentPoints`.
    window
        The window every response covers. Default: steady.
    executor
        Runs the per-subdomain work (linearizations, trace products).
        Default: :class:`SerialExecutor`.
    """

    def __init__(
        self,
        space: InnerProductSpaceProtocol[V, A],
        subdomains: Sequence[CoupledSubdomain[V, Array]],
        schur: SchurBuilderProtocol[V, Array, F],
        points: Optional[PointStoreProtocol[V, Array]] = None,
        window: Optional[TimeWindow] = None,
        executor: Optional[SubdomainExecutorProtocol] = None,
    ) -> None:
        if not isinstance(space, InnerProductSpaceProtocol):
            raise TypeError(
                "space must satisfy InnerProductSpaceProtocol, got "
                f"{type(space).__name__}"
            )
        if not isinstance(schur, SchurBuilderProtocol):
            raise TypeError(
                f"schur must satisfy SchurBuilderProtocol, got {type(schur).__name__}"
            )
        if executor is not None and not isinstance(
            executor, SubdomainExecutorProtocol
        ):
            raise TypeError(
                "executor must satisfy SubdomainExecutorProtocol, got "
                f"{type(executor).__name__}"
            )
        for i, sd in enumerate(subdomains):
            if not isinstance(sd.response, LocalResponseProtocol):
                raise TypeError(
                    f"subdomain {i}'s response must satisfy LocalResponseProtocol"
                )
            if not isinstance(sd.transfer, PortTransferProtocol):
                raise TypeError(
                    f"subdomain {i}'s transfer must satisfy PortTransferProtocol"
                )
            if sd.response.nport() != sd.transfer.nport():
                raise ValueError(
                    f"subdomain {i}: the response has {sd.response.nport()} port "
                    f"rows, the transfer {sd.transfer.nport()}"
                )
            if sd.response.nparams() != 0:
                raise ValueError(f"subdomain {i}'s response has parameters")
        self._space = space
        self._subdomains = tuple(subdomains)
        self._schur = schur
        self._store: PointStoreProtocol[V, Array] = (
            KeepCurrentPoints() if points is None else points
        )
        self._window = TimeWindow.steady() if window is None else window
        self._executor: SubdomainExecutorProtocol = (
            SerialExecutor() if executor is None else executor
        )
        self._derivs: ResidualDerivatives[V, F] = ResidualDerivatives.state_only(
            ResidualJacobian.general(self._jacobian)
        )

    def subdomains(self) -> tuple[CoupledSubdomain[V, Array], ...]:
        """Each subdomain's response and port transfer."""
        return self._subdomains

    def linearize_all(self, state: V) -> Sequence[ResponsePointProtocol[Array]]:
        def linearize(i: int) -> ResponsePointProtocol[Array]:
            sd = self._subdomains[i]
            params = sd.response.bkd().zeros((0, 1))
            port_data = sd.transfer.scatter(state)
            return sd.response.linearize(port_data, params, self._window)

        return self._executor.map(linearize, range(len(self._subdomains)))

    def same_state(self, a: V, b: V) -> bool:
        difference = self._space.axpby(1.0, a, -1.0, b)
        # Only decides reuse; nothing computed flows from it.
        return bool((self._space.norm(difference) <= 0.0).item())

    def copy_state(self, state: V) -> V:
        return self._space.copy(state)

    def __call__(self, state: V) -> V:
        total = self._space.zeros(1)
        points = self._store.points(state, self)
        for sd, point in zip(self._subdomains, points):
            total = sd.transfer.gather_add(point.flux(), total)
        return total

    def _jacobian(
        self, state: V, label: LinearizationPoint
    ) -> LinearizedOperator[V, F]:
        points = self._store.points(state, self)
        return LinearizedOperator(
            self._schur.build(self._space, self._subdomains, points, self._executor),
            label,
        )

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._derivs

    def parameters_version(self) -> int:
        return 0
