r"""The skeleton residual and its Schur-complement Jacobian.

The coupled problem is stated on the skeleton, the one space every
subdomain's ports read from:

.. math::

    S(\lambda) = \sum_i E_i^\top r_i(E_i \lambda) + C \lambda = 0,

with :math:`r_i` subdomain :math:`i`'s local response, :math:`E_i` its
port transfer, and :math:`C` an optional fixed coupling
(:class:`CouplingProtocol`). :math:`C` is nonzero only where pieces
receive different kinds of data, so one piece's input is the value
another returns (:mod:`~pyapprox.pde.coupling.kind_layout` numbers such
a system and its entries); the residual never asks what a response
receives or returns. Its Jacobian, the Schur complement
(Steklov–Poincaré operator) :math:`J = \sum_i E_i^\top (\partial
r_i/\partial \lambda_i) E_i + C`, is formed by an injected
:class:`SchurBuilderProtocol` from the
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
from pyapprox.pde.coupling.ports import PortRowsProtocol, PortTransferProtocol
from pyapprox.pde.coupling.responses import (
    LocalResponseProtocol,
    ResponsePointProtocol,
    TimeWindow,
    TraceJVP,
    TraceVJP,
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


@dataclass(frozen=True)
class CoupledParameters(Generic[V, Array]):
    """The coupled problem's parameters :math:`p`: single-column vectors
    of ``space``, each subdomain reading its own parameters through its
    entry of ``rows`` (``None`` for a subdomain without parameters), in
    the order of the residual's subdomains. Rows of different subdomains
    may overlap: a shared parameter is read by each.

    The skeleton has no parameters of its own: :math:`S(\\lambda; p)`
    depends on :math:`p` only through the responses. It holds :math:`p`
    so that the coupled solution can be differentiated in it."""

    space: VectorSpaceProtocol[V]
    rows: Sequence[Optional[PortRowsProtocol[V, Array]]]


@runtime_checkable
class CouplingProtocol(Protocol[V]):
    r"""A fixed linear coupling :math:`C` added to the skeleton residual
    and its Jacobian."""

    def apply_add(self, x: V, y: V) -> V:
        r""":math:`y + C x` for a block ``x``; ``y`` is not modified."""
        ...

    def apply_transpose_add(self, x: V, y: V) -> V:
        r""":math:`y + C^\top x` for a block ``x``; ``y`` is not modified."""
        ...


class RowCoupling(Generic[V, Array]):
    r""":math:`C x = R^\top (s \odot U x)`: entry :math:`j` adds
    :math:`s_j` times unknown :math:`U_j` to row :math:`R_j`.

    Stack-neutral: the unknowns and rows are port rows of the skeleton's
    stack, the signs a column on the port blocks' backend. Costs
    O(nentries) per column.

    Parameters
    ----------
    unknowns
        The unknown of each entry.
    rows
        The residual row of each entry.
    signs
        The sign of each entry, shape (nentries, 1).

    Raises
    ------
    TypeError
        If ``unknowns`` or ``rows`` does not satisfy
        :class:`~pyapprox.pde.coupling.ports.PortRowsProtocol`.
    ValueError
        If the entry counts disagree.
    """

    def __init__(
        self,
        unknowns: PortRowsProtocol[V, Array],
        rows: PortRowsProtocol[V, Array],
        signs: Array,
    ) -> None:
        for name, value in (("unknowns", unknowns), ("rows", rows)):
            if not isinstance(value, PortRowsProtocol):
                raise TypeError(
                    f"{name} must satisfy PortRowsProtocol, got {type(value).__name__}"
                )
        n = unknowns.nrows()
        if rows.nrows() != n or tuple(signs.shape) != (n, 1):
            raise ValueError(
                f"{n} unknowns need {n} rows and signs of shape ({n}, 1), got "
                f"{rows.nrows()} rows and signs of shape {tuple(signs.shape)}"
            )
        self._unknowns, self._rows, self._signs = unknowns, rows, signs

    def apply_add(self, x: V, y: V) -> V:
        return self._rows.add_to(y, self._signs * self._unknowns.take(x))

    def apply_transpose_add(self, x: V, y: V) -> V:
        return self._unknowns.add_to(y, self._signs * self._rows.take(x))


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
        coupling: Optional[CouplingProtocol[V]],
    ) -> ActionOperatorProtocol[V, F_co]:
        """The Jacobian, with ``coupling`` added when it is not ``None``."""
        ...


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


def _adjoints(
    points: Sequence[ResponsePointProtocol[Array]],
) -> Optional[list[TraceVJP[Array]]]:
    """Every point's ``vjp_trace``, or ``None`` when one has none."""
    adjoints = []
    for point in points:
        adjoint = point.derivatives().vjp_trace
        if adjoint is None:
            return None
        adjoints.append(adjoint)
    return adjoints


class MatrixFreeSchur(Generic[V, Array]):
    """Builds :class:`SchurOperator`: applied, never stored. Pairs with
    any solver that only applies its operator (Krylov)."""

    def build(
        self,
        space: VectorSpaceProtocol[V],
        subdomains: Sequence[CoupledSubdomain[V, Array]],
        points: Sequence[ResponsePointProtocol[Array]],
        executor: SubdomainExecutorProtocol,
        coupling: Optional[CouplingProtocol[V]],
    ) -> "SchurOperator[V, Array]":
        return SchurOperator(space, subdomains, points, executor, coupling)


class DenseSchur(Generic[Array]):
    r"""Assembles the skeleton Jacobian densely on a skeleton of backend
    arrays: :math:`J = \sum_i \tilde E_i^\top S_i \tilde E_i + C`, with
    :math:`S_i` the trace directional derivatives applied to the port's
    unit vectors, :math:`\tilde E_i` the transfer applied to the
    skeleton's, and :math:`C` the coupling applied to them. Pairs with a
    dense direct solver. Costs ``nport`` trace directional derivatives
    per subdomain per build.

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
        coupling: Optional[CouplingProtocol[Array]],
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
        if coupling is not None:
            matrix = coupling.apply_add(identity, matrix)
        return BackendDenseOperator(matrix, descriptor, bkd)


class SchurOperator(Generic[V, Array]):
    r""":math:`J = \sum_i E_i^\top (\partial r_i/\partial \lambda_i) E_i + C`
    at one set of response points, applied matrix-free; :math:`C` is
    absent when ``coupling`` is ``None``. Transposable when every point's
    bundle carries ``vjp_trace``: :math:`J^\top = \sum_i E_i^\top
    (\partial r_i/\partial \lambda_i)^\top E_i + C^\top`.

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
        coupling: Optional[CouplingProtocol[V]] = None,
    ) -> None:
        self._space = space
        self._transfers = [sd.transfer for sd in subdomains]
        self._tangents = _tangents(points)
        self._adjoints = _adjoints(points)
        self._executor = executor
        self._coupling = coupling

    def domain(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def properties(self) -> OperatorProperties:
        """Transposable exactly when every point carries ``vjp_trace``."""
        return OperatorProperties(transposable=self._adjoints is not None)

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        def local(i: int) -> Array:
            return self._tangents[i](self._transfers[i].scatter(x))

        total = self._sum(local, x)
        if self._coupling is not None:
            total = self._coupling.apply_add(x, total)
        return total

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        """Raises ``TypeError`` when a point carries no ``vjp_trace``."""
        adjoints = self._adjoints
        if adjoints is None:
            raise TypeError(
                "a response point carries no vjp_trace, so the Schur operator "
                "has no transpose action"
            )

        def local(i: int) -> Array:
            return adjoints[i](self._transfers[i].scatter(y))

        total = self._sum(local, y)
        if self._coupling is not None:
            total = self._coupling.apply_transpose_add(y, total)
        return total

    def _sum(self, local: Callable[[int], Array], x: V) -> V:
        r""":math:`\sum_i E_i^\top` ``local(i)``."""
        blocks = self._executor.map(local, range(len(self._transfers)))
        total = self._space.zeros(self._space.ncols(x))
        for transfer, block in zip(self._transfers, blocks):
            total = transfer.gather_add(block, total)
        return total


class SkeletonParamOperator(Generic[V, Array]):
    r""":math:`S_p = \sum_i E_i^\top (\partial r_i/\partial p_i) P_i` at
    one set of response points, applied matrix-free, :math:`P_i` the
    reading of subdomain :math:`i`'s parameters from :math:`p`. Its
    transpose is :math:`\sum_i P_i^\top (\partial r_i/\partial
    p_i)^\top E_i`. Subdomains without parameters contribute nothing.

    Raises
    ------
    TypeError
        From :meth:`apply` (:meth:`apply_transpose`) when a subdomain with
        parameters has a point without ``jvp_param`` (``vjp_param``).
    """

    def __init__(
        self,
        space: VectorSpaceProtocol[V],
        parameters: CoupledParameters[V, Array],
        subdomains: Sequence[CoupledSubdomain[V, Array]],
        points: Sequence[ResponsePointProtocol[Array]],
        executor: SubdomainExecutorProtocol,
    ) -> None:
        self._space, self._parameters = space, parameters
        self._transfers = [sd.transfer for sd in subdomains]
        self._bundles = [point.derivatives() for point in points]
        self._executor = executor
        self._with_params = [
            i for i, rows in enumerate(parameters.rows) if rows is not None
        ]

    def domain(self) -> SpaceDescriptor:
        return self._parameters.space.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def properties(self) -> OperatorProperties:
        """Transposable exactly when every subdomain with parameters has a
        point carrying ``vjp_param``."""
        return OperatorProperties(
            transposable=all(
                self._bundles[i].vjp_param is not None for i in self._with_params
            )
        )

    def format(self) -> MatrixFree:
        return MatrixFree()

    def _rows(self, i: int) -> PortRowsProtocol[V, Array]:
        rows = self._parameters.rows[i]
        if rows is None:
            raise RuntimeError(f"subdomain {i} has no parameters")
        return rows

    def apply(self, x: V, out: Optional[V] = None) -> V:
        def local(i: int) -> Array:
            jvp = self._bundles[i].jvp_param
            if jvp is None:
                raise TypeError(f"subdomain {i}'s response point has no jvp_param")
            return jvp(self._rows(i).take(x))

        blocks = self._executor.map(local, self._with_params)
        total = self._space.zeros(self._parameters.space.ncols(x))
        for i, block in zip(self._with_params, blocks):
            total = self._transfers[i].gather_add(block, total)
        return total

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        def local(i: int) -> Array:
            vjp = self._bundles[i].vjp_param
            if vjp is None:
                raise TypeError(f"subdomain {i}'s response point has no vjp_param")
            return vjp(self._transfers[i].scatter(y))

        blocks = self._executor.map(local, self._with_params)
        params = self._parameters.space
        total = params.zeros(self._space.ncols(y))
        for i, block in zip(self._with_params, blocks):
            total = self._rows(i).add_to(total, block)
        return total


class SkeletonResidual(Generic[V, A, Array, F]):
    r""":math:`S(\lambda) = \sum_i E_i^\top r_i(E_i \lambda) + C \lambda`.

    Satisfies :class:`~pyapprox.implicitfunction.residual.ResidualProtocol`
    with a Jacobian in the format its Schur builder forms. Without
    ``parameters`` every response must have none, and the parameter
    version is 0. With them, :math:`S(\lambda; p)`: each response
    linearizes at its own parameters read from :math:`p`
    (:meth:`set_param`), and the bundle carries the parameter Jacobian
    :class:`SkeletonParamOperator`, built from the responses' parameter
    derivatives.

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
    coupling
        The fixed coupling :math:`C` of a system whose pieces receive
        different kinds of data (:class:`RowCoupling` from a
        :class:`~pyapprox.pde.coupling.kind_layout.KindLayout`'s entries),
        or ``None`` when every piece receives the same kind. In a
        distributed run exactly one rank holds it. The layout, not the
        residual, checks the responses' declared kinds. Default: ``None``.
    parameters
        The coupled parameters and each subdomain's reading of them, or
        ``None`` when no response has parameters. They start at zero.
        Default: ``None``.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If a response and its transfer disagree on the port count, a
        response has parameters without ``parameters``, or
        ``parameters`` has another entry count than the subdomains or
        rows of another count than their response's parameters.
    """

    def __init__(
        self,
        space: InnerProductSpaceProtocol[V, A],
        subdomains: Sequence[CoupledSubdomain[V, Array]],
        schur: SchurBuilderProtocol[V, Array, F],
        points: Optional[PointStoreProtocol[V, Array]] = None,
        window: Optional[TimeWindow] = None,
        executor: Optional[SubdomainExecutorProtocol] = None,
        coupling: Optional[CouplingProtocol[V]] = None,
        parameters: Optional[CoupledParameters[V, Array]] = None,
    ) -> None:
        if coupling is not None and not isinstance(coupling, CouplingProtocol):
            raise TypeError(
                f"coupling must satisfy CouplingProtocol, got {type(coupling).__name__}"
            )
        if not isinstance(space, InnerProductSpaceProtocol):
            raise TypeError(
                "space must satisfy InnerProductSpaceProtocol, got "
                f"{type(space).__name__}"
            )
        if not isinstance(schur, SchurBuilderProtocol):
            raise TypeError(
                f"schur must satisfy SchurBuilderProtocol, got {type(schur).__name__}"
            )
        if executor is not None and not isinstance(executor, SubdomainExecutorProtocol):
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
        _check_parameters(subdomains, parameters)
        self._space = space
        self._subdomains = tuple(subdomains)
        self._schur = schur
        self._store: PointStoreProtocol[V, Array] = (
            KeepCurrentPoints() if points is None else points
        )
        self._window = TimeWindow.steady() if window is None else window
        self._coupling = coupling
        self._executor: SubdomainExecutorProtocol = (
            SerialExecutor() if executor is None else executor
        )
        self._parameters = parameters
        self._param: Optional[V] = (
            None if parameters is None else parameters.space.zeros(1)
        )
        self._version = 0
        jacobian: ResidualJacobian[V, F] = ResidualJacobian.general(self._jacobian)
        self._derivs: ResidualDerivatives[V, F] = (
            ResidualDerivatives.state_only(jacobian)
            if parameters is None
            else ResidualDerivatives.first_order(
                parameters.space, jacobian, self._param_jacobian
            )
        )

    def subdomains(self) -> tuple[CoupledSubdomain[V, Array], ...]:
        """Each subdomain's response and port transfer."""
        return self._subdomains

    def coupling(self) -> Optional[CouplingProtocol[V]]:
        """The fixed coupling, or ``None``."""
        return self._coupling

    def nparams(self) -> int:
        """The number of coupled parameters; 0 without them."""
        if self._parameters is None:
            return 0
        return self._parameters.space.descriptor().size

    def param(self) -> V:
        """The coupled parameters, a single column.

        Raises
        ------
        ValueError
            If the residual has no parameters.
        """
        if self._param is None:
            raise ValueError("this skeleton residual has no parameters")
        return self._param

    def set_param(self, param: V) -> None:
        """Set the coupled parameters from a single column of their
        space; moves the parameter version and drops the kept points.

        Raises
        ------
        ValueError
            If the residual has no parameters.
        """
        if self._parameters is None:
            raise ValueError("this skeleton residual has no parameters")
        self._param = self._parameters.space.copy(param)
        self._version += 1
        self._store.clear()

    def _own_params(self, i: int) -> Array:
        response = self._subdomains[i].response
        rows = None if self._parameters is None else self._parameters.rows[i]
        if rows is None or self._param is None:
            return response.bkd().zeros((0, 1))
        return rows.take(self._param)

    def linearize_all(self, state: V) -> Sequence[ResponsePointProtocol[Array]]:
        def linearize(i: int) -> ResponsePointProtocol[Array]:
            sd = self._subdomains[i]
            port_data = sd.transfer.scatter(state)
            return sd.response.linearize(port_data, self._own_params(i), self._window)

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
            total = sd.transfer.gather_add(point.output(), total)
        if self._coupling is not None:
            total = self._coupling.apply_add(state, total)
        return total

    def _jacobian(
        self, state: V, label: LinearizationPoint
    ) -> LinearizedOperator[V, F]:
        points = self._store.points(state, self)
        return LinearizedOperator(
            self._schur.build(
                self._space, self._subdomains, points, self._executor, self._coupling
            ),
            label,
        )

    def _param_jacobian(self, state: V) -> SkeletonParamOperator[V, Array]:
        if self._parameters is None:
            raise RuntimeError(
                "the residual has no parameters; check derivatives() before calling"
            )
        return SkeletonParamOperator(
            self._space,
            self._parameters,
            self._subdomains,
            self._store.points(state, self),
            self._executor,
        )

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._derivs

    def parameters_version(self) -> int:
        """The number of :meth:`set_param` calls so far."""
        return self._version


class CoupledParameterSetter(Generic[Array, F]):
    """Sets a backend skeleton residual's coupled parameters from a 1-D
    array, as :class:`~pyapprox.implicitfunction.steady_function.
    SteadyImplicitFunction` passes them: the setter that makes the coupled
    problem a function of its parameters.

    Satisfies :class:`~pyapprox.implicitfunction.steady_function.
    ParameterSetterProtocol`.

    Raises
    ------
    ValueError
        If ``residual`` has no parameters.
    """

    def __init__(
        self, residual: SkeletonResidual[Array, Array, Array, F], bkd: Backend[Array]
    ) -> None:
        if residual.nparams() == 0:
            raise ValueError("the skeleton residual has no parameters to set")
        self._residual, self._bkd = residual, bkd

    def nparams(self) -> int:
        return self._residual.nparams()

    def set_param(self, param: Array) -> None:
        """Apply ``param``, shape ``(nparams,)``."""
        self._residual.set_param(self._bkd.reshape(param, (-1, 1)))


def _check_parameters(
    subdomains: Sequence[CoupledSubdomain[V, Array]],
    parameters: Optional[CoupledParameters[V, Array]],
) -> None:
    """Every response's parameter count matches its reading of the
    coupled parameters; without them, every response has none."""
    if parameters is None:
        for i, sd in enumerate(subdomains):
            if sd.response.nparams() != 0:
                raise ValueError(
                    f"subdomain {i}'s response has parameters; pass parameters"
                )
        return
    if len(parameters.rows) != len(subdomains):
        raise ValueError(
            f"parameters has {len(parameters.rows)} entries for "
            f"{len(subdomains)} subdomains"
        )
    for i, (sd, rows) in enumerate(zip(subdomains, parameters.rows)):
        count = 0 if rows is None else rows.nrows()
        if count != sd.response.nparams():
            raise ValueError(
                f"subdomain {i}'s response has {sd.response.nparams()} parameters, "
                f"its rows {count}"
            )
