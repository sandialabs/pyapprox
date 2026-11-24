r"""Port transfers: moving data between the skeleton and one subdomain.

A subdomain's *port block* is the skeleton data it touches, held in a
small array local to the rank that owns the subdomain. It is the only
thing that crosses between stacks: the skeleton stack takes a block out
of a skeleton vector and adds one back (:class:`PortRowsProtocol`), and
each subdomain converts the block to its own vector type.

A port transfer :math:`E_i` maps skeleton data to the subdomain's port
data; its transpose adds a subdomain's tested flux back into a skeleton
vector, which keeps the flux sum conservative.
"""

from typing import Generic, Optional, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.pde.coupling.geometry import FacePoint
from pyapprox.pde.coupling.spaces import FaceValues
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import backend_dtype_name
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    LinearSolverProtocol,
    OperatorProperties,
    SpaceDescriptor,
)
from pyapprox.util.linalg.operators.algebra import Product

V = TypeVar("V")
P = TypeVar("P")
F = TypeVar("F")
F_co = TypeVar("F_co", covariant=True)
# Element data only flows in to a sink: I for DOF maps, E for values.
I_contra = TypeVar("I_contra", contravariant=True)
E_contra = TypeVar("E_contra", contravariant=True)


@runtime_checkable
class PortRowsProtocol(Protocol, Generic[V, P]):
    r"""Rows :math:`R` of a skeleton space, read into and added from a
    port block of type ``P``; supplied by the skeleton's stack."""

    def nrows(self) -> int:
        """The number of rows in the port block."""
        ...

    def take(self, x: V) -> P:
        r""":math:`R x`, shape (nrows, ncols(x))."""
        ...

    def add_to(self, y: V, block: P) -> V:
        r""":math:`y + R^\top` ``block``; ``y`` is not modified."""
        ...


@runtime_checkable
class PortTransferProtocol(Protocol, Generic[V, P]):
    r"""A port transfer :math:`E_i` and its transpose."""

    def nport(self) -> int:
        """The number of rows of the subdomain's port data."""
        ...

    def scatter(self, lam: V) -> P:
        r"""The port data :math:`E_i \lambda`."""
        ...

    def gather_add(self, block: P, y: V) -> V:
        r""":math:`y + E_i^\top` ``block``; ``y`` is not modified."""
        ...


class MatchingNodalTransfer(Generic[V, P]):
    """Port nodes that coincide with skeleton DOFs: :math:`E_i = R`.

    Parameters
    ----------
    rows
        The skeleton rows of the port, in the subdomain's port order.
    """

    def __init__(self, rows: PortRowsProtocol[V, P]) -> None:
        if not isinstance(rows, PortRowsProtocol):
            raise TypeError(
                f"rows must satisfy PortRowsProtocol, got {type(rows).__name__}"
            )
        self._rows = rows

    def nport(self) -> int:
        return self._rows.nrows()

    def scatter(self, lam: V) -> P:
        return self._rows.take(lam)

    def gather_add(self, block: P, y: V) -> V:
        return self._rows.add_to(y, block)


class SplitRowsTransfer(Generic[V, Array]):
    r"""Port data read from one set of rows and output added into
    another, both oriented by a sign: :math:`E_i = s R_{\text{in}}` on
    the way in, :math:`s R_{\text{out}}^\top` on the way out.

    For a skeleton whose unknowns and residual rows are numbered apart
    (:class:`~pyapprox.pde.coupling.kind_layout.KindLayout`'s ``inputs``
    and ``outputs``): a piece reads the unknowns of the kind it receives
    and adds into the rows of the kind it returns. The sign orients a
    piece's data, as the dual condition's signed jump needs; with
    :math:`R_{\text{in}} = R_{\text{out}}` and :math:`s = 1` it is
    :class:`MatchingNodalTransfer`.

    Parameters
    ----------
    inputs
        The rows the port data are read from, in the port order.
    outputs
        The rows the output is added into, in the port order.
    sign
        :math:`s`, ``1.0`` or ``-1.0``.

    Raises
    ------
    TypeError
        If ``inputs`` or ``outputs`` does not satisfy
        :class:`PortRowsProtocol`.
    ValueError
        If their row counts differ or ``sign`` is not ``1.0`` or ``-1.0``.
    """

    def __init__(
        self,
        inputs: PortRowsProtocol[V, Array],
        outputs: PortRowsProtocol[V, Array],
        sign: float = 1.0,
    ) -> None:
        for name, rows in (("inputs", inputs), ("outputs", outputs)):
            if not isinstance(rows, PortRowsProtocol):
                raise TypeError(
                    f"{name} must satisfy PortRowsProtocol, got {type(rows).__name__}"
                )
        if inputs.nrows() != outputs.nrows():
            raise ValueError(
                f"inputs have {inputs.nrows()} rows, outputs {outputs.nrows()}"
            )
        if sign not in (1.0, -1.0):
            raise ValueError(f"sign must be 1.0 or -1.0, got {sign}")
        self._inputs, self._outputs, self._sign = inputs, outputs, sign

    def nport(self) -> int:
        return self._inputs.nrows()

    def scatter(self, lam: V) -> Array:
        return self._sign * self._inputs.take(lam)

    def gather_add(self, block: Array, y: V) -> V:
        return self._outputs.add_to(y, self._sign * block)


@runtime_checkable
class PortMapProtocol(Protocol, Generic[P]):
    """What :class:`PortMatrixTransfer` asks of its local factor: a
    linear map between port blocks, known by its action and transpose."""

    def domain(self) -> SpaceDescriptor:
        """The space of the input of :meth:`apply`: the port's rows."""
        ...

    def range(self) -> SpaceDescriptor:
        """The space of the output of :meth:`apply`: the port data."""
        ...

    def apply(self, x: P) -> P:
        """:math:`L x` for a block ``x`` in the domain."""
        ...

    def apply_transpose(self, y: P) -> P:
        r""":math:`L^\top y` for a block ``y`` in the range."""
        ...


class PortMatrixTransfer(Generic[V, P]):
    r"""A port transfer :math:`E_i = L R`: the skeleton rows the port
    reads, then a local linear map to the port data.

    :math:`R` takes the skeleton DOFs the port reads; :math:`L` maps them
    to the subdomain's port data, e.g. the skeleton function at port nodes
    that are not skeleton DOFs (:func:`port_interpolation`). The transpose
    adds a tested flux back as :math:`R^\top L^\top`, so fluxes stay
    conservative. Neither the skeleton vector type ``V`` nor how
    :math:`L` is stored (dense, face-blocked, low-rank) is inspected:
    which transfer a port uses is the choice of the builder of
    :math:`L`. Ports at skeleton nodes need no map:
    :class:`MatchingNodalTransfer`.

    Parameters
    ----------
    rows
        The skeleton rows the port reads, :math:`R`.
    local
        :math:`L`, with domain size ``rows.nrows()``.

    Raises
    ------
    TypeError
        If ``rows`` or ``local`` does not satisfy its protocol.
    ValueError
        If ``local``'s domain size differs from ``rows.nrows()``.
    """

    def __init__(self, rows: PortRowsProtocol[V, P], local: PortMapProtocol[P]) -> None:
        if not isinstance(rows, PortRowsProtocol):
            raise TypeError(
                f"rows must satisfy PortRowsProtocol, got {type(rows).__name__}"
            )
        if not isinstance(local, PortMapProtocol):
            raise TypeError(
                f"local must satisfy PortMapProtocol, got {type(local).__name__}"
            )
        if local.domain().size != rows.nrows():
            raise ValueError(
                f"local's domain has size {local.domain().size}, the rows "
                f"{rows.nrows()}"
            )
        self._rows, self._local = rows, local

    def nport(self) -> int:
        return self._local.range().size

    def scatter(self, lam: V) -> P:
        return self._local.apply(self._rows.take(lam))

    def gather_add(self, block: P, y: V) -> V:
        return self._rows.add_to(y, self._local.apply_transpose(block))


class NormalProjectionTransfer(Generic[V, Array]):
    r"""A scalar skeleton read by a vector port: :math:`E = s\, (n \otimes)
    E_0`, the scalar transfer :math:`E_0`'s port values spread along each
    port node's unit normal, so its transpose :math:`E^\top = s\,
    E_0^\top (n \cdot)` reads the normal component of a vector block back.

    For a scalar interface unknown (a pressure, a normal velocity) beside
    a subdomain whose port data and output are vectors (a traction in, a
    velocity out). Port data and output are node-major: row :math:`k a +
    c` is component :math:`c` at port node :math:`a`, :math:`k` the
    dimension. The sign :math:`s` orients the port, as signs do in port
    transfers: e.g. :math:`s = -1` makes the traction :math:`-p\, n` of an
    interface pressure :math:`p`, and then gathers :math:`-n \cdot u`.
    The normals enter by a constant product, so a graph through the
    skeleton values survives.

    Parameters
    ----------
    inner
        :math:`E_0`, the scalar transfer, one port value per node.
    normals
        The unit normal at each of ``inner``'s port nodes. Shape: (nnodes,
        ndim).
    bkd
        The backend of the port blocks.
    sign
        :math:`s`.

    Raises
    ------
    TypeError
        If ``inner`` does not satisfy :class:`PortTransferProtocol`.
    ValueError
        If ``normals`` is not (inner.nport(), ndim) or a normal is not of
        unit length.
    """

    def __init__(
        self,
        inner: PortTransferProtocol[V, Array],
        normals: Array,
        bkd: Backend[Array],
        sign: float = 1.0,
    ) -> None:
        if not isinstance(inner, PortTransferProtocol):
            raise TypeError(
                f"inner must satisfy PortTransferProtocol, got {type(inner).__name__}"
            )
        nnodes = inner.nport()
        if normals.ndim != 2 or normals.shape[0] != nnodes:
            raise ValueError(
                f"normals must have shape ({nnodes}, ndim), got {tuple(normals.shape)}"
            )
        lengths = bkd.sqrt(bkd.sum(normals**2, axis=1))
        if bkd.to_float(bkd.max(bkd.abs(lengths - 1.0))) > 1e-12:
            raise ValueError("the normals must be of unit length")
        ndim = int(normals.shape[1])
        # (ndim * nnodes, nnodes): row k a + c holds n_c at node a.
        placement = bkd.asarray(
            [
                [1.0 if b == a else 0.0 for b in range(nnodes)]
                for a in range(nnodes)
                for _ in range(ndim)
            ]
        )
        spread = sign * (bkd.reshape(normals, (-1,))[:, None] * placement)
        self._inner, self._spread = inner, spread
        self._read = bkd.transpose(spread)
        self._nport = ndim * nnodes

    def nport(self) -> int:
        return self._nport

    def scatter(self, lam: V) -> Array:
        return self._spread @ self._inner.scatter(lam)

    def gather_add(self, block: Array, y: V) -> V:
        return self._inner.gather_add(self._read @ block, y)


class TimeComponentTransfer(Generic[Array]):
    r"""A transfer on a skeleton carrying ``ncomponents`` values per DOF,
    built from one on the plain skeleton: :math:`E \otimes I`, each
    component transferred alone.

    For a skeleton unknown in a space in time per window (one trace per
    unknown of the trial space), laid out as a
    :class:`~pyapprox.pde.coupling.spaces.ComponentSkeletonSpace` lays out
    its vectors and as the windowed port data are: entry ``q * d + k`` is
    DOF (or port row) ``d``'s component ``k`` of ``q``. Any plain transfer
    (interpolation, projection, a normal projection) then serves a space
    in time unchanged.

    Parameters
    ----------
    inner
        :math:`E`, the transfer on the plain skeleton.
    ncomponents
        :math:`q`, the values per DOF.
    bkd
        The backend of the skeleton vectors and port blocks.

    Raises
    ------
    TypeError
        If ``inner`` does not satisfy :class:`PortTransferProtocol`.
    ValueError
        If ``ncomponents`` is less than 1.
    """

    def __init__(
        self,
        inner: PortTransferProtocol[Array, Array],
        ncomponents: int,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(inner, PortTransferProtocol):
            raise TypeError(
                f"inner must satisfy PortTransferProtocol, got {type(inner).__name__}"
            )
        if ncomponents < 1:
            raise ValueError(f"ncomponents must be at least 1, got {ncomponents}")
        self._inner, self._q, self._bkd = inner, ncomponents, bkd

    def _interleaved(self, parts: list[Array]) -> Array:
        stacked = self._bkd.stack(parts, axis=1)
        n, q, ncols = stacked.shape
        return self._bkd.reshape(stacked, (n * q, ncols))

    def nport(self) -> int:
        return self._inner.nport() * self._q

    def scatter(self, lam: Array) -> Array:
        q = self._q
        return self._interleaved([self._inner.scatter(lam[k::q]) for k in range(q)])

    def gather_add(self, block: Array, y: Array) -> Array:
        q, bkd = self._q, self._bkd
        nskel, ncols = int(y.shape[0]) // q, int(block.shape[1])
        zero = bkd.zeros((nskel, ncols))
        parts = [self._inner.gather_add(block[k::q], zero) for k in range(q)]
        return y + self._interleaved(parts)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._inner!r}, ncomponents={self._q})"


@runtime_checkable
class FaceValuesProtocol(Protocol, Generic[Array]):
    """What :func:`port_interpolation` asks of a skeleton field space: its
    functions on a face at reference points, and the backend of their
    reference data."""

    def face_values(self, face: int, points: Array) -> FaceValues[Array]: ...

    def bkd(self) -> Backend[Array]: ...


@runtime_checkable
class CrossPointRuleProtocol(Protocol, Generic[Array]):
    """How a port node lying on several faces' closure reads the skeleton.

    Each incident face gives the node a row: the values, shape
    (k, ndofs), of the port's skeleton functions there through that
    face. On a continuous space the rows agree; on a facet space they
    differ, and the rule decides.
    """

    def row(self, node: int, faces: Sequence[int], rows: Sequence[Array]) -> Array:
        """The node's row from its incident faces' rows (at least two)."""
        ...


class AverageIncidentFaces(Generic[Array]):
    """A cross-point node reads the mean of its incident faces' values."""

    def row(self, node: int, faces: Sequence[int], rows: Sequence[Array]) -> Array:
        total = rows[0]
        for r in rows[1:]:
            total = total + r
        return total / float(len(rows))


class RefuseCrossPoints(Generic[Array]):
    """A port node on several faces' closure is an error."""

    def row(self, node: int, faces: Sequence[int], rows: Sequence[Array]) -> Array:
        raise ValueError(f"port node {node} lies on several faces {tuple(faces)}")


def _interpolation_rows(
    space: FaceValuesProtocol[Array],
    located: Sequence[Sequence[FacePoint[Array]]],
    dofs: Sequence[int],
    cross_points: CrossPointRuleProtocol[Array],
) -> Array:
    """The skeleton functions of ``dofs`` at each located point, as exchange
    data: shape (k * npoints, len(dofs)), node-major (point ``n``,
    component ``c`` is row ``k n + c``, as interleaved field DOFs are
    numbered).

    A point on one face reads that face's values; a point on several asks
    ``cross_points``. The values enter by products with constant 0/1
    placement arrays, so a graph through them survives.

    Raises
    ------
    TypeError
        If ``space`` or ``cross_points`` does not satisfy its protocol.
    ValueError
        If ``located`` is empty or holds a point on no face, or a face's
        DOFs are not all in ``dofs``.
    """
    if not isinstance(space, FaceValuesProtocol):
        raise TypeError(
            f"space must satisfy FaceValuesProtocol, got {type(space).__name__}"
        )
    if not isinstance(cross_points, CrossPointRuleProtocol):
        raise TypeError(
            "cross_points must satisfy CrossPointRuleProtocol, got "
            f"{type(cross_points).__name__}"
        )
    if not located:
        raise ValueError("the port must hold at least one point")
    bkd = space.bkd()
    column = {dof: j for j, dof in enumerate(dofs)}
    node_rows = []
    for node, faces in enumerate(located):
        if not faces:
            raise ValueError(f"port node {node} lies on no face")
        rows = []
        for fp in faces:
            fv = space.face_values(fp.face, fp.point)
            missing = [d for d in fv.dofs if d not in column]
            if missing:
                raise ValueError(
                    f"face {fp.face} reads DOFs {missing} outside the port's DOFs"
                )
            # (ndofs_face, len(dofs)) placement of the face's DOFs.
            placement = [
                [1.0 if column[d] == j else 0.0 for j in range(len(dofs))]
                for d in fv.dofs
            ]
            rows.append(fv.values[0].T @ bkd.asarray(placement))
        if len(rows) == 1:
            node_rows.append(rows[0])
        else:
            node_rows.append(cross_points.row(node, [fp.face for fp in faces], rows))
    return bkd.concatenate(node_rows, axis=0)


@runtime_checkable
class PortBlockAdapterProtocol(Protocol, Generic[Array, P]):
    """A port stack's one conversion of exchange blocks (interface-sized
    backend arrays, as the skeleton's port rows deliver them) to and from
    its own block type ``P``. Each port stack has one, whatever the
    skeleton's stack is."""

    def to_port(self, block: Array, descriptor: SpaceDescriptor) -> P:
        """``block``, shape (descriptor.size, ncols), as a block of ``P``."""
        ...

    def to_exchange(self, block: P) -> Array:
        """``block`` as an exchange block, shape (size, ncols)."""
        ...


class AdaptedPortRows(Generic[V, Array, P]):
    """Port rows delivering blocks of a port stack ``P``: the skeleton's
    exchange rows, converted at the port block by the port stack's
    adapter.

    Satisfies :class:`PortRowsProtocol` with ``P`` as the port block, so a
    :class:`PortMatrixTransfer` whose local map acts on ``P`` reads a
    skeleton of any stack.

    Parameters
    ----------
    rows
        The skeleton's rows of the port, delivering exchange blocks.
    adapter
        The port stack's adapter.
    descriptor
        The space of the port blocks, of size ``rows.nrows()``: the
        domain of the transfer's local map.

    Raises
    ------
    TypeError
        If ``rows`` or ``adapter`` does not satisfy its protocol.
    ValueError
        If ``descriptor``'s size differs from ``rows.nrows()``.
    """

    def __init__(
        self,
        rows: PortRowsProtocol[V, Array],
        adapter: PortBlockAdapterProtocol[Array, P],
        descriptor: SpaceDescriptor,
    ) -> None:
        if not isinstance(rows, PortRowsProtocol):
            raise TypeError(
                f"rows must satisfy PortRowsProtocol, got {type(rows).__name__}"
            )
        if not isinstance(adapter, PortBlockAdapterProtocol):
            raise TypeError(
                "adapter must satisfy PortBlockAdapterProtocol, got "
                f"{type(adapter).__name__}"
            )
        if descriptor.size != rows.nrows():
            raise ValueError(
                f"descriptor has size {descriptor.size}, the rows {rows.nrows()}"
            )
        self._rows, self._adapter, self._descriptor = rows, adapter, descriptor

    def nrows(self) -> int:
        return self._rows.nrows()

    def take(self, x: V) -> P:
        return self._adapter.to_port(self._rows.take(x), self._descriptor)

    def add_to(self, y: V, block: P) -> V:
        return self._rows.add_to(y, self._adapter.to_exchange(block))


@runtime_checkable
class PortTraceProtocol(Protocol, Generic[Array]):
    r"""What the projection builders ask of a port: its trace functions
    :math:`\phi` and the quadrature of the port, element by element.

    Elements are batched, each with the same number of local trace
    functions and quadrature points. The trace functions are scalar; a
    port of :math:`k`-vector data carries :math:`\phi_a e_c`, numbered
    node-major (port data row :math:`k a + c`).
    """

    def bkd(self) -> Backend[Array]: ...

    def nnodes(self) -> int:
        """The number of trace functions: the port's nodes."""
        ...

    def element_nodes(self) -> Array:
        """Each element's port nodes, int64. Shape: (nelems, nlocal)."""
        ...

    def points(self) -> Array:
        """The physical quadrature points, element-major (point
        ``e * nquad + q``). Shape: (ndim, nelems * nquad)."""
        ...

    def weights(self) -> Array:
        """The quadrature weights in the physical measure. Shape:
        (nelems, nquad)."""
        ...

    def values(self) -> Array:
        """Each element's trace functions at its points. Shape: (nelems,
        nquad, nlocal)."""
        ...


@runtime_checkable
class PointLocatorProtocol(Protocol, Generic[Array]):
    """What :func:`port_interpolation` and :func:`port_trace_coupling` ask
    of a geometry: where physical points lie on the skeleton's faces."""

    def locate(
        self, points: Array, faces: Sequence[int], tol: float = 1e-12
    ) -> tuple[tuple[FacePoint[Array], ...], ...]: ...


@runtime_checkable
class PortMatrixSinkProtocol(Protocol, Generic[P, F_co, I_contra, E_contra]):
    """What the port builders ask of a port stack's matrix sink:
    element blocks in, the stack's operator out. ``I_contra`` is the
    array type of DOF maps, ``E_contra`` that of element values."""

    def add_element_matrices(
        self,
        row_dofs: I_contra,
        col_dofs: I_contra,
        values: E_contra,
        weights: Optional[E_contra] = None,
    ) -> None: ...

    def assemble(
        self, properties: Optional[OperatorProperties] = None
    ) -> ActionOperatorProtocol[P, F_co]: ...


@runtime_checkable
class PortAssemblyProtocol(Protocol, Generic[P, F_co, I_contra, E_contra]):
    """What the port builders (:func:`port_interpolation`,
    :func:`port_trace_mass`, :func:`port_trace_coupling`) ask of a port
    stack: a matrix sink for a pattern of element blocks, e.g. an
    assembly stack of :mod:`pyapprox.pde.assembly` or, for a torch graph
    through the port, a
    :class:`~pyapprox.pde.assembly.backend_dense_sink.BackendDenseMatrixAssembly`."""

    def matrix_sink(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: I_contra,
        col_dofs: I_contra,
    ) -> PortMatrixSinkProtocol[P, F_co, I_contra, E_contra]: ...


def trace_shape(trace: PortTraceProtocol[Array]) -> tuple[int, int, int]:
    """A trace's ``(nelems, nquad, nlocal)``, checked across its arrays.

    Raises
    ------
    TypeError
        If ``trace`` does not satisfy :class:`PortTraceProtocol`.
    ValueError
        If its arrays disagree in shape or it has no elements.
    """
    if not isinstance(trace, PortTraceProtocol):
        raise TypeError(
            f"trace must satisfy PortTraceProtocol, got {type(trace).__name__}"
        )
    nodes, weights, values = trace.element_nodes(), trace.weights(), trace.values()
    if nodes.ndim != 2 or nodes.shape[0] == 0:
        raise ValueError(
            f"element_nodes must have shape (nelems, nlocal), got {tuple(nodes.shape)}"
        )
    nelems, nlocal = int(nodes.shape[0]), int(nodes.shape[1])
    if weights.ndim != 2 or weights.shape[0] != nelems:
        raise ValueError(
            f"weights must have shape ({nelems}, nquad), got {tuple(weights.shape)}"
        )
    nquad = int(weights.shape[1])
    if tuple(values.shape) != (nelems, nquad, nlocal):
        raise ValueError(
            f"values must have shape ({nelems}, {nquad}, {nlocal}), "
            f"got {tuple(values.shape)}"
        )
    return nelems, nquad, nlocal


def _component_rows(trace: PortTraceProtocol[Array], ncomponents: int) -> Array:
    """Each element's port data rows, node-major. Shape: (nelems,
    nlocal * ncomponents)."""
    bkd = trace.bkd()
    nodes = trace.element_nodes()
    offsets = bkd.asarray(list(range(ncomponents)), dtype=bkd.index_dtype())
    rows = nodes[:, :, None] * ncomponents + offsets[None, None, :]
    return bkd.reshape(rows, (int(nodes.shape[0]), -1))


def port_trace_mass(
    trace: PortTraceProtocol[Array],
    ncomponents: int,
    assembly: PortAssemblyProtocol[P, F, Array, Array],
) -> ActionOperatorProtocol[P, F]:
    r"""The port's trace mass :math:`M_p = \int \phi^\top \phi`, assembled
    by the port stack and declared symmetric positive definite.

    Each element adds :math:`\Phi_e^\top W_e \Phi_e \otimes I_k` at its
    port data rows (node-major), computed from the trace's backend
    arrays; ``assembly`` chooses the storage.

    Parameters
    ----------
    trace
        The port's trace functions and quadrature.
    ncomponents
        :math:`k`, the components of the port data.
    assembly
        The port stack's assembly.

    Returns
    -------
    ActionOperatorProtocol
        On the ``"port"`` space of size ``k * trace.nnodes()``.

    Raises
    ------
    TypeError
        If ``trace`` or ``assembly`` does not satisfy its protocol.
    ValueError
        If ``ncomponents`` is not positive or the trace's arrays disagree
        in shape.
    """
    nelems, _, nlocal = trace_shape(trace)
    if not isinstance(assembly, PortAssemblyProtocol):
        raise TypeError(
            f"assembly must satisfy PortAssemblyProtocol, got {type(assembly).__name__}"
        )
    if ncomponents < 1:
        raise ValueError(f"ncomponents must be >= 1, got {ncomponents}")
    bkd, k = trace.bkd(), ncomponents
    phi, weights = trace.values(), trace.weights()
    scalar = bkd.einsum("eq,eqa,eqb->eab", weights, phi, phi)
    eye = bkd.eye(k, dtype=bkd.real_dtype())
    blocks = bkd.reshape(
        bkd.einsum("eab,cd->eacbd", scalar, eye), (nelems, nlocal * k, nlocal * k)
    )
    rows = _component_rows(trace, k)
    port = SpaceDescriptor("port", k * trace.nnodes(), backend_dtype_name(bkd))
    sink = assembly.matrix_sink(port, port, rows, rows)
    sink.add_element_matrices(rows, rows, blocks)
    return sink.assemble(OperatorProperties(symmetric=True, spd=True))


def port_interpolation(
    space: FaceValuesProtocol[Array],
    points: Array,
    geometry: PointLocatorProtocol[Array],
    faces: Sequence[int],
    dofs: Sequence[int],
    cross_points: CrossPointRuleProtocol[Array],
    assembly: PortAssemblyProtocol[P, F, Array, Array],
) -> ActionOperatorProtocol[P, F]:
    r"""The local map :math:`L` of a :class:`PortMatrixTransfer` that
    interpolates: the skeleton functions of ``dofs`` at each port node,
    assembled by the port stack.

    The port nodes are located on ``faces``; node ``n``'s rows (node-major,
    component ``c`` is row ``k n + c``) are one element block. A node on
    one face reads that face's values; a node on several asks
    ``cross_points``. Each block spans every column, so the pattern is
    dense, rows times ``dofs``, as in :func:`port_trace_coupling`; a
    face-blocked pattern is the remedy for large ports.

    Parameters
    ----------
    space
        The skeleton field space, of :math:`k` components.
    points
        The port nodes, in the order of the port data. Shape: (ndim,
        nnodes).
    geometry
        Locates the port nodes on the skeleton's faces.
    faces
        The faces the port lies on.
    dofs
        The skeleton DOFs the port reads, in the order of the transfer's
        rows :math:`R`.
    cross_points
        The rule for nodes on several faces.
    assembly
        The port stack's assembly.

    Returns
    -------
    ActionOperatorProtocol
        From the ``"port-rows"`` space of size ``len(dofs)`` to the
        ``"port"`` space of size ``k * nnodes``.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If there are no port nodes, a node lies on none of ``faces``, or a
        face's DOFs are not all in ``dofs``.
    """
    if not isinstance(geometry, PointLocatorProtocol):
        raise TypeError(
            f"geometry must satisfy PointLocatorProtocol, got {type(geometry).__name__}"
        )
    if not isinstance(assembly, PortAssemblyProtocol):
        raise TypeError(
            f"assembly must satisfy PortAssemblyProtocol, got {type(assembly).__name__}"
        )
    matrix = _interpolation_rows(
        space, geometry.locate(points, faces), dofs, cross_points
    )
    bkd = space.bkd()
    nnodes, ncols = int(points.shape[1]), len(dofs)
    k = int(matrix.shape[0]) // nnodes
    blocks = bkd.reshape(matrix, (nnodes, k, ncols))
    rows = bkd.asarray(
        [[k * n + c for c in range(k)] for n in range(nnodes)],
        dtype=bkd.index_dtype(),
    )
    columns = bkd.asarray([list(range(ncols))] * nnodes, dtype=bkd.index_dtype())
    dtype = backend_dtype_name(bkd)
    sink = assembly.matrix_sink(
        SpaceDescriptor("port", k * nnodes, dtype),
        SpaceDescriptor("port-rows", ncols, dtype),
        rows,
        columns,
    )
    sink.add_element_matrices(rows, columns, blocks)
    return sink.assemble()


def port_trace_coupling(
    space: FaceValuesProtocol[Array],
    trace: PortTraceProtocol[Array],
    geometry: PointLocatorProtocol[Array],
    faces: Sequence[int],
    dofs: Sequence[int],
    cross_points: CrossPointRuleProtocol[Array],
    assembly: PortAssemblyProtocol[P, F, Array, Array],
) -> ActionOperatorProtocol[P, F]:
    r"""The coupling :math:`B = \int \phi^\top \psi` of the port's trace
    functions with the skeleton functions of ``dofs``, assembled by the
    port stack.

    The trace's quadrature points are located on ``faces`` and the
    skeleton functions evaluated there as exchange data
    (:func:`port_interpolation`'s rows, node-major per point); each
    element adds :math:`\Phi_e^\top W_e \Psi_e` at its port data rows.
    Each element's block spans every column, so the pattern is dense,
    rows times ``dofs``; a face-blocked pattern is the remedy for large
    ports.

    Parameters
    ----------
    space
        The skeleton field space, of :math:`k` components.
    trace
        The port's trace functions and quadrature.
    geometry
        Locates the trace's points on the skeleton's faces.
    faces
        The faces the port lies on.
    dofs
        The skeleton DOFs the port reads, in the order of the transfer's
        rows.
    cross_points
        The rule for quadrature points on several faces.
    assembly
        The port stack's assembly.

    Returns
    -------
    ActionOperatorProtocol
        From the ``"port-rows"`` space of size ``len(dofs)`` to the
        ``"port"`` space of size ``k * trace.nnodes()``.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If the trace's arrays disagree in shape, a point lies on none of
        ``faces``, or a face's DOFs are not all in ``dofs``.
    """
    nelems, nquad, _ = trace_shape(trace)
    if not isinstance(geometry, PointLocatorProtocol):
        raise TypeError(
            f"geometry must satisfy PointLocatorProtocol, got {type(geometry).__name__}"
        )
    if not isinstance(assembly, PortAssemblyProtocol):
        raise TypeError(
            f"assembly must satisfy PortAssemblyProtocol, got {type(assembly).__name__}"
        )
    psi = _interpolation_rows(
        space, geometry.locate(trace.points(), faces), dofs, cross_points
    )
    k = int(psi.shape[0]) // (nelems * nquad)
    psi = trace.bkd().reshape(psi, (nelems, nquad, k, len(dofs)))
    return _tested_coupling(trace, psi, "port", assembly)


def _tested_coupling(
    trace: PortTraceProtocol[Array],
    psi: Array,
    range_name: str,
    assembly: PortAssemblyProtocol[P, F, Array, Array],
) -> ActionOperatorProtocol[P, F]:
    r""":math:`\int \phi^\top \psi` for the trace functions :math:`\phi` and
    the columns ``psi`` at the trace's points, shape (nelems, nquad, k,
    ncols), assembled into the ``range_name`` space by the port stack."""
    bkd = trace.bkd()
    nelems, _, k, ncols = (int(n) for n in psi.shape)
    nlocal = int(trace.element_nodes().shape[1])
    blocks = bkd.reshape(
        bkd.einsum("eq,eqa,eqcj->eacj", trace.weights(), trace.values(), psi),
        (nelems, nlocal * k, ncols),
    )
    rows = _component_rows(trace, k)
    columns = bkd.asarray([list(range(ncols))] * nelems, dtype=bkd.index_dtype())
    dtype = backend_dtype_name(bkd)
    sink = assembly.matrix_sink(
        SpaceDescriptor(range_name, k * trace.nnodes(), dtype),
        SpaceDescriptor("port-rows", ncols, dtype),
        rows,
        columns,
    )
    sink.add_element_matrices(rows, columns, blocks)
    return sink.assemble()


def _same_points(
    first: PortTraceProtocol[Array], second: PortTraceProtocol[Array]
) -> None:
    """Refuse two traces whose quadrature points or weights differ."""
    bkd = first.bkd()
    if tuple(first.weights().shape) != tuple(second.weights().shape) or tuple(
        first.points().shape
    ) != tuple(second.points().shape):
        raise ValueError("the traces must share their quadrature points and weights")
    gap = bkd.max(bkd.abs(first.points() - second.points())) + bkd.max(
        bkd.abs(first.weights() - second.weights())
    )
    if bkd.to_float(gap) > 1e-12:
        raise ValueError("the traces must share their quadrature points and weights")


def port_trace_pairing(
    test: PortTraceProtocol[Array],
    trial: PortTraceProtocol[Array],
    ncomponents: int,
    assembly: PortAssemblyProtocol[P, F, Array, Array],
) -> ActionOperatorProtocol[P, F]:
    r"""The pairing :math:`\tilde M = \int \tilde\psi^\top \psi \otimes I_k`
    of a port's test functions with its trial functions, assembled by the
    port stack: the mass of a mortar projection, square when there are as
    many test as trial functions and not symmetric in general.

    Parameters
    ----------
    test, trial
        The two traces, on the same quadrature points and weights and
        with the same elements.
    ncomponents
        :math:`k`, the components of the port data.
    assembly
        The port stack's assembly.

    Returns
    -------
    ActionOperatorProtocol
        From the ``"port"`` space of size ``k * trial.nnodes()`` to the
        ``"port-test"`` space of size ``k * test.nnodes()``.

    Raises
    ------
    TypeError
        If a trace or ``assembly`` does not satisfy its protocol.
    ValueError
        If ``ncomponents`` is not positive, a trace's arrays disagree in
        shape, or the traces differ in elements, points or weights.
    """
    nelems, _, nlocal = trace_shape(test)
    if trace_shape(trial)[0] != nelems:
        raise ValueError("the traces must have the same elements")
    _same_points(test, trial)
    if not isinstance(assembly, PortAssemblyProtocol):
        raise TypeError(
            f"assembly must satisfy PortAssemblyProtocol, got {type(assembly).__name__}"
        )
    if ncomponents < 1:
        raise ValueError(f"ncomponents must be >= 1, got {ncomponents}")
    bkd, k = test.bkd(), ncomponents
    mlocal = int(trial.element_nodes().shape[1])
    scalar = bkd.einsum(
        "eq,eqa,eqb->eab", test.weights(), test.values(), trial.values()
    )
    eye = bkd.eye(k, dtype=bkd.real_dtype())
    blocks = bkd.reshape(
        bkd.einsum("eab,cd->eacbd", scalar, eye), (nelems, nlocal * k, mlocal * k)
    )
    rows, columns = _component_rows(test, k), _component_rows(trial, k)
    dtype = backend_dtype_name(bkd)
    sink = assembly.matrix_sink(
        SpaceDescriptor("port-test", k * test.nnodes(), dtype),
        SpaceDescriptor("port", k * trial.nnodes(), dtype),
        rows,
        columns,
    )
    sink.add_element_matrices(rows, columns, blocks)
    return sink.assemble()


def port_trace_mortar_coupling(
    space: FaceValuesProtocol[Array],
    test: PortTraceProtocol[Array],
    ends: PortTraceProtocol[Array],
    end_points: Array,
    geometry: PointLocatorProtocol[Array],
    faces: Sequence[int],
    dofs: Sequence[int],
    cross_points: CrossPointRuleProtocol[Array],
    assembly: PortAssemblyProtocol[P, F, Array, Array],
) -> ActionOperatorProtocol[P, F]:
    r"""The coupling :math:`\hat B = \int \tilde\psi^\top (\phi - \sum_e
    \phi(x_e)\, \psi_e)` of a mortar projection: the port's test functions
    against the skeleton functions of ``dofs`` less their values at the
    fixed ends ``x_e`` carried by the end functions :math:`\psi_e`.

    With :func:`port_trace_pairing` as the mass, :func:`port_projection`
    gives :math:`L = \tilde M^{-1} \hat B`: the port data :math:`u` with
    :math:`\int \tilde\psi\, (u_h - \lambda) = 0`, where :math:`u_h` is the
    trial function of the data plus the end functions at the skeleton's
    end values; its transpose tests a flux recovered in the span of the
    test functions, constants included, against the skeleton functions.

    Parameters
    ----------
    space
        The skeleton field space, of :math:`k` components.
    test, ends
        The test and end traces (:func:`~pyapprox.pde.coupling.traces.
        mortar_segment_traces`), on the same points and weights.
    end_points
        The ends' points, in the end trace's node order. Shape: (ndim,
        nends).
    geometry
        Locates points on the skeleton's faces.
    faces
        The faces the port lies on.
    dofs
        The skeleton DOFs the port reads, in the order of the transfer's
        rows.
    cross_points
        The rule for points on several faces.
    assembly
        The port stack's assembly.

    Returns
    -------
    ActionOperatorProtocol
        From the ``"port-rows"`` space of size ``len(dofs)`` to the
        ``"port-test"`` space of size ``k * test.nnodes()``.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If a trace's arrays disagree in shape, the traces differ in
        elements, points or weights, ``end_points`` has not one column per
        end, a point lies on none of ``faces``, or a face's DOFs are not
        all in ``dofs``.
    """
    nelems, nquad, _ = trace_shape(test)
    if trace_shape(ends)[0] != nelems:
        raise ValueError("the traces must have the same elements")
    _same_points(test, ends)
    if not isinstance(geometry, PointLocatorProtocol):
        raise TypeError(
            f"geometry must satisfy PointLocatorProtocol, got {type(geometry).__name__}"
        )
    if not isinstance(assembly, PortAssemblyProtocol):
        raise TypeError(
            f"assembly must satisfy PortAssemblyProtocol, got {type(assembly).__name__}"
        )
    nends = ends.nnodes()
    if end_points.ndim != 2 or end_points.shape[1] != nends:
        raise ValueError(
            f"end_points must have shape (ndim, {nends}), got {tuple(end_points.shape)}"
        )
    bkd, ncols = test.bkd(), len(dofs)
    psi = _interpolation_rows(
        space, geometry.locate(test.points(), faces), dofs, cross_points
    )
    k = int(psi.shape[0]) // (nelems * nquad)
    psi = bkd.reshape(psi, (nelems, nquad, k, ncols))
    at_ends = bkd.reshape(
        _interpolation_rows(
            space, geometry.locate(end_points, faces), dofs, cross_points
        ),
        (nends, k, ncols),
    )
    # Each slot's end function at the points, placed in its end's column
    # by a constant 0/1 product, so a graph through the values survives.
    slots = ends.element_nodes()
    placement = bkd.asarray(
        [
            [
                [1.0 if bkd.to_int(slots[e, a]) == n else 0.0 for n in range(nends)]
                for a in range(int(slots.shape[1]))
            ]
            for e in range(nelems)
        ]
    )
    end_values = bkd.einsum("eqa,ean->eqn", ends.values(), placement)
    corrected = psi - bkd.einsum("eqn,ncj->eqcj", end_values, at_ends)
    return _tested_coupling(test, corrected, "port-test", assembly)


def port_projection(
    mass: ActionOperatorProtocol[P, F],
    coupling: ActionOperatorProtocol[P, F],
    solver: LinearSolverProtocol[P, F],
) -> Product[P]:
    r"""The local map :math:`L = M_p^{-1} B` of a :class:`PortMatrixTransfer`
    that projects: the port data are the :math:`L^2` projection of the
    skeleton function onto the port's trace space.

    Composed, never formed: :math:`M_p^{-1} B` is dense even when
    :math:`M_p` and :math:`B` are sparse. ``apply`` is
    :math:`x \mapsto M_p^{-1}(B x)` and ``apply_transpose`` is
    :math:`y \mapsto B^\top M_p^{-\top} y`, which recovers the flux
    function from a flux tested against the trace functions and tests it
    against the skeleton functions. ``solver`` factors :math:`M_p` once,
    and must solve with its transpose.

    A port whose ends are fixed (an outer Dirichlet condition) takes the
    mortar pair instead: :func:`port_trace_pairing` of its test and trial
    traces as the mass and :func:`port_trace_mortar_coupling` as the
    coupling.

    Parameters
    ----------
    mass
        :math:`M_p`, e.g. from :func:`port_trace_mass`, or the mortar
        pairing.
    coupling
        :math:`B`, e.g. from :func:`port_trace_coupling`, or the mortar
        coupling.
    solver
        A solver for ``mass``'s format.

    Raises
    ------
    ValueError
        If ``coupling``'s range is not ``mass``'s.
    """
    handle = solver.setup(mass)
    return Product([handle, coupling])


class BackendPortRows(Generic[Array]):
    """Port rows of a skeleton stored as backend arrays.

    Satisfies :class:`PortRowsProtocol` with backend arrays as both the
    skeleton vectors and the port blocks. Both operations index the
    port rows, costing O(nport) per column; the add writes through the
    backend's functional ``index_update`` (the rows are distinct), so a
    torch graph through the operands is preserved.

    Parameters
    ----------
    descriptor
        The skeleton space.
    dofs
        The skeleton DOF of each port row, distinct.
    bkd
        Computational backend.
    """

    def __init__(
        self, descriptor: SpaceDescriptor, dofs: Sequence[int], bkd: Backend[Array]
    ) -> None:
        if len(set(dofs)) != len(dofs) or any(
            not 0 <= d < descriptor.size for d in dofs
        ):
            raise ValueError(
                f"dofs must be distinct and lie in [0, {descriptor.size}), "
                f"got {tuple(dofs)}"
            )
        self._descriptor = descriptor
        self._bkd = bkd
        self._rows = bkd.asarray(list(dofs), dtype=bkd.index_dtype())
        self._nrows = len(dofs)

    def nrows(self) -> int:
        return self._nrows

    def _check(self, x: Array, rows: int, name: str) -> None:
        if x.ndim != 2 or x.shape[0] != rows:
            raise ValueError(
                f"{name} must have shape ({rows}, ncols), got {tuple(x.shape)}"
            )

    def take(self, x: Array) -> Array:
        self._check(x, self._descriptor.size, "x")
        return x[self._rows]

    def add_to(self, y: Array, block: Array) -> Array:
        self._check(y, self._descriptor.size, "y")
        self._check(block, self._nrows, "block")
        return self._bkd.index_update(y, self._rows, y[self._rows] + block)


@runtime_checkable
class EntityDofsProtocol(Protocol):
    """What :func:`subdomain_port_dofs` asks of a skeleton space: the DOFs
    each skeleton entity carries. Any discretization of the interface
    provides it, whatever its faces' shape or basis."""

    def entity_dofs(self, entity: int) -> tuple[int, ...]:
        """The DOFs on ``entity`` itself, empty if it carries none."""
        ...


@runtime_checkable
class SubdomainEntitiesProtocol(Protocol):
    """What :func:`subdomain_port_dofs` asks of a partition: the skeleton
    entities on each subdomain's boundary."""

    def subdomain_entities(self, subdomain: int) -> tuple[int, ...]:
        """The skeleton entities on ``subdomain``'s boundary."""
        ...


def subdomain_port_dofs(
    topology: SubdomainEntitiesProtocol, space: EntityDofsProtocol, subdomain: int
) -> tuple[int, ...]:
    """The skeleton DOFs on ``subdomain``'s closure entities, increasing.

    Raises
    ------
    TypeError
        If ``topology`` or ``space`` does not satisfy its protocol.
    """
    if not isinstance(topology, SubdomainEntitiesProtocol):
        raise TypeError(
            "topology must satisfy SubdomainEntitiesProtocol, got "
            f"{type(topology).__name__}"
        )
    if not isinstance(space, EntityDofsProtocol):
        raise TypeError(
            f"space must satisfy EntityDofsProtocol, got {type(space).__name__}"
        )
    return tuple(
        sorted(
            dof
            for entity in topology.subdomain_entities(subdomain)
            for dof in space.entity_dofs(entity)
        )
    )
