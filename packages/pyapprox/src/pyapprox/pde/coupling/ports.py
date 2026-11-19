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

from typing import Generic, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import SpaceDescriptor

V = TypeVar("V")
P = TypeVar("P")


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
        self._rows = bkd.asarray(list(dofs), dtype=bkd.int64_dtype())
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
