r"""Protocols for stack-owned vectors and the operations on them.

A *linear-algebra stack* owns vectors, operators and solvers that work
together. Generic code is written against these protocols and is
parameterized by the stack's vector type ``V``; it never indexes,
slices, reshapes or multiplies a vector directly, so a stack may store
vectors in any way, including distributed across processes.

**Blocks of vectors.** The unit is a block of ``ncols`` vectors of one
space (a multivector); a single vector is a block with one column.
Reductions of blocks are small dense arrays of type ``A``, replicated
on every process.

**The optional ``out`` argument is a hint.** Every operation returns
its result, and callers always use the returned value::

    y = space.axpby(a, x, b, y, out=y)

A stack that writes in place writes into ``out`` and returns it; a stack
that cannot (for example one whose arrays record operations for
automatic differentiation) ignores ``out`` and returns a new vector.
Either way the calling code is the same. An in-place interface can always
be used functionally, but not the reverse, which is why ``out`` exists.

**Aliasing.** An operation may write into the operand it accumulates
into, so ``out`` may be ``y`` in :meth:`VectorSpaceProtocol.axpby` and
``x`` in :meth:`VectorSpaceProtocol.scale`. Any other ``out`` must share
no storage with an operand. In particular ``out`` is never the ``x`` of
``axpby`` or ``copy``, and an operator's ``apply`` never writes into its
own input: an in-place stack would read input it had already
overwritten. Stacks may check this and raise ``ValueError``.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, Self, TypeVar, runtime_checkable

from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor

V = TypeVar("V")
A_co = TypeVar("A_co", covariant=True)
F_co = TypeVar("F_co", covariant=True)


@dataclass(frozen=True)
class OperatorProperties:
    r"""Structural properties an operator declares.

    Declared by whoever builds the operator and kept separate from its
    storage, so a matrix-free operator can declare them too. Solvers rely
    on them (a conjugate-gradient solver needs ``spd``; a solver may use
    its forward solve as the transpose only when ``symmetric``), so each
    implies the weaker ones, and inconsistent combinations are refused.

    Parameters
    ----------
    symmetric : bool
        :math:`A = A^\top` (the algebraic transpose, no conjugation).
    spd : bool
        Symmetric positive definite. Implies ``symmetric``.
    diagonal : bool
        Only diagonal entries are nonzero. Implies ``symmetric``.
    identity : bool
        :math:`A = I`. Implies ``diagonal`` and ``spd``.

    Raises
    ------
    ValueError
        If a property is declared without one it implies.
    """

    symmetric: bool = False
    spd: bool = False
    diagonal: bool = False
    identity: bool = False

    def __post_init__(self) -> None:
        implications = (
            ("spd", self.spd, "symmetric", self.symmetric),
            ("diagonal", self.diagonal, "symmetric", self.symmetric),
            ("identity", self.identity, "diagonal", self.diagonal),
            ("identity", self.identity, "spd", self.spd),
        )
        for name, value, implied, implied_value in implications:
            if value and not implied_value:
                raise ValueError(f"{name}=True requires {implied}=True")


@runtime_checkable
class VectorSpaceProtocol(Protocol, Generic[V, A_co]):
    r"""Operations on blocks of vectors of one space.

    ``V`` is the stack's vector type, a block of ``ncols`` vectors of
    :meth:`descriptor`; ``A_co`` is the small dense array type of reduction
    results.
    """

    def descriptor(self) -> SpaceDescriptor:
        """The space every vector handled here belongs to."""
        ...

    def zeros(self, ncols: int = 1) -> V:
        """Return a block of ``ncols`` zero vectors."""
        ...

    def zeros_like(self, x: V) -> V:
        """Return a zero block with as many columns as ``x``."""
        ...

    def ncols(self, x: V) -> int:
        """Return the number of vectors in the block ``x``."""
        ...

    def copy(self, x: V, out: Optional[V] = None) -> V:
        """Return a copy of ``x``. ``out`` must not alias ``x``."""
        ...

    def scale(self, a: float, x: V, out: Optional[V] = None) -> V:
        """Return ``a x``. ``out`` may be ``x``."""
        ...

    def axpby(self, a: float, x: V, b: float, y: V, out: Optional[V] = None) -> V:
        """Return ``a x + b y``. ``out`` may be ``y``, never ``x``."""
        ...

    def dot(self, x: V, y: V) -> A_co:
        r"""Return the block inner product :math:`x^H y`.

        A global reduction, replicated on every process. Shape
        ``(ncols(x), ncols(y))``. Conjugate-linear in ``x`` for complex
        spaces.
        """
        ...

    def norm(self, x: V) -> A_co:
        """Return the Euclidean norm of each vector of ``x``.

        A global reduction, replicated on every process. Shape
        ``(1, ncols(x))``.
        """
        ...

    def divide_entries(self, x: V, d: V, out: Optional[V] = None) -> V:
        """Return ``x`` with each row divided by the matching entry of the
        single vector ``d``, for every column. ``out`` may be ``x``."""
        ...

    def all_finite(self, x: V) -> bool:
        """Whether every entry of ``x`` is finite: a global reduction."""
        ...


@runtime_checkable
class ActionOperatorProtocol(Protocol, Generic[V, F_co]):
    r"""A linear operator from :meth:`domain` to :meth:`range`.

    ``V`` is the stack's vector type and ``F_co`` the type of the format
    witness :meth:`format`. A consumer that needs storage, such as a
    direct solver, declares the format it accepts, so pairing it with an
    operator of another format is a static type error.

    ``apply`` and ``apply_transpose`` act on blocks of vectors and take
    the optional ``out`` hint of :class:`VectorSpaceProtocol`; ``out``
    must never alias the input.
    """

    def domain(self) -> SpaceDescriptor:
        """The space of the input of :meth:`apply`."""
        ...

    def range(self) -> SpaceDescriptor:
        """The space of the output of :meth:`apply`."""
        ...

    def apply(self, x: V, out: Optional[V] = None) -> V:
        """Return :math:`A x` for a block ``x`` in the domain."""
        ...

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        r"""Return :math:`A^\top y` for a block ``y`` in the range.

        The plain algebraic transpose: no conjugation and no weighting
        by an inner product.
        """
        ...

    def properties(self) -> OperatorProperties:
        """The declared structural properties."""
        ...

    def format(self) -> F_co:
        """The format witness: how, and whether, the operator is stored."""
        ...


@runtime_checkable
class DiagonalCapableProtocol(ActionOperatorProtocol[V, F_co], Protocol[V, F_co]):
    """An operator that can also return its diagonal.

    Cheap preconditioners (Jacobi, smoothers) need the diagonal of an
    operator that may never be assembled, so the capability is declared
    independently of the format.
    """

    def diagonal(self) -> V:
        """Return the diagonal as a single vector of the domain.

        Requires ``domain() == range()``.
        """
        ...


@runtime_checkable
class RowSetProtocol(Protocol, Generic[V]):
    r"""A stack-owned set of rows of one space.

    Generic code holds no integer row arrays: it acts on rows through
    these operations, and each stack stores the set as its library does
    (an index array, a distributed index set). Under distribution each
    process holds the rows it owns, so no operation communicates.

    With :math:`P` the projection that zeros the rows in the set, the
    operations are fused forms of the products a row-constrained
    operator needs. ``out`` follows :class:`VectorSpaceProtocol`: it may
    be the operand accumulated into (``y``), never another operand.
    """

    def descriptor(self) -> SpaceDescriptor:
        """The space whose rows the set selects."""
        ...

    def count(self) -> int:
        """The number of rows in the set, over all processes."""
        ...

    def zero_rows(self, x: V, out: Optional[V] = None) -> V:
        """Return :math:`P x`: ``x`` with the rows in the set zeroed.
        ``out`` may be ``x``."""
        ...

    def replace_rows(self, y: V, x: V, out: Optional[V] = None) -> V:
        """Return :math:`P y + (I - P) x`: ``y`` with the rows in the set
        taken from ``x``."""
        ...

    def add_rows(self, y: V, x: V, out: Optional[V] = None) -> V:
        """Return :math:`y + (I - P) x`: ``y`` plus the rows of ``x`` in
        the set."""
        ...

    def difference(self, other: Self) -> Self:
        """Return the rows in this set and not in ``other``."""
        ...
