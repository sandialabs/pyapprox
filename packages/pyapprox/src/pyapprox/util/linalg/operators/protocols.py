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

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.util.linalg.operators.layout import Layout

V = TypeVar("V")
A_co = TypeVar("A_co", covariant=True)


@runtime_checkable
class VectorSpaceProtocol(Protocol, Generic[V, A_co]):
    r"""Operations on blocks of vectors of one space.

    ``V`` is the stack's vector type, a block of ``ncols`` vectors of
    :meth:`layout`; ``A_co`` is the small dense array type of reduction
    results.
    """

    def layout(self) -> Layout:
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
