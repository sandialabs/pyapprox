r"""Operator combinators: sums, transposes, products and format erasure.

Each combinator is itself an operator. Its action is computed from its
operands' actions; its format, when a consumer asks for one, from its
operands' formats through :class:`~.formats.FormatProtocol`, so a sum of
CSR operators can be factored by a sparse direct solver while a sum that
is only ever applied is never assembled.

Formats must agree. A sum of operators of different formats has no
common storage, and mypy reports it at the call site (the inferred format
type falls outside :class:`~.formats.FormatProtocol`); erase the formats
explicitly with :func:`as_matrix_free` to combine them by action alone.

Declared properties propagate conservatively: a combinator declares a
property only when it holds for every admissible operand, so a solver
never relies on a property that does not hold.
"""

from typing import Generic, Optional, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.formats import FormatProtocol, MatrixFree
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    LinearMapProtocol,
    OperatorProperties,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


def scaled_sum_properties(
    terms: Sequence[tuple[float, OperatorProperties]],
) -> OperatorProperties:
    r"""Properties of :math:`\sum_i c_i A_i`.

    Symmetric, diagonal or transposable if every term is; SPD if every
    term is SPD with a positive coefficient; never the identity.
    """
    symmetric = all(props.symmetric for _, props in terms)
    diagonal = all(props.diagonal for _, props in terms)
    spd = all(props.spd and c > 0 for c, props in terms)
    transposable = all(props.transposable for _, props in terms)
    return OperatorProperties(
        symmetric=symmetric, spd=spd, diagonal=diagonal, transposable=transposable
    )


def product_properties(factors: Sequence[OperatorProperties]) -> OperatorProperties:
    r"""Properties of :math:`A_1 A_2 \cdots A_k`: transposable if every
    factor is, and nothing more unless all are diagonal.

    A product of symmetric (or SPD) matrices is in general neither. When
    every factor is diagonal, the product is diagonal, hence equal to its
    transpose (symmetric); it is SPD when every factor has a positive
    diagonal (SPD), and the identity when every factor is.
    """
    if not all(props.diagonal for props in factors):
        return OperatorProperties(
            transposable=all(props.transposable for props in factors)
        )
    return OperatorProperties(
        symmetric=True,
        spd=all(props.spd for props in factors),
        diagonal=True,
        identity=all(props.identity for props in factors),
    )


def _require_spaces(
    op: LinearMapProtocol[V],
    domain: SpaceDescriptor,
    range_: SpaceDescriptor,
    what: str,
) -> None:
    domain.require_same(op.domain(), f"domain of {what}")
    range_.require_same(op.range(), f"range of {what}")


class ScaledSum(Generic[V, F]):
    r"""The operator :math:`\sum_i c_i A_i`.

    Parameters
    ----------
    terms : sequence of (float, operator)
        Nonempty. Every operator maps the same domain to the same range.
        Its stored format exists only when every term has the same format
        type ``F``; terms of mixed formats still sum and apply.
    space : VectorSpaceProtocol
        Linear combinations in the range, used to accumulate ``apply``.
    domain_space : VectorSpaceProtocol, optional
        Linear combinations in the domain, used to accumulate
        ``apply_transpose``. Defaults to ``space``, which then must be
        over the domain too (a square sum).

    Raises
    ------
    ValueError
        If ``terms`` is empty or a descriptor disagrees.
    """

    def __init__(
        self,
        terms: Sequence[tuple[float, ActionOperatorProtocol[V, F]]],
        space: VectorSpaceProtocol[V],
        domain_space: Optional[VectorSpaceProtocol[V]] = None,
    ) -> None:
        if not terms:
            raise ValueError("a scaled sum needs at least one term")
        self._terms = tuple(terms)
        first = self._terms[0][1]
        self._domain, self._range = first.domain(), first.range()
        for _, op in self._terms[1:]:
            _require_spaces(op, self._domain, self._range, "a summed operator")
        self._space = space
        self._domain_space = space if domain_space is None else domain_space
        self._range.require_same(space.descriptor(), "space of a scaled sum")
        self._domain.require_same(
            self._domain_space.descriptor(), "domain space of a scaled sum"
        )
        self._properties = scaled_sum_properties(
            [(c, op.properties()) for c, op in self._terms]
        )
        self._format: Optional[F] = None

    def domain(self) -> SpaceDescriptor:
        return self._domain

    def range(self) -> SpaceDescriptor:
        return self._range

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> F:
        """The operands' formats combined, computed on first request.

        The sum's action needs no format, so only callers asking for
        storage (a direct solver, a row edit) reach this.

        Raises
        ------
        TypeError
            If the terms' formats are of different types: they have no
            combined storage. Wrap the sum in :func:`as_matrix_free` to
            use it without one.
        """
        if self._format is None:
            formats = [op.format() for _, op in self._terms]
            kinds = sorted({type(fmt).__name__ for fmt in formats})
            if len(kinds) > 1:
                raise TypeError(
                    f"a scaled sum of terms in different formats ({', '.join(kinds)}) "
                    "has no combined format; wrap it in as_matrix_free to use it "
                    "without one"
                )
            coefficients = [c for c, _ in self._terms]
            self._format = formats[0].scaled_sum(
                coefficients[0], list(zip(coefficients[1:], formats[1:]))
            )
        return self._format

    def apply(self, x: V, out: Optional[V] = None) -> V:
        (c0, op0), *rest = self._terms
        y = op0.apply(x, out=out)
        if c0 != 1.0:
            y = self._space.scale(c0, y, out=y)
        for c, op in rest:
            y = self._space.axpby(c, op.apply(x), 1.0, y, out=y)
        return y

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        (c0, op0), *rest = self._terms
        x = op0.apply_transpose(y, out=out)
        if c0 != 1.0:
            x = self._domain_space.scale(c0, x, out=x)
        for c, op in rest:
            x = self._domain_space.axpby(c, op.apply_transpose(y), 1.0, x, out=x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nterms={len(self._terms)})"


@runtime_checkable
class RowPartitionProtocol(Protocol, Generic[V]):
    """A space's vectors split by rows into consecutive parts.

    What :class:`BlockAssembled` needs of a stack: its column
    concatenation (``VectorSpaceProtocol.concatenate``) joins vectors,
    not rows, so splitting and joining rows is the stack's to supply.
    """

    def space(self) -> VectorSpaceProtocol[V]:
        """The whole space."""
        ...

    def parts(self) -> Sequence[VectorSpaceProtocol[V]]:
        """The space of each part, in row order."""
        ...

    def split(self, x: V) -> Sequence[V]:
        """The parts of ``x``, each a vector of its part's space."""
        ...

    def join(self, parts: Sequence[V]) -> V:
        """The vector whose parts are ``parts``, as one new vector."""
        ...


def block_properties(
    blocks: Sequence[Sequence[Optional[OperatorProperties]]],
) -> OperatorProperties:
    r"""Properties of the block operator :math:`[A_{ij}]`, a None block
    zero.

    Transposable if every present block is. Only a block-diagonal
    operator (no off-diagonal block, every diagonal block present)
    inherits more: symmetric, SPD, diagonal or the identity when every
    diagonal block is.
    """
    present = [props for row in blocks for props in row if props is not None]
    transposable = all(props.transposable for props in present)
    block_diagonal = all(
        (i == j) == (props is not None)
        for i, row in enumerate(blocks)
        for j, props in enumerate(row)
    )
    if not block_diagonal:
        return OperatorProperties(transposable=transposable)
    return OperatorProperties(
        symmetric=all(props.symmetric for props in present),
        spd=all(props.spd for props in present),
        diagonal=all(props.diagonal for props in present),
        identity=all(props.identity for props in present),
        transposable=transposable,
    )


class BlockAssembled(Generic[V, F]):
    r"""The square block operator :math:`A = [A_{ij}]`, a None block zero.

    Applied block by block through the stack's row partition; its format,
    on request, is the blocks' formats assembled by
    :meth:`~.formats.FormatProtocol.block_matrix`, so the stack's own
    direct solver factors it. Its vectors are the stack's whole vectors;
    :class:`~.blocks.BlockOperator` is the counterpart over tuples of
    vectors, with no stored format.

    Parameters
    ----------
    blocks : sequence of sequences of operators or None
        A square grid, row by row, with at least one present block. Block
        :math:`(i, j)` maps part :math:`j` to part :math:`i`.
    partition : RowPartitionProtocol
        The parts of the operator's domain and range.

    Raises
    ------
    ValueError
        If the grid is not square, matches not the partition, has no
        present block, or a block's spaces disagree with its parts.
    """

    def __init__(
        self,
        blocks: Sequence[Sequence[Optional[ActionOperatorProtocol[V, F]]]],
        partition: RowPartitionProtocol[V],
    ) -> None:
        self._blocks = tuple(tuple(row) for row in blocks)
        self._partition = partition
        self._parts = tuple(partition.parts())
        nparts = len(self._parts)
        if len(self._blocks) != nparts or any(
            len(row) != nparts for row in self._blocks
        ):
            raise ValueError(
                f"a block operator over {nparts} parts needs a {nparts}x{nparts} "
                "grid of blocks"
            )
        self._present = [
            (i, j, op)
            for i, row in enumerate(self._blocks)
            for j, op in enumerate(row)
            if op is not None
        ]
        if not self._present:
            raise ValueError("a block operator needs at least one present block")
        for i, j, op in self._present:
            _require_spaces(
                op,
                self._parts[j].descriptor(),
                self._parts[i].descriptor(),
                f"block ({i}, {j})",
            )
        self._space = partition.space().descriptor()
        self._properties = block_properties(
            [
                [None if op is None else op.properties() for op in row]
                for row in self._blocks
            ]
        )
        self._format: Optional[F] = None

    def domain(self) -> SpaceDescriptor:
        return self._space

    def range(self) -> SpaceDescriptor:
        return self._space

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> F:
        """The blocks' formats assembled, computed on first request.

        Raises
        ------
        TypeError
            If the blocks' formats are of different types, or the format
            cannot store the block operator.
        """
        if self._format is None:
            formats = [
                [None if op is None else op.format() for op in row]
                for row in self._blocks
            ]
            present = [fmt for row in formats for fmt in row if fmt is not None]
            kinds = sorted({type(fmt).__name__ for fmt in present})
            if len(kinds) > 1:
                raise TypeError(
                    f"a block operator of blocks in different formats "
                    f"({', '.join(kinds)}) has no assembled format; wrap it in "
                    "as_matrix_free to use it without one"
                )
            sizes = [part.descriptor().size for part in self._parts]
            self._format = present[0].block_matrix(formats, sizes)
        return self._format

    def _product(self, x: V, transpose: bool) -> V:
        pieces = self._partition.split(x)
        result: list[Optional[V]] = [None] * len(self._parts)
        for i, j, op in self._present:
            # A_ij maps part j to part i; its transpose part i to part j.
            if transpose:
                target, term = j, op.apply_transpose(pieces[i])
            else:
                target, term = i, op.apply(pieces[j])
            previous = result[target]
            result[target] = (
                term
                if previous is None
                else self._parts[target].axpby(1.0, term, 1.0, previous, out=previous)
            )
        ncols = self._partition.space().ncols(x)
        return self._partition.join(
            [
                self._parts[k].zeros(ncols) if piece is None else piece
                for k, piece in enumerate(result)
            ]
        )

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._product(x, transpose=False)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self._product(y, transpose=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nparts={len(self._parts)})"


class Transposed(Generic[V, F]):
    r"""The operator :math:`A^\top`, in ``A``'s format.

    Parameters
    ----------
    op : ActionOperatorProtocol
        The operator to transpose.

    Raises
    ------
    ValueError
        If ``op`` has no transpose: :math:`A^\top` would have no action.
    """

    def __init__(self, op: ActionOperatorProtocol[V, F]) -> None:
        if not op.properties().transposable:
            raise ValueError(
                f"{op!r} has no transpose (properties().transposable is False)"
            )
        self._op = op
        self._format: Optional[F] = None

    def domain(self) -> SpaceDescriptor:
        return self._op.range()

    def range(self) -> SpaceDescriptor:
        return self._op.domain()

    def properties(self) -> OperatorProperties:
        # Each property holds for A^T exactly when it holds for A.
        return self._op.properties()

    def format(self) -> F:
        """The operand's format transposed, computed on first request."""
        if self._format is None:
            self._format = self._op.format().transposed()
        return self._format

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._op.apply_transpose(x, out=out)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self._op.apply(y, out=out)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._op!r})"


class Product(Generic[V]):
    r"""The operator :math:`A_1 A_2 \cdots A_k`, applied right to left.

    Matrix-free: forming a product of stored matrices would fill in their
    sparsity, so a consumer that needs one forms it explicitly. Factors
    may be of any format, including solver handles acting as inverses.

    Parameters
    ----------
    factors : sequence of operators
        Nonempty, ``[A_1, ..., A_k]``; the domain of each factor is the
        range of the next.

    Raises
    ------
    ValueError
        If ``factors`` is empty or consecutive descriptors disagree.
    """

    def __init__(self, factors: Sequence[LinearMapProtocol[V]]) -> None:
        if not factors:
            raise ValueError("a product needs at least one factor")
        self._factors = tuple(factors)
        for left, right in zip(self._factors, self._factors[1:]):
            left.domain().require_same(right.range(), "range of a product factor")
        self._properties = product_properties(
            [factor.properties() for factor in self._factors]
        )

    def domain(self) -> SpaceDescriptor:
        return self._factors[-1].domain()

    def range(self) -> SpaceDescriptor:
        return self._factors[0].range()

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        *inner, outer = reversed(self._factors)
        for factor in inner:
            x = factor.apply(x)
        return outer.apply(x, out=out)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        *inner, outer = self._factors
        for factor in inner:
            y = factor.apply_transpose(y)
        return outer.apply_transpose(y, out=out)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nfactors={len(self._factors)})"


class _MatrixFreeView(Generic[V]):
    """An operator's action and properties, with its format erased."""

    def __init__(self, op: LinearMapProtocol[V]) -> None:
        self._op = op

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return self._op.properties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._op.apply(x, out=out)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self._op.apply_transpose(y, out=out)

    def __repr__(self) -> str:
        return f"as_matrix_free({self._op!r})"


def as_matrix_free(
    op: LinearMapProtocol[V],
) -> ActionOperatorProtocol[V, MatrixFree]:
    """Erase ``op``'s format, keeping its action and properties.

    The explicit step that lets operators of different formats be summed:
    their sum is then known only through its action.
    """
    return _MatrixFreeView(op)
