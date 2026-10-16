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

from typing import Any, Generic, Optional, Sequence, TypeVar

from pyapprox.util.linalg.operators.formats import FormatProtocol, MatrixFree
from pyapprox.util.linalg.operators.layout import Layout
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    OperatorProperties,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


def scaled_sum_properties(
    terms: Sequence[tuple[float, OperatorProperties]],
) -> OperatorProperties:
    r"""Properties of :math:`\sum_i c_i A_i`.

    Symmetric or diagonal if every term is; SPD if every term is SPD with
    a positive coefficient; never the identity.
    """
    symmetric = all(props.symmetric for _, props in terms)
    diagonal = all(props.diagonal for _, props in terms)
    spd = all(props.spd and c > 0 for c, props in terms)
    return OperatorProperties(symmetric=symmetric, spd=spd, diagonal=diagonal)


def product_properties(factors: Sequence[OperatorProperties]) -> OperatorProperties:
    r"""Properties of :math:`A_1 A_2 \cdots A_k`: none unless all are diagonal.

    Nothing is declared unless every factor is diagonal: a product of
    symmetric (or SPD) matrices is in general neither. When every factor
    is diagonal, the product is diagonal, hence equal to its transpose
    (symmetric); it is SPD when every factor has a positive diagonal
    (SPD), and the identity when every factor is.
    """
    if not all(props.diagonal for props in factors):
        return OperatorProperties()
    return OperatorProperties(
        symmetric=True,
        spd=all(props.spd for props in factors),
        diagonal=True,
        identity=all(props.identity for props in factors),
    )


def _require_spaces(
    op: ActionOperatorProtocol[Any, Any], domain: Layout, range_: Layout, what: str
) -> None:
    domain.require_same(op.domain(), f"domain of {what}")
    range_.require_same(op.range(), f"range of {what}")


class ScaledSum(Generic[V, F]):
    r"""The operator :math:`\sum_i c_i A_i`.

    Parameters
    ----------
    terms : sequence of (float, operator)
        Nonempty. Every operator maps the same domain to the same range
        and has the same format type ``F``.
    space : VectorSpaceProtocol
        The vector space of the range, used to accumulate ``apply``.
    domain_space : VectorSpaceProtocol, optional
        The vector space of the domain, used to accumulate
        ``apply_transpose``. Defaults to ``space``, which then must be a
        space of the domain too.

    Raises
    ------
    ValueError
        If ``terms`` is empty or a layout disagrees.
    """

    def __init__(
        self,
        terms: Sequence[tuple[float, ActionOperatorProtocol[V, F]]],
        space: VectorSpaceProtocol[V, Any],
        domain_space: Optional[VectorSpaceProtocol[V, Any]] = None,
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
        self._range.require_same(space.layout(), "space of a scaled sum")
        self._domain.require_same(
            self._domain_space.layout(), "domain space of a scaled sum"
        )
        self._properties = scaled_sum_properties(
            [(c, op.properties()) for c, op in self._terms]
        )
        self._format: Optional[F] = None

    def domain(self) -> Layout:
        return self._domain

    def range(self) -> Layout:
        return self._range

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> F:
        """The operands' formats combined, computed on first request."""
        if self._format is None:
            (c0, op0), *rest = self._terms
            self._format = op0.format().scaled_sum(
                c0, [(c, op.format()) for c, op in rest]
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


class Transposed(Generic[V, F]):
    r"""The operator :math:`A^\top`, in ``A``'s format.

    Parameters
    ----------
    op : ActionOperatorProtocol
        The operator to transpose.
    """

    def __init__(self, op: ActionOperatorProtocol[V, F]) -> None:
        self._op = op
        self._format: Optional[F] = None

    def domain(self) -> Layout:
        return self._op.range()

    def range(self) -> Layout:
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
        If ``factors`` is empty or consecutive layouts disagree.
    """

    def __init__(self, factors: Sequence[ActionOperatorProtocol[V, Any]]) -> None:
        if not factors:
            raise ValueError("a product needs at least one factor")
        self._factors = tuple(factors)
        for left, right in zip(self._factors, self._factors[1:]):
            left.domain().require_same(right.range(), "range of a product factor")
        self._properties = product_properties(
            [factor.properties() for factor in self._factors]
        )

    def domain(self) -> Layout:
        return self._factors[-1].domain()

    def range(self) -> Layout:
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

    def __init__(self, op: ActionOperatorProtocol[V, Any]) -> None:
        self._op = op

    def domain(self) -> Layout:
        return self._op.domain()

    def range(self) -> Layout:
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
    op: ActionOperatorProtocol[V, Any],
) -> ActionOperatorProtocol[V, MatrixFree]:
    """Erase ``op``'s format, keeping its action and properties.

    The explicit step that lets operators of different formats be summed:
    their sum is then known only through its action.
    """
    return _MatrixFreeView(op)
