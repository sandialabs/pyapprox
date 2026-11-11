r"""Rows of a residual pinned to data: :math:`u_R = g(p)`.

A residual whose rows :math:`R` are constrained to data has those rows'
equations replaced by :math:`u_R - g_R = 0`. A declaration names the rows
and the data (:class:`PinnedValuesProtocol`): a row set of the state
space, the data as a column whose other rows are ignored, and how the
data depend on the parameters (:class:`PinnedValuesParamDerivatives`).

Two layers act on a declaration:

- :class:`RowReplacement` is the row algebra alone, with :math:`P` the
  projection zeroing the rows :math:`R`: a residual block's own operator
  becomes :math:`P A + (I - P)` (identity rows), a coupling block
  :math:`P A`, and an adjoint loses its rows :math:`R`, which carry no
  curvature of the replaced equation.
- :class:`RowPin` adds the data: the replaced rows read :math:`u_R -
  g_R`, a linear solve's right-hand side carries :math:`g_R`, and the
  rows' parameter Jacobian and curvature are the data's, negated
  (:meth:`RowPin.data_param_jacobian`, :meth:`RowPin.data_param_curvature`),
  read on the rows by :class:`PinnedValuesData`: None where the
  declaration does not make a derivative available, a zero operator
  where it states one zero. A consumer refuses the first
  (:func:`require_data_derivative`) and may skip the second (its
  properties declare ``zero``).

A time-indexed declaration (:mod:`pyapprox.ode.pinned_rows`) is a family
of these, one per time.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from pyapprox.implicitfunction.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    StatedZero,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    MatrixFree,
    OperatorProperties,
    RowReplacedOperator,
    RowSetProtocol,
    RowsZeroedOperator,
    SpaceDescriptor,
    VectorSpaceProtocol,
    ZeroOperator,
)

V = TypeVar("V")
W = TypeVar("W")
F = TypeVar("F", bound=FormatProtocol)

ValuesParamFn = Callable[[], DerivativeOperator[V]]
r""":math:`\partial g/\partial p`, parameters to states; its rows outside
:math:`R` are ignored."""
ValuesCurvatureFn = Callable[[V], DerivativeOperator[V]]
r""":math:`w \mapsto \sum_i w_i\, \partial^2 g_i/\partial p^2`, parameters
to parameters, for a weight column :math:`w` zero outside :math:`R`."""


@dataclass(frozen=True)
class PinnedValuesParamDerivatives(Generic[V]):
    r"""How a declaration's data :math:`g(p)` depend on the parameters.

    Each field is a function, :data:`~pyapprox.implicitfunction.
    residual_derivatives.STATED_ZERO` (the data do not depend on the
    parameters that way), or None (not available: a consumer needing it
    refuses). Build with the named constructors.

    Attributes
    ----------
    values : ValuesParamFn, STATED_ZERO or None
        :math:`\partial g/\partial p`.
    curvature : ValuesCurvatureFn, STATED_ZERO or None
        The weighted second derivatives of :math:`g`.
    """

    values: Union[ValuesParamFn[V], StatedZero, None] = None
    curvature: Union[ValuesCurvatureFn[V], StatedZero, None] = None

    @staticmethod
    def independent() -> "PinnedValuesParamDerivatives[W]":
        """Data that do not depend on the parameters."""
        return PinnedValuesParamDerivatives(STATED_ZERO, STATED_ZERO)

    @staticmethod
    def unavailable() -> "PinnedValuesParamDerivatives[W]":
        """Data whose parameter derivatives are not known."""
        return PinnedValuesParamDerivatives()

    @staticmethod
    def first_order(
        values: Union[ValuesParamFn[W], StatedZero],
    ) -> "PinnedValuesParamDerivatives[W]":
        """First derivatives only."""
        return PinnedValuesParamDerivatives(values)

    @staticmethod
    def second_order(
        values: Union[ValuesParamFn[W], StatedZero],
        curvature: Union[ValuesCurvatureFn[W], StatedZero],
    ) -> "PinnedValuesParamDerivatives[W]":
        """First and second derivatives."""
        return PinnedValuesParamDerivatives(values, curvature)


@runtime_checkable
class PinnedValuesProtocol(Protocol[V]):
    """Rows of a state pinned to data, and the data."""

    def rows(self) -> RowSetProtocol[V]:
        """The rows :math:`R`, a row set of the state space."""
        ...

    def values(self) -> V:
        r""":math:`g`: a single column of the state space whose rows
        :math:`R` hold the data; its other rows are ignored."""
        ...

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[V]:
        """How the data depend on the parameters; which fields are present
        is fixed for the declaration's life."""
        ...


class RowsKept(Generic[V]):
    r""":math:`(I - P) A`: an operator with the rows outside :math:`R`
    zeroed; ``MatrixFree``. ``out`` is ignored."""

    def __init__(
        self,
        op: DerivativeOperator[V],
        rows: RowSetProtocol[V],
        states: VectorSpaceProtocol[V],
    ) -> None:
        self._op, self._rows, self._states = op, rows, states

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        y = self._op.apply(x)
        return self._states.axpby(1.0, y, -1.0, self._rows.zero_rows(y))

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        kept = self._states.axpby(1.0, y, -1.0, self._rows.zero_rows(y))
        return self._op.apply_transpose(kept)


def undeclared_data_derivative(what: str, owner: object) -> ValueError:
    """The refusal of a data derivative a consumer needs and ``owner`` does
    not make available, naming both."""
    return ValueError(
        f"{type(owner).__name__} does not declare {what} of its data, "
        "which these derivatives need: declare it, or that the data do "
        "not depend on the parameters (independent())"
    )


def require_data_derivative(
    op: Optional[DerivativeOperator[V]], what: str, owner: object
) -> DerivativeOperator[V]:
    """``op``, a data derivative a consumer needs.

    Raises
    ------
    ValueError
        If ``op`` is None (not available), naming ``owner`` and ``what``.
    """
    if op is None:
        raise undeclared_data_derivative(what, owner)
    return op


def kept_rows(rows: RowSetProtocol[V], states: VectorSpaceProtocol[V], x: V) -> V:
    r""":math:`(I - P) x`: ``x`` on the rows :math:`R`, zero elsewhere."""
    return states.axpby(1.0, x, -1.0, rows.zero_rows(x))


class PinnedValuesData(Generic[V]):
    r"""A declaration's data derivatives read on its rows :math:`R`: what a
    consumer adds there; a zero operator where the declaration states zero,
    None where it does not make the derivative available.

    Parameters
    ----------
    declaration : PinnedValuesProtocol
        The rows and their data.
    states : VectorSpaceProtocol
        The state space.
    """

    def __init__(
        self, declaration: PinnedValuesProtocol[V], states: VectorSpaceProtocol[V]
    ) -> None:
        self._declaration, self._states = declaration, states
        self._rows = declaration.rows()
        self._bundle = declaration.parameter_derivatives()

    def values_param_jacobian(
        self, params: VectorSpaceProtocol[V]
    ) -> Optional[DerivativeOperator[V]]:
        r""":math:`(I - P)\, \partial g/\partial p`, from ``params`` to the
        states."""
        fn = self._bundle.values
        if fn is None:
            return None
        if isinstance(fn, StatedZero):
            return ZeroOperator(params, self._states)
        return RowsKept(fn(), self._rows, self._states)

    def values_param_curvature(
        self, weights: V, params: VectorSpaceProtocol[V]
    ) -> Optional[DerivativeOperator[V]]:
        r""":math:`\sum_{i \in R} w_i\, \partial^2 g_i/\partial p^2`, on
        ``params``."""
        fn = self._bundle.curvature
        if fn is None:
            return None
        if isinstance(fn, StatedZero):
            return ZeroOperator(params)
        return fn(kept_rows(self._rows, self._states, weights))


class RowReplacement(Generic[V]):
    r"""The row algebra of pinning the rows :math:`R`: identity rows in a
    block's own operator, the rows zeroed in its couplings and in an
    adjoint (see the module docstring).

    Parameters
    ----------
    rows : RowSetProtocol
        The rows :math:`R`.
    """

    def __init__(self, rows: RowSetProtocol[V]) -> None:
        self._rows = rows

    def rows(self) -> RowSetProtocol[V]:
        """The rows :math:`R`."""
        return self._rows

    def own(self, op: ActionOperatorProtocol[V, F]) -> RowReplacedOperator[V, F]:
        """A block's own operator with the rows :math:`R` identity rows."""
        return RowReplacedOperator(op, self._rows)

    def coupled(self, op: ActionOperatorProtocol[V, F]) -> RowsZeroedOperator[V, F]:
        """A coupling block with the rows :math:`R` zeroed."""
        return RowsZeroedOperator(op, self._rows)

    def adjoint(self, block: V, space: VectorSpaceProtocol[V]) -> V:
        """A new copy of an adjoint block with the rows :math:`R` zeroed."""
        return self._rows.zero_rows(space.copy(block))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nrows={self._rows.count()})"


class RowPin(Generic[V]):
    r"""Pins a declaration's rows of a residual block to its data: the row
    algebra of :class:`RowReplacement` and the data's terms.

    Parameters
    ----------
    declaration : PinnedValuesProtocol
        The rows and their data.

    Raises
    ------
    TypeError
        If ``declaration`` does not satisfy
        :class:`PinnedValuesProtocol`.
    """

    def __init__(self, declaration: PinnedValuesProtocol[V]) -> None:
        if not isinstance(declaration, PinnedValuesProtocol):
            raise TypeError(
                "declaration must satisfy PinnedValuesProtocol, got "
                f"{type(declaration).__name__}"
            )
        self._declaration = declaration
        self._algebra = RowReplacement(declaration.rows())

    def rows(self) -> RowSetProtocol[V]:
        """The rows :math:`R`."""
        return self._algebra.rows()

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[V]:
        """How the data depend on the parameters, as declared."""
        return self._declaration.parameter_derivatives()

    def residual(self, block: V, state: V, space: VectorSpaceProtocol[V]) -> V:
        r"""A new residual block, its rows :math:`R` reading :math:`u_R -
        g_R` for ``state`` the block's unknown."""
        pinned = space.axpby(1.0, state, -1.0, self._declaration.values())
        return self._algebra.rows().replace_rows(block, pinned)

    def right_hand_side(self, rhs: V) -> V:
        """A new right-hand side for a linear solve with a row-replaced
        operator: ``rhs`` with its rows :math:`R` the data."""
        return self._algebra.rows().replace_rows(rhs, self._declaration.values())

    def own(self, op: ActionOperatorProtocol[V, F]) -> RowReplacedOperator[V, F]:
        return self._algebra.own(op)

    def coupled(self, op: ActionOperatorProtocol[V, F]) -> RowsZeroedOperator[V, F]:
        return self._algebra.coupled(op)

    def adjoint(self, block: V, space: VectorSpaceProtocol[V]) -> V:
        return self._algebra.adjoint(block, space)

    def data_param_jacobian(
        self, space: VectorSpaceProtocol[V], params: VectorSpaceProtocol[V]
    ) -> Optional[DerivativeOperator[V]]:
        r""":math:`(I - P)\, \partial g/\partial p`: the pinned rows'
        parameter Jacobian is its negative. A zero operator when the data
        do not depend on the parameters, None when the declaration does
        not make it available."""
        return PinnedValuesData(self._declaration, space).values_param_jacobian(
            params
        )

    def data_param_curvature(
        self,
        weights: V,
        space: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> Optional[DerivativeOperator[V]]:
        r""":math:`\sum_{i \in R} w_i\, \partial^2 g_i/\partial p^2` for an
        adjoint block ``weights`` unzeroed: the pinned rows' parameter
        curvature is its negative. A zero operator when the data do not
        depend on the parameters, None when the declaration does not make
        it available."""
        return PinnedValuesData(self._declaration, space).values_param_curvature(
            weights, params
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nrows={self._algebra.rows().count()})"
