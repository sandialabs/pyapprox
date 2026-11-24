r"""Essential rows of a second-order system pinned in both state blocks.

A displacement prescribed on rows :math:`R`, :math:`u_R = g(t)`, fixes the
velocity there too, :math:`v_R = \dot g(t)`. :class:`DoubledPinnedRows`
declares both for the reduced state :math:`w = [u; v]` as a
:class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol`: rows :math:`R` of
each block, values :math:`[g; \dot g]` and rates :math:`[\dot g;
\ddot g]`. Every shared pin (``RowPinning``, ``SplitRowPinning``,
``PinnedRowsConstraint``, ``differentiated_rows``) then applies it to the
reduced system unchanged.

It reads a declaration of the displacement rows that states its data's
time derivatives by order (:class:`TimeDerivativeRowsProtocol`) and a
storage that lifts row sets of one block to row sets of :math:`w`
(:class:`RowLiftingProtocol`), so it serves any stack and any
discretization: nothing here names one.
"""

from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives
from pyapprox.ode.residual_derivatives import StatedZero
from pyapprox.util.linalg.operators import RowSetProtocol

V = TypeVar("V")


@runtime_checkable
class TimeDerivativeRowsProtocol(Protocol[V]):
    """The displacement rows pinned to data, with the data's time
    derivatives by order, as :class:`DoubledPinnedRows` reads them."""

    def rows(self) -> RowSetProtocol[V]:
        """The rows :math:`R`, a row set of one block."""
        ...

    def values(self, time: float) -> V:
        r""":math:`g(t)`, a single column of one block; rows outside
        :math:`R` are ignored."""
        ...

    def time_derivative(self, order: int) -> Optional[Callable[[float], V]]:
        r""":math:`t \mapsto \partial_t^k g(t)` for ``order``
        :math:`k \ge 1`, laid out as :meth:`values`; None when not
        supplied. Fixed for the declaration's life."""
        ...

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[V]:
        """How the data depend on the parameters."""
        ...


@runtime_checkable
class RowLiftingProtocol(Protocol[V]):
    """What :class:`DoubledPinnedRows` needs of a storage of
    :math:`w = [u; v]`; both storages of
    :mod:`pyapprox.ode.second_order` satisfy it."""

    def join(self, displacement: V, velocity: V) -> V:
        """The vector with blocks ``displacement`` and ``velocity``."""
        ...

    def lift_rows(
        self, displacement: RowSetProtocol[V], velocity: RowSetProtocol[V]
    ) -> RowSetProtocol[V]:
        """The rows of :math:`w` that are ``displacement``'s rows of the
        first block and ``velocity``'s of the second."""
        ...


class _JoinedDerivative(Generic[V]):
    r""":math:`t \mapsto [\partial_t^k g; \partial_t^{k+1} g]`."""

    def __init__(
        self,
        blocks: RowLiftingProtocol[V],
        displacement: Callable[[float], V],
        velocity: Callable[[float], V],
    ) -> None:
        self._blocks = blocks
        self._displacement, self._velocity = displacement, velocity

    def __call__(self, time: float) -> V:
        return self._blocks.join(self._displacement(time), self._velocity(time))


class DoubledPinnedRows(Generic[V]):
    r"""Rows :math:`R` of both blocks of :math:`w = [u; v]` pinned to
    :math:`[g; \dot g]`.

    Satisfies :class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol` and
    :class:`TimeDerivativeRowsProtocol` on :math:`w`: its order-:math:`k`
    derivative is :math:`[\partial_t^k g; \partial_t^{k+1} g]`, present
    when the declaration supplies order :math:`k + 1`. So :meth:`rates`,
    :math:`[\dot g; \ddot g]`, needs :math:`\ddot g`, which a pin at every
    block does not read.

    Parameters
    ----------
    declaration : TimeDerivativeRowsProtocol
        The displacement rows and their data.
    blocks : RowLiftingProtocol
        The storage of :math:`w`, whose block space is the declaration's.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If the declaration supplies no rate, needed as the velocity's
        data, or its data are not declared independent of the parameters
        (the doubled data's parameter derivatives would join operators of
        both blocks, which this declaration does not build).
    """

    def __init__(
        self,
        declaration: TimeDerivativeRowsProtocol[V],
        blocks: RowLiftingProtocol[V],
    ) -> None:
        if not isinstance(declaration, TimeDerivativeRowsProtocol):
            raise TypeError(
                "declaration must satisfy TimeDerivativeRowsProtocol, got "
                f"{type(declaration).__name__}"
            )
        if not isinstance(blocks, RowLiftingProtocol):
            raise TypeError(
                f"blocks must satisfy RowLiftingProtocol, got {type(blocks).__name__}"
            )
        rate = declaration.time_derivative(1)
        if rate is None:
            raise ValueError(
                "the declaration supplies no rate of its data; the velocity "
                "rows are pinned to it"
            )
        bundle = declaration.parameter_derivatives()
        if not (
            isinstance(bundle.values, StatedZero)
            and isinstance(bundle.rates, StatedZero)
        ):
            raise ValueError(
                "the declaration's data are not declared independent of the "
                "parameters; doubled data depending on them are not supported"
            )
        self._declaration, self._blocks, self._rate = declaration, blocks, rate
        base = declaration.rows()
        self._rows = blocks.lift_rows(base, base)

    def rows(self) -> RowSetProtocol[V]:
        r"""Rows :math:`R` of each block, a row set of :math:`w`."""
        return self._rows

    def values(self, time: float) -> V:
        r""":math:`[g(t); \dot g(t)]`, a single column of :math:`w`."""
        return self._blocks.join(self._declaration.values(time), self._rate(time))

    def time_derivative(self, order: int) -> Optional[Callable[[float], V]]:
        r""":math:`t \mapsto [\partial_t^k g; \partial_t^{k+1} g]`; None
        when the declaration does not supply order :math:`k + 1`.

        Raises
        ------
        ValueError
            If ``order`` is less than 1.
        """
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        displacement = self._declaration.time_derivative(order)
        velocity = self._declaration.time_derivative(order + 1)
        if displacement is None or velocity is None:
            return None
        return _JoinedDerivative(self._blocks, displacement, velocity)

    def rates(self) -> Optional[Callable[[float], V]]:
        r""":math:`t \mapsto [\dot g; \ddot g]`; None without
        :math:`\ddot g`."""
        return self.time_derivative(1)

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[V]:
        """Independent of the parameters, as the declaration's data are."""
        return PinnedRowsParamDerivatives.independent()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._declaration!r})"
