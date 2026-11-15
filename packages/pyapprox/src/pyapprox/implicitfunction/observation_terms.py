r"""One-state terms built on a linear observation of the state.

An observation is a linear map :math:`H` from a state to observed values
(:class:`StateObservationProtocol`), applied forward and transposed. The
misfit :class:`ObservationMisfit` and the observed values
:class:`ObservedValue` reach the state only through it, so they are
generic in the state's stack, and serve as terms of any functional of one
state: a steady solution's, or each member a trajectory's stored-state
term builds at a stored state. :class:`StateEntries` observes entries of
a backend state column.
"""

from typing import Generic, Optional, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.implicitfunction.functional_terms import (
    LinearizationProtocol,
    StateTermDerivatives,
)
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.backend_dense import backend_dtype_name
from pyapprox.util.linalg.operators import RowSetProtocol, SpaceDescriptor

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)


@runtime_checkable
class StateObservationProtocol(Protocol[V, A]):
    r"""A linear observation :math:`H` of a state: forward, the observed
    values; transposed, a state from value weights. Being linear, it is
    its own derivative."""

    def nobs(self) -> int:
        """The number of observed values."""
        ...

    def domain(self) -> SpaceDescriptor:
        """The space of the vectors observed, so a consumer can check it
        against the space it applies the observation in."""
        ...

    def apply(self, state: V) -> A:
        r""":math:`H y`, a column of ``nobs`` values."""
        ...

    def apply_transpose(self, values: A) -> V:
        r""":math:`H^\top r` for a column ``r`` of ``nobs`` values."""
        ...


class StateEntries(Generic[Array]):
    r"""The observation of entries ``rows`` of a backend state column,
    :math:`H` the rows of the identity; a row may repeat.

    Parameters
    ----------
    bkd : Backend
        The backend.
    nstates : int
        The length of a state.
    rows : sequence of int
        The observed entries, each in ``[0, nstates)``.

    Raises
    ------
    ValueError
        If ``rows`` is empty or an entry is out of range.
    """

    def __init__(self, bkd: Backend[Array], nstates: int, rows: Sequence[int]) -> None:
        rows = list(rows)
        if not rows:
            raise ValueError("an observation needs at least one entry")
        if any(row < 0 or row >= nstates for row in rows):
            raise ValueError(f"observed entries {rows} are not all in [0, {nstates})")
        self._bkd = bkd
        self._domain = SpaceDescriptor("state", nstates, backend_dtype_name(bkd))
        # The selection matrix, built from plain integers (constant data).
        self._matrix = bkd.asarray(
            [
                [1.0 if column == row else 0.0 for column in range(nstates)]
                for row in rows
            ]
        )

    def nobs(self) -> int:
        return int(self._matrix.shape[0])

    def domain(self) -> SpaceDescriptor:
        """The backend state space of ``nstates`` entries."""
        return self._domain

    def apply(self, state: Array) -> Array:
        return self._bkd.dot(self._matrix, state)

    def apply_transpose(self, values: Array) -> Array:
        return self._bkd.dot(self._matrix.T, values)


class RowWeights(Generic[Array]):
    r"""The observation :math:`H = W^\top` of a backend state column by
    weight columns :math:`W` that vanish off a row set's rows: entry
    :math:`k` is :math:`w_k^\top y`, read on those rows alone.

    Weights off the rows are refused rather than dropped, so a row the
    caller meant to read but that is not in the set (a natural boundary's
    row, say, where a residual vanishes) is an error, not a silent zero.

    Parameters
    ----------
    bkd : Backend
        The backend.
    rows : RowSetProtocol
        The rows read.
    weights : Array
        :math:`W`, shape ``(rows' size, nobs)``.

    Raises
    ------
    TypeError
        If ``rows`` does not satisfy ``RowSetProtocol``.
    ValueError
        If ``weights`` is not 2-D with a row per entry of the rows' space
        and at least one column, or has a nonzero weight off the rows.
    """

    def __init__(
        self, bkd: Backend[Array], rows: RowSetProtocol[Array], weights: Array
    ) -> None:
        if not isinstance(rows, RowSetProtocol):
            raise TypeError(
                f"rows must satisfy RowSetProtocol, got {type(rows).__name__}"
            )
        size = rows.descriptor().size
        if weights.ndim != 2 or weights.shape[0] != size or weights.shape[1] < 1:
            raise ValueError(
                f"weights must have shape ({size}, nobs) with nobs >= 1, got "
                f"{tuple(weights.shape)}"
            )
        # Zeroing the set's rows leaves only the weights off them.
        off_rows = rows.zero_rows(weights)
        if bkd.to_float(bkd.max(bkd.abs(off_rows))) != 0.0:
            raise ValueError(
                "weights must vanish off the rows read; a weight is nonzero "
                "on a row outside the set"
            )
        self._bkd, self._rows, self._weights = bkd, rows, weights

    def nobs(self) -> int:
        return int(self._weights.shape[1])

    def domain(self) -> SpaceDescriptor:
        return self._rows.descriptor()

    def apply(self, state: Array) -> Array:
        return self._bkd.dot(self._weights.T, state)

    def apply_transpose(self, values: Array) -> Array:
        return self._bkd.dot(self._weights, values)


class _MisfitLinearization(Generic[V, A]):
    r""":math:`D x = r^\top H x / \sigma^2` for the residual :math:`r = H y
    - d`; transposed, :math:`H^\top (r\, \omega) / \sigma^2`, the product
    taken in the values, so no state vector meets an array."""

    def __init__(
        self,
        operator: StateObservationProtocol[V, A],
        residual: A,
        bkd: Backend[A],
        scale: float,
    ) -> None:
        self._operator, self._residual = operator, residual
        self._bkd, self._scale = bkd, scale

    def apply(self, x: V) -> A:
        observed = self._operator.apply(x)
        return self._scale * self._bkd.sum(
            self._residual * observed, axis=0, keepdims=True
        )

    def apply_transpose(self, weights: A) -> V:
        return self._operator.apply_transpose(self._scale * self._residual * weights)


def _no_block(state: V, weights: A, x: V) -> Optional[V]:
    """A second-derivative block that is zero."""
    return None


class ObservationMisfit(Generic[V, A]):
    r"""The misfit :math:`\|H y - d\|^2 / (2\sigma^2)` of one observation
    of a state, a one-state term with a value of one entry.

    It does not read the parameters, so its parameter derivatives are zero
    and its state second derivative is :math:`H^\top H / \sigma^2`.

    Parameters
    ----------
    bkd : Backend
        The backend of the values.
    operator : StateObservationProtocol
        :math:`H`.
    data : A
        :math:`d`, a column of ``operator.nobs()`` values.
    noise_std : float, default 1
        :math:`\sigma`, fixed.

    Raises
    ------
    TypeError
        If ``operator`` does not satisfy :class:`StateObservationProtocol`.
    ValueError
        If the datum's shape differs from the observation's, or
        ``noise_std`` is not positive.
    """

    def __init__(
        self,
        bkd: Backend[A],
        operator: StateObservationProtocol[V, A],
        data: A,
        noise_std: float = 1.0,
    ) -> None:
        if not isinstance(operator, StateObservationProtocol):
            raise TypeError(
                "each observation must satisfy StateObservationProtocol, "
                f"got {type(operator).__name__}"
            )
        if tuple(data.shape) != (operator.nobs(), 1):
            raise ValueError(
                f"the data must have shape ({operator.nobs()}, 1), got "
                f"{tuple(data.shape)}"
            )
        if not noise_std > 0.0:
            raise ValueError(f"noise_std must be positive, got {noise_std}")
        self._bkd, self._operator, self._data = bkd, operator, data
        self._scale = 1.0 / noise_std**2
        self._derivatives: StateTermDerivatives[V, A] = (
            StateTermDerivatives.second_order(
                self._wrt_state,
                None,
                self._state_state,
                _no_block,
                _no_block,
                _no_block,
            )
        )

    def nqoi(self) -> int:
        return 1

    def _residual(self, state: V) -> A:
        return self._operator.apply(state) - self._data

    def value(self, state: V) -> A:
        residual = self._residual(state)
        return (0.5 * self._scale) * self._bkd.sum(
            residual * residual, axis=0, keepdims=True
        )

    def derivatives(self) -> StateTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_state(self, state: V) -> LinearizationProtocol[V, A]:
        return _MisfitLinearization(
            self._operator, self._residual(state), self._bkd, self._scale
        )

    def _state_state(self, state: V, weights: A, w: V) -> Optional[V]:
        operator = self._operator
        return operator.apply_transpose(self._scale * operator.apply(w) * weights)


class ObservedValue(Generic[V, A]):
    r"""The observation :math:`H y` of a state, a one-state term with
    ``H.nobs()`` entries: linear, so :math:`H` is its derivative and its
    second derivatives are zero.

    Parameters
    ----------
    operator : StateObservationProtocol
        :math:`H`.

    Raises
    ------
    TypeError
        If ``operator`` does not satisfy :class:`StateObservationProtocol`.
    """

    def __init__(self, operator: StateObservationProtocol[V, A]) -> None:
        if not isinstance(operator, StateObservationProtocol):
            raise TypeError(
                "operator must satisfy StateObservationProtocol, got "
                f"{type(operator).__name__}"
            )
        self._operator = operator
        self._derivatives: StateTermDerivatives[V, A] = (
            StateTermDerivatives.second_order(
                self._wrt_state, None, _no_block, _no_block, _no_block, _no_block
            )
        )

    def nqoi(self) -> int:
        return self._operator.nobs()

    def value(self, state: V) -> A:
        return self._operator.apply(state)

    def derivatives(self) -> StateTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_state(self, state: V) -> LinearizationProtocol[V, A]:
        return self._operator
