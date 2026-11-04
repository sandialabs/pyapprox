r"""Native terms of a functional of a trajectory: observations of stored
states, their misfit, and a Tikhonov regularization of the parameters.

Each is a term of :mod:`pyapprox.ode.trajectory_terms`: it holds no
trajectory, is handed each stored state it reads, and gives its value and
its derivatives in its bundle, so one term serves the value sink and every
sweep, on any store.

An observation is a linear map :math:`H` from a stored state to observed
values (:class:`ObservationOperatorProtocol`), applied forward and
transposed; the misfit and the endpoint terms reach states only through
it, so they are generic in the state's stack.
:class:`StateEntries` observes entries of a backend state column.
"""

from typing import (
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.trajectory_terms import (
    BackendScalarGradient,
    LinearizationProtocol,
    ParameterTermDerivatives,
    StateTermDerivatives,
)
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)


@runtime_checkable
class ObservationOperatorProtocol(Protocol[V, A]):
    r"""A linear observation :math:`H` of a stored state: forward, the
    observed values; transposed, a state from value weights. Being linear,
    it is its own derivative."""

    def nobs(self) -> int:
        """The number of observed values."""
        ...

    def apply(self, state: V) -> A:
        r""":math:`H y`, a column of ``nobs`` values."""
        ...

    def apply_transpose(self, values: A) -> V:
        r""":math:`H^\top r` for a column ``r`` of ``nobs`` values."""
        ...


def _selection(bkd: Backend[Array], rows: Sequence[int], size: int) -> Array:
    """The ``(len(rows), size)`` matrix selecting ``rows``, built from
    plain integers (constant data)."""
    return bkd.asarray(
        [[1.0 if column == row else 0.0 for column in range(size)] for row in rows]
    )


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
        self._matrix = _selection(bkd, rows, nstates)

    def nobs(self) -> int:
        return int(self._matrix.shape[0])

    def apply(self, state: Array) -> Array:
        return self._bkd.dot(self._matrix, state)

    def apply_transpose(self, values: Array) -> Array:
        return self._bkd.dot(self._matrix.T, values)


class _MisfitLinearization(Generic[V, A]):
    r""":math:`D x = r^\top H x / \sigma^2` for the residual :math:`r = H y
    - d`; transposed, :math:`H^\top (r\, \omega) / \sigma^2`, the product
    taken in the values, so no state vector meets an array."""

    def __init__(
        self,
        operator: ObservationOperatorProtocol[V, A],
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


class MisfitTerm(Generic[V, A]):
    r"""The misfit :math:`\sum_m \|H_m y_m - d_m\|^2 / (2\sigma^2)` of
    observations of stored states, a state term with a value of one entry.

    It reads stored state ``m`` for each observed index ``m``; it does not
    read the parameters, so its parameter derivatives are zero and its
    state second derivative is :math:`H_m^\top H_m / \sigma^2`.

    Parameters
    ----------
    bkd : Backend
        The backend of the values.
    observations : mapping
        For each observed stored-state index (non-negative), the
        observation :math:`H_m` and the data :math:`d_m`, a column of
        ``H_m.nobs()`` values.
    noise_std : float, default 1
        :math:`\sigma`, fixed.

    Raises
    ------
    TypeError
        If an observation does not satisfy
        :class:`ObservationOperatorProtocol`.
    ValueError
        If no state is observed, an index is negative, a datum's shape
        differs from its observation's, or ``noise_std`` is not positive.
    """

    def __init__(
        self,
        bkd: Backend[A],
        observations: Mapping[int, Tuple[ObservationOperatorProtocol[V, A], A]],
        noise_std: float = 1.0,
    ) -> None:
        if not observations:
            raise ValueError("a misfit needs at least one observed state")
        if not noise_std > 0.0:
            raise ValueError(f"noise_std must be positive, got {noise_std}")
        for index, (operator, data) in observations.items():
            if index < 0:
                raise ValueError(
                    f"observed state indices must be non-negative, got {index}"
                )
            if not isinstance(operator, ObservationOperatorProtocol):
                raise TypeError(
                    "each observation must satisfy ObservationOperatorProtocol, "
                    f"got {type(operator).__name__}"
                )
            if tuple(data.shape) != (operator.nobs(), 1):
                raise ValueError(
                    f"the data at state {index} must have shape "
                    f"({operator.nobs()}, 1), got {tuple(data.shape)}"
                )
        self._bkd = bkd
        self._observations = dict(observations)
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

    @staticmethod
    def from_state_times(
        bkd: Backend[Array],
        nstates: int,
        observed: Sequence[Tuple[int, Sequence[int]]],
        values: Array,
        noise_std: float = 1.0,
    ) -> "MisfitTerm[Array, Array]":
        r"""The misfit of backend states observed entry by entry: for each
        ``(entry, indices)`` of ``observed``, state entry ``entry`` at the
        stored states ``indices``, the flat ``values`` listing each pair's
        data in that order (all of the first pair's, then the second's).

        Parameters
        ----------
        bkd : Backend
            The backend.
        nstates : int
            The length of a state.
        observed : sequence of (int, sequence of int)
            The observed entries and the stored states each is observed at.
        values : Array
            The data, shape ``(nobs,)``.
        noise_std : float, default 1
            :math:`\sigma`.

        Raises
        ------
        ValueError
            If ``values`` does not hold one datum per observation.
        """
        nobs = sum(len(indices) for _, indices in observed)
        if tuple(values.shape) != (nobs,):
            raise ValueError(
                f"values must have shape ({nobs},) for {nobs} observations, got "
                f"{tuple(values.shape)}"
            )
        # The rows observed at each stored state, and their positions in
        # ``values``, in the order they are listed.
        rows: dict[int, List[int]] = {}
        positions: dict[int, List[int]] = {}
        offset = 0
        for entry, indices in observed:
            for position, index in enumerate(indices):
                rows.setdefault(index, []).append(entry)
                positions.setdefault(index, []).append(offset + position)
            offset += len(indices)
        column = bkd.reshape(values, (nobs, 1))
        observations: dict[
            int, Tuple[ObservationOperatorProtocol[Array, Array], Array]
        ] = {
            index: (
                StateEntries(bkd, nstates, rows[index]),
                bkd.dot(_selection(bkd, positions[index], nobs), column),
            )
            for index in sorted(rows)
        }
        return MisfitTerm(bkd, observations, noise_std)

    def nqoi(self) -> int:
        return 1

    def reads(self, index: int, last: bool) -> bool:
        return index in self._observations

    def _residual(
        self, index: int, state: V
    ) -> Tuple[ObservationOperatorProtocol[V, A], A]:
        operator, data = self._observations[index]
        return operator, operator.apply(state) - data

    def value(self, index: int, time: float, state: V) -> A:
        _, residual = self._residual(index, state)
        return (0.5 * self._scale) * self._bkd.sum(
            residual * residual, axis=0, keepdims=True
        )

    def derivatives(self) -> StateTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_state(
        self, index: int, time: float, state: V
    ) -> LinearizationProtocol[V, A]:
        operator, residual = self._residual(index, state)
        return _MisfitLinearization(operator, residual, self._bkd, self._scale)

    def _state_state(
        self, index: int, time: float, state: V, weights: A, w: V
    ) -> Optional[V]:
        operator, _ = self._observations[index]
        return operator.apply_transpose(self._scale * operator.apply(w) * weights)


def _no_block(index: int, time: float, state: V, weights: A, x: V) -> Optional[V]:
    """A second-derivative block that is zero."""
    return None


class EndpointTerm(Generic[V, A]):
    r"""The observation :math:`H y_N` of the last stored state, a state
    term with ``H.nobs()`` entries: linear, so :math:`H` is its derivative
    and its second derivatives are zero.

    Parameters
    ----------
    operator : ObservationOperatorProtocol
        :math:`H`.

    Raises
    ------
    TypeError
        If ``operator`` does not satisfy :class:`ObservationOperatorProtocol`.
    """

    def __init__(self, operator: ObservationOperatorProtocol[V, A]) -> None:
        if not isinstance(operator, ObservationOperatorProtocol):
            raise TypeError(
                "operator must satisfy ObservationOperatorProtocol, got "
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

    def reads(self, index: int, last: bool) -> bool:
        return last

    def value(self, index: int, time: float, state: V) -> A:
        return self._operator.apply(state)

    def derivatives(self) -> StateTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_state(
        self, index: int, time: float, state: V
    ) -> LinearizationProtocol[V, A]:
        return self._operator


class TikhonovTerm(Generic[Array]):
    r"""The regularization :math:`(\alpha/2)\, p^\top W p` of the
    parameters last set (:math:`W = I` by default), a parameter term.

    Parameters
    ----------
    bkd : Backend
        The backend.
    alpha : float
        :math:`\alpha`.
    weight : Array, optional
        :math:`W`, symmetric, shape ``(nparams, nparams)``.

    Raises
    ------
    ValueError
        If ``weight`` is not square or not symmetric.
    """

    def __init__(
        self, bkd: Backend[Array], alpha: float, weight: Optional[Array] = None
    ) -> None:
        if weight is not None:
            if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
                raise ValueError(
                    f"weight must be square, got shape {tuple(weight.shape)}"
                )
            asymmetry = bkd.to_float(bkd.max(bkd.abs(weight - weight.T)))
            if asymmetry > 1e-12 * max(bkd.to_float(bkd.max(bkd.abs(weight))), 1.0):
                raise ValueError("weight must be symmetric")
        self._bkd, self._alpha, self._weight = bkd, alpha, weight
        self._param: Optional[Array] = None
        self._derivatives: ParameterTermDerivatives[Array, Array] = (
            ParameterTermDerivatives.second_order(self._wrt_param, self._param_param)
        )

    def set_param(self, param: Array) -> None:
        """Set :math:`p`, a column."""
        if param.ndim != 2 or param.shape[1] != 1:
            raise ValueError(
                f"param must be a column (nparams, 1), got {tuple(param.shape)}"
            )
        self._param = param

    def _weighted(self, v: Array) -> Array:
        return v if self._weight is None else self._bkd.dot(self._weight, v)

    def _current(self) -> Array:
        if self._param is None:
            raise ValueError("set_param must be called before the term is read")
        return self._param

    def nqoi(self) -> int:
        return 1

    def value(self) -> Array:
        param = self._current()
        return (0.5 * self._alpha) * self._bkd.dot(param.T, self._weighted(param))

    def derivatives(self) -> ParameterTermDerivatives[Array, Array]:
        return self._derivatives

    def _wrt_param(self) -> BackendScalarGradient[Array]:
        return BackendScalarGradient(
            self._alpha * self._weighted(self._current()), self._bkd
        )

    def _param_param(self, weights: Array, v: Array) -> Array:
        return self._alpha * self._weighted(v) * weights
