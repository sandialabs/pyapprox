r"""Stored-state terms of a functional of a trajectory built on
observations: the misfit of observed states and the observation of the
last state.

Each is a family of :mod:`pyapprox.ode.trajectory_terms`: it reads chosen
stored states and builds there a one-state term of
:mod:`pyapprox.implicitfunction.observation_terms`, so one family serves
the value sink and every sweep, on any store. :class:`MisfitTerm` builds
the :class:`ObservationMisfit` of each observed state and
:class:`EndpointTerm` the :class:`ObservedValue` of the last.
"""

from typing import Generic, List, Mapping, Sequence, Tuple, TypeVar

from pyapprox.implicitfunction.functional_terms import StateTermProtocol
from pyapprox.implicitfunction.observation_terms import (
    ObservationMisfit,
    ObservedValue,
    StateEntries,
    StateObservationProtocol,
)
from pyapprox.ode.trajectory_terms import (
    AtStoredStates,
    StoredStates,
    StoredStateTermDerivatives,
)
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)


class MisfitTerm(Generic[V, A]):
    r"""The misfit :math:`\sum_m \|H_m y_m - d_m\|^2 / (2\sigma^2)` of
    observations of stored states, a state term with a value of one entry.

    It reads stored state ``m`` for each observed index ``m``, its member
    there the :class:`ObservationMisfit` of :math:`(H_m, d_m)`.

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
        :class:`StateObservationProtocol`.
    ValueError
        If no state is observed, an index is negative, a datum's shape
        differs from its observation's, or ``noise_std`` is not positive.
    """

    def __init__(
        self,
        bkd: Backend[A],
        observations: Mapping[int, Tuple[StateObservationProtocol[V, A], A]],
        noise_std: float = 1.0,
    ) -> None:
        if not observations:
            raise ValueError("a misfit needs at least one observed state")
        if not noise_std > 0.0:
            raise ValueError(f"noise_std must be positive, got {noise_std}")
        self._members: dict[int, ObservationMisfit[V, A]] = {}
        for index, (operator, data) in observations.items():
            if index < 0:
                raise ValueError(
                    f"observed state indices must be non-negative, got {index}"
                )
            try:
                member = ObservationMisfit(bkd, operator, data, noise_std)
            except ValueError as error:
                raise ValueError(f"at stored state {index}: {error}") from error
            self._members[index] = member
        self._derivatives: StoredStateTermDerivatives[V, A] = (
            StoredStateTermDerivatives.of_members(self.at, list(self._members.values()))
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
            int, Tuple[StateObservationProtocol[Array, Array], Array]
        ] = {
            index: (
                StateEntries(bkd, nstates, rows[index]),
                StateEntries(bkd, nobs, positions[index]).apply(column),
            )
            for index in sorted(rows)
        }
        return MisfitTerm(bkd, observations, noise_std)

    def nqoi(self) -> int:
        return 1

    def reads(self, index: int, last: bool) -> bool:
        return index in self._members

    def at(self, index: int, time: float) -> StateTermProtocol[V, A]:
        return self._members[index]

    def derivatives(self) -> StoredStateTermDerivatives[V, A]:
        return self._derivatives


class EndpointTerm(AtStoredStates[V, A]):
    r"""The observation :math:`H y_N` of the last stored state: the
    :class:`ObservedValue` of :math:`H` read at the last state alone.

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
        super().__init__(ObservedValue(operator), StoredStates(last=True))
