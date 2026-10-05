"""Protocols for objectives of a subset of the design variables.

A subset objective scores a set of chosen design variables, for example
sensors or groups of observations, with every unchosen one switched off.
Values are minimized, like every pyapprox objective.

The incremental protocol lets a search grow a chosen set without
rescoring it from scratch. Its state is opaque and owned by the objective:
a plain implementation keeps the chosen indices and rescores, a fast one
keeps whatever its updates need, such as covariances conditioned on the
chosen observations. Solvers see only the protocol, so either can be
passed to the same solver.
"""

from typing import Generic, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend

State = TypeVar("State")


@runtime_checkable
class SubsetObjectiveProtocol(Protocol, Generic[Array]):
    """The value of choosing a subset of the design variables.

    Methods
    -------
    bkd()
        Get the computational backend.
    ncandidates()
        Number of design variables a subset is drawn from.
    value(subset)
        The value, to be minimized, of choosing ``subset``.
    """

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        ...

    def ncandidates(self) -> int:
        """Number of design variables a subset is drawn from."""
        ...

    def value(self, subset: Sequence[int]) -> float:
        """The value of choosing ``subset``, to be minimized.

        Parameters
        ----------
        subset : Sequence[int]
            Distinct indices of the chosen design variables, in any order.
            May be empty.

        Returns
        -------
        float
            The objective with exactly ``subset`` switched on.
        """
        ...


@runtime_checkable
class IncrementalSubsetObjectiveProtocol(Protocol, Generic[Array, State]):
    """A subset objective that scores additions to a chosen set.

    ``State`` stands for a chosen set; its contents are the objective's
    own. States are values: ``add`` returns a new state and leaves its
    argument usable, so a search can keep several.

    Methods
    -------
    bkd()
        Get the computational backend.
    ncandidates()
        Number of design variables a subset is drawn from.
    initial_state()
        The state of the empty set.
    values_after(state, additions)
        Values of the chosen set extended by each addition.
    add(state, addition)
        The state of the chosen set extended by ``addition``.
    """

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        ...

    def ncandidates(self) -> int:
        """Number of design variables a subset is drawn from."""
        ...

    def initial_state(self) -> State:
        """The state of the empty set."""
        ...

    def values_after(self, state: State, additions: Sequence[Sequence[int]]) -> Array:
        """Values of the chosen set extended by each addition.

        Parameters
        ----------
        state : State
            The chosen set.
        additions : Sequence[Sequence[int]]
            Sets of design variables to try adding, each disjoint from the
            chosen set. Single candidates are one-element sets; a batch
            search passes larger ones.

        Returns
        -------
        Array
            The value of the chosen set plus each addition, to be
            minimized. Shape: (len(additions),)
        """
        ...

    def add(self, state: State, addition: Sequence[int]) -> State:
        """The state of the chosen set extended by ``addition``.

        Parameters
        ----------
        state : State
            The chosen set; not modified.
        addition : Sequence[int]
            Design variables to add, disjoint from the chosen set.
        """
        ...
