"""Subset objectives from objectives of relaxed design weights.

Choosing a subset of design variables is evaluating a relaxed objective at
the 0/1 design that switches exactly that subset on. This is exact for
any objective whose zero weight removes a design variable, such as the
moment-Gaussian ``DesignObjective``, and it works for every criterion and
through any ``ParameterizedObjective``, where a design variable is a group
of observations.

Not every ``OEDObjectiveProtocol`` qualifies. The KL-OED and risk-based
objectives weight the noise precision, with noise variance
``sigma^2 / w``, so they are undefined at ``w = 0``; ``BruteForceKLOEDSolver``
gives unchosen observations a small positive weight instead.
"""

import math
from typing import Generic, Sequence, Tuple

from pyapprox.expdesign.protocols.objective import OEDObjectiveProtocol
from pyapprox.expdesign.protocols.subset import SubsetObjectiveProtocol
from pyapprox.util.backends.protocols import Array, Backend


def _check_subset(subset: Sequence[int], ncandidates: int) -> None:
    bad = [ii for ii in subset if not 0 <= ii < ncandidates]
    if bad:
        raise ValueError(
            f"indices {bad} are outside the {ncandidates} design variables"
        )
    if len(set(subset)) != len(subset):
        raise ValueError(f"subset {list(subset)} repeats design variables")


class BinaryDesignSubsetObjective(Generic[Array]):
    """``objective`` at the 0/1 design that switches ``subset`` on.

    Satisfies ``SubsetObjectiveProtocol``.

    Parameters
    ----------
    objective : OEDObjectiveProtocol[Array]
        An objective of relaxed design variables in [0, 1] that is defined
        at zero weights, where a zero weight removes its design variable.
        The moment-Gaussian ``DesignObjective`` does so exactly; the
        precision-weighted KL-OED and risk objectives do not, and give a
        non-finite value, which raises. Unchosen variables get weight
        exactly 0, not a small positive weight.
    """

    def __init__(self, objective: OEDObjectiveProtocol[Array]) -> None:
        if not isinstance(objective, OEDObjectiveProtocol):
            raise TypeError(
                "objective must satisfy OEDObjectiveProtocol, got "
                f"{type(objective).__name__}"
            )
        self._objective = objective

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._objective.bkd()

    def ncandidates(self) -> int:
        """Number of design variables of ``objective``."""
        return self._objective.nvars()

    def design(self, subset: Sequence[int]) -> Array:
        """The 0/1 design that switches ``subset`` on. Shape: (ncandidates, 1)"""
        _check_subset(subset, self.ncandidates())
        design = self.bkd().zeros((self.ncandidates(), 1))
        for ii in subset:
            design[ii, 0] = 1.0
        return design

    def value(self, subset: Sequence[int]) -> float:
        """``objective`` at ``design(subset)``.

        Raises
        ------
        ValueError
            If the value is not finite, as for an objective undefined at
            zero weights.
        """
        value = self.bkd().to_float(self._objective(self.design(subset))[0, 0])
        if not math.isfinite(value):
            raise ValueError(
                f"objective is {value} at the 0/1 design of subset "
                f"{list(subset)}; it must be defined at zero weights"
            )
        return value


class ReevaluatingIncremental(Generic[Array]):
    """The incremental protocol for any subset objective, by rescoring.

    Satisfies ``IncrementalSubsetObjectiveProtocol`` with the chosen
    indices, sorted, as its state. Each value rescores the whole extended
    set, so it suits every criterion; an objective with a cheaper update
    can implement the protocol itself and be passed to the same solvers.

    Parameters
    ----------
    subset_objective : SubsetObjectiveProtocol[Array]
        The objective to rescore.
    """

    def __init__(self, subset_objective: SubsetObjectiveProtocol[Array]) -> None:
        if not isinstance(subset_objective, SubsetObjectiveProtocol):
            raise TypeError(
                "subset_objective must satisfy SubsetObjectiveProtocol, got "
                f"{type(subset_objective).__name__}"
            )
        self._subset_objective = subset_objective

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._subset_objective.bkd()

    def ncandidates(self) -> int:
        """Number of design variables a subset is drawn from."""
        return self._subset_objective.ncandidates()

    def initial_state(self) -> Tuple[int, ...]:
        """No design variables chosen."""
        return ()

    def _extend(
        self, state: Tuple[int, ...], addition: Sequence[int]
    ) -> Tuple[int, ...]:
        overlap = set(state).intersection(addition)
        if overlap:
            raise ValueError(f"design variables {sorted(overlap)} are already chosen")
        extended = state + tuple(addition)
        _check_subset(extended, self.ncandidates())
        return tuple(sorted(extended))

    def values_after(
        self, state: Tuple[int, ...], additions: Sequence[Sequence[int]]
    ) -> Array:
        """The value of ``state`` plus each addition. Shape: (len(additions),)"""
        values = [
            self._subset_objective.value(self._extend(state, addition))
            for addition in additions
        ]
        return self.bkd().asarray(values)

    def add(self, state: Tuple[int, ...], addition: Sequence[int]) -> Tuple[int, ...]:
        """The sorted indices of ``state`` and ``addition``."""
        return self._extend(state, addition)
