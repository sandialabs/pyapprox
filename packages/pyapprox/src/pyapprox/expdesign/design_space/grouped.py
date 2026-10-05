r"""Designs that switch groups of observations together.

A group design has one weight :math:`v_j` per group of observations, and
each observation takes its group's weight: :math:`w = P v` with
:math:`P_{ij} = 1` when observation :math:`i` is in group :math:`j`. Which
observations form a group is entirely the caller's: a sensor's
measurements at every time, every sensor at one time, or singletons for a
fully independent design are all just different groups.
"""

from typing import Generic, List, Sequence

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend


class GroupedDesign(Generic[Array]):
    """The map ``v -> w = P v`` from group weights to observation weights.

    Satisfies ``FunctionProtocol`` with a constant Jacobian ``P``, so it can
    parameterize an objective through ``ParameterizedObjective``.

    Parameters
    ----------
    groups : Sequence[Sequence[int]]
        Observation indices of each group, in any order. Groups must be
        non-empty and disjoint, so every ``w_i`` stays in [0, 1].
        Observations in no group always get weight 0.
    nobs : int
        Number of observations.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self, groups: Sequence[Sequence[int]], nobs: int, bkd: Backend[Array]
    ) -> None:
        if len(groups) == 0:
            raise ValueError("groups is empty")
        seen: set[int] = set()
        for jj, group in enumerate(groups):
            if len(group) == 0:
                raise ValueError(f"group {jj} is empty")
            bad = [ii for ii in group if not 0 <= ii < nobs]
            if bad:
                raise ValueError(
                    f"group {jj} has indices {bad} outside the {nobs} observations"
                )
            repeated = seen.intersection(group)
            if repeated or len(set(group)) != len(group):
                raise ValueError(
                    f"group {jj} repeats observations {sorted(repeated)}; groups "
                    "must be disjoint so every weight stays in [0, 1]"
                )
            seen.update(group)
        self._groups: List[List[int]] = [list(group) for group in groups]
        self._nobs = nobs
        self._bkd = bkd
        matrix = bkd.zeros((nobs, len(groups)))
        for jj, group in enumerate(self._groups):
            for ii in group:
                matrix[ii, jj] = 1.0
        self._matrix = matrix
        self._derivatives = Derivatives.first_order(jacobian=self._jacobian)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of groups, the design variables."""
        return len(self._groups)

    def nqoi(self) -> int:
        """Number of observations."""
        return self._nobs

    def groups(self) -> List[List[int]]:
        """Observation indices of each group."""
        return [list(group) for group in self._groups]

    def matrix(self) -> Array:
        """``P``. Shape: (nobs, ngroups)"""
        return self._matrix

    def __call__(self, samples: Array, /) -> Array:
        """``w = P v``. Shape: (ngroups, n) to (nobs, n)"""
        if samples.ndim != 2 or samples.shape[0] != self.nvars():
            raise ValueError(
                f"samples must have shape ({self.nvars()}, n), got "
                f"{tuple(samples.shape)}"
            )
        return self._bkd.dot(self._matrix, samples)

    def _jacobian(self, sample: Array) -> Array:
        return self._matrix

    def derivatives(self) -> Derivatives[Array]:
        """First-order bundle: the constant Jacobian ``P``."""
        return self._derivatives
