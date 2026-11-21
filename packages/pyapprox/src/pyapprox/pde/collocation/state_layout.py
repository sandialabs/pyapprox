"""Where each field of a collocation state sits.

A collocation state stacks its fields, each on its own nodes: field
``k`` occupies the rows ``offset(k) .. offset(k) + npts(k) - 1``, and the
node ``i`` of field ``k`` is the row ``offset(k) + i``. When every field
shares one grid of ``npts`` nodes (:meth:`StateLayout.uniform`), that row
is ``i + k * npts``. Fields on different grids, such as a velocity on the
Gauss-Lobatto nodes and a pressure on interior Gauss nodes, give each
field its own node count.
"""

from typing import List, Optional, Sequence, Tuple


class StateLayout:
    """The fields of a collocation state, each with its own node count,
    stacked in order.

    Parameters
    ----------
    npts_per_field : sequence of int
        The number of nodes of each field, in stacking order.
    names : sequence of str, optional
        A name for each field. Default ``("0", "1", ...)``.

    Raises
    ------
    ValueError
        If there are no fields, a field has no nodes, or ``names`` has
        another length or repeats a name.
    """

    def __init__(
        self,
        npts_per_field: Sequence[int],
        names: Optional[Sequence[str]] = None,
    ) -> None:
        counts = tuple(int(n) for n in npts_per_field)
        if not counts:
            raise ValueError("a state layout needs at least one field")
        if min(counts) < 1:
            raise ValueError(f"every field needs at least one node, got {counts}")
        labels = (
            tuple(str(k) for k in range(len(counts))) if names is None else tuple(names)
        )
        if len(labels) != len(counts):
            raise ValueError(f"{len(labels)} names given for {len(counts)} fields")
        if len(set(labels)) != len(labels):
            raise ValueError(f"field names repeat: {labels}")
        self._npts = counts
        self._names = labels
        offsets = [0]
        for count in counts[:-1]:
            offsets.append(offsets[-1] + count)
        self._offsets = tuple(offsets)

    @classmethod
    def uniform(
        cls, npts: int, nfields: int, names: Optional[Sequence[str]] = None
    ) -> "StateLayout":
        """``nfields`` fields on one grid of ``npts`` nodes."""
        return cls([npts] * nfields, names)

    def nfields(self) -> int:
        """The number of fields."""
        return len(self._npts)

    def nstates(self) -> int:
        """The number of state entries: the fields' node counts summed."""
        return self._offsets[-1] + self._npts[-1]

    def names(self) -> Tuple[str, ...]:
        """The field names, in stacking order."""
        return self._names

    def field(self, name: str) -> int:
        """The position of the field called ``name``.

        Raises
        ------
        ValueError
            If no field has that name.
        """
        if name not in self._names:
            raise ValueError(f"no field {name!r}; fields are {self._names}")
        return self._names.index(name)

    def npts(self, field: int) -> int:
        """The number of nodes of ``field``."""
        self._check_field(field)
        return self._npts[field]

    def offset(self, field: int) -> int:
        """The row of node 0 of ``field``."""
        self._check_field(field)
        return self._offsets[field]

    def rows(self, field: int, node_indices: Sequence[int]) -> List[int]:
        """The state rows of ``field`` at its nodes ``node_indices``.

        Raises
        ------
        ValueError
            If ``field`` is out of range or a node index is not in
            ``[0, npts(field))``.
        """
        npts = self.npts(field)
        indices = [int(index) for index in node_indices]
        if indices and (min(indices) < 0 or max(indices) >= npts):
            raise ValueError(
                f"node indices of field {field} must lie in [0, {npts}), got "
                f"[{min(indices)}, {max(indices)}]"
            )
        start = self._offsets[field]
        return [start + index for index in indices]

    def is_uniform(self) -> bool:
        """Whether every field has the same number of nodes."""
        return len(set(self._npts)) == 1

    def _check_field(self, field: int) -> None:
        if not 0 <= field < len(self._npts):
            raise ValueError(f"field {field} is not in [0, {len(self._npts)})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateLayout):
            return NotImplemented
        return self._npts == other._npts and self._names == other._names

    def __hash__(self) -> int:
        return hash((self._npts, self._names))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._npts)}, names={list(self._names)})"
