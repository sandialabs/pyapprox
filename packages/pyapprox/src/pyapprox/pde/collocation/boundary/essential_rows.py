r"""A collocation system's essential (Dirichlet) rows and their data.

The rows of the state pinned to boundary data :math:`g(t)`, declared as
:class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol` reads them. The
declaration does nothing to a residual: the shared pins (``RowPin`` for a
steady solve or a step's blocks, ``RowPinning``, ``PinnedRowsConstraint``)
apply it.
"""

from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Sequence

from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalCoefficientProtocol,
    as_nodal_coefficient,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet


@dataclass(frozen=True)
class DirichletPart(Generic[Array]):
    """One part of a Dirichlet declaration: a component of the state fixed
    to data on a set of boundary nodes.

    Attributes
    ----------
    boundary_indices : Array
        Mesh node indices of the part. Shape: (nboundary,)
    data : float, NodalCoefficientProtocol or callable
        The data at those nodes, in the forms ``as_nodal_coefficient``
        takes: a constant, a ``NodalField`` of nodal values, or a
        ``NodalSupplier`` evaluating a declared ``f(coords, time)`` at the
        boundary nodes' coordinates.
    component : int
        The state component fixed (0 for a scalar physics).
    """

    boundary_indices: Array
    data: CoefficientInput[Array]
    component: int = 0


class DirichletRows(Generic[Array]):
    r"""The essential rows of a collocation system and their data, as
    :class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol` declares them.

    A part's rows are its boundary node indices offset by
    ``component * npts`` in the component-stacked state. Parts may share
    rows (neighboring sides share corner nodes): a later part's data win a
    shared row.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    npts : int
        The number of collocation nodes.
    ncomponents : int
        The number of state components; the state has
        ``ncomponents * npts`` entries.
    parts : sequence of DirichletPart
        The parts, in the order that settles shared rows.

    Raises
    ------
    ValueError
        If there are no parts, a component is not in
        ``[0, ncomponents)``, or a boundary index is not in ``[0, npts)``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        npts: int,
        ncomponents: int,
        parts: Sequence[DirichletPart[Array]],
    ) -> None:
        if not parts:
            raise ValueError("a Dirichlet declaration needs at least one part")
        self._bkd = bkd
        self._nstates = npts * ncomponents
        self._rows_per_part: List[Array] = []
        self._data: List[NodalCoefficientProtocol[Array]] = []
        all_rows: List[int] = []
        for part in parts:
            if not 0 <= part.component < ncomponents:
                raise ValueError(
                    f"component {part.component} is not in [0, {ncomponents})"
                )
            # Read once at construction to build the row set
            indices = [bkd.to_int(index) for index in part.boundary_indices]
            if indices and (min(indices) < 0 or max(indices) >= npts):
                raise ValueError(
                    f"boundary indices must lie in [0, {npts}), got "
                    f"[{min(indices)}, {max(indices)}]"
                )
            rows = [index + part.component * npts for index in indices]
            all_rows.extend(rows)
            self._rows_per_part.append(bkd.asarray(rows, dtype=bkd.int64_dtype()))
            self._data.append(as_nodal_coefficient(part.data, len(rows), bkd))
        descriptor = backend_state_space(bkd, self._nstates).descriptor()
        self._rows = BackendRowSet(descriptor, all_rows, bkd)

    def rows(self) -> BackendRowSet[Array]:
        """The pinned rows of the component-stacked state."""
        return self._rows

    def values(self, time: float) -> Array:
        """A column holding :math:`g(t)` on the pinned rows and zero
        elsewhere. Shape: (nstates, 1)

        Raises
        ------
        ValueError
            If a part's data do not have one value per boundary node.
        """
        bkd = self._bkd
        column = bkd.zeros((self._nstates,))
        for rows, data in zip(self._rows_per_part, self._data):
            values = data.values(time)
            if values.shape != rows.shape:
                raise ValueError(
                    f"Dirichlet data must have shape {tuple(rows.shape)}, got "
                    f"{tuple(values.shape)}"
                )
            column = bkd.index_update(column, rows, values)
        return bkd.reshape(column, (self._nstates, 1))

    def rates(self) -> Optional[Callable[[float], Array]]:
        r"""None: :math:`\dot g` is not supplied. A pin at every block, or
        at the step end of a stiffly accurate method, needs none."""
        return None

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        """Independent of the parameters."""
        return PinnedRowsParamDerivatives.independent()

    def is_time_invariant(self) -> bool:
        """Whether every part's data are declared time-invariant."""
        return not any(data.is_time_dependent() for data in self._data)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nparts={len(self._data)}, "
            f"nrows={self._rows.count()})"
        )
