r"""A collocation system's essential (Dirichlet) rows and their data.

The rows of the state pinned to boundary data :math:`g(t; p)`, declared as
:class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol` reads them. The
declaration does nothing to a residual: the shared pins (``RowPin`` for a
steady solve or a step's blocks, ``RowPinning``, ``PinnedRowsConstraint``)
apply it, including the data's parameter derivatives.

Data that depend on the parameters satisfy
:class:`ParameterizedBoundaryDataProtocol`; the data object, not the
declaration, owns the map from the parameters to the data and its
derivatives. :class:`MappedBoundaryData` is one such object, built on a
field map.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives
from pyapprox.ode.residual_derivatives import (
    DerivativeOperator,
    backend_spaces,
    backend_state_space,
)
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalCoefficientProtocol,
    as_nodal_coefficient,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
    FieldMapWithHVPProtocol,
    field_map_has_hvp,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator, BackendRowSet
from pyapprox.util.linalg.operators import SpaceDescriptor


@dataclass(frozen=True)
class BoundaryDataParamDerivatives(Generic[Array]):
    r"""The parameter derivatives of boundary data :math:`g(t; p)` on a
    part's nodes.

    Attributes
    ----------
    jacobian : callable
        ``time -> dg/dp``. Shape: (nboundary, nparams)
    curvature : callable or None
        ``(time, weights) -> sum_i w_i d^2 g_i/dp^2`` for weights of shape
        (nboundary,). Shape: (nparams, nparams). None when the data do not
        supply second derivatives; a second-order consumer then refuses.
    """

    jacobian: Callable[[float], Array]
    curvature: Optional[Callable[[float, Array], Array]] = None


@runtime_checkable
class ParameterizedBoundaryDataProtocol(Protocol, Generic[Array]):
    """Boundary data that depend on the parameters, read by
    :class:`DirichletRows`."""

    def values(self, time: float) -> Array:
        """:math:`g(t; p)` at the part's nodes. Shape: (nboundary,)"""
        ...

    def is_time_dependent(self) -> bool:
        """Whether the data change with time at fixed parameters."""
        ...

    def version(self) -> int:
        """Moves whenever the parameters are written."""
        ...

    def nparams(self) -> int:
        """The number of parameters."""
        ...

    def param_derivatives(self) -> BoundaryDataParamDerivatives[Array]:
        """The data's parameter derivatives; which fields are present is
        fixed for the object's life."""
        ...


class MappedBoundaryData(Generic[Array]):
    r"""Time-independent boundary data :math:`g = m(p)` from a field map
    over the part's nodes.

    The parameters are written with :meth:`set_parameters`. The curvature
    is supplied when the map provides a usable adjoint-weighted HVP; it is
    assembled column by column (``nparams`` HVPs), as parameter counts of
    boundary data are small.

    Parameters
    ----------
    field_map : FieldMapProtocol
        Maps the parameters to the data at the part's nodes.
    params : Array
        The initial parameters. Shape: (nparams,)
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        field_map: FieldMapProtocol[Array],
        params: Array,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(field_map, FieldMapProtocol):
            raise TypeError(
                "field_map must satisfy FieldMapProtocol, got "
                f"{type(field_map).__name__}"
            )
        self._field_map = field_map
        self._bkd = bkd
        self._hvp_map: Optional[FieldMapWithHVPProtocol[Array]] = (
            field_map
            if isinstance(field_map, FieldMapWithHVPProtocol)
            and field_map_has_hvp(field_map)
            else None
        )
        self._version = -1
        self.set_parameters(params)

    def set_parameters(self, params: Array) -> None:
        """Write the parameters. Shape: (nparams,)"""
        if params.shape != (self._field_map.nvars(),):
            raise ValueError(
                f"params must have shape ({self._field_map.nvars()},), got "
                f"{tuple(params.shape)}"
            )
        self._params = params
        self._version += 1

    def version(self) -> int:
        """The number of parameter writes after the first."""
        return self._version

    def values(self, time: float) -> Array:
        """:math:`m(p)`. Shape: (nboundary,)"""
        return self._field_map(self._params)

    def is_time_dependent(self) -> bool:
        return False

    def nparams(self) -> int:
        return self._field_map.nvars()

    def _jacobian(self, time: float) -> Array:
        return self._field_map.jacobian(self._params)

    def _curvature(self, time: float, weights: Array) -> Array:
        if self._hvp_map is None:
            raise RuntimeError("the field map supplies no HVP")
        bkd, nparams = self._bkd, self.nparams()
        identity = bkd.eye(nparams)
        return bkd.stack(
            [
                self._hvp_map.hvp(self._params, weights, identity[:, k])
                for k in range(nparams)
            ],
            axis=1,
        )

    def param_derivatives(self) -> BoundaryDataParamDerivatives[Array]:
        return BoundaryDataParamDerivatives(
            self._jacobian,
            None if self._hvp_map is None else self._curvature,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._field_map!r})"


# Data evaluated at the same node by two suppliers agree to round-off, so
# a larger relative difference is a real mismatch
_SHARED_ROW_RTOL = 1e-10


def component_state_rows(
    bkd: Backend[Array],
    boundary_indices: Array,
    component: int,
    npts: int,
    ncomponents: int,
) -> List[int]:
    """The rows of the component-stacked state for a component's boundary
    nodes: ``boundary_indices + component * npts``.

    Raises
    ------
    ValueError
        If ``component`` is not in ``[0, ncomponents)`` or an index is not
        in ``[0, npts)``.
    """
    if not 0 <= component < ncomponents:
        raise ValueError(f"component {component} is not in [0, {ncomponents})")
    # Read once at construction to build the row set
    indices = [bkd.to_int(index) for index in boundary_indices]
    if indices and (min(indices) < 0 or max(indices) >= npts):
        raise ValueError(
            f"boundary indices must lie in [0, {npts}), got "
            f"[{min(indices)}, {max(indices)}]"
        )
    return [index + component * npts for index in indices]


BoundaryDataInput = Union[
    CoefficientInput[Array], ParameterizedBoundaryDataProtocol[Array]
]


@dataclass(frozen=True)
class DirichletPart(Generic[Array]):
    """One part of a Dirichlet declaration: a component of the state fixed
    to data on a set of boundary nodes.

    Attributes
    ----------
    boundary_indices : Array
        Mesh node indices of the part. Shape: (nboundary,)
    data : float, NodalCoefficientProtocol, callable or
        ParameterizedBoundaryDataProtocol
        The data at those nodes: a constant, a ``NodalField`` of nodal
        values, or a ``NodalSupplier`` evaluating a declared
        ``f(coords, time)`` at the boundary nodes' coordinates (all
        independent of the parameters, in the forms
        ``as_nodal_coefficient`` takes); or data depending on the
        parameters.
    component : int
        The state component fixed (0 for a scalar physics).
    """

    boundary_indices: Array
    data: BoundaryDataInput[Array]
    component: int = 0


class _PlacedDataJacobian(Generic[Array]):
    r""":math:`t \mapsto \partial g/\partial p`, each parameterized part's
    rows placed in a parameters-to-states operator; zero on the rows of
    parts independent of the parameters."""

    def __init__(self, rows: "DirichletRows[Array]") -> None:
        self._rows = rows

    def __call__(self, time: float) -> DerivativeOperator[Array]:
        return self._rows._data_jacobian(time)


class _SummedDataCurvature(Generic[Array]):
    r""":math:`(t, w) \mapsto \sum_{i \in R} w_i\, \partial^2 g_i/\partial
    p^2`, each parameterized part weighted on the rows it supplies."""

    def __init__(self, rows: "DirichletRows[Array]") -> None:
        self._rows = rows

    def __call__(self, time: float, weights: Array) -> DerivativeOperator[Array]:
        return self._rows._data_curvature(time, weights)


class DirichletRows(Generic[Array]):
    r"""The essential rows of a collocation system and their data, as
    :class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol` declares them.

    A part's rows are its boundary node indices offset by
    ``component * npts`` in the component-stacked state. Parts may share
    rows (neighboring sides share corner nodes): a later part's data win a
    shared row, and its parameter derivatives with them.

    Parameter derivatives are stated zero when no part depends on the
    parameters. Otherwise :math:`\partial g/\partial p` is supplied, the
    curvature only when every parameterized part supplies it, and the
    rate's derivatives are not available (no rate is supplied).

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
        ``[0, ncomponents)``, a boundary index is not in ``[0, npts)``, or
        parameterized parts disagree on the number of parameters.
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
        rows_per_part = [self._part_rows(part, npts, ncomponents) for part in parts]
        self._rows_per_part = [
            bkd.asarray(rows, dtype=bkd.int64_dtype()) for rows in rows_per_part
        ]
        self._data: List[NodalCoefficientProtocol[Array]] = []
        # (part position, data) of each part depending on the parameters
        self._parameterized: List[
            tuple[int, ParameterizedBoundaryDataProtocol[Array]]
        ] = []
        for position, (part, rows) in enumerate(zip(parts, rows_per_part)):
            if isinstance(part.data, ParameterizedBoundaryDataProtocol):
                self._data.append(part.data)
                self._parameterized.append((position, part.data))
            else:
                self._data.append(as_nodal_coefficient(part.data, len(rows), bkd))
        # 1 where the part supplies the row, 0 where a later part does
        self._wins = [
            bkd.asarray(
                [
                    0.0
                    if any(row in later for later in rows_per_part[k + 1 :])
                    else 1.0
                    for row in rows
                ]
            )
            for k, rows in enumerate(rows_per_part)
        ]
        self._shared = self._shared_rows(rows_per_part)
        self._states = backend_state_space(bkd, self._nstates).descriptor()
        self._rows = BackendRowSet(
            self._states, [row for rows in rows_per_part for row in rows], bkd
        )
        self._params = self._parameter_space()

    def _part_rows(
        self, part: DirichletPart[Array], npts: int, ncomponents: int
    ) -> List[int]:
        return component_state_rows(
            self._bkd, part.boundary_indices, part.component, npts, ncomponents
        )

    def _shared_rows(
        self, rows_per_part: List[List[int]]
    ) -> List[Tuple[int, int, Array, Array, List[int]]]:
        """Each pair of parts sharing rows: (earlier part, later part, the
        positions of the shared rows in each part's data, the rows)."""
        shared = []
        for first, first_rows in enumerate(rows_per_part):
            for second in range(first + 1, len(rows_per_part)):
                second_rows = rows_per_part[second]
                common = [row for row in first_rows if row in second_rows]
                if common:
                    shared.append(
                        (
                            first,
                            second,
                            self._bkd.asarray(
                                [first_rows.index(row) for row in common],
                                dtype=self._bkd.int64_dtype(),
                            ),
                            self._bkd.asarray(
                                [second_rows.index(row) for row in common],
                                dtype=self._bkd.int64_dtype(),
                            ),
                            common,
                        )
                    )
        return shared

    def _check_shared_rows_agree(self, part_values: List[Array]) -> None:
        bkd = self._bkd
        for first, second, first_at, second_at, rows in self._shared:
            first_values = part_values[first][first_at]
            second_values = part_values[second][second_at]
            scale = bkd.maximum(
                bkd.maximum(bkd.abs(first_values), bkd.abs(second_values)),
                bkd.ones(first_values.shape),
            )
            mismatch = bkd.to_float(
                bkd.max(bkd.abs(first_values - second_values) / scale)
            )
            if mismatch > _SHARED_ROW_RTOL:
                raise ValueError(
                    f"Dirichlet parts {first} and {second} disagree on their "
                    f"shared rows {rows} (relative mismatch {mismatch:.3e}): "
                    "no smooth solution satisfies both. Make the data agree "
                    "there, or leave the shared nodes out of one part to "
                    "choose which condition holds"
                )

    def _parameter_space(self) -> Optional[SpaceDescriptor]:
        counts = {data.nparams() for _, data in self._parameterized}
        if not counts:
            return None
        if len(counts) > 1:
            raise ValueError(
                "parameterized Dirichlet parts disagree on the number of "
                f"parameters: {sorted(counts)}"
            )
        _, params = backend_spaces(self._bkd, self._nstates, counts.pop())
        return params.descriptor()

    def rows(self) -> BackendRowSet[Array]:
        """The pinned rows of the component-stacked state."""
        return self._rows

    def values(self, time: float) -> Array:
        """A column holding :math:`g(t)` on the pinned rows and zero
        elsewhere. Shape: (nstates, 1)

        Raises
        ------
        ValueError
            If a part's data do not have one value per boundary node, or
            two parts' data on a shared row (a corner) differ by more than
            a relative 1e-10: collocation needs a smooth solution, which
            cannot take two values there.
        """
        bkd = self._bkd
        part_values = []
        for rows, data in zip(self._rows_per_part, self._data):
            values = data.values(time)
            if values.shape != rows.shape:
                raise ValueError(
                    f"Dirichlet data must have shape {tuple(rows.shape)}, got "
                    f"{tuple(values.shape)}"
                )
            part_values.append(values)
        self._check_shared_rows_agree(part_values)
        column = bkd.zeros((self._nstates,))
        for rows, values in zip(self._rows_per_part, part_values):
            column = bkd.index_update(column, rows, values)
        return bkd.reshape(column, (self._nstates, 1))

    def rates(self) -> Optional[Callable[[float], Array]]:
        r"""None: :math:`\dot g` is not supplied. A pin at every block, or
        at the step end of a stiffly accurate method, needs none."""
        return None

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        """Stated zero when no part depends on the parameters; otherwise
        the values' Jacobian, their curvature when every parameterized part
        supplies one, and no rate derivatives."""
        if not self._parameterized:
            return PinnedRowsParamDerivatives.independent()
        curvature_available = all(
            data.param_derivatives().curvature is not None
            for _, data in self._parameterized
        )
        return PinnedRowsParamDerivatives(
            values=_PlacedDataJacobian(self),
            rates=None,
            values_curvature=(
                _SummedDataCurvature(self) if curvature_available else None
            ),
            rates_curvature=None,
        )

    def _require_params(self) -> SpaceDescriptor:
        if self._params is None:
            raise RuntimeError("no part depends on the parameters")
        return self._params

    def _data_jacobian(self, time: float) -> DerivativeOperator[Array]:
        bkd, params = self._bkd, self._require_params()
        parameterized = dict(self._parameterized)
        matrix = bkd.zeros((self._nstates, params.size))
        # Every part writes its rows in order, so a later part wins a
        # shared row; a part independent of the parameters writes zeros
        for position, rows in enumerate(self._rows_per_part):
            shape = (rows.shape[0], params.size)
            data = parameterized.get(position)
            if data is None:
                matrix = bkd.index_update(matrix, rows, bkd.zeros(shape))
                continue
            jacobian = data.param_derivatives().jacobian(time)
            if jacobian.shape != shape:
                raise ValueError(
                    f"boundary data Jacobian must have shape {shape}, got "
                    f"{tuple(jacobian.shape)}"
                )
            matrix = bkd.index_update(matrix, rows, jacobian)
        return BackendDenseOperator(matrix, params, bkd, range=self._states)

    def _data_curvature(self, time: float, weights: Array) -> DerivativeOperator[Array]:
        bkd, params = self._bkd, self._require_params()
        flat = bkd.flatten(weights)
        matrix = bkd.zeros((params.size, params.size))
        for position, data in self._parameterized:
            curvature = data.param_derivatives().curvature
            if curvature is None:
                raise RuntimeError("a parameterized part supplies no curvature")
            rows = self._rows_per_part[position]
            matrix = matrix + curvature(time, flat[rows] * self._wins[position])
        return BackendDenseOperator(matrix, params, bkd)

    def is_time_invariant(self) -> bool:
        """Whether every part's data are declared time-invariant."""
        return not any(data.is_time_dependent() for data in self._data)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nparts={len(self._data)}, "
            f"nrows={self._rows.count()})"
        )
