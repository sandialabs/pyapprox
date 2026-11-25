r"""A collocation system second order in time, as the doubled first-order
system the tableau and multistep steppers integrate.

:func:`collocation_second_order_system` builds, from a
:class:`~pyapprox.pde.models.collocation.closed_system.CollocationSystem`
whose physics is second order in time,
:math:`\mathrm{diag}(\rho)\, \ddot u = f(u, t; p) + g(u, \dot u, t)`, the
system of :math:`w = [u; v]` (:func:`~pyapprox.ode.second_order.reduction.
second_order_system`) with its boundary rows:

- interior node: :math:`\dot u = v` and :math:`\rho \dot v = f + g`;
- essential node: :math:`u` pinned to :math:`g(t)` and :math:`v` to
  :math:`\dot g(t)`
  (:class:`~pyapprox.ode.second_order.doubled_rows.DoubledPinnedRows`);
- closure node (traction, Robin): the displacement row stays
  :math:`\dot u = v`; the velocity row carries the closure, in the form the
  caller injects, in the blocks the placement selects
  (:class:`~pyapprox.ode.affine_rows.AffineRowPinning`).

The displacement rows are edited only by essential pins, so the first
block row stays zero and identity and either storage of the blocks
serves. The mass is the physics' nodal density, the same on every row:
a pinned or closure row's mass multiplies a row the pin replaces.

Each choice goes to the one class that reads it, with no default: the
storage (``blocks``), where the pins act (``placement``), whether the
essential rows carry the rate of their data (``treatment``), the closure
form (``closure_form``) and the initial velocity's parameter dependence.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.affine_rows import AffineRowPinning, AffineRowsProtocol
from pyapprox.ode.differentiated_rows import (
    RowRatesTreatmentProtocol,
    differentiated_rows,
)
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.pinned_rows import (
    PinnedRowsParamDerivatives,
    PinnedRowsProtocol,
    PinPlacementProtocol,
    RowPinning,
    RowPinningProtocol,
)
from pyapprox.ode.second_order.closure_forms import (
    ClosureBlocksProtocol,
    DisplacementConstraintProtocol,
)
from pyapprox.ode.second_order.doubled_rows import (
    DoubledPinnedRows,
    TimeDerivativeRowsProtocol,
)
from pyapprox.ode.second_order.reduction import (
    InitialVelocityDerivatives,
    StateBlocksProtocol,
    VelocityForceProtocol,
    second_order_system,
)
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.closure_pinning import ClosureRelation
from pyapprox.pde.models.collocation.unconstrained_system import _weighted_system
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import FormatProtocol, RowSetProtocol

G = TypeVar("G", bound=FormatProtocol)


@runtime_checkable
class SecondOrderInteriorOperatorProtocol(
    CollocationInteriorOperatorProtocol[Array], Protocol, Generic[Array]
):
    r"""An interior operator second order in time that declares the nodal
    weights of its second time derivative,
    :math:`\mathrm{diag}(\rho)\, \ddot u = f(u, t)`."""

    def interior_density_weights(self) -> Array:
        """The nodal weights, positive. Shape: (nstates,)"""
        ...


@runtime_checkable
class SecondOrderStorageProtocol(
    StateBlocksProtocol[Array, Array, DenseFormat[Array], G], Protocol[Array, G]
):
    """A storage of :math:`w = [u; v]` over collocation's dense blocks
    that also lifts row sets of one block to row sets of :math:`w`, as the
    doubled essential rows and the closure form need; both storages of
    :mod:`pyapprox.ode.second_order` satisfy it."""

    def lift_rows(
        self, displacement: RowSetProtocol[Array], velocity: RowSetProtocol[Array]
    ) -> RowSetProtocol[Array]:
        """The rows of :math:`w` that are ``displacement``'s rows of the
        first block and ``velocity``'s of the second."""
        ...


@runtime_checkable
class ClosureFormProtocol(Protocol[Array, G]):
    r"""How the closure rows' relation :math:`c(u, t) = 0` on the
    displacement is placed on the velocity rows of :math:`w`, e.g.
    :class:`~pyapprox.ode.second_order.closure_forms.DirectClosureForm`."""

    def relation(
        self,
        constraint: DisplacementConstraintProtocol[Array, DenseFormat[Array]],
        blocks: ClosureBlocksProtocol[Array, DenseFormat[Array], G],
    ) -> AffineRowsProtocol[Array, G]:
        """The relation on :math:`w` the closure pin imposes."""
        ...


class _ZeroRates(Generic[Array]):
    r""":math:`t \mapsto 0`, a column of :math:`w`."""

    def __init__(self, zero: Array) -> None:
        self._zero = zero

    def __call__(self, time: float) -> Array:
        return self._zero


class _NoDoubledRows(Generic[Array]):
    """The essential declaration of :math:`w` for a system without
    essential rows: no rows, zero data."""

    def __init__(
        self,
        blocks: SecondOrderStorageProtocol[Array, G],
        half_rows: RowSetProtocol[Array],
    ) -> None:
        empty = half_rows.difference(half_rows)
        self._rows = blocks.lift_rows(empty, empty)
        self._zero = blocks.space().zeros()

    def rows(self) -> RowSetProtocol[Array]:
        return self._rows

    def values(self, time: float) -> Array:
        return self._zero

    def rates(self) -> Optional[Callable[[float], Array]]:
        return _ZeroRates(self._zero)

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        return PinnedRowsParamDerivatives.independent()


@dataclass(frozen=True)
class CollocationSecondOrderSystem(Generic[Array, G]):
    """What a transient solve of a collocation system second order in time
    and its sweeps read.

    Built by :func:`collocation_second_order_system`; it satisfies the
    discretization and parameter-setter protocols of
    :mod:`pyapprox.ode.transient_implicit_map`.

    Attributes
    ----------
    solved : MassFormSystem
        The doubled system stepped, with the essential rows in
        differentiated form when a treatment was given.
    collocation : CollocationSystem
        The physics with its essential and closure declarations.
    doubled : DoubledPinnedRows or None
        The essential rows of both blocks; None without essential rows.
    row_pinning : RowPinningProtocol or None
        Pins the doubled essential rows and imposes the closure relation
        on the velocity rows; None without boundary rows.
    parameters : ParameterState or None
        The shared parameters; none for a system without them.
    """

    solved: MassFormSystem[Array, Array, G]
    collocation: CollocationSystem[Array]
    doubled: Optional[DoubledPinnedRows[Array]]
    row_pinning: Optional[RowPinningProtocol[Array, G]]
    parameters: Optional[ParameterState[Array]]

    def system(self) -> MassFormSystem[Array, Array, G]:
        """The solved system."""
        return self.solved

    def pinning(self) -> Optional[RowPinningProtocol[Array, G]]:
        """The boundary rows' pin."""
        return self.row_pinning

    def initial_state(self, state: Array, time: float) -> Array:
        r"""``state`` :math:`= [u_0; v_0]`, a column of :math:`w`, with its
        essential entries set to :math:`[g(t); \dot g(t)]`; the closure rows
        are left as given."""
        if self.doubled is None:
            return state
        return self.doubled.rows().replace_rows(state, self.doubled.values(time))

    def nparams(self) -> int:
        """The number of parameters; 0 without them."""
        return 0 if self.parameters is None else self.parameters.nparams()

    def set_param(self, param: Array) -> None:
        """Set the parameters from a column of shape ``(nparams, 1)``.

        Raises
        ------
        ValueError
            If the system has no parameters.
        """
        if self.parameters is None:
            raise ValueError("this system has no parameters to set")
        self.parameters.set_param(param[:, 0])


def collocation_second_order_system(
    system: CollocationSystem[Array],
    blocks: SecondOrderStorageProtocol[Array, G],
    placement: PinPlacementProtocol,
    treatment: Optional[RowRatesTreatmentProtocol],
    closure_form: ClosureFormProtocol[Array, G],
    initial_velocity: InitialVelocityDerivatives[Array],
    velocity_force: Optional[VelocityForceProtocol[Array, DenseFormat[Array]]] = None,
    parameters: Optional[ParameterState[Array]] = None,
) -> CollocationSecondOrderSystem[Array, G]:
    r"""The doubled first-order system of a collocation system second order
    in time (see the module docstring).

    Parameters
    ----------
    system : CollocationSystem
        The physics, satisfying :class:`SecondOrderInteriorOperatorProtocol`,
        with its essential and closure declarations. The essential rows
        must declare their data's time derivatives
        (:class:`~pyapprox.ode.second_order.doubled_rows.
        TimeDerivativeRowsProtocol`).
    blocks : SecondOrderStorageProtocol
        How :math:`w` is stored and solved with: assembled or condensed.
    placement : PinPlacementProtocol
        The blocks in which the boundary rows are pinned.
    treatment : RowRatesTreatmentProtocol or None
        How the essential rows carry the rate of their data in every
        block, or None to leave them raw. Given, the data must supply their
        second time derivative.
    closure_form : ClosureFormProtocol
        How the closure rows' relation is placed on the velocity rows;
        read only when the system has closure rows.
    initial_velocity : InitialVelocityDerivatives
        :math:`\partial v_0/\partial p`.
    velocity_force : VelocityForceProtocol, optional
        A damping force :math:`g(u, v, t)`; none for an undamped system.
    parameters : ParameterState, optional
        The parameters; none for a system without them.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol, the physics declares
        no density weights, or the essential rows declare no time
        derivatives of their data.
    ValueError
        If the density weights are not one per state or not positive, a
        treatment is given without essential rows, the essential data
        supply no rate (or no second rate a treatment needs) or depend on
        the parameters, or a closure cannot be pinned.
    """
    if not isinstance(system, CollocationSystem):
        raise TypeError(
            f"system must be a CollocationSystem, got {type(system).__name__}"
        )
    if not isinstance(blocks, SecondOrderStorageProtocol):
        raise TypeError(
            "blocks must satisfy SecondOrderStorageProtocol, got "
            f"{type(blocks).__name__}"
        )
    if not isinstance(placement, PinPlacementProtocol):
        raise TypeError(
            "placement must satisfy PinPlacementProtocol, got "
            f"{type(placement).__name__}"
        )
    if not isinstance(closure_form, ClosureFormProtocol):
        raise TypeError(
            "closure_form must satisfy ClosureFormProtocol, got "
            f"{type(closure_form).__name__}"
        )
    physics = system.physics()
    if not isinstance(physics, SecondOrderInteriorOperatorProtocol):
        raise TypeError(
            "the physics must declare its density weights "
            "(SecondOrderInteriorOperatorProtocol), got "
            f"{type(physics).__name__}"
        )
    weights = physics.interior_density_weights()
    bkd = system.bkd()
    if not bkd.to_bool(bkd.all(weights > 0.0)):
        raise ValueError("the density weights must be positive on every row")
    restoring = _weighted_system(system, weights, parameters, None)
    reduced = second_order_system(restoring, blocks, initial_velocity, velocity_force)
    doubled = _doubled_rows(system, blocks)
    if treatment is None:
        solved = reduced
    elif doubled is None:
        raise ValueError(
            "a treatment of the essential rows was given, but the system has "
            "no essential rows"
        )
    else:
        solved = differentiated_rows(reduced, doubled, treatment)
    return CollocationSecondOrderSystem(
        solved=solved,
        collocation=system,
        doubled=doubled,
        row_pinning=_pinning(
            system, blocks, placement, closure_form, doubled, parameters
        ),
        parameters=parameters,
    )


def _doubled_rows(
    system: CollocationSystem[Array], blocks: SecondOrderStorageProtocol[Array, G]
) -> Optional[DoubledPinnedRows[Array]]:
    essential = system.essential()
    if essential is None:
        return None
    if not isinstance(essential, TimeDerivativeRowsProtocol):
        raise TypeError(
            "the essential rows must declare their data's time derivatives "
            "(TimeDerivativeRowsProtocol): the velocity rows are pinned to "
            f"the data's rate; got {type(essential).__name__}"
        )
    return DoubledPinnedRows(essential, blocks)


def _pinning(
    system: CollocationSystem[Array],
    blocks: SecondOrderStorageProtocol[Array, G],
    placement: PinPlacementProtocol,
    closure_form: ClosureFormProtocol[Array, G],
    doubled: Optional[DoubledPinnedRows[Array]],
    parameters: Optional[ParameterState[Array]],
) -> Optional[RowPinningProtocol[Array, G]]:
    if not system.closures():
        return None if doubled is None else RowPinning(doubled, placement)
    constraint = ClosureRelation(system, parameters)
    essential: PinnedRowsProtocol[Array] = (
        _NoDoubledRows(blocks, constraint.rows()) if doubled is None else doubled
    )
    return AffineRowPinning(
        closure_form.relation(constraint, blocks),
        essential,
        blocks.space(),
        placement,
    )
