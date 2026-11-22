r"""A transient collocation system for the tableau and multistep steppers.

:func:`collocation_transient_system` builds, from a
:class:`~pyapprox.pde.models.collocation.closed_system.CollocationSystem`,
what a transient solve and its sweeps read: the system stepped, the pin of
its boundary rows, the unconstrained system and the shared parameter state.
The result satisfies the discretization and parameter-setter protocols of
:mod:`pyapprox.ode.transient_implicit_map`, so
:class:`~pyapprox.ode.transient_implicit_map.TransientImplicitMap` solves
and differentiates it with an injected method, setups, steps and
functional, and
:class:`~pyapprox.implicitfunction.implicit_map.ImplicitMapFunction` gives
it the ``FunctionProtocol`` face.

How the boundary rows are treated in time is the caller's choice, with no
default: where the pin acts (``placement``: the step end, every block) and
whether the essential rows carry the rate of their data (``treatment``:
:class:`~pyapprox.ode.differentiated_rows.CoupledRows`,
:class:`~pyapprox.ode.differentiated_rows.SymmetricRows`, or None). Which
combination to prefer depends on the method; a step-end pin, for one,
leaves an explicit multistage method's inner stages unpinned, and Heun's
then does not preserve a steady state.
"""

from dataclasses import dataclass
from typing import Generic, Optional

from pyapprox.ode.differentiated_rows import (
    RowRatesTreatmentProtocol,
    differentiated_rows,
)
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.pinned_rows import PinPlacementProtocol
from pyapprox.ode.residual_derivatives import StatedZero
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.closure_pinning import ClosurePinning
from pyapprox.pde.models.collocation.unconstrained_system import (
    collocation_unconstrained_system,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import LinearSolverProtocol

_Dense = MassFormSystem[Array, Array, DenseFormat[Array]]


@dataclass(frozen=True)
class CollocationTransientSystem(Generic[Array]):
    """What a transient collocation solve and its sweeps read.

    Built by :func:`collocation_transient_system`; the fields share one
    collocation system and one parameter state.

    Attributes
    ----------
    solved : MassFormSystem
        The system stepped: the raw interior system, with the essential
        rows in differentiated form when a treatment was given.
    unconstrained : MassFormSystem
        The raw interior system, no row treated; not solved.
    collocation : CollocationSystem
        The physics with its essential and closure declarations.
    row_pinning : ClosurePinning
        Pins the essential rows and imposes the closure rows in the blocks
        its placement selects.
    parameters : ParameterState or None
        The shared parameters; none for a system without them.
    """

    solved: MassFormSystem[Array, Array, DenseFormat[Array]]
    unconstrained: MassFormSystem[Array, Array, DenseFormat[Array]]
    collocation: CollocationSystem[Array]
    row_pinning: ClosurePinning[Array]
    parameters: Optional[ParameterState[Array]]

    # The transient-discretization and parameter-setter protocols of
    # :mod:`pyapprox.ode.transient_implicit_map`.

    def system(self) -> MassFormSystem[Array, Array, DenseFormat[Array]]:
        """The solved system."""
        return self.solved

    def pinning(self) -> ClosurePinning[Array]:
        """The boundary rows' pin."""
        return self.row_pinning

    def initial_state(self, state: Array, time: float) -> Array:
        """``state``, shape ``(nstates, 1)``, with its essential entries
        set to their data at ``time``; the closure rows are left as given
        (:meth:`initial_incompatibility` reports how far they are from
        their relation)."""
        essential = self.collocation.essential()
        if essential is None:
            return state
        return essential.rows().replace_rows(state, essential.values(time))

    def initial_incompatibility(self, state: Array, time: float) -> tuple[float, float]:
        r"""How far ``state``, shape ``(nstates, 1)``, is from the boundary
        data at ``time``: the largest :math:`|u_0 - g(t)|` on the essential
        rows, before :meth:`initial_state` sets them, and the largest
        :math:`|c(u_0, t)|` on the closure rows.

        A mismatch is reported, not refused: an impulsively started problem
        is legitimate, and the essential values are set either way.
        """
        bkd = self.collocation.bkd()
        essential = self.collocation.essential()
        mismatch = 0.0
        if essential is not None:
            gap = state - essential.values(time)
            zero = bkd.zeros(gap.shape)
            on_rows = gap - essential.rows().replace_rows(gap, zero)
            mismatch = bkd.to_float(bkd.max(bkd.abs(on_rows)))
        residual = 0.0
        for mask, closure in zip(
            self.collocation.closure_masks(), self.collocation.closures()
        ):
            rows = mask * closure.residual(state[:, 0], time)
            residual = max(residual, bkd.to_float(bkd.max(bkd.abs(rows))))
        return mismatch, residual

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


def collocation_transient_system(
    system: CollocationSystem[Array],
    placement: PinPlacementProtocol,
    treatment: Optional[RowRatesTreatmentProtocol],
    parameters: Optional[ParameterState[Array]] = None,
    solver: Optional[LinearSolverProtocol[Array, DenseFormat[Array]]] = None,
) -> CollocationTransientSystem[Array]:
    r"""The transient system of ``system``.

    Parameters
    ----------
    system : CollocationSystem
        The physics with its essential and closure declarations.
    placement : PinPlacementProtocol
        The blocks in which the boundary rows are pinned:
        :class:`~pyapprox.ode.pinned_rows.AtStepEnd`,
        :class:`~pyapprox.ode.pinned_rows.AtEveryBlock`, or another.
    treatment : RowRatesTreatmentProtocol or None
        How the essential rows carry the rate of their data in every block
        (:class:`~pyapprox.ode.differentiated_rows.CoupledRows`,
        :class:`~pyapprox.ode.differentiated_rows.SymmetricRows`), or None
        to leave them raw. Given, the essential data must supply their
        rate.
    parameters : ParameterState, optional
        The parameters; none for a system without them.
    solver : LinearSolverProtocol, optional
        Solves with the mass and the stage operators; a dense LU by
        default.

    Raises
    ------
    TypeError
        If ``system`` is not a ``CollocationSystem``.
    ValueError
        If a treatment is given for a system without essential rows, the
        essential data supply no rate a treatment needs, a closure cannot
        be pinned, or the essential data depend on the parameters: their
        initial values would then too, which the sweeps do not
        differentiate.
    """
    if not isinstance(system, CollocationSystem):
        raise TypeError(
            f"system must be a CollocationSystem, got {type(system).__name__}"
        )
    essential = system.essential()
    if parameters is not None and essential is not None:
        values = essential.parameter_derivatives().values
        if not isinstance(values, StatedZero):
            raise ValueError(
                "the essential data are not declared independent of the "
                "parameters; the initial state's essential entries would then "
                "depend on them, which this system does not differentiate"
            )
    unconstrained: _Dense[Array] = collocation_unconstrained_system(
        system, parameters, solver
    )
    if treatment is None:
        solved = unconstrained
    elif essential is None:
        raise ValueError(
            "a treatment of the essential rows was given, but the system has "
            "no essential rows"
        )
    else:
        solved = differentiated_rows(unconstrained, essential, treatment)
    return CollocationTransientSystem(
        solved=solved,
        unconstrained=unconstrained,
        collocation=system,
        row_pinning=ClosurePinning(system, placement, parameters),
        parameters=parameters,
    )
