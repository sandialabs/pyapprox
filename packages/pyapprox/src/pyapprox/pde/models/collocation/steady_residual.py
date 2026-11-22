r"""The steady residual of a collocation system.

The closed right-hand side :math:`\tilde f(u, t^*; p)` at one time
:math:`t^*`, with the essential rows pinned to their data :math:`g(t^*)`:

.. math::

    R(u; p) = P \tilde f(u, t^*; p) + (I - P)(u - g(t^*)),

:math:`P` zeroing the essential rows. :math:`\tilde f` already carries the
closure rows (Robin, flux, traction, periodic), so the pin only replaces
the essential ones.

The residual holds no solver: a caller solves it with
:class:`~pyapprox.implicitfunction.steady_system.SteadySystem`, which takes
the setup strategy, and differentiates it with
:class:`~pyapprox.implicitfunction.steady_derivatives.SteadyDerivatives`.
"""

from typing import Optional

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import RowPin
from pyapprox.ode.pinned_rows import PinnedRowsProtocol, values_at_time
from pyapprox.ode.residual_at_time import RightHandSideAtTime
from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_state_space
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.models.collocation.closed_system import (
    ClosedRightHandSide,
    CollocationSystem,
    NoEssentialRows,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import DenseFormat


def _refuse_time_dependence(system: CollocationSystem[Array]) -> None:
    """Refuse an omitted time when any piece is declared time-dependent,
    naming the piece."""
    physics = system.physics()
    if not physics.interior_is_time_invariant():
        raise ValueError(
            f"{type(physics).__name__} has data declared time-dependent, so "
            "its steady state depends on the time; pass the time"
        )
    essential = system.essential()
    if essential is not None and not essential.is_time_invariant():
        raise ValueError(
            "the essential data are time-dependent, so the steady state "
            "depends on the time; pass the time"
        )
    for k, closure in enumerate(system.closures()):
        if not closure.is_time_invariant():
            raise ValueError(
                f"closure {k} ({type(closure).__name__}) has data declared "
                "time-dependent, so the steady state depends on the time; "
                "pass the time"
            )


def collocation_steady_residual(
    system: CollocationSystem[Array],
    parameters: Optional[ParameterState[Array]] = None,
    time: Optional[float] = None,
) -> PinnedResidual[Array, DenseFormat[Array]]:
    r"""The steady residual of ``system`` at ``time``.

    Parameters
    ----------
    system : CollocationSystem
        The physics with its essential and closure declarations.
    parameters : ParameterState, optional
        The parameters; none for a system without them. The initial state
        plays no part, so its parameter derivatives are stated zero.
    time : float, optional
        :math:`t^*`, the time the data are frozen at. Omitted, the interior
        operator and every declaration must be declared time-invariant,
        and any time gives the same residual.

    Raises
    ------
    TypeError
        If ``system`` is not a ``CollocationSystem`` or ``parameters`` is
        not a ``ParameterState``.
    ValueError
        If ``time`` is omitted and a piece is declared time-dependent (the
        message names it); if the parameterization writes objects other
        than the physics; or if boundary data hold parameters of their own.
    """
    if not isinstance(system, CollocationSystem):
        raise TypeError(
            f"system must be a CollocationSystem, got {type(system).__name__}"
        )
    if time is None:
        _refuse_time_dependence(system)
        # Time-invariant data read the same at every time.
        time = 0.0
    bkd, nstates = system.bkd(), system.nstates()
    rhs = ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO)
    declared = system.essential()
    essential: PinnedRowsProtocol[Array] = (
        NoEssentialRows(bkd, nstates) if declared is None else declared
    )
    return PinnedResidual(
        RightHandSideAtTime(rhs, time, BackendColumnForm(bkd)),
        RowPin(values_at_time(essential, time)),
        backend_state_space(bkd, nstates),
    )
