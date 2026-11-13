r"""The steady residual of a Galerkin system: :math:`R = P F + (I - P)(u - g)`.

The spatial residual :math:`F(u, t^*; p)` at one time :math:`t^*`, with
its essential rows pinned to their data :math:`g(t^*)`:

.. math::

    R(u; p) = P F(u, t^*; p) + (I - P)(u - g(t^*)),

:math:`P` zeroing the essential rows. Its sign is :math:`F`'s, so the
adjoint of a functional of its solution is the one the steady problem
of :math:`F` has always had.

The residual holds no solver: a caller solves it with
:class:`~pyapprox.implicitfunction.steady_system.SteadySystem`, which takes
the setup strategy, and differentiates it with
:class:`~pyapprox.implicitfunction.steady_derivatives.SteadyDerivatives`.
"""

from typing import Optional, TypeVar

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import RowPin
from pyapprox.ode.pinned_rows import values_at_time
from pyapprox.ode.residual_at_time import RightHandSideAtTime
from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.galerkin.protocols.discretization import (
    SteadyDiscretizationProtocol,
)
from pyapprox.pde.galerkin.time_integration.essential_pinned_rows import (
    EssentialPinnedRows,
)
from pyapprox.pde.models.galerkin.unconstrained_system import (
    GalerkinUnconstrainedRightHandSide,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import FormatProtocol

F = TypeVar("F", bound=FormatProtocol)


def galerkin_steady_residual(
    discretization: SteadyDiscretizationProtocol[Array, Array, F],
    parameters: Optional[ParameterState[Array]] = None,
    time: Optional[float] = None,
) -> PinnedResidual[Array, F]:
    r"""The steady residual of ``discretization`` at ``time``.

    Parameters
    ----------
    discretization : SteadyDiscretizationProtocol
        The system on a stack: skfem's SciPy stack, dense backend arrays,
        or any other.
    parameters : ParameterState, optional
        The parameters; none for a system without them. The initial state
        plays no part, so its parameter derivatives are stated zero.
    time : float, optional
        :math:`t^*`, the time the data is frozen at. Omitted, the data
        must be declared time-invariant, and any time gives the same
        residual.

    Raises
    ------
    TypeError
        If ``discretization`` does not satisfy
        ``SteadyDiscretizationProtocol`` or ``parameters`` is not a
        ``ParameterState``.
    ValueError
        If ``time`` is omitted and the spatial operator or the essential
        constraints are declared time-dependent: the steady state then
        depends on which time is meant. Also if the parameterization
        writes objects the system does not hold.
    """
    if not isinstance(discretization, SteadyDiscretizationProtocol):
        raise TypeError(
            "discretization must satisfy SteadyDiscretizationProtocol, got "
            f"{type(discretization).__name__}"
        )
    system = discretization.system()
    if time is None:
        if not system.spatial_operator().is_time_invariant():
            raise ValueError(
                f"{type(system.spatial_operator()).__name__} has data "
                "declared time-dependent, so its steady state depends on "
                "the time; pass the time"
            )
        if not system.constraint_set().is_time_invariant():
            raise ValueError(
                "the essential constraints are time-dependent, so the "
                "steady state depends on the time; pass the time"
            )
        # Time-invariant data reads the same at every time.
        time = 0.0
    rhs: GalerkinUnconstrainedRightHandSide[Array, F] = (
        GalerkinUnconstrainedRightHandSide(
            discretization, parameters, initial_state=STATED_ZERO
        )
    )
    return PinnedResidual(
        RightHandSideAtTime(rhs, time, BackendColumnForm(system.bkd())),
        RowPin(values_at_time(EssentialPinnedRows(discretization), time)),
        discretization.space(),
    )
