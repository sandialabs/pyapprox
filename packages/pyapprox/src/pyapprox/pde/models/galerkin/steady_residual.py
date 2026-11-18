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
the setup strategy (:func:`solve_galerkin_steady` does both steps), and
differentiates it with
:class:`~pyapprox.implicitfunction.steady_derivatives.SteadyDerivatives`.
"""

from typing import Optional, Tuple, TypeVar

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import RowPin
from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
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
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

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


def solve_galerkin_steady(
    discretization: SteadyDiscretizationProtocol[Array, Array, F],
    setups: SetupStrategyProtocol[SteadyRequest[Array, F], Array, F],
    nonlinear: NonlinearSolverProtocol[Array, Array],
    parameters: Optional[ParameterState[Array]] = None,
    time: Optional[float] = None,
    initial_guess: Optional[Array] = None,
) -> Tuple[PinnedResidual[Array, F], SteadyPoint[Array, Array]]:
    r"""Solve the steady residual of ``discretization``.

    :func:`galerkin_steady_residual` solved by
    :class:`~pyapprox.implicitfunction.steady_system.SteadySystem`: the
    residual at the parameters' current values and ``time``, from
    ``initial_guess``.

    Parameters
    ----------
    discretization : SteadyDiscretizationProtocol
        The system on a stack; its state space must have an inner product.
    setups : SetupStrategyProtocol
        Sets up the Jacobian solves; its linear solver pairs with the
        discretization's Jacobian format.
    nonlinear : NonlinearSolverProtocol
        The nonlinear solver, with its tolerances.
    parameters : ParameterState, optional
        The parameters; none for a system without them.
    time : float, optional
        As for :func:`galerkin_steady_residual`.
    initial_guess : Array, optional
        Shape ``(nstates, 1)``; zero when omitted.

    Returns
    -------
    tuple
        The residual solved, so that a functional of the solution (an
        essential-boundary flux) reads the same parameters and time, and
        the :class:`~pyapprox.implicitfunction.steady_system.SteadyPoint`.
        A failed solve does not raise: check ``converged()``.

    Raises
    ------
    TypeError
        As for :func:`galerkin_steady_residual` and ``SteadySystem``.
    ValueError
        If ``initial_guess`` is not of shape ``(nstates, 1)``, and as for
        :func:`galerkin_steady_residual`.
    """
    residual = galerkin_steady_residual(discretization, parameters, time)
    space = discretization.space()
    nstates = space.descriptor().size
    if initial_guess is None:
        initial_guess = space.zeros()
    elif tuple(initial_guess.shape) != (nstates, 1):
        raise ValueError(
            f"initial_guess must have shape ({nstates}, 1), got "
            f"{tuple(initial_guess.shape)}"
        )
    steady: SteadySystem[Array, Array, F] = SteadySystem(residual, space, setups)
    return residual, steady.solve(nonlinear, initial_guess)
