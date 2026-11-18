r"""A collocation steady functional as a function of the parameters.

A configured collocation problem is a :class:`ParameterizedCollocationSystem`:
the physics with its boundary declarations, the parameterization that
writes into it, and the basis that functionals read. The function

.. math::

    q(p) = Q(u(p);\, p), \qquad R(u(p);\, p) = 0,

is then built in three moves, in this order because a state term that
reads the parameters must read the same object the function sets:

1. ``parameters = ParameterState(configured.parameterization)``;
2. the functional, its terms built with ``parameters`` where they read
   them;
3. ``collocation_steady_function(configured.system, parameters,
   functional)``.

The result is a :class:`~pyapprox.implicitfunction.steady_function.
SteadyImplicitFunction` over :func:`~pyapprox.pde.models.collocation.
steady_residual.collocation_steady_residual`.
"""

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from pyapprox.implicitfunction.setups import (
    RecomputeAlways,
    SetupStrategyProtocol,
    SteadyRequest,
)
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import (
    SteadyImplicitFunction,
    SteadyJacobianProtocol,
)
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton, NonlinearSolverProtocol

B = TypeVar("B")


@dataclass(frozen=True)
class ParameterizedCollocationSystem(Generic[Array, B]):
    """A configured collocation problem: what a steady or transient
    function of the parameters is built from.

    Attributes
    ----------
    system : CollocationSystem
        The physics with its essential and closure declarations.
    parameterization : ParameterizationProtocol
        Writes the parameters into ``system``'s physics.
    basis : B
        The collocation basis, for functionals that interpolate or
        integrate the state.
    """

    system: CollocationSystem[Array]
    parameterization: ParameterizationProtocol[Array]
    basis: B


def collocation_steady_function(
    system: CollocationSystem[Array],
    parameters: ParameterState[Array],
    functional: StateFunctional[Array, Array],
    init_state: Optional[Array] = None,
    setups: Optional[
        SetupStrategyProtocol[
            SteadyRequest[Array, DenseFormat[Array]], Array, DenseFormat[Array]
        ]
    ] = None,
    nonlinear: Optional[NonlinearSolverProtocol[Array, Array]] = None,
    time: Optional[float] = None,
    jacobian: Optional[SteadyJacobianProtocol] = None,
) -> SteadyImplicitFunction[Array, DenseFormat[Array]]:
    r"""The functional of ``system``'s steady state as a function of the
    parameters.

    Parameters
    ----------
    system : CollocationSystem
        The physics with its essential and closure declarations.
    parameters : ParameterState
        The parameters; the same object the functional's terms read.
    functional : StateFunctional
        :math:`Q(u;\, p)`.
    init_state : Array, optional
        The initial guess of every solve, shape ``(nstates, 1)``; default
        zero.
    setups : SetupStrategyProtocol, optional
        The linear-solver setups; default dense LU recomputed each
        Newton step.
    nonlinear : NonlinearSolverProtocol, optional
        Default ``Newton`` with absolute tolerance 1e-10, no relative
        tolerance, 50 iterations and a line search.
    time : float, optional
        The time the data are frozen at; omitted, every piece must be
        declared time-invariant (see ``collocation_steady_residual``).
    jacobian : SteadyJacobianProtocol, optional
        How the Jacobian is formed; default by the number of entries and
        parameters (see ``SteadyImplicitFunction``).

    Raises
    ------
    TypeError
        If an argument is of the wrong type.
    ValueError
        If ``init_state`` has the wrong shape, or ``time`` is omitted and
        a piece is declared time-dependent.
    """
    bkd, nstates = system.bkd(), system.nstates()
    if init_state is None:
        init_state = bkd.zeros((nstates, 1))
    if setups is None:
        setups = RecomputeAlways(BackendDenseLU(bkd))
    if nonlinear is None:
        nonlinear = Newton(atol=1e-10, rtol=0.0, maxiters=50, line_search=True)
    return SteadyImplicitFunction(
        collocation_steady_residual(system, parameters, time),
        backend_state_space(bkd, nstates),
        parameters,
        setups,
        nonlinear,
        functional,
        init_state,
        bkd,
        jacobian,
    )
