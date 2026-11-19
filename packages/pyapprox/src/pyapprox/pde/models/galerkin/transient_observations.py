r"""Quantities of a transient Galerkin solution, as terms of trajectory
functionals.

Point probes and nodal advective outflux at chosen stored states
(:func:`probe_observations`, :func:`nodal_outflux_observations`), and the
time integral of a weighted squared norm of the solution
(:func:`time_integrated_weighted_l2`). Each is built from the native
terms of :mod:`pyapprox.ode.trajectory_terms` and
:mod:`pyapprox.ode.time_integral`, so the value sink and every sweep read
it, on any store. The essential-boundary flux is in
:mod:`pyapprox.pde.models.galerkin.essential_boundary_flux`.
"""

from typing import List, Sequence, TypeVar

from pyapprox.implicitfunction.observation_terms import (
    ObservedValue,
    WeightedEntries,
)
from pyapprox.ode.integrands import WeightedSquaredNorm
from pyapprox.ode.setups import MassShiftStrategy
from pyapprox.ode.stepping_methods import SteppingMethodProtocol
from pyapprox.ode.time_integral import TimeIntegral
from pyapprox.ode.trajectory_terms import (
    AtStoredStates,
    StoredStates,
    TrajectoryFunctional,
)
from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.pde.models.galerkin.transient_system import GalerkinTransientSystem
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import FormatProtocol
from pyapprox.util.linalg.symmetric_weight import SymmetricWeightProtocol

F = TypeVar("F", bound=FormatProtocol)


def probe_observations(
    basis: LagrangeBasis[Array],
    points: Array,
    stored_states: Sequence[int],
) -> TrajectoryFunctional[Array, Array]:
    r"""The solution at points and stored states.

    Entry ``j * npts + s`` is :math:`u(x_s, t_{m_j})` for
    :math:`m_j` = ``stored_states[j]``, evaluated exactly in the element
    containing :math:`x_s`.

    Parameters
    ----------
    basis : LagrangeBasis
        Scalar basis of the solution.
    points : Array
        Sensor locations. Shape: (ndim, npts).
    stored_states : sequence of int
        Indices of the stored states observed, non-negative (0 the
        initial state).

    Raises
    ------
    ValueError
        If ``stored_states`` is empty or an index is negative.
    """
    indices, weights = basis.probe_rows(points)
    observation = WeightedEntries(basis.bkd(), basis.ndofs(), indices, weights)
    return _at_stored_states(basis.bkd(), observation, stored_states)


def nodal_outflux_observations(
    bkd: Backend[Array],
    nstates: int,
    boundary_dofs: Array,
    normal_velocity: Array,
    stored_states: Sequence[int],
) -> TrajectoryFunctional[Array, Array]:
    r"""The pointwise advective outflux :math:`u\, \beta\cdot n` at
    boundary DOFs and stored states.

    Entry ``j * nbdofs + s`` is :math:`u\, \beta\cdot n` at
    ``boundary_dofs[s]`` and stored state ``stored_states[j]``, in the
    order of ``boundary_dofs`` (sort them by coordinate beforehand for a
    profile along the boundary).

    The normal velocity is fixed data. If the velocity is a model
    parameter, the outflux's direct dependence on it is missed; use this
    only where :math:`\beta\cdot n` does not vary with the parameters (for
    a stream-function velocity whose perturbation vanishes on the
    boundary, for example).

    Parameters
    ----------
    bkd : Backend
        The backend.
    nstates : int
        The number of states.
    boundary_dofs : Array
        Boundary DOF indices. Integer, shape: (nbdofs,).
    normal_velocity : Array
        :math:`\beta\cdot n` at those DOFs. Shape: (nbdofs,).
    stored_states : sequence of int
        Indices of the stored states observed, non-negative.

    Raises
    ------
    ValueError
        If ``boundary_dofs`` and ``normal_velocity`` are not both of shape
        ``(nbdofs,)``, a DOF is out of range, ``stored_states`` is empty or
        an index is negative.
    """
    if boundary_dofs.ndim != 1 or tuple(boundary_dofs.shape) != tuple(
        normal_velocity.shape
    ):
        raise ValueError(
            "boundary_dofs and normal_velocity must both have shape (nbdofs,), "
            f"got {tuple(boundary_dofs.shape)} and {tuple(normal_velocity.shape)}"
        )
    observation = WeightedEntries(
        bkd,
        nstates,
        bkd.reshape(boundary_dofs, (-1, 1)),
        bkd.reshape(normal_velocity, (-1, 1)),
    )
    return _at_stored_states(bkd, observation, stored_states)


def time_integrated_weighted_l2(
    system: GalerkinTransientSystem[Array, F],
    method: SteppingMethodProtocol[Array, Array],
    setups: MassShiftStrategy[Array, Array, F],
    weight: SymmetricWeightProtocol[Array],
) -> TimeIntegral[Array, Array]:
    r""":math:`\int_{t_0}^{t_N} y^\top W y\, dt`, integrated by the method
    the trajectory is solved with.

    The integral is the method applied to :math:`\dot z = y^\top W y`
    alongside the solve, so its nodes are the method's own (a stage of the
    step where the method has one), not a rule chosen apart from it. Use
    its :meth:`~pyapprox.ode.time_integral.TimeIntegral.sink` during the
    solve for the value, and its :meth:`~pyapprox.ode.time_integral.
    TimeIntegral.trajectory_functional` on the stored trajectory for the
    sweeps.

    Parameters
    ----------
    system : GalerkinTransientSystem
        The system the trajectory is solved on.
    method : SteppingMethodProtocol
        The method it is solved with.
    setups : MassShiftStrategy
        Sets up the linear solves of the stepper built here to read the
        method's quadrature and spaces; it solves nothing.
    weight : SymmetricWeightProtocol
        :math:`W`, e.g. a subdomain-weighted mass matrix.

    Raises
    ------
    TypeError
        If ``method`` or ``weight`` does not satisfy its protocol.
    ValueError
        If ``weight`` does not act on the system's states.
    """
    if not isinstance(method, SteppingMethodProtocol):
        raise TypeError(
            f"method must satisfy SteppingMethodProtocol, got {type(method).__name__}"
        )
    integrand = WeightedSquaredNorm(weight, system.constraints.bkd())
    nstates = system.solved.space.descriptor().size
    if weight.nstates() != nstates:
        raise ValueError(
            f"weight acts on {weight.nstates()} states, the system has {nstates}"
        )
    stepper = method.stepper(system.solved, setups, system.row_pinning)
    return TimeIntegral(integrand, stepper, system.constraints.bkd())


def _at_stored_states(
    bkd: Backend[Array],
    observation: WeightedEntries[Array],
    stored_states: Sequence[int],
) -> TrajectoryFunctional[Array, Array]:
    """``observation`` at each of ``stored_states``, stacked in that order.

    Raises
    ------
    ValueError
        If ``stored_states`` is empty or an index is negative.
    """
    if not stored_states:
        raise ValueError("stored_states must name at least one stored state")
    nobs = observation.nobs()
    per_state: List[TrajectoryFunctional[Array, Array]] = [
        TrajectoryFunctional.from_terms(
            bkd,
            nobs,
            state_terms=(
                AtStoredStates(ObservedValue(observation), StoredStates([index])),
            ),
        )
        for index in stored_states
    ]
    return TrajectoryFunctional.stack(bkd, per_state)
