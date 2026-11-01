"""Shared helpers for the dynamical-systems tests.

- :func:`step_times`: a fixed-step grid, the last step shortened to end
  at the final time.
- :func:`solve_batched`: a batched learned system's trajectory under a
  stepping method, as one array.
"""

import math
from typing import List, Tuple

from pyapprox.ode.integration import Integrator
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.stepping_methods import SteppingMethodProtocol
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore, trajectory_array
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.util.backends.protocols import Array


def step_times(init_time: float, final_time: float, deltat: float) -> List[float]:
    """Times :math:`t_0 + n \\Delta t` up to ``final_time``, the last step
    shortened to end there when the interval is not a whole number of
    steps (to a relative 1e-12)."""
    nfull = math.floor((final_time - init_time) / deltat * (1 + 1e-12))
    times = [init_time + deltat * n for n in range(nfull + 1)]
    if final_time - times[-1] > 1e-12 * deltat:
        times.append(final_time)
    return times


def solve_batched(
    wrapper: BatchedBoundODEResidual[Array],
    method: SteppingMethodProtocol[Array, Array],
    init_state: Array,
    times: List[float],
) -> Tuple[Array, Array]:
    """The trajectory at the wrapper's current coefficients, shape
    (nstates, ntimes), and its times, shape (ntimes,).

    Parameters
    ----------
    wrapper : BatchedBoundODEResidual
        The learned system, its trajectories batched.
    method : SteppingMethodProtocol
        Bound to the wrapper's system.
    init_state : Array
        The batched initial state, flat or a single column.
    times : list of float
        The step boundaries; they must determine their sizes exactly (see
        :meth:`~pyapprox.ode.step_plan.GivenSizes.from_times`), as the
        trajectory-matching loss requires of them too.
    """
    bkd = wrapper.bkd()
    stepper = method.stepper(wrapper.mass_form_system())
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    steps = GivenSizes.from_times(times)
    Integrator(stepper).solve(bkd.reshape(init_state, (-1, 1)), steps, store)
    return trajectory_array(store, bkd), bkd.asarray(store.times())
