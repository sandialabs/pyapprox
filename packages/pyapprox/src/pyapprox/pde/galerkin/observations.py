"""Observation functionals of Galerkin solutions at selected time steps.

Both build a ``TimeIndexedObservationFunctional``: point probes of the
finite element solution, and the pointwise advective outflux through
boundary DOFs.
"""

from typing import Sequence

from pyapprox.ode.functionals.time_indexed_observation import (
    TimeIndexedObservationFunctional,
)
from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.util.backends.protocols import Array, Backend


def probe_observation_functional(
    basis: LagrangeBasis[Array],
    points: Array,
    time_indices: Sequence[int],
    nparams: int,
) -> TimeIndexedObservationFunctional[Array]:
    """Observe the solution at points and time steps.

    QoI ``j * npts + s`` is :math:`u(x_s, t_{n_j})`, evaluated exactly
    in the element containing :math:`x_s` (barycentric for P1).

    Parameters
    ----------
    basis : LagrangeBasis
        Scalar basis of the solution.
    points : Array
        Sensor locations. Shape: (ndim, npts)
    time_indices : Sequence[int]
        Trajectory columns observed.
    nparams : int
        Total number of model parameters.

    Returns
    -------
    TimeIndexedObservationFunctional
        The observations, sensor index fastest.
    """
    indices, weights = basis.probe_rows(points)
    return TimeIndexedObservationFunctional(
        indices, weights, time_indices, basis.ndofs(), nparams, basis.bkd()
    )


def nodal_outflux_functional(
    boundary_dofs: Array,
    normal_velocity: Array,
    time_indices: Sequence[int],
    nstates: int,
    nparams: int,
    bkd: Backend[Array],
) -> TimeIndexedObservationFunctional[Array]:
    """Observe the pointwise advective outflux :math:`u \\, \\beta\\cdot n`
    at boundary DOFs and time steps.

    QoI ``j * nbdofs + s`` is :math:`u \\, \\beta\\cdot n` at
    ``boundary_dofs[s]`` and step ``time_indices[j]``, in the order of
    ``boundary_dofs`` (sort them by coordinate beforehand for a profile
    along the boundary).

    The normal velocity is fixed data. If the velocity is a model
    parameter, this functional misses the outflux's direct dependence on
    it; use it only where :math:`\\beta\\cdot n` does not vary with the
    parameters (for a stream-function velocity whose perturbation
    vanishes on the boundary, for example).

    Parameters
    ----------
    boundary_dofs : Array
        Boundary DOF indices. Integer, shape: (nbdofs,)
    normal_velocity : Array
        :math:`\\beta\\cdot n` at those DOFs. Shape: (nbdofs,)
    time_indices : Sequence[int]
        Trajectory columns observed.
    nstates : int
        Total number of state variables.
    nparams : int
        Total number of model parameters.
    bkd : Backend
        Backend for array operations.

    Returns
    -------
    TimeIndexedObservationFunctional
        The outflux, boundary DOF index fastest.
    """
    if boundary_dofs.shape != normal_velocity.shape or boundary_dofs.ndim != 1:
        raise ValueError(
            "boundary_dofs and normal_velocity must both have shape "
            f"(nbdofs,), got {boundary_dofs.shape} and "
            f"{normal_velocity.shape}"
        )
    return TimeIndexedObservationFunctional(
        bkd.reshape(boundary_dofs, (-1, 1)),
        bkd.reshape(normal_velocity, (-1, 1)),
        time_indices,
        nstates,
        nparams,
        bkd,
    )
