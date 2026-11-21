"""The stream-function velocity map drives a Galerkin transport model.

Advection-diffusion on the unit square (P1 on 6x6 quadrilaterals,
diffusivity 0.05, unit forcing, zero Dirichlet on the left and right
sides) with the velocity of a three-by-three stream-function expansion
around psi = sin(pi x) sin(pi y) + y, stepped by backward Euler with
dt = 0.025 to T = 0.1 through ``TransientImplicitMap``: the final state is
finite with one value per DOF, and the velocity the physics advects with
is the map's at the sample.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import math
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.implicit_map import ArrayParameters, ImplicitMapFunction
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import backward_euler
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldVelocity,
)
from pyapprox.pde.field_maps.stream_function import StreamFunctionVelocityMap
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.transient_system import galerkin_transient_system
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]


class _CellsPlusUniform(Generic[Array]):
    """psi = sin(pi x) sin(pi y) + y."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def gradient(self, points: Array) -> Array:
        bkd, x, y = self._bkd, points[0], points[1]
        return bkd.stack(
            [
                math.pi * bkd.cos(math.pi * x) * bkd.sin(math.pi * y),
                math.pi * bkd.sin(math.pi * x) * bkd.cos(math.pi * y) + 1.0,
            ],
            axis=0,
        )


def test_velocity_map_drives_transient_solve(numpy_bkd: NumpyBkd) -> None:
    """The map plugs into the ADR parameterization's velocity_map, and the
    velocity it writes is the one the physics advects with."""
    bkd = numpy_bkd
    mesh = StructuredMesh2D(6, 6, [(0.0, 1.0), (0.0, 1.0)], bkd)
    basis = LagrangeBasis(mesh, degree=1)
    vel_basis = VectorLagrangeBasis(mesh, degree=1)
    npsi = 3
    i = np.arange(1, npsi + 1)
    velocity_map = StreamFunctionVelocityMap(
        bkd,
        vel_basis.dof_coordinates()[:, 0::2],
        bkd.asarray(1.5**2 / np.add.outer(i**2, i**2) ** 2.5),
        _CellsPlusUniform(bkd),
        vel_basis.component_layout(),
    )
    velocity = NodalFieldVelocity(vel_basis, np.zeros(vel_basis.ndofs()))
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis, dofs=0.05 * np.ones(basis.ndofs())),
        bkd=bkd,
        velocity=velocity,
        forcing=NodalFieldForcing(basis, dofs=np.ones(basis.ndofs())),
    )
    system = compose_galerkin_system(
        physics,
        [DirichletBC(basis, name, 0.0, bkd) for name in ("left", "right")],
    )
    parameters = ParameterState(
        AdvectionDiffusionParameterization(physics, bkd=bkd, velocity_map=velocity_map)
    )
    transient = galerkin_transient_system(
        skfem_discretization(system), ScipySparseLU(), parameters
    )
    nstates = basis.ndofs()
    final_state = TrajectoryFunctional.from_terms(
        bkd,
        nstates,
        state_terms=(EndpointTerm(StateEntries(bkd, nstates, list(range(nstates)))),),
    )
    implicit: TransientImplicitMap[_Np, _Np, Any] = TransientImplicitMap(
        transient,
        transient,
        RungeKutta(backward_euler(), Newton(atol=1e-10, rtol=0.0, maxiters=10)),
        RecomputeAlways(transient.solved.solver),
        FixedStep(0.025, (0.0, 0.1)),
        bkd.zeros((nstates, 1)),
        FixedTrajectoryFunctional(final_state),
    )
    model = ImplicitMapFunction(implicit, ArrayParameters(bkd))
    eta = bkd.asarray(np.random.default_rng(0).standard_normal(npsi**2))
    values = model(eta[:, None])
    assert tuple(values.shape) == (nstates, 1)
    assert np.all(np.isfinite(bkd.to_numpy(values)))
    # The parameterization wrote the map's velocity into the physics.
    bkd.assert_allclose(bkd.asarray(velocity.dofs()), velocity_map(eta), rtol=1e-12)
