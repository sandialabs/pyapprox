"""The stream-function velocity map drives a Galerkin transport model."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import math
from typing import Any

import numpy as np
from pyapprox.ode.config import TimeIntegrationConfig
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
from pyapprox.pde.models.galerkin.transient import (
    GalerkinTransientForwardModel,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd


class _CellsPlusUniform:
    """psi = sin(pi x) sin(pi y) + y."""

    def __init__(self, bkd: Any) -> None:
        self._bkd = bkd

    def gradient(self, points: Any) -> Any:
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
    mesh = StructuredMesh2D(6, 6, [[0.0, 1.0], [0.0, 1.0]], bkd)
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
    parameterization = AdvectionDiffusionParameterization(
        physics, bkd=bkd, velocity_map=velocity_map
    )
    config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.025,
        newton_tol=1e-10,
        newton_maxiter=10,
        lumped_mass=False,
        verbosity=0,
    )
    model = GalerkinTransientForwardModel(
        system, parameterization, bkd.zeros((basis.ndofs(),)), config, bkd
    )
    eta = bkd.asarray(np.random.default_rng(0).standard_normal(npsi**2))
    final_state = model(eta[:, None])
    assert tuple(final_state.shape) == (basis.ndofs(), 1)
    assert np.all(np.isfinite(bkd.to_numpy(final_state)))
    # The parameterization wrote the map's velocity into the physics.
    bkd.assert_allclose(
        bkd.asarray(velocity.dofs()), velocity_map(eta), rtol=1e-12
    )
