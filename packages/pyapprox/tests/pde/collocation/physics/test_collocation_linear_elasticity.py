"""First-order transient paths refuse collocation elasticity.

Elasticity is second order in time (rho u_tt = div sigma + f). The
collocation transient paths integrate ``du/dt = f(u, t)``, which for it
is a viscous relaxation with no inertia or density, so
``CollocationModel.solve_transient`` and ``TransientForwardModel`` refuse
both ``LinearElasticityPhysics`` and ``HyperelasticityPhysics``. Steady
solves and the physics' mass matrix stay available.
"""

from typing import Any, List, Union

import numpy as np
import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.mesh import (
    TransformedMesh2D,
    create_uniform_mesh_2d,
)
from pyapprox.pde.collocation.physics import (
    HyperelasticityPhysics,
    LinearElasticityPhysics,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    create_hyperelasticity,
)
from pyapprox.pde.collocation.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.manufactured import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.pde.models.collocation.transient import TransientForwardModel
from pyapprox.pde.parameterizations.collocation_elasticity import (
    CollocationElasticityParameterization,
)
from pyapprox.pde.parameterizations.collocation_hyperelasticity import (
    CollocationHyperelasticityParameterization,
)
from pyapprox.pde.time_order import SecondOrderInTimeProtocol
from pyapprox.util.backends.protocols import Array, Backend

_NPTS_1D = 8
_REFUSAL = "second order in time"
_PHYSICS_NAMES = ["LinearElasticityPhysics", "HyperelasticityPhysics"]

_Physics = Union[LinearElasticityPhysics[Any], HyperelasticityPhysics[Any]]


def _time_config() -> TimeIntegrationConfig[Any]:
    return TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.01,
        newton_tol=1e-10,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )


def _basis(bkd: Backend[Array]) -> ChebyshevBasis2D[Array]:
    return ChebyshevBasis2D(TransformedMesh2D(_NPTS_1D, _NPTS_1D, bkd), bkd)


def _nodes(bkd: Backend[Array]) -> Array:
    """The collocation nodes of ``_basis``. Shape: (2, npts)."""
    basis = _basis(bkd)
    xx, yy = bkd.meshgrid(basis.nodes_x(), basis.nodes_y(), indexing="xy")
    return bkd.stack([bkd.flatten(xx), bkd.flatten(yy)], axis=0)


def _set_zero_dirichlet(physics: _Physics, bkd: Backend[Array]) -> None:
    """Homogeneous Dirichlet on all four sides for both components."""
    mesh = create_uniform_mesh_2d((_NPTS_1D, _NPTS_1D), (-1.0, 1.0, -1.0, 1.0), bkd)
    npts = physics.basis().npts()
    bcs: List[BoundaryConditionProtocol[Array]] = []
    for side in range(4):
        boundary_idx = mesh.boundary_indices(side)
        bcs.append(zero_dirichlet_bc(bkd, boundary_idx))
        bcs.append(zero_dirichlet_bc(bkd, boundary_idx + npts))
    physics.set_boundary_conditions(bcs)


def _manufactured(
    bkd: Backend[Array],
) -> ManufacturedLinearElasticityEquations[Array]:
    """A steady displacement vanishing on the boundary."""
    return ManufacturedLinearElasticityEquations(
        sol_strs=["(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*x"],
        nvars=2,
        lambda_str="1.0",
        mu_str="1.0",
        bkd=bkd,
        oned=True,
    )


def _physics(bkd: Backend[Array], physics_name: str) -> _Physics:
    basis = _basis(bkd)
    physics: _Physics
    if physics_name == "LinearElasticityPhysics":
        values = _manufactured(bkd).functions["forcing"](_nodes(bkd))
        stacked = bkd.concatenate([values[:, 0], values[:, 1]])

        def forcing(time: float) -> Array:
            return stacked

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=forcing
        )
    else:
        physics = create_hyperelasticity(basis, bkd, mu=1.0, lamda=1.5)
    _set_zero_dirichlet(physics, bkd)
    return physics


def _parameterization(
    physics: _Physics, bkd: Backend[Array]
) -> Union[
    CollocationElasticityParameterization[Array],
    CollocationHyperelasticityParameterization[Array],
]:
    """A two-mode Young's-modulus map; never applied, the model refuses."""
    npts = physics.basis().npts()
    youngs_modulus_map = MeshKLEFieldMap(
        bkd,
        bkd.asarray(np.full(npts, 2.5)),
        bkd.asarray(np.ones((npts, 2))),
    )
    if isinstance(physics, LinearElasticityPhysics):
        return CollocationElasticityParameterization(
            physics,
            youngs_modulus_map=youngs_modulus_map,
            poisson_ratio=0.25,
            bkd=bkd,
        )
    return CollocationHyperelasticityParameterization(
        physics,
        youngs_modulus_map=youngs_modulus_map,
        poisson_ratio=0.25,
        bkd=bkd,
    )


class TestCollocationElasticityTimeOrder:
    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_declares_second_order(
        self, bkd: Backend[Array], physics_name: str
    ) -> None:
        physics = _physics(bkd, physics_name)
        assert isinstance(physics, SecondOrderInTimeProtocol)
        message = physics.first_order_transient_refusal()
        assert physics_name in message
        assert _REFUSAL in message

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_solve_transient_is_refused(
        self, bkd: Backend[Array], physics_name: str
    ) -> None:
        physics = _physics(bkd, physics_name)
        model = CollocationModel(physics, bkd)
        with pytest.raises(TypeError, match=_REFUSAL):
            model.solve_transient(bkd.zeros((physics.nstates(),)), _time_config())

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_transient_forward_model_is_refused(
        self, bkd: Backend[Array], physics_name: str
    ) -> None:
        physics = _physics(bkd, physics_name)
        with pytest.raises(TypeError, match=_REFUSAL):
            TransientForwardModel(
                physics,
                bkd,
                bkd.zeros((physics.nstates(),)),
                _time_config(),
                parameterization=_parameterization(physics, bkd),
            )

    def test_steady_solve_still_works(self, bkd: Backend[Array]) -> None:
        physics = _physics(bkd, "LinearElasticityPhysics")
        exact = _manufactured(bkd).functions["solution"](_nodes(bkd))
        solution = CollocationModel(physics, bkd).solve_steady(
            bkd.zeros((physics.nstates(),)), tol=1e-10
        )
        bkd.assert_allclose(
            solution,
            bkd.concatenate([exact[:, 0], exact[:, 1]]),
            atol=1e-8,
        )

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_mass_matrix_still_available(
        self, bkd: Backend[Array], physics_name: str
    ) -> None:
        physics = _physics(bkd, physics_name)
        nstates = physics.nstates()
        bkd.assert_allclose(physics.mass_matrix(), bkd.eye(nstates))
