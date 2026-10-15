"""First-order transient paths refuse Galerkin linear elasticity.

Linear elasticity is second order in time (rho u_tt = div sigma + f).
Every Galerkin transient path integrates ``M du/dt = F(u, t)``, which for
elasticity is a viscous relaxation with no inertia or density, so each
path refuses the physics. Steady solves, and direct use of the physics'
own mass matrix, stay available.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolution,
    create_elasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import CompositeLinearElasticity
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    GalerkinModel,
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.models.galerkin.transient import (
    GalerkinTransientForwardModel,
)
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.pde.time_order import SecondOrderInTimeProtocol
from pyapprox.util.backends.numpy import NumpyBkd

_NumpyArray = NDArray[Any]

_REFUSAL = "second order in time"


def _time_config(lumped_mass: bool = False) -> TimeIntegrationConfig[_NumpyArray]:
    return TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.05,
        newton_tol=1e-10,
        newton_maxiter=5,
        lumped_mass=lumped_mass,
        verbosity=0,
    )


def _setup(
    bkd: NumpyBkd,
) -> Tuple[
    CompositeLinearElasticity[_NumpyArray],
    GalerkinSystem[_NumpyArray],
    _NumpyArray,
]:
    """Quadratic manufactured displacement, reproduced exactly by P2.

    Lame parameters lambda = mu = 1 (E = 2.5, nu = 0.25); Dirichlet on
    all four sides. Returns the physics, its system, and the exact
    displacement at the DOFs.
    """
    functions, ndim = create_elasticity_manufactured_test(
        bounds=[0.0, 1.0, 0.0, 1.0],
        sol_strs=["x*(1-x)", "y*(1-y)"],
        lambda_str="1.0",
        mu_str="1.0",
        bkd=bkd,
    )
    mesh = StructuredMesh2D(nx=4, ny=4, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    basis = VectorLagrangeBasis(mesh, degree=2)
    ms = GalerkinManufacturedSolution(basis, functions, bkd)
    physics = CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=2.5,
        poisson_ratio=0.25,
        bkd=bkd,
        body_force=ms.forcing(),
    )
    bcs = [ms.dirichlet(name) for name in ("left", "right", "bottom", "top")]
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    values = np.asarray(ms.solution()(dof_coords, 0.0))  # (ndofs, ndim)
    ndofs = basis.ndofs()
    exact = values[np.arange(ndofs), np.arange(ndofs) % ndim]
    return physics, compose_galerkin_system(physics, bcs), bkd.asarray(exact)


class TestGalerkinLinearElasticityTimeOrder:
    def test_declares_second_order(self, numpy_bkd: NumpyBkd) -> None:
        physics, _, _ = _setup(numpy_bkd)
        assert isinstance(physics, SecondOrderInTimeProtocol)
        message = physics.first_order_transient_refusal()
        for phrase in (
            "CompositeLinearElasticity",
            "rho u_tt = div sigma + f",
            "M du/dt = F(u, t)",
            "no inertia or density",
            "steady",
            "second-order time integration",
        ):
            assert phrase in message

    def test_system_mass_is_refused(self, numpy_bkd: NumpyBkd) -> None:
        _, system, _ = _setup(numpy_bkd)
        with pytest.raises(TypeError, match=_REFUSAL):
            system.mass_matrix()

    @pytest.mark.parametrize("lumped_mass", [False, True])
    def test_hand_built_adapter_is_refused(
        self, numpy_bkd: NumpyBkd, lumped_mass: bool
    ) -> None:
        _, system, _ = _setup(numpy_bkd)
        with pytest.raises(TypeError, match=_REFUSAL):
            GalerkinPhysicsToODEResidualAdapter(system, lumped_mass=lumped_mass)

    @pytest.mark.parametrize("lumped_mass", [False, True])
    def test_model_solve_transient_is_refused(
        self, numpy_bkd: NumpyBkd, lumped_mass: bool
    ) -> None:
        _, system, exact = _setup(numpy_bkd)
        model = GalerkinModel(system, numpy_bkd)
        with pytest.raises(TypeError, match=_REFUSAL):
            model.solve_transient(exact, _time_config(lumped_mass))
        with pytest.raises(TypeError, match=_REFUSAL):
            model.adapter()

    def test_forward_model_is_refused(self, numpy_bkd: NumpyBkd) -> None:
        physics, system, exact = _setup(numpy_bkd)
        parameterization = create_galerkin_lame_parameterization(physics, numpy_bkd)
        with pytest.raises(TypeError, match=_REFUSAL):
            GalerkinTransientForwardModel(
                system, parameterization, exact, _time_config(), numpy_bkd
            )

    def test_steady_solve_still_works(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        _, system, exact = _setup(bkd)
        solution = GalerkinModel(system, bkd).solve_steady(
            bkd.zeros((system.nstates(),))
        )
        bkd.assert_allclose(solution, exact, atol=1e-10)

    def test_physics_mass_still_available(self, numpy_bkd: NumpyBkd) -> None:
        """Reduced models read the physics' mass directly."""
        bkd = numpy_bkd
        physics, _, exact = _setup(bkd)
        mass = physics.mass_matrix()
        nstates = physics.nstates()
        assert mass.shape == (nstates, nstates)
        bkd.assert_allclose(
            physics.mass_solve(bkd.asarray(mass @ exact)), exact, atol=1e-10
        )
