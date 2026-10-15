"""First-order transient paths refuse Galerkin hyperelasticity.

Hyperelasticity is second order in time (rho u_tt = Div P + f). Every
Galerkin transient path integrates ``M du/dt = F(u, t)``, which for it
is a viscous relaxation with no inertia or density, so each path refuses
both ``HyperelasticityPhysics`` and ``CompositeHyperelasticityPhysics``.
Steady solves, and direct use of the physics' own mass matrix, stay
available.

NumPy only: skfem assembly at each nonlinear step is numpy-based.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolution,
    create_hyperelasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import (
    CompositeHyperelasticityPhysics,
    HyperelasticityPhysics,
)
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
_Physics = Union[
    HyperelasticityPhysics[_NumpyArray],
    CompositeHyperelasticityPhysics[_NumpyArray],
]

_REFUSAL = "second order in time"
_PHYSICS_NAMES = ["HyperelasticityPhysics", "CompositeHyperelasticityPhysics"]


def _time_config() -> TimeIntegrationConfig[_NumpyArray]:
    return TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.05,
        newton_tol=1e-10,
        newton_maxiter=5,
        lumped_mass=False,
        verbosity=0,
    )


def _setup(
    bkd: NumpyBkd, physics_name: str
) -> Tuple[_Physics, GalerkinSystem[_NumpyArray], _NumpyArray]:
    """1D Neo-Hookean bar with a manufactured steady displacement.

    Lame parameters lambda = mu = 1 (E = 2.5, nu = 0.25); Dirichlet at
    both ends. Returns the physics, its system, and the exact
    displacement at the DOFs.
    """
    stress: NeoHookeanStress[_NumpyArray] = NeoHookeanStress(1.0, 1.0)
    functions, _ = create_hyperelasticity_manufactured_test(
        bounds=[0.0, 1.0],
        sol_strs=["0.1*x**2*(1-x)**2"],
        stress_model=stress,
        bkd=bkd,
    )
    mesh = StructuredMesh1D(nx=30, bounds=(0.0, 1.0), bkd=bkd)
    basis = VectorLagrangeBasis(mesh, degree=2)
    ms = GalerkinManufacturedSolution(basis, functions, bkd)
    physics: _Physics
    if physics_name == "HyperelasticityPhysics":
        physics = HyperelasticityPhysics(
            basis=basis, stress_model=stress, bkd=bkd, body_force=ms.forcing()
        )
    else:
        physics = CompositeHyperelasticityPhysics.from_uniform(
            basis, 2.5, 0.25, bkd, body_force=ms.forcing()
        )
    bcs = [ms.dirichlet(name) for name in ("left", "right")]
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    exact = np.asarray(ms.solution()(dof_coords, 0.0))[:, 0]
    return physics, compose_galerkin_system(physics, bcs), bkd.asarray(exact)


class TestGalerkinHyperelasticityTimeOrder:
    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_declares_second_order(
        self, numpy_bkd: NumpyBkd, physics_name: str
    ) -> None:
        physics, _, _ = _setup(numpy_bkd, physics_name)
        assert isinstance(physics, SecondOrderInTimeProtocol)
        message = physics.first_order_transient_refusal()
        assert physics_name in message
        assert _REFUSAL in message

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_hand_built_adapter_is_refused(
        self, numpy_bkd: NumpyBkd, physics_name: str
    ) -> None:
        _, system, _ = _setup(numpy_bkd, physics_name)
        with pytest.raises(TypeError, match=physics_name):
            GalerkinPhysicsToODEResidualAdapter(system)

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_model_solve_transient_is_refused(
        self, numpy_bkd: NumpyBkd, physics_name: str
    ) -> None:
        _, system, exact = _setup(numpy_bkd, physics_name)
        model = GalerkinModel(system, numpy_bkd)
        with pytest.raises(TypeError, match=_REFUSAL):
            model.solve_transient(exact, _time_config())

    def test_forward_model_is_refused(self, numpy_bkd: NumpyBkd) -> None:
        """Only the composite physics has a Galerkin parameterization."""
        physics, system, exact = _setup(numpy_bkd, "CompositeHyperelasticityPhysics")
        assert isinstance(physics, CompositeHyperelasticityPhysics)
        parameterization = create_galerkin_lame_parameterization(physics, numpy_bkd)
        with pytest.raises(TypeError, match=_REFUSAL):
            GalerkinTransientForwardModel(
                system, parameterization, exact, _time_config(), numpy_bkd
            )

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_steady_solve_still_works(
        self, numpy_bkd: NumpyBkd, physics_name: str
    ) -> None:
        bkd = numpy_bkd
        _, system, exact = _setup(bkd, physics_name)
        solution = GalerkinModel(system, bkd).solve_steady(
            bkd.zeros((system.nstates(),))
        )
        bkd.assert_allclose(solution, exact, atol=1e-6)

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_physics_mass_still_available(
        self, numpy_bkd: NumpyBkd, physics_name: str
    ) -> None:
        """Reduced models read the physics' mass directly."""
        bkd = numpy_bkd
        physics, _, exact = _setup(bkd, physics_name)
        mass = physics.mass_matrix()
        bkd.assert_allclose(
            physics.mass_solve(bkd.asarray(mass @ exact)), exact, atol=1e-10
        )
