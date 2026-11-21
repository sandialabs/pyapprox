"""First-order transient paths refuse Galerkin hyperelasticity.

Hyperelasticity is second order in time (rho u_tt = Div P + f). Every
Galerkin transient path integrates ``M du/dt = F(u, t)``, which for it
is a viscous relaxation with no inertia or density, so each path refuses
both ``HyperelasticityPhysics`` and ``CompositeHyperelasticityPhysics``
(the hand-built ODE adapter, and ``galerkin_transient_system`` for the
composite class, the one with a parameterization). Steady solves, and
direct use of the physics' own mass matrix, stay available. The problem
is a :mod:`pyapprox.pde.verification` manufactured problem.

NumPy only: skfem assembly at each nonlinear step is numpy-based.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple, Union

from numpy.typing import NDArray

from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import (
    CompositeHyperelasticityPhysics,
    HyperelasticityPhysics,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
    skfem_ode_residual,
)
from pyapprox.pde.models.galerkin.transient_system import galerkin_transient_system
from pyapprox.pde.models.galerkin.verification import (
    StructuredBox,
    VectorLagrange,
    exact_dof_values,
    vector_boundary_conditions,
    vector_forcing,
)
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.time_order import SecondOrderInTimeProtocol
from pyapprox.pde.verification.conditions import Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.solid_mechanics import (
    equilibrium_terms,
    neo_hookean_stress,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from tests._helpers.galerkin_steady_solve import steady_solve

_NumpyArray = NDArray[Any]
_Physics = Union[
    HyperelasticityPhysics[_NumpyArray],
    CompositeHyperelasticityPhysics[_NumpyArray],
]

_REFUSAL = "second order in time"
_PHYSICS_NAMES = ["HyperelasticityPhysics", "CompositeHyperelasticityPhysics"]
_LINE = Box([(0.0, 1.0)])


def _setup(
    bkd: NumpyBkd, physics_name: str
) -> Tuple[_Physics, GalerkinSystem[_NumpyArray], _NumpyArray]:
    r"""A neo-Hookean bar, :math:`-\nabla\cdot P = f` with :math:`u =
    x^2 (1-x)^2 / 10`, :math:`\lambda = \mu = 1` (E = 2.5, nu = 0.25
    through ``from_uniform`` for the composite class), Dirichlet at both
    ends; P2 on 30 cells.

    Returns the physics, its system, and the exact displacement at the
    DOFs.
    """
    problem = ManufacturedProblem(
        {"u": "0.1*x**2*(1 - x)**2"},
        equilibrium_terms(neo_hookean_stress(1, 1, 1)),
        _LINE,
        {side: Dirichlet() for side in _LINE.boundaries()},
    )
    (mesh,) = StructuredBox([30]).meshes(_LINE)
    basis = VectorLagrange(2).build(mesh)
    rows = problem.fields()
    force = vector_forcing(problem, rows)
    physics: _Physics
    if physics_name == "HyperelasticityPhysics":
        physics = HyperelasticityPhysics(
            basis, NeoHookeanStress(1.0, 1.0), bkd, body_force=force
        )
    else:
        physics = CompositeHyperelasticityPhysics.from_uniform(
            basis, 2.5, 0.25, bkd, body_force=force
        )
    system = compose_galerkin_system(
        physics, vector_boundary_conditions(problem, basis, rows)
    )
    return physics, system, bkd.asarray(exact_dof_values(problem, basis, rows))


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
            skfem_ode_residual(system)

    def test_transient_system_is_refused(self, numpy_bkd: NumpyBkd) -> None:
        """Only the composite physics has a Galerkin parameterization."""
        physics, system, _ = _setup(numpy_bkd, "CompositeHyperelasticityPhysics")
        assert isinstance(physics, CompositeHyperelasticityPhysics)
        parameters = ParameterState(
            create_galerkin_lame_parameterization(physics, numpy_bkd)
        )
        with pytest.raises(TypeError, match=_REFUSAL):
            galerkin_transient_system(
                skfem_discretization(system), ScipySparseLU(), parameters
            )

    @pytest.mark.parametrize("physics_name", _PHYSICS_NAMES)
    def test_steady_solve_still_works(
        self, numpy_bkd: NumpyBkd, physics_name: str
    ) -> None:
        bkd = numpy_bkd
        _, system, exact = _setup(bkd, physics_name)
        result = steady_solve(system)
        assert result.converged, result.residual_norm
        bkd.assert_allclose(result.solution, exact, atol=1e-6)

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
