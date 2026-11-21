"""First-order transient paths refuse Galerkin linear elasticity.

Linear elasticity is second order in time (rho u_tt = div sigma + f).
Every Galerkin transient path integrates ``M du/dt = F(u, t)``, which for
elasticity is a viscous relaxation with no inertia or density, so each
path refuses the physics: the system's mass, the hand-built ODE adapter
and ``galerkin_transient_system``. Steady solves, and direct use of the
physics' own mass matrix, stay available. The problem is a
:mod:`pyapprox.pde.verification` manufactured problem.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple

from numpy.typing import NDArray

from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import CompositeLinearElasticity
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
    linear_stress,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from tests._helpers.galerkin_steady_solve import steady_solve

_NumpyArray = NDArray[Any]

_REFUSAL = "second order in time"
_SQUARE = Box([(0.0, 1.0)] * 2)


def _setup(
    bkd: NumpyBkd,
) -> Tuple[
    CompositeLinearElasticity[_NumpyArray],
    GalerkinSystem[_NumpyArray],
    _NumpyArray,
]:
    r"""A quadratic displacement :math:`(x(1-x),\, y(1-y))` with
    :math:`-\nabla\cdot\sigma = f`, :math:`\lambda = \mu = 1` (E = 2.5,
    nu = 0.25 through ``from_uniform``), Dirichlet on all four sides; P2
    on four quadrilaterals per side reproduces it.

    Returns the physics, its system, and the exact displacement at the
    DOFs.
    """
    problem = ManufacturedProblem(
        {"u": "x*(1 - x)", "v": "y*(1 - y)"},
        equilibrium_terms(linear_stress(2, 1, 1)),
        _SQUARE,
        {side: Dirichlet() for side in _SQUARE.boundaries()},
    )
    (mesh,) = StructuredBox([4]).meshes(_SQUARE)
    basis = VectorLagrange(2).build(mesh)
    rows = problem.fields()
    physics = CompositeLinearElasticity.from_uniform(
        basis, 2.5, 0.25, bkd, body_force=vector_forcing(problem, rows)
    )
    system = compose_galerkin_system(
        physics, vector_boundary_conditions(problem, basis, rows)
    )
    return physics, system, bkd.asarray(exact_dof_values(problem, basis, rows))


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
            skfem_ode_residual(system, lumped_mass=lumped_mass)

    def test_transient_system_is_refused(self, numpy_bkd: NumpyBkd) -> None:
        physics, system, _ = _setup(numpy_bkd)
        parameters = ParameterState(
            create_galerkin_lame_parameterization(physics, numpy_bkd)
        )
        with pytest.raises(TypeError, match=_REFUSAL):
            galerkin_transient_system(
                skfem_discretization(system), ScipySparseLU(), parameters
            )

    def test_steady_solve_still_works(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        _, system, exact = _setup(bkd)
        result = steady_solve(system)
        assert result.converged, result.residual_norm
        bkd.assert_allclose(result.solution, exact, atol=1e-10)

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
