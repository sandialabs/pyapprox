r"""``solve_galerkin_steady`` on the backend's dense stack.

The mock P1 operator of ``tests._helpers.mock_elementwise`` on 8 cells of
[0, 1] (advection speed 0.3, reaction :math:`u^2`), u(0) = 1, u(1) = 2,
its modal parameterization at p = (0.3, -0.2, 0.1, 0.4, -0.1, 0.2), solved
by Newton (atol 1e-12) with dense LU, on both backends.

Checks that do not depend on the residual's derivation: the solution
holds the boundary data and makes the operator's unconstrained rows
vanish; the residual returned is the one solved (zero at the solution,
same parameter version); started at the solution, Newton takes one step.
"""

from typing import Generic

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.pde.boundary import DirichletConstraintSet
from pyapprox.pde.galerkin.system import GalerkinSteadySystem
from pyapprox.pde.galerkin.time_integration.dense_discretization import (
    DenseSteadyDiscretization,
)
from pyapprox.pde.models.galerkin.steady_residual import solve_galerkin_steady
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.mock_elementwise import (
    MockP1Operator,
    ModalCoefficientParameterization,
)
from tests._helpers.rom_problem import FixedValue

_NCELLS = 8
_PARAM = [0.3, -0.2, 0.1, 0.4, -0.1, 0.2]


class _Case(Generic[Array]):
    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        self.operator: MockP1Operator[Array] = MockP1Operator(
            _NCELLS, True, bkd, velocity=0.3
        )
        self.n = _NCELLS + 1
        constraint_set = DirichletConstraintSet(
            [FixedValue(0, 1.0, bkd), FixedValue(self.n - 1, 2.0, bkd)], self.n, bkd
        )
        system = GalerkinSteadySystem(self.operator, constraint_set)
        self.discretization: DenseSteadyDiscretization[Array] = (
            DenseSteadyDiscretization(system)
        )
        self.parameters = ParameterState(
            ModalCoefficientParameterization(self.operator, bkd)
        )
        self.parameters.set_param(bkd.asarray(_PARAM))

    def setups(self) -> RecomputeAlways[Array, DenseFormat[Array]]:
        return RecomputeAlways(BackendDenseLU(self.bkd))

    def newton(self) -> Newton[Array, Array]:
        return Newton(atol=1e-12, rtol=0.0, maxiters=30)


class TestSolveGalerkinSteady:
    def test_solves_the_problem(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        _, point = solve_galerkin_steady(
            case.discretization, case.setups(), case.newton(), case.parameters
        )
        assert point.converged()
        state = point.solution()
        bkd.assert_allclose(state[[0, -1], 0], bkd.asarray([1.0, 2.0]), atol=1e-12)
        raw = case.operator.spatial_residual(state[:, 0], 0.0)
        bkd.assert_allclose(raw[1:-1], bkd.zeros((case.n - 2,)), atol=1e-10)

    def test_returns_the_residual_solved(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        residual, point = solve_galerkin_steady(
            case.discretization, case.setups(), case.newton(), case.parameters
        )
        assert residual.parameters_version() == point.parameters_version()
        bkd.assert_allclose(
            residual(point.solution()), bkd.zeros((case.n, 1)), atol=1e-10
        )

    def test_starts_from_the_initial_guess(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        _, point = solve_galerkin_steady(
            case.discretization, case.setups(), case.newton(), case.parameters
        )
        _, again = solve_galerkin_steady(
            case.discretization,
            case.setups(),
            case.newton(),
            case.parameters,
            initial_guess=point.solution(),
        )
        assert again.converged()
        assert again.info().iterations == 1
        assert point.info().iterations > 1

    def test_refuses_a_guess_of_the_wrong_shape(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        with pytest.raises(ValueError, match="initial_guess"):
            solve_galerkin_steady(
                case.discretization,
                case.setups(),
                case.newton(),
                case.parameters,
                initial_guess=bkd.zeros((case.n,)),
            )
