r"""A steady system with a dense Jacobian, on the backend's dense stack.

The mock P1 operator of ``tests._helpers.mock_elementwise`` on 8 cells of
[0, 1] (advection speed 0.3, with and without the reaction :math:`u^2`),
u(0) = 1, u(1) = 2, its modal parameterization at p = (0.3, -0.2, 0.1,
0.4, -0.1, 0.2).

- ``DenseSteadyDiscretization`` gives the system's spaces, its two
  essential rows, and the operator's own Jacobian, on both backends.
- ``galerkin_steady_residual`` over it equals the steady view's value,
  state Jacobian and parameter Jacobian (``steady_constrained_derivatives``)
  to 1e-12.
- Solved by ``SteadySystem`` (Newton, atol 1e-12, dense LU, from zero) the
  steady state holds the boundary data, makes the unconstrained rows of
  :math:`F` vanish to 1e-10, and agrees across NumPy and Torch to 1e-12.
"""

from typing import Generic

import numpy as np
import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.boundary import DirichletConstraintSet
from pyapprox.pde.galerkin.system import GalerkinSteadySystem
from pyapprox.pde.galerkin.time_integration.dense_discretization import (
    DenseSteadyDiscretization,
)
from pyapprox.pde.models.galerkin.constrained_derivatives import (
    steady_constrained_derivatives,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.linalg.operators import LinearizationPoint
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.mock_elementwise import (
    MockP1Operator,
    ModalCoefficientParameterization,
)
from tests._helpers.rom_problem import FixedValue

_NCELLS = 8
_PARAM = [0.3, -0.2, 0.1, 0.4, -0.1, 0.2]
_POINT = LinearizationPoint(0, ("test",))


class _Case(Generic[Array]):
    def __init__(self, bkd: Backend[Array], reaction: bool) -> None:
        self.bkd = bkd
        self.operator: MockP1Operator[Array] = MockP1Operator(
            _NCELLS, reaction, bkd, velocity=0.3
        )
        nnodes = _NCELLS + 1
        constraint_set = DirichletConstraintSet(
            [FixedValue(0, 1.0, bkd), FixedValue(nnodes - 1, 2.0, bkd)], nnodes, bkd
        )
        self.system: GalerkinSteadySystem[Array] = GalerkinSteadySystem(
            self.operator, constraint_set
        )
        self.parameterization: ModalCoefficientParameterization[Array] = (
            ModalCoefficientParameterization(self.operator, bkd)
        )
        self.discretization: DenseSteadyDiscretization[Array] = (
            DenseSteadyDiscretization(self.system)
        )
        self.parameters = ParameterState(self.parameterization)
        self.param = bkd.asarray(_PARAM)
        self.parameters.set_param(self.param)
        self.n = nnodes

    def solve(self) -> Array:
        bkd = self.bkd
        residual = galerkin_steady_residual(self.discretization, self.parameters)
        system = SteadySystem(
            residual,
            self.discretization.space(),
            RecomputeAlways(BackendDenseLU(bkd)),
        )
        newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=30)
        point = system.solve(newton, bkd.zeros((self.n, 1)))
        assert point.converged()
        return point.state()


class TestTheDiscretization:
    def test_spaces_rows_and_jacobian(self, bkd: Backend[Array]) -> None:
        case: _Case[Array] = _Case(bkd, reaction=True)
        disc = case.discretization
        assert disc.system() is case.system
        assert disc.nstates() == case.n
        assert disc.space().descriptor().size == case.n
        assert disc.parameter_space(6).descriptor().size == 6
        assert disc.essential_rows().count() == 2
        state = bkd.asarray(np.linspace(0.5, 1.5, case.n))
        direction = bkd.asarray(np.linspace(-1.0, 1.0, case.n))[:, None]
        expected = bkd.dot(case.operator.spatial_jacobian(state, 0.0), direction)
        bkd.assert_allclose(
            disc.assemble_jacobian(state, 0.0).apply(direction), expected, rtol=1e-14
        )

    def test_refuses_an_object_that_is_not_a_system(self) -> None:
        with pytest.raises(TypeError, match="GalerkinSteadySystemProtocol"):
            DenseSteadyDiscretization(object())  # type: ignore[arg-type]


class TestTheSteadyResidual:
    @pytest.mark.parametrize("reaction", [False, True])
    def test_equals_the_steady_view(self, numpy_bkd: NumpyBkd, reaction: bool) -> None:
        bkd = numpy_bkd
        case: _Case[NumpyArray] = _Case(bkd, reaction)
        residual = galerkin_steady_residual(case.discretization, case.parameters)
        view = case.system.steady()
        old = steady_constrained_derivatives(
            view, case.parameterization.param_derivatives()
        )
        state = bkd.linspace(0.5, 1.5, case.n)
        bkd.assert_allclose(
            residual(state[:, None])[:, 0], view.steady_residual(state), atol=1e-12
        )
        bundle = residual.derivatives()
        assert bundle.jacobian is not None and bundle.param is not None
        jacobian = bundle.jacobian(state[:, None], _POINT)
        eye = bkd.eye(case.n)
        dense = bkd.hstack([jacobian.apply(eye[:, j : j + 1]) for j in range(case.n)])
        bkd.assert_allclose(dense, view.steady_jacobian(state), atol=1e-12)
        assert old.param_jacobian is not None
        param_jacobian = bundle.param(state[:, None])
        eye6 = bkd.eye(6)
        dense_param = bkd.hstack(
            [param_jacobian.apply(eye6[:, j : j + 1]) for j in range(6)]
        )
        bkd.assert_allclose(
            dense_param, old.param_jacobian(state, case.param), atol=1e-12
        )


class TestTheSteadySolve:
    @pytest.mark.parametrize("reaction", [False, True])
    def test_solves_the_problem(self, bkd: Backend[Array], reaction: bool) -> None:
        case = _Case(bkd, reaction)
        state = case.solve()
        bkd.assert_allclose(state[[0, -1], 0], bkd.asarray([1.0, 2.0]), atol=1e-12)
        raw = case.operator.spatial_residual(state[:, 0], 0.0)
        bkd.assert_allclose(raw[1:-1], bkd.zeros((case.n - 2,)), atol=1e-10)

    @pytest.mark.parametrize("reaction", [False, True])
    def test_agrees_across_backends(
        self, bkd: Backend[Array], numpy_bkd: NumpyBkd, reaction: bool
    ) -> None:
        reference = _Case(numpy_bkd, reaction).solve()
        state = _Case(bkd, reaction).solve()
        bkd.assert_allclose(state, bkd.asarray(reference), atol=1e-12)
