r"""The steady residual of a Galerkin system equals the steady view's.

``galerkin_steady_residual`` (the unconstrained right-hand side at t*,
its essential rows pinned to their data at t*) is compared with the
steady view and ``steady_constrained_derivatives`` on four problems:

- the 1D diffusion problem of ``tests._helpers.galerkin_steady``,
  linear and with the reaction :math:`u^2`, at p = (0.4, -0.3, 0.2);
- the elastic plate of the same helper (Robin support, traction, body
  force) at (E, nu) = (1, 0.3);
- the heat problem of ``tests._helpers.galerkin_heat`` (time-dependent
  forcing, moving right end 0.5 sin 4t) frozen at t* = 0.03, the
  residual given the time and the view a snapshot.

At u = linspace(0.1, 0.9), adjoint linspace(-1, 1) and seeded random
directions, the value, the state Jacobian (dense), the parameter
Jacobian and the four adjoint-weighted second-derivative products agree
to 1e-12, relative. Solved by ``SteadySystem`` (stack ``Newton``, atol
1e-13, SuperLU, ``KeepAll``) from zero, the steady state equals
``SteadyStateSolver`` on the view to 1e-10 relative, with one
factorization per Newton step plus one for the handle at the solution.

Parameter-dependent essential data: the nonlinear diffusion problem with
the right end pinned to :math:`g(p) = 0.2 p_0^2 + 0.1 p_1 p_2` instead of
zero, through a declaration supplying :math:`\partial g/\partial p` and
its curvature; the quantity is the state at x = 0.9. The gradient and the
Hessian-vector product of ``SteadyDerivatives`` pass central finite
differences (smallest relative error below 1e-6). Wrong-model control:
the same data declared independent of p give a gradient whose
finite-difference error stays above 1e-2.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse

from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import (
    PinnedValuesParamDerivatives,
    RowPin,
)
from pyapprox.implicitfunction.residual_derivatives import DerivativeOperator
from pyapprox.implicitfunction.setups import KeepAll, RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.residual_at_time import RightHandSideAtTime
from pyapprox.ode.residual_derivatives import STATED_ZERO, dense_operator
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.constrained_derivatives import (
    steady_constrained_derivatives,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.models.galerkin.unconstrained_system import (
    GalerkinUnconstrainedRightHandSide,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    InnerProductSpaceProtocol,
    LinearizationPoint,
    RowSetProtocol,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.galerkin_heat import HEAT_PARAM, MovingEndHeat
from tests._helpers.galerkin_steady import (
    ADR_PARAM,
    PLATE_PARAM,
    elastic_plate,
    steady_adr,
)
from tests._helpers.linalg.counting_solver import CountingSolver

_Np = NDArray[Any]
_HEAT_TIME = 0.03
_POINT = LinearizationPoint(0, ("test",))
_RTOL = 1e-12
_CASES = ["linear", "nonlinear", "elastic plate", "heat snapshot"]


class _Case:
    """A problem's new residual and old view, at its parameters."""

    def __init__(self, bkd: NumpyBkd, name: str) -> None:
        self.bkd = bkd
        self.time: Optional[float] = None
        system: GalerkinSystem[_Np]
        parameterization: ParameterizationProtocol[_Np]
        if name == "heat snapshot":
            heat = MovingEndHeat(bkd)
            system, parameterization = heat.system, heat.parameterization
            param, self.time = HEAT_PARAM, _HEAT_TIME
            self.right_end = int(heat.ends[1])
        elif name == "elastic plate":
            system, parameterization = elastic_plate(bkd)
            param = PLATE_PARAM
        else:
            system, parameterization, _ = steady_adr(bkd, name == "nonlinear")
            param = ADR_PARAM
        self.system = system
        self.discretization = skfem_discretization(system)
        self.param = param[:, 0]
        self.nparams: int = self.param.shape[0]
        self.parameters = ParameterState(parameterization)
        self.parameters.set_param(self.param)
        self.residual = galerkin_steady_residual(
            self.discretization, self.parameters, self.time
        )
        self.view = (
            system.steady() if self.time is None else system.steady_snapshot(self.time)
        )
        self.old = steady_constrained_derivatives(
            self.view, parameterization.param_derivatives()
        )
        n = system.nstates()
        self.n = n
        self.state = np.linspace(0.1, 0.9, n)
        self.adj = np.linspace(-1.0, 1.0, n)
        rng = np.random.default_rng(11)
        self.wvec: _Np = rng.normal(size=n)
        self.vec: _Np = rng.normal(size=(self.nparams,))

    def dense(self, op: DerivativeOperator[_Np], ncols: int) -> _Np:
        return np.hstack([op.apply(np.eye(ncols)[:, j : j + 1]) for j in range(ncols)])


def _close(actual: _Np, expected: _Np, rtol: float = _RTOL) -> None:
    scale = max(float(np.max(np.abs(expected))), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=rtol * scale)


class TestAgainstTheSteadyView:
    @pytest.mark.parametrize("name", _CASES)
    def test_the_value(self, numpy_bkd: NumpyBkd, name: str) -> None:
        case = _Case(numpy_bkd, name)
        value = case.residual(case.state[:, None])[:, 0]
        _close(value, case.view.steady_residual(case.state))

    def test_the_pinned_rows_read_the_data_at_the_time(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        case = _Case(numpy_bkd, "heat snapshot")
        value = case.residual(case.state[:, None])[:, 0]
        right = case.right_end
        _close(
            value[right : right + 1],
            np.asarray([0.9 - 0.5 * np.sin(4.0 * _HEAT_TIME)]),
        )

    @pytest.mark.parametrize("name", _CASES)
    def test_the_jacobians(self, numpy_bkd: NumpyBkd, name: str) -> None:
        case = _Case(numpy_bkd, name)
        bundle = case.residual.derivatives()
        assert bundle.jacobian is not None and bundle.param is not None
        old_jacobian = case.view.steady_jacobian(case.state)
        assert issparse(old_jacobian)
        state = case.state[:, None]
        _close(
            case.dense(bundle.jacobian(state, _POINT), case.n),
            csr_matrix(old_jacobian).toarray(),
        )
        assert case.old.param_jacobian is not None
        _close(
            case.dense(bundle.param(state), case.nparams),
            case.old.param_jacobian(case.state, case.param),
        )

    @pytest.mark.parametrize("name", _CASES)
    def test_the_second_derivatives(self, numpy_bkd: NumpyBkd, name: str) -> None:
        case = _Case(numpy_bkd, name)
        bundle = case.residual.derivatives()
        old = case.old
        assert bundle.state_state is not None and bundle.state_param is not None
        assert bundle.param_param is not None
        assert old.state_state_hvp is not None and old.state_param_hvp is not None
        assert old.param_state_hvp is not None and old.param_param_hvp is not None
        state, adj = case.state[:, None], case.adj[:, None]
        wvec, vec = case.wvec[:, None], case.vec[:, None]
        _close(
            bundle.state_state(state, adj).apply(wvec)[:, 0],
            old.state_state_hvp(case.state, case.adj, case.wvec),
        )
        _close(
            bundle.state_param(state, adj).apply(vec)[:, 0],
            old.state_param_hvp(case.state, case.param, case.adj, case.vec),
        )
        _close(
            bundle.state_param(state, adj).apply_transpose(wvec)[:, 0],
            old.param_state_hvp(case.state, case.param, case.adj, case.wvec),
        )
        _close(
            bundle.param_param(state, adj).apply(vec)[:, 0],
            old.param_param_hvp(case.state, case.param, case.adj, case.vec),
        )

    def test_the_version_is_the_shared_parameters(self, numpy_bkd: NumpyBkd) -> None:
        case = _Case(numpy_bkd, "linear")
        assert case.residual.parameters_version() == 1
        case.parameters.set_param(case.param)
        assert case.residual.parameters_version() == 2


class TestTheSteadySolve:
    @pytest.mark.parametrize("name", _CASES)
    def test_it_equals_the_steady_state_solver(
        self, numpy_bkd: NumpyBkd, name: str
    ) -> None:
        case = _Case(numpy_bkd, name)
        space = case.discretization.space()
        assert isinstance(space, InnerProductSpaceProtocol)
        lu: CountingSolver[_Np, Any] = CountingSolver(ScipySparseLU())
        system = SteadySystem(case.residual, space, KeepAll(lu))
        point = system.solve(
            Newton(atol=1e-13, rtol=0.0, maxiters=30), np.zeros((case.n, 1))
        )
        assert point.converged()
        iterations = point.info().iterations
        assert (iterations == 1) == (name != "nonlinear")
        old = SteadyStateSolver(case.view, tol=1e-13).solve()
        _close(point.state()[:, 0], old.solution, 1e-10)
        point.jacobian_handle()
        assert lu.setups == (1 if iterations == 1 else iterations + 1)


class TestTheBuilderRefuses:
    def test_time_dependent_data_without_a_time(self, numpy_bkd: NumpyBkd) -> None:
        heat = MovingEndHeat(numpy_bkd)
        with pytest.raises(ValueError, match="time-dependent"):
            galerkin_steady_residual(
                skfem_discretization(heat.system),
                ParameterState(heat.parameterization),
            )

    def test_a_system_that_is_not_discretized(self, numpy_bkd: NumpyBkd) -> None:
        system, _, _ = steady_adr(numpy_bkd, False)
        with pytest.raises(TypeError, match="GalerkinStackDiscretization"):
            galerkin_steady_residual(system)  # type: ignore[arg-type]

    def test_parameters_that_are_not_a_parameter_state(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system, parameterization, _ = steady_adr(numpy_bkd, False)
        with pytest.raises(TypeError, match="ParameterState"):
            galerkin_steady_residual(
                skfem_discretization(system),
                parameterization,  # type: ignore[arg-type]
            )


class TestWithoutParameters:
    def test_value_and_state_only_bundle(self, numpy_bkd: NumpyBkd) -> None:
        system, _, _ = steady_adr(numpy_bkd, True)
        residual = galerkin_steady_residual(skfem_discretization(system))
        bundle = residual.derivatives()
        assert bundle.jacobian is not None and bundle.param is None
        state = np.linspace(0.1, 0.9, system.nstates())
        _close(residual(state[:, None])[:, 0], system.steady().steady_residual(state))
        assert residual.parameters_version() == 0


class _RightEndData:
    r"""The essential rows of the diffusion problem, the right end pinned
    to :math:`g(p) = 0.2 p_0^2 + 0.1 p_1 p_2` and the left to zero;
    ``declared`` chooses whether the data say they depend on p."""

    def __init__(
        self,
        rows: RowSetProtocol[_Np],
        right: int,
        nstates: int,
        states: VectorSpaceProtocol[_Np],
        params: VectorSpaceProtocol[_Np],
        bkd: NumpyBkd,
        declared: bool,
    ) -> None:
        self._rows, self._right, self._n = rows, right, nstates
        self._p = ADR_PARAM[:, 0].copy()
        self._jacobian: Callable[[], DerivativeOperator[_Np]] = dense_operator(
            self._jacobian_matrix, params, states, bkd
        )
        self._curvature: Callable[[_Np], DerivativeOperator[_Np]] = dense_operator(
            self._curvature_matrix, params, params, bkd, symmetric=True
        )
        self._declared = declared

    def set_param(self, param: _Np) -> None:
        self._p = param.copy()

    def rows(self) -> RowSetProtocol[_Np]:
        return self._rows

    def values(self) -> _Np:
        g = np.zeros((self._n, 1))
        p = self._p
        g[self._right, 0] = 0.2 * p[0] ** 2 + 0.1 * p[1] * p[2]
        return g

    def _jacobian_matrix(self) -> _Np:
        p = self._p
        matrix = np.zeros((self._n, 3))
        matrix[self._right] = [0.4 * p[0], 0.1 * p[2], 0.1 * p[1]]
        return matrix

    def _curvature_matrix(self, weights: _Np) -> _Np:
        w = float(weights[self._right, 0])
        return w * np.array([[0.4, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.1, 0.0]])

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[_Np]:
        if not self._declared:
            return PinnedValuesParamDerivatives.independent()
        return PinnedValuesParamDerivatives.second_order(
            self._jacobian, self._curvature
        )


class _DataProblem:
    """The nonlinear diffusion problem pinned to ``_RightEndData``; one
    ``set_param`` moves the coefficients and the data."""

    def __init__(self, bkd: NumpyBkd, declared: bool) -> None:
        self.bkd = bkd
        system, parameterization, basis = steady_adr(bkd, True)
        discretization = skfem_discretization(system)
        space = discretization.space()
        assert isinstance(space, InnerProductSpaceProtocol)
        n = system.nstates()
        self.n = n
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        right = int(bkd.to_numpy(basis.get_dofs("right"))[0])
        observed = int(np.argmin(np.abs(coords - 0.9)))
        self.parameters = ParameterState(parameterization)
        self.data = _RightEndData(
            discretization.essential_rows(),
            right,
            n,
            space,
            discretization.parameter_space(3),
            bkd,
            declared,
        )
        rhs: GalerkinUnconstrainedRightHandSide[_Np, Any] = (
            GalerkinUnconstrainedRightHandSide(
                discretization, self.parameters, initial_state=STATED_ZERO
            )
        )
        residual = PinnedResidual(
            RightHandSideAtTime(rhs, 0.0, BackendColumnForm(bkd)),
            RowPin(self.data),
            space,
        )
        self.system = SteadySystem(residual, space, RecomputeAlways(ScipySparseLU()))
        self.derivatives: SteadyDerivatives[_Np, _Np, Any] = SteadyDerivatives(
            residual,
            space,
            StateFunctional.from_terms(
                bkd,
                1,
                state_terms=(ObservedValue(StateEntries(bkd, n, [observed])),),
            ),
        )

    def solve(self, param: _Np) -> SteadyPoint[_Np, _Np]:
        self.parameters.set_param(param[:, 0])
        self.data.set_param(param[:, 0])
        point = self.system.solve(
            Newton(atol=1e-13, rtol=0.0, maxiters=30), np.zeros((self.n, 1))
        )
        assert point.converged()
        return point

    def value(self, samples: _Np) -> _Np:
        return np.hstack(
            [
                self.derivatives.value(self.solve(samples[:, k : k + 1]))
                for k in range(samples.shape[1])
            ]
        )

    def gradient(self, sample: _Np) -> _Np:
        derivs = self.derivatives
        return derivs.gradient(derivs.adjoint(self.solve(sample)))

    def hessian_matrix(self, sample: _Np) -> _Np:
        hessian = self.derivatives.hessian()
        assert hessian is not None
        adjoint = self.derivatives.adjoint(self.solve(sample))
        return np.hstack(
            [hessian.hvp(adjoint, np.eye(3)[:, j : j + 1]) for j in range(3)]
        )


def _min_error(
    bkd: NumpyBkd, nqoi: int, fun: Callable[[_Np], _Np], jacobian: Callable[[_Np], _Np]
) -> float:
    errors = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=3, fun=fun, jacobian=jacobian, bkd=bkd
        )
    ).check_derivatives(ADR_PARAM, relative=True, central=True)[0]
    return bkd.to_float(bkd.min(errors))


def _gradients(problem: _DataProblem) -> Callable[[_Np], _Np]:
    def gradients(samples: _Np) -> _Np:
        return np.hstack(
            [problem.gradient(samples[:, k : k + 1]) for k in range(samples.shape[1])]
        )

    return gradients


def _gradient_row(problem: _DataProblem) -> Callable[[_Np], _Np]:
    def gradient_row(sample: _Np) -> _Np:
        return problem.gradient(sample).T

    return gradient_row


class TestParameterDependentEssentialData:
    def test_the_data_move_the_solution(self, numpy_bkd: NumpyBkd) -> None:
        """Guards against a vacuous check: the pinned value is not the
        zero the problem's own boundary condition holds."""
        problem = _DataProblem(numpy_bkd, True)
        p = ADR_PARAM[:, 0]
        expected = 0.2 * p[0] ** 2 + 0.1 * p[1] * p[2]
        assert abs(expected) > 1e-2
        point = problem.solve(ADR_PARAM)
        assert np.max(np.abs(point.state())) > abs(expected) / 2

    def test_gradient_and_hvp(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        problem = _DataProblem(bkd, True)
        assert _min_error(bkd, 1, problem.value, _gradient_row(problem)) < 1e-6
        assert _min_error(bkd, 3, _gradients(problem), problem.hessian_matrix) < 1e-6

    def test_data_declared_independent_give_a_wrong_gradient(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        problem = _DataProblem(bkd, False)
        assert _min_error(bkd, 1, problem.value, _gradient_row(problem)) > 1e-2
