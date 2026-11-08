r"""The pinned steady residual of a Galerkin system equals the steady view's.

The heat problem of ``tests._helpers.galerkin_heat`` (1D, P1 on 10
cells, log-KLE diffusivity at p = (0.4, -0.3, 0.2), forcing (1 + x)(1 +
sin 5t), u(0) = 0, moving right end 0.5 sin 4t), frozen at t* = 0.03.

``PinnedResidual`` of the unconstrained right-hand side at t*
(``RightHandSideAtTime``) with the essential rows pinned to their data at
t* (``values_at_time`` of ``EssentialPinnedRows``) is compared with the
steady view (``SteadyView.snapshot``) and ``steady_constrained_derivatives``
at u = linspace(0.1, 0.9), adjoint linspace(-1, 1) and seeded random
directions: the value, the state Jacobian (dense), the parameter
Jacobian, and the four adjoint-weighted second-derivative products agree
to 1e-12, relative. The pinned rows of the value read u - g(t*), which
for the moving end is 0.9 - 0.5 sin 0.12.

Solved by ``SteadySystem`` with stack ``Newton`` (atol 1e-13) and SuperLU
from zero, the steady state equals ``SteadyStateSolver`` on the view to
1e-12 relative, in one step and one factorization, which the solved
point's handle reuses.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import RowPin
from pyapprox.implicitfunction.residual_derivatives import DerivativeOperator
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.ode.pinned_rows import values_at_time
from pyapprox.ode.residual_at_time import RightHandSideAtTime
from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.galerkin.time_integration.essential_pinned_rows import (
    EssentialPinnedRows,
)
from pyapprox.pde.models.galerkin.constrained_derivatives import (
    steady_constrained_derivatives,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.models.galerkin.unconstrained_system import (
    GalerkinUnconstrainedRightHandSide,
)
from pyapprox.pde.steady_view import SteadyView
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    InnerProductSpaceProtocol,
    LinearizationPoint,
)
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.galerkin_heat import HEAT_PARAM, MovingEndHeat
from tests._helpers.linalg.counting_solver import CountingSolver

_Np = NDArray[Any]
_TIME = 0.03
_POINT = LinearizationPoint(0, ("test",))
_RTOL = 1e-12


class _Case:
    def __init__(self, bkd: NumpyBkd) -> None:
        self.bkd = bkd
        self.heat = MovingEndHeat(bkd)
        discretization = skfem_discretization(self.heat.system)
        self.discretization = discretization
        adapter = create_galerkin_physics_ode_residual(
            discretization, ScipySparseLU(), self.heat.parameterization
        )
        self.param = HEAT_PARAM[:, 0]
        adapter.set_param(self.param)
        rhs: GalerkinUnconstrainedRightHandSide[_Np, Any] = (
            GalerkinUnconstrainedRightHandSide(
                discretization, adapter.parameter_state(), initial_state=STATED_ZERO
            )
        )
        self.residual = RightHandSideAtTime(rhs, _TIME, BackendColumnForm(bkd))
        self.pinned = PinnedResidual(
            self.residual,
            RowPin(values_at_time(EssentialPinnedRows(discretization), _TIME)),
            discretization.space(),
        )
        self.view = SteadyView.snapshot(
            self.heat.system.spatial_operator(),
            self.heat.system.constraint_set(),
            _TIME,
        )
        self.old = steady_constrained_derivatives(
            self.view, self.heat.parameterization.param_derivatives()
        )
        n = self.heat.system.nstates()
        self.n = n
        self.state = np.linspace(0.1, 0.9, n)
        self.adj = np.linspace(-1.0, 1.0, n)
        rng = np.random.default_rng(11)
        self.wvec = rng.normal(size=n)
        self.vec = rng.normal(size=3)

    def column(self, x: _Np) -> _Np:
        return x[:, None]

    def dense(self, op: DerivativeOperator[_Np], ncols: int) -> _Np:
        return np.hstack([op.apply(np.eye(ncols)[:, j : j + 1]) for j in range(ncols)])


def _close(actual: _Np, expected: _Np) -> None:
    scale = max(float(np.max(np.abs(expected))), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_RTOL * scale)


class TestAgainstTheSteadyView:
    def test_the_value(self, numpy_bkd: NumpyBkd) -> None:
        case = _Case(numpy_bkd)
        value = case.pinned(case.column(case.state))[:, 0]
        _close(value, case.view.steady_residual(case.state))
        right = int(case.heat.ends[1])
        _close(
            value[right : right + 1],
            np.asarray([0.9 - 0.5 * np.sin(4.0 * _TIME)]),
        )

    def test_the_jacobians(self, numpy_bkd: NumpyBkd) -> None:
        case = _Case(numpy_bkd)
        bundle = case.pinned.derivatives()
        assert bundle.jacobian is not None and bundle.param is not None
        old_jacobian = case.view.steady_jacobian(case.state)
        assert issparse(old_jacobian)
        state = case.column(case.state)
        _close(
            case.dense(bundle.jacobian(state, _POINT), case.n),
            csr_matrix(old_jacobian).toarray(),
        )
        assert case.old.param_jacobian is not None
        _close(
            case.dense(bundle.param(state), 3),
            case.old.param_jacobian(case.state, case.param),
        )

    def test_the_second_derivatives(self, numpy_bkd: NumpyBkd) -> None:
        case = _Case(numpy_bkd)
        bundle = case.pinned.derivatives()
        old = case.old
        assert bundle.state_state is not None and bundle.state_param is not None
        assert bundle.param_param is not None
        assert old.state_state_hvp is not None and old.state_param_hvp is not None
        assert old.param_state_hvp is not None and old.param_param_hvp is not None
        state, adj = case.column(case.state), case.column(case.adj)
        wvec, vec = case.column(case.wvec), case.column(case.vec)
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
        case = _Case(numpy_bkd)
        version = case.pinned.parameters_version()
        assert version == case.residual.parameters_version() > 0


class TestTheSteadySolve:
    def test_it_equals_the_steady_state_solver(self, numpy_bkd: NumpyBkd) -> None:
        case = _Case(numpy_bkd)
        space = case.discretization.space()
        assert isinstance(space, InnerProductSpaceProtocol)
        lu: CountingSolver[_Np, Any] = CountingSolver(ScipySparseLU())
        system = SteadySystem(case.pinned, space, lu)
        assert system.is_affine()
        point = system.solve(
            Newton(atol=1e-13, rtol=0.0, maxiters=20), np.zeros((case.n, 1))
        )
        assert point.converged() and point.info().iterations == 1
        old = SteadyStateSolver(case.view).solve()
        _close(point.state()[:, 0], old.solution)
        point.jacobian_handle()
        assert lu.setups == 1
