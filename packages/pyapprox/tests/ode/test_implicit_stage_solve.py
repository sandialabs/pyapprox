"""Implicit steppers solving through residual-form systems.

Each implicit stepper's Newton solve, transposed diagonal solve and
adjoint initial condition must agree with a dense solve with
:math:`M - cJ`, whether the system is the dense default or an injected
matrix-free one with GMRES; the matrix the sweeps read is exactly
:math:`M - cJ`. Also the stage solve's own contract.
"""

from typing import Generic, TypeVar, Union

import pytest

from pyapprox.ode.implicit_stage_solve import ImplicitStageSolve, newton_matrix
from pyapprox.ode.implicit_steppers import (
    BackwardEulerHVP,
    BackwardEulerStepper,
    CrankNicolsonHVP,
    ImplicitMidpointHVP,
)
from pyapprox.ode.mass_matrix_system import dense_system, matrix_free_system
from pyapprox.ode.protocols import ResidualFormSystemProtocol
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendVectorSpace,
)
from pyapprox.util.linalg.operators import (
    GMRES,
    CachingSolver,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearSolverProtocol,
    SolveFailedError,
    SpaceDescriptor,
)
from tests._helpers.linalg.no_transpose_solver import NoTransposeSolver
from tests._helpers.ode_models import LotkaVolterra

F = TypeVar("F")

_MASS = [[2.0, 0.5], [0.5, 1.0]]
_DT = 0.1
_Stepper = Union[
    BackwardEulerHVP[Array], CrankNicolsonHVP[Array], ImplicitMidpointHVP[Array]
]


def _model(bkd: Backend[Array]) -> LotkaVolterra[Array]:
    return LotkaVolterra(bkd, mass=bkd.asarray(_MASS))


def _ctx(bkd: Backend[Array]) -> StepContext[Array]:
    return StepContext(t_prev=0.2, deltat=_DT, y_prev=bkd.asarray([9.0, 4.5]))


def _linearization(
    name: str, ctx: StepContext[Array], state: Array
) -> tuple[Array, float, float]:
    """Each method's (linearization state, time, coefficient c)."""
    if name == "midpoint":
        return 0.5 * (ctx.y_prev + state), ctx.t_prev + 0.5 * ctx.deltat, 0.5 * _DT
    if name == "crank-nicolson":
        return state, ctx.t_curr, 0.5 * _DT
    return state, ctx.t_curr, _DT


_NAMES = ["backward-euler", "crank-nicolson", "midpoint"]


class _CountingSystem(Generic[Array, F]):
    """A residual-form system that counts its shifted Jacobians."""

    def __init__(self, inner: ResidualFormSystemProtocol[Array, Array, F]) -> None:
        self._inner = inner
        self._builds = 0

    def builds(self) -> int:
        return self._builds

    def residual(self, state: Array, state_rate: Array, time: float) -> Array:
        return self._inner.residual(state, state_rate, time)

    def shifted_jacobian(
        self,
        state: Array,
        state_rate: Array,
        time: float,
        shift: float,
        point: LinearizationPoint,
    ) -> JacobianSnapshotProtocol[Array, F]:
        self._builds += 1
        return self._inner.shifted_jacobian(state, state_rate, time, shift, point)

    def solver(self) -> LinearSolverProtocol[Array, F]:
        return self._inner.solver()


def _stepper(
    name: str, model: LotkaVolterra[Array], matrix_free: bool
) -> _Stepper[Array]:
    """The method's stepper, with the default (dense) system or an
    injected matrix-free one solved by GMRES."""
    bkd = model.bkd()
    space = BackendVectorSpace(SpaceDescriptor("state", 2), bkd)
    system = (
        matrix_free_system(model, nstates=2, solver=GMRES(space, bkd, tol=1e-13))
        if matrix_free
        else None
    )
    if name == "midpoint":
        return ImplicitMidpointHVP(model, system)
    if name == "crank-nicolson":
        return CrankNicolsonHVP(model, system)
    return BackwardEulerHVP(model, system)


@pytest.mark.parametrize("matrix_free", [False, True], ids=["dense", "free"])
@pytest.mark.parametrize("name", _NAMES)
class TestImplicitSteppers:
    def test_newton_and_adjoint_solves(
        self, bkd: Backend[Array], name: str, matrix_free: bool
    ) -> None:
        model = _model(bkd)
        stepper = _stepper(name, model, matrix_free)
        ctx = _ctx(bkd)
        stepper.bind(ctx)
        state = bkd.asarray([10.0, 5.0])
        y_lin, t_lin, c = _linearization(name, ctx, state)
        newton = newton_matrix(model, y_lin, t_lin, c)
        rhs = bkd.asarray([1.0, -2.0])
        bkd.assert_allclose(
            stepper.linsolve(state, rhs), bkd.solve(newton, rhs), rtol=1e-10
        )
        # The matrix the time-step sweeps read, formed beside the solves.
        bkd.assert_allclose(stepper.jacobian(state), newton, rtol=1e-14)
        # The adjoint's diagonal block is the transposed Newton operator.
        diag = stepper.adjoint_diag_jacobian(ctx, state)
        bkd.assert_allclose(diag.solve(rhs), bkd.solve(newton.T, rhs), rtol=1e-10)
        bkd.assert_allclose(
            diag.solve_transpose(rhs), bkd.solve(newton, rhs), rtol=1e-10
        )
        dqdu = bkd.asarray([0.3, 0.7])
        bkd.assert_allclose(
            stepper.adjoint_initial_condition(ctx, state, dqdu),
            bkd.solve(newton.T, -dqdu),
            rtol=1e-10,
        )


class TestImplicitStageSolve:
    def test_injected_system_wins_and_matrix_comes_from_the_stepper(
        self, bkd: Backend[Array]
    ) -> None:
        model = _model(bkd)
        stage = ImplicitStageSolve(model, matrix_free_system(model, nstates=2))
        state = bkd.asarray([10.0, 5.0])
        rate = bkd.zeros((2,))
        op = stage.newton_operator(
            state, rate, 0.0, _DT, LinearizationPoint(0), lambda: bkd.eye(2)
        )
        bkd.assert_allclose(op.as_matrix(), bkd.eye(2))
        newton = newton_matrix(model, state, 0.0, _DT)
        x = bkd.asarray([[1.0, 0.0], [0.5, 2.0]])
        bkd.assert_allclose(op.apply(x), bkd.dot(newton, x), rtol=1e-12)
        bkd.assert_allclose(op.apply_transpose(x), bkd.dot(newton.T, x), rtol=1e-12)

    def test_the_shifted_jacobian_is_built_on_first_use_only(
        self, bkd: Backend[Array]
    ) -> None:
        """Asking for the matrix alone never assembles the shifted
        Jacobian; solves and actions build it once and share it."""
        model = _model(bkd)
        system = _CountingSystem(dense_system(model))
        op = ImplicitStageSolve(model, system).newton_operator(
            bkd.asarray([10.0, 5.0]),
            bkd.zeros((2,)),
            0.0,
            _DT,
            LinearizationPoint(0),
            lambda: bkd.eye(2),
        )
        op.as_matrix()
        assert system.builds() == 0
        rhs = bkd.asarray([1.0, -2.0])
        newton = newton_matrix(model, bkd.asarray([10.0, 5.0]), 0.0, _DT)
        bkd.assert_allclose(op.solve(rhs), bkd.solve(newton, rhs), rtol=1e-12)
        op.apply(rhs)
        op.solve_transpose(rhs)
        assert system.builds() == 1

    def test_a_failed_solve_raises(self, bkd: Backend[Array]) -> None:
        model = _model(bkd)
        system = matrix_free_system(
            model,
            nstates=2,
            solver=GMRES(
                BackendVectorSpace(SpaceDescriptor("state", 2), bkd), bkd, maxiter=1
            ),
        )
        op = ImplicitStageSolve(model, system).newton_operator(
            bkd.asarray([10.0, 5.0]),
            bkd.zeros((2,)),
            0.0,
            _DT,
            LinearizationPoint(0),
            lambda: bkd.eye(2),
        )
        with pytest.raises(SolveFailedError, match="relative residual"):
            op.solve(bkd.asarray([1.0, -2.0]))

    def test_a_solver_without_transposes_is_refused(self, bkd: Backend[Array]) -> None:
        model = _model(bkd)
        system = dense_system(model, solver=NoTransposeSolver(bkd))
        op = ImplicitStageSolve(model, system).newton_operator(
            bkd.asarray([10.0, 5.0]),
            bkd.zeros((2,)),
            0.0,
            _DT,
            LinearizationPoint(0),
            lambda: bkd.eye(2),
        )
        bkd.assert_allclose(
            op.solve(bkd.asarray([1.0, 0.0])),
            bkd.solve(
                newton_matrix(model, bkd.asarray([10.0, 5.0]), 0.0, _DT),
                bkd.asarray([1.0, 0.0]),
            ),
            rtol=1e-12,
        )
        with pytest.raises(TypeError, match="no transposed solve"):
            op.solve_transpose(bkd.asarray([1.0, 0.0]))


class TestStagePointsAndCapabilities:
    def test_every_linearization_gets_a_new_label(self, bkd: Backend[Array]) -> None:
        """A caching solver never reuses one Newton iterate's setup for
        another at the same state and time."""
        stage = ImplicitStageSolve(_model(bkd))
        assert stage.stage_point(0.3) != stage.stage_point(0.3)
        model = _model(bkd)
        cache = CachingSolver(BackendDenseLU(bkd), 4)
        stepper = BackwardEulerHVP(model, dense_system(model, solver=cache))
        stepper.bind(_ctx(bkd))
        state, rhs = bkd.asarray([10.0, 5.0]), bkd.asarray([1.0, -2.0])
        stepper.linsolve(state, rhs)
        stepper.linsolve(state, rhs)
        assert cache.setups() == 2 and cache.hits() == 0

    @pytest.mark.parametrize("name", _NAMES)
    def test_adjoint_tiers_refuse_a_solver_without_transposes(
        self, bkd: Backend[Array], name: str
    ) -> None:
        model = _model(bkd)
        system = dense_system(model, solver=NoTransposeSolver(bkd))
        with pytest.raises(TypeError, match="needs transposed solves"):
            if name == "midpoint":
                ImplicitMidpointHVP(model, system)
            elif name == "crank-nicolson":
                CrankNicolsonHVP(model, system)
            else:
                BackwardEulerHVP(model, system)
        # A forward-only stepper does not need them.
        BackwardEulerStepper(model, system)
