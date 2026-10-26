"""Explicit steppers solving with their mass through a mass system.

Forward Euler and Heun given a dense or a matrix-free (GMRES) system must
reproduce, method by method, the steppers that solve with the model's own
mass: the step residual and its solve, the sensitivity block, the
adjoint's blocks and boundary solves, and the HVP terms. These are all
the time-step sweeps read, so gradients and HVPs follow. Also
``MassSolve``'s own contract, that the mass is set up once, and the
refusal of a singular mass. These tests go with ``MassSolve``.
"""

from typing import Generic, Optional, Union

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerHVP
from pyapprox.ode.explicit_steppers.heun import HeunHVP, HeunStepper
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.mass_solve import MassSolve
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.semi_explicit_system import dense_system, matrix_free_system
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    GMRES,
    ActionOperatorProtocol,
    SolveFailedError,
    SolverCapabilities,
    SolverHandleProtocol,
    SpaceDescriptor,
)
from pyapprox.util.rootfinding.newton import NewtonSolver
from tests._helpers.linalg.no_transpose_solver import NoTransposeSolver
from tests._helpers.ode_models import LotkaVolterra

_MASS = [[2.0, 0.5], [0.3, 1.0]]
_OTHER_MASS = [[1.0, -0.2], [0.4, 3.0]]
_Stepper = Union[ForwardEulerHVP[Array], HeunHVP[Array]]


def _model(
    bkd: Backend[Array], mass: list[list[float]] = _MASS
) -> LotkaVolterra[Array]:
    return LotkaVolterra(bkd, mass=bkd.asarray(mass))


def _gmres(bkd: Backend[Array], maxiter: int = 1000) -> GMRES[Array, Array]:
    space = BackendVectorSpace(SpaceDescriptor("state", 2), bkd)
    return GMRES(space, bkd, tol=1e-13, maxiter=maxiter)


def _stepper(
    name: str,
    model: LotkaVolterra[Array],
    kind: Optional[str],
    system_model: Optional[LotkaVolterra[Array]] = None,
) -> _Stepper[Array]:
    """The method's stepper, solving with the model's own mass when
    ``kind`` is None, else through a dense or a matrix-free system built
    from ``system_model`` (default: the model)."""
    heun = name == "heun"
    if kind is None:
        return HeunHVP(model) if heun else ForwardEulerHVP(model)
    source = model if system_model is None else system_model
    if kind == "free":
        free = matrix_free_system(source, nstates=2, solver=_gmres(model.bkd()))
        return HeunHVP(model, free) if heun else ForwardEulerHVP(model, free)
    dense = dense_system(source)
    return HeunHVP(model, dense) if heun else ForwardEulerHVP(model, dense)


def _ctx(bkd: Backend[Array]) -> StepContext[Array]:
    return StepContext(t_prev=0.2, deltat=0.1, y_prev=bkd.asarray([9.0, 4.5]))


@pytest.mark.parametrize("kind", ["dense", "free"])
@pytest.mark.parametrize("name", ["forward-euler", "heun"])
class TestExplicitSteppersThroughASystem:
    def test_every_mass_use_matches_the_models_mass(
        self, bkd: Backend[Array], name: str, kind: str
    ) -> None:
        model = _model(bkd)
        reference = _stepper(name, model, None)
        stepper = _stepper(name, model, kind)
        ctx = _ctx(bkd)
        reference.bind(ctx)
        stepper.bind(ctx)
        state = bkd.asarray([10.0, 5.0])
        rhs, dqdu = bkd.asarray([1.0, -2.0]), bkd.asarray([0.3, 0.7])
        adj, wvec = bkd.asarray([0.4, -1.1]), bkd.asarray([0.6, 0.2])
        vvec = bkd.asarray([0.5, -0.3, 0.8, 0.1])
        identity = bkd.eye(2)
        pairs = [
            (stepper(state), reference(state)),
            (stepper.linsolve(state, rhs), reference.linsolve(state, rhs)),
            (stepper.jacobian(state), reference.jacobian(state)),
            (
                stepper.off_diag_operator(ctx, state).apply(identity),
                reference.off_diag_operator(ctx, state).apply(identity),
            ),
            (stepper.param_jacobian(ctx, state), reference.param_jacobian(ctx, state)),
            (
                stepper.off_diag_operator(ctx, state).apply_transpose(identity),
                reference.off_diag_operator(ctx, state).apply_transpose(identity),
            ),
            (
                stepper.adjoint_diag_jacobian(ctx, state).solve(rhs),
                reference.adjoint_diag_jacobian(ctx, state).solve(rhs),
            ),
            (
                stepper.adjoint_diag_jacobian(ctx, state).apply(rhs),
                reference.adjoint_diag_jacobian(ctx, state).apply(rhs),
            ),
            (
                stepper.adjoint_initial_condition(ctx, state, dqdu),
                reference.adjoint_initial_condition(ctx, state, dqdu),
            ),
            (
                stepper.adjoint_final_solution(ctx, state, adj, dqdu),
                reference.adjoint_final_solution(ctx, state, adj, dqdu),
            ),
            (
                stepper.param_param_hvp(ctx, state, adj, vvec),
                reference.param_param_hvp(ctx, state, adj, vvec),
            ),
            (
                stepper.prev_state_state_hvp(ctx, state, adj, wvec),
                reference.prev_state_state_hvp(ctx, state, adj, wvec),
            ),
            (
                stepper.prev_state_param_hvp(ctx, state, adj, vvec),
                reference.prev_state_param_hvp(ctx, state, adj, vvec),
            ),
            (
                stepper.prev_param_state_hvp(ctx, state, adj, wvec),
                reference.prev_param_state_hvp(ctx, state, adj, wvec),
            ),
        ]
        for actual, expected in pairs:
            bkd.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_a_systems_mass_replaces_the_models(
        self, bkd: Backend[Array], name: str, kind: str
    ) -> None:
        """Given a system, the stepper never reads the model's mass: every
        use, the matrix-forming ones included, is the system's. The system
        comes from the same right-hand side with a different mass, so the
        reference is the stepper of that model."""
        model, other = _model(bkd), _model(bkd, _OTHER_MASS)
        stepper = _stepper(name, model, kind, system_model=other)
        reference = _stepper(name, other, None)
        ctx = _ctx(bkd)
        stepper.bind(ctx)
        reference.bind(ctx)
        state, rhs = bkd.asarray([10.0, 5.0]), bkd.asarray([1.0, -2.0])
        adj, dqdu = bkd.asarray([0.4, -1.1]), bkd.asarray([0.3, 0.7])
        pairs = [
            (stepper(state), reference(state)),
            (stepper.linsolve(state, rhs), reference.linsolve(state, rhs)),
            (stepper.jacobian(state), reference.jacobian(state)),
            (
                stepper.off_diag_operator(ctx, state).apply(bkd.eye(2)),
                reference.off_diag_operator(ctx, state).apply(bkd.eye(2)),
            ),
            (
                stepper.adjoint_initial_condition(ctx, state, dqdu),
                reference.adjoint_initial_condition(ctx, state, dqdu),
            ),
            (
                stepper.adjoint_final_solution(ctx, state, adj, dqdu),
                reference.adjoint_final_solution(ctx, state, adj, dqdu),
            ),
        ]
        for actual, expected in pairs:
            bkd.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_gradient_and_hvp_through_the_integrator(
        self, bkd: Backend[Array], name: str, kind: str
    ) -> None:
        """The adjoint gradient and HVP of the final prey population with
        respect to the four coefficients, by the derivative checker: finite
        differences of Q against the gradient, and of the gradient against
        H v, through ten steps solved with the system."""
        y0 = bkd.asarray([10.0, 5.0])

        def operator() -> TimeAdjointOperatorWithHVP[Array]:
            stepper = _stepper(name, _model(bkd), kind)
            newton = NewtonSolver(stepper)
            newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
            integrator = TimeIntegrator(0.0, 0.5, 0.05, newton)
            return TimeAdjointOperatorWithHVP(
                integrator, EndpointFunctional(0, 2, 4, bkd)
            )

        def fun(samples: Array) -> Array:
            return bkd.hstack(
                [operator()(y0, samples[:, k : k + 1]) for k in range(samples.shape[1])]
            )

        wrapper = FunctionWithJacobianAndHVPFromCallable(
            nvars=4,
            fun=fun,
            jacobian=lambda sample: operator().jacobian(y0, sample),
            hvp=lambda sample, vec: operator().hvp(y0, sample, vec),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        base = bkd.asarray([[1.0], [0.1], [0.075], [1.5]])
        grad_errors, hvp_errors = checker.check_derivatives(base, relative=True)
        # The errors fall tenfold per decade of the finite-difference step
        # until rounding: Heun's floor is near 1.1e-7, while a missing term
        # plateaus at O(1), which the ratio catches.
        for errors in (grad_errors, hvp_errors):
            assert float(bkd.to_numpy(bkd.min(errors))) <= 1e-6
            assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 5e-6

    def test_refuses_a_singular_mass(
        self, bkd: Backend[Array], name: str, kind: str
    ) -> None:
        """The mass the stepper uses decides: the system's when given (here
        built from the same singular model), else the model's."""
        model = _model(bkd, [[1.0, 0.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match=r"singular: rows \[1\]"):
            _stepper(name, model, kind)
        with pytest.raises(ValueError, match=r"singular: rows \[1\]"):
            _stepper(name, model, None)


class _CountingLU(Generic[Array]):
    """A dense LU that counts its setups."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._lu = BackendDenseLU(bkd)
        self._setups = 0

    def setups(self) -> int:
        return self._setups

    def capabilities(self) -> SolverCapabilities:
        return self._lu.capabilities()

    def setup(
        self, op: ActionOperatorProtocol[Array, DenseFormat[Array]]
    ) -> SolverHandleProtocol[Array]:
        self._setups += 1
        return self._lu.setup(op)


class TestMassSolve:
    def test_solves_vectors_and_blocks(self, bkd: Backend[Array]) -> None:
        """A 1-D vector and a 2-D block, with GMRES, which takes the block
        a column at a time."""
        model = _model(bkd)
        system = matrix_free_system(model, nstates=2, solver=_gmres(bkd))
        mass = MassSolve(system, bkd).mass_matrix()
        matrix = bkd.asarray(_MASS)
        vector = bkd.asarray([1.0, -2.0])
        block = bkd.asarray([[1.0, 0.0, 3.0], [-2.0, 1.0, 0.5]])
        for rhs in (vector, block):
            bkd.assert_allclose(mass.solve(rhs), bkd.solve(matrix, rhs), rtol=1e-11)
            bkd.assert_allclose(
                mass.solve_transpose(rhs), bkd.solve(matrix.T, rhs), rtol=1e-11
            )
            bkd.assert_allclose(mass.apply(rhs), bkd.dot(matrix, rhs), rtol=1e-14)
            bkd.assert_allclose(
                mass.apply_transpose(rhs), bkd.dot(matrix.T, rhs), rtol=1e-14
            )
        bkd.assert_allclose(mass.as_matrix(), matrix)
        assert not mass.is_singular() and mass.zero_rows() == []

    def test_sets_the_mass_up_once(self, bkd: Backend[Array]) -> None:
        model = _model(bkd)
        solver = _CountingLU(bkd)
        stepper = HeunHVP(model, dense_system(model, solver=solver))
        assert solver.setups() == 0
        ctx = _ctx(bkd)
        stepper.bind(ctx)
        state = bkd.asarray([10.0, 5.0])
        stepper.linsolve(state, stepper(state))
        offdiag = stepper.off_diag_operator(ctx, state)
        offdiag.apply(bkd.eye(2))
        offdiag.apply_transpose(bkd.eye(2))
        stepper.adjoint_initial_condition(ctx, state, bkd.asarray([0.3, 0.7]))
        assert solver.setups() == 1

    def test_a_failed_solve_raises(self, bkd: Backend[Array]) -> None:
        model = _model(bkd)
        system = matrix_free_system(model, nstates=2, solver=_gmres(bkd, maxiter=1))
        mass = MassSolve(system, bkd).mass_matrix()
        with pytest.raises(SolveFailedError, match="relative residual"):
            mass.solve(bkd.asarray([1.0, -2.0]))

    def test_a_solver_without_transposes_is_refused(self, bkd: Backend[Array]) -> None:
        model = _model(bkd)
        mass = MassSolve(
            dense_system(model, solver=NoTransposeSolver(bkd)), bkd
        ).mass_matrix()
        rhs = bkd.asarray([1.0, -2.0])
        bkd.assert_allclose(mass.solve(rhs), bkd.solve(bkd.asarray(_MASS), rhs))
        with pytest.raises(TypeError, match="no transposed solve"):
            mass.solve_transpose(rhs)

    def test_adjoint_tiers_refuse_a_solver_without_transposes(
        self, bkd: Backend[Array]
    ) -> None:
        model = _model(bkd)
        system = dense_system(model, solver=NoTransposeSolver(bkd))
        with pytest.raises(TypeError, match="needs transposed solves"):
            ForwardEulerHVP(model, system)
        with pytest.raises(TypeError, match="needs transposed solves"):
            HeunHVP(model, system)
        # A forward-only stepper does not need them.
        HeunStepper(model, system)

    def test_refuses_a_non_system(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="MassSystemProtocol"):
            ForwardEulerHVP(_model(bkd), bkd.eye(2))  # type: ignore[arg-type]
