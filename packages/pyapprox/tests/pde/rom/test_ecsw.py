"""ECSW hyper-reduction against the exact projection it approximates.

Problems (``elementwise_rom_problem``): the 1D ADR factories of
``tests/_helpers/rom_problems.py`` on 200 P1 cells (linear, u^2
reaction, exp kappa; skfem, so only when it is installed), 12 training
samples, POD rank 4 in the mass metric; and the closed-form mock P1
factories of ``tests/_helpers/mock_elementwise.py`` (linear, u^2
reaction; 200 cells, trapezoid metric, no assembly library), which pass
the same checks. Training targets: reduced residual and reduced Jacobian at the
projected training snapshots (240 rows by 200 element columns).
Weights from EarlyStoppingLawsonHanson.

Measured on these problems (ROM error is the max state error at the 4
held-out parameters, relative to the exact-Galerkin ROM's):
tolerance 1e-10 stops at the solver's roundoff floor (NNLS relative
residual ~1.6e-7) keeping 24-25 elements with the exact-Galerkin error;
1e-3 keeps 13-15 elements at 1.01-1.41x; 1e-2 keeps 12 but costs 7.9x
on the u^2 problem. The reduced Jacobian stays symmetric (to ~1e-12)
and -J coercive (min eigenvalue 7-11) at every tolerance.

Checks: unit weights over every element reproduce the exact projection;
a tight fit reproduces the exact-Galerkin ROM; a matched tolerance keeps
at most a tenth of the elements at no more than twice the error;
non-negative weights keep the reduced Jacobian symmetric and coercive;
the training matrix is reused across tolerances; DerivativeChecker on
the state Jacobian. Wrong-model controls: dropping a kept element or
flipping a weight's sign worsens agreement with the exact projection by
more than 10x.
"""

from typing import List, Tuple

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.elementwise import ElementSet
from pyapprox.pde.rom import (
    ElementSampledResidual,
    JacobianTargets,
    LinearTrialSpace,
    ProjectedResidual,
    ReducedResidualProtocol,
    ResidualTargets,
    build_ecsw,
    ecsw_training_matrix,
    fit_ecsw,
    fit_pod_trial_space,
    full_state,
    solve_reduced_steady,
    unit_weight_residual,
)
from pyapprox.util.linalg.nnls import EarlyStoppingLawsonHanson, NNLSResult
from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.rom_problem import RomProblem

_RANK = 4


class _Setup:
    """A problem with its trial space, exact projection and training data."""

    def __init__(self, problem: RomProblem) -> None:
        bkd = problem.bkd
        homogenized = problem.homogenized(problem.train)
        self.problem = problem
        self.parameterization = problem.residual.parameterization()
        self.trial: LinearTrialSpace[NumpyArray] = fit_pod_trial_space(
            homogenized, bkd, latent_dim=_RANK, metric=problem.metric
        )
        self.exact = ProjectedResidual(problem.residual, self.trial, problem.lift)
        assert problem.elementwise is not None
        self.elementwise = problem.elementwise
        self.full = unit_weight_residual(
            self.elementwise, self.parameterization, self.trial, problem.lift
        )
        self.codes = self.trial.project(homogenized.states)
        self.matrix = ecsw_training_matrix(
            self.full,
            self.codes,
            problem.train,
            (ResidualTargets(), JacobianTargets()),
        )

    def weights(self, tolerance: float) -> NNLSResult[NumpyArray]:
        solver = EarlyStoppingLawsonHanson(self.problem.bkd, tolerance)
        return fit_ecsw(self.matrix, solver, self.problem.bkd)

    def build(
        self, weights: NNLSResult[NumpyArray]
    ) -> ElementSampledResidual[NumpyArray]:
        return build_ecsw(
            self.elementwise,
            self.parameterization,
            self.trial,
            self.problem.lift,
            weights,
        ).residual

    def agreement(self, reduced: ReducedResidualProtocol[NumpyArray]) -> float:
        """Worst residual gap to the exact projection on training data,
        relative to the largest exact residual entry."""
        gaps: List[float] = []
        scales: List[float] = []
        for kk in range(int(self.codes.shape[1])):
            code, param = self.codes[:, kk], self.problem.train[:, kk]
            exact = self.exact.residual(code, 0.0, param)
            gaps.append(
                float(np.max(np.abs(reduced.residual(code, 0.0, param) - exact)))
            )
            scales.append(float(np.max(np.abs(exact))))
        return max(gaps) / max(scales)

    def rom_error(self, reduced: ReducedResidualProtocol[NumpyArray]) -> float:
        """Max state error at the held-out parameters."""
        bkd = self.problem.bkd
        truth = self.problem.solve(self.problem.held_out).states
        errors: List[float] = []
        for jj in range(int(truth.shape[1])):
            codes = solve_reduced_steady(
                reduced, self.problem.held_out[:, jj], bkd.zeros((_RANK,))
            )
            state = full_state(self.trial, self.problem.lift, codes)
            errors.append(float(np.max(np.abs(state - truth[:, jj]))))
        return max(errors)

    def with_weights(
        self, indices: List[int], weights: NumpyArray
    ) -> ElementSampledResidual[NumpyArray]:
        bkd = self.problem.bkd
        elements = ElementSet(bkd.asarray(indices, dtype=bkd.int64_dtype()), bkd)
        return ElementSampledResidual.with_weights(
            self.elementwise.restrict(elements),
            self.parameterization,
            self.trial,
            self.problem.lift,
            weights,
        )


def _kept(weights: NNLSResult[NumpyArray]) -> Tuple[List[int], NumpyArray]:
    values = np.asarray(weights.weights)
    indices = [int(e) for e in np.flatnonzero(values > 0.0)]
    return indices, values[indices]


class TestTrainingMatrix:
    def test_row_counts(self, elementwise_rom_problem: RomProblem) -> None:
        setup = _Setup(elementwise_rom_problem)
        ntrain = int(setup.codes.shape[1])
        nelements = setup.elementwise.nelements()
        assert setup.matrix.shape == (ntrain * (_RANK + _RANK**2), nelements)

    def test_unit_weights_reproduce_exact_projection(
        self, elementwise_rom_problem: RomProblem
    ) -> None:
        setup = _Setup(elementwise_rom_problem)
        rng = np.random.default_rng(3)
        for jj in range(3):
            code = rng.normal(size=_RANK) * 0.1
            param = elementwise_rom_problem.held_out[:, jj]
            exact_res = setup.exact.residual(code, 0.0, param)
            exact_jac = setup.exact.state_jacobian(code, 0.0, param)
            elementwise_rom_problem.bkd.assert_allclose(
                setup.full.residual(code, 0.0, param),
                exact_res,
                atol=1e-10 * float(np.max(np.abs(exact_res))),
            )
            elementwise_rom_problem.bkd.assert_allclose(
                setup.full.state_jacobian(code, 0.0, param),
                exact_jac,
                atol=1e-10 * float(np.max(np.abs(exact_jac))),
            )


class TestFit:
    def test_tight_fit_matches_exact_galerkin(
        self, elementwise_rom_problem: RomProblem
    ) -> None:
        setup = _Setup(elementwise_rom_problem)
        ecsw = setup.build(setup.weights(1e-10))
        exact_error = setup.rom_error(setup.exact)
        assert abs(setup.rom_error(ecsw) - exact_error) <= 0.01 * exact_error

    def test_matched_tolerance_is_sparse_and_accurate(
        self, elementwise_rom_problem: RomProblem
    ) -> None:
        setup = _Setup(elementwise_rom_problem)
        weights = setup.weights(1e-3)
        indices, _ = _kept(weights)
        assert len(indices) <= setup.elementwise.nelements() // 10
        ratio = setup.rom_error(setup.build(weights)) / setup.rom_error(setup.exact)
        assert ratio <= 2.0

    def test_training_matrix_reused_across_tolerances(
        self, elementwise_rom_problem: RomProblem
    ) -> None:
        setup = _Setup(elementwise_rom_problem)
        tight, loose = setup.weights(1e-6), setup.weights(1e-3)
        assert len(_kept(loose)[0]) <= len(_kept(tight)[0])

    def test_weights_keep_jacobian_symmetric_and_coercive(
        self, elementwise_rom_problem: RomProblem
    ) -> None:
        setup = _Setup(elementwise_rom_problem)
        weights = setup.weights(1e-3)
        assert float(np.min(np.asarray(weights.weights))) >= 0.0
        ecsw = setup.build(weights)
        jacobian = np.asarray(
            ecsw.state_jacobian(
                setup.codes[:, 0], 0.0, elementwise_rom_problem.train[:, 0]
            )
        )
        scale = float(np.max(np.abs(jacobian)))
        assert float(np.max(np.abs(jacobian - jacobian.T))) <= 1e-10 * scale
        assert float(np.min(np.linalg.eigvalsh(-jacobian))) > 0.0


class TestControls:
    def test_dropping_a_kept_element(self, elementwise_rom_problem: RomProblem) -> None:
        setup = _Setup(elementwise_rom_problem)
        indices, values = _kept(setup.weights(1e-6))
        fitted = setup.agreement(setup.with_weights(indices, values))
        heaviest = int(np.argmax(values))
        dropped = setup.agreement(
            setup.with_weights(
                indices[:heaviest] + indices[heaviest + 1 :],
                np.delete(values, heaviest),
            )
        )
        assert dropped > 10.0 * fitted

    def test_flipping_a_weight_sign(self, elementwise_rom_problem: RomProblem) -> None:
        setup = _Setup(elementwise_rom_problem)
        indices, values = _kept(setup.weights(1e-6))
        fitted = setup.agreement(setup.with_weights(indices, values))
        flipped_values = values.copy()
        flipped_values[int(np.argmax(values))] *= -1.0
        flipped = setup.agreement(setup.with_weights(indices, flipped_values))
        assert flipped > 10.0 * fitted


class TestDerivatives:
    def test_state_jacobian(self, elementwise_rom_problem: RomProblem) -> None:
        setup = _Setup(elementwise_rom_problem)
        ecsw = setup.build(setup.weights(1e-6))
        param = elementwise_rom_problem.held_out[:, 0]
        bkd = elementwise_rom_problem.bkd

        def fun(sample: NumpyArray) -> NumpyArray:
            return ecsw.residual(sample[:, 0], 0.0, param)[:, None]

        def jac(sample: NumpyArray) -> NumpyArray:
            return ecsw.state_jacobian(sample[:, 0], 0.0, param)

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=_RANK, nvars=_RANK, fun=fun, jacobian=jac, bkd=bkd
            )
        )
        code = np.random.default_rng(4).normal(size=(_RANK, 1)) * 0.1
        errors = checker.check_derivatives(code, verbosity=0, central=True)[0]
        assert float(checker.error_ratio(errors)) <= 1e-6
