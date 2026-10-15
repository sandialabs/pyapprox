"""Guarantees of exact projection ROMs, run against every test problem.

The problems come from ``tests/_helpers/rom_problems.py`` through the
``rom_problem`` and ``affine_rom_problem`` fixtures; each declares
whether it is affine (state and parameters) and coercive. None of these
tests needs a known PDE solution, only the discrete system. Each
guarantee has a scope:

- Root (every problem, any nonlinearity): a training snapshot's
  projection is a root of the exact reduced residual whenever the trial
  space holds the snapshot, because F vanishes on the unconstrained rows
  at the snapshot and the trial space never reads the constrained ones.
  Solver-independent.
- Reproduction and full-space limit (every problem): Newton on the
  reduced residual, started at zero codes (the snapshot mean), returns
  the full-order solution: at training parameters for a trial space
  holding every snapshot, and at any parameter for one holding every
  unconstrained DOF. Assumes the reduced root is unique near the start,
  true for these problems.
- Quasi-optimality (coercive problems): the reduced error is a bounded
  multiple of the best approximation error, never below it.
- Affine agreement, reproduction and limit (affine problems): the affine
  fit equals the exact projection off the training set.

Hyper-reduced residuals are approximations and get their own tier
(agreement within the fit tolerance; exactness only with unit weights);
curved trial spaces have no full-space limit.
"""

from typing import List

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.rom import (
    AgreementCheck,
    LinearTrialSpace,
    ProjectedResidual,
    ReducedResidualProtocol,
    fit_affine_projection,
    fit_pod_trial_space,
    full_state,
    projection_errors,
    solve_reduced_steady,
)
from pyapprox.surrogates.kle import as_basis_operator
from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.rom_problems import RomProblem

# Measured max ratio 1.85 (rank 3) for the linear ADR problem over ranks
# 1-6 and its held-out samples; the bound leaves headroom without hiding
# a broken projection.
_QUASI_OPTIMALITY_BOUND = 3.0


def _free_dof_trial_space(problem: RomProblem) -> LinearTrialSpace[NumpyArray]:
    """Every unconstrained DOF as a basis vector, zero reference."""
    bkd = problem.bkd
    nstates = problem.nstates()
    constrained = {int(dof) for dof in bkd.to_numpy(problem.constraint_set.dofs())}
    free = [ii for ii in range(nstates) if ii not in constrained]
    return LinearTrialSpace(
        as_basis_operator(bkd.asarray(np.eye(nstates)[:, free]), bkd),
        bkd.zeros((nstates,)),
    )


def _full_pod(problem: RomProblem) -> LinearTrialSpace[NumpyArray]:
    homogenized = problem.homogenized(problem.train)
    return fit_pod_trial_space(homogenized, problem.bkd, metric=problem.metric)


def _assert_reproduces(
    problem: RomProblem,
    reduced: ReducedResidualProtocol[NumpyArray],
    trial: LinearTrialSpace[NumpyArray],
    samples: NumpyArray,
) -> None:
    truth = problem.solve(samples).states
    for jj in range(int(samples.shape[1])):
        codes = solve_reduced_steady(
            reduced, samples[:, jj], problem.bkd.zeros((trial.ncodes(),))
        )
        problem.bkd.assert_allclose(
            full_state(trial, problem.lift, codes), truth[:, jj], atol=1e-10
        )


def _quasi_optimality_ratios(problem: RomProblem, affine: bool) -> List[float]:
    bkd = problem.bkd
    homogenized = problem.homogenized(problem.train)
    truth = problem.homogenized(problem.held_out)
    ratios: List[float] = []
    for rank in range(1, 7):
        trial = fit_pod_trial_space(
            homogenized, bkd, latent_dim=rank, metric=problem.metric
        )
        reduced: ReducedResidualProtocol[NumpyArray] = (
            fit_affine_projection(
                problem.residual, trial, problem.lift, problem.reference_params
            )
            if affine
            else ProjectedResidual(problem.residual, trial, problem.lift)
        )
        best = projection_errors(trial, truth.states)
        for jj in range(int(truth.states.shape[1])):
            codes = solve_reduced_steady(
                reduced, problem.held_out[:, jj], bkd.zeros((rank,))
            )
            rom = trial.decode(codes[:, None]) - truth.states[:, jj : jj + 1]
            rom_error = bkd.to_float(problem.metric.norm(rom)[0])
            ratios.append(rom_error / bkd.to_float(best[jj]))
    return ratios


class TestEveryProblem:
    def test_projected_snapshot_is_a_root(self, rom_problem: RomProblem) -> None:
        problem = rom_problem
        homogenized = problem.homogenized(problem.train)
        trial = _full_pod(problem)
        projected = ProjectedResidual(problem.residual, trial, problem.lift)
        codes = trial.project(homogenized.states)
        for jj in range(int(codes.shape[1])):
            params = problem.train[:, jj]
            residual = projected.residual(codes[:, jj], 0.0, params)
            jacobian = projected.state_jacobian(codes[:, jj], 0.0, params)
            # Relative to the operator's scale: snapshots lie in the span
            # only to the POD rank floor, and the residual magnifies that
            # leftover by roughly 1/h^2. Measured at most 3.7e-13 (linear),
            # 5.6e-17 (u^2 reaction) and 3.2e-12 (exp kappa).
            assert float(np.max(np.abs(residual))) <= 1e-10 * float(
                np.max(np.abs(jacobian))
            )

    def test_reproduces_training_solutions(self, rom_problem: RomProblem) -> None:
        trial = _full_pod(rom_problem)
        projected = ProjectedResidual(rom_problem.residual, trial, rom_problem.lift)
        _assert_reproduces(rom_problem, projected, trial, rom_problem.train)

    def test_full_space_limit(self, rom_problem: RomProblem) -> None:
        trial = _free_dof_trial_space(rom_problem)
        projected = ProjectedResidual(rom_problem.residual, trial, rom_problem.lift)
        _assert_reproduces(rom_problem, projected, trial, rom_problem.held_out)

    def test_quasi_optimality(self, rom_problem: RomProblem) -> None:
        if not rom_problem.coercive:
            pytest.skip(f"{rom_problem.name} is not declared coercive")
        ratios = _quasi_optimality_ratios(rom_problem, affine=False)
        assert min(ratios) >= 1.0 - 1e-8
        assert max(ratios) <= _QUASI_OPTIMALITY_BOUND


class TestAffineProblems:
    def test_reproduces_training_solutions(
        self, affine_rom_problem: RomProblem
    ) -> None:
        problem = affine_rom_problem
        trial = _full_pod(problem)
        fitted = fit_affine_projection(
            problem.residual, trial, problem.lift, problem.reference_params
        )
        _assert_reproduces(problem, fitted, trial, problem.train)

    def test_full_space_limit(self, affine_rom_problem: RomProblem) -> None:
        problem = affine_rom_problem
        trial = _free_dof_trial_space(problem)
        fitted = fit_affine_projection(
            problem.residual, trial, problem.lift, problem.reference_params
        )
        _assert_reproduces(problem, fitted, trial, problem.held_out)

    def test_agrees_with_exact_projection_off_training_set(
        self, affine_rom_problem: RomProblem
    ) -> None:
        problem = affine_rom_problem
        bkd = problem.bkd
        homogenized = problem.homogenized(problem.train)
        trial = fit_pod_trial_space(
            homogenized, bkd, latent_dim=4, metric=problem.metric
        )
        fitted = fit_affine_projection(
            problem.residual, trial, problem.lift, problem.reference_params
        )
        exact = ProjectedResidual(problem.residual, trial, problem.lift)
        codes = bkd.asarray(np.random.default_rng(2).normal(size=(4, 3)))
        # Away from the offset, so the check covers T z as well as G.
        AgreementCheck(problem.held_out, codes=codes, rtol=1e-10).check(fitted, exact)
        fitted_jac = fitted.derivatives().param_jacobian
        exact_jac = exact.derivatives().param_jacobian
        assert fitted_jac is not None and exact_jac is not None
        for jj in range(3):
            truth = exact_jac(codes[:, jj], 0.0, problem.held_out[:, jj])
            bkd.assert_allclose(
                fitted_jac(codes[:, jj], 0.0, problem.held_out[:, jj]),
                truth,
                rtol=1e-10,
                atol=1e-10 * float(bkd.to_float(bkd.max(bkd.abs(truth)))),
            )

    def test_quasi_optimality(self, affine_rom_problem: RomProblem) -> None:
        if not affine_rom_problem.coercive:
            pytest.skip(f"{affine_rom_problem.name} is not declared coercive")
        ratios = _quasi_optimality_ratios(affine_rom_problem, affine=True)
        assert min(ratios) >= 1.0 - 1e-8
        assert max(ratios) <= _QUASI_OPTIMALITY_BOUND
