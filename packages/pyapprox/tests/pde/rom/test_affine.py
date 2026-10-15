"""What is specific to the affine fit: its refusals, its controls, and an
analytic anchor.

The guarantees the affine fit shares with every exact projection
(reproduction, full-space limit, agreement off the training set,
quasi-optimality) run in ``test_conformance.py`` over the shared
problems of ``tests/_helpers/rom_problems.py``. Here:

- Refusals: a nonlinear parameterization (exp kappa), a state
  nonlinearity (u^2), time-dependent forcing, and a curved trial space.
- False affinity: a parameterization that claims affinity over the exp
  map passes the declaration checks, and AgreementCheck against the
  exact projection catches it.
- Reproduction controls: a lift without the boundary data, or a trial
  space missing the snapshot, breaks reproduction.
- Analytic anchor: -u'' = 2p, u(0) = 1, u(1) = 2 has u = 1 + x + p x (1-x)
  for every p, and 1D P1 Galerkin with constant coefficient and exact
  load is exact at the nodes, so a trial space written down analytically
  (reference 1 + x, mode x (1-x)) must give the exact PDE solution with
  no snapshots; the wrong mode x^2 (1-x) cannot.
"""

from typing import Any, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldForcing,
    TimeDependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.models.galerkin import GalerkinParameterizedResidual
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.pde.rom import (
    AffineProjectedResidual,
    AgreementCheck,
    AlgebraicLift,
    LinearTrialSpace,
    ProjectedResidual,
    SnapshotSet,
    TangentOperatorProtocol,
    fit_affine_projection,
    fit_pod_trial_space,
    full_state,
    solve_reduced_steady,
)
from pyapprox.surrogates.kle import as_basis_operator
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.rom_problems import (
    RomProblem,
    galerkin_adr_linear,
    galerkin_adr_nonlinear_params,
    galerkin_adr_nonlinear_state,
)

_Points = NDArray[np.floating[Any]]


def _pod(problem: RomProblem, rank: int) -> LinearTrialSpace[NumpyArray]:
    homogenized = problem.homogenized(problem.train)
    return fit_pod_trial_space(
        homogenized, problem.bkd, latent_dim=rank, metric=problem.metric
    )


def _max_error(problem: RomProblem, rom_state: NumpyArray, params: NumpyArray) -> float:
    truth = problem.solve(params[:, None]).states[:, 0]
    return float(np.max(np.abs(rom_state - truth)))


class _ClaimsAffine:
    """Delegates to a parameterization but claims affinity regardless."""

    def __init__(self, inner: ParameterizationProtocol[NumpyArray]) -> None:
        self._inner = inner

    def nparams(self) -> int:
        return self._inner.nparams()

    def targets(self) -> Tuple[object, ...]:
        return self._inner.targets()

    def owned_coefficients(self) -> Tuple[str, ...]:
        return self._inner.owned_coefficients()

    def apply(self, params_1d: NumpyArray) -> None:
        self._inner.apply(params_1d)

    def param_derivatives(self) -> ParamDerivatives[NumpyArray]:
        return self._inner.param_derivatives()

    def is_affine_in_params(self) -> bool:
        return True


class _ZeroLift:
    """A lift without the boundary data: the wrong-model control."""

    def __init__(self, nstates: int, bkd: NumpyBkd) -> None:
        self._nstates = nstates
        self._bkd = bkd

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nstates(self) -> int:
        return self._nstates

    def is_time_invariant(self) -> bool:
        return True

    def lift(self, time: float) -> NumpyArray:
        return self._bkd.zeros((self._nstates,))


class _CurvedTrialSpace:
    """Satisfies TrialSpaceProtocol but not LinearTrialSpaceProtocol."""

    def __init__(self, nstates: int, bkd: NumpyBkd) -> None:
        self._nstates = nstates
        self._bkd = bkd

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nstates(self) -> int:
        return self._nstates

    def ncodes(self) -> int:
        return 1

    def decode(self, codes: NumpyArray) -> NumpyArray:
        return self._bkd.zeros((self._nstates, int(codes.shape[1])))

    def tangent(self, code: NumpyArray) -> TangentOperatorProtocol[NumpyArray]:
        return as_basis_operator(self._bkd.zeros((self._nstates, 1)), self._bkd)

    def weighted_curvature(self, code: NumpyArray, weights: NumpyArray) -> NumpyArray:
        return self._bkd.zeros((1, 1))


def _pulsing(coords: _Points, time: float) -> _Points:
    return np.full(coords.shape[1], 1.0 + time)


def _time_dependent_forcing_residual(
    bkd: NumpyBkd,
) -> Tuple[GalerkinParameterizedResidual[NumpyArray], AlgebraicLift[NumpyArray]]:
    """Linear diffusivity map, forcing (1 + t): affine but time-dependent."""
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        forcing=TimeDependent(_pulsing),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 1.0, bkd),
            DirichletBC(basis, "right", 2.0, bkd),
        ],
    )
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    diffusivity_map = MeshKLEFieldMap(
        bkd,
        bkd.asarray(np.ones(coords.shape[0])),
        bkd.asarray(0.2 * np.sin(np.pi * coords)[:, None]),
    )
    param_obj = AdvectionDiffusionParameterization(
        physics, diffusivity_map=diffusivity_map, bkd=bkd
    )
    return (
        GalerkinParameterizedResidual(system.spatial_operator(), param_obj),
        AlgebraicLift(system.constraint_set(), system.nstates()),
    )


class TestAffineProjectedResidual:
    def test_rejects_wrong_shapes(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="coupling"):
            AffineProjectedResidual(
                bkd.zeros((2,)),
                bkd.zeros((2, 2)),
                bkd.zeros((2, 3)),
                bkd.zeros((2, 3, 3)),
                bkd.zeros((3,)),
                bkd,
            )


class TestRefusals:
    def test_nonlinear_parameterization(self, numpy_bkd: NumpyBkd) -> None:
        trial = _pod(galerkin_adr_linear(numpy_bkd), rank=3)
        problem = galerkin_adr_nonlinear_params(numpy_bkd)
        with pytest.raises(ValueError, match="affine in the parameters"):
            fit_affine_projection(
                problem.residual, trial, problem.lift, problem.reference_params
            )

    def test_nonlinear_state(self, numpy_bkd: NumpyBkd) -> None:
        trial = _pod(galerkin_adr_linear(numpy_bkd), rank=3)
        problem = galerkin_adr_nonlinear_state(numpy_bkd)
        with pytest.raises(ValueError, match="affine in the state"):
            fit_affine_projection(
                problem.residual, trial, problem.lift, problem.reference_params
            )

    def test_time_dependent_forcing(self, numpy_bkd: NumpyBkd) -> None:
        trial = _pod(galerkin_adr_linear(numpy_bkd), rank=3)
        residual, lift = _time_dependent_forcing_residual(numpy_bkd)
        with pytest.raises(ValueError, match="time-invariant"):
            fit_affine_projection(residual, trial, lift, numpy_bkd.zeros((1,)))

    def test_curved_trial_space(self, numpy_bkd: NumpyBkd) -> None:
        problem = galerkin_adr_linear(numpy_bkd)
        curved = _CurvedTrialSpace(problem.nstates(), numpy_bkd)
        # Passes a curved trial space on purpose to exercise the runtime
        # guard for callers that are not type-checked. mypy rejects this
        # call (arg-type), which is the point: left unsuppressed.
        with pytest.raises(TypeError, match="LinearTrialSpaceProtocol"):
            fit_affine_projection(
                problem.residual, curved, problem.lift, problem.reference_params
            )


class TestFalseAffinity:
    def test_agreement_check_catches_it(self, numpy_bkd: NumpyBkd) -> None:
        trial = _pod(galerkin_adr_linear(numpy_bkd), rank=4)
        problem = galerkin_adr_nonlinear_params(numpy_bkd)
        liar = GalerkinParameterizedResidual(
            problem.residual.spatial_operator(),
            _ClaimsAffine(problem.residual.parameterization()),
        )
        fitted = fit_affine_projection(
            liar, trial, problem.lift, problem.reference_params
        )
        with pytest.raises(ValueError, match="departs"):
            AgreementCheck(problem.held_out).check(
                fitted, ProjectedResidual(liar, trial, problem.lift)
            )


class TestReproductionControls:
    def test_lift_without_boundary_data(self, numpy_bkd: NumpyBkd) -> None:
        problem = galerkin_adr_linear(numpy_bkd)
        trial = _pod(problem, rank=9)
        wrong = fit_affine_projection(
            problem.residual,
            trial,
            _ZeroLift(problem.nstates(), numpy_bkd),
            problem.reference_params,
        )
        params = problem.train[:, 0]
        codes = solve_reduced_steady(wrong, params, numpy_bkd.zeros((9,)))
        rom_state = full_state(trial, problem.lift, codes)
        assert _max_error(problem, rom_state, params) > 1e-3

    def test_trial_space_missing_the_snapshot(self, numpy_bkd: NumpyBkd) -> None:
        problem = galerkin_adr_linear(numpy_bkd)
        homogenized = problem.homogenized(problem.train)
        others = SnapshotSet(homogenized.states[:, 1:], homogenized.params[:, 1:])
        trial = fit_pod_trial_space(
            others, numpy_bkd, latent_dim=2, metric=problem.metric
        )
        fitted = fit_affine_projection(
            problem.residual, trial, problem.lift, problem.reference_params
        )
        params = problem.train[:, 0]
        codes = solve_reduced_steady(fitted, params, numpy_bkd.zeros((2,)))
        rom_state = full_state(trial, problem.lift, codes)
        assert _max_error(problem, rom_state, params) > 1e-6


class TestAnalyticAnchor:
    """-u'' = 2p, u(0) = 1, u(1) = 2: u = 1 + x + p x (1 - x)."""

    def _rom_and_exact(
        self, bkd: NumpyBkd, wrong_mode: bool, param: float
    ) -> Tuple[NumpyArray, NumpyArray]:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=NodalFieldForcing(basis),
        )
        system = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 1.0, bkd),
                DirichletBC(basis, "right", 2.0, bkd),
            ],
        )
        x = bkd.to_numpy(basis.dof_coordinates())[0]
        forcing_map = MeshKLEFieldMap(
            bkd,
            bkd.asarray(np.zeros(x.shape[0])),
            bkd.asarray(np.full((x.shape[0], 1), 2.0)),
        )
        param_obj = AdvectionDiffusionParameterization(
            physics, forcing_map=forcing_map, bkd=bkd
        )
        residual = GalerkinParameterizedResidual(system.spatial_operator(), param_obj)
        lift = AlgebraicLift(system.constraint_set(), system.nstates())
        mode = x**2 * (1.0 - x) if wrong_mode else x * (1.0 - x)
        trial = LinearTrialSpace(
            as_basis_operator(bkd.asarray(mode[:, None]), bkd),
            bkd.asarray(1.0 + x) - lift.lift(0.0),
        )
        fitted = fit_affine_projection(residual, trial, lift, bkd.zeros((1,)))
        codes = solve_reduced_steady(fitted, bkd.asarray([param]), bkd.zeros((1,)))
        exact: NumpyArray = 1.0 + x + param * x * (1.0 - x)
        return full_state(trial, lift, codes), exact

    @pytest.mark.parametrize("param", [-1.5, 0.3, 2.0])
    def test_exact_mode_gives_the_pde_solution(
        self, numpy_bkd: NumpyBkd, param: float
    ) -> None:
        rom, exact = self._rom_and_exact(numpy_bkd, wrong_mode=False, param=param)
        numpy_bkd.assert_allclose(rom, exact, atol=1e-12)

    def test_wrong_mode_cannot(self, numpy_bkd: NumpyBkd) -> None:
        rom, exact = self._rom_and_exact(numpy_bkd, wrong_mode=True, param=2.0)
        assert float(np.max(np.abs(rom - exact))) > 1e-2
