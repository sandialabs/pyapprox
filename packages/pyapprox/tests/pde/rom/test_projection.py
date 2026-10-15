"""The exact projected residual, the reference for every reduced residual.

Setup: -(kappa u')' + u^2 = 1 on [0, 1] (the reaction makes F nonlinear
in u), u(0) = 1, u(1) = 2, P1 on 10 cells, kappa = exp(W p) with three
smooth modes. Steady snapshots at 12 parameter samples, homogenized by
the algebraic lift; POD in the assembled mass metric.

Reproduction (derivation-free): when the trial space contains a
training snapshot, the reduced residual vanishes at that snapshot's
projection, because the raw residual vanishes on every unconstrained
row and the trial space is zero on the constrained ones. It holds only
if the raw residual, the lift and the Galerkin test space are all
right. Wrong-model controls: dropping the lift, or a trial space that
does not contain the snapshot, leaves a residual many orders larger.

Derivatives: DerivativeChecker on the state Jacobian (linear trial
space, and a test-local quadratic decoder where the curvature term
matters) and on the parameter Jacobian. Control: the quadratic decoder
with its curvature dropped fails the state-Jacobian check.
"""

from typing import Any

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    TimeIndependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.models.galerkin import GalerkinParameterizedResidual
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.rom import (
    AlgebraicLift,
    LinearTrialSpace,
    LinearTrialSpaceProtocol,
    ProjectedResidual,
    ReducedResidualProtocol,
    SnapshotSet,
    TangentOperatorProtocol,
    TrialSpaceProtocol,
    collect_steady_snapshots,
    fit_pod_trial_space,
    homogenize,
)
from pyapprox.surrogates.kle import as_basis_operator
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.inner_product import MassInnerProduct
from tests._helpers.adjoint_checks import NumpyArray

_NSAMPLES = 12


def _unit_forcing(coords: Any) -> Any:
    return np.ones(coords.shape[1])


def _square(x: Any, u: Any) -> Any:
    return u**2


def _square_deriv(x: Any, u: Any) -> Any:
    return 2.0 * u


def _square_deriv2(x: Any, u: Any) -> Any:
    return np.full_like(u, 2.0)


class _Problem:
    """The steady problem, its homogenized snapshots and its lift."""

    def __init__(self, bkd: NumpyBkd) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            reaction=CallableReaction(_square, _square_deriv, _square_deriv2),
            forcing=TimeIndependent(_unit_forcing),
        )
        system = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 1.0, bkd),
                DirichletBC(basis, "right", 2.0, bkd),
            ],
        )
        coords = bkd.to_numpy(basis.dof_coordinates())[0]
        modes = np.stack(
            [0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1) for k in range(3)],
            axis=1,
        )
        kle = MeshKLEFieldMap(
            bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
        )
        exp = _ExpTransform(bkd)
        param_obj = AdvectionDiffusionParameterization(
            physics,
            diffusivity_map=TransformedFieldMap(
                kle, exp, exp, bkd, transform_deriv2=exp
            ),
            bkd=bkd,
        )
        state_eq = GalerkinParameterizedStateEquation(system.steady(), param_obj, bkd)
        samples = np.random.default_rng(5).uniform(-0.5, 0.5, size=(3, _NSAMPLES))
        snapshots = collect_steady_snapshots(
            state_eq, bkd.asarray(samples), bkd.zeros((system.nstates(), 1))
        )
        constraint_set = system.constraint_set()
        self.lift = AlgebraicLift(constraint_set, system.nstates())
        self.homogenized = homogenize(snapshots, self.lift, constraint_set)
        self.metric = MassInnerProduct(physics.mass_matrix(), bkd)
        self.residual = GalerkinParameterizedResidual(
            system.spatial_operator(), param_obj
        )
        self.bkd = bkd

    def full_trial_space(self) -> LinearTrialSpace[NumpyArray]:
        return fit_pod_trial_space(self.homogenized, self.bkd, metric=self.metric)


class _ZeroLift:
    """A lift that carries no boundary data: the wrong-model control."""

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


class _QuadraticTrialSpace:
    r"""phi(z) = u_bar + V z + W (z kron z), test-local.

    J(z) = V + W (I kron z + z kron I); the curvature of w^T phi is
    S + S^T with S = reshape(W^T w, (r, r)). ``drop_curvature`` returns
    zeros instead, the wrong-model control.
    """

    def __init__(
        self,
        reference: NumpyArray,
        linear: NumpyArray,
        quadratic: NumpyArray,
        bkd: NumpyBkd,
        drop_curvature: bool = False,
    ) -> None:
        self._reference = reference
        self._linear = linear
        self._quadratic = quadratic
        self._bkd = bkd
        self._drop_curvature = drop_curvature

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nstates(self) -> int:
        return int(self._linear.shape[0])

    def ncodes(self) -> int:
        return int(self._linear.shape[1])

    def decode(self, codes: NumpyArray) -> NumpyArray:
        kron = np.stack([np.kron(c, c) for c in codes.T], axis=1)
        return self._reference[:, None] + self._linear @ codes + self._quadratic @ kron

    def tangent(self, code: NumpyArray) -> TangentOperatorProtocol[NumpyArray]:
        eye = np.eye(self.ncodes())
        dkron = np.kron(eye, code[:, None]) + np.kron(code[:, None], eye)
        return as_basis_operator(self._linear + self._quadratic @ dkron, self._bkd)

    def weighted_curvature(self, code: NumpyArray, weights: NumpyArray) -> NumpyArray:
        ncodes = self.ncodes()
        if self._drop_curvature:
            return np.zeros((ncodes, ncodes))
        half = (self._quadratic.T @ weights).reshape(ncodes, ncodes)
        return half + half.T


def _residual_norm(
    projected: ReducedResidualProtocol[NumpyArray],
    codes: NumpyArray,
    params: NumpyArray,
) -> float:
    return float(np.linalg.norm(projected.residual(codes, 0.0, params)))


class TestProtocols:
    def test_linear_trial_space_satisfies_both(self, numpy_bkd: NumpyBkd) -> None:
        trial = _Problem(numpy_bkd).full_trial_space()
        assert isinstance(trial, TrialSpaceProtocol)
        assert isinstance(trial, LinearTrialSpaceProtocol)
        code = np.zeros(trial.ncodes())
        assert trial.tangent(code) is trial.basis()
        numpy_bkd.assert_allclose(
            trial.weighted_curvature(code, np.ones(trial.nstates())),
            np.zeros((trial.ncodes(), trial.ncodes())),
        )

    def test_curved_trial_space_is_not_linear(self, numpy_bkd: NumpyBkd) -> None:
        curved = _QuadraticTrialSpace(
            np.zeros(5), np.ones((5, 2)), np.ones((5, 4)), numpy_bkd
        )
        assert isinstance(curved, TrialSpaceProtocol)
        assert not isinstance(curved, LinearTrialSpaceProtocol)

    def test_projected_residual_satisfies_protocol(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        projected = ProjectedResidual(
            problem.residual, problem.full_trial_space(), problem.lift
        )
        assert isinstance(projected, ReducedResidualProtocol)
        assert projected.derivatives().param_jacobian is not None

    def test_rejects_mismatched_sizes(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        with pytest.raises(ValueError, match="states"):
            ProjectedResidual(
                problem.residual,
                problem.full_trial_space(),
                _ZeroLift(3, numpy_bkd),
            )


class TestReproduction:
    def test_residual_vanishes_at_projected_snapshot(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        trial = problem.full_trial_space()
        projected = ProjectedResidual(problem.residual, trial, problem.lift)
        snapshots = problem.homogenized.states
        codes = trial.project(snapshots)
        # The snapshot is in the trial space, so decoding is exact.
        numpy_bkd.assert_allclose(trial.decode(codes), snapshots, atol=1e-12)
        for jj in range(_NSAMPLES):
            norm = _residual_norm(
                projected, codes[:, jj], problem.homogenized.params[:, jj]
            )
            assert norm < 1e-10

    def test_dropping_the_lift_breaks_it(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        trial = problem.full_trial_space()
        wrong = ProjectedResidual(
            problem.residual,
            trial,
            _ZeroLift(trial.nstates(), numpy_bkd),
        )
        codes = trial.project(problem.homogenized.states)
        norm = _residual_norm(wrong, codes[:, 0], problem.homogenized.params[:, 0])
        assert norm > 1e-2

    def test_trial_space_without_the_snapshot_breaks_it(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        problem = _Problem(numpy_bkd)
        others = SnapshotSet(
            problem.homogenized.states[:, 1:], problem.homogenized.params[:, 1:]
        )
        trial = fit_pod_trial_space(
            others, numpy_bkd, latent_dim=1, metric=problem.metric
        )
        projected = ProjectedResidual(problem.residual, trial, problem.lift)
        held_out = problem.homogenized.states[:, :1]
        codes = trial.project(held_out)
        norm = _residual_norm(projected, codes[:, 0], problem.homogenized.params[:, 0])
        assert norm > 1e-6


def _check_state_jacobian(
    projected: ProjectedResidual[NumpyArray],
    codes: NumpyArray,
    params: NumpyArray,
    bkd: NumpyBkd,
) -> float:
    ncodes = projected.ncodes()

    def fun(sample: NumpyArray) -> NumpyArray:
        return projected.residual(sample[:, 0], 0.0, params)[:, None]

    def jac(sample: NumpyArray) -> NumpyArray:
        return projected.state_jacobian(sample[:, 0], 0.0, params)

    checker = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=ncodes, nvars=ncodes, fun=fun, jacobian=jac, bkd=bkd
        )
    )
    # Central differences: with a linear trial space the reduced residual
    # is quadratic in the codes (the u^2 reaction), so forward differences
    # reach roundoff after a few steps and their min/max ratio lands
    # either side of 1e-6 from run to run; central differences are exact
    # for a quadratic and robust otherwise.
    errors = checker.check_derivatives(codes[:, None], verbosity=0, central=True)[0]
    return float(checker.error_ratio(errors))


class TestDerivatives:
    def test_state_jacobian_linear_trial_space(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        trial = fit_pod_trial_space(
            problem.homogenized, numpy_bkd, latent_dim=4, metric=problem.metric
        )
        projected = ProjectedResidual(problem.residual, trial, problem.lift)
        codes = np.random.default_rng(6).normal(size=4) * 0.1
        ratio = _check_state_jacobian(
            projected, codes, problem.homogenized.params[:, 0], numpy_bkd
        )
        assert ratio <= 1e-6

    def _curved(self, problem: _Problem, drop_curvature: bool) -> Any:
        trial = fit_pod_trial_space(
            problem.homogenized, problem.bkd, latent_dim=3, metric=problem.metric
        )
        rng = np.random.default_rng(7)
        quadratic = 0.3 * trial.apply(np.eye(3)) @ rng.normal(size=(3, 9))
        return _QuadraticTrialSpace(
            trial.reference(),
            trial.apply(np.eye(3)),
            quadratic,
            problem.bkd,
            drop_curvature=drop_curvature,
        )

    def test_state_jacobian_curved_trial_space(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        projected = ProjectedResidual(
            problem.residual, self._curved(problem, False), problem.lift
        )
        codes = np.array([0.3, -0.2, 0.1])
        ratio = _check_state_jacobian(
            projected, codes, problem.homogenized.params[:, 0], numpy_bkd
        )
        assert ratio <= 1e-6

    def test_dropping_curvature_fails_the_check(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        projected = ProjectedResidual(
            problem.residual, self._curved(problem, True), problem.lift
        )
        codes = np.array([0.3, -0.2, 0.1])
        ratio = _check_state_jacobian(
            projected, codes, problem.homogenized.params[:, 0], numpy_bkd
        )
        assert ratio > 1e-3

    def test_param_jacobian(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        problem = _Problem(bkd)
        trial = fit_pod_trial_space(
            problem.homogenized, bkd, latent_dim=4, metric=problem.metric
        )
        projected = ProjectedResidual(problem.residual, trial, problem.lift)
        param_jac = projected.derivatives().param_jacobian
        assert param_jac is not None
        codes = np.random.default_rng(8).normal(size=4) * 0.1

        def fun(sample: NumpyArray) -> NumpyArray:
            return projected.residual(codes, 0.0, sample[:, 0])[:, None]

        def jac(sample: NumpyArray) -> NumpyArray:
            return param_jac(codes, 0.0, sample[:, 0])

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=4, nvars=3, fun=fun, jacobian=jac, bkd=bkd
            )
        )
        errors = checker.check_derivatives(
            problem.homogenized.params[:, :1], verbosity=0, central=True
        )[0]
        assert float(checker.error_ratio(errors)) <= 1e-6
