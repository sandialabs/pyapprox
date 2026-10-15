"""POD trial spaces and their truncation diagnostics.

The check is the POD optimality identity, which no step of the fit
computes: for a basis of the r leading modes of centered snapshots in
the metric M,

    sum_j ||s_j - Pi_r s_j||_M^2 = sum_{i > r} sigma_i^2,

with sigma the M-weighted singular values of the same centered
snapshots, and V^T M V = I. Synthetic setup: 20 states, 8 snapshots of
rank 5 plus a small full-rank perturbation, positive diagonal metric,
both backends. Galerkin setup: homogenized steady snapshots of
-(kappa u')' = 1, u(0)=1, u(1)=2, P1 on 10 cells, kappa = exp(W p) with
three modes, at 12 parameter samples, in the assembled mass metric.

Wrong-model control: a basis fitted in the Euclidean metric but used in
a diagonal metric with weights spread over two decades (0.1 to 10)
loses the identity (its error exceeds the M-POD tail) and its Gram is
not the identity. The control uses the synthetic metric because on a
uniform mesh the mass matrix is nearly a multiple of the identity, and
Euclidean and mass-weighted POD agree to about 1e-5 relative there.
"""

from typing import Any, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.constitutive.coefficient_functions import (
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
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.rom import (
    AlgebraicLift,
    LinearTrialSpace,
    SnapshotSet,
    TrialSpaceProtocol,
    collect_steady_snapshots,
    fit_pod_trial_space,
    homogenize,
    projection_errors,
    snapshot_spectrum,
)
from pyapprox.surrogates.kle import as_basis_operator
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    InnerProductProtocol,
    MassInnerProduct,
)
from tests._helpers.adjoint_checks import NumpyArray


def _assert_pod_identity(
    bkd: Backend[Any],
    snapshots: SnapshotSet[Any],
    metric: InnerProductProtocol[Any],
    ranks: range,
) -> None:
    sigma = snapshot_spectrum(snapshots, bkd, metric=metric)
    for rank in ranks:
        trial = fit_pod_trial_space(snapshots, bkd, latent_dim=rank, metric=metric)
        errors = projection_errors(trial, snapshots.states)
        bkd.assert_allclose(
            bkd.sum(errors**2)[None],
            bkd.sum(sigma[rank:] ** 2)[None],
            rtol=1e-8,
            atol=1e-12 * float(bkd.to_float(bkd.sum(sigma**2))),
        )
        bkd.assert_allclose(trial.gram(), bkd.eye(rank), atol=1e-10)


def _synthetic(bkd: Backend[Any]) -> Tuple[SnapshotSet[Any], Any]:
    rng = np.random.default_rng(1)
    low_rank = rng.normal(size=(20, 5)) @ rng.normal(size=(5, 8))
    states = low_rank + 1e-3 * rng.normal(size=(20, 8))
    weights = 10.0 ** rng.uniform(-1.0, 1.0, size=20)
    return (
        SnapshotSet(bkd.asarray(states), bkd.zeros((1, 8))),
        DiagonalInnerProduct(bkd.asarray(weights), bkd),
    )


def _unit_forcing(coords: Any) -> Any:
    return np.ones(coords.shape[1])


def _galerkin(
    bkd: NumpyBkd,
) -> Tuple[SnapshotSet[NumpyArray], MassInnerProduct[NumpyArray], Any]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
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
        diffusivity_map=TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp),
        bkd=bkd,
    )
    state_eq = GalerkinParameterizedStateEquation(system.steady(), param_obj, bkd)
    samples = np.random.default_rng(2).uniform(-0.5, 0.5, size=(3, 12))
    snapshots = collect_steady_snapshots(
        state_eq, bkd.asarray(samples), bkd.zeros((system.nstates(), 1))
    )
    constraint_set = system.constraint_set()
    homogenized = homogenize(
        snapshots,
        AlgebraicLift(constraint_set, system.nstates()),
        constraint_set,
    )
    ends = bkd.concatenate([basis.get_dofs("left"), basis.get_dofs("right")])
    return (
        homogenized,
        MassInnerProduct(physics.mass_matrix(), bkd),
        ends,
    )


class TestLinearTrialSpace:
    def test_satisfies_protocol(self, bkd: Backend[Any]) -> None:
        snapshots, metric = _synthetic(bkd)
        trial = fit_pod_trial_space(snapshots, bkd, latent_dim=3, metric=metric)
        assert isinstance(trial, TrialSpaceProtocol)
        assert (trial.nstates(), trial.ncodes()) == (20, 3)

    def test_project_inverts_decode(self, bkd: Backend[Any]) -> None:
        snapshots, metric = _synthetic(bkd)
        trial = fit_pod_trial_space(snapshots, bkd, latent_dim=3, metric=metric)
        codes = bkd.asarray(np.random.default_rng(3).normal(size=(3, 4)))
        bkd.assert_allclose(trial.project(trial.decode(codes)), codes, rtol=1e-10)

    def test_projection_is_metric_orthogonal_for_any_basis(
        self, bkd: Backend[Any]
    ) -> None:
        """The residual is M-orthogonal to the span even when V^T M V != I."""
        snapshots, metric = _synthetic(bkd)
        raw = bkd.asarray(np.random.default_rng(4).normal(size=(20, 3)))
        trial = LinearTrialSpace(as_basis_operator(raw, bkd), bkd.zeros((20,)), metric)
        residual = snapshots.states - trial.decode(trial.project(snapshots.states))
        bkd.assert_allclose(
            bkd.dot(raw.T, metric.apply(residual)),
            bkd.zeros((3, 8)),
            atol=1e-10,
        )

    def test_rejects_wrong_reference_shape(self, bkd: Backend[Any]) -> None:
        raw = bkd.ones((5, 2))
        with pytest.raises(ValueError, match="reference"):
            LinearTrialSpace(as_basis_operator(raw, bkd), bkd.zeros((4,)))


class TestPODIdentity:
    def test_synthetic_diagonal_metric(self, bkd: Backend[Any]) -> None:
        snapshots, metric = _synthetic(bkd)
        _assert_pod_identity(bkd, snapshots, metric, range(1, 7))

    def test_galerkin_mass_metric(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        snapshots, metric, _ = _galerkin(bkd)
        _assert_pod_identity(bkd, snapshots, metric, range(1, 6))

    def test_basis_vanishes_on_constrained_dofs(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        snapshots, metric, ends = _galerkin(bkd)
        trial = fit_pod_trial_space(snapshots, bkd, latent_dim=3, metric=metric)
        # Newton leaves ~1e-14 on the ends; the smallest kept mode divides
        # it by its singular value.
        bkd.assert_allclose(
            trial.apply(bkd.eye(3))[ends, :], bkd.zeros((2, 3)), atol=1e-10
        )
        bkd.assert_allclose(trial.reference()[ends], bkd.zeros((2,)), atol=1e-12)

    def test_euclidean_basis_fails_in_weighted_metric(self, bkd: Backend[Any]) -> None:
        snapshots, metric = _synthetic(bkd)
        rank = 3
        sigma = snapshot_spectrum(snapshots, bkd, metric=metric)
        euclidean = fit_pod_trial_space(snapshots, bkd, latent_dim=rank)
        wrong = LinearTrialSpace(euclidean.basis(), euclidean.reference(), metric)
        error = float(
            bkd.to_float(bkd.sum(projection_errors(wrong, snapshots.states) ** 2))
        )
        tail = float(bkd.to_float(bkd.sum(sigma[rank:] ** 2)))
        assert error > 1.01 * tail
        assert (
            float(bkd.to_float(bkd.max(bkd.abs(wrong.gram() - bkd.eye(rank))))) > 1e-3
        )
