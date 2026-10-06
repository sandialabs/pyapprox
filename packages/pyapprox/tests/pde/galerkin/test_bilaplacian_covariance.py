"""Covariance, factors and sampling of ``BiLaplacianPrior`` for each noise
mass, and the mesh independence of its pointwise standard deviation."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.bilaplacian import BiLaplacianPrior
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.noise_mass import (
    ConsistentNoiseMass,
    CovarianceOperatorNoiseMass,
    LumpedNoiseMass,
    NoiseMassProtocol,
)
from pyapprox.probability.covariance.dense import (
    DenseCholeskyCovarianceOperator,
)
from pyapprox.util.backends.protocols import Array, Backend
from scipy.sparse import issparse

from tests._helpers.markers import slow_test

_GAMMA, _DELTA = 10.0, 100.0


def _dense(matrix: Any) -> NDArray[np.floating[Any]]:
    if issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _basis(bkd: Backend[Array], nelem: int) -> LagrangeBasis[Array]:
    mesh = StructuredMesh2D(
        nelem, nelem, [[0.0, 1.0], [0.0, 1.0]], bkd, element_type="tri"
    )
    return LagrangeBasis(mesh, degree=1)


def _lumped(bkd: Backend[Array], basis: Any) -> NoiseMassProtocol[Array]:
    return LumpedNoiseMass(basis, bkd)


def _consistent(bkd: Backend[Array], basis: Any) -> NoiseMassProtocol[Array]:
    return ConsistentNoiseMass(basis, bkd)


def _dense_consistent(
    bkd: Backend[Array], basis: Any
) -> NoiseMassProtocol[Array]:
    mass = bkd.asarray(_dense(ConsistentNoiseMass(basis, bkd).mass_matrix()))
    return CovarianceOperatorNoiseMass(DenseCholeskyCovarianceOperator(mass, bkd))


_MASSES: Dict[str, Callable[[Backend[Any], Any], NoiseMassProtocol[Any]]] = {
    "lumped": _lumped,
    "consistent": _consistent,
    "dense_consistent": _dense_consistent,
}


def _prior(
    bkd: Backend[Array],
    mass: str,
    nelem: int = 5,
    robin_alpha: Optional[float] = None,
) -> BiLaplacianPrior[Array]:
    basis = _basis(bkd, nelem)
    return BiLaplacianPrior.with_uniform_robin(
        basis,
        1.0,
        10.0,
        bkd,
        robin_alpha=robin_alpha,
        noise_mass=_MASSES[mass](bkd, basis),
    )


def _reference_covariance(prior: BiLaplacianPrior[Any]) -> Any:
    """K^{-1} M K^{-1} computed independently of the prior's methods."""
    stiffness = _dense(prior.stiffness_matrix())
    mass = _dense(prior.noise_mass().mass_matrix())
    inverse = np.linalg.inv(stiffness)
    return inverse @ mass @ inverse


@pytest.mark.parametrize("mass", list(_MASSES))
class TestCovariance:
    def test_covariance_definition(self, bkd: Backend[Array], mass: str) -> None:
        prior = _prior(bkd, mass)
        bkd.assert_allclose(
            prior.covariance(),
            bkd.asarray(_reference_covariance(prior)),
            rtol=1e-10,
            atol=1e-14,
        )

    def test_factor_reproduces_covariance(
        self, bkd: Backend[Array], mass: str
    ) -> None:
        prior = _prior(bkd, mass)
        factor = prior.covariance_factor()
        ndofs = prior.stiffness_matrix().shape[0]
        assert tuple(factor.shape) == (ndofs, prior.nnoise())
        bkd.assert_allclose(
            factor @ factor.T, prior.covariance(), rtol=1e-10, atol=1e-14
        )

    def test_apply_covariance_matches_dense(
        self, bkd: Backend[Array], mass: str
    ) -> None:
        prior = _prior(bkd, mass)
        rng = np.random.default_rng(0)
        vectors = bkd.asarray(rng.normal(size=(prior.covariance().shape[0], 3)))
        bkd.assert_allclose(
            prior.apply_covariance(vectors),
            prior.covariance() @ vectors,
            rtol=1e-10,
            atol=1e-14,
        )

    def test_full_expansion_reproduces_covariance(
        self, bkd: Backend[Array], mass: str
    ) -> None:
        prior = _prior(bkd, mass)
        ndofs = prior.covariance().shape[0]
        full = prior.truncated_covariance_factor(ndofs)
        bkd.assert_allclose(
            full @ full.T, prior.covariance(), rtol=1e-8, atol=1e-14
        )

    def test_truncation_gives_leading_eigenpairs(
        self, bkd: Backend[Array], mass: str
    ) -> None:
        """The sparse rank-r eigenpairs are the leading eigenpairs of the
        dense full solve, they solve K phi = mu M phi, and they are
        M-orthonormal. Compared only through sign-free quantities: the
        two solvers may orient a mode differently when its extreme
        entries tie to rounding."""
        prior = _prior(bkd, mass)
        ndofs = prior.covariance().shape[0]
        rank = 5
        mu, modes = prior.generalized_eigenpairs(rank)
        mu_full, _ = prior.generalized_eigenpairs(ndofs)
        bkd.assert_allclose(mu, mu_full[:rank], rtol=1e-8)
        stiffness = bkd.asarray(_dense(prior.stiffness_matrix()))
        mass_matrix = bkd.asarray(_dense(prior.noise_mass().mass_matrix()))
        bkd.assert_allclose(
            stiffness @ modes, (mass_matrix @ modes) * mu, rtol=1e-8, atol=1e-10
        )
        bkd.assert_allclose(
            modes.T @ mass_matrix @ modes, bkd.eye(rank), rtol=1e-8, atol=1e-10
        )
        # The truncated factor scales each mode by 1/mu: mu is an
        # eigenvalue of the precision, so the covariance's is 1/mu^2.
        bkd.assert_allclose(
            prior.truncated_covariance_factor(rank), modes / mu, rtol=1e-12
        )


def test_consistent_equals_dense_adapter(bkd: Backend[Array]) -> None:
    """The element-assembled factor and the dense Cholesky factor differ,
    but both have B B^T = M, so the covariances agree exactly."""
    consistent = _prior(bkd, "consistent")
    dense = _prior(bkd, "dense_consistent")
    assert consistent.nnoise() > dense.nnoise()
    bkd.assert_allclose(
        consistent.covariance(), dense.covariance(), rtol=1e-12, atol=1e-16
    )


def test_lumped_differs_from_consistent(numpy_bkd: Backend[Array]) -> None:
    lumped = numpy_bkd.to_numpy(_prior(numpy_bkd, "lumped").covariance())
    consistent = numpy_bkd.to_numpy(_prior(numpy_bkd, "consistent").covariance())
    assert np.abs(lumped - consistent).max() > 1e-3 * np.abs(lumped).max()


def test_default_samples_unchanged(numpy_bkd: Backend[Array]) -> None:
    """The default noise mass is the lumped mass the prior always used."""
    basis = _basis(numpy_bkd, 5)
    default = BiLaplacianPrior.with_uniform_robin(basis, 1.0, 10.0, numpy_bkd)
    explicit = BiLaplacianPrior.with_uniform_robin(
        basis, 1.0, 10.0, numpy_bkd, noise_mass=LumpedNoiseMass(basis, numpy_bkd)
    )
    numpy_bkd.assert_allclose(
        default.rvs(4, rng=np.random.default_rng(7)),
        explicit.rvs(4, rng=np.random.default_rng(7)),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("mass", list(_MASSES))
def test_sample_variance_matches_covariance(
    numpy_bkd: Backend[Array], mass: str
) -> None:
    """Pointwise sample variances match diag(covariance()) within Monte
    Carlo error: the relative std error of a variance estimate from N
    zero-mean samples is sqrt(2/N), and we allow 5 of them."""
    prior = _prior(numpy_bkd, mass, nelem=4)
    nsamples = 4000
    samples = numpy_bkd.to_numpy(
        prior.rvs(nsamples, rng=np.random.default_rng(11))
    )
    sample_variance = np.mean(samples**2, axis=1)
    variance = np.diag(numpy_bkd.to_numpy(prior.covariance()))
    tolerance = 5.0 * np.sqrt(2.0 / nsamples)
    numpy_bkd.assert_allclose(
        numpy_bkd.asarray(sample_variance),
        numpy_bkd.asarray(variance),
        rtol=tolerance,
    )


@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_field_map_over_factor_passes_derivative_check(
    bkd: Backend[Array], mass: str
) -> None:
    prior = _prior(bkd, mass, nelem=3)
    factor = prior.covariance_factor()
    mean = bkd.full((factor.shape[0],), -3.0)
    field_map = MeshKLEFieldMap(bkd, mean, factor)
    assert field_map.nvars() == prior.nnoise()
    wrapper = FunctionWithJacobianFromCallable(
        nqoi=factor.shape[0],
        nvars=field_map.nvars(),
        fun=lambda samples: bkd.stack(
            [field_map(samples[:, ii]) for ii in range(samples.shape[1])],
            axis=1,
        ),
        jacobian=lambda sample: field_map.jacobian(sample[:, 0]),
        bkd=bkd,
    )
    rng = np.random.default_rng(3)
    params = bkd.asarray(rng.normal(size=(field_map.nvars(), 1)))
    errors = DerivativeChecker(wrapper).check_derivatives(params)[0]
    assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-6


def _std_at(prior: BiLaplacianPrior[Any], nodes: Any, bkd: Backend[Any]) -> Any:
    """Exact pointwise std at ``nodes`` through sparse solves."""
    ndofs = prior.stiffness_matrix().shape[0]
    unit = np.zeros((ndofs, len(nodes)))
    unit[nodes, np.arange(len(nodes))] = 1.0
    applied = bkd.to_numpy(prior.apply_covariance(bkd.asarray(unit)))
    return np.sqrt(applied[nodes, np.arange(len(nodes))])


def _center(basis: Any, bkd: Backend[Any]) -> int:
    coords = bkd.to_numpy(basis.dof_coordinates())
    return int(np.argmin((coords[0] - 0.5) ** 2 + (coords[1] - 0.5) ** 2))


@pytest.mark.parametrize(
    "nelem,neumann_lumped,neumann_consistent,robin_lumped,robin_consistent",
    [
        (19, 0.01138, 0.01130, 0.00867, 0.00855),
        (39, 0.01132, 0.01129, 0.00860, 0.00856),
        pytest.param(79, 0.01130, 0.01129, 0.00858, 0.00857, marks=slow_test),
    ],
)
def test_center_std_under_refinement(
    numpy_bkd: Backend[Array],
    nelem: int,
    neumann_lumped: float,
    neumann_consistent: float,
    robin_lumped: float,
    robin_consistent: float,
) -> None:
    """The center std converges under refinement, for both masses.

    Setup: unit square, ``StructuredMesh2D(nelem, nelem,
    element_type="tri")``, P1, gamma = 10, delta = 100, the node nearest
    (0.5, 0.5), exact diagonal through ``apply_covariance``. Neumann is no
    boundary term; Robin is ``with_uniform_robin``'s default coefficient,
    ``sqrt(gamma * delta) * 1.42``.
    """
    basis = _basis(numpy_bkd, nelem)
    center = _center(basis, numpy_bkd)
    cases = (
        ([], LumpedNoiseMass(basis, numpy_bkd), neumann_lumped),
        ([], ConsistentNoiseMass(basis, numpy_bkd), neumann_consistent),
        (None, LumpedNoiseMass(basis, numpy_bkd), robin_lumped),
        (None, ConsistentNoiseMass(basis, numpy_bkd), robin_consistent),
    )
    for bcs, noise_mass, expected in cases:
        if bcs is None:
            prior = BiLaplacianPrior.with_uniform_robin(
                basis, _GAMMA, _DELTA, numpy_bkd, noise_mass=noise_mass
            )
        else:
            prior = BiLaplacianPrior(
                basis, _GAMMA, _DELTA, numpy_bkd, bcs, noise_mass=noise_mass
            )
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(_std_at(prior, [center], numpy_bkd)),
            numpy_bkd.asarray([expected]),
            rtol=6e-4,
        )


def test_robin_edges_and_corners_near_free_space(
    numpy_bkd: Backend[Array],
) -> None:
    """With Robin coefficient sqrt(gamma delta)/1.42 (lumped mass,
    h = 1/39, gamma = 10, delta = 100), the std at the bottom-edge
    midpoint and the corner (0, 0) is within 10% of the free-space value
    1/sqrt(4 pi gamma delta)."""
    basis = _basis(numpy_bkd, 39)
    coords = numpy_bkd.to_numpy(basis.dof_coordinates())
    edge = int(np.argmin((coords[0] - 0.5) ** 2 + coords[1] ** 2))
    corner = int(np.argmin(coords[0] ** 2 + coords[1] ** 2))
    prior = BiLaplacianPrior.with_uniform_robin(
        basis,
        _GAMMA,
        _DELTA,
        numpy_bkd,
        robin_alpha=float(np.sqrt(_GAMMA * _DELTA) / 1.42),
    )
    free_space = 1.0 / np.sqrt(4.0 * np.pi * _GAMMA * _DELTA)
    ratios = _std_at(prior, [edge, corner], numpy_bkd) / free_space
    numpy_bkd.assert_allclose(
        numpy_bkd.asarray(ratios), numpy_bkd.asarray([1.0, 1.0]), atol=0.1
    )


def test_rejects_non_noise_mass(numpy_bkd: Backend[Array]) -> None:
    with pytest.raises(TypeError, match="NoiseMassProtocol"):
        BiLaplacianPrior(
            _basis(numpy_bkd, 3),
            1.0,
            10.0,
            numpy_bkd,
            [],
            noise_mass=object(),  # type: ignore[arg-type]
        )


def test_truncation_rank_is_validated(numpy_bkd: Backend[Array]) -> None:
    prior = _prior(numpy_bkd, "lumped", nelem=3)
    ndofs = prior.stiffness_matrix().shape[0]
    for rank in (0, ndofs + 1):
        with pytest.raises(ValueError, match="rank must be"):
            prior.truncated_covariance_factor(rank)
