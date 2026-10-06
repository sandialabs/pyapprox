"""Tests for the white-noise mass weightings."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
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


def _dense(matrix: Any) -> NDArray[np.floating[Any]]:
    if issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _basis(
    bkd: Backend[Array], element_type: str = "tri", degree: int = 1
) -> LagrangeBasis[Array]:
    mesh = StructuredMesh2D(
        4, 4, [[0.0, 1.0], [0.0, 1.0]], bkd, element_type=element_type
    )
    return LagrangeBasis(mesh, degree=degree)


def _factor(noise_mass: NoiseMassProtocol[Array], bkd: Backend[Array]) -> Array:
    """``B`` as a dense backend array, by applying it to the identity."""
    return noise_mass.apply_factor(bkd.eye(noise_mass.nnoise()))


def _consistent_mass(bkd: Backend[Array], basis: Any) -> Array:
    return bkd.asarray(_dense(ConsistentNoiseMass(basis, bkd).mass_matrix()))


class TestConsistentNoiseMass:
    @pytest.mark.parametrize(
        "element_type,degree,nloc",
        [("tri", 1, 3), ("tri", 2, 6), ("quad", 1, 4)],
    )
    def test_factor_reproduces_mass(
        self, bkd: Backend[Array], element_type: str, degree: int, nloc: int
    ) -> None:
        """B B^T = M exactly, and nnoise is the number of local DOFs."""
        basis = _basis(bkd, element_type, degree)
        noise_mass = ConsistentNoiseMass(basis, bkd)
        assert isinstance(noise_mass, NoiseMassProtocol)
        nelem = basis.skfem_basis().mesh.nelements
        assert noise_mass.nnoise() == nloc * nelem
        factor = _factor(noise_mass, bkd)
        assert tuple(factor.shape) == (basis.ndofs(), nloc * nelem)
        bkd.assert_allclose(
            factor @ factor.T,
            bkd.asarray(_dense(noise_mass.mass_matrix())),
            rtol=1e-12,
            atol=1e-15,
        )


class TestLumpedNoiseMass:
    def test_factor_is_square_root_of_row_sums(
        self, bkd: Backend[Array]
    ) -> None:
        basis = _basis(bkd)
        noise_mass = LumpedNoiseMass(basis, bkd)
        assert isinstance(noise_mass, NoiseMassProtocol)
        assert noise_mass.nnoise() == basis.ndofs()
        row_sums = bkd.sum(_consistent_mass(bkd, basis), axis=1)
        bkd.assert_allclose(
            bkd.asarray(np.diag(_dense(noise_mass.mass_matrix()))),
            row_sums,
            rtol=1e-12,
        )
        bkd.assert_allclose(
            _factor(noise_mass, bkd),
            bkd.diag(bkd.sqrt(row_sums)),
            rtol=1e-12,
        )

    def test_matches_prior_lumped_mass(self, bkd: Backend[Array]) -> None:
        """The default weighting is the prior's existing lumped mass."""
        basis = _basis(bkd)
        prior = BiLaplacianPrior.with_uniform_robin(basis, 1.0, 10.0, bkd)
        bkd.assert_allclose(
            bkd.asarray(np.diag(_dense(prior.noise_mass().mass_matrix()))),
            prior.lumped_mass(),
            rtol=1e-12,
        )


class TestCovarianceOperatorNoiseMass:
    def test_dense_cholesky_is_square_consistent_factor(
        self, bkd: Backend[Array]
    ) -> None:
        basis = _basis(bkd)
        mass = _consistent_mass(bkd, basis)
        noise_mass = CovarianceOperatorNoiseMass(
            DenseCholeskyCovarianceOperator(mass, bkd)
        )
        assert isinstance(noise_mass, NoiseMassProtocol)
        assert noise_mass.nnoise() == basis.ndofs()
        factor = _factor(noise_mass, bkd)
        bkd.assert_allclose(factor @ factor.T, mass, rtol=1e-12, atol=1e-15)
        bkd.assert_allclose(
            bkd.asarray(_dense(noise_mass.mass_matrix())), mass, rtol=1e-12
        )

    def test_rejects_non_operator(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="CovarianceOperatorProtocol"):
            CovarianceOperatorNoiseMass(object())  # type: ignore[arg-type]


def _noise_masses(
    bkd: Backend[Array], basis: Any
) -> Tuple[NoiseMassProtocol[Array], ...]:
    return (
        LumpedNoiseMass(basis, bkd),
        ConsistentNoiseMass(basis, bkd),
        CovarianceOperatorNoiseMass(
            DenseCholeskyCovarianceOperator(_consistent_mass(bkd, basis), bkd)
        ),
    )


def test_apply_factor_rejects_wrong_noise_length(
    numpy_bkd: Backend[Array],
) -> None:
    basis = _basis(numpy_bkd)
    for noise_mass in _noise_masses(numpy_bkd, basis):
        with pytest.raises(ValueError, match="noise must have shape"):
            noise_mass.apply_factor(
                numpy_bkd.zeros((noise_mass.nnoise() + 1, 1))
            )
