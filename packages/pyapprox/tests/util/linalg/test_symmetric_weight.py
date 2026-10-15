"""Tests for the storage-owning symmetric weights."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    EuclideanInnerProduct,
)
from pyapprox.util.linalg.symmetric_weight import (
    DenseMatrixWeight,
    ScipySparseWeight,
    SymmetricWeightProtocol,
    TorchSparseWeight,
)

_N = 7


def _p1_mass() -> csr_matrix:
    """1D P1 mass on [0, 1]: symmetric positive definite, tridiagonal."""
    h = 1.0 / (_N - 1)
    return diags([h / 6.0, 2.0 * h / 3.0, h / 6.0], [-1, 0, 1], shape=(_N, _N)).tocsr()


def _zone_mass() -> csr_matrix:
    """The P1 mass zeroed outside rows/columns 2..4: only semidefinite."""
    mass = _p1_mass().toarray()
    keep = np.zeros(_N, dtype=bool)
    keep[2:5] = True
    mass[~keep, :] = 0.0
    mass[:, ~keep] = 0.0
    return csr_matrix(mass)


class TestSymmetricWeights:
    @pytest.mark.parametrize("build", ["dense", "scipy"])
    def test_apply_matches_matrix_product(
        self, bkd: Backend[Array], build: str
    ) -> None:
        for matrix in (_p1_mass(), _zone_mass()):
            weight: SymmetricWeightProtocol[Array]
            if build == "dense":
                weight = DenseMatrixWeight(bkd.asarray(matrix.toarray()), bkd)
            else:
                weight = ScipySparseWeight(matrix, bkd)
            assert isinstance(weight, SymmetricWeightProtocol)
            assert weight.nstates() == _N
            x = bkd.asarray(np.random.default_rng(3).normal(size=(_N, 4)))
            bkd.assert_allclose(
                weight.apply(x),
                bkd.asarray(matrix.toarray() @ bkd.to_numpy(x)),
                rtol=1e-12,
            )

    def test_inner_products_are_weights(self, bkd: Backend[Array]) -> None:
        """Every inner product is a (definite) symmetric weight."""
        assert isinstance(EuclideanInnerProduct(_N, bkd), SymmetricWeightProtocol)
        assert isinstance(
            DiagonalInnerProduct(bkd.ones((_N,)), bkd), SymmetricWeightProtocol
        )

    def test_dense_rejects_nonsquare_and_asymmetric(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="nstates, nstates"):
            DenseMatrixWeight(bkd.ones((_N, _N + 1)), bkd)
        asymmetric = np.eye(_N)
        asymmetric[0, 1] = 1.0
        with pytest.raises(ValueError, match="symmetric"):
            DenseMatrixWeight(bkd.asarray(asymmetric), bkd)

    def test_scipy_rejects_dense_and_asymmetric(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="scipy sparse"):
            ScipySparseWeight(_p1_mass().toarray(), bkd)
        asymmetric = _p1_mass().tolil()
        asymmetric[0, 3] = 1.0
        with pytest.raises(ValueError, match="symmetric"):
            ScipySparseWeight(asymmetric.tocsr(), bkd)
        with pytest.raises(ValueError, match="nstates, nstates"):
            ScipySparseWeight(csr_matrix(np.ones((_N, _N + 1))), bkd)


class TestTorchSparseWeight:
    def test_apply_matches_matrix_product(self, torch_bkd: Backend[Array]) -> None:
        bkd = torch_bkd
        matrix = _zone_mass()
        weight = TorchSparseWeight(matrix, bkd)
        assert isinstance(weight, SymmetricWeightProtocol)
        x = bkd.asarray(np.random.default_rng(4).normal(size=(_N, 3)))
        bkd.assert_allclose(
            weight.apply(x),
            bkd.asarray(matrix.toarray() @ bkd.to_numpy(x)),
            rtol=1e-12,
        )

    def test_preserves_the_graph(self, torch_bkd: Backend[Array]) -> None:
        """d(x^T W x)/dx = 2 W x through autograd, so apply keeps x in
        the graph (a scipy product would detach it)."""
        import torch

        bkd = torch_bkd
        matrix = _p1_mass()
        weight = TorchSparseWeight(matrix, bkd)
        x = torch.tensor(
            np.random.default_rng(6).normal(size=(_N, 1)), requires_grad=True
        )
        value = torch.sum(x * weight.apply(x))
        (grad,) = torch.autograd.grad(value, x)
        bkd.assert_allclose(
            grad,
            bkd.asarray(2.0 * matrix.toarray() @ x.detach().numpy()),
            rtol=1e-12,
        )

    def test_rejects_asymmetric(self, torch_bkd: Backend[Array]) -> None:
        asymmetric = _p1_mass().tolil()
        asymmetric[0, 3] = 1.0
        with pytest.raises(ValueError, match="symmetric"):
            TorchSparseWeight(asymmetric.tocsr(), torch_bkd)
