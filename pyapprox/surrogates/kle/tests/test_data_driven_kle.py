"""Tests for DataDrivenKLE and PrincipalComponentAnalysis.

Ports legacy tests from pyapprox/surrogates/affine/tests/test_kle.py.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.surrogates.affine.univariate.globalpoly import (
    LegendrePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    GaussQuadratureRule,
)
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle.data_driven_kle import DataDrivenKLE
from pyapprox.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.surrogates.kle.pca import PrincipalComponentAnalysis
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests, slow_test  # noqa: F401


def _gauss_legendre_quad(lb, ub, npts, bkd):
    """Gauss-Legendre quadrature on [lb, ub] for Lebesgue integration."""
    poly = LegendrePolynomial1D(bkd)
    poly.set_nterms(npts)
    quad_rule = GaussQuadratureRule(poly)
    pts, wts = quad_rule(npts)
    dom_len = ub - lb
    half_len = dom_len / 2.0
    mid = (lb + ub) / 2.0
    pts = pts * half_len + mid
    wts = wts * dom_len
    return pts, wts[:, 0]


class TestDataDrivenKLE(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(1)
        self._bkd = self.bkd()

    @slow_test
    def test_data_driven_kle_vs_mesh_kle(self) -> None:
        """Port of legacy test_data_driven_kle (part 1).

        Build MeshKLE (no weights), generate 10k samples, build
        DataDrivenKLE from realizations, verify eigenvalues match.
        """
        bkd = self._bkd
        nterms = 3
        level = 6
        len_scale, sigma = 1.0, 1.0
        lb, ub = 0.0, 2.0
        npts = 2**level + 1

        mesh_coords, quad_weights = _gauss_legendre_quad(lb, ub, npts, bkd)

        lenscale_arr = bkd.array([len_scale])
        kernel = ExponentialKernel(lenscale_arr, (0.01, 100.0), 1, bkd)

        kle = MeshKLE(
            mesh_coords,
            kernel,
            sigma=sigma,
            nterms=nterms,
            quad_weights=None,
            bkd=bkd,
        )

        nsamples = 10000
        samples = bkd.asarray(np.random.normal(0.0, 1.0, (nterms, nsamples)))
        kle_realizations = kle(samples)

        kle_data = DataDrivenKLE(
            kle_realizations,
            nterms=nterms,
            bkd=bkd,
        )
        bkd.assert_allclose(
            kle_data._sqrt_eig_vals,
            kle._sqrt_eig_vals,
            atol=1e-2,
            rtol=1e-2,
        )

    @slow_test
    def test_data_driven_kle_with_weights(self) -> None:
        """Port of legacy test_data_driven_kle (part 3).

        MeshKLE with quad weights -> generate samples ->
        DataDrivenKLE with same weights -> eigenvalues match.
        """
        bkd = self._bkd
        nterms = 3
        level = 6
        len_scale, sigma = 1.0, 1.0
        lb, ub = 0.0, 2.0
        npts = 2**level + 1

        mesh_coords, quad_weights = _gauss_legendre_quad(lb, ub, npts, bkd)

        lenscale_arr = bkd.array([len_scale])
        kernel = ExponentialKernel(lenscale_arr, (0.01, 100.0), 1, bkd)

        kle = MeshKLE(
            mesh_coords,
            kernel,
            sigma=sigma,
            nterms=nterms,
            quad_weights=quad_weights,
            bkd=bkd,
        )

        nsamples = 10000
        samples = bkd.asarray(np.random.normal(0.0, 1.0, (nterms, nsamples)))
        kle_realizations = kle(samples)

        kle_data = DataDrivenKLE(
            kle_realizations,
            nterms=nterms,
            quad_weights=quad_weights,
            bkd=bkd,
        )
        bkd.assert_allclose(
            kle_data._sqrt_eig_vals,
            kle._sqrt_eig_vals,
            atol=1e-2,
            rtol=1e-2,
        )


class TestDataDrivenKLENumpy(TestDataDrivenKLE[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDataDrivenKLETorch(TestDataDrivenKLE[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestPCA(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(1)
        self._bkd = self.bkd()

    def test_pca_low_rank_recovery(self) -> None:
        """Port of legacy test_PCA_low_rank_matrix_recovery.

        PCA with nterms=rank recovers low-rank matrix exactly.
        """
        bkd = self._bkd
        B = bkd.asarray(np.random.rand(5, 3))
        A = B.T @ B  # shape (3, 3), rank <= 3

        rank_A = bkd.rank(A)

        pca = PrincipalComponentAnalysis(A, rank_A, bkd=bkd)

        # Verify the reduced basis is orthogonal
        bkd.assert_allclose(
            pca.eigenvectors().T @ pca.eigenvectors(),
            bkd.eye(rank_A),
            atol=1e-14,
        )

        # Reduce and expand A using PCA
        reduced_A = pca.reduce_state(A)
        recovered_A = pca.expand_reduced_state(reduced_A)

        # Verify that the recovered matrix matches the original
        bkd.assert_allclose(recovered_A, A, rtol=1e-6, atol=1e-8)

    def test_pca_reduce_expand_cycle(self) -> None:
        """Test that reduce -> expand is identity for normalized snapshots."""
        bkd = self._bkd
        ncoords, nsamples = 10, 5
        data = bkd.asarray(np.random.rand(ncoords, nsamples))
        rank = bkd.rank(data)

        pca = PrincipalComponentAnalysis(data, rank, bkd=bkd)

        # Use the normalized snapshots that PCA actually operates on
        normalized = pca.snapshots()

        # Project normalized data onto reduced basis and back
        reduced = pca.reduce_state(normalized)
        expanded = pca.expand_reduced_state(reduced)

        bkd.assert_allclose(expanded, normalized, rtol=1e-6, atol=1e-8)


class TestPCANumpy(TestPCA[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPCATorch(TestPCA[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
