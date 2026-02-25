"""Tests for FEM-aware KLE factory functions.

Tests ``create_fem_galerkin_kle``, ``create_fem_nystrom_nodes_kle``, and
``create_fem_nystrom_quadrature_kle`` from ``kle_factory``.  All tests use
small 1D skfem meshes (20-30 elements) to keep kernel matrices tiny and
avoid the memory problems that arise with large meshes.

These tests live here (not in ``surrogates/kle/tests``) because the
factory functions depend on skfem, which is a PDE-layer dependency.
"""

import unittest

import numpy as np
from skfem import Basis, ElementLineP1, MeshLine, asm
from skfem.models.poisson import mass

from pyapprox.pde.field_maps.kle_factory import (
    create_fem_galerkin_kle,
    create_fem_nystrom_nodes_kle,
    create_fem_nystrom_quadrature_kle,
)
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle.analytical import (
    AnalyticalExponentialKLE1D,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


def _make_1d_setup(nelems=25, nterms=3, corr_len=1.0, dom_len=2.0):
    """Create a small 1D skfem mesh and kernel for testing.

    Returns
    -------
    basis : skfem Basis
    kernel : ExponentialKernel
    analytical : AnalyticalExponentialKLE1D
    bkd : NumpyBkd
    nterms : int
    """
    bkd = NumpyBkd()
    mesh = MeshLine(np.linspace(0.0, dom_len, nelems + 1))
    basis = Basis(mesh, ElementLineP1())

    lenscale = bkd.array([corr_len])
    kernel = ExponentialKernel(lenscale, (0.01, 100.0), 1, bkd)

    analytical = AnalyticalExponentialKLE1D(
        corr_len=corr_len,
        sigma2=1.0,
        dom_len=dom_len,
        nterms=nterms,
    )
    return basis, kernel, analytical, bkd, nterms


class TestFEMKLE(unittest.TestCase):
    """Tests for FEM KLE factory functions (numpy-only, requires skfem)."""

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    # --- Galerkin KLE tests ---

    def test_galerkin_eigenvalues_vs_analytical(self) -> None:
        """GalerkinKLE eigenvalues converge to analytical values."""
        basis, kernel, analytical, bkd, nterms = _make_1d_setup(nelems=30)
        kle = create_fem_galerkin_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        exact_eig = analytical.eigenvalues()
        bkd.assert_allclose(kle.eigenvalues(), bkd.array(exact_eig), rtol=5e-3)

    def test_galerkin_m_orthonormality(self) -> None:
        """GalerkinKLE eigenvectors are M-orthonormal."""
        basis, kernel, analytical, bkd, nterms = _make_1d_setup(nelems=25)
        kle = create_fem_galerkin_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )

        M = asm(mass, basis)
        M_dense = bkd.asarray(M.toarray())
        vecs = kle.eigenvectors()  # (ndofs, nterms)

        # v_i^T M v_j = delta_{ij}
        gram = vecs.T @ M_dense @ vecs
        identity = bkd.array(np.eye(nterms))
        bkd.assert_allclose(gram, identity, atol=1e-10)

    def test_galerkin_shapes(self) -> None:
        """GalerkinKLE has correct output shapes."""
        basis, kernel, _, bkd, nterms = _make_1d_setup(nelems=20)
        kle = create_fem_galerkin_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        ndofs = basis.N
        self.assertEqual(kle.eigenvalues().shape, (nterms,))
        self.assertEqual(kle.eigenvectors().shape, (ndofs, nterms))
        self.assertEqual(kle.weighted_eigenvectors().shape, (ndofs, nterms))
        self.assertEqual(kle.nterms(), nterms)
        self.assertEqual(kle.nvars(), nterms)

        # Evaluate
        coef = bkd.array(np.random.randn(nterms, 4))
        result = kle(coef)
        self.assertEqual(result.shape, (ndofs, 4))

    # --- Nystrom nodes KLE tests ---

    def test_nystrom_nodes_eigenvalues_vs_analytical(self) -> None:
        """Nystrom-at-nodes eigenvalues converge to analytical values."""
        basis, kernel, analytical, bkd, nterms = _make_1d_setup(nelems=30)
        kle = create_fem_nystrom_nodes_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        exact_eig = analytical.eigenvalues()
        bkd.assert_allclose(kle.eigenvalues(), bkd.array(exact_eig), rtol=5e-3)

    def test_nystrom_nodes_weighted_orthonormality(self) -> None:
        """Nystrom-at-nodes eigenvectors are orthonormal under lumped mass."""
        basis, kernel, _, bkd, nterms = _make_1d_setup(nelems=25)
        kle = create_fem_nystrom_nodes_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )

        # Lumped mass weights
        M = asm(mass, basis)
        w = bkd.asarray(np.array(M.sum(axis=1)).ravel())

        vecs = kle.eigenvectors()  # (nnodes, nterms)
        gram = vecs.T @ (w[:, None] * vecs)
        identity = bkd.array(np.eye(nterms))
        bkd.assert_allclose(gram, identity, atol=1e-6)

    def test_nystrom_nodes_shapes(self) -> None:
        """Nystrom-at-nodes has correct output shapes."""
        basis, kernel, _, bkd, nterms = _make_1d_setup(nelems=20)
        kle = create_fem_nystrom_nodes_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        nnodes = basis.mesh.p.shape[1]
        self.assertEqual(kle.eigenvalues().shape, (nterms,))
        self.assertEqual(kle.eigenvectors().shape, (nnodes, nterms))
        self.assertEqual(kle.nterms(), nterms)

    # --- Nystrom quadrature KLE tests ---

    def test_nystrom_quad_eigenvalues_vs_analytical(self) -> None:
        """Nystrom-at-quadrature eigenvalues converge to analytical values."""
        basis, kernel, analytical, bkd, nterms = _make_1d_setup(nelems=20)
        kle = create_fem_nystrom_quadrature_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        exact_eig = analytical.eigenvalues()
        bkd.assert_allclose(kle.eigenvalues(), bkd.array(exact_eig), rtol=5e-3)

    def test_nystrom_quad_weighted_orthonormality(self) -> None:
        """Nystrom-at-quadrature eigenvectors are orthonormal under dx weights."""
        basis, kernel, _, bkd, nterms = _make_1d_setup(nelems=20)
        kle = create_fem_nystrom_quadrature_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )

        w = bkd.asarray(basis.dx.ravel())
        vecs = kle.eigenvectors()
        gram = vecs.T @ (w[:, None] * vecs)
        identity = bkd.array(np.eye(nterms))
        bkd.assert_allclose(gram, identity, atol=1e-6)

    def test_nystrom_quad_shapes(self) -> None:
        """Nystrom-at-quadrature has correct output shapes."""
        basis, kernel, _, bkd, nterms = _make_1d_setup(nelems=20)
        kle = create_fem_nystrom_quadrature_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        nquad_total = basis.dx.size
        self.assertEqual(kle.eigenvalues().shape, (nterms,))
        self.assertEqual(kle.eigenvectors().shape, (nquad_total, nterms))
        self.assertEqual(kle.nterms(), nterms)

    # --- Cross-method agreement tests ---

    def test_all_methods_eigenvalues_agree(self) -> None:
        """All three FEM KLE methods produce similar eigenvalues."""
        basis, kernel, _, bkd, nterms = _make_1d_setup(nelems=30)

        kle_galerkin = create_fem_galerkin_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        kle_nodes = create_fem_nystrom_nodes_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        kle_quad = create_fem_nystrom_quadrature_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )

        # All should agree to within discretization error
        bkd.assert_allclose(
            kle_galerkin.eigenvalues(),
            kle_nodes.eigenvalues(),
            rtol=5e-3,
        )
        bkd.assert_allclose(
            kle_galerkin.eigenvalues(),
            kle_quad.eigenvalues(),
            rtol=5e-3,
        )

    def test_eigenvalue_convergence_with_refinement(self) -> None:
        """Eigenvalues improve with mesh refinement for Nystrom-at-nodes."""
        bkd = NumpyBkd()
        nterms = 3
        corr_len = 1.0
        dom_len = 2.0

        analytical = AnalyticalExponentialKLE1D(
            corr_len=corr_len,
            sigma2=1.0,
            dom_len=dom_len,
            nterms=nterms,
        )
        exact_eig = bkd.array(analytical.eigenvalues())

        errors = []
        for nelems in [10, 20, 40]:
            mesh = MeshLine(np.linspace(0.0, dom_len, nelems + 1))
            basis = Basis(mesh, ElementLineP1())
            lenscale = bkd.array([corr_len])
            kernel = ExponentialKernel(lenscale, (0.01, 100.0), 1, bkd)
            kle = create_fem_nystrom_nodes_kle(
                basis,
                kernel,
                nterms=nterms,
                sigma=1.0,
                bkd=bkd,
            )
            err = float(bkd.max(bkd.abs(kle.eigenvalues() - exact_eig) / exact_eig))
            errors.append(err)

        # Error should decrease with refinement
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_galerkin_eigenvalues_positive(self) -> None:
        """GalerkinKLE eigenvalues are all positive."""
        basis, kernel, _, bkd, nterms = _make_1d_setup(nelems=20)
        kle = create_fem_galerkin_kle(
            basis,
            kernel,
            nterms=nterms,
            sigma=1.0,
            bkd=bkd,
        )
        self.assertTrue(bkd.all_bool(kle.eigenvalues() > 0))


if __name__ == "__main__":
    unittest.main()
