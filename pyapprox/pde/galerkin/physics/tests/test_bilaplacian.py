"""Tests for BiLaplacianPrior."""

from typing import Any, Generic

import unittest

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.galerkin.mesh.structured import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import RobinBC
from pyapprox.pde.galerkin.bilaplacian import BiLaplacianPrior

from pyapprox.util.test_utils import load_tests  # noqa: F401

try:
    from skfem import MeshQuad, Basis
    from skfem.element import ElementQuad1
except ImportError:
    raise ImportError("scikit-fem required for tests")


class _SkfemMeshWrapper(Generic[Array]):
    """Thin wrapper around a raw skfem mesh for GalerkinMeshProtocol."""

    def __init__(self, skfem_mesh, bkd: Backend[Array]):
        self._skfem_mesh = skfem_mesh
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return self._skfem_mesh.p.shape[0]

    def nelements(self) -> int:
        return self._skfem_mesh.nelements

    def nnodes(self) -> int:
        return self._skfem_mesh.nvertices

    def nodes(self) -> Array:
        return self._bkd.asarray(
            self._skfem_mesh.p.astype(np.float64)
        )

    def elements(self) -> Array:
        return self._bkd.asarray(
            self._skfem_mesh.t.astype(np.int64)
        )

    def skfem_mesh(self) -> Any:
        return self._skfem_mesh

    def boundary_nodes(self, boundary_id: str) -> Array:
        facets = self._skfem_mesh.boundaries[boundary_id]
        nodes = np.unique(self._skfem_mesh.facets[:, facets])
        return self._bkd.asarray(nodes.astype(np.int64))


class TestBiLaplacianPrior(unittest.TestCase):
    """Tests for BiLaplacianPrior class."""

    def setUp(self):
        self._bkd = NumpyBkd()

    def _make_legacy_mesh_and_basis(self):
        """Create mesh matching legacy get_mesh([0,1,0,1], 1)."""
        skfem_mesh = (
            MeshQuad.init_tensor(
                np.linspace(0, 1, 3), np.linspace(0, 1, 3)
            )
            .refined(1)
            .with_boundaries(
                {
                    "left": lambda x: x[0] == 0.0,
                    "right": lambda x: x[0] == 1.0,
                    "bottom": lambda x: x[1] == 0.0,
                    "top": lambda x: x[1] == 1.0,
                }
            )
        )
        mesh_wrapper = _SkfemMeshWrapper(skfem_mesh, self._bkd)
        basis = LagrangeBasis(mesh_wrapper, degree=1)
        return basis

    def test_regression_matches_legacy(self):
        """Regression test matching legacy BiLaplacianPrior output.

        Replicates test_finite_elements.py:1242-1292.
        """
        basis = self._make_legacy_mesh_and_basis()
        gamma, delta = 1, 0.5
        aniso = np.array([[1.0, 0.0], [0.0, 1 / 20.0]])

        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma, delta, self._bkd, anisotropic_tensor=aniso
        )

        np.random.seed(1)
        samples = prior.rvs(2)
        samples_np = self._bkd.to_numpy(samples)

        reference_samples = np.array(
            [
                [0.42253777, 0.02968439],
                [0.17642463, -0.76812773],
                [0.21693848, -1.0817779],
                [0.40217442, 0.01687668],
                [0.42879581, -0.52078908],
                [0.36093003, -0.77522621],
                [-0.23375438, -0.1081412],
                [0.32022388, -0.5853546],
                [-0.1171723, -0.3977024],
                [-0.12397468, 1.4035203],
                [0.05872044, 0.29104518],
                [0.26992423, -0.22092672],
                [0.44287723, -0.66328011],
                [0.15643527, -0.75555845],
                [-0.03193684, 1.63982625],
                [-0.10756252, -0.10867872],
                [-0.16320015, -0.68081064],
                [0.17240703, -0.54417394],
                [-0.20866122, -0.38950437],
                [0.62974185, 1.50672728],
                [-0.05177725, -0.53887669],
                [-0.13495736, 1.61777272],
                [0.07485801, -0.60787973],
                [0.40618714, 1.80283875],
                [0.05299376, -0.52887806],
            ]
        )
        self._bkd.assert_allclose(
            self._bkd.asarray(samples_np),
            self._bkd.asarray(reference_samples),
            rtol=1e-6,
        )

    def test_sample_shape_2d(self):
        """Samples have correct shape (ndofs, nsamples)."""
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)
        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )
        rng = np.random.default_rng(42)
        for nsamples in [1, 5, 10]:
            samples = prior.rvs(nsamples, rng=rng)
            samples_np = self._bkd.to_numpy(samples)
            self.assertEqual(samples_np.shape, (basis.ndofs(), nsamples))

    def test_sample_shape_1d(self):
        """BiLaplacianPrior works on 1D meshes."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self._bkd)
        basis = LagrangeBasis(mesh, degree=1)
        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )
        rng = np.random.default_rng(42)
        samples = prior.rvs(5, rng=rng)
        samples_np = self._bkd.to_numpy(samples)
        self.assertEqual(samples_np.shape, (basis.ndofs(), 5))

    def test_stiffness_spd_2d(self):
        """Stiffness matrix is symmetric positive definite."""
        mesh = StructuredMesh2D(
            nx=3, ny=3, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)
        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )
        K_raw = prior.stiffness_matrix()
        K = K_raw.toarray() if issparse(K_raw) else self._bkd.to_numpy(K_raw)
        # Check symmetry
        self._bkd.assert_allclose(
            self._bkd.asarray(K),
            self._bkd.asarray(K.T),
            rtol=1e-12,
        )
        # Check positive definite (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(K)
        self.assertTrue(np.all(eigvals > 0))

    def test_stiffness_spd_1d(self):
        """Stiffness matrix is SPD for 1D case."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self._bkd)
        basis = LagrangeBasis(mesh, degree=1)
        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )
        K_raw = prior.stiffness_matrix()
        K = K_raw.toarray() if issparse(K_raw) else self._bkd.to_numpy(K_raw)
        self._bkd.assert_allclose(
            self._bkd.asarray(K),
            self._bkd.asarray(K.T),
            rtol=1e-12,
        )
        eigvals = np.linalg.eigvalsh(K)
        self.assertTrue(np.all(eigvals > 0))

    def test_zero_mean(self):
        """Samples have approximately zero mean."""
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)
        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )
        rng = np.random.default_rng(42)
        samples = prior.rvs(1000, rng=rng)
        samples_np = self._bkd.to_numpy(samples)
        mean = np.mean(samples_np, axis=1)
        self.assertTrue(
            np.all(np.abs(mean) < 0.3),
            f"Mean should be near zero, got max |mean|={np.max(np.abs(mean)):.4f}",
        )

    def test_anisotropy_effect(self):
        """Anisotropic tensor affects variance in different directions.

        With K = [[1, 0], [0, 1/20]], y-direction correlation is shorter,
        so y-direction variance at interior nodes should be different from
        x-direction behavior.
        """
        mesh = StructuredMesh2D(
            nx=8, ny=8, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)

        # Isotropic
        prior_iso = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )
        rng_iso = np.random.default_rng(42)
        samples_iso = self._bkd.to_numpy(prior_iso.rvs(500, rng=rng_iso))

        # Anisotropic: small K22 -> short y-correlation
        aniso = np.array([[1.0, 0.0], [0.0, 1 / 20.0]])
        prior_aniso = BiLaplacianPrior.with_uniform_robin(
            basis,
            gamma=1.0,
            delta=0.5,
            bkd=self._bkd,
            anisotropic_tensor=aniso,
        )
        rng_aniso = np.random.default_rng(42)
        samples_aniso = self._bkd.to_numpy(
            prior_aniso.rvs(500, rng=rng_aniso)
        )

        # Anisotropic samples should have different variance than isotropic
        var_iso = np.var(samples_iso, axis=1)
        var_aniso = np.var(samples_aniso, axis=1)
        self.assertFalse(
            np.allclose(var_iso, var_aniso, rtol=0.1),
            "Anisotropic and isotropic should produce different variances",
        )

    def test_parameter_scaling(self):
        """Different gamma*delta produces different sample variance.

        Larger gamma and delta increase the stiffness (stronger diffusion
        and reaction), resulting in smaller sample variance.
        """
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)

        # Small gamma*delta -> weaker stiffness -> larger variance
        prior_small = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=0.1, delta=0.1, bkd=self._bkd
        )
        rng1 = np.random.default_rng(42)
        samples_small = self._bkd.to_numpy(prior_small.rvs(500, rng=rng1))

        # Large gamma*delta -> stronger stiffness -> smaller variance
        prior_large = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=10.0, delta=10.0, bkd=self._bkd
        )
        rng2 = np.random.default_rng(42)
        samples_large = self._bkd.to_numpy(prior_large.rvs(500, rng=rng2))

        var_small = np.mean(np.var(samples_small, axis=1))
        var_large = np.mean(np.var(samples_large, axis=1))
        self.assertGreater(
            var_small,
            var_large,
            f"Smaller gamma*delta should give larger variance: "
            f"{var_small:.4f} vs {var_large:.4f}",
        )

    def test_custom_robin_alpha(self):
        """Custom robin_alpha produces different samples than default."""
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)

        prior_default = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )
        prior_custom = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd, robin_alpha=5.0
        )

        rng1 = np.random.default_rng(42)
        samples_default = self._bkd.to_numpy(prior_default.rvs(3, rng=rng1))
        rng2 = np.random.default_rng(42)
        samples_custom = self._bkd.to_numpy(prior_custom.rvs(3, rng=rng2))

        self.assertFalse(
            np.allclose(samples_default, samples_custom),
            "Custom robin_alpha should produce different samples",
        )

    def test_custom_boundary_conditions(self):
        """Constructor accepts user-created RobinBC list."""
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)

        # Only apply Robin BC on left and right
        robin_bcs = [
            RobinBC(basis, "left", alpha=1.0, value_func=0.0, bkd=self._bkd),
            RobinBC(
                basis, "right", alpha=1.0, value_func=0.0, bkd=self._bkd
            ),
        ]
        prior = BiLaplacianPrior(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd,
            boundary_conditions=robin_bcs,
        )
        rng = np.random.default_rng(42)
        samples = prior.rvs(3, rng=rng)
        self.assertEqual(
            self._bkd.to_numpy(samples).shape, (basis.ndofs(), 3)
        )

    def test_rng_reproducibility(self):
        """Using the same rng seed produces identical samples."""
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)
        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )

        rng1 = np.random.default_rng(123)
        samples1 = self._bkd.to_numpy(prior.rvs(5, rng=rng1))
        rng2 = np.random.default_rng(123)
        samples2 = self._bkd.to_numpy(prior.rvs(5, rng=rng2))

        self._bkd.assert_allclose(
            self._bkd.asarray(samples1),
            self._bkd.asarray(samples2),
            rtol=1e-12,
        )

    def test_invalid_anisotropic_tensor_shape(self):
        """Raises ValueError for wrong anisotropic_tensor shape."""
        mesh = StructuredMesh2D(
            nx=4, ny=4, bounds=[[0, 1], [0, 1]], bkd=self._bkd
        )
        basis = LagrangeBasis(mesh, degree=1)

        with self.assertRaises(ValueError):
            BiLaplacianPrior.with_uniform_robin(
                basis,
                gamma=1.0,
                delta=0.5,
                bkd=self._bkd,
                anisotropic_tensor=np.array([[1.0, 0.0, 0.0]]),
            )

    def test_repr(self):
        """repr includes ndofs, gamma, delta."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self._bkd)
        basis = LagrangeBasis(mesh, degree=1)
        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=1.0, delta=0.5, bkd=self._bkd
        )
        r = repr(prior)
        self.assertIn("BiLaplacianPrior", r)
        self.assertIn("ndofs=11", r)
        self.assertIn("gamma=1.0", r)
        self.assertIn("delta=0.5", r)
