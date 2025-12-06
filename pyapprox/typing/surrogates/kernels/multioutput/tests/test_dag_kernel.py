"""
Tests for DAGMultiOutputKernel.
"""

import unittest
from typing import Any, Generic
import numpy as np
from numpy.typing import NDArray
import networkx as nx

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.surrogates.kernels import (
    Matern32Kernel,
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.typing.surrogates.kernels.multioutput import (
    DAGMultiOutputKernel,
)
from pyapprox.typing.surrogates.kernels.scalings import PolynomialScaling


def create_matern_kernel(nu, lenscale, lenscale_bounds, nvars, bkd):
    """Create a Matern kernel with the specified nu value."""
    if nu == 1.5:
        return Matern32Kernel(lenscale, lenscale_bounds, nvars, bkd)
    elif nu == 2.5:
        return Matern52Kernel(lenscale, lenscale_bounds, nvars, bkd)
    elif nu == np.inf:
        return SquaredExponentialKernel(lenscale, lenscale_bounds, nvars, bkd)
    else:
        raise ValueError(f"Unsupported nu value: {nu}")


class TestDAGMultiOutputKernel(Generic[Array], unittest.TestCase):
    """Base test class for DAGMultiOutputKernel."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nvars = 1

    def _create_constant_scaling(self, value: float) -> PolynomialScaling:
        """Helper to create constant scaling using PolynomialScaling with degree 0."""
        return PolynomialScaling([value], (0.1, 2.0), self._bkd, nvars=self._nvars)

    def _create_linear_scaling(self, c0: float, c1: float) -> PolynomialScaling:
        """Helper to create linear scaling using PolynomialScaling with degree 1."""
        return PolynomialScaling([c0, c1], (0.1, 2.0), self._bkd)

    def test_sequential_structure(self):
        """Test sequential structure: 0 -> 1 -> 2."""
        # Create sequential DAG using networkx
        dag = nx.DiGraph()
        dag.add_nodes_from([0, 1, 2])
        dag.add_edges_from([(0, 1), (1, 2)])

        # Create discrepancy kernels
        kernels = [
            create_matern_kernel(2.5, [1.0], (0.1, 10.0), self._nvars, self._bkd),
            create_matern_kernel(2.5, [0.8], (0.1, 10.0), self._nvars, self._bkd),
            create_matern_kernel(2.5, [0.5], (0.1, 10.0), self._nvars, self._bkd),
        ]

        # Create constant scalings
        edge_scalings = {
            (0, 1): self._create_constant_scaling(0.9),
            (1, 2): self._create_constant_scaling(0.85),
        }

        # Create DAG kernel
        dag_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)

        # Test data
        X = self._bkd.array([[0.0, 0.5]])
        X_list = [X, X, X]

        # Compute kernel matrix
        K = dag_kernel(X_list)

        # Verify shape
        self.assertEqual(K.shape, (6, 6))

        # Verify symmetry
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

        # Verify positive definiteness (all eigenvalues positive)
        eigvals = np.linalg.eigvalsh(K)
        self.assertTrue(np.all(eigvals > -1e-10))

    def test_tree_structure_three_nodes(self):
        """Test tree structure with three nodes: 0 -> 1, 0 -> 2.

        This represents a structure where outputs 1 and 2 independently
        depend on output 0, with no direct connection between 1 and 2.
        """
        # Create tree DAG: root 0 with two independent children
        dag = nx.DiGraph()
        dag.add_nodes_from([0, 1, 2])
        dag.add_edges_from([(0, 1), (0, 2)])

        # Create discrepancy kernels
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.7], (0.1, 10.0), self._nvars, self._bkd)
        k2 = Matern52Kernel([0.7], (0.1, 10.0), self._nvars, self._bkd)
        kernels = [k0, k1, k2]

        # Create scalings for both edges from root
        edge_scalings = {
            (0, 1): self._create_constant_scaling(0.9),
            (0, 2): self._create_constant_scaling(0.85),
        }

        # Create DAG kernel
        dag_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)

        # Test data - use same points for simplicity
        X = self._bkd.array([[0.0, 1.0]])
        X_list = [X, X, X]

        # Compute kernel matrix
        K = dag_kernel(X_list)

        # Verify shape: 3 outputs × 2 points each = 6×6
        self.assertEqual(K.shape, (6, 6))

        # Verify block structure
        # K_00: just k0
        K_00 = k0(X, X)
        np.testing.assert_allclose(K[:2, :2], K_00, rtol=1e-10)

        # K_11: ρ_01² k0 + k1
        K_11_expected = 0.9**2 * k0(X, X) + k1(X, X)
        np.testing.assert_allclose(K[2:4, 2:4], K_11_expected, rtol=1e-10)

        # K_22: ρ_02² k0 + k2
        K_22_expected = 0.85**2 * k0(X, X) + k2(X, X)
        np.testing.assert_allclose(K[4:6, 4:6], K_22_expected, rtol=1e-10)

        # K_10: ρ_01 k0
        K_10_expected = 0.9 * k0(X, X)
        np.testing.assert_allclose(K[2:4, :2], K_10_expected, rtol=1e-10)

        # K_20: ρ_02 k0
        K_20_expected = 0.85 * k0(X, X)
        np.testing.assert_allclose(K[4:6, :2], K_20_expected, rtol=1e-10)

        # K_21: outputs 1 and 2 share only root 0, so K_21 = ρ_01 * ρ_02 * k0
        K_21_expected = 0.9 * 0.85 * k0(X, X)
        np.testing.assert_allclose(K[4:6, 2:4], K_21_expected, rtol=1e-10)

        # Verify symmetry
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_diamond_structure(self):
        """Test diamond structure: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3.

        Node 3 has multiple paths from node 0, so it benefits from
        information through both intermediate nodes.
        """
        # Create diamond DAG
        dag = nx.DiGraph()
        dag.add_nodes_from([0, 1, 2, 3])
        dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

        # Create kernels
        kernels = [
            Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.7], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.7], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.4], (0.1, 10.0), self._nvars, self._bkd),
        ]

        # Create scalings
        edge_scalings = {
            (0, 1): self._create_constant_scaling(0.9),
            (0, 2): self._create_constant_scaling(0.85),
            (1, 3): self._create_constant_scaling(0.8),
            (2, 3): self._create_constant_scaling(0.75),
        }

        # Create DAG kernel
        dag_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)

        # Test with single point for simplicity
        X = self._bkd.array([[0.0]])
        X_list = [X, X, X, X]

        # Compute kernel
        K = dag_kernel(X_list)

        # Verify shape
        self.assertEqual(K.shape, (4, 4))

        # For node 3, there are two paths from node 0:
        # Path 1: 0 -> 1 -> 3 with scaling ρ_01 * ρ_13 = 0.9 * 0.8 = 0.72
        # Path 2: 0 -> 2 -> 3 with scaling ρ_02 * ρ_23 = 0.85 * 0.75 = 0.6375
        #
        # K_33 includes all path-pair contributions:
        # For ancestor 0: all pairs of paths [0,1,3] and [0,2,3]
        # For ancestor 1: path [1,3]
        # For ancestor 2: path [2,3]
        # For ancestor 3 (self): path [3]

        k0_val = kernels[0](X, X)[0, 0]  # = 1
        k1_val = kernels[1](X, X)[0, 0]  # = 1
        k2_val = kernels[2](X, X)[0, 0]  # = 1
        k3_val = kernels[3](X, X)[0, 0]  # = 1

        rho_01_13 = 0.9 * 0.8  # = 0.72
        rho_02_23 = 0.85 * 0.75  # = 0.6375

        # K_33 = sum over ancestors k, sum over all path pairs from k to 3:
        # Ancestor 0: (w1 + w2)² where w1=0.72, w2=0.6375
        # = w1²*k0 + 2*w1*w2*k0 + w2²*k0 = (w1+w2)² * k0
        ancestor_0_contrib = (rho_01_13 + rho_02_23)**2 * k0_val

        # Ancestor 1: path [1,3] with scaling 0.8
        ancestor_1_contrib = 0.8**2 * k1_val

        # Ancestor 2: path [2,3] with scaling 0.75
        ancestor_2_contrib = 0.75**2 * k2_val

        # Ancestor 3 (self): path [3] with scaling 1
        ancestor_3_contrib = k3_val

        K_33_expected = (
            ancestor_0_contrib +
            ancestor_1_contrib +
            ancestor_2_contrib +
            ancestor_3_contrib
        )

        np.testing.assert_allclose(K[3, 3], K_33_expected, rtol=1e-10)

    def test_spatially_varying_scaling(self):
        """Test DAG kernel with spatially varying linear scaling."""
        # Simple 0 -> 1 structure
        dag = nx.DiGraph()
        dag.add_nodes_from([0, 1])
        dag.add_edge(0, 1)

        # Create kernels
        k0 = Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd)
        k1 = Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd)

        # Linear scaling: ρ(x) = 0.9 + 0.1*x
        edge_scalings = {
            (0, 1): self._create_linear_scaling(0.9, 0.1),
        }

        # Create kernel
        dag_kernel = DAGMultiOutputKernel(dag, [k0, k1], edge_scalings)

        # Test at points where we can verify scaling
        X0 = self._bkd.array([[-1.0, 0.0, 1.0]])
        X1 = self._bkd.array([[-1.0, 1.0]])

        K = dag_kernel([X0, X1])

        # Verify K_10 block has spatially varying scaling
        # At x=-1: ρ(-1) = 0.9 - 0.1 = 0.8
        # At x=0: ρ(0) = 0.9
        # At x=1: ρ(1) = 0.9 + 0.1 = 1.0

        # K_10[i,j] = ρ(X1[i]) * k0(X1[i], X0[j])
        # Note: only the scaling on the higher-fidelity side (output 1) applies
        rho_X1 = edge_scalings[(0, 1)].eval_scaling(X1)  # [[0.8], [1.0]]

        K_10_expected = rho_X1 * k0(X1, X0)

        np.testing.assert_allclose(K[3:5, :3], K_10_expected, rtol=1e-10)

    def test_block_format(self):
        """Test block_format output option."""
        # Simple 0 -> 1 structure
        dag = nx.DiGraph()
        dag.add_nodes_from([0, 1])
        dag.add_edge(0, 1)

        kernels = [
            Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.8], (0.1, 10.0), self._nvars, self._bkd),
        ]

        edge_scalings = {
            (0, 1): self._create_constant_scaling(0.9),
        }

        dag_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)

        # Test data
        X0 = self._bkd.array([[0.0, 1.0]])
        X1 = self._bkd.array([[0.5]])

        # Get blocks
        blocks = dag_kernel([X0, X1], block_format=True)

        # Verify structure
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 2)
        self.assertEqual(len(blocks[1]), 2)

        # Verify block shapes
        self.assertEqual(blocks[0][0].shape, (2, 2))  # K_00
        self.assertEqual(blocks[0][1].shape, (2, 1))  # K_01
        self.assertEqual(blocks[1][0].shape, (1, 2))  # K_10
        self.assertEqual(blocks[1][1].shape, (1, 1))  # K_11

    def test_default_constant_scaling(self):
        """Test that edges without explicit scaling default to constant 1.0."""
        dag = nx.DiGraph()
        dag.add_nodes_from([0, 1])
        dag.add_edge(0, 1)

        kernels = [
            Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.8], (0.1, 10.0), self._nvars, self._bkd),
        ]

        # Don't provide edge_scalings - should default to 1.0
        dag_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings=None)

        X = self._bkd.array([[0.0]])
        K = dag_kernel([X, X])

        # With ρ = 1.0, K_11 = k0 + k1
        K_11_expected = kernels[0](X, X) + kernels[1](X, X)
        np.testing.assert_allclose(K[1:, 1:], K_11_expected, rtol=1e-10)

    def test_precision_matrix_3model_fork(self) -> None:
        """
        Test precision matrix for 3-model fork DAG: 0 -> 1, 0 -> 2.

        In this structure, models 1 and 2 are conditionally independent
        given model 0. The precision matrix (inverse covariance) should
        have zeros in the blocks corresponding to this conditional independence:
        K_inv[1, 2] = 0 and K_inv[2, 1] = 0.
        """
        # Create fork DAG: 0 -> 1, 0 -> 2
        dag = nx.DiGraph()
        dag.add_nodes_from([0, 1, 2])
        dag.add_edges_from([(0, 1), (0, 2)])

        # Create kernels
        kernels = [
            Matern52Kernel([1.0], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd),
            Matern52Kernel([0.5], (0.1, 10.0), self._nvars, self._bkd),
        ]

        # Use constant scalings (PolynomialScaling with degree 0)
        edge_scalings = {
            (0, 1): PolynomialScaling([0.9], (0.5, 1.5), self._bkd, nvars=self._nvars),
            (0, 2): PolynomialScaling([0.85], (0.5, 1.5), self._bkd, nvars=self._nvars),
        }

        dag_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)

        # Create test points (same for all models)
        np.random.seed(42)
        n_samples = 5
        X_np = np.random.randn(self._nvars, n_samples)
        X = self._bkd.array(X_np)
        X_list = [X, X, X]

        # Compute full covariance matrix
        K = dag_kernel(X_list, block_format=False)
        K_np = self._bkd.to_numpy(K)

        # Compute precision matrix (use numpy for linalg operations in tests)
        K_inv = np.linalg.inv(K_np)

        # Extract blocks: each model has n_samples rows/cols
        # Block [1, 2] corresponds to rows [n_samples:2*n_samples, 2*n_samples:3*n_samples]
        K_inv_12 = K_inv[n_samples:2*n_samples, 2*n_samples:3*n_samples]
        K_inv_21 = K_inv[2*n_samples:3*n_samples, n_samples:2*n_samples]

        # Conditional independence: models 1 and 2 are independent given model 0
        # Therefore, precision matrix blocks K_inv[1, 2] and K_inv[2, 1] should be zero
        np.testing.assert_allclose(
            K_inv_12, 0.0, atol=1e-10,
            err_msg="Precision matrix K_inv[1,2] should be zero (conditional independence)"
        )
        np.testing.assert_allclose(
            K_inv_21, 0.0, atol=1e-10,
            err_msg="Precision matrix K_inv[2,1] should be zero (conditional independence)"
        )


class TestDAGMultiOutputKernelNumpy(TestDAGMultiOutputKernel[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
