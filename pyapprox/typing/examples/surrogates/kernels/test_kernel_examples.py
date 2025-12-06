"""
Test that kernel examples can run without matplotlib.

This script tests the core functionality of kernel examples without requiring
matplotlib for plotting.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys

from pyapprox.typing.surrogates.kernels import MaternKernel, PolynomialScaling
from pyapprox.typing.surrogates.kernels.multioutput import (
    MultiLevelKernel,
    DAGMultiOutputKernel,
)
from pyapprox.typing.surrogates.kernels.plot_kernel_matrix import (
    plot_kernel_matrix_heatmap,
    plot_kernel_matrix_surface,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd


class TestKernelMatrixPlotting(unittest.TestCase):
    """Test kernel matrix plotting examples work."""

    def setUp(self):
        self.bkd = NumpyBkd()

    @patch('pyapprox.typing.surrogates.kernels.plot_kernel_matrix.np.linspace')
    def test_matern_kernel_matrix(self, mock_linspace):
        """Test Matern kernel matrix computation."""
        # Create kernel
        kernel = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, self.bkd)

        # Mock linspace
        x = self.bkd.to_numpy(self.bkd.array([0.0, 1.0, 2.0]))
        mock_linspace.return_value = x

        # Mock axes
        mock_ax = MagicMock()
        mock_ax.imshow.return_value = MagicMock()

        # Test heatmap
        plot_kernel_matrix_heatmap(kernel, (-2.0, 2.0), mock_ax, npts=3)
        self.assertTrue(mock_ax.imshow.called)

        # Verify kernel matrix shape
        K_plot = mock_ax.imshow.call_args[0][0]
        self.assertEqual(K_plot.shape, (3, 3))

    def test_multilevel_kernel_matrix(self):
        """Test multi-level kernel matrix computation."""
        # Create 2-level kernel
        k0 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, self.bkd)
        k1 = MaternKernel(2.5, [0.5], (0.1, 10.0), 1, self.bkd)
        scaling = PolynomialScaling([0.9], (0.5, 1.5), self.bkd, nvars=1)

        ml_kernel = MultiLevelKernel([k0, k1], [scaling])

        # Sample points
        x = self.bkd.array([[-1.0, 0.0, 1.0]])
        X_list = [x, x]

        # Compute kernel matrix
        K_full = ml_kernel(X_list)

        # Check shape
        self.assertEqual(K_full.shape, (6, 6))  # 3 + 3 samples

        # Check block structure
        K_np = self.bkd.to_numpy(K_full)
        n0 = 3
        n1 = 3

        K_00 = K_np[:n0, :n0]
        K_01 = K_np[:n0, n0:]
        K_10 = K_np[n0:, :n0]
        K_11 = K_np[n0:, n0:]

        # All blocks should have correct shapes
        self.assertEqual(K_00.shape, (3, 3))
        self.assertEqual(K_01.shape, (3, 3))
        self.assertEqual(K_10.shape, (3, 3))
        self.assertEqual(K_11.shape, (3, 3))

        # Check symmetry: K_01.T should equal K_10
        self.assertTrue(
            self.bkd.allclose(
                self.bkd.array(K_01.T),
                self.bkd.array(K_10),
                rtol=1e-10
            )
        )

    def test_dag_kernel_matrix_nonhierarchical(self):
        """Test DAG kernel matrix with non-hierarchical structure."""
        import networkx as nx

        # Create non-hierarchical DAG (diamond structure)
        # Output 2 depends on both outputs 0 and 1
        dag = nx.DiGraph()
        dag.add_edges_from([
            (0, 2),  # 0 -> 2
            (1, 2),  # 1 -> 2
        ])

        # Create base kernels for each output
        k0 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, self.bkd)
        k1 = MaternKernel(2.5, [0.8], (0.1, 10.0), 1, self.bkd)
        k2 = MaternKernel(2.5, [0.6], (0.1, 10.0), 1, self.bkd)

        base_kernels = [k0, k1, k2]

        # Create scalings for edges
        # (0, 2) edge
        scaling_02 = PolynomialScaling([0.9], (0.5, 1.5), self.bkd, nvars=1)
        # (1, 2) edge
        scaling_12 = PolynomialScaling([0.85], (0.5, 1.5), self.bkd, nvars=1)

        scalings = {(0, 2): scaling_02, (1, 2): scaling_12}

        # Create DAG kernel
        dag_kernel = DAGMultiOutputKernel(dag, base_kernels, scalings)

        # Sample points
        x = self.bkd.array([[-1.0, 0.0, 1.0]])
        X_list = [x, x, x]  # Same points for all outputs

        # Compute kernel matrix
        K_full = dag_kernel(X_list)

        # Check shape
        self.assertEqual(K_full.shape, (9, 9))  # 3 outputs × 3 samples each

        # Check symmetry
        K_np = self.bkd.to_numpy(K_full)
        self.assertTrue(
            self.bkd.allclose(
                self.bkd.array(K_np),
                self.bkd.array(K_np.T),
                rtol=1e-10
            )
        )

        # Extract blocks
        n0 = 3
        n1 = 3
        n2 = 3

        K_00 = K_np[:n0, :n0]
        K_11 = K_np[n0:2*n0, n0:2*n0]
        K_22 = K_np[2*n0:, 2*n0:]
        K_02 = K_np[:n0, 2*n0:]
        K_12 = K_np[n0:2*n0, 2*n0:]

        # Auto-covariances should be positive definite
        self.assertTrue(self.bkd.all_bool(self.bkd.array(K_00.diagonal()) > 0))
        self.assertTrue(self.bkd.all_bool(self.bkd.array(K_11.diagonal()) > 0))
        self.assertTrue(self.bkd.all_bool(self.bkd.array(K_22.diagonal()) > 0))

        # Cross-covariances should be non-zero (due to correlations)
        self.assertTrue(self.bkd.all_bool(self.bkd.array(K_02) != 0))
        self.assertTrue(self.bkd.all_bool(self.bkd.array(K_12) != 0))


if __name__ == "__main__":
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKernelMatrixPlotting)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
