"""Tests for visualization module."""

import unittest

import numpy as np

# Skip tests if matplotlib is not available
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestAllocationVisualization(unittest.TestCase):
    """Tests for allocation visualization functions."""

    def test_plot_allocation(self):
        """Test plot_allocation function."""
        from pyapprox.typing.stats.visualization import plot_allocation

        nsamples = np.array([100, 500, 2000])
        ax = plot_allocation(nsamples)

        self.assertIsNotNone(ax)
        plt.close()

    def test_plot_allocation_with_names(self):
        """Test plot_allocation with custom model names."""
        from pyapprox.typing.stats.visualization import plot_allocation

        nsamples = np.array([100, 500, 2000])
        ax = plot_allocation(nsamples, model_names=["HF", "MF", "LF"])

        self.assertIsNotNone(ax)
        plt.close()

    def test_plot_samples_per_model(self):
        """Test plot_samples_per_model function."""
        from pyapprox.typing.stats.visualization import plot_samples_per_model

        results = {
            "MFMC": np.array([100, 500, 2000]),
            "MLMC": np.array([150, 300, 1500]),
        }
        ax = plot_samples_per_model(results)

        self.assertIsNotNone(ax)
        plt.close()


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestComparisonVisualization(unittest.TestCase):
    """Tests for comparison visualization functions."""

    def test_plot_estimator_comparison(self):
        """Test plot_estimator_comparison function."""
        from pyapprox.typing.stats.visualization import plot_estimator_comparison

        results = {
            "mc": {"variance": 0.1, "nsamples": np.array([10])},
            "mfmc": {"variance": 0.05, "nsamples": np.array([10, 50])},
        }
        ax = plot_estimator_comparison(results)

        self.assertIsNotNone(ax)
        plt.close()

    def test_plot_variance_vs_cost(self):
        """Test plot_variance_vs_cost function."""
        from pyapprox.typing.stats.visualization import plot_variance_vs_cost

        costs = [10, 100, 1000]
        variances = {
            "MC": [0.1, 0.01, 0.001],
            "MFMC": [0.05, 0.005, 0.0005],
        }
        ax = plot_variance_vs_cost(costs, variances)

        self.assertIsNotNone(ax)
        plt.close()


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestCorrelationVisualization(unittest.TestCase):
    """Tests for correlation visualization functions."""

    def test_plot_correlation_matrix(self):
        """Test plot_correlation_matrix function."""
        from pyapprox.typing.stats.visualization import plot_correlation_matrix

        cov = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        ax = plot_correlation_matrix(cov)

        self.assertIsNotNone(ax)
        plt.close()

    def test_plot_correlation_matrix_with_names(self):
        """Test plot_correlation_matrix with custom model names."""
        from pyapprox.typing.stats.visualization import plot_correlation_matrix

        cov = np.array([[1.0, 0.9], [0.9, 1.0]])
        ax = plot_correlation_matrix(cov, model_names=["HF", "LF"])

        self.assertIsNotNone(ax)
        plt.close()


if __name__ == "__main__":
    unittest.main()
