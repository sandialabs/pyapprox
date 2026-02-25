"""Tests for sparse grid point plotter.

Numpy-only tests using matplotlib Agg backend.
"""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyapprox.surrogates.sparsegrids.plot import (
    plot_sparse_grid_points,
)
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestPlotSparseGridPoints(unittest.TestCase):
    """Smoke tests for plot_sparse_grid_points."""

    def test_2d_selected_only(self) -> None:
        """2D plot with selected samples only, no candidates."""
        fig, ax = plt.subplots()
        selected = np.array([[0.0, 0.5, -0.5, 1.0],
                             [0.0, 0.5, -0.5, -1.0]])
        result = plot_sparse_grid_points(ax, selected)
        self.assertIn("selected", result)
        self.assertIsNotNone(result["selected"])
        self.assertIsNone(result["candidate"])
        plt.close(fig)

    def test_2d_with_candidates(self) -> None:
        """2D plot with both selected and candidate samples."""
        fig, ax = plt.subplots()
        selected = np.array([[0.0, 0.5], [0.0, 0.5]])
        candidates = np.array([[-0.5, 1.0], [-0.5, -1.0]])
        result = plot_sparse_grid_points(ax, selected, candidates)
        self.assertIsNotNone(result["selected"])
        self.assertIsNotNone(result["candidate"])
        plt.close(fig)

    def test_1d_selected_only(self) -> None:
        """1D plot with selected samples (scatter at y=0)."""
        fig, ax = plt.subplots()
        selected = np.array([[0.0, 0.5, -0.5, 1.0, -1.0]])
        result = plot_sparse_grid_points(ax, selected)
        self.assertIsNotNone(result["selected"])
        self.assertIsNone(result["candidate"])
        plt.close(fig)

    def test_1d_with_candidates(self) -> None:
        """1D plot with both selected and candidate samples."""
        fig, ax = plt.subplots()
        selected = np.array([[0.0, 0.5]])
        candidates = np.array([[-0.5, 1.0]])
        result = plot_sparse_grid_points(ax, selected, candidates)
        self.assertIsNotNone(result["selected"])
        self.assertIsNotNone(result["candidate"])
        plt.close(fig)

    def test_axis_labels_and_title(self) -> None:
        """Axis labels and title are applied."""
        fig, ax = plt.subplots()
        selected = np.array([[0.0, 1.0], [0.0, 1.0]])
        plot_sparse_grid_points(
            ax, selected,
            axis_labels=["$z_1$", "$z_2$"],
            title="Test Grid",
        )
        self.assertEqual(ax.get_xlabel(), "$z_1$")
        self.assertEqual(ax.get_ylabel(), "$z_2$")
        self.assertEqual(ax.get_title(), "Test Grid")
        plt.close(fig)

    def test_empty_candidates(self) -> None:
        """Empty candidate array handled correctly."""
        fig, ax = plt.subplots()
        selected = np.array([[0.0, 1.0], [0.0, 1.0]])
        candidates = np.zeros((2, 0))
        result = plot_sparse_grid_points(ax, selected, candidates)
        self.assertIsNotNone(result["selected"])
        self.assertIsNone(result["candidate"])
        plt.close(fig)

    def test_3d_raises(self) -> None:
        """3D input raises ValueError."""
        fig, ax = plt.subplots()
        selected = np.array([[0.0], [0.0], [0.0]])
        with self.assertRaises(ValueError):
            plot_sparse_grid_points(ax, selected)
        plt.close(fig)

    def test_returns_expected_keys(self) -> None:
        """Return dict has exactly 'selected' and 'candidate' keys."""
        fig, ax = plt.subplots()
        selected = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = plot_sparse_grid_points(ax, selected)
        self.assertEqual(set(result.keys()), {"selected", "candidate"})
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
