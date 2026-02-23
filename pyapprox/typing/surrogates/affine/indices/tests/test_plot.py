"""Tests for the index-set plotting utilities.

Numpy-only (no dual-backend needed).  All tests use the ``Agg`` backend so
they run headlessly in CI.
"""

import unittest

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from pyapprox.typing.surrogates.affine.indices.plot import (  # noqa: E402
    _resolve_colors,
    _resolve_labels,
    format_index_axes,
    plot_index_sets,
    plot_indices_2d,
    plot_indices_3d,
)


# Simple 2D index set: (0,0), (1,0), (0,1), (2,0)
_INDICES_2D = np.array([[0, 1, 0, 2], [0, 0, 1, 0]])

# Simple 3D index set
_INDICES_3D = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


# ---- _resolve_colors ----

class TestResolveColors(unittest.TestCase):
    def test_uniform_string(self):
        result = _resolve_colors("red", _INDICES_2D)
        self.assertEqual(result, ["red"] * 4)

    def test_per_index_list(self):
        cols = ["a", "b", "c", "d"]
        result = _resolve_colors(cols, _INDICES_2D)
        self.assertEqual(result, cols)

    def test_callable(self):
        result = _resolve_colors(
            lambda idx: "blue" if idx[0] == 0 else "red",
            _INDICES_2D,
        )
        self.assertEqual(result, ["blue", "red", "blue", "red"])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _resolve_colors(["a", "b"], _INDICES_2D)


# ---- _resolve_labels ----

class TestResolveLabels(unittest.TestCase):
    def test_none(self):
        result = _resolve_labels(None, _INDICES_2D)
        self.assertEqual(result, [None] * 4)

    def test_true_auto(self):
        result = _resolve_labels(True, _INDICES_2D)
        self.assertEqual(result, ["(0,0)", "(1,0)", "(0,1)", "(2,0)"])

    def test_per_index_list(self):
        lbl = ["a", "b", "c", "d"]
        result = _resolve_labels(lbl, _INDICES_2D)
        self.assertEqual(result, lbl)

    def test_callable(self):
        result = _resolve_labels(
            lambda idx: f"L{idx[0]}",
            _INDICES_2D,
        )
        self.assertEqual(result, ["L0", "L1", "L0", "L2"])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _resolve_labels(["a"], _INDICES_2D)


# ---- plot_indices_2d ----

class TestPlotIndices2D(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def test_smoke(self):
        rects, texts = plot_indices_2d(self.ax, _INDICES_2D)
        self.assertEqual(len(rects), 4)
        self.assertEqual(len(texts), 0)  # no labels by default

    def test_with_labels(self):
        rects, texts = plot_indices_2d(self.ax, _INDICES_2D, labels=True)
        self.assertEqual(len(rects), 4)
        self.assertEqual(len(texts), 4)
        self.assertEqual(texts[0].get_text(), "(0,0)")

    def test_artist_types(self):
        rects, texts = plot_indices_2d(
            self.ax, _INDICES_2D, labels=True,
        )
        for r in rects:
            self.assertIsInstance(r, Rectangle)
        for t in texts:
            self.assertIsInstance(t, Text)

    def test_callable_colors(self):
        rects, _ = plot_indices_2d(
            self.ax, _INDICES_2D,
            colors=lambda idx: "#FF0000" if idx[0] == 0 else "#0000FF",
        )
        self.assertEqual(len(rects), 4)

    def test_wrong_nvars_raises(self):
        with self.assertRaises(ValueError):
            plot_indices_2d(self.ax, _INDICES_3D)

    def test_rect_position(self):
        rects, _ = plot_indices_2d(
            self.ax, _INDICES_2D, box_width=0.8, box_height=0.8,
        )
        # First index is (0, 0), rect should be centred there
        r0 = rects[0]
        x, y = r0.get_xy()
        np.testing.assert_allclose(x, -0.4)
        np.testing.assert_allclose(y, -0.4)
        np.testing.assert_allclose(r0.get_width(), 0.8)
        np.testing.assert_allclose(r0.get_height(), 0.8)

    def test_custom_box_size(self):
        rects, _ = plot_indices_2d(
            self.ax, _INDICES_2D, box_width=1.0, box_height=0.6,
        )
        r = rects[0]
        np.testing.assert_allclose(r.get_width(), 1.0)
        np.testing.assert_allclose(r.get_height(), 0.6)


# ---- plot_indices_3d ----

class TestPlotIndices3D(unittest.TestCase):
    def setUp(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")

    def tearDown(self):
        plt.close(self.fig)

    def test_smoke(self):
        voxel_result, texts = plot_indices_3d(self.ax, _INDICES_3D)
        self.assertIsNotNone(voxel_result)
        self.assertEqual(len(texts), 0)

    def test_with_labels(self):
        _, texts = plot_indices_3d(self.ax, _INDICES_3D, labels=True)
        self.assertEqual(len(texts), 4)
        self.assertEqual(texts[0].get_text(), "(0,0,0)")

    def test_callable_colors(self):
        _, texts = plot_indices_3d(
            self.ax, _INDICES_3D,
            colors=lambda idx: "#2C7FB8CC" if idx[2] == 0 else "#E74C3CCC",
        )
        self.assertEqual(len(texts), 0)

    def test_wrong_nvars_raises(self):
        with self.assertRaises(ValueError):
            plot_indices_3d(self.ax, _INDICES_2D)


# ---- format_index_axes ----

class TestFormatIndexAxes(unittest.TestCase):
    def test_2d_limits_and_ticks(self):
        fig, ax = plt.subplots()
        format_index_axes(ax, _INDICES_2D, pad=0.5)
        self.assertAlmostEqual(ax.get_xlim()[0], -0.5)
        self.assertAlmostEqual(ax.get_xlim()[1], 2.5)
        self.assertAlmostEqual(ax.get_ylim()[0], -0.5)
        self.assertAlmostEqual(ax.get_ylim()[1], 1.5)
        # Integer ticks
        np.testing.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        np.testing.assert_array_equal(ax.get_yticks(), [0, 1])
        plt.close(fig)

    def test_2d_axis_labels(self):
        fig, ax = plt.subplots()
        format_index_axes(
            ax, _INDICES_2D,
            axis_labels=["dim 1", "dim 2"],
        )
        self.assertEqual(ax.get_xlabel(), "dim 1")
        self.assertEqual(ax.get_ylabel(), "dim 2")
        plt.close(fig)

    def test_2d_max_indices_overrides_limits(self):
        fig, ax = plt.subplots()
        # _INDICES_2D has max (2, 1), but we fix limits to (5, 3)
        format_index_axes(ax, _INDICES_2D, pad=0.5, max_indices=[5, 3])
        self.assertAlmostEqual(ax.get_xlim()[1], 5.5)
        self.assertAlmostEqual(ax.get_ylim()[1], 3.5)
        np.testing.assert_array_equal(ax.get_xticks(), [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(ax.get_yticks(), [0, 1, 2, 3])
        plt.close(fig)

    def test_3d_limits(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        format_index_axes(ax, _INDICES_3D, pad=0.5)
        self.assertAlmostEqual(ax.get_xlim()[0], -0.5)
        # Voxels are centred on integer coords, same formula as 2D
        self.assertAlmostEqual(ax.get_xlim()[1], 1.5)  # max=1 + pad=0.5
        self.assertAlmostEqual(ax.get_zlim()[1], 1.5)  # max=1 + pad=0.5
        plt.close(fig)

    def test_3d_view_init(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        format_index_axes(ax, _INDICES_3D, view_init=(20, 60))
        self.assertAlmostEqual(ax.elev, 20)
        self.assertAlmostEqual(ax.azim, 60)
        plt.close(fig)


# ---- plot_index_sets ----

class TestPlotIndexSets(unittest.TestCase):
    def test_selected_only_2d(self):
        fig, ax = plt.subplots()
        result = plot_index_sets(ax, _INDICES_2D, selected_labels=True)
        self.assertEqual(len(result["selected"][0]), 4)
        self.assertEqual(len(result["selected"][1]), 4)
        self.assertEqual(len(result["candidates"][0]), 0)
        plt.close(fig)

    def test_selected_and_candidates_2d(self):
        fig, ax = plt.subplots()
        cand = np.array([[3, 0], [0, 2]])
        result = plot_index_sets(
            ax, _INDICES_2D, cand,
            selected_labels=True, candidate_labels=True,
        )
        self.assertEqual(len(result["selected"][0]), 4)
        self.assertEqual(len(result["candidates"][0]), 2)
        plt.close(fig)

    def test_3d_dispatch(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        result = plot_index_sets(ax, _INDICES_3D)
        self.assertIn("selected", result)
        plt.close(fig)

    def test_wrong_nvars_raises(self):
        fig, ax = plt.subplots()
        bad = np.array([[0, 1], [0, 0], [0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            plot_index_sets(ax, bad)
        plt.close(fig)

    def test_mismatched_candidate_nvars_raises(self):
        fig, ax = plt.subplots()
        cand_bad = np.array([[0], [0], [0]])
        with self.assertRaises(ValueError):
            plot_index_sets(ax, _INDICES_2D, cand_bad)
        plt.close(fig)

    def test_callable_colors(self):
        fig, ax = plt.subplots()
        result = plot_index_sets(
            ax, _INDICES_2D,
            selected_colors=lambda idx: "blue" if idx[1] == 0 else "red",
        )
        self.assertEqual(len(result["selected"][0]), 4)
        plt.close(fig)

    def test_format_axes_applies_labels(self):
        fig, ax = plt.subplots()
        plot_index_sets(
            ax, _INDICES_2D,
            axis_labels=["$k_1$", "$k_2$"],
        )
        self.assertEqual(ax.get_xlabel(), "$k_1$")
        self.assertEqual(ax.get_ylabel(), "$k_2$")
        plt.close(fig)

    def test_format_axes_false(self):
        fig, ax = plt.subplots()
        plot_index_sets(
            ax, _INDICES_2D,
            format_axes=False,
        )
        # Limits should not have been set by format_index_axes
        # (default matplotlib limits)
        self.assertEqual(ax.get_xlabel(), "")
        plt.close(fig)

    def test_max_indices_fixes_limits(self):
        fig, ax = plt.subplots()
        # Small index set but large fixed limits — for animation stability
        small = np.array([[0, 1], [0, 0]])
        plot_index_sets(ax, small, max_indices=[8, 3])
        self.assertAlmostEqual(ax.get_xlim()[1], 8.5)
        self.assertAlmostEqual(ax.get_ylim()[1], 3.5)
        plt.close(fig)


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401, E402
