"""Tests for MFNet builders and helpers."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.surrogates.mfnets.builders import (
    build_chain_mfnet,
    build_dag_mfnet,
)
from pyapprox.surrogates.mfnets.helpers import (
    generate_synthetic_data,
    randomize_coefficients,
)
from pyapprox.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    MFNetNode,
    RootMFNetNode,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401

# --- Builder Tests ---


class TestBuildChainMFNet(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_two_node_chain(self) -> None:
        """2-node chain: leaf(0) -> root(1)."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=1, nqoi=1, nnodes=2, bkd=bkd)
        self.assertEqual(len(net.topo_order()), 2)
        self.assertIsInstance(net.node(0), LeafMFNetNode)
        self.assertIsInstance(net.node(1), RootMFNetNode)

    def test_three_node_chain(self) -> None:
        """3-node chain: leaf(0) -> interior(1) -> root(2)."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=1, nqoi=1, nnodes=3, bkd=bkd)
        self.assertEqual(len(net.topo_order()), 3)
        self.assertIsInstance(net.node(0), LeafMFNetNode)
        self.assertIsInstance(net.node(1), MFNetNode)
        self.assertIsInstance(net.node(2), RootMFNetNode)

    def test_chain_forward_eval(self) -> None:
        """Chain network evaluates and returns correct shape."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=1, nqoi=1, nnodes=2, bkd=bkd)
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        out = net(samples)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 10)

    def test_chain_noise_std_list(self) -> None:
        """Per-node noise_std list is propagated."""
        bkd = self._bkd
        net = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=3,
            bkd=bkd,
            noise_std=[0.1, 0.2, 0.3],
        )
        for i, expected in enumerate([0.1, 0.2, 0.3]):
            actual = float(bkd.to_numpy(net.node(i).noise_std()))
            self.assertAlmostEqual(actual, expected, places=6)

    def test_chain_noise_std_wrong_length_raises(self) -> None:
        """noise_std list with wrong length raises ValueError."""
        bkd = self._bkd
        with self.assertRaises(ValueError):
            build_chain_mfnet(
                nvars=1,
                nqoi=1,
                nnodes=3,
                bkd=bkd,
                noise_std=[0.1, 0.2],
            )

    def test_single_node_raises(self) -> None:
        """nnodes=1 raises ValueError."""
        bkd = self._bkd
        with self.assertRaises(ValueError):
            build_chain_mfnet(nvars=1, nqoi=1, nnodes=1, bkd=bkd)

    def test_custom_levels(self) -> None:
        """Custom polynomial levels produce different-sized models."""
        bkd = self._bkd
        net_low = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=2,
            bkd=bkd,
            leaf_level=1,
            delta_level=1,
        )
        net_high = build_chain_mfnet(
            nvars=1,
            nqoi=1,
            nnodes=2,
            bkd=bkd,
            leaf_level=5,
            delta_level=5,
        )
        # Higher levels = more parameters
        nparams_low = net_low.hyp_list().nparams()
        nparams_high = net_high.hyp_list().nparams()
        self.assertGreater(nparams_high, nparams_low)

    def test_multivariate(self) -> None:
        """Chain works with nvars > 1."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=2, nqoi=1, nnodes=2, bkd=bkd)
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        out = net(samples)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 10)


# --- DAG Builder Tests ---


class TestBuildDAGMFNet(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_diamond_dag(self) -> None:
        """Diamond: 0->1, 0->2, 1->3, 2->3."""
        bkd = self._bkd
        net = build_dag_mfnet(
            nvars=1,
            nqoi=1,
            edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
            bkd=bkd,
        )
        self.assertEqual(len(net.topo_order()), 4)
        self.assertIsInstance(net.node(0), LeafMFNetNode)
        self.assertIsInstance(net.node(3), RootMFNetNode)
        # Nodes 1 and 2 are interior (have both children and parents)
        self.assertIsInstance(net.node(1), MFNetNode)
        self.assertIsInstance(net.node(2), MFNetNode)

    def test_diamond_forward_eval(self) -> None:
        """Diamond DAG evaluates without error."""
        bkd = self._bkd
        net = build_dag_mfnet(
            nvars=1,
            nqoi=1,
            edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
            bkd=bkd,
        )
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        out = net(samples)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 10)

    def test_dag_chain_matches_topology(self) -> None:
        """DAG with chain edges has correct topology."""
        bkd = self._bkd
        net = build_dag_mfnet(
            nvars=1,
            nqoi=1,
            edges=[(0, 1), (1, 2)],
            bkd=bkd,
        )
        self.assertEqual(len(net.topo_order()), 3)
        self.assertIsInstance(net.node(0), LeafMFNetNode)
        self.assertIsInstance(net.node(1), MFNetNode)
        self.assertIsInstance(net.node(2), RootMFNetNode)

    def test_dag_per_node_config(self) -> None:
        """Per-node config overrides are applied."""
        bkd = self._bkd
        net = build_dag_mfnet(
            nvars=1,
            nqoi=1,
            edges=[(0, 1)],
            bkd=bkd,
            node_configs={
                0: {"leaf_level": 5},
                1: {"noise_std": 0.5},
            },
        )
        actual_noise = float(bkd.to_numpy(net.node(1).noise_std()))
        self.assertAlmostEqual(actual_noise, 0.5, places=6)

    def test_empty_edges_raises(self) -> None:
        """Empty edge list raises ValueError."""
        bkd = self._bkd
        with self.assertRaises(ValueError):
            build_dag_mfnet(nvars=1, nqoi=1, edges=[], bkd=bkd)

    def test_two_leaf_dag(self) -> None:
        """Two leaves feeding one root: 0->2, 1->2."""
        bkd = self._bkd
        net = build_dag_mfnet(
            nvars=1,
            nqoi=1,
            edges=[(0, 2), (1, 2)],
            bkd=bkd,
        )
        self.assertIsInstance(net.node(0), LeafMFNetNode)
        self.assertIsInstance(net.node(1), LeafMFNetNode)
        self.assertIsInstance(net.node(2), RootMFNetNode)
        # Root should have nscaled_qoi = 2 (one from each leaf)
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        out = net(samples)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 10)


# --- Synthetic Data Tests ---


class TestGenerateSyntheticData(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_noise_free_data(self) -> None:
        """Noise-free data matches network subgraph_values."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=1, nqoi=1, nnodes=2, bkd=bkd)
        randomize_coefficients(net, bkd, seed=10)

        samples, values = generate_synthetic_data(
            net,
            bkd,
            nsamples_per_node=[20, 15],
            seed=42,
        )

        # Check shapes
        self.assertEqual(samples[0].shape, (1, 20))
        self.assertEqual(values[0].shape, (1, 20))
        self.assertEqual(samples[1].shape, (1, 15))
        self.assertEqual(values[1].shape, (1, 15))

        # Verify values match subgraph eval
        for node_id in net.topo_order():
            expected = net.subgraph_values(samples[node_id], node_id)
            bkd.assert_allclose(values[node_id], expected)

    def test_noisy_data_shape(self) -> None:
        """Noisy data has correct shapes and differs from clean."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=1, nqoi=1, nnodes=2, bkd=bkd)
        randomize_coefficients(net, bkd, seed=10)

        samples_clean, values_clean = generate_synthetic_data(
            net,
            bkd,
            nsamples_per_node=[20, 15],
            noise_std=0.0,
            seed=42,
        )
        samples_noisy, values_noisy = generate_synthetic_data(
            net,
            bkd,
            nsamples_per_node=[20, 15],
            noise_std=0.1,
            seed=42,
        )

        # Same samples (same seed, noise added after sampling)
        bkd.assert_allclose(samples_clean[0], samples_noisy[0])
        # Values should differ due to noise
        diff = bkd.to_numpy(values_noisy[0] - values_clean[0])
        self.assertGreater(float(np.max(np.abs(diff))), 1e-10)

    def test_reproducibility(self) -> None:
        """Same seed produces identical data."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=1, nqoi=1, nnodes=2, bkd=bkd)
        randomize_coefficients(net, bkd, seed=10)

        s1, v1 = generate_synthetic_data(
            net,
            bkd,
            nsamples_per_node=[20, 15],
            seed=99,
        )
        s2, v2 = generate_synthetic_data(
            net,
            bkd,
            nsamples_per_node=[20, 15],
            seed=99,
        )
        for i in range(2):
            bkd.assert_allclose(s1[i], s2[i])
            bkd.assert_allclose(v1[i], v2[i])


# --- Randomize Coefficients Tests ---


class TestRandomizeCoefficients(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_sets_nonzero_coefficients(self) -> None:
        """After randomize, model predictions change from zero init."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=1, nqoi=1, nnodes=2, bkd=bkd)

        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        out_before = bkd.to_numpy(net(samples))

        randomize_coefficients(net, bkd, seed=10)
        out_after = bkd.to_numpy(net(samples))

        # Output should change
        self.assertGreater(float(np.max(np.abs(out_after - out_before))), 1e-10)

    def test_reproducibility(self) -> None:
        """Same seed produces identical coefficients."""
        bkd = self._bkd
        net1 = build_chain_mfnet(nvars=1, nqoi=1, nnodes=2, bkd=bkd)
        net2 = build_chain_mfnet(nvars=1, nqoi=1, nnodes=2, bkd=bkd)

        randomize_coefficients(net1, bkd, seed=10)
        randomize_coefficients(net2, bkd, seed=10)

        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        bkd.assert_allclose(net1(samples), net2(samples))

    def test_three_node_chain(self) -> None:
        """Randomize works on 3-node chain with discrepancy models."""
        bkd = self._bkd
        net = build_chain_mfnet(nvars=1, nqoi=1, nnodes=3, bkd=bkd)
        randomize_coefficients(net, bkd, seed=10)

        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        out = net(samples)
        # Should produce non-trivial output
        self.assertGreater(float(np.max(np.abs(bkd.to_numpy(out)))), 1e-10)


# --- Concrete backend test classes ---


class TestBuildChainNumpy(TestBuildChainMFNet[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBuildChainTorch(TestBuildChainMFNet[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestBuildDAGNumpy(TestBuildDAGMFNet[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBuildDAGTorch(TestBuildDAGMFNet[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestSyntheticDataNumpy(TestGenerateSyntheticData[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSyntheticDataTorch(TestGenerateSyntheticData[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestRandomizeNumpy(TestRandomizeCoefficients[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRandomizeTorch(TestRandomizeCoefficients[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
