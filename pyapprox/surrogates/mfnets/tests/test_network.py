"""Tests for MFNet network class."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.surrogates.affine.basis import MultiIndexBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import MonomialBasis1D
from pyapprox.surrogates.mfnets.edges import MFNetEdge
from pyapprox.surrogates.mfnets.network import MFNet
from pyapprox.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    RootMFNetNode,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


def _create_monomial_expansion(
    bkd: Backend[Array], nvars: int = 1, nqoi: int = 1, max_level: int = 2
) -> BasisExpansion[Array]:
    """Create a BasisExpansion with monomial basis."""
    bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = MultiIndexBasis.__new__(MultiIndexBasis)
    MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
    expansion = BasisExpansion(basis, bkd, nqoi=nqoi)
    return expansion


class TestMFNet(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def _build_two_node_network(self) -> Any:
        """Build a simple 2-node network: leaf(0) -> root(1).

        Leaf model: f0(x) = polynomial(x), nvars=1, nqoi=1
        Root model: f1(x, q0) = polynomial(x, q0), nvars=2, nqoi=1
          where q0 is the leaf output.
        """
        bkd = self._bkd
        # Leaf: 1D polynomial -> 1 output
        leaf_model = _create_monomial_expansion(bkd, nvars=1, nqoi=1)
        np.random.seed(10)
        leaf_model.set_coefficients(
            bkd.asarray(np.random.randn(leaf_model.nterms(), 1))
        )

        # Root: 2D polynomial (x, q0) -> 1 output
        root_model = _create_monomial_expansion(bkd, nvars=2, nqoi=1)
        np.random.seed(20)
        root_model.set_coefficients(
            bkd.asarray(np.random.randn(root_model.nterms(), 1))
        )

        net = MFNet(nvars=1, bkd=bkd)
        net.add_node(LeafMFNetNode(0, leaf_model, noise_std=0.1, bkd=bkd))
        net.add_node(RootMFNetNode(1, root_model, noise_std=0.1, bkd=bkd))
        net.add_edge(MFNetEdge(child_node_id=0, parent_node_id=1, bkd=bkd))
        net.validate()
        return net, leaf_model, root_model

    def test_two_node_forward_eval(self) -> None:
        """Test that 2-node forward eval matches manual computation."""
        bkd = self._bkd
        net, leaf_model, root_model = self._build_two_node_network()

        np.random.seed(99)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))

        # Manual: leaf output
        leaf_out = leaf_model(samples)  # (1, 10)
        # Root input: (x, q0)
        augmented = bkd.vstack([samples, leaf_out])  # (2, 10)
        root_out = root_model(augmented)  # (1, 10)

        # Network output
        net_out = net(samples)

        bkd.assert_allclose(net_out, root_out, rtol=1e-12)

    def test_subgraph_values_leaf(self) -> None:
        """Test subgraph_values for leaf node returns model output."""
        bkd = self._bkd
        net, leaf_model, _ = self._build_two_node_network()

        np.random.seed(99)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))

        leaf_vals = net.subgraph_values(samples, node_id=0)
        expected = leaf_model(samples)
        bkd.assert_allclose(leaf_vals, expected, rtol=1e-12)

    def test_nqoi(self) -> None:
        net, _, _ = self._build_two_node_network()
        self.assertEqual(net.nqoi(), 1)

    def test_topo_order(self) -> None:
        net, _, _ = self._build_two_node_network()
        topo = net.topo_order()
        # Leaf (0) must come before root (1)
        self.assertEqual(topo.index(0) < topo.index(1), True)

    def test_nodes_classification(self) -> None:
        net, _, _ = self._build_two_node_network()
        leaves = net.leaf_nodes()
        roots = net.root_nodes()
        self.assertEqual(len(leaves), 1)
        self.assertEqual(len(roots), 1)
        self.assertEqual(leaves[0].node_id(), 0)
        self.assertEqual(roots[0].node_id(), 1)

    def test_hyp_list_aggregation(self) -> None:
        net, _, _ = self._build_two_node_network()
        hyps = net.hyp_list()
        # Each node: model params + noise_std param
        node0 = net.node(0)
        node1 = net.node(1)
        expected_nparams = node0.hyp_list().nparams() + node1.hyp_list().nparams()
        self.assertEqual(hyps.nparams(), expected_nparams)

    def test_not_validated_raises(self) -> None:
        net = MFNet(nvars=1, bkd=self._bkd)
        with self.assertRaises(RuntimeError):
            net.nqoi()
        with self.assertRaises(RuntimeError):
            np.random.seed(1)
            s = self._bkd.asarray(np.random.randn(1, 5))
            net(s)

    def test_three_node_chain(self) -> None:
        """Build 3-node chain: leaf(0) -> mid(1) -> root(2)."""
        bkd = self._bkd

        # Leaf: f0(x), nvars=1, nqoi=1
        leaf_model = _create_monomial_expansion(bkd, nvars=1, nqoi=1)
        np.random.seed(10)
        leaf_model.set_coefficients(
            bkd.asarray(np.random.randn(leaf_model.nterms(), 1))
        )

        # Mid: f1(x, q0), nvars=2, nqoi=1 (interior node)
        from pyapprox.surrogates.mfnets.nodes import MFNetNode

        mid_model = _create_monomial_expansion(bkd, nvars=2, nqoi=1)
        np.random.seed(20)
        mid_model.set_coefficients(bkd.asarray(np.random.randn(mid_model.nterms(), 1)))

        # Root: f2(x, q1), nvars=2, nqoi=1
        root_model = _create_monomial_expansion(bkd, nvars=2, nqoi=1)
        np.random.seed(30)
        root_model.set_coefficients(
            bkd.asarray(np.random.randn(root_model.nterms(), 1))
        )

        net = MFNet(nvars=1, bkd=bkd)
        net.add_node(LeafMFNetNode(0, leaf_model, noise_std=0.1, bkd=bkd))
        net.add_node(MFNetNode(1, mid_model, noise_std=0.1, bkd=bkd))
        net.add_node(RootMFNetNode(2, root_model, noise_std=0.1, bkd=bkd))
        net.add_edge(MFNetEdge(0, 1, bkd=bkd))
        net.add_edge(MFNetEdge(1, 2, bkd=bkd))
        net.validate()

        # Manual evaluation
        np.random.seed(99)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 8)))

        q0 = leaf_model(samples)  # (1, 8)
        aug1 = bkd.vstack([samples, q0])
        q1 = mid_model(aug1)  # (1, 8)
        aug2 = bkd.vstack([samples, q1])
        q2 = root_model(aug2)  # (1, 8)

        net_out = net(samples)
        bkd.assert_allclose(net_out, q2, rtol=1e-12)

    def test_set_training_data(self) -> None:
        bkd = self._bkd
        net, _, _ = self._build_two_node_network()

        s0 = bkd.asarray(np.random.randn(1, 5))
        v0 = bkd.asarray(np.random.randn(1, 5))
        s1 = bkd.asarray(np.random.randn(1, 3))
        v1 = bkd.asarray(np.random.randn(1, 3))

        net.set_training_data([s0, s1], [v0, v1])
        self.assertIsNotNone(net.train_samples())
        self.assertEqual(len(net.train_samples()), 2)

    def test_memoization_avoids_recomputation(self) -> None:
        """Verify cache is shared across root evaluations."""
        bkd = self._bkd
        net, leaf_model, _ = self._build_two_node_network()

        np.random.seed(99)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))

        cache = {}
        net.subgraph_values(samples, node_id=1, cache=cache)
        # Cache should contain both node 0 and node 1
        self.assertIn(0, cache)
        self.assertIn(1, cache)


# --- Concrete backend test classes ---


class TestMFNetNumpy(TestMFNet[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMFNetTorch(TestMFNet[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
