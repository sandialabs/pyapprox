"""Tests for MFNet nodes and edges."""

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
from pyapprox.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    MFNetNode,
    RootMFNetNode,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


def _create_monomial_expansion(
    bkd: Backend[Array], nvars: int = 1, nqoi: int = 1, max_level: int = 2
) -> BasisExpansion[Array]:
    """Create a simple BasisExpansion with monomial basis for testing."""
    bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = MultiIndexBasis.__new__(MultiIndexBasis)
    MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
    expansion = BasisExpansion(basis, bkd, nqoi=nqoi)
    np.random.seed(42)
    coef = bkd.asarray(np.random.randn(expansion.nterms(), nqoi))
    expansion.set_coefficients(coef)
    return expansion


class TestMFNetNodes(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def test_node_construction(self) -> None:
        model = _create_monomial_expansion(self._bkd)
        node = MFNetNode(node_id=0, model=model, noise_std=0.1, bkd=self._bkd)
        self.assertEqual(node.node_id(), 0)
        self.assertIsNotNone(node.model())
        # noise_std is stored in log space, check exp recovers it
        self._bkd.assert_allclose(
            node.noise_std(),
            self._bkd.asarray([0.1]),
            rtol=1e-12,
        )

    def test_node_negative_id_raises(self) -> None:
        model = _create_monomial_expansion(self._bkd)
        with self.assertRaises(ValueError):
            MFNetNode(node_id=-1, model=model, noise_std=0.1, bkd=self._bkd)

    def test_node_hyp_list_aggregation(self) -> None:
        model = _create_monomial_expansion(self._bkd)
        node = MFNetNode(
            node_id=0,
            model=model,
            noise_std=0.5,
            bkd=self._bkd,
            fixed_noise_std=False,
        )
        hyps = node.hyp_list()
        # Model params + 1 noise_std param
        model_nparams = model.hyp_list().nparams()
        self.assertEqual(hyps.nparams(), model_nparams + 1)
        # noise_std is active, so nactive_params should include it
        self.assertEqual(hyps.nactive_params(), model.hyp_list().nactive_params() + 1)

    def test_node_hyp_list_fixed_noise(self) -> None:
        model = _create_monomial_expansion(self._bkd)
        node = MFNetNode(
            node_id=0,
            model=model,
            noise_std=0.5,
            bkd=self._bkd,
            fixed_noise_std=True,
        )
        hyps = node.hyp_list()
        # noise_std is fixed, so nactive_params should NOT include it
        self.assertEqual(hyps.nactive_params(), model.hyp_list().nactive_params())

    def test_leaf_node_validation(self) -> None:
        model = _create_monomial_expansion(self._bkd)
        node = LeafMFNetNode(node_id=0, model=model, noise_std=0.1, bkd=self._bkd)
        # Leaf with no children or parents: should pass validation
        node.validate(nvars_global=1)
        self.assertTrue(node.is_leaf())
        self.assertIsNotNone(node.active_sample_vars())

    def test_leaf_node_with_children_raises(self) -> None:
        model = _create_monomial_expansion(self._bkd)
        node = LeafMFNetNode(node_id=0, model=model, noise_std=0.1, bkd=self._bkd)
        node.set_children_ids([1])
        with self.assertRaises(ValueError):
            node.validate(nvars_global=1)

    def test_root_node_validation(self) -> None:
        model = _create_monomial_expansion(self._bkd, nvars=2)
        node = RootMFNetNode(node_id=1, model=model, noise_std=0.1, bkd=self._bkd)
        # Root with no parents: should pass validation
        node.set_children_ids([0])
        node.validate(nvars_global=1)
        self.assertTrue(node.is_root())

    def test_root_node_with_parents_raises(self) -> None:
        model = _create_monomial_expansion(self._bkd)
        node = RootMFNetNode(node_id=1, model=model, noise_std=0.1, bkd=self._bkd)
        node.set_parent_ids([2])
        with self.assertRaises(ValueError):
            node.validate(nvars_global=1)

    def test_interior_node_needs_both(self) -> None:
        model = _create_monomial_expansion(self._bkd, nvars=2)
        node = MFNetNode(node_id=1, model=model, noise_std=0.1, bkd=self._bkd)
        # Interior with only children should fail
        node.set_children_ids([0])
        with self.assertRaises(ValueError):
            node.validate(nvars_global=1)

    def test_active_sample_vars_default(self) -> None:
        model = _create_monomial_expansion(self._bkd)
        node = LeafMFNetNode(node_id=0, model=model, noise_std=0.1, bkd=self._bkd)
        node.validate(nvars_global=3)
        expected = self._bkd.asarray([0, 1, 2], dtype=int)
        self._bkd.assert_allclose(
            self._bkd.asarray(node.active_sample_vars(), dtype=float),
            self._bkd.asarray(expected, dtype=float),
        )


class TestMFNetEdges(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_edge_construction(self) -> None:
        edge = MFNetEdge(child_node_id=0, parent_node_id=1, bkd=self._bkd)
        self.assertEqual(edge.child_node_id(), 0)
        self.assertEqual(edge.parent_node_id(), 1)
        self.assertIsNone(edge.output_ids())

    def test_edge_validate_defaults_to_all(self) -> None:
        edge = MFNetEdge(child_node_id=0, parent_node_id=1, bkd=self._bkd)
        edge.validate(child_nqoi=3)
        expected = self._bkd.asarray([0, 1, 2], dtype=int)
        self._bkd.assert_allclose(
            self._bkd.asarray(edge.output_ids(), dtype=float),
            self._bkd.asarray(expected, dtype=float),
        )

    def test_edge_validate_custom_ids(self) -> None:
        ids = self._bkd.asarray([0, 2], dtype=int)
        edge = MFNetEdge(
            child_node_id=0,
            parent_node_id=1,
            bkd=self._bkd,
            child_output_ids=ids,
        )
        edge.validate(child_nqoi=3)
        self._bkd.assert_allclose(
            self._bkd.asarray(edge.output_ids(), dtype=float),
            self._bkd.asarray(ids, dtype=float),
        )

    def test_edge_validate_out_of_range_raises(self) -> None:
        ids = self._bkd.asarray([0, 3], dtype=int)
        edge = MFNetEdge(
            child_node_id=0,
            parent_node_id=1,
            bkd=self._bkd,
            child_output_ids=ids,
        )
        with self.assertRaises(ValueError):
            edge.validate(child_nqoi=3)


# --- Concrete backend test classes ---


class TestMFNetNodesNumpy(TestMFNetNodes[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMFNetNodesTorch(TestMFNetNodes[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMFNetEdgesNumpy(TestMFNetEdges[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMFNetEdgesTorch(TestMFNetEdges[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
