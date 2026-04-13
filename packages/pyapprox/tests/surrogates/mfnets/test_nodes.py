"""Tests for MFNet nodes and edges."""

import numpy as np
import pytest

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


def _create_monomial_expansion(
    bkd, nvars: int = 1, nqoi: int = 1, max_level: int = 2
) -> BasisExpansion:
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


class TestMFNetNodes:

    def test_node_construction(self, bkd) -> None:
        model = _create_monomial_expansion(bkd)
        node = MFNetNode(node_id=0, model=model, noise_std=0.1, bkd=bkd)
        assert node.node_id() == 0
        assert node.model() is not None
        # noise_std is stored in log space, check exp recovers it
        bkd.assert_allclose(
            node.noise_std(),
            bkd.asarray([0.1]),
            rtol=1e-12,
        )

    def test_node_negative_id_raises(self, bkd) -> None:
        model = _create_monomial_expansion(bkd)
        with pytest.raises(ValueError):
            MFNetNode(node_id=-1, model=model, noise_std=0.1, bkd=bkd)

    def test_node_hyp_list_aggregation(self, bkd) -> None:
        model = _create_monomial_expansion(bkd)
        node = MFNetNode(
            node_id=0,
            model=model,
            noise_std=0.5,
            bkd=bkd,
            fixed_noise_std=False,
        )
        hyps = node.hyp_list()
        # Model params + 1 noise_std param
        model_nparams = model.hyp_list().nparams()
        assert hyps.nparams() == model_nparams + 1
        # noise_std is active, so nactive_params should include it
        assert hyps.nactive_params() == model.hyp_list().nactive_params() + 1

    def test_node_hyp_list_fixed_noise(self, bkd) -> None:
        model = _create_monomial_expansion(bkd)
        node = MFNetNode(
            node_id=0,
            model=model,
            noise_std=0.5,
            bkd=bkd,
            fixed_noise_std=True,
        )
        hyps = node.hyp_list()
        # noise_std is fixed, so nactive_params should NOT include it
        assert hyps.nactive_params() == model.hyp_list().nactive_params()

    def test_leaf_node_validation(self, bkd) -> None:
        model = _create_monomial_expansion(bkd)
        node = LeafMFNetNode(node_id=0, model=model, noise_std=0.1, bkd=bkd)
        # Leaf with no children or parents: should pass validation
        node.validate(nvars_global=1)
        assert node.is_leaf()
        assert node.active_sample_vars() is not None

    def test_leaf_node_with_children_raises(self, bkd) -> None:
        model = _create_monomial_expansion(bkd)
        node = LeafMFNetNode(node_id=0, model=model, noise_std=0.1, bkd=bkd)
        node.set_children_ids([1])
        with pytest.raises(ValueError):
            node.validate(nvars_global=1)

    def test_root_node_validation(self, bkd) -> None:
        model = _create_monomial_expansion(bkd, nvars=2)
        node = RootMFNetNode(node_id=1, model=model, noise_std=0.1, bkd=bkd)
        # Root with no parents: should pass validation
        node.set_children_ids([0])
        node.validate(nvars_global=1)
        assert node.is_root()

    def test_root_node_with_parents_raises(self, bkd) -> None:
        model = _create_monomial_expansion(bkd)
        node = RootMFNetNode(node_id=1, model=model, noise_std=0.1, bkd=bkd)
        node.set_parent_ids([2])
        with pytest.raises(ValueError):
            node.validate(nvars_global=1)

    def test_interior_node_needs_both(self, bkd) -> None:
        model = _create_monomial_expansion(bkd, nvars=2)
        node = MFNetNode(node_id=1, model=model, noise_std=0.1, bkd=bkd)
        # Interior with only children should fail
        node.set_children_ids([0])
        with pytest.raises(ValueError):
            node.validate(nvars_global=1)

    def test_active_sample_vars_default(self, bkd) -> None:
        model = _create_monomial_expansion(bkd)
        node = LeafMFNetNode(node_id=0, model=model, noise_std=0.1, bkd=bkd)
        node.validate(nvars_global=3)
        expected = bkd.asarray([0, 1, 2], dtype=int)
        bkd.assert_allclose(
            bkd.asarray(node.active_sample_vars(), dtype=float),
            bkd.asarray(expected, dtype=float),
        )


class TestMFNetEdges:

    def test_edge_construction(self, bkd) -> None:
        edge = MFNetEdge(child_node_id=0, parent_node_id=1, bkd=bkd)
        assert edge.child_node_id() == 0
        assert edge.parent_node_id() == 1
        assert edge.output_ids() is None

    def test_edge_validate_defaults_to_all(self, bkd) -> None:
        edge = MFNetEdge(child_node_id=0, parent_node_id=1, bkd=bkd)
        edge.validate(child_nqoi=3)
        expected = bkd.asarray([0, 1, 2], dtype=int)
        bkd.assert_allclose(
            bkd.asarray(edge.output_ids(), dtype=float),
            bkd.asarray(expected, dtype=float),
        )

    def test_edge_validate_custom_ids(self, bkd) -> None:
        ids = bkd.asarray([0, 2], dtype=int)
        edge = MFNetEdge(
            child_node_id=0,
            parent_node_id=1,
            bkd=bkd,
            child_output_ids=ids,
        )
        edge.validate(child_nqoi=3)
        bkd.assert_allclose(
            bkd.asarray(edge.output_ids(), dtype=float),
            bkd.asarray(ids, dtype=float),
        )

    def test_edge_validate_out_of_range_raises(self, bkd) -> None:
        ids = bkd.asarray([0, 3], dtype=int)
        edge = MFNetEdge(
            child_node_id=0,
            parent_node_id=1,
            bkd=bkd,
            child_output_ids=ids,
        )
        with pytest.raises(ValueError):
            edge.validate(child_nqoi=3)
