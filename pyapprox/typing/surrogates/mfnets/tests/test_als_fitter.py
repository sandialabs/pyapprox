"""Tests for MFNet ALS fitter."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.affine.univariate import MonomialBasis1D
from pyapprox.typing.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.typing.surrogates.affine.basis import MultiIndexBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion

from pyapprox.typing.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    MFNetNode,
    RootMFNetNode,
)
from pyapprox.typing.surrogates.mfnets.edges import MFNetEdge
from pyapprox.typing.surrogates.mfnets.network import MFNet
from pyapprox.typing.surrogates.mfnets.discrepancy import (
    MultiplicativeAdditiveDiscrepancy,
)
from pyapprox.typing.surrogates.mfnets.fitters.als_fitter import (
    MFNetALSFitter,
)


def _create_expansion(
    bkd: Backend[Array], nvars: int, nqoi: int, max_level: int = 2
) -> BasisExpansion[Array]:
    bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = MultiIndexBasis.__new__(MultiIndexBasis)
    MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=nqoi)


def _build_mfnet_with_discrepancy(
    bkd: Backend[Array],
    leaf_coef: Array,
    scale_coef: Array,
    delta_coef: Array,
    leaf_level: int = 3,
    scale_level: int = 1,
    delta_level: int = 3,
) -> MFNet[Array]:
    """Build a 2-node MFNet with discrepancy root."""
    leaf_model = _create_expansion(bkd, nvars=1, nqoi=1, max_level=leaf_level)
    leaf_model.set_coefficients(leaf_coef)

    scaling = _create_expansion(bkd, nvars=1, nqoi=1, max_level=scale_level)
    scaling.set_coefficients(scale_coef)
    delta = _create_expansion(bkd, nvars=1, nqoi=1, max_level=delta_level)
    delta.set_coefficients(delta_coef)
    root_model = MultiplicativeAdditiveDiscrepancy(
        [scaling], delta, nscaled_qoi=1, bkd=bkd
    )

    net = MFNet(nvars=1, bkd=bkd)
    net.add_node(LeafMFNetNode(0, leaf_model, noise_std=1e-2, bkd=bkd))
    net.add_node(RootMFNetNode(1, root_model, noise_std=1e-2, bkd=bkd))
    net.add_edge(MFNetEdge(0, 1, bkd=bkd))
    net.validate()
    return net


class TestMFNetALSFitter(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def _make_truth_and_data(self, n_leaf: int = 30, n_root: int = 25) -> Any:
        bkd = self._bkd
        np.random.seed(10)

        leaf_nterms = _create_expansion(bkd, 1, 1, 3).nterms()
        scale_nterms = _create_expansion(bkd, 1, 1, 1).nterms()
        delta_nterms = _create_expansion(bkd, 1, 1, 3).nterms()

        true_leaf = bkd.asarray(np.random.randn(leaf_nterms, 1))
        true_scale = bkd.asarray(np.random.randn(scale_nterms, 1))
        true_delta = bkd.asarray(np.random.randn(delta_nterms, 1))

        true_net = _build_mfnet_with_discrepancy(
            bkd, true_leaf, true_scale, true_delta
        )

        np.random.seed(42)
        s_leaf = bkd.asarray(np.random.uniform(-1, 1, (1, n_leaf)))
        s_root = bkd.asarray(np.random.uniform(-1, 1, (1, n_root)))
        v_leaf = true_net.subgraph_values(s_leaf, 0)
        v_root = true_net.subgraph_values(s_root, 1)

        return true_net, [s_leaf, s_root], [v_leaf, v_root]

    def test_als_convergence(self) -> None:
        """Test ALS converges to zero residual with noise-free data."""
        bkd = self._bkd
        true_net, samples, values = self._make_truth_and_data()

        # Build fresh network with zero initial coefficients
        leaf_nterms = true_net.node(0).model().hyp_list().nparams()
        scale_nterms = true_net.node(1).model().scaling_models()[0].hyp_list().nparams()
        delta_nterms = true_net.node(1).model().delta_model().hyp_list().nparams()

        fit_net = _build_mfnet_with_discrepancy(
            bkd,
            bkd.zeros((leaf_nterms, 1)),
            bkd.zeros((scale_nterms, 1)),
            bkd.zeros((delta_nterms, 1)),
        )

        fitter = MFNetALSFitter(bkd, max_sweeps=20, tol=1e-12)
        result = fitter.fit(fit_net, samples, values)

        # Check predictions match
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        true_out = true_net(test_samples)
        fit_out = result.surrogate()(test_samples)
        bkd.assert_allclose(fit_out, true_out, atol=1e-6)

        # Loss should have decreased
        self.assertLess(result.loss_history()[-1], result.loss_history()[0] + 1e-10)

    def test_als_three_node_chain(self) -> None:
        """Test ALS on a 3-node chain: leaf -> mid -> root."""
        bkd = self._bkd

        # True models
        np.random.seed(10)
        leaf_model = _create_expansion(bkd, nvars=1, nqoi=1, max_level=2)
        leaf_model.set_coefficients(bkd.asarray(np.random.randn(leaf_model.nterms(), 1)))

        mid_scale = _create_expansion(bkd, nvars=1, nqoi=1, max_level=0)
        mid_scale.set_coefficients(bkd.asarray(np.random.randn(mid_scale.nterms(), 1)))
        mid_delta = _create_expansion(bkd, nvars=1, nqoi=1, max_level=2)
        mid_delta.set_coefficients(bkd.asarray(np.random.randn(mid_delta.nterms(), 1)))
        mid_model = MultiplicativeAdditiveDiscrepancy(
            [mid_scale], mid_delta, nscaled_qoi=1, bkd=bkd
        )

        root_scale = _create_expansion(bkd, nvars=1, nqoi=1, max_level=0)
        root_scale.set_coefficients(bkd.asarray(np.random.randn(root_scale.nterms(), 1)))
        root_delta = _create_expansion(bkd, nvars=1, nqoi=1, max_level=2)
        root_delta.set_coefficients(bkd.asarray(np.random.randn(root_delta.nterms(), 1)))
        root_model = MultiplicativeAdditiveDiscrepancy(
            [root_scale], root_delta, nscaled_qoi=1, bkd=bkd
        )

        true_net = MFNet(nvars=1, bkd=bkd)
        true_net.add_node(LeafMFNetNode(0, leaf_model, noise_std=1e-2, bkd=bkd))
        true_net.add_node(MFNetNode(1, mid_model, noise_std=1e-2, bkd=bkd))
        true_net.add_node(RootMFNetNode(2, root_model, noise_std=1e-2, bkd=bkd))
        true_net.add_edge(MFNetEdge(0, 1, bkd=bkd))
        true_net.add_edge(MFNetEdge(1, 2, bkd=bkd))
        true_net.validate()

        # Generate data
        np.random.seed(42)
        s0 = bkd.asarray(np.random.uniform(-1, 1, (1, 25)))
        s1 = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        s2 = bkd.asarray(np.random.uniform(-1, 1, (1, 15)))
        v0 = true_net.subgraph_values(s0, 0)
        v1 = true_net.subgraph_values(s1, 1)
        v2 = true_net.subgraph_values(s2, 2)

        # Build fresh network with zero initial params
        leaf_f = _create_expansion(bkd, nvars=1, nqoi=1, max_level=2)
        mid_sf = _create_expansion(bkd, nvars=1, nqoi=1, max_level=0)
        mid_df = _create_expansion(bkd, nvars=1, nqoi=1, max_level=2)
        mid_mf = MultiplicativeAdditiveDiscrepancy(
            [mid_sf], mid_df, nscaled_qoi=1, bkd=bkd
        )
        root_sf = _create_expansion(bkd, nvars=1, nqoi=1, max_level=0)
        root_df = _create_expansion(bkd, nvars=1, nqoi=1, max_level=2)
        root_mf = MultiplicativeAdditiveDiscrepancy(
            [root_sf], root_df, nscaled_qoi=1, bkd=bkd
        )

        fit_net = MFNet(nvars=1, bkd=bkd)
        fit_net.add_node(LeafMFNetNode(0, leaf_f, noise_std=1e-2, bkd=bkd))
        fit_net.add_node(MFNetNode(1, mid_mf, noise_std=1e-2, bkd=bkd))
        fit_net.add_node(RootMFNetNode(2, root_mf, noise_std=1e-2, bkd=bkd))
        fit_net.add_edge(MFNetEdge(0, 1, bkd=bkd))
        fit_net.add_edge(MFNetEdge(1, 2, bkd=bkd))
        fit_net.validate()

        fitter = MFNetALSFitter(bkd, max_sweeps=30, tol=1e-14)
        result = fitter.fit(fit_net, [s0, s1, s2], [v0, v1, v2])

        # Verify predictions
        np.random.seed(123)
        test_s = bkd.asarray(np.random.uniform(-1, 1, (1, 40)))
        true_out = true_net(test_s)
        fit_out = result.surrogate()(test_s)
        bkd.assert_allclose(fit_out, true_out, atol=1e-5)


# --- Concrete backend test classes ---

class TestALSFitterNumpy(TestMFNetALSFitter[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestALSFitterTorch(TestMFNetALSFitter[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
