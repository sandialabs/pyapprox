"""End-to-end recovery tests for MFNets.

Strategy: create a true MFNet with random coefficients, generate training
data from it, then fit a fresh MFNet and verify predictions match.
"""

from typing import List

import numpy as np
import pytest

from pyapprox.surrogates.mfnets.edges import MFNetEdge
from pyapprox.surrogates.mfnets.fitters.als_fitter import (
    MFNetALSFitter,
)
from pyapprox.surrogates.mfnets.fitters.gradient_fitter import (
    MFNetGradientFitter,
)
from pyapprox.surrogates.mfnets.network import MFNet
from pyapprox.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    MFNetNode,
    RootMFNetNode,
)
from pyapprox.surrogates.mfnets.registry import create_node_model
from pyapprox.util.test_utils import (
    slower_test,
)


def _build_two_node_mfnet(
    bkd,
    leaf_level: int = 3,
    scale_level: int = 1,
    delta_level: int = 3,
    seed: int = 0,
) -> MFNet:
    """Build a 2-node MFNet with random coefficients."""
    np.random.seed(seed)
    leaf = create_node_model(
        "basis_expansion", bkd, nvars=1, nqoi=1, max_level=leaf_level
    )
    nterms_leaf = leaf.nterms()
    leaf.set_coefficients(bkd.asarray(np.random.randn(nterms_leaf, 1)))

    root = create_node_model(
        "multiplicative_additive",
        bkd,
        nvars_x=1,
        nqoi=1,
        nscaled_qoi=1,
        scale_level=scale_level,
        delta_level=delta_level,
    )
    # Set random coefficients for all sub-models
    for sm in root.scaling_models():
        sm.set_coefficients(bkd.asarray(np.random.randn(sm.nterms(), sm.nqoi())))
    root.delta_model().set_coefficients(
        bkd.asarray(np.random.randn(root.delta_model().nterms(), root.nqoi()))
    )

    net = MFNet(nvars=1, bkd=bkd)
    net.add_node(LeafMFNetNode(0, leaf, noise_std=1e-2, bkd=bkd))
    net.add_node(RootMFNetNode(1, root, noise_std=1e-2, bkd=bkd))
    net.add_edge(MFNetEdge(0, 1, bkd=bkd))
    net.validate()
    return net


def _build_blank_two_node_mfnet(
    bkd,
    leaf_level: int = 3,
    scale_level: int = 1,
    delta_level: int = 3,
) -> MFNet:
    """Build a 2-node MFNet with zero coefficients."""
    leaf = create_node_model(
        "basis_expansion", bkd, nvars=1, nqoi=1, max_level=leaf_level
    )
    root = create_node_model(
        "multiplicative_additive",
        bkd,
        nvars_x=1,
        nqoi=1,
        nscaled_qoi=1,
        scale_level=scale_level,
        delta_level=delta_level,
    )
    net = MFNet(nvars=1, bkd=bkd)
    net.add_node(LeafMFNetNode(0, leaf, noise_std=1e-2, bkd=bkd))
    net.add_node(RootMFNetNode(1, root, noise_std=1e-2, bkd=bkd))
    net.add_edge(MFNetEdge(0, 1, bkd=bkd))
    net.validate()
    return net


def _generate_data(
    net: MFNet, bkd, n_per_node: List[int], seed: int = 42
):
    """Generate noise-free training data from a true network."""
    np.random.seed(seed)
    samples_list = []
    values_list = []
    for i, n in enumerate(n_per_node):
        s = bkd.asarray(np.random.uniform(-1, 1, (1, n)))
        v = net.subgraph_values(s, i)
        samples_list.append(s)
        values_list.append(v)
    return samples_list, values_list


class TestRecovery:

    def test_als_recovers_two_node(self, bkd) -> None:
        """ALS should recover predictions for a 2-node network."""
        true_net = _build_two_node_mfnet(bkd, seed=10)
        samples, values = _generate_data(true_net, bkd, [30, 25])

        fit_net = _build_blank_two_node_mfnet(bkd)
        fitter = MFNetALSFitter(bkd, max_sweeps=20, tol=1e-14)
        result = fitter.fit(fit_net, samples, values)

        np.random.seed(123)
        test_s = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        true_out = true_net(test_s)
        fit_out = result.surrogate()(test_s)
        bkd.assert_allclose(fit_out, true_out, atol=1e-6)

    @slower_test
    def test_gradient_recovers_two_node(self, bkd) -> None:
        """Gradient fitter should recover predictions for a 2-node network."""
        true_net = _build_two_node_mfnet(bkd, seed=20)
        samples, values = _generate_data(true_net, bkd, [30, 25])

        fit_net = _build_blank_two_node_mfnet(bkd)
        # Use ALS to get a good initial guess
        als = MFNetALSFitter(bkd, max_sweeps=5, tol=1e-10)
        als.fit(fit_net, samples, values)
        # Then refine with gradient
        fitter = MFNetGradientFitter(bkd)
        result = fitter.fit(fit_net, samples, values)

        np.random.seed(123)
        test_s = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        true_out = true_net(test_s)
        fit_out = result.surrogate()(test_s)
        bkd.assert_allclose(fit_out, true_out, atol=1e-4)

    @pytest.mark.slow_on("TorchBkd")
    def test_als_recovers_three_node_chain(self, bkd) -> None:
        """ALS on 3-node chain: leaf -> mid -> root."""
        np.random.seed(30)

        # True network
        leaf = create_node_model("basis_expansion", bkd, nvars=1, nqoi=1, max_level=2)
        leaf.set_coefficients(bkd.asarray(np.random.randn(leaf.nterms(), 1)))
        mid = create_node_model(
            "multiplicative_additive",
            bkd,
            nvars_x=1,
            nqoi=1,
            nscaled_qoi=1,
            scale_level=0,
            delta_level=2,
        )
        for sm in mid.scaling_models():
            sm.set_coefficients(bkd.asarray(np.random.randn(sm.nterms(), sm.nqoi())))
        mid.delta_model().set_coefficients(
            bkd.asarray(np.random.randn(mid.delta_model().nterms(), mid.nqoi()))
        )
        root = create_node_model(
            "multiplicative_additive",
            bkd,
            nvars_x=1,
            nqoi=1,
            nscaled_qoi=1,
            scale_level=0,
            delta_level=2,
        )
        for sm in root.scaling_models():
            sm.set_coefficients(bkd.asarray(np.random.randn(sm.nterms(), sm.nqoi())))
        root.delta_model().set_coefficients(
            bkd.asarray(np.random.randn(root.delta_model().nterms(), root.nqoi()))
        )

        true_net = MFNet(nvars=1, bkd=bkd)
        true_net.add_node(LeafMFNetNode(0, leaf, noise_std=1e-2, bkd=bkd))
        true_net.add_node(MFNetNode(1, mid, noise_std=1e-2, bkd=bkd))
        true_net.add_node(RootMFNetNode(2, root, noise_std=1e-2, bkd=bkd))
        true_net.add_edge(MFNetEdge(0, 1, bkd=bkd))
        true_net.add_edge(MFNetEdge(1, 2, bkd=bkd))
        true_net.validate()

        samples, values = _generate_data(true_net, bkd, [30, 25, 20])

        # Blank network
        leaf_f = create_node_model("basis_expansion", bkd, nvars=1, nqoi=1, max_level=2)
        mid_f = create_node_model(
            "multiplicative_additive",
            bkd,
            nvars_x=1,
            nqoi=1,
            nscaled_qoi=1,
            scale_level=0,
            delta_level=2,
        )
        root_f = create_node_model(
            "multiplicative_additive",
            bkd,
            nvars_x=1,
            nqoi=1,
            nscaled_qoi=1,
            scale_level=0,
            delta_level=2,
        )
        fit_net = MFNet(nvars=1, bkd=bkd)
        fit_net.add_node(LeafMFNetNode(0, leaf_f, noise_std=1e-2, bkd=bkd))
        fit_net.add_node(MFNetNode(1, mid_f, noise_std=1e-2, bkd=bkd))
        fit_net.add_node(RootMFNetNode(2, root_f, noise_std=1e-2, bkd=bkd))
        fit_net.add_edge(MFNetEdge(0, 1, bkd=bkd))
        fit_net.add_edge(MFNetEdge(1, 2, bkd=bkd))
        fit_net.validate()

        fitter = MFNetALSFitter(bkd, max_sweeps=30, tol=1e-14)
        result = fitter.fit(fit_net, samples, values)

        np.random.seed(123)
        test_s = bkd.asarray(np.random.uniform(-1, 1, (1, 40)))
        true_out = true_net(test_s)
        fit_out = result.surrogate()(test_s)
        bkd.assert_allclose(fit_out, true_out, atol=1e-5)

    def test_registry_list(self, bkd) -> None:
        """Verify registry contains built-in models."""
        from pyapprox.surrogates.mfnets.registry import list_node_models

        models = list_node_models()
        assert "basis_expansion" in models
        assert "multiplicative_additive" in models

    def test_registry_unknown_raises(self, bkd) -> None:
        """Verify unknown model name raises KeyError."""
        from pyapprox.surrogates.mfnets.registry import create_node_model

        with pytest.raises(KeyError):
            create_node_model("nonexistent", bkd)
