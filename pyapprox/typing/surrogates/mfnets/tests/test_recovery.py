"""End-to-end recovery tests for MFNets.

Strategy: create a true MFNet with random coefficients, generate training
data from it, then fit a fresh MFNet and verify predictions match.
"""

import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    MFNetNode,
    RootMFNetNode,
)
from pyapprox.typing.surrogates.mfnets.edges import MFNetEdge
from pyapprox.typing.surrogates.mfnets.network import MFNet
from pyapprox.typing.surrogates.mfnets.registry import create_node_model
from pyapprox.typing.surrogates.mfnets.fitters.gradient_fitter import (
    MFNetGradientFitter,
)
from pyapprox.typing.surrogates.mfnets.fitters.als_fitter import (
    MFNetALSFitter,
)


def _build_two_node_mfnet(
    bkd: Backend[Array],
    leaf_level: int = 3,
    scale_level: int = 1,
    delta_level: int = 3,
    seed: int = 0,
) -> MFNet[Array]:
    """Build a 2-node MFNet with random coefficients."""
    np.random.seed(seed)
    leaf = create_node_model(
        "basis_expansion", bkd, nvars=1, nqoi=1, max_level=leaf_level
    )
    nterms_leaf = leaf.nterms()
    leaf.set_coefficients(bkd.asarray(np.random.randn(nterms_leaf, 1)))

    root = create_node_model(
        "multiplicative_additive", bkd,
        nvars_x=1, nqoi=1, nscaled_qoi=1,
        scale_level=scale_level, delta_level=delta_level,
    )
    # Set random coefficients for all sub-models
    for sm in root.scaling_models():
        sm.set_coefficients(
            bkd.asarray(np.random.randn(sm.nterms(), sm.nqoi()))
        )
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
    bkd: Backend[Array],
    leaf_level: int = 3,
    scale_level: int = 1,
    delta_level: int = 3,
) -> MFNet[Array]:
    """Build a 2-node MFNet with zero coefficients."""
    leaf = create_node_model(
        "basis_expansion", bkd, nvars=1, nqoi=1, max_level=leaf_level
    )
    root = create_node_model(
        "multiplicative_additive", bkd,
        nvars_x=1, nqoi=1, nscaled_qoi=1,
        scale_level=scale_level, delta_level=delta_level,
    )
    net = MFNet(nvars=1, bkd=bkd)
    net.add_node(LeafMFNetNode(0, leaf, noise_std=1e-2, bkd=bkd))
    net.add_node(RootMFNetNode(1, root, noise_std=1e-2, bkd=bkd))
    net.add_edge(MFNetEdge(0, 1, bkd=bkd))
    net.validate()
    return net


def _generate_data(
    net: MFNet[Array], bkd: Backend[Array],
    n_per_node: List[int], seed: int = 42
) -> Any:
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


class TestRecovery(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_als_recovers_two_node(self) -> None:
        """ALS should recover predictions for a 2-node network."""
        bkd = self._bkd
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

    def test_gradient_recovers_two_node(self) -> None:
        """Gradient fitter should recover predictions for a 2-node network."""
        bkd = self._bkd
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

    def test_als_recovers_three_node_chain(self) -> None:
        """ALS on 3-node chain: leaf -> mid -> root."""
        bkd = self._bkd
        np.random.seed(30)

        # True network
        leaf = create_node_model("basis_expansion", bkd, nvars=1, nqoi=1, max_level=2)
        leaf.set_coefficients(bkd.asarray(np.random.randn(leaf.nterms(), 1)))
        mid = create_node_model(
            "multiplicative_additive", bkd,
            nvars_x=1, nqoi=1, nscaled_qoi=1,
            scale_level=0, delta_level=2,
        )
        for sm in mid.scaling_models():
            sm.set_coefficients(bkd.asarray(np.random.randn(sm.nterms(), sm.nqoi())))
        mid.delta_model().set_coefficients(
            bkd.asarray(np.random.randn(mid.delta_model().nterms(), mid.nqoi()))
        )
        root = create_node_model(
            "multiplicative_additive", bkd,
            nvars_x=1, nqoi=1, nscaled_qoi=1,
            scale_level=0, delta_level=2,
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
            "multiplicative_additive", bkd,
            nvars_x=1, nqoi=1, nscaled_qoi=1,
            scale_level=0, delta_level=2,
        )
        root_f = create_node_model(
            "multiplicative_additive", bkd,
            nvars_x=1, nqoi=1, nscaled_qoi=1,
            scale_level=0, delta_level=2,
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

    def test_registry_list(self) -> None:
        """Verify registry contains built-in models."""
        from pyapprox.typing.surrogates.mfnets.registry import list_node_models
        models = list_node_models()
        self.assertIn("basis_expansion", models)
        self.assertIn("multiplicative_additive", models)

    def test_registry_unknown_raises(self) -> None:
        """Verify unknown model name raises KeyError."""
        from pyapprox.typing.surrogates.mfnets.registry import create_node_model
        with self.assertRaises(KeyError):
            create_node_model("nonexistent", self._bkd)


# --- Concrete backend test classes ---

class TestRecoveryNumpy(TestRecovery[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRecoveryTorch(TestRecovery[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
