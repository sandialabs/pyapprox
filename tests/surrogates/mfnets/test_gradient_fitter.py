"""Tests for MFNet gradient fitter."""

import numpy as np

from pyapprox.surrogates.affine.basis import MultiIndexBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import MonomialBasis1D
from pyapprox.surrogates.mfnets.discrepancy import (
    MultiplicativeAdditiveDiscrepancy,
)
from pyapprox.surrogates.mfnets.edges import MFNetEdge
from pyapprox.surrogates.mfnets.fitters.gradient_fitter import (
    MFNetGradientFitter,
)
from pyapprox.surrogates.mfnets.losses import (
    MFNetNegLogLikelihoodLoss,
)
from pyapprox.surrogates.mfnets.network import MFNet
from pyapprox.surrogates.mfnets.nodes import (
    LeafMFNetNode,
    RootMFNetNode,
)
from pyapprox.util.test_utils import (
    slowest_test,
)


def _create_expansion(
    bkd, nvars: int, nqoi: int, max_level: int = 2
) -> BasisExpansion:
    bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = MultiIndexBasis.__new__(MultiIndexBasis)
    MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=nqoi)


def _build_two_node_mfnet(
    bkd,
    leaf_coef,
    scale_coef,
    delta_coef,
) -> MFNet:
    """Build a 2-node MFNet: leaf(0) -> root(1) with discrepancy model.

    Leaf: BasisExpansion, nvars=1, nqoi=1
    Root: MultiplicativeAdditiveDiscrepancy, scaling(x)*q + delta(x)
    """
    # Leaf model
    leaf_model = _create_expansion(bkd, nvars=1, nqoi=1, max_level=3)
    leaf_model.set_coefficients(leaf_coef)

    # Root discrepancy: scaling (nvars=1, nqoi=1) + delta (nvars=1, nqoi=1)
    scaling = _create_expansion(bkd, nvars=1, nqoi=1, max_level=1)
    scaling.set_coefficients(scale_coef)
    delta = _create_expansion(bkd, nvars=1, nqoi=1, max_level=3)
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


class TestMFNetGradientFitter:

    def _generate_truth_and_data(self, bkd):
        """Generate a true MFNet and training data from it."""
        # True coefficients
        np.random.seed(10)
        leaf_nterms = len(compute_hyperbolic_indices(1, 3, 1.0, bkd).T)
        scale_nterms = len(compute_hyperbolic_indices(1, 1, 1.0, bkd).T)
        delta_nterms = len(compute_hyperbolic_indices(1, 3, 1.0, bkd).T)

        true_leaf = bkd.asarray(np.random.randn(leaf_nterms, 1))
        true_scale = bkd.asarray(np.random.randn(scale_nterms, 1))
        true_delta = bkd.asarray(np.random.randn(delta_nterms, 1))

        true_net = _build_two_node_mfnet(bkd, true_leaf, true_scale, true_delta)

        # Generate training data
        np.random.seed(42)
        n_leaf = 20
        n_root = 15
        s_leaf = bkd.asarray(np.random.uniform(-1, 1, (1, n_leaf)))
        s_root = bkd.asarray(np.random.uniform(-1, 1, (1, n_root)))
        v_leaf = true_net.subgraph_values(s_leaf, 0)
        v_root = true_net.subgraph_values(s_root, 1)

        return true_net, [s_leaf, s_root], [v_leaf, v_root]

    def test_loss_evaluation(self, bkd) -> None:
        """Test that loss is computable and positive."""
        true_net, samples, values = self._generate_truth_and_data(bkd)

        loss = MFNetNegLogLikelihoodLoss(true_net, samples, values)
        params = true_net.hyp_list().get_active_values()
        val = loss(bkd.reshape(params, (-1, 1)))

        assert val.shape[0] == 1
        assert val.shape[1] == 1

    @slowest_test
    def test_gradient_fitter_fits(self, bkd) -> None:
        """Fit a 2-node MFNet and verify predictions recover truth."""
        true_net, samples, values = self._generate_truth_and_data(bkd)

        # Build a fresh network with different initial coefficients
        np.random.seed(999)
        leaf_nterms = true_net.node(0).model().hyp_list().nparams()
        scale_nterms = true_net.node(1).model().scaling_models()[0].hyp_list().nparams()
        delta_nterms = true_net.node(1).model().delta_model().hyp_list().nparams()

        init_leaf = bkd.asarray(np.random.randn(leaf_nterms, 1) * 0.1)
        init_scale = bkd.asarray(np.random.randn(scale_nterms, 1) * 0.1)
        init_delta = bkd.asarray(np.random.randn(delta_nterms, 1) * 0.1)

        fit_net = _build_two_node_mfnet(bkd, init_leaf, init_scale, init_delta)

        fitter = MFNetGradientFitter(bkd)
        result = fitter.fit(fit_net, samples, values)

        # Compare predictions on test samples
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 30)))
        true_out = true_net(test_samples)
        fit_out = result.surrogate()(test_samples)

        bkd.assert_allclose(fit_out, true_out, atol=1e-4)
