"""Tests for LayerPropagator (Phase 7)."""

import numpy as np
import pytest

import networkx as nx

from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerOutputDist,
    LayerPropagator,
)
from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
    InducingPoints,
)
from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
    GaussianVariationalDistribution,
)
from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
    GaussianLikelihood,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import ZeroMean
from pyapprox.surrogates.kernels.matern import Matern52Kernel


def _make_layer(bkd, nvars=1, num_inducing=5, noise_std=0.1,
                with_likelihood=True, seed=42):
    rng = np.random.RandomState(seed)
    kernel = Matern52Kernel(
        lenscale=[1.0] * nvars,
        lenscale_bounds=(0.1, 10.0),
        nvars=nvars,
        bkd=bkd,
        fixed=True,
    )
    mean = ZeroMean(bkd)
    locs = bkd.array(rng.randn(nvars, num_inducing))
    ip = InducingPoints(
        nvars=nvars,
        num_inducing=num_inducing,
        bkd=bkd,
        inducing_locations=locs,
        inducing_bounds=(-5.0, 5.0),
    )
    vd = GaussianVariationalDistribution(num_inducing, bkd)
    lik = None
    if with_likelihood:
        lik = GaussianLikelihood(noise_std, (1e-6, 1.0), bkd)
    ip.hyp_list().set_all_inactive()
    vd.hyp_list().set_all_inactive()
    if lik is not None:
        lik.hyp_list().set_all_inactive()
    return DGPLayer(kernel, mean, ip, vd, bkd, likelihood=lik)


class TestLayerPropagatorSingleNode:
    """Single-layer propagation: predict_at must match layer.predict_marginal."""

    def test_single_node_mean_var(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        layers = {0: layer}

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(2, 8))

        prop = LayerPropagator(bkd)
        means, variances, weights = prop.predict_at(
            dag, layers, X, target_node=0,
        )

        assert means.shape == (1, 1, 8)
        assert variances.shape == (1, 1, 8)
        assert weights.shape == (1,)

        expected_mean, expected_var = layer.predict_marginal(X)
        bkd.assert_allclose(means[0], expected_mean, rtol=1e-12)
        bkd.assert_allclose(variances[0], expected_var, rtol=1e-12)
        bkd.assert_allclose(weights, bkd.ones((1,)), rtol=1e-12)

    def test_single_node_predict_mean_and_std(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        layers = {0: layer}

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(2, 8))

        prop = LayerPropagator(bkd)
        mean, std = prop.predict_mean_and_std(
            dag, layers, X, target_node=0,
        )

        expected_mean, expected_var = layer.predict_marginal(X)
        expected_std = bkd.sqrt(expected_var)
        bkd.assert_allclose(mean, expected_mean, rtol=1e-12)
        bkd.assert_allclose(std, expected_std, rtol=1e-12)

    def test_single_node_no_samples_at_leaf(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=5)
        layers = {0: layer}

        X = bkd.array(np.random.RandomState(0).randn(1, 5))
        prop = LayerPropagator(bkd)
        cache = prop.forward(dag, layers, X)
        assert cache[0].samples is None


class TestLayerPropagatorTwoLayerChain:
    """Two-layer chain: 0 -> 1."""

    def test_chain_shapes(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=3, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(2, 6))
        S = 5

        prop = LayerPropagator(bkd)
        means, variances, weights = prop.predict_at(
            dag, layers, X, target_node=1, n_samples=S,
        )

        assert means.shape == (S, 1, 6)
        assert variances.shape == (S, 1, 6)
        assert weights.shape == (S,)

    def test_chain_predict_mean_and_std_shapes(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=3, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(2, 6))

        prop = LayerPropagator(bkd)
        mean, std = prop.predict_mean_and_std(
            dag, layers, X, target_node=1, n_samples=10,
        )

        assert mean.shape == (1, 6)
        assert std.shape == (1, 6)

    def test_chain_root_matches_standalone(self, bkd):
        """Root node in a chain still matches standalone predict_marginal."""
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=3, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(2, 6))

        prop = LayerPropagator(bkd)
        means, variances, weights = prop.predict_at(
            dag, layers, X, target_node=0, n_samples=5,
        )

        expected_mean, expected_var = layer0.predict_marginal(X)
        bkd.assert_allclose(means[0], expected_mean, rtol=1e-12)
        bkd.assert_allclose(variances[0], expected_var, rtol=1e-12)

    def test_chain_variance_nonnegative(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        X = bkd.array(np.random.RandomState(0).randn(1, 10))
        prop = LayerPropagator(bkd)
        _, variances, _ = prop.predict_at(
            dag, layers, X, target_node=1, n_samples=5,
        )
        assert float(bkd.to_numpy(bkd.min(variances))) >= 0.0


class TestLayerPropagatorDiamond:
    """Diamond DAG: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3."""

    def test_diamond_shapes(self, bkd):
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

        nvars_x = 1
        M = 4
        layer0 = _make_layer(bkd, nvars=nvars_x, num_inducing=M, seed=10)
        layer1 = _make_layer(bkd, nvars=nvars_x + 1, num_inducing=M, seed=20)
        layer2 = _make_layer(bkd, nvars=nvars_x + 1, num_inducing=M, seed=30)
        layer3 = _make_layer(
            bkd, nvars=nvars_x + 2, num_inducing=M, seed=40,
        )
        layers = {0: layer0, 1: layer1, 2: layer2, 3: layer3}

        X = bkd.array(np.random.RandomState(0).randn(nvars_x, 5))
        S = 3

        prop = LayerPropagator(bkd)
        means, variances, weights = prop.predict_at(
            dag, layers, X, target_node=3, n_samples=S,
        )

        assert means.shape == (S, 1, 5)
        assert variances.shape == (S, 1, 5)
        assert weights.shape == (S,)

    def test_diamond_all_nodes_cached(self, bkd):
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

        nvars_x = 1
        M = 3
        layer0 = _make_layer(bkd, nvars=nvars_x, num_inducing=M, seed=10)
        layer1 = _make_layer(bkd, nvars=nvars_x + 1, num_inducing=M, seed=20)
        layer2 = _make_layer(bkd, nvars=nvars_x + 1, num_inducing=M, seed=30)
        layer3 = _make_layer(
            bkd, nvars=nvars_x + 2, num_inducing=M, seed=40,
        )
        layers = {0: layer0, 1: layer1, 2: layer2, 3: layer3}

        X = bkd.array(np.random.RandomState(0).randn(nvars_x, 4))
        prop = LayerPropagator(bkd)
        cache = prop.forward(dag, layers, X, n_samples=3)

        assert set(cache.keys()) == {0, 1, 2, 3}
        assert cache[3].samples is None


class TestLayerPropagatorSampleForward:
    def test_sample_forward_shapes(self, bkd):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        X = bkd.array(np.random.RandomState(0).randn(1, 6))
        S = 7

        prop = LayerPropagator(bkd)
        sample_cache = prop.sample_forward(dag, layers, X, n_samples=S)

        assert sample_cache[0].shape == (S, 1, 6)
        assert sample_cache[1].shape == (S, 1, 6)

    def test_sample_forward_single_node(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=2, num_inducing=4, seed=10)
        layers = {0: layer}

        X = bkd.array(np.random.RandomState(0).randn(2, 5))
        S = 3

        prop = LayerPropagator(bkd)
        sample_cache = prop.sample_forward(dag, layers, X, n_samples=S)
        assert sample_cache[0].shape == (S, 1, 5)


class TestPropagatorMCvsGH:
    """MC propagation should converge to tensor-GH propagation."""

    def test_two_layer_mc_converges_to_gh(self, numpy_bkd):
        """Leaf mean/std under MC at large S should match GH."""
        from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
            MonteCarloRule,
            TensorProductGHRule,
        )

        bkd = numpy_bkd
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        X = bkd.array(np.random.RandomState(0).randn(1, 5))

        gh_rule = TensorProductGHRule(order=7)
        prop_gh = LayerPropagator(bkd, rule=gh_rule)
        mean_gh, std_gh = prop_gh.predict_mean_and_std(
            dag, layers, X, target_node=1, n_samples=49,
        )

        mc_rule = MonteCarloRule(rng=np.random.default_rng(42))
        prop_mc = LayerPropagator(bkd, rule=mc_rule)
        mean_mc, std_mc = prop_mc.predict_mean_and_std(
            dag, layers, X, target_node=1, n_samples=10000,
        )

        bkd.assert_allclose(mean_mc, mean_gh, rtol=0.05, atol=0.02)
        bkd.assert_allclose(std_mc, std_gh, rtol=0.10, atol=0.02)

    def test_chain_leaf_variance_at_least_layer_variance(self, numpy_bkd):
        """A 2-layer DGP's leaf std should be at least the root layer's
        intrinsic std far from inducing points. Propagation adds
        uncertainty; it should never collapse it."""
        bkd = numpy_bkd
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        X_far = bkd.array([[10.0, -10.0, 15.0]])

        prop = LayerPropagator(bkd)
        _, std_chain = prop.predict_mean_and_std(
            dag, layers, X_far, target_node=1, n_samples=500,
        )

        _, var_root = layer0.predict_marginal(X_far)
        std_root = bkd.sqrt(var_root)

        std_chain_np = bkd.to_numpy(std_chain).ravel()
        std_root_np = bkd.to_numpy(std_root).ravel()

        for i in range(3):
            assert std_chain_np[i] >= 0.5 * std_root_np[i], (
                f"Chain std {std_chain_np[i]:.4f} collapsed below "
                f"root std {std_root_np[i]:.4f} at point {i}"
            )

    def test_two_layer_sobol_converges_to_gh(self, numpy_bkd):
        """Sobol propagation at large S should match tensor GH."""
        from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
            SobolRule,
            TensorProductGHRule,
        )

        bkd = numpy_bkd
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        X = bkd.array(np.random.RandomState(0).randn(1, 5))

        gh_rule = TensorProductGHRule(order=7)
        prop_gh = LayerPropagator(bkd, rule=gh_rule)
        mean_gh, std_gh = prop_gh.predict_mean_and_std(
            dag, layers, X, target_node=1, n_samples=49,
        )

        sobol_rule = SobolRule(seed=42)
        prop_sobol = LayerPropagator(bkd, rule=sobol_rule)
        mean_sobol, std_sobol = prop_sobol.predict_mean_and_std(
            dag, layers, X, target_node=1, n_samples=2048,
        )

        bkd.assert_allclose(mean_sobol, mean_gh, rtol=0.03, atol=0.01)
        bkd.assert_allclose(std_sobol, std_gh, rtol=0.05, atol=0.01)


class TestPropagatorDimensionConsistency:
    """Both forward and sample_forward must request the same number of
    dimensions from the rule on the same DAG."""

    def test_forward_and_sample_request_same_dimensions(self, numpy_bkd):
        """Use a custom rule that records the n_layers it is asked for.
        Confirm forward() and sample_forward() both pass the same value."""
        from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
            MonteCarloRule,
        )

        bkd = numpy_bkd

        class _RecordingRule:
            def __init__(self):
                self._inner = MonteCarloRule(
                    rng=np.random.default_rng(0),
                )
                self.observed_dims: list[int] = []

            def __call__(self, n_samples, n_layers, bkd):
                self.observed_dims.append(n_layers)
                return self._inner(n_samples, n_layers, bkd)

        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=4, seed=20)
        layers = {0: layer0, 1: layer1}

        X = bkd.array(np.random.RandomState(0).randn(1, 5))
        rule = _RecordingRule()
        prop = LayerPropagator(bkd, rule=rule)

        prop.predict_mean_and_std(
            dag, layers, X, target_node=1, n_samples=5,
        )
        prop.sample_forward(dag, layers, X, n_samples=5)

        assert len(rule.observed_dims) == 2
        assert rule.observed_dims[0] == rule.observed_dims[1], (
            f"forward() requested {rule.observed_dims[0]} dimensions "
            f"but sample_forward() requested {rule.observed_dims[1]}; "
            f"they must agree for the same DAG."
        )
        assert rule.observed_dims[0] == 2
