"""Tests for DGPMaximumLikelihoodFitter (Phase 9)."""

import numpy as np
import pytest

import networkx as nx

from pyapprox.optimization.minimize.adam.adam_optimizer import AdamOptimizer
from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
)
from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
    DGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPOptimizedFitResult,
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
                with_likelihood=True, fixed=False, seed=42):
    rng = np.random.RandomState(seed)
    kernel = Matern52Kernel(
        lenscale=[1.0] * nvars,
        lenscale_bounds=(0.1, 10.0),
        nvars=nvars,
        bkd=bkd,
        fixed=fixed,
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
    return DGPLayer(kernel, mean, ip, vd, bkd, likelihood=lik)


class TestDGPFitterNoActiveParams:
    def test_no_optimization_when_all_fixed(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            fixed=True, seed=10)
        layer.inducing_points().hyp_list().set_all_inactive()
        layer.variational_dist().hyp_list().set_all_inactive()
        if layer.likelihood() is not None:
            layer.likelihood().hyp_list().set_all_inactive()

        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 8))
        y = bkd.array(np.sin(rng.randn(1, 8)))
        data = {0: (X, y)}

        fitter = DGPMaximumLikelihoodFitter(bkd, n_propagation=1)
        result = fitter.fit(dgp, data)

        assert isinstance(result, GPOptimizedFitResult)
        assert result.optimization_result() is None


class TestDGPFitterSingleLayer:
    def test_neg_elbo_decreases(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 10))
        y = bkd.array(np.sin(X))
        data = {0: (X, y)}

        optimizer = AdamOptimizer(lr=1e-2, maxiter=200, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        assert isinstance(result, GPOptimizedFitResult)
        initial_nll = float(
            bkd.to_numpy(result.neg_log_marginal_likelihood())[0, 0]
        )
        from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
            DGPELBOLoss,
        )
        initial_loss = DGPELBOLoss(dgp, data, n_propagation=1)
        initial_val = float(
            bkd.to_numpy(
                initial_loss(dgp.hyp_list().get_active_values())
            )[0, 0]
        )
        assert initial_nll < initial_val

    def test_fitted_dgp_can_predict(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 10))
        y = bkd.array(np.sin(X))
        data = {0: (X, y)}

        optimizer = AdamOptimizer(lr=1e-2, maxiter=100, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        fitted_dgp = result.surrogate()
        assert fitted_dgp.is_fitted()

        X_test = bkd.array(rng.randn(1, 5))
        mean = fitted_dgp.predict(X_test)
        std = fitted_dgp.predict_std(X_test)
        assert mean.shape == (1, 5)
        assert std.shape == (1, 5)
        assert np.all(np.isfinite(bkd.to_numpy(mean)))
        assert np.all(np.isfinite(bkd.to_numpy(std)))

    def test_result_predict_delegates(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 6))
        y = bkd.array(np.sin(X))
        data = {0: (X, y)}

        optimizer = AdamOptimizer(lr=1e-2, maxiter=50, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        X_test = bkd.array(rng.randn(1, 3))
        bkd.assert_allclose(result.predict(X_test),
                            result.surrogate().predict(X_test),
                            rtol=1e-12)
        bkd.assert_allclose(result(X_test),
                            result.surrogate().predict(X_test),
                            rtol=1e-12)

    def test_clone_independence(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 6))
        y = bkd.array(np.sin(X))
        data = {0: (X, y)}

        old_vals = bkd.to_numpy(dgp.hyp_list().get_values()).copy()

        optimizer = AdamOptimizer(lr=1e-2, maxiter=50, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        fitter.fit(dgp, data)

        bkd.assert_allclose(
            dgp.hyp_list().get_values(),
            bkd.array(old_vals),
            rtol=1e-12,
        )


class TestTorchAdamOptimizer:
    def test_torch_adam_matches_generic_adam(self, torch_bkd):
        import torch
        bkd = torch_bkd
        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 10))
        y = bkd.array(np.sin(X))

        def _fit_with(optimizer_cls, seed=42):
            torch.manual_seed(seed)
            np.random.seed(seed)
            dag = nx.DiGraph()
            dag.add_node(0)
            layer = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
            prop = LayerPropagator(bkd)
            dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)
            data = {0: (X, y)}
            optimizer = optimizer_cls(lr=1e-2, maxiter=200, verbosity=0)
            fitter = DGPMaximumLikelihoodFitter(
                bkd, optimizer=optimizer, n_propagation=1,
            )
            result = fitter.fit(dgp, data)
            return float(
                bkd.to_numpy(result.neg_log_marginal_likelihood())[0, 0]
            )

        from pyapprox.optimization.minimize.adam.torch_adam_optimizer import (
            TorchAdamOptimizer,
        )
        nll_generic = _fit_with(AdamOptimizer)
        nll_torch = _fit_with(TorchAdamOptimizer)

        assert abs(nll_generic - nll_torch) / abs(nll_generic) < 1e-4


class TestDGPFitterChain:
    def test_chain_neg_elbo_decreases(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        layer0 = _make_layer(bkd, nvars=1, num_inducing=3,
                             with_likelihood=False, seed=10)
        layer1 = _make_layer(bkd, nvars=2, num_inducing=3, seed=20)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(
            dag, {0: layer0, 1: layer1}, prop, bkd,
        )

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 8))
        y = bkd.array(np.sin(rng.randn(1, 8)))
        data = {1: (X, y)}

        from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
            DGPELBOLoss,
        )
        initial_loss = DGPELBOLoss(dgp, data, n_propagation=1)
        initial_val = float(
            bkd.to_numpy(
                initial_loss(dgp.hyp_list().get_active_values())
            )[0, 0]
        )

        optimizer = AdamOptimizer(lr=1e-2, maxiter=200, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, data)

        final_nll = float(
            bkd.to_numpy(result.neg_log_marginal_likelihood())[0, 0]
        )
        assert final_nll < initial_val


class TestDGPFitterConvergence:
    """Tests verifying the fitter converges to known optima."""

    def test_single_layer_fit_reaches_titsias_optimum(self, torch_bkd):
        """Single-layer DGP at M=N: Adam reaches the closed-form
        Titsias optimum to within 0.01 nat."""
        from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
            DGPELBOLoss,
        )
        from pyapprox.surrogates.gaussianprocess.inducing.titsias import (
            titsias_optimal_whitened_q_u,
        )

        bkd = torch_bkd
        rng = np.random.RandomState(42)
        nvars = 1
        N = 8
        noise_std = 0.1
        noise_var = noise_std ** 2
        nugget = 1e-8

        X_train = bkd.array(rng.randn(nvars, N))
        y_train = bkd.array(np.sin(rng.randn(1, N)))

        kernel_ref = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        K_uu = kernel_ref(X_train, X_train) + bkd.eye(N) * nugget
        L_uu = bkd.cholesky(K_uu)
        m_star, L_star = titsias_optimal_whitened_q_u(
            K_uu, kernel_ref(X_train, X_train), y_train[0, :],
            bkd.asarray([noise_var]), L_uu, bkd,
        )

        ip_opt = InducingPoints(
            nvars=nvars, num_inducing=N, bkd=bkd,
            inducing_locations=X_train, inducing_bounds=(-10.0, 10.0),
        )
        vd_opt = GaussianVariationalDistribution(N, bkd, m_star, L_star)
        lik_opt = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
        layer_opt = DGPLayer(
            kernel_ref, ZeroMean(bkd), ip_opt, vd_opt, bkd,
            likelihood=lik_opt, nugget=nugget,
        )
        ip_opt.hyp_list().set_all_inactive()
        vd_opt.hyp_list().set_all_inactive()
        lik_opt.hyp_list().set_all_inactive()

        dag = nx.DiGraph()
        dag.add_node(0)
        prop_opt = LayerPropagator(bkd)
        dgp_opt = DeepGaussianProcess(dag, {0: layer_opt}, prop_opt, bkd)
        loss_opt = DGPELBOLoss(dgp_opt, {0: (X_train, y_train)})
        optimal_neg_elbo = float(bkd.to_numpy(
            loss_opt(dgp_opt.hyp_list().get_active_values())
        )[0, 0])

        kernel_init = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip_init = InducingPoints(
            nvars=nvars, num_inducing=N, bkd=bkd,
            inducing_locations=X_train, inducing_bounds=(-10.0, 10.0),
        )
        vd_init = GaussianVariationalDistribution(N, bkd)
        lik_init = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
        layer_init = DGPLayer(
            kernel_init, ZeroMean(bkd), ip_init, vd_init, bkd,
            likelihood=lik_init, nugget=nugget,
        )
        kernel_init.hyp_list().set_all_inactive()
        ip_init.hyp_list().set_all_inactive()
        lik_init.hyp_list().set_all_inactive()

        prop_init = LayerPropagator(bkd)
        dgp_init = DeepGaussianProcess(dag, {0: layer_init}, prop_init, bkd)

        optimizer = AdamOptimizer(lr=1e-2, maxiter=2000, verbosity=0)
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp_init, {0: (X_train, y_train)})

        fitted_neg_elbo = float(bkd.to_numpy(
            result.neg_log_marginal_likelihood()
        )[0, 0])

        assert abs(fitted_neg_elbo - optimal_neg_elbo) < 0.01, (
            f"Adam did not reach Titsias optimum: "
            f"fitted neg_elbo = {fitted_neg_elbo:.6f}, "
            f"optimal neg_elbo = {optimal_neg_elbo:.6f}, "
            f"gap = {fitted_neg_elbo - optimal_neg_elbo:.6f} nats"
        )

    @pytest.mark.slow_on("*")
    def test_two_layer_linear_kernel_layer1_reaches_titsias(self, torch_bkd):
        """2-layer DGP with linear kernels, layer 0 q(u) FROZEN at a
        chosen near-deterministic state, layer 1 q(u) free. Adam->L-BFGS-B
        should reach the Titsias-optimal layer-1 q(u).

        Uses 2D inputs to avoid linear-kernel rank deficiency (K = X.T @ X
        is rank min(d_x, M); 1D gives rank 1 with dreadful conditioning).

        With layer 0 frozen, the propagated input to layer 1 is
        approximately [X, mu_0(X)] (deterministic because L_tilde_0 is
        tiny). The Titsias formula gives layer 1's closed-form optimum
        for this fixed augmented input.

        EXERCISES: 2-layer propagation with GH, ChainedOptimizer
        (Adam -> L-BFGS-B), Titsias closed-form comparison at layer 1.
        """
        from pyapprox.optimization.minimize.chained.chained_optimizer import (
            ChainedOptimizer,
        )
        from pyapprox.optimization.minimize.scipy.lbfgsb import (
            LBFGSBOptimizer,
        )
        from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
            TensorProductGHRule,
        )
        from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
            DGPELBOLoss,
        )
        from pyapprox.surrogates.gaussianprocess.inducing.titsias import (
            titsias_optimal_whitened_q_u,
        )
        from pyapprox.surrogates.kernels.linear import LinearKernel

        bkd = torch_bkd
        rng = np.random.RandomState(42)
        nvars_x = 2
        N = 8
        signal_var = 1.0
        noise_std = 0.1
        noise_var = noise_std ** 2
        nugget = 1e-6

        X_train = bkd.array(rng.randn(nvars_x, N))
        X_np = bkd.to_numpy(X_train)
        y_train = bkd.array(
            (X_np[0, :] * X_np[1, :]).reshape(1, -1)
        )

        # --- Layer 0: frozen at a chosen near-deterministic q(u_0) ---
        m_tilde_0 = bkd.array(rng.randn(N))
        l_tilde_diag = 1e-6
        L_tilde_0 = bkd.eye(N) * l_tilde_diag

        kernel0 = LinearKernel(
            signal_variance=signal_var,
            signal_variance_bounds=(0.1, 10.0),
            nvars=nvars_x, bkd=bkd, fixed=True,
        )
        K_uu_0 = kernel0(X_train, X_train) + bkd.eye(N) * nugget
        L_uu_0 = bkd.cholesky(K_uu_0)

        # mu_0(X) = alpha_0.T @ m_tilde_0 where alpha_0 = L_uu_0^{-1} K_u0X
        alpha_0 = bkd.solve_triangular(
            L_uu_0, kernel0(X_train, X_train), lower=True,
        )
        mu_0_X = alpha_0.T @ m_tilde_0

        # --- Layer 1's augmented input: [X, mu_0(X)] ---
        h0_dim = 1
        nvars_layer1 = nvars_x + h0_dim
        X_aug = bkd.vstack([X_train, bkd.reshape(mu_0_X, (1, N))])

        # --- Compute layer 1's Titsias optimum for fixed X_aug ---
        kernel1_ref = LinearKernel(
            signal_variance=signal_var,
            signal_variance_bounds=(0.1, 10.0),
            nvars=nvars_layer1, bkd=bkd, fixed=True,
        )
        K_uu_1 = kernel1_ref(X_aug, X_aug) + bkd.eye(N) * nugget
        L_uu_1 = bkd.cholesky(K_uu_1)
        m_star_1, L_star_1 = titsias_optimal_whitened_q_u(
            K_uu_1, kernel1_ref(X_aug, X_aug), y_train[0, :],
            bkd.asarray([noise_var]), L_uu_1, bkd,
        )

        # --- Build optimal DGP (all params frozen) and eval neg_elbo ---
        def _build_frozen_layer0():
            k = LinearKernel(
                signal_variance=signal_var,
                signal_variance_bounds=(0.1, 10.0),
                nvars=nvars_x, bkd=bkd, fixed=True,
            )
            ip = InducingPoints(
                nvars=nvars_x, num_inducing=N, bkd=bkd,
                inducing_locations=X_train, inducing_bounds=(-10.0, 10.0),
            )
            vd = GaussianVariationalDistribution(N, bkd, m_tilde_0, L_tilde_0)
            layer = DGPLayer(
                k, ZeroMean(bkd), ip, vd, bkd,
                likelihood=None, nugget=nugget,
            )
            k.hyp_list().set_all_inactive()
            ip.hyp_list().set_all_inactive()
            vd.hyp_list().set_all_inactive()
            return layer

        def _build_layer1(m_tilde, L_tilde, freeze_vd=False):
            k = LinearKernel(
                signal_variance=signal_var,
                signal_variance_bounds=(0.1, 10.0),
                nvars=nvars_layer1, bkd=bkd, fixed=True,
            )
            ip = InducingPoints(
                nvars=nvars_layer1, num_inducing=N, bkd=bkd,
                inducing_locations=X_aug, inducing_bounds=(-10.0, 10.0),
            )
            vd = GaussianVariationalDistribution(N, bkd, m_tilde, L_tilde)
            lik = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
            layer = DGPLayer(
                k, ZeroMean(bkd), ip, vd, bkd,
                likelihood=lik, nugget=nugget,
            )
            k.hyp_list().set_all_inactive()
            ip.hyp_list().set_all_inactive()
            lik.hyp_list().set_all_inactive()
            if freeze_vd:
                vd.hyp_list().set_all_inactive()
            return layer

        dag = nx.DiGraph()
        dag.add_edge(0, 1)

        layer0_opt = _build_frozen_layer0()
        layer1_opt = _build_layer1(m_star_1, L_star_1, freeze_vd=True)
        prop_opt = LayerPropagator(bkd, rule=TensorProductGHRule(order=7))
        dgp_opt = DeepGaussianProcess(
            dag, {0: layer0_opt, 1: layer1_opt}, prop_opt, bkd,
        )
        loss_opt = DGPELBOLoss(
            dgp_opt, {1: (X_train, y_train)}, n_propagation=49,
        )
        optimal_neg_elbo = float(bkd.to_numpy(
            loss_opt(dgp_opt.hyp_list().get_active_values())
        )[0, 0])

        # --- Build fresh DGP: layer 0 FROZEN (same state), layer 1 at prior ---
        layer0_init = _build_frozen_layer0()
        layer1_init = _build_layer1(None, None, freeze_vd=False)
        prop_init = LayerPropagator(bkd, rule=TensorProductGHRule(order=7))
        dgp_init = DeepGaussianProcess(
            dag, {0: layer0_init, 1: layer1_init}, prop_init, bkd,
        )

        chained_opt = ChainedOptimizer(
            AdamOptimizer(lr=1e-2, maxiter=500, verbosity=0),
            LBFGSBOptimizer(maxiter=100, verbosity=0),
        )
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=chained_opt, n_propagation=49,
        )
        result = fitter.fit(dgp_init, {1: (X_train, y_train)})

        fitted_neg_elbo = float(bkd.to_numpy(
            result.neg_log_marginal_likelihood()
        )[0, 0])

        assert abs(fitted_neg_elbo - optimal_neg_elbo) < 0.05, (
            f"Adam->L-BFGS-B did not reach layer-1 Titsias optimum: "
            f"fitted = {fitted_neg_elbo:.6f}, "
            f"optimal = {optimal_neg_elbo:.6f}, "
            f"gap = {fitted_neg_elbo - optimal_neg_elbo:.6f} nats"
        )

    def test_two_layer_fit_reproducible_under_fixed_seed(self, torch_bkd):
        """Two identical fits with the same RNG seed produce
        bit-identical fitted parameters."""
        from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
            MonteCarloRule,
        )

        bkd = torch_bkd
        rng_data = np.random.RandomState(42)
        X_train = bkd.array(rng_data.randn(1, 8))
        y_train = bkd.array(np.sin(rng_data.randn(1, 8)))
        data = {1: (X_train, y_train)}

        def build_and_fit(rng_seed: int) -> np.ndarray:
            np.random.seed(rng_seed)
            mc_rule = MonteCarloRule(rng=np.random.default_rng(rng_seed))

            dag = nx.DiGraph()
            dag.add_edge(0, 1)
            layer0 = _make_layer(
                bkd, nvars=1, num_inducing=3,
                with_likelihood=False, fixed=False, seed=99,
            )
            layer1 = _make_layer(
                bkd, nvars=2, num_inducing=3, fixed=False, seed=100,
            )
            prop = LayerPropagator(bkd, rule=mc_rule)
            dgp = DeepGaussianProcess(
                dag, {0: layer0, 1: layer1}, prop, bkd,
            )

            optimizer = AdamOptimizer(lr=1e-2, maxiter=200, verbosity=0)
            fitter = DGPMaximumLikelihoodFitter(
                bkd, optimizer=optimizer, n_propagation=1,
            )
            fitter.fit(dgp, data)
            return bkd.to_numpy(dgp.hyp_list().get_values()).copy()

        params_1 = build_and_fit(rng_seed=42)
        params_2 = build_and_fit(rng_seed=42)

        np.testing.assert_array_equal(
            params_1, params_2,
            err_msg=(
                "Two fits with same seed produced different parameters. "
                "Possible causes: hidden global RNG, torch nondeterminism, "
                "state leaking between fitter calls."
            ),
        )
