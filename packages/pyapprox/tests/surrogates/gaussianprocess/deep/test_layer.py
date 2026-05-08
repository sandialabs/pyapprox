"""Tests for DGPLayer (Phase 5)."""

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
    InducingPoints,
)
from pyapprox.surrogates.gaussianprocess.inducing.titsias import (
    titsias_optimal_whitened_q_u,
)
from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
    GaussianVariationalDistribution,
)
from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
    GaussianLikelihood,
)
from pyapprox.surrogates.gaussianprocess.mean_functions import ZeroMean
from pyapprox.surrogates.gaussianprocess.variational import (
    VariationalGaussianProcess,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


def _make_layer(
    bkd,
    nvars=1,
    num_inducing=5,
    noise_std=0.1,
    with_likelihood=True,
    fixed=True,
    seed=42,
):
    """Helper to create a DGPLayer for tests."""
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
    if fixed:
        ip.hyp_list().set_all_inactive()
        vd.hyp_list().set_all_inactive()
        if lik is not None:
            lik.hyp_list().set_all_inactive()
    return DGPLayer(kernel, mean, ip, vd, bkd, likelihood=lik)


class TestDGPLayerBasic:
    def test_predict_marginal_shape(self, bkd):
        layer = _make_layer(bkd, nvars=2, num_inducing=4)
        h = bkd.array(np.random.RandomState(0).randn(2, 10))
        mean, var = layer.predict_marginal(h)
        assert mean.shape == (1, 10)
        assert var.shape == (1, 10)

    def test_variance_nonnegative(self, bkd):
        layer = _make_layer(bkd, nvars=2, num_inducing=6)
        h = bkd.array(np.random.RandomState(1).randn(2, 20))
        _, var = layer.predict_marginal(h)
        assert float(bkd.to_numpy(bkd.min(var))) >= 0.0

    def test_sample_shape(self, bkd):
        layer = _make_layer(bkd, nvars=1, num_inducing=4)
        h = bkd.array(np.random.RandomState(2).randn(1, 8))
        samples = layer.sample(h, n_samples=7)
        assert samples.shape == (7, 1, 8)

    def test_kl_prior_is_zero(self, bkd):
        """KL = 0 when q(u) = prior (m_tilde=0, L_tilde=I)."""
        layer = _make_layer(bkd)
        kl = layer.kl_to_prior()
        bkd.assert_allclose(
            bkd.asarray([kl]), bkd.zeros((1,)), atol=1e-12,
        )

    def test_hyp_list_count(self, bkd):
        nvars = 2
        M = 5
        layer = _make_layer(bkd, nvars=nvars, num_inducing=M, fixed=False)
        n_kernel = layer.kernel().hyp_list().nparams()
        n_mean = layer.mean_function().hyp_list().nparams()
        n_ip = layer.inducing_points().hyp_list().nparams()
        n_vd = layer.variational_dist().hyp_list().nparams()
        n_lik = layer.likelihood().hyp_list().nparams()
        expected = n_kernel + n_mean + n_ip + n_vd + n_lik
        assert layer.hyp_list().nparams() == expected

    def test_hyp_list_no_likelihood(self, bkd):
        layer = _make_layer(bkd, with_likelihood=False, fixed=False)
        n_kernel = layer.kernel().hyp_list().nparams()
        n_mean = layer.mean_function().hyp_list().nparams()
        n_ip = layer.inducing_points().hyp_list().nparams()
        n_vd = layer.variational_dist().hyp_list().nparams()
        expected = n_kernel + n_mean + n_ip + n_vd
        assert layer.hyp_list().nparams() == expected

    def test_clone_unfitted_independence(self, bkd):
        layer = _make_layer(bkd, fixed=False)
        clone = layer._clone_unfitted()
        old_vals = bkd.to_numpy(layer.hyp_list().get_values()).copy()
        clone.hyp_list().set_values(
            clone.hyp_list().get_values() + bkd.ones(
                (clone.hyp_list().nparams(),)
            ) * 0.5
        )
        bkd.assert_allclose(
            layer.hyp_list().get_values(),
            bkd.array(old_vals),
            rtol=1e-12,
        )


class TestDGPLayerMatchesExactGP:
    """Tier 1: M=N SVGP = exact GP with Titsias-optimal q(u).

    Three-way consistency: ExactGP, VGP, and DGPLayer must agree on
    latent predictions (mean and std) when M=N, Z=X, and q(u) is set
    to the Titsias optimum.

    ExactGP uses signal-only kernel with nugget=noise_var so that
    predict/predict_std return latent quantities directly.

    nugget << noise_var so the jitter plays no role in the comparison.
    """

    def test_three_way_latent_consistency(self, bkd):
        rng = np.random.RandomState(123)
        nvars = 1
        N = 6
        noise_std = 0.1
        noise_var = noise_std**2
        nugget = 1e-8

        X_train = bkd.array(rng.randn(nvars, N))
        y_train = bkd.array(np.sin(rng.randn(nvars, N)[:1, :]))
        X_test = bkd.array(rng.randn(nvars, 15))

        # --- ExactGP (ground truth) ---
        exact_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        exact_gp = ExactGaussianProcess(
            exact_kernel, nvars, bkd, nugget=noise_var,
        )
        exact_gp._fit_internal(X_train, y_train)
        exact_mean = exact_gp.predict(X_test)
        exact_std = exact_gp.predict_std(X_test)

        # --- VGP (collapsed Titsias) ---
        vgp_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip_vgp = InducingPoints(
            nvars=nvars, num_inducing=N, bkd=bkd,
            inducing_locations=X_train, inducing_bounds=(-10.0, 10.0),
        )
        lik_vgp = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
        ip_vgp.hyp_list().set_all_inactive()
        lik_vgp.hyp_list().set_all_inactive()
        vgp = VariationalGaussianProcess(
            vgp_kernel, nvars, ip_vgp, lik_vgp, bkd,
            mean_function=ZeroMean(bkd), nugget=nugget,
        )
        vgp._fit_internal(X_train, y_train)
        vgp_mean = vgp.predict(X_test)
        vgp_std = vgp.predict_std(X_test)

        # --- DGPLayer (Hensman explicit q(u)) ---
        layer_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip_layer = InducingPoints(
            nvars=nvars, num_inducing=N, bkd=bkd,
            inducing_locations=X_train, inducing_bounds=(-10.0, 10.0),
        )
        K_uu = layer_kernel(X_train, X_train)
        K_uu_nug = K_uu + bkd.eye(N) * nugget
        L_uu = bkd.cholesky(K_uu_nug)

        m_tilde, L_tilde = titsias_optimal_whitened_q_u(
            K_uu_nug, K_uu, y_train[0, :], bkd.asarray([noise_var]),
            L_uu, bkd,
        )
        vd = GaussianVariationalDistribution(N, bkd, m_tilde, L_tilde)

        lik_layer = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
        layer = DGPLayer(
            layer_kernel, ZeroMean(bkd), ip_layer, vd, bkd,
            likelihood=lik_layer, nugget=nugget,
        )
        ip_layer.hyp_list().set_all_inactive()
        vd.hyp_list().set_all_inactive()
        lik_layer.hyp_list().set_all_inactive()

        layer_mean, layer_var = layer.predict_marginal(X_test)
        layer_std = bkd.sqrt(layer_var)

        # Three-way mean comparison
        bkd.assert_allclose(exact_mean, vgp_mean, rtol=1e-4, atol=1e-6)
        bkd.assert_allclose(exact_mean, layer_mean, rtol=1e-4, atol=1e-6)
        bkd.assert_allclose(vgp_mean, layer_mean, rtol=1e-4, atol=1e-6)

        # Three-way latent std comparison
        bkd.assert_allclose(exact_std, vgp_std, rtol=1e-4, atol=1e-6)
        bkd.assert_allclose(exact_std, layer_std, rtol=1e-4, atol=1e-6)
        bkd.assert_allclose(vgp_std, layer_std, rtol=1e-4, atol=1e-6)

    def test_three_way_low_noise_regression(self, bkd):
        """Consistency at near-zero noise with looser tolerance.

        When noise_var ~ nugget, implementations diverge because the
        effective diagonal perturbation differs (ExactGP sees noise_var
        alone; VGP/DGPLayer see noise_var in the data term plus nugget
        in K_uu). This test checks they don't catastrophically diverge.
        """
        rng = np.random.RandomState(123)
        nvars = 1
        N = 6
        noise_std = 1e-3
        noise_var = noise_std**2
        nugget = 1e-6

        X_train = bkd.array(rng.randn(nvars, N))
        y_train = bkd.array(np.sin(rng.randn(nvars, N)[:1, :]))
        X_test = bkd.array(rng.randn(nvars, 15))

        exact_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        exact_gp = ExactGaussianProcess(
            exact_kernel, nvars, bkd, nugget=noise_var,
        )
        exact_gp._fit_internal(X_train, y_train)
        exact_mean = exact_gp.predict(X_test)
        exact_std = exact_gp.predict_std(X_test)

        vgp_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip_vgp = InducingPoints(
            nvars=nvars, num_inducing=N, bkd=bkd,
            inducing_locations=X_train, inducing_bounds=(-10.0, 10.0),
        )
        lik_vgp = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
        ip_vgp.hyp_list().set_all_inactive()
        lik_vgp.hyp_list().set_all_inactive()
        vgp = VariationalGaussianProcess(
            vgp_kernel, nvars, ip_vgp, lik_vgp, bkd,
            mean_function=ZeroMean(bkd), nugget=nugget,
        )
        vgp._fit_internal(X_train, y_train)
        vgp_mean = vgp.predict(X_test)
        vgp_std = vgp.predict_std(X_test)

        layer_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip_layer = InducingPoints(
            nvars=nvars, num_inducing=N, bkd=bkd,
            inducing_locations=X_train, inducing_bounds=(-10.0, 10.0),
        )
        K_uu = layer_kernel(X_train, X_train)
        K_uu_nug = K_uu + bkd.eye(N) * nugget
        L_uu = bkd.cholesky(K_uu_nug)
        m_tilde, L_tilde = titsias_optimal_whitened_q_u(
            K_uu_nug, K_uu, y_train[0, :], bkd.asarray([noise_var]),
            L_uu, bkd,
        )
        vd = GaussianVariationalDistribution(N, bkd, m_tilde, L_tilde)
        lik_layer = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
        layer = DGPLayer(
            layer_kernel, ZeroMean(bkd), ip_layer, vd, bkd,
            likelihood=lik_layer, nugget=nugget,
        )
        ip_layer.hyp_list().set_all_inactive()
        vd.hyp_list().set_all_inactive()
        lik_layer.hyp_list().set_all_inactive()
        layer_mean, layer_var = layer.predict_marginal(X_test)
        layer_std = bkd.sqrt(layer_var)

        bkd.assert_allclose(exact_mean, vgp_mean, rtol=1e-3, atol=1e-4)
        bkd.assert_allclose(exact_mean, layer_mean, rtol=1e-3, atol=1e-4)
        bkd.assert_allclose(exact_std, vgp_std, rtol=0.1, atol=1e-3)
        bkd.assert_allclose(exact_std, layer_std, rtol=0.1, atol=1e-3)

    def test_interpolation_noise_free_limit(self, bkd):
        """With noise_var << nugget, latent std at training inputs ~ sqrt(nugget)."""
        rng = np.random.RandomState(123)
        nvars = 1
        N = 6
        noise_std = 1e-4
        noise_var = noise_std**2
        nugget = 1e-6

        X_train = bkd.array(rng.randn(nvars, N))
        y_train = bkd.array(np.sin(rng.randn(nvars, N)[:1, :]))

        exact_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        exact_gp = ExactGaussianProcess(
            exact_kernel, nvars, bkd, nugget=noise_var,
        )
        exact_gp._fit_internal(X_train, y_train)
        exact_std_train = exact_gp.predict_std(X_train)
        assert float(bkd.to_numpy(bkd.max(exact_std_train))) < 1e-2

        layer_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip_layer = InducingPoints(
            nvars=nvars, num_inducing=N, bkd=bkd,
            inducing_locations=X_train, inducing_bounds=(-10.0, 10.0),
        )
        K_uu = layer_kernel(X_train, X_train)
        K_uu_nug = K_uu + bkd.eye(N) * nugget
        L_uu = bkd.cholesky(K_uu_nug)
        m_tilde, L_tilde = titsias_optimal_whitened_q_u(
            K_uu_nug, K_uu, y_train[0, :], bkd.asarray([noise_var]),
            L_uu, bkd,
        )
        vd = GaussianVariationalDistribution(N, bkd, m_tilde, L_tilde)
        lik_layer = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
        layer = DGPLayer(
            layer_kernel, ZeroMean(bkd), ip_layer, vd, bkd,
            likelihood=lik_layer, nugget=nugget,
        )
        ip_layer.hyp_list().set_all_inactive()
        vd.hyp_list().set_all_inactive()
        lik_layer.hyp_list().set_all_inactive()

        _, layer_var_train = layer.predict_marginal(X_train)
        layer_std_train = bkd.sqrt(layer_var_train)
        assert float(bkd.to_numpy(bkd.max(layer_std_train))) < 1e-2


class TestDGPLayerPriorSampling:
    """Tier 0: prior samples match prior moments at rate 1/sqrt(S)."""

    def test_prior_sample_mean(self, numpy_bkd):
        bkd = numpy_bkd
        rng = np.random.RandomState(99)
        nvars = 1
        M = 5
        N_test = 8
        S = 5000

        layer = _make_layer(bkd, nvars=nvars, num_inducing=M, seed=77)
        h = bkd.array(rng.randn(nvars, N_test))

        mean_pred, _ = layer.predict_marginal(h)

        samples = layer.sample(h, n_samples=S)  # (S, 1, N_test)
        emp_mean = bkd.reshape(
            bkd.asarray([float(bkd.mean(samples[:, 0, j]))
                         for j in range(N_test)]),
            (1, N_test),
        )

        bkd.assert_allclose(emp_mean, mean_pred, atol=0.1)


class TestDGPLayerValidation:
    def test_mismatched_inducing_raises(self, bkd):
        kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=1, bkd=bkd, fixed=True,
        )
        ip = InducingPoints(
            nvars=1, num_inducing=5, bkd=bkd,
            inducing_locations=bkd.array(np.zeros((1, 5))),
            inducing_bounds=(-5.0, 5.0),
        )
        vd = GaussianVariationalDistribution(3, bkd)
        with pytest.raises(ValueError, match="inducing_points has M=5"):
            DGPLayer(kernel, ZeroMean(bkd), ip, vd, bkd)
