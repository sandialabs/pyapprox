"""Tests for DGPELBOLoss (Phase 8)."""

import numpy as np
import pytest

import networkx as nx

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
)
from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
    DGPELBOLoss,
    TorchDGPELBOLoss,
)
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.inducing.titsias import (
    titsias_optimal_whitened_q_u,
)
from pyapprox.surrogates.gaussianprocess.variational import (
    VariationalGaussianProcess,
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
                with_likelihood=True, fixed=True, seed=42):
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


class TestDGPELBOLossSingleLayer:
    def test_elbo_is_finite(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=4, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 8))
        y = bkd.array(np.sin(rng.randn(1, 8)))
        data = {0: (X, y)}

        loss = DGPELBOLoss(dgp, data, n_propagation=1)
        params = dgp.hyp_list().get_active_values()
        result = loss(params)
        assert result.shape == (1, 1)
        assert np.isfinite(float(bkd.to_numpy(result).ravel()[0]))

    def test_elbo_returns_shape(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=2, num_inducing=3, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(2, 5))
        y = bkd.array(rng.randn(1, 5))
        data = {0: (X, y)}

        loss = DGPELBOLoss(dgp, data, n_propagation=3)
        result = loss(dgp.hyp_list().get_active_values())
        assert result.shape == (1, 1)

    def test_no_likelihood_raises(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            with_likelihood=False, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        X = bkd.array(np.random.RandomState(0).randn(1, 5))
        y = bkd.array(np.random.RandomState(1).randn(1, 5))
        data = {0: (X, y)}

        loss = DGPELBOLoss(dgp, data, n_propagation=1)
        with pytest.raises(ValueError, match="no likelihood"):
            loss(dgp.hyp_list().get_active_values())


class TestDGPELBOLossChain:
    def test_chain_elbo_is_finite(self, bkd):
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
        X = bkd.array(rng.randn(1, 6))
        y = bkd.array(np.sin(rng.randn(1, 6)))
        data = {1: (X, y)}

        loss = DGPELBOLoss(dgp, data, n_propagation=3)
        result = loss(dgp.hyp_list().get_active_values())
        assert result.shape == (1, 1)
        assert np.isfinite(float(bkd.to_numpy(result).ravel()[0]))

    def test_kl_contributes_to_elbo(self, bkd):
        """With non-prior q(u), KL > 0 shifts the ELBO."""
        rng = np.random.RandomState(42)
        dag = nx.DiGraph()
        dag.add_node(0)

        M = 4
        m_init = bkd.array(rng.randn(M))
        layer = _make_layer(bkd, nvars=1, num_inducing=M, seed=10)
        layer.variational_dist()._mean_param.set_values(m_init)

        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        X = bkd.array(rng.randn(1, 5))
        y = bkd.array(rng.randn(1, 5))
        data = {0: (X, y)}

        loss = DGPELBOLoss(dgp, data, n_propagation=1)
        neg_elbo = loss(dgp.hyp_list().get_active_values())

        kl = dgp.kl_total()
        kl_np = bkd.to_numpy(bkd.reshape(kl, (1,)))
        assert float(kl_np[0]) > 0.0


class TestDGPELBOLossInterface:
    def test_nvars_and_nqoi(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            fixed=False, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        X = bkd.array(np.random.RandomState(0).randn(1, 5))
        y = bkd.array(np.random.RandomState(1).randn(1, 5))
        data = {0: (X, y)}

        loss = DGPELBOLoss(dgp, data)
        assert loss.nvars() == dgp.hyp_list().nactive_params()
        assert loss.nqoi() == 1

    def test_repr(self, bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        X = bkd.array(np.random.RandomState(0).randn(1, 5))
        y = bkd.array(np.random.RandomState(1).randn(1, 5))
        data = {0: (X, y)}

        loss = DGPELBOLoss(dgp, data, n_propagation=5)
        r = repr(loss)
        assert "DGPELBOLoss" in r
        assert "n_propagation=5" in r


class TestDGPELBOMatchesExactGP:
    """Tier 1: At Titsias optimum with M=N, DGP ELBO = exact GP log ML.

    ExactGP uses nugget as the sole diagonal perturbation (K + nugget*I).
    VGP/DGP separate nugget (jitter in K_uu) from noise_var (in the data
    term). When noise_var >> nugget they agree because the trace
    correction vanishes at M=N. Use noise_var=0.01 >> nugget=1e-8.
    """

    def test_single_layer_elbo_equals_exact_gp_nlml(self, bkd):
        rng = np.random.RandomState(42)
        nvars = 1
        N = 6
        noise_std = 0.1
        noise_var = noise_std**2
        nugget = 1e-8

        X_train = bkd.array(rng.randn(nvars, N))
        y_train = bkd.array(np.sin(rng.randn(1, N)))

        # ExactGP: K + noise_var*I (nugget=noise_var)
        exact_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        exact_gp = ExactGaussianProcess(
            exact_kernel, nvars, bkd, nugget=noise_var,
        )
        exact_gp._fit_internal(X_train, y_train)
        exact_nlml = exact_gp.neg_log_marginal_likelihood()

        # VGP (collapsed Titsias) as intermediate check
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
        vgp_neg_elbo = vgp.neg_log_marginal_likelihood()

        # DGPLayer at Titsias optimum with M=N, inducing=training
        layer_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip = InducingPoints(
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
        lik = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)

        layer = DGPLayer(
            layer_kernel, ZeroMean(bkd), ip, vd, bkd,
            likelihood=lik, nugget=nugget,
        )
        ip.hyp_list().set_all_inactive()
        vd.hyp_list().set_all_inactive()
        lik.hyp_list().set_all_inactive()

        dag = nx.DiGraph()
        dag.add_node(0)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        loss = DGPELBOLoss(dgp, {0: (X_train, y_train)}, n_propagation=1)
        dgp_neg_elbo = loss(dgp.hyp_list().get_active_values())

        # DGP ELBO = VGP ELBO (both use Hensman/Titsias decomposition)
        bkd.assert_allclose(
            dgp_neg_elbo.ravel(),
            bkd.reshape(vgp_neg_elbo, (1,)),
            rtol=1e-4,
        )
        # VGP ELBO = ExactGP NLL at M=N (trace correction vanishes)
        bkd.assert_allclose(
            bkd.reshape(vgp_neg_elbo, (1,)),
            bkd.reshape(exact_nlml, (1,)),
            rtol=1e-4,
        )
        # Transitive: DGP ELBO = ExactGP NLL
        bkd.assert_allclose(
            dgp_neg_elbo.ravel(),
            bkd.reshape(exact_nlml, (1,)),
            rtol=1e-4,
        )


class TestDGPELBOMatchesVGP:
    """Tier 2: At Titsias optimum, DGP ELBO = VGP collapsed ELBO."""

    def test_single_layer_elbo_equals_vgp_elbo(self, bkd):
        rng = np.random.RandomState(42)
        nvars = 1
        N = 8
        M = 5
        noise_std = 0.1
        noise_var = noise_std**2
        nugget = 1e-8

        X_train = bkd.array(rng.randn(nvars, N))
        y_train = bkd.array(np.sin(rng.randn(1, N)))
        Z = bkd.array(rng.randn(nvars, M))

        vgp_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip_vgp = InducingPoints(
            nvars=nvars, num_inducing=M, bkd=bkd,
            inducing_locations=Z, inducing_bounds=(-10.0, 10.0),
        )
        lik_vgp = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
        ip_vgp.hyp_list().set_all_inactive()
        lik_vgp.hyp_list().set_all_inactive()
        vgp = VariationalGaussianProcess(
            vgp_kernel, nvars, ip_vgp, lik_vgp, bkd,
            mean_function=ZeroMean(bkd), nugget=nugget,
        )
        vgp._fit_internal(X_train, y_train)
        vgp_neg_elbo = vgp.neg_log_marginal_likelihood()

        layer_kernel = Matern52Kernel(
            lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        ip_layer = InducingPoints(
            nvars=nvars, num_inducing=M, bkd=bkd,
            inducing_locations=Z, inducing_bounds=(-10.0, 10.0),
        )
        K_uu = layer_kernel(Z, Z)
        K_uf = layer_kernel(Z, X_train)
        K_uu_nug = K_uu + bkd.eye(M) * nugget
        L_uu = bkd.cholesky(K_uu_nug)

        m_tilde, L_tilde = titsias_optimal_whitened_q_u(
            K_uu_nug, K_uf, y_train[0, :], bkd.asarray([noise_var]),
            L_uu, bkd,
        )
        vd = GaussianVariationalDistribution(M, bkd, m_tilde, L_tilde)
        lik_layer = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)

        layer = DGPLayer(
            layer_kernel, ZeroMean(bkd), ip_layer, vd, bkd,
            likelihood=lik_layer, nugget=nugget,
        )
        ip_layer.hyp_list().set_all_inactive()
        vd.hyp_list().set_all_inactive()
        lik_layer.hyp_list().set_all_inactive()

        dag = nx.DiGraph()
        dag.add_node(0)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        loss = DGPELBOLoss(dgp, {0: (X_train, y_train)}, n_propagation=1)
        dgp_neg_elbo = loss(dgp.hyp_list().get_active_values())

        bkd.assert_allclose(
            dgp_neg_elbo.ravel(),
            bkd.reshape(vgp_neg_elbo, (1,)),
            rtol=1e-4,
        )

    def test_elbo_monotone_in_inducing_count(self, bkd):
        """More inducing points => tighter ELBO (lower neg_elbo).

        Monotonicity requires nested inducing sets: each larger set
        contains the smaller set as a prefix. Use training points as
        the pool and take first M for each level.
        """
        rng = np.random.RandomState(42)
        nvars = 1
        N = 10
        noise_std = 0.1
        noise_var = noise_std**2
        nugget = 1e-8

        X_train = bkd.array(rng.randn(nvars, N))
        y_train = bkd.array(np.sin(rng.randn(1, N)))

        neg_elbos = []
        for M in [3, 5, 8, N]:
            Z = X_train[:, :M]

            kernel = Matern52Kernel(
                lenscale=[1.0], lenscale_bounds=(0.1, 10.0),
                nvars=nvars, bkd=bkd, fixed=True,
            )
            ip = InducingPoints(
                nvars=nvars, num_inducing=M, bkd=bkd,
                inducing_locations=Z, inducing_bounds=(-10.0, 10.0),
            )
            K_uu = kernel(Z, Z)
            K_uf = kernel(Z, X_train)
            K_uu_nug = K_uu + bkd.eye(M) * nugget
            L_uu = bkd.cholesky(K_uu_nug)

            m_tilde, L_tilde = titsias_optimal_whitened_q_u(
                K_uu_nug, K_uf, y_train[0, :], bkd.asarray([noise_var]),
                L_uu, bkd,
            )
            vd = GaussianVariationalDistribution(M, bkd, m_tilde, L_tilde)
            lik = GaussianLikelihood(noise_std, (1e-8, 1.0), bkd)
            layer = DGPLayer(
                kernel, ZeroMean(bkd), ip, vd, bkd,
                likelihood=lik, nugget=nugget,
            )
            ip.hyp_list().set_all_inactive()
            vd.hyp_list().set_all_inactive()
            lik.hyp_list().set_all_inactive()

            dag = nx.DiGraph()
            dag.add_node(0)
            prop = LayerPropagator(bkd)
            dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

            loss = DGPELBOLoss(
                dgp, {0: (X_train, y_train)}, n_propagation=1,
            )
            val = float(bkd.to_numpy(loss(
                dgp.hyp_list().get_active_values()
            ).ravel())[0])
            neg_elbos.append(val)

        for i in range(len(neg_elbos) - 1):
            assert neg_elbos[i] >= neg_elbos[i + 1] - 1e-6, (
                f"ELBO not monotone: M values gave neg_elbos {neg_elbos}"
            )


class TestTorchDGPELBOLoss:
    def test_numpy_backend_raises(self, numpy_bkd):
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(numpy_bkd, nvars=1, num_inducing=3, seed=10)
        prop = LayerPropagator(numpy_bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, numpy_bkd)

        X = numpy_bkd.array(np.random.RandomState(0).randn(1, 5))
        y = numpy_bkd.array(np.random.RandomState(1).randn(1, 5))
        data = {0: (X, y)}

        with pytest.raises(TypeError, match="TorchBkd"):
            TorchDGPELBOLoss(dgp, data)

    def test_jacobian_shape(self, torch_bkd):
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            fixed=False, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 5))
        y = bkd.array(np.sin(rng.randn(1, 5)))
        data = {0: (X, y)}

        loss = TorchDGPELBOLoss(dgp, data, n_propagation=1)
        params = dgp.hyp_list().get_active_values()
        jac = loss.jacobian(params)
        assert jac.shape == (1, loss.nvars())

    def test_autograd_jacobian_matches_fd(self, torch_bkd):
        """DerivativeChecker: autograd gradient matches finite differences."""
        bkd = torch_bkd
        dag = nx.DiGraph()
        dag.add_node(0)
        layer = _make_layer(bkd, nvars=1, num_inducing=3,
                            fixed=False, seed=10)
        prop = LayerPropagator(bkd)
        dgp = DeepGaussianProcess(dag, {0: layer}, prop, bkd)

        rng = np.random.RandomState(0)
        X = bkd.array(rng.randn(1, 5))
        y = bkd.array(np.sin(rng.randn(1, 5)))
        data = {0: (X, y)}

        loss = TorchDGPELBOLoss(dgp, data, n_propagation=1)
        n_active = loss.nvars()

        def value_fn(sample):
            return loss(sample[:, 0])

        def jac_fn(sample):
            return loss.jacobian(sample[:, 0])

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=n_active, fun=value_fn,
            jacobian=jac_fn, bkd=bkd,
        )

        checker = DerivativeChecker(wrapper)
        x0 = bkd.reshape(
            dgp.hyp_list().get_active_values(), (n_active, 1),
        )
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        errors = checker.check_derivatives(
            x0, fd_eps=fd_eps, relative=True, verbosity=0,
        )
        jac_error = errors[0]
        assert bkd.all_bool(bkd.isfinite(jac_error))
        jac_ratio = float(checker.error_ratio(jac_error))
        assert jac_ratio < 1e-6, f"Jacobian error ratio: {jac_ratio}"
