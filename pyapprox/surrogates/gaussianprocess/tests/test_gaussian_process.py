import unittest
import copy
import time

import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from functools import partial

from sklearn.gaussian_process.kernels import _approx_fprime

from pyapprox.surrogates.gaussianprocess.kernels import (
    Matern, WhiteKernel, RBF, ConstantKernel
)
from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    GaussianProcess,
    RandomGaussianProcessRealizations, gaussian_process_pointwise_variance,
    integrate_gaussian_process, extract_covariance_kernel, gaussian_tau,
    gaussian_u, compute_varpi, compute_varsigma_sq, variance_of_mean,
    gaussian_P, compute_v_sq, compute_zeta, mean_of_variance, gaussian_nu,
    gaussian_Pi, compute_psi, compute_chi, compute_phi, gaussian_lamda,
    compute_varrho, compute_xi, gaussian_xi_1, compute_varphi,
    marginalize_gaussian_process, compute_expected_sobol_indices,
    IVARSampler, CholeskySampler, GreedyVarianceOfMeanSampler,
    GreedyIntegratedVarianceSampler,
    RBF_integrated_posterior_variance_gradient_wrt_samples, integrate_tau_P,
    RBF_gradient_wrt_samples,  gaussian_grad_P_offdiag_term1,
    gaussian_grad_P_offdiag_term2, gaussian_grad_P_diag_term1,
    gaussian_grad_P_diag_term2, integrate_grad_P,
    RBF_posterior_variance_jacobian_wrt_samples, matern_gradient_wrt_samples,
    AdaptiveGaussianProcess, AdaptiveCholeskyGaussianProcessFixedKernel
)
from pyapprox.expdesign.low_discrepancy_sequences import (
    sobol_sequence, transformed_halton_sequence
)
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.surrogates.approximate import approximate
from pyapprox.analysis.sensitivity_analysis import (
    get_sobol_indices,
    get_main_and_total_effect_indices_from_pce
)
from pyapprox.util.utilities import (
    cartesian_product, outer_product, get_all_sample_combinations,
    approx_jacobian, check_gradients
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.surrogates.orthopoly.quadrature import (
    gauss_jacobi_pts_wts_1D, gauss_hermite_pts_wts_1D
)
from pyapprox.variables.density import tensor_product_pdf


def compute_mean_and_variance_of_gaussian_process(gp, length_scale,
                                                  train_samples,
                                                  A_inv, kernel_var,
                                                  train_vals, quad_samples,
                                                  quad_weights):
    # just use for testing purposes
    # computing variance_of_variance requires splitting up terms
    # like done in code so no point computing this quantity as it cannot
    # test if the splitting procedure is correct.
    nvars = quad_samples.shape[0]
    gp_vals, gp_std = gp(quad_samples, return_std=True)
    gp_vals = gp_vals[:, 0]
    mean_of_mean = gp_vals.dot(quad_weights)
    quad_samples_WWXX = get_all_sample_combinations(
        quad_samples, quad_samples)
    quad_weights_WWXX = outer_product([quad_weights]*2)

    L = np.linalg.cholesky(A_inv)

    ww, xx = quad_samples_WWXX[:nvars], quad_samples_WWXX[nvars:]
    dists_ww_xx = np.sum((ww.T/length_scale-xx.T/length_scale)**2, axis=1)
    dists_ww_tt = cdist(train_samples.T/length_scale, ww.T/length_scale,
                        metric='sqeuclidean')
    dists_xx_tt = cdist(train_samples.T/length_scale, xx.T/length_scale,
                        metric='sqeuclidean')
    t_ww = np.exp(-.5*dists_ww_tt)
    t_xx = np.exp(-.5*dists_xx_tt)
    prior_cov_ww_xx = kernel_var*np.exp(-.5*dists_ww_xx)
    post_cov_ww_xx = prior_cov_ww_xx - kernel_var*np.sum(
        t_ww.T.dot(L)*t_xx.T.dot(L), axis=1)

    var_of_mean = post_cov_ww_xx.dot(quad_weights_WWXX)

    mean_of_var = (gp_vals**2+gp_std**2).dot(quad_weights) - (
        var_of_mean+mean_of_mean**2)

    return mean_of_mean, var_of_mean, mean_of_var


def compute_intermediate_quantities_with_monte_carlo(mu_scalar, sigma_scalar,
                                                     length_scale,
                                                     train_samples,
                                                     A_inv, kernel_var,
                                                     train_vals):
    nsamples_mc = 20000
    nvars = length_scale.shape[0]
    xx = np.random.normal(mu_scalar, sigma_scalar, (nvars, nsamples_mc))
    yy = np.random.normal(mu_scalar, sigma_scalar, (nvars, nsamples_mc))
    zz = np.random.normal(mu_scalar, sigma_scalar, (nvars, nsamples_mc))
    dists_xx_tt = cdist(train_samples.T/length_scale, xx.T/length_scale,
                        metric='sqeuclidean')
    dists_yy_tt = cdist(train_samples.T/length_scale, yy.T/length_scale,
                        metric='sqeuclidean')
    dists_zz_tt = cdist(train_samples.T/length_scale, zz.T/length_scale,
                        metric='sqeuclidean')
    dists_xx_yy = np.sum((xx.T/length_scale-yy.T/length_scale)**2, axis=1)
    dists_xx_zz = np.sum((xx.T/length_scale-zz.T/length_scale)**2, axis=1)
    t_xx = np.exp(-.5*dists_xx_tt)
    t_yy = np.exp(-.5*dists_yy_tt)
    t_zz = np.exp(-.5*dists_zz_tt)

    mean_gp_xx = t_xx.T.dot(A_inv.dot(train_vals))*kernel_var
    mean_gp_yy = t_yy.T.dot(A_inv.dot(train_vals))*kernel_var
    prior_cov_xx_xx = np.ones((xx.shape[1]))
    L = np.linalg.cholesky(A_inv)
    post_cov_xx_xx = prior_cov_xx_xx - np.sum(
        (t_xx.T.dot(L))**2, axis=1)
    prior_cov_xx_yy = np.exp(-.5*dists_xx_yy)
    post_cov_xx_yy = prior_cov_xx_yy - np.sum(
        t_xx.T.dot(L)*t_yy.T.dot(L), axis=1)
    # assert np.allclose(np.sum(
    #    t_xx.T.dot(L)*t_yy.T.dot(L),axis=1),np.diag(t_xx.T.dot(A_inv).dot(t_yy)))
    prior_cov_xx_zz = np.exp(-.5*dists_xx_zz)
    post_cov_xx_zz = prior_cov_xx_zz - np.sum(
        t_xx.T.dot(L)*t_zz.T.dot(L), axis=1)

    eta_mc = mean_gp_xx.mean()
    varrho_mc = (mean_gp_xx*post_cov_xx_yy).mean()
    phi_mc = (mean_gp_xx*mean_gp_yy*post_cov_xx_yy).mean()
    CC_mc = (post_cov_xx_yy*post_cov_xx_zz).mean()
    chi_mc = (post_cov_xx_yy**2).mean()
    M_sq_mc = (mean_gp_xx**2).mean()
    v_sq_mc = (post_cov_xx_xx).mean()
    varsigma_sq_mc = (post_cov_xx_yy).mean()
    P_mc = (t_xx.dot(t_xx.T))/xx.shape[1]
    lamda_mc = (prior_cov_xx_yy*t_yy).mean(axis=1)
    CC1_mc = (prior_cov_xx_yy*prior_cov_xx_zz).mean()
    Pi_mc = np.zeros((train_samples.shape[1], train_samples.shape[1]))
    for ii in range(train_samples.shape[1]):
        for jj in range(ii, train_samples.shape[1]):
            Pi_mc[ii, jj] = (prior_cov_xx_yy*t_xx[ii, :]*t_yy[jj, :]).mean()
            Pi_mc[jj, ii] = Pi_mc[ii, jj]

    return eta_mc, varrho_mc, phi_mc, CC_mc, chi_mc, M_sq_mc, v_sq_mc, \
        varsigma_sq_mc, P_mc, lamda_mc, CC1_mc, Pi_mc


def verify_quantities(reference_quantities, quantities, tols):
    assert len(reference_quantities) == len(quantities) == len(tols)
    ii = 0
    for q_ref, q, tol in zip(reference_quantities, quantities, tols):
        assert np.allclose(q_ref, q, rtol=tol), (ii, q_ref, q, tol)
        ii += 1


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_gaussian_process_random_realization_interpolation(self):
        nvars = 1
        lb, ub = 0, 1
        ntrain_samples = 5
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples)]*nvars)
        train_vals = func(train_samples)

        kernel = Matern(0.4, length_scale_bounds='fixed', nu=np.inf)
        kernel = ConstantKernel(
            constant_value=2., constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(kernel)

        gp.fit(train_samples, train_vals)

        ngp_realizations, ninterpolation_samples = 2, 17
        nvalidation_samples = 20
        rand_noise = np.random.normal(
            0, 1, (ninterpolation_samples+nvalidation_samples,
                   ngp_realizations))
        candidate_samples = np.random.uniform(lb, ub, (nvars, 1000))
        gp_realizations = RandomGaussianProcessRealizations(gp, alpha=1e-14)
        gp_realizations.fit(
            candidate_samples, rand_noise, ninterpolation_samples,
            nvalidation_samples)
        interp_random_gp_vals = gp_realizations(
            gp.map_from_canonical(
                gp_realizations.selected_canonical_samples))
        # print(np.absolute(gp_realizations.train_vals-interp_random_gp_vals))
        # adding alpha means we wont interpolate the data exactly
        assert np.allclose(
            gp_realizations.train_vals, interp_random_gp_vals,
            rtol=1e-6, atol=1e-6)
        samples = gp.map_from_canonical(np.hstack((
            gp_realizations.selected_canonical_samples,
            gp_realizations.canonical_validation_samples)))
        random_gp_vals = gp.predict_random_realization(
            samples, rand_noise[: samples.shape[1]])
        print(interp_random_gp_vals -
              random_gp_vals[:gp_realizations.selected_canonical_samples.shape[1]])
        assert np.allclose(
            interp_random_gp_vals,
            random_gp_vals[:gp_realizations.selected_canonical_samples.shape[1]],
            atol=5e-8)

    def test_gaussian_process_pointwise_variance(self):
        nvars = 1
        lb, ub = 0, 1
        ntrain_samples = 5
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples)]*nvars)
        train_vals = func(train_samples)

        kernel = Matern(0.4, length_scale_bounds='fixed', nu=np.inf)
        kernel = ConstantKernel(
            constant_value=2., constant_value_bounds='fixed')*kernel
        kernel += WhiteKernel(noise_level=1e-5, noise_level_bounds='fixed')
        gp = GaussianProcess(kernel)

        gp.fit(train_samples, train_vals)

        samples = np.random.uniform(0, 1, (nvars, 100))
        pred_vals, stdev1 = gp(samples, return_std=True)

        variance2 = gaussian_process_pointwise_variance(
            kernel, samples, train_samples)

        assert np.allclose(stdev1**2, variance2)

    def test_integrate_gaussian_process_gaussian(self):

        nvars = 2
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        mu_scalar, sigma_scalar = 3, 1
        # mu_scalar, sigma_scalar = 0, 1

        univariate_variables = [stats.norm(mu_scalar, sigma_scalar)]*nvars
        variable = IndependentMarginalsVariable(
            univariate_variables)

        lb, ub = univariate_variables[0].interval(0.99999)

        ntrain_samples = 5
        # ntrain_samples = 20

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples)]*nvars)
        train_vals = func(train_samples)

        nu = np.inf
        nvars = train_samples.shape[0]
        length_scale = np.array([1]*nvars)
        kernel = Matern(length_scale, length_scale_bounds=(1e-2, 10), nu=nu)
        # fix kernel variance
        kernel = ConstantKernel(
            constant_value=2., constant_value_bounds='fixed')*kernel
        # optimize kernel variance
        # kernel = ConstantKernel(
        #    constant_value=3,constant_value_bounds=(0.1, 10))*kernel
        # optimize gp noise
        # kernel += WhiteKernel(noise_level_bounds=(1e-8, 1))
        # fix gp noise
        # kernel += WhiteKernel(noise_level=1e-5, noise_level_bounds='fixed')
        # white kernel K(x_i,x_j) is only nonzeros when x_i=x_j, i.e.
        # it is not used when calling gp.predict
        gp = GaussianProcess(kernel, n_restarts_optimizer=10, alpha=1e-8)
        gp.fit(train_samples, train_vals)
        # print(gp.kernel_)

        # xx=np.linspace(lb,ub,101)
        # plt.plot(xx,func(xx[np.newaxis,:]))
        # gp_mean,gp_std = gp(xx[np.newaxis,:],return_std=True)
        # gp_mean = gp_mean[:,0]
        # plt.plot(xx,gp_mean)
        # plt.plot(train_samples[0,:],train_vals[:,0],'o')
        # plt.fill_between(xx,gp_mean-2*gp_std,gp_mean+2*gp_std,alpha=0.5)
        # plt.show()

        # import time
        # t0 = time.time()
        (expected_random_mean, variance_random_mean, expected_random_var,
         variance_random_var, intermediate_quantities) =\
             integrate_gaussian_process(gp, variable, return_full=True,
                                        nquad_samples=100)
        # print('time', time.time()-t0)

        # mu and sigma should match variable
        kernel_types = [Matern]
        kernel = extract_covariance_kernel(gp.kernel_, kernel_types)
        length_scale = np.atleast_1d(kernel.length_scale)
        constant_kernel = extract_covariance_kernel(
            gp.kernel_, [ConstantKernel])
        if constant_kernel is not None:
            kernel_var = constant_kernel.constant_value
        else:
            kernel_var = 1

        Kinv_y = gp.alpha_
        mu = np.array([mu_scalar]*nvars)[:, np.newaxis]
        sigma = np.array([sigma_scalar]*nvars)[:, np.newaxis]
        # Notes sq exp kernel is exp(-dists/delta). Sklearn RBF kernel is
        # exp(-.5*dists/L**2)
        delta = 2*length_scale[:, np.newaxis]**2

        # Kinv_y is inv(kernel_var*A).dot(y). Thus multiply by kernel_var to
        # get formula in notes
        Ainv_y = Kinv_y*kernel_var
        L_inv = solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]), lower=False)
        K_inv = L_inv.dot(L_inv.T)
        # K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get A_inv
        A_inv = K_inv*kernel_var

        # Verify quantities used to compute mean and variance of mean of GP
        # This is redundant know but helped to isolate incorrect terms
        # when initialing writing tests
        tau_true = gaussian_tau(train_samples, delta, mu, sigma)
        u_true = gaussian_u(delta, sigma)
        varpi_true = compute_varpi(tau_true, A_inv)
        varsigma_sq_true = compute_varsigma_sq(u_true, varpi_true)
        verify_quantities(
            [tau_true, u_true, varpi_true, varsigma_sq_true],
            intermediate_quantities[:4], [1e-8]*4)

        # Verify mean and variance of mean of GP
        true_expected_random_mean = tau_true.dot(Ainv_y)
        true_variance_random_mean = variance_of_mean(
            kernel_var, varsigma_sq_true)
        assert np.allclose(
            true_expected_random_mean, expected_random_mean)
        # print(true_variance_random_mean,variance_random_mean)
        assert np.allclose(
            true_variance_random_mean, variance_random_mean)

        # Verify quantities used to compute mean of variance of GP
        # This is redundant know but helped to isolate incorrect terms
        # when initialing writing tests
        P_true = gaussian_P(train_samples, delta, mu, sigma)
        v_sq_true = compute_v_sq(A_inv, P_true)
        zeta_true = compute_zeta(train_vals, A_inv, P_true)
        verify_quantities(
            [P_true, v_sq_true], intermediate_quantities[4:6], [1e-8]*2)

        true_expected_random_var = mean_of_variance(
            zeta_true, v_sq_true, kernel_var, true_expected_random_mean,
            true_variance_random_mean)
        assert np.allclose(true_expected_random_var, expected_random_var)

        # Verify quantities used to compute variance of variance of GP
        # This is redundant know but helped to isolate incorrect terms
        # when initialing writing tests
        nu_true = gaussian_nu(delta, sigma)
        varphi_true = compute_varphi(A_inv, P_true)
        Pi_true = gaussian_Pi(train_samples, delta, mu, sigma)
        psi_true = compute_psi(A_inv, Pi_true)
        chi_true = compute_chi(nu_true, varphi_true, psi_true)
        phi_true = compute_phi(train_vals, A_inv, Pi_true, P_true)
        lamda_true = gaussian_lamda(train_samples, delta, mu, sigma)
        varrho_true = compute_varrho(
            lamda_true, A_inv, train_vals, P_true, tau_true)
        xi_1_true = gaussian_xi_1(delta, sigma)
        xi_true = compute_xi(xi_1_true, lamda_true, tau_true, P_true, A_inv)
        verify_quantities(
            [zeta_true, nu_true, varphi_true, Pi_true, psi_true, chi_true,
             phi_true, lamda_true, varrho_true, xi_1_true, xi_true],
            intermediate_quantities[6:17], [1e-8]*11)

        if nvars == 1:
            nxx = 100
        else:
            nxx = 15
            xx, ww = gauss_hermite_pts_wts_1D(nxx)
            xx = xx*sigma_scalar + mu_scalar
            quad_points = cartesian_product([xx]*nvars)
            quad_weights = outer_product([ww]*nvars)
            mean_of_mean_quad, variance_of_mean_quad, mean_of_variance_quad = \
                compute_mean_and_variance_of_gaussian_process(
                    gp, length_scale, train_samples, A_inv, kernel_var,
                    train_vals, quad_points, quad_weights)

        assert np.allclose(mean_of_mean_quad, expected_random_mean)
        assert np.allclose(variance_of_mean_quad, variance_random_mean)
        assert np.allclose(mean_of_variance_quad, expected_random_var)

        nsamples, final_tol = int(1e5), 1.3e-2
        # Below nsamples is killed by github actions due to memory usage
        # nsamples, final_tol = int(1e6), 5e-3
        random_means, random_variances = [], []
        xx, ww = gauss_hermite_pts_wts_1D(nxx)
        xx = xx*sigma_scalar + mu_scalar
        quad_points = cartesian_product([xx]*nvars)
        quad_weights = outer_product([ww]*nvars)
        vals = gp.predict_random_realization(quad_points, nsamples)
        # when quad_points.shape is less than 1000 just compute
        # realizations exactly
        # vals = evaluate_random_gaussian_process_realizations_via_interpolation(
        #     gp, quad_points, nsamples)
        I, I2 = vals.T.dot(quad_weights), (vals.T**2).dot(quad_weights)
        random_means = I
        random_variances = I2-I**2
        # random_I2sq = I2**2
        # random_I2Isq = I2*I**2
        # random_I4 = I**4

        # print('MC expected random mean', np.mean(random_means))
        # print('MC variance random mean', np.var(random_means))
        # print('MC expected random variance', np.mean(random_variances))
        # print('MC variance random variance', np.var(random_variances))
        # print('expected random mean', expected_random_mean)
        # print('variance random mean', variance_random_mean)
        # print('expected random variance', expected_random_var)
        # print('variance random variance', variance_random_var)
        assert np.allclose(
            np.mean(random_means), expected_random_mean, rtol=1e-3)
        assert np.allclose(
            np.var(random_means), variance_random_mean, rtol=5e-3)
        assert np.allclose(
            expected_random_var, np.mean(random_variances), rtol=1e-3)
        # print(variance_random_var-np.var(random_variances),
        #     np.var(random_variances))
        assert np.allclose(
            variance_random_var, np.var(random_variances), rtol=final_tol)

    def test_integrate_gaussian_process_uniform(self):
        nvars = 1
        constant = 1e3
        nugget = 0
        normalize_y = True
        def func(x): return constant*np.sum((2*x-.5)**2, axis=0)[:, np.newaxis]

        univariate_variables = [stats.uniform(0, 1)]
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)

        ntrain_samples = 7
        train_samples = (np.cos(
            np.linspace(0, np.pi, ntrain_samples))[np.newaxis, :]+1)/2
        train_vals = func(train_samples)

        nu = np.inf
        kernel = Matern(length_scale_bounds=(1e-2, 10), nu=nu)
        # kernel needs to be multiplied by a constant kernel
        if not normalize_y:
            kernel = ConstantKernel(
                constant_value=constant, constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(
            kernel, n_restarts_optimizer=5, normalize_y=normalize_y,
            alpha=nugget)

        # This code block shows that for same rand_noise different samples xx
        # will produce different realizations
        # xx = np.linspace(0, 1, 101)
        # rand_noise = np.random.normal(0, 1, (xx.shape[0], 1))
        # yy = gp.predict_random_realization(xx[None, :], rand_noise)
        # plt.plot(xx, yy)
        # xx = np.linspace(0, 1, 97)
        # rand_noise = np.random.normal(0, 1, (xx.shape[0], 1))
        # yy = gp.predict_random_realization(xx[None, :], rand_noise)
        # plt.plot(xx, yy)
        # plt.show()

        gp.set_variable_transformation(var_trans)
        gp.fit(train_samples, train_vals)
        # avoid training data when checking variance
        zz = np.linspace(0.01, 0.99, 100)
        mean, std = gp(zz[None, :], return_std=True)
        vals = gp.predict_random_realization(
            zz[None, :], 100000,
            truncated_svd={'nsingular_vals': 10, 'tol': 1e-8})
        assert np.allclose(vals.std(axis=1), std, rtol=5e-3)

        expected_random_mean, variance_random_mean, expected_random_var, \
            variance_random_var = integrate_gaussian_process(gp, variable)

        true_mean = constant*7/12
        true_var = constant**2*61/80-true_mean**2

        print('True mean', true_mean)
        print('Expected random mean', expected_random_mean)
        print('Variance random mean', variance_random_mean)
        std_random_mean = np.sqrt(variance_random_mean)
        print('Stdev random mean', std_random_mean)
        print('Expected random mean +/- 3 stdev',
              [expected_random_mean-3*std_random_mean,
               expected_random_mean+3*std_random_mean])
        print('True var', true_var)
        print('Expected random var', expected_random_var)
        print('Variance random var', variance_random_var)

        nsamples = int(2e5)
        random_means = []
        xx, ww = gauss_jacobi_pts_wts_1D(50, 0, 0)
        xx = (xx+1)/2
        quad_points = cartesian_product([xx]*nvars)
        quad_weights = outer_product([ww]*nvars)
        vals = gp.predict_random_realization(quad_points, nsamples)
        random_means = vals.T.dot(quad_weights)
        random_variances = ((vals)**2).T.dot(quad_weights)-random_means**2

        print('MC expected random mean', np.mean(random_means))
        print('MC variance random mean', np.var(random_means))
        print('MC expected random var', np.mean(random_variances))
        print('MC variance random var', "{:e}".format(
            np.var(random_variances)))

        assert np.allclose(true_mean, expected_random_mean, rtol=1e-3)
        assert np.allclose(expected_random_var, true_var, rtol=1e-3)
        assert np.allclose(
            np.mean(random_means), expected_random_mean, rtol=1e-3)
        assert np.allclose(
            np.var(random_means), variance_random_mean, rtol=3e-3)
        assert np.allclose(
            np.mean(random_variances), expected_random_var, rtol=1e-3)
        assert np.allclose(
            np.var(random_variances), variance_random_var, rtol=3e-3)

        # xx=np.linspace(-1,1,101)
        # plt.plot(xx,func(xx[np.newaxis,:]))
        # gp_mean,gp_std = gp(xx[np.newaxis,:],return_std=True)
        # gp_mean = gp_mean[:,0]
        # plt.plot(xx,gp_mean)
        # plt.plot(train_samples[0,:],train_vals[:,0],'o')
        # plt.fill_between(xx,gp_mean-2*gp_std,gp_mean+2*gp_std,alpha=0.5)
        # vals = gp.predict_random_realization(xx[np.newaxis,:])
        # plt.plot(xx,vals)
        # plt.show()

    def test_integrate_gaussian_process_uniform_mixed_bounds(self):
        nvars = 2
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        ntrain_samples = 25
        train_samples = np.cos(
            np.random.uniform(0, np.pi, (nvars, ntrain_samples)))
        train_samples[1, :] = (train_samples[1, :]+1)/2
        train_vals = func(train_samples)

        univariate_variables = [stats.uniform(-1, 2), stats.uniform(0, 1)]
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)

        nu = np.inf
        length_scale = np.ones(nvars)
        kernel = Matern(length_scale, length_scale_bounds=(1e-2, 10), nu=nu)
        gp = GaussianProcess(kernel, n_restarts_optimizer=1)
        gp.set_variable_transformation(var_trans)
        gp.fit(train_samples, train_vals)

        expected_random_mean, variance_random_mean, expected_random_var, \
            variance_random_var = integrate_gaussian_process(gp, variable)

        true_mean = 2/3
        true_var = 28/45-true_mean**2

        print('True mean', true_mean)
        print('Expected random mean', expected_random_mean)
        std_random_mean = np.sqrt(variance_random_mean)
        print('Variance random mean', variance_random_mean)
        print('Stdev random mean', std_random_mean)
        print('Expected random mean +/- 3 stdev',
              [expected_random_mean-3*std_random_mean,
               expected_random_mean+3*std_random_mean])
        assert np.allclose(true_mean, expected_random_mean, rtol=5e-2)

        print('True var', true_var)
        print('Expected random var', expected_random_var)
        assert np.allclose(expected_random_var, true_var, rtol=1e-2)

        nsamples = int(1e4)
        random_means = []
        xx, ww = gauss_jacobi_pts_wts_1D(20, 0, 0)
        quad_points = cartesian_product([xx, (xx+1)/2])
        quad_weights = outer_product([ww]*nvars)
        vals = gp.predict_random_realization(quad_points, nsamples)
        random_means = vals.T.dot(quad_weights)

        print('MC expected random mean', np.mean(random_means))
        print('MC variance random mean', np.var(random_means))
        assert np.allclose(
            np.mean(random_means), expected_random_mean, rtol=1e-5)
        assert np.allclose(
            np.var(random_means), variance_random_mean, rtol=1e-5)

    def test_marginalize_gaussian_process_uniform(self):
        nvars = 2
        a = np.array([1, 0.25])
        # a = np.array([1, 1])

        def func(x):
            return np.sum(a[:, None]*(2*x-1)**2, axis=0)[:, np.newaxis]

        ntrain_samples = 30
        # train_samples = np.random.uniform(0, 1, (nvars, ntrain_samples))
        train_samples = sobol_sequence(nvars, ntrain_samples)
        train_vals = func(train_samples)

        univariate_variables = [stats.uniform(0, 1)]*nvars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)

        nu = np.inf
        kernel_var = 2.
        length_scale = np.array([1]*nvars)
        kernel = Matern(length_scale, length_scale_bounds=(1e-2, 10), nu=nu)
        kernel = ConstantKernel(
            constant_value=kernel_var, constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(kernel, n_restarts_optimizer=1, alpha=1e-8,
                             normalize_y=True)
        gp.set_variable_transformation(var_trans)
        gp.fit(train_samples, train_vals)

        validation_samples = np.random.uniform(0, 1, (nvars, 100))
        validation_vals = func(validation_samples)
        error = np.linalg.norm(validation_vals-gp(validation_samples)) / \
            np.linalg.norm(validation_vals)
        print(error)
        # only satisfied with more training data. But more training data
        # makes it hard to test computation of marginal gp mean and variances
        # assert error < 1e-3

        # true_mean = 1/3*a.sum()
        expected_random_mean, variance_random_mean, expected_random_var, \
            variance_random_var, intermediate_quantities = \
            integrate_gaussian_process(gp, variable, return_full=True)
        # only satisfied with more training data. But more training data
        # makes it hard to test computation of marginal gp mean and variances
        # print(expected_random_mean - true_mean)
        # assert abs(expected_random_mean - true_mean)/true_mean < 1e-3

        center = False
        marginalized_gps = marginalize_gaussian_process(
            gp, variable, center=center)

        expected_random_mean = integrate_gaussian_process(gp, variable)[0]
        for ii in range(nvars):
            gp_ii = marginalized_gps[ii]
            if center is True:
                assert np.allclose(gp_ii.mean, expected_random_mean)
            # variable_ii = IndependentMarginalsVariable(
            #     [univariate_variables[ii]])

            # kernel must be evaluated in canonical space
            # gp must be evaluated in user space
            xx = np.linspace(0, 1, 11)
            xx, ww = gauss_jacobi_pts_wts_1D(20, 0, 0)
            xx = (xx+1)/2
            A_inv = np.linalg.inv(gp.L_.dot(gp.L_.T))
            K_pred = gp_ii.kernel_(2*xx[:, None]-1, gp_ii.X_train_)
            variance_ii = kernel_var*gp_ii.kernel_.k2.u-np.diag(
                K_pred.dot(A_inv).dot(K_pred.T))
            # print(variance_ii)
            assert variance_ii.min() > -1e-7
            # stdev_ii = np.sqrt(np.maximum(0, variance_ii))
            vals_ii, std_ii = gp_ii(xx[np.newaxis, :], return_std=True)
            vals_ii += gp_ii.mean
            # print(stdev_ii-std_ii)
            # assert np.allclose(stdev_ii, std_ii, atol=5e-8)

            xx_quad, ww_quad = gauss_jacobi_pts_wts_1D(10, 0, 0)
            xx_quad = (xx_quad+1)/2

            # gp_ii_mean = gp_ii(xx_quad[None, :])[:, 0].dot(ww_quad)+gp_ii.mean

            xx_2d = cartesian_product([xx_quad, xx])
            if ii == 0:
                xx_2d = np.vstack([xx_2d[1], xx_2d[0]])
            nreps = 10000
            marginalized_vals = []
            all_vals = gp.predict_random_realization(xx_2d, nreps)
            for jj in range(nreps):
                vals_flat = all_vals[:, jj]
                vals = vals_flat.reshape(
                    xx_quad.shape[0], xx.shape[0], order='F')
                # check we reshaped correctly
                assert np.allclose(
                    vals[:xx_quad.shape[0], 0], vals_flat[:xx_quad.shape[0]])
                assert np.allclose(
                    vals[:xx_quad.shape[0], 1],
                    vals_flat[xx_quad.shape[0]:xx_quad.shape[0]*2])
                marginalized_vals.append(vals.T.dot(ww_quad))

            mc_mean_ii = np.mean(np.asarray(marginalized_vals), axis=0)
            mc_stdev_ii = np.std(np.asarray(marginalized_vals), axis=0)
            print(mc_mean_ii-vals_ii[:, 0])
            print(mc_stdev_ii-std_ii)
            assert np.allclose(mc_mean_ii, vals_ii[:, 0], atol=2e-3)
            assert np.allclose(mc_stdev_ii, std_ii, atol=2e-3)
            # plt.plot(xx, vals_ii[:, 0], label='marginalized GP')
            # plt.plot(xx, mc_mean_ii, ':', label='MC marginalized GP')
            # plt.fill_between(
            #     xx, vals_ii[:, 0]-2*std_ii, vals_ii[:, 0]+2*std_ii,
            #     color='gray', alpha=0.5)
            # vals = gp(np.vstack((xx[np.newaxis, :], xx[np.newaxis, :]*0)))
            # plt.plot(xx, vals, '--', label='2D cross section')
            # plt.plot(train_samples[ii, :], train_vals, 'bo',
            #          label='Projected train data')
            # plt.legend()
            # plt.show()

        from pyapprox.surrogates.polychaos.gpc import \
            marginalize_polynomial_chaos_expansion
        pce = approximate(
            train_samples, train_vals, 'polynomial_chaos',
            {'basis_type': 'hyperbolic_cross', 'variable': variable,
             'options': {'max_degree': 4}}).approx
        print(pce.coefficients[0, 0])
        marginalized_pces = []
        for ii in range(nvars):
            inactive_idx = np.hstack((np.arange(ii), np.arange(ii+1, nvars)))
            marginalized_pces.append(
                marginalize_polynomial_chaos_expansion(
                    pce, inactive_idx, center=center))
            # xx = np.linspace(0, 1, 101)
            # plt.plot(xx, marginalized_pces[ii](xx[None, :]), '-')
            # plt.plot(xx, marginalized_gps[ii](xx[None, :]), '--')
            # plt.show()
            assert np.allclose(
                marginalized_pces[ii](xx[None, :]),
                marginalized_gps[ii](xx[None, :]), rtol=1e-2, atol=1.3e-2)

    def test_compute_sobol_indices_gaussian_process_uniform_1d(self):
        nvars = 1
        univariate_variables = [stats.uniform(0, 1)]*nvars
        variable = IndependentMarginalsVariable(
            univariate_variables)

        degree = 2
        def func(xx):
            return np.sum(xx**degree, axis=0)[:, None]

        ntrain_samples = 21
        # train_samples = np.random.uniform(0, 1, (nvars, ntrain_samples))
        # from pyapprox.expdesign.low_discrepancy_sequences import sobol_sequence
        # train_samples = sobol_sequence(
        #     nvars, ntrain_samples, variable=variable, start_index=1)
        train_samples = np.linspace(0, 1, ntrain_samples)[None, :]
        train_vals = func(train_samples)

        # var_trans = AffineTransform(variable)

        nu = np.inf
        kernel_var = 1.
        length_scale = np.array([1]*nvars)
        kernel = Matern(length_scale, length_scale_bounds=(1e-2, 10), nu=nu)
        # kernel = Matern(length_scale, length_scale_bounds='fixed', nu=nu)
        # condition number of kernel at training data significantly effects
        # accuracy. Making alpha larger can help
        kernel = ConstantKernel(
            constant_value=kernel_var, constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(kernel, n_restarts_optimizer=1, alpha=1e-6)
        # gp.set_variable_transformation(var_trans)
        gp.fit(train_samples, train_vals)
        print(gp.kernel_)

        # import matplotlib.pyplot as plt
        xx = np.linspace(0, 1, 21)
        print(gp(xx[None, :], return_std=True)[1], 's')
        # plt.plot(xx, gp(xx[None, :]))
        # plt.plot(train_samples[0, :], train_vals[:, 0], 'o')
        # plt.plot(xx, func(xx[None, :]), '--')
        # plt.show()

        validation_samples = np.random.uniform(0, 1, (nvars, ntrain_samples))
        validation_vals = func(validation_samples)
        error = np.linalg.norm(validation_vals - gp(validation_samples)) / \
            np.linalg.norm(validation_vals)
        print(error)

        nquad_samples = 10
        expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var = integrate_gaussian_process(
                gp, variable, nquad_samples=nquad_samples)
        print('v', expected_random_var, expected_random_mean)

        true_mean = 1/(degree+1)
        unnormalized_main_effect_0 = 1/(2*degree+1) -\
            true_mean**2
        true_unnormalized_main_effects = np.array(
            [[unnormalized_main_effect_0]]).T
        true_var = np.sum(true_unnormalized_main_effects)
        print(expected_random_var[0, 0], true_var)
        assert np.allclose(expected_random_var, true_var, atol=1e-4)

        order = 1
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, np.where(
            interaction_terms.max(axis=0) == 1)[0]]

        nquad_samples = 100
        sobol_indices, total_effects, mean, variance = \
            compute_expected_sobol_indices(
                gp, variable, interaction_terms, nquad_samples=nquad_samples)

        print(variance, true_var)
        true_unnormalized_sobol_indices = true_unnormalized_main_effects
        true_sobol_indices = true_unnormalized_sobol_indices/true_var
        assert np.allclose(true_sobol_indices, sobol_indices, atol=1e-4)

    def test_compute_sobol_indices_gaussian_process_uniform_2d(self):
        nvars = 2
        a = np.array([1, 0.25])
        # a = np.array([1, 1])

        def func(x):
            return np.sum(a[:, None]*(2*x-1)**2, axis=0)[:, np.newaxis]

        ntrain_samples = 100
        # train_samples = np.random.uniform(0, 1, (nvars, ntrain_samples))
        from pyapprox.expdesign.low_discrepancy_sequences import sobol_sequence
        train_samples = sobol_sequence(nvars, ntrain_samples)
        train_vals = func(train_samples)

        univariate_variables = [stats.uniform(0, 1)]*nvars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        # var_trans = AffineTransform(variable)

        nu = np.inf
        kernel_var = 1.
        length_scale = np.array([1]*nvars)
        kernel = Matern(length_scale, length_scale_bounds=(1e-2, 10), nu=nu)
        kernel = ConstantKernel(
            constant_value=kernel_var, constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(kernel, n_restarts_optimizer=1, alpha=1e-6)
        # gp.set_variable_transformation(var_trans)
        gp.fit(train_samples, train_vals)

        validation_samples = np.random.uniform(0, 1, (nvars, ntrain_samples))
        validation_vals = func(validation_samples)
        error = np.linalg.norm(validation_vals - gp(validation_samples)) / \
            np.linalg.norm(validation_vals)
        print(error)

        nquad_samples = 50
        expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var = integrate_gaussian_process(
                gp, variable, nquad_samples=nquad_samples)
        # print('v',variance_random_mean, expected_random_var)

        true_mean = 1/3*a.sum()
        unnormalized_main_effect_0 = a[0]**2/5+(2*a[0]*a[1])/9+a[1]**2/9 -\
            true_mean**2
        unnormalized_main_effect_1 = a[1]**2/5+(2*a[0]*a[1])/9+a[0]**2/9 -\
            true_mean**2
        true_unnormalized_main_effects = np.array(
            [[unnormalized_main_effect_0, unnormalized_main_effect_1]]).T
        true_var = np.sum(true_unnormalized_main_effects)

        order = 2
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, np.where(
            interaction_terms.max(axis=0) == 1)[0]]

        nquad_samples = 100
        sobol_indices, total_effects, mean, variance = \
            compute_expected_sobol_indices(
                gp, variable, interaction_terms, nquad_samples=nquad_samples)
        true_unnormalized_sobol_indices = np.vstack((
            true_unnormalized_main_effects, [[0]]))
        true_sobol_indices = true_unnormalized_sobol_indices/true_var
        print("GP sobol indices debug", np.absolute(
            sobol_indices-true_sobol_indices) - 4e-5-2e-5*true_sobol_indices)
        print(sobol_indices)
        print(true_sobol_indices)
        assert np.allclose(
            sobol_indices, true_sobol_indices, rtol=2e-5, atol=4e-5)

    def test_compute_sobol_indices_gaussian_process_uniform_3d(self):
        nvars = 3
        coef = np.array([1, 0.25, 0.25])

        def func(x):
            return (np.prod(x[:2], axis=0)+np.sum(
                coef[:, None]*(2*x-1)**2, axis=0))[:, np.newaxis]

        ntrain_samples = 300
        # train_samples = np.random.uniform(0, 1, (nvars, ntrain_samples))
        train_samples = sobol_sequence(nvars, ntrain_samples)
        train_vals = func(train_samples)

        univariate_variables = [stats.uniform(0, 1)]*nvars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        # var_trans = AffineTransform(variable)

        nu = np.inf
        kernel_var = 1.
        length_scale = np.array([1]*nvars)
        kernel = Matern(length_scale, length_scale_bounds=(1e-2, 10), nu=nu)
        kernel = ConstantKernel(
            constant_value=kernel_var, constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(kernel, n_restarts_optimizer=1, alpha=1e-6,
                             normalize_y=True)
        # gp.set_variable_transformation(var_trans)
        gp.fit(train_samples, train_vals)

        validation_samples = np.random.uniform(0, 1, (nvars, ntrain_samples))
        validation_vals = func(validation_samples)
        error = np.linalg.norm(validation_vals - gp(validation_samples)) / \
            np.linalg.norm(validation_vals)
        print(error)

        pce = approximate(
            train_samples, train_vals, 'polynomial_chaos',
            {'basis_type': 'hyperbolic_cross', 'variable': variable,
             'options': {'max_degree': 4}}).approx
        assert np.linalg.norm(validation_vals - pce(validation_samples)) / \
            np.linalg.norm(validation_vals) < 3e-15

        pce_interaction_terms, pce_sobol_indices = get_sobol_indices(
            pce.get_coefficients(), pce.get_indices(), max_order=3)
        pce_main_effects, pce_total_effects = \
            get_main_and_total_effect_indices_from_pce(
                pce.coefficients, pce.indices)

        interaction_terms = np.zeros(
            (nvars, len(pce_interaction_terms)), dtype=int)
        for ii, idx in enumerate(pce_interaction_terms):
            interaction_terms[idx, ii] = 1

        nquad_samples = 100
        sobol_indices, total_effects, mean, variance = \
            compute_expected_sobol_indices(
                gp, variable, interaction_terms, nquad_samples=nquad_samples)
        # print(sobol_indices, pce_sobol_indices)
        assert np.allclose(
            sobol_indices, pce_sobol_indices, rtol=1e-4, atol=1e-4)
        # print(total_effects, pce_total_effects)
        assert np.allclose(
            total_effects, pce_total_effects, rtol=1e-4, atol=1e-4)

    def test_generate_gp_realizations(self):
        bounds = np.array(
            [0.2, 0.6, 1.15e-8, 1.15e-4, 0.2e-3, 160.e-3, 0.02, 0.1, 1., 5.,
             2., 8., 0.1, 0.5, 600., 1800., 0.2, 1., 7.e-7, 3.e-6])
        # I = [0, 1, 7, 8]
        # I = [0, 1, 2, 3]
        # bounds = bounds[I]
        # lb, ub = bounds[:2]
        length_scale = (bounds[1::2]-bounds[::2])/1
        univariate_variables = [
            stats.uniform(bounds[2*ii], bounds[2*ii+1]-bounds[2*ii])
            for ii in range(len(bounds)//2)]
        variable = IndependentMarginalsVariable(
            univariate_variables)
        # lb, ub = 1e4, 1e5
        # # lb, ub = 1e0, 1e1
        # variable =  IndependentMarginalsVariable(
        #     [stats.uniform(lb, ub-lb)])#, stats.uniform(1e4, 1e5-1e4)])
        # length_scale = (ub-lb)/10

        nvars = variable.num_vars()

        fkernel = Matern(length_scale, length_scale_bounds='fixed', nu=np.inf)
        fkernel = ConstantKernel(
            constant_value=1e5, constant_value_bounds='fixed')*fkernel

        coef = np.random.normal(0, 1, (1000, 1))
        kernel_samples = generate_independent_random_samples(
            variable, coef.shape[0])

        def fun(samples):
            return fkernel(samples.T, kernel_samples.T).dot(coef)

        if nvars == 1:
            ntrain_samples = 5
        else:
            ntrain_samples = 106

        # train_samples = sobol_sequence(
        #     nvars, ntrain_samples, variable=variable)
        marginal_icdfs = [v.ppf for v in variable.marginals()]
        start_index = 2000
        # start_index needs to be large so validation samples
        # are not the same as candidate samples used by
        # RandomGaussianProcessRealizations
        train_samples = transformed_halton_sequence(
            marginal_icdfs, nvars, ntrain_samples, start_index)

        train_vals = fun(train_samples)
        normalize_y, constant_value = True, 1
        # normalize_y, constant_value = False, fkernel.k1.constant_value

        alpha = 1e-8
        var_trans = AffineTransform(variable)
        kernel = Matern(
            np.ones(nvars), length_scale_bounds=(1e-1, 1e1), nu=np.inf)
        kernel = ConstantKernel(
            constant_value=constant_value,
            constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(kernel, alpha=alpha, normalize_y=normalize_y)
        gp.set_variable_transformation(var_trans=var_trans)
        gp.fit(train_samples, train_vals)
        print(gp.kernel_)
        print(fkernel)

        # X, Y, Z = get_meshgrid_function_data(fun, bounds, 30)
        # cset = plt.contourf(
        #     X, Y, Z, levels=np.linspace(Z.min(),Z.max(), 20))
        # plt.show()

        ngp_realizations = 100000
        # nvalidation_samples = 400
        # ncandidate_samples = 1000
        # ninterpolation_samples = 500
        # gp_realizations = generate_gp_realizations(
        #     gp, ngp_realizations, ninterpolation_samples,
        #     nvalidation_samples, ncandidate_samples,
        #     variable, use_cholesky=False, alpha=alpha)

        if nvars == 1:
            lb, ub = variable.interval(1)
            validation_samples = np.linspace(lb, ub, 101)[None, :]
        else:
            validation_samples = generate_independent_random_samples(
                variable, 10)

        mean_vals, std = gp(validation_samples, return_std=True)

        print('error', np.linalg.norm(mean_vals-fun(validation_samples)) /
              np.linalg.norm(fun(validation_samples)))

        # realization_vals = gp_realizations(validation_samples)
        realization_vals = gp.predict_random_realization(
            validation_samples, ngp_realizations)
        # realization_vals = gp.sample_y(
        #    validation_samples.T, ngp_realizations)[:, 0, :]

        std_error = np.linalg.norm(
            std-realization_vals.std(axis=1))/np.linalg.norm(std)
        # print(std, realization_vals.std(axis=1))
        mean_error = np.linalg.norm(
            mean_vals[:, 0]-realization_vals.mean(axis=1))/np.linalg.norm(
                mean_vals[:, 0])
        print('std of realizations error', std_error)
        print('mean of realizations error', mean_error)
        assert std_error < 3e-3
        assert mean_error < 1e-3

        # plt.plot(validation_samples[0, :], realization_vals, '-k', lw=0.5)
        # plt.plot(validation_samples[0, :], mean_vals, '-r')
        # plt.plot(validation_samples[0, :], realization_vals.mean(axis=1),
        #          '--b')
        # plt.plot(train_samples[0, :], train_vals[:, 0], 'or')
        # plt.plot(validation_samples[0, :], fun(validation_samples), 'g')
        # plt.plot(validation_samples[0, :], mean_vals[:, 0]+std, 'k-')
        # plt.plot(validation_samples[0, :],
        #     realization_vals.mean(axis=1)+realization_vals.std(axis=1),
        #     'y--')
        # plt.show()

    def check_gp_with_matern_gradient_wrt_samples(self, nu):

        nvars = 1
        lb, ub = 0, 1
        ntrain_samples = 20
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples)]*nvars)
        train_vals = func(train_samples)

        kernel = Matern(0.4, length_scale_bounds='fixed', nu=nu)
        kernel = ConstantKernel(
            constant_value=2., constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(kernel)
        gp.fit(train_samples, train_vals)

        x0 = np.full((nvars, 1), 0.5)
        errors = check_gradients(gp, True, x0, disp=False)
        assert errors.min() < 8.2e-6 and errors.max() > 0.6

        kernel = Matern(0.4, length_scale_bounds='fixed', nu=nu)
        gp = GaussianProcess(kernel)
        gp.fit(train_samples, train_vals)

        x0 = np.full((nvars, 1), 0.5)
        errors = check_gradients(gp, True, x0, disp=False)
        assert errors.min() < 1.1e-5 and errors.max() > 0.6

        if nu != np.inf:
            return
        kernel = RBF(0.4, length_scale_bounds='fixed')
        kernel = ConstantKernel(
            constant_value=2., constant_value_bounds='fixed')*kernel
        gp = GaussianProcess(kernel)
        gp.fit(train_samples, train_vals)

        x0 = np.full((nvars, 1), 0.5)
        errors = check_gradients(gp, True, x0, disp=False)
        assert errors.min() < 9.2e-6 and errors.max() > 0.6

    def test_gp_with_matern_gradient_wrt_samples(self):
        self.check_gp_with_matern_gradient_wrt_samples(3/2)
        self.check_gp_with_matern_gradient_wrt_samples(5/2)
        self.check_gp_with_matern_gradient_wrt_samples(np.inf)


class TestSamplers(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_cholesky_sampler_basic_restart(self):
        nvars = 1
        variables = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)]*nvars)
        sampler = CholeskySampler(nvars, 100, variables)
        kernel = Matern(1, length_scale_bounds='fixed', nu=np.inf)
        sampler.set_kernel(kernel)

        num_samples = 10
        samples = sampler(num_samples)[0]

        sampler2 = CholeskySampler(nvars, 100, variables)
        sampler2.set_kernel(kernel)
        samples2 = sampler2(num_samples//2)[0]
        samples2 = np.hstack([samples2, sampler2(num_samples)[0]])
        assert np.allclose(samples2, samples)

    def test_cholesky_sampler_restart_with_changed_kernel(self):
        nvars = 1
        variables = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)]*nvars)
        kernel1 = Matern(1, length_scale_bounds='fixed', nu=np.inf)
        kernel2 = Matern(0.1, length_scale_bounds='fixed', nu=np.inf)

        num_samples = 10
        sampler = CholeskySampler(nvars, 100, variables)
        sampler.set_kernel(kernel1)
        samples = sampler(num_samples)[0]

        sampler2 = CholeskySampler(nvars, 100, variables)
        sampler2.set_kernel(kernel1)
        samples2 = sampler2(num_samples//2)[0]
        sampler2.set_kernel(kernel2)
        samples2 = np.hstack([samples2, sampler2(num_samples)[0]])

        # plt.plot(samples[0, :], samples[0, :]*0, 'o')
        # plt.plot(samples2[0, :], samples2[0, :]*0,'x')
        # plt.show()
        assert np.allclose(sampler2.pivots[:num_samples//2],
                           sampler.pivots[:num_samples//2])
        assert not np.allclose(samples2, samples)

    def test_cholesky_sampler_restart_with_changed_weight_function(self):
        nvars = 1
        variables = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)]*nvars)
        kernel1 = Matern(1, length_scale_bounds='fixed', nu=np.inf)

        def wfunction1(x): return np.ones(x.shape[1])

        def wfunction2(x): return x[0, :]**2

        num_samples = 10
        sampler = CholeskySampler(nvars, 100, variables)
        sampler.set_kernel(kernel1)
        sampler.set_weight_function(wfunction1)
        samples = sampler(num_samples)[0]

        sampler2 = CholeskySampler(nvars, 100, variables)
        sampler2.set_kernel(kernel1)
        sampler2.set_weight_function(wfunction1)
        samples2 = sampler2(num_samples//2)[0]
        sampler2.set_weight_function(wfunction2)
        samples2 = np.hstack([samples2, sampler2(num_samples)[0]])

        assert np.allclose(sampler2.pivots[:num_samples//2],
                           sampler.pivots[:num_samples//2])

        assert not np.allclose(samples2, samples)

    def test_cholesky_sampler_adaptive_gp_fixed_kernel(self):
        nvars = 1
        # variables = IndependentMarginalsVariable(
        #    [stats.uniform(-1, 2)]*nvars)

        def func(samples): return np.array(
                [np.sum(samples**2, axis=0), np.sum(samples**3, axis=0)]).T

        validation_samples = np.random.uniform(0, 1, (nvars, 100))
        nsamples = 3

        kernel = Matern(1, length_scale_bounds='fixed', nu=3.5)
        sampler1 = CholeskySampler(nvars, 100, None)
        sampler1.set_kernel(copy.deepcopy(kernel))
        gp1 = AdaptiveCholeskyGaussianProcessFixedKernel(sampler1, func)
        gp1.refine(nsamples)
        vals1 = gp1(validation_samples)

        # currently AdaptiveGaussianProcess can only handle scalar QoI
        # so only test first QoI of func.
        def func2(samples): return func(samples)[:, :1]

        sampler2 = CholeskySampler(nvars, 100, None)
        sampler2.set_kernel(copy.deepcopy(kernel))
        gp2 = AdaptiveGaussianProcess(kernel=kernel, alpha=1e-12)
        gp2.setup(func2, sampler2)
        gp2.refine(nsamples)
        vals2 = gp2(validation_samples)

        assert np.allclose(gp1.train_samples, gp2.X_train_.T)
        assert np.allclose(gp1.train_values[:, 0:1], gp2.y_train_)
        print(vals1.shape, vals2.shape)
        assert np.allclose(vals1[:, 0:1], vals2)

        # xx = np.linspace(0,1,101)
        # plt.plot(xx,gp1(xx[np.newaxis,:]),'-r')
        # plt.plot(xx,gp2(xx[np.newaxis,:]),'-k')
        # plt.plot(gp1.train_samples[0,:],gp1.train_values[:,0],'ro')
        # plt.plot(gp2.X_train_.T[0,:],gp2.y_train_,'ks')
        # plt.show()

        gp1.refine(2*nsamples)
        vals1 = gp1(validation_samples)
        gp2.refine(2*nsamples)
        vals2 = gp2(validation_samples)
        assert np.allclose(vals1[:, 0:1], vals2)

    def test_cholesky_sampler_adaptive_gp_fixed_kernel_II(self):
        np.random.seed(1)
        nvars = 10
        sampler_length_scale = 0.5
        sampler_matern_nu = np.inf
        ncandidate_samples = 1000

        alpha_stat, beta_stat = 20, 20
        variables = IndependentMarginalsVariable(
            [stats.beta(a=alpha_stat, b=beta_stat)]*nvars)

        generate_samples = partial(
            generate_independent_random_samples, variables)

        weight_function = partial(
            tensor_product_pdf,
            univariate_pdfs=partial(stats.beta.pdf, a=alpha_stat, b=beta_stat))

        def gp_mean_function(kernel, samples, alpha, X):
            return kernel(X.T, samples.T).dot(alpha)

        def random_gaussian_process(kernel, samples):
            alpha = np.random.normal(0, 1, (samples.shape[1], 1))
            return partial(gp_mean_function, kernel, samples, alpha)

        ntrials = 100
        lb, ub = 0, 1
        func_length_scale = sampler_length_scale
        func_matern_nu = sampler_matern_nu
        func_kernel = Matern(
            func_length_scale, length_scale_bounds='fixed', nu=func_matern_nu)
        funcs = [random_gaussian_process(func_kernel, np.random.uniform(
            lb, ub, (nvars, 1000))) for n in range(ntrials)]

        def multiple_qoi_function(funcs, samples):
            return np.array([f(samples)[:, 0] for f in funcs]).T

        func = partial(multiple_qoi_function, funcs)

        sampler_kernel = Matern(
            sampler_length_scale, length_scale_bounds='fixed',
            nu=sampler_matern_nu)

        weight_function = None
        sampler = CholeskySampler(
            nvars, ncandidate_samples, variables,
            generate_random_samples=generate_samples)
        sampler.set_kernel(copy.deepcopy(sampler_kernel))
        sampler.set_weight_function(weight_function)

        nvalidation_samples = 1000
        generate_validation_samples = generate_samples
        validation_samples = generate_validation_samples(nvalidation_samples)
        validation_values = func(validation_samples)

        class Callback(object):
            def __init__(self, validation_samples, validation_values,
                         norm_ord=2):
                self.errors, self.nsamples, self.condition_numbers = [], [], []
                self.validation_samples = validation_samples
                self.validation_values = validation_values
                self.norm = partial(np.linalg.norm, ord=norm_ord)

            def __call__(self, approx):
                pred_values = approx(self.validation_samples)
                assert pred_values.shape == self.validation_values.shape
                error = self.norm(
                    pred_values-self.validation_values, axis=0)/self.norm(
                        self.validation_values, axis=0)
                self.errors.append(error)
                self.nsamples.append(approx.num_training_samples())
                self.condition_numbers.append(approx.condition_number())

        callback = Callback(validation_samples, validation_values)
        gp = AdaptiveCholeskyGaussianProcessFixedKernel(sampler, func)

        # checkpoints = [5, 10, 100, 500]
        checkpoints = [5, 10, 20, 50, 100, 200, 300, 500, 1000]
        nsteps = len(checkpoints)
        for ii in range(nsteps):
            gp.refine(checkpoints[ii])
            callback(gp)

        # print(np.median(callback.errors,axis=1))
        assert np.median(callback.errors, axis=1)[-1] < 1e-3
        # plt.loglog(checkpoints, np.median(callback.errors ,axis=1))
        # plt.show()

    def test_RBF_posterior_variance_gradient_wrt_samples(self):
        nvars = 2
        lb, ub = 0, 1
        ntrain_samples_1d = 10

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples_1d)]*nvars)

        length_scale = [0.1, 0.2][:nvars]
        kernel = RBF(length_scale, length_scale_bounds='fixed')

        pred_samples = np.random.uniform(0, 1, (nvars, 3))
        x0 = train_samples[:, :1]
        grad = RBF_gradient_wrt_samples(
            x0, pred_samples, length_scale)

        fd_grad = approx_jacobian(
            lambda x: kernel(x, pred_samples.T)[0, :], x0[:, 0])
        assert np.allclose(grad, fd_grad, atol=1e-6)
        errors = check_gradients(
            lambda x: kernel(x.T, pred_samples.T)[0, :],
            lambda x: RBF_gradient_wrt_samples(
                x, pred_samples, length_scale), x0)
        assert errors.min() < 1e-6

        jac = RBF_posterior_variance_jacobian_wrt_samples(
            train_samples, pred_samples, kernel)

        x0 = train_samples.flatten(order='F')
        assert np.allclose(
            train_samples, x0.reshape(train_samples.shape, order='F'))

        def func(x_flat):
            return gaussian_process_pointwise_variance(
                kernel, pred_samples, x_flat.reshape(
                    train_samples.shape, order='F'))
        fd_jac = approx_jacobian(func, x0)

        # print(jac, '\n\n',f d_jac)
        # print('\n', np.absolute(jac-fd_jac).max())
        assert np.allclose(jac, fd_jac, atol=1e-5)

        errors = check_gradients(
            func,
            lambda x: RBF_posterior_variance_jacobian_wrt_samples(
                x.reshape(nvars, x.shape[0]//nvars, order='F'),
                pred_samples, kernel), x0[:, np.newaxis])
        assert errors.min() < 5e-6

    def check_matern_gradient_wrt_samples(self, nu):
        nvars = 2
        lb, ub = 0, 1
        ntrain_samples_1d = 3

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples_1d)]*nvars)

        length_scale = [0.1, 0.2][:nvars]
        kernel = Matern(length_scale, length_scale_bounds='fixed', nu=nu)

        pred_samples = np.random.uniform(lb, ub, (nvars, 1))
        x0 = train_samples[:, :1]
        grad = matern_gradient_wrt_samples(
            nu, x0, pred_samples, length_scale)
        # K = kernel(x0.T, pred_samples.T)

        fd_grad = approx_jacobian(
            lambda x: kernel(x, pred_samples.T)[0, :], x0[:, 0])
        assert np.allclose(grad, fd_grad, atol=1e-6)
        errors = check_gradients(
            lambda x: kernel(x.T, pred_samples.T)[0, :],
            lambda x: matern_gradient_wrt_samples(
                nu, x, pred_samples, length_scale), x0)
        assert errors.min() < 1e-6

    def test_matern_gradient_wrt_samples(self):
        self.check_matern_gradient_wrt_samples(3/2)
        self.check_matern_gradient_wrt_samples(5/2)
        self.check_matern_gradient_wrt_samples(np.inf)


    def test_RBF_posterior_variance_gradient_wrt_samples_subset(
            self):
        nvars = 2
        lb, ub = 0, 1
        ntrain_samples_1d = 10
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples_1d)]*nvars)
        # train_vals = func(train_samples)

        new_samples_index = train_samples.shape[1]-10

        length_scale = [0.1, 0.2][:nvars]
        kernel = RBF(length_scale, length_scale_bounds='fixed')

        pred_samples = np.random.uniform(0, 1, (nvars, 3))
        jac = RBF_posterior_variance_jacobian_wrt_samples(
            train_samples, pred_samples, kernel, new_samples_index)

        x0 = train_samples.flatten(order='F')
        assert np.allclose(
            train_samples, x0.reshape(train_samples.shape, order='F'))

        def func(x_flat):
            return gaussian_process_pointwise_variance(
                kernel, pred_samples, x_flat.reshape(
                    train_samples.shape, order='F'))
        fd_jac = approx_jacobian(func, x0)[:, new_samples_index*nvars:]

        # print(jac, '\n\n',f d_jac)
        # print('\n', np.absolute(jac-fd_jac).max())
        assert np.allclose(jac, fd_jac, atol=1e-5)

    def test_integrate_grad_P(self):
        nvars = 2
        # univariate_variables = [stats.norm()]*nvars
        # lb, ub = univariate_variables[0].interval(0.99999)
        lb, ub = -2, 2

        ntrain_samples_1d = 2

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples_1d)]*nvars)

        length_scale = [0.5, 0.4][:nvars]
        kernel = RBF(length_scale, length_scale_bounds='fixed')

        # the shorter the length scale the larger the number of quadrature
        # points is needed
        xx_1d, ww_1d = gauss_hermite_pts_wts_1D(100)
        grad_P = integrate_grad_P(
            [xx_1d]*nvars, [ww_1d]*nvars, train_samples, length_scale)[0]

        a, b = train_samples[:, 0]
        mu = [0]*nvars
        sigma = [1]*nvars
        term1 = gaussian_grad_P_diag_term1(a, length_scale[0], mu[0], sigma[0])
        term2 = gaussian_grad_P_diag_term2(b, length_scale[1], mu[1], sigma[1])
        assert np.allclose(
            term1,
            gaussian_grad_P_offdiag_term1(
                a, a, length_scale[0], mu[0], sigma[0]))
        assert np.allclose(
            term2,
            gaussian_grad_P_offdiag_term2(
                b, b, length_scale[1], mu[1], sigma[1]))
        assert np.allclose(
            term1,
            ((xx_1d-a)/length_scale[0]**2*np.exp(
                -(xx_1d-a)**2/(2*length_scale[0]**2))**2).dot(ww_1d))
        assert np.allclose(
            term2,
            (np.exp(-(xx_1d-b)**2/(2*length_scale[1]**2))**2).dot(ww_1d))

        pred_samples = cartesian_product([xx_1d]*nvars)
        weights = outer_product([ww_1d]*nvars)
        for ii in range(train_samples.shape[1]):
            x0 = train_samples[:, ii:ii+1]
            grad = RBF_gradient_wrt_samples(
                x0, pred_samples, length_scale)
            for jj in range(train_samples.shape[1]):
                x1 = train_samples[:, jj:jj+1]
                K = kernel(pred_samples.T, x1.T)
                for kk in range(nvars):
                    grad_P_quad = (grad[:, kk:kk+1]*K).T.dot(weights)
                    t1 = gaussian_grad_P_offdiag_term1(
                        x0[kk, 0], x1[kk, 0], length_scale[kk],
                        mu[kk], sigma[kk])
                    t2 = gaussian_grad_P_offdiag_term2(
                        x0[1-kk, 0], x1[1-kk, 0], length_scale[1-kk],
                        mu[1-kk], sigma[1-kk])
                    grad_P_exact = t1*t2
                    if ii == jj:
                        grad_P_quad *= 2
                        grad_P_exact *= 2
                        assert np.allclose(grad_P_quad, grad_P_exact)
                        assert np.allclose(grad_P_quad, grad_P[nvars*ii+kk, jj])
                        # assert False
                        # assert np.allclose(grad_P_mc, grad_P[kk,ii,jj])

    def test_integrate_grad_P_II(self):
        nvars = 2
        # univariate_variables = [stats.norm()]*nvars
        # lb, ub = univariate_variables[0].interval(0.99999)
        lb, ub = -2, 2

        ntrain_samples_1d = 4

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples_1d)]*nvars)

        length_scale = [0.5, 0.4, 0.6][:nvars]
        kernel = RBF(length_scale, length_scale_bounds='fixed')

        # the shorter the length scale the larger the number of quadrature
        # points is needed
        xx_1d, ww_1d = gauss_hermite_pts_wts_1D(100)
        grad_P, P = integrate_grad_P(
            [xx_1d]*nvars, [ww_1d]*nvars, train_samples, length_scale)

        def func1(xtr):
            xtr = xtr.reshape((nvars, train_samples.shape[1]), order='F')
            P = 1
            for kk in range(nvars):
                P *= integrate_tau_P(
                    xx_1d, ww_1d, xtr[kk:kk+1, :], length_scale[kk])[1]

            vals = P.flatten(order='F')
            # vals=[P[0,0], P[1,0], ..., P[N-1,0], P[0,1], P[1,1], ... P[N-1,1]
            #         ...]
            return vals

        x0 = train_samples.flatten(order='F')
        P_fd_jac = approx_jacobian(func1, x0)
        ntrain_samples = train_samples.shape[1]
        assert np.allclose(
            P_fd_jac.shape, (ntrain_samples**2, nvars*ntrain_samples))
        P_fd_jac_res = P_fd_jac.reshape(
            (ntrain_samples, ntrain_samples, nvars*ntrain_samples), order='F')
        assert np.allclose(_approx_fprime(x0, lambda x: func1(x).reshape(
            ntrain_samples, ntrain_samples, order='F'),
                                          np.sqrt(np.finfo(float).eps)), P_fd_jac_res)

        # Consider 3 training samples with
        # P = P00 P01 P02
        #     P10 P11 P12
        #     P20 P21 P22
        # for kth training sample grad_P stores
        # Pk0 Pk1 Pk2
        # All entries not involving k are zero, e.g. for k=0 the terms
        # P11 P12 P21 P22 will be zero such that
        # dP/d(x[0,n]) = C00 C01 C02
        #                C10  0   0
        #                C20  0   0
        jac = np.empty_like(P_fd_jac)
        for kk in range(ntrain_samples):
            for nn in range(nvars):
                tmp = np.zeros((ntrain_samples, ntrain_samples))
                tmp[kk, :] = grad_P[kk*nvars+nn, :]
                tmp[:, kk] = tmp[kk, :]
                assert np.allclose(P_fd_jac_res[:, :, kk*nvars+nn], tmp)

        def func2(xtr):
            xtr = xtr.reshape((nvars, train_samples.shape[1]), order='F')
            A_inv = np.linalg.inv(kernel(xtr.T))
            return A_inv.flatten(order='F')

        def func3(xtr):
            xtr = xtr.reshape((nvars, train_samples.shape[1]), order='F')
            P = 1
            for kk in range(nvars):
                P *= integrate_tau_P(
                    xx_1d, ww_1d, xtr[kk:kk+1, :], length_scale[kk])[1]
                A_inv = np.linalg.inv(kernel(xtr.T))
                val = np.sum(A_inv*P)
            return -val

        A_fd_jac = approx_jacobian(func2, x0).reshape((
            ntrain_samples, ntrain_samples, nvars*ntrain_samples), order='F')
        assert np.allclose(_approx_fprime(x0, lambda x: func2(x).reshape(
            ntrain_samples, ntrain_samples, order='F'),
                                          np.sqrt(np.finfo(float).eps)), A_fd_jac)

        A_inv = np.linalg.inv(kernel(train_samples.T))
        assert np.allclose(func3(x0), -np.sum(A_inv*P))
        obj_fd_split = - \
            np.sum(A_fd_jac*P[:, :, np.newaxis] +
                   P_fd_jac_res*A_inv[:, :, np.newaxis], axis=(0, 1))

        obj_fd_jac = approx_jacobian(func3, x0)[0, :]
        assert np.allclose(obj_fd_split, obj_fd_jac)

        assert np.allclose(
            P, func1(x0).reshape((ntrain_samples, ntrain_samples), order='F'))
        jac = np.zeros((nvars*ntrain_samples))
        jac1 = np.zeros((nvars*ntrain_samples))
        AinvPAinv = (A_inv.dot(P).dot(A_inv))
        for kk in range(ntrain_samples):
            K_train_grad_all_train_points_kk = \
                RBF_gradient_wrt_samples(
                    train_samples[:, kk:kk+1], train_samples, length_scale)
            # Use the follow properties for tmp3 and tmp4
            # Do sparse matrix element wise product
            # 0 a 0   D00 D01 D02
            # a b c x D10 D11 D12
            # 0 c 0   D20 D21 D22
            # =2*(a*D01 b*D11 + c*D21)-b*D11
            #
            # Trace [RCRP] = Trace[RPRC] for symmetric matrices
            tmp3 = -2*np.sum(
                K_train_grad_all_train_points_kk.T*AinvPAinv[:, kk], axis=1)
            tmp3 -= -K_train_grad_all_train_points_kk[kk, :]*AinvPAinv[kk, kk]
            jac1[kk*nvars:(kk+1)*nvars] = -tmp3
            tmp4 = 2*np.sum(grad_P[kk*nvars:(kk+1)*nvars]*A_inv[:, kk], axis=1)
            tmp4 -= grad_P[kk*nvars:(kk+1)*nvars, kk]*A_inv[kk, kk]
            jac1[kk*nvars:(kk+1)*nvars] -= tmp4
            # check these numpy operations with an explicit loop calculation
            for nn in range(nvars):
                tmp1 = np.zeros((ntrain_samples, ntrain_samples))
                tmp1[kk, :] = grad_P[kk*nvars+nn, :]
                tmp1[:, kk] = tmp1[kk, :]
                assert np.allclose(P_fd_jac_res[:, :, kk*nvars+nn], tmp1)
                tmp2 = np.zeros((ntrain_samples, ntrain_samples))
                tmp2[kk, :] = K_train_grad_all_train_points_kk[:, nn]
                tmp2[:, kk] = tmp2[kk, :]
                tmp2 = -A_inv.dot(tmp2.dot(A_inv))
                assert np.allclose(
                    A_fd_jac[:, :, kk*nvars+nn], tmp2, atol=1e-6)
                jac[kk*nvars+nn] -= np.sum(tmp2*P+A_inv*tmp1)

        assert np.allclose(jac, obj_fd_jac)
        assert np.allclose(jac1, obj_fd_jac)

        jac2 = \
            RBF_integrated_posterior_variance_gradient_wrt_samples(
                train_samples, [xx_1d]*nvars, [ww_1d]*nvars, kernel)
        assert np.allclose(jac2, obj_fd_jac)

    def test_RBF_integrated_posterior_variance_gradient_wrt_sample_subset(self):
        nvars = 2
        lb, ub = -1, 1
        ntrain_samples_1d = 10
        def func1(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        train_samples = cartesian_product(
            [np.linspace(lb, ub, ntrain_samples_1d)]*nvars)
        # train_vals = func1(train_samples)

        new_samples_index = train_samples.shape[1]-10

        length_scale = [0.1, 0.2][:nvars]
        kernel = RBF(length_scale, length_scale_bounds='fixed')

        xx_1d, ww_1d = gauss_jacobi_pts_wts_1D(100, 0, 0)
        t0 = time.time()
        jac = RBF_integrated_posterior_variance_gradient_wrt_samples(
            train_samples, [xx_1d]*nvars, [ww_1d]*nvars, kernel,
            new_samples_index)
        print(time.time()-t0)

        x0 = train_samples.flatten(order='F')
        assert np.allclose(
            train_samples, x0.reshape(train_samples.shape, order='F'))

        def func(x_flat):
            xtr = x_flat.reshape((nvars, train_samples.shape[1]), order='F')
            P = 1
            for kk in range(nvars):
                P *= integrate_tau_P(
                    xx_1d, ww_1d, xtr[kk:kk+1, :], length_scale[kk])[1]
                A_inv = np.linalg.inv(kernel(xtr.T))
                val = np.sum(A_inv*P)
            return -val

        t0 = time.time()
        fd_jac = approx_jacobian(func, x0)[0, new_samples_index*nvars:]
        print(time.time()-t0)

        print(jac, '\n\n', fd_jac)
        # print('\n', np.absolute(jac-fd_jac).max())
        assert np.allclose(jac, fd_jac, atol=1e-5)

    def test_monte_carlo_gradient_based_ivar_sampler(self):
        nvars = 2
        variables = IndependentMarginalsVariable(
            [stats.beta(20, 20)]*nvars)
        generate_random_samples = partial(
            generate_independent_random_samples, variables)

        # correlation length affects ability to check gradient.
        # As kernel matrix gets more ill conditioned then gradients get worse
        greedy_method = 'ivar'
        # greedy_method = 'chol'
        use_gauss_quadrature = False
        kernel = Matern(.1, length_scale_bounds='fixed', nu=np.inf)
        sampler = IVARSampler(
            nvars, 1000, 1000, generate_random_samples, variables,
            greedy_method, use_gauss_quadrature=use_gauss_quadrature,
            nugget=1e-14)
        sampler.set_kernel(copy.deepcopy(kernel))

        def weight_function(samples):
            return np.prod([variables.marginals()[ii].pdf(samples[ii, :])
                            for ii in range(samples.shape[0])], axis=0)

        if greedy_method == 'chol':
            sampler.set_weight_function(weight_function)

        # nature of training samples affects ability to check gradient. As
        # training samples makes kernel matrix more ill conditioned then
        # gradients get worse
        ntrain_samples_1d = 10
        train_samples = cartesian_product(
            [np.linspace(0, 1, ntrain_samples_1d)]*nvars)
        x0 = train_samples.flatten(order='F')
        if not use_gauss_quadrature:
            # gradients not currently implemented when using quadrature
            errors = check_gradients(
                sampler.objective, sampler.objective_gradient,
                x0[:, np.newaxis], disp=False)
            assert errors.min() < 4e-6

        # gsampler = sampler.greedy_sampler
        # print(np.linalg.norm(gsampler.candidate_samples))
        # print(np.linalg.norm(sampler.pred_samples))

        ntrain_samples = 20
        new_samples1 = sampler(ntrain_samples)[0].copy()

        val1 = gaussian_process_pointwise_variance(
            sampler.greedy_sampler.kernel, sampler.pred_samples,
            sampler.training_samples).mean()
        val2 = gaussian_process_pointwise_variance(
            sampler.greedy_sampler.kernel, sampler.pred_samples,
            sampler.init_guess).mean()
        # can't just call sampler.objective here because self.training_points
        # has been updated and calling objective(sampler.training_samples)
        # will evaluate objective with training samples repeated twice.
        # Similarly init guess will be concatenated with self.training_samples
        # if passed to objective at this point
        # print(val1, val2)
        assert (val1 < val2)

        sampler(2*ntrain_samples)[0]
        # currently the following check will fail because a different set
        # of prediction samples will be generated by greedy sampler
        # assert np.allclose(
        #     1+sampler.greedy_sampler.best_obj_vals[ntrain_samples-1], val1)

        assert np.allclose(
            new_samples1, sampler.training_samples[:, :ntrain_samples],
            atol=1e-12)

        assert np.allclose(
            sampler.greedy_sampler.training_samples[:, :ntrain_samples],
            new_samples1, atol=1e-12)

        val1 = gaussian_process_pointwise_variance(
            sampler.greedy_sampler.kernel, sampler.pred_samples,
            sampler.training_samples).mean()
        # initial guess used by optimizer does not contain
        # fixed training points already selected so add here
        greedy_samples = np.hstack(
            [sampler.training_samples[:, :ntrain_samples],
             sampler.init_guess])
        val2 = gaussian_process_pointwise_variance(
            sampler.greedy_sampler.kernel, sampler.pred_samples,
            greedy_samples).mean()
        print(val1, val2)
        assert (val1 < val2)

        # plt.plot(sampler.training_samples[0, :],
        #          sampler.training_samples[1, :], 'o')
        # plt.plot(sampler.greedy_sampler.training_samples[0, :],
        #          sampler.greedy_sampler.training_samples[1, :], 'x')
        # plt.plot(sampler.init_guess[0, :],
        #         sampler.init_guess[1, :], '^')
        # plt.show()

    def test_quadrature_gradient_based_ivar_sampler(self):
        nvars = 2
        variables = IndependentMarginalsVariable(
            [stats.beta(20, 20)]*nvars)
        generate_random_samples = partial(
            generate_independent_random_samples, variables)

        # correlation length affects ability to check gradient.
        # As kerenl matrix gets more ill conditioned then gradients get worse
        greedy_method = 'ivar'
        # greedy_method = 'chol'
        use_gauss_quadrature = True
        kernel = Matern(.1, length_scale_bounds='fixed', nu=np.inf)
        sampler = IVARSampler(
            nvars, 1000, 1000, generate_random_samples, variables,
            greedy_method, use_gauss_quadrature=use_gauss_quadrature,
            nugget=1e-8)
        sampler.set_kernel(copy.deepcopy(kernel))

        def weight_function(samples):
            return np.prod([variables.marginals()[ii].pdf(samples[ii, :])
                            for ii in range(samples.shape[0])], axis=0)

        if greedy_method == 'chol':
            sampler.set_weight_function(weight_function)

        # nature of training samples affects ability to check gradient. As
        # training samples makes kernel matrix more ill conditioned then
        # gradients get worse
        ntrain_samples_1d = 10
        train_samples = cartesian_product(
            [np.linspace(0, 1, ntrain_samples_1d)]*nvars)
        x0 = train_samples.flatten(order='F')
 
        errors = check_gradients(
            sampler.objective, sampler.objective_gradient,
            x0[:, np.newaxis], disp=False, fd_eps=3*np.logspace(-13, 0, 14)[::-1])
        print(errors)
        assert errors.min() < 3e-5

        # gsampler = sampler.greedy_sampler
        # print(np.linalg.norm(gsampler.candidate_samples))
        # print(np.linalg.norm(sampler.pred_samples))

        ntrain_samples = 20
        new_samples1 = sampler(ntrain_samples)[0].copy()

        A_inv = np.linalg.inv(kernel(sampler.training_samples.T))
        P = sampler.compute_P(sampler.training_samples)
        val1 = 1-np.trace(A_inv.dot(P))
        A_inv = np.linalg.inv(kernel(sampler.init_guess.T))
        P = sampler.compute_P(sampler.init_guess)
        val2 = 1-np.trace(A_inv.dot(P))

        # can't just call sampler.objective here because self.training_points
        # has been updated and calling objective(sampler.training_samples)
        # will evaluate objective with training samples repeated twice.
        # Similarly init guess will be concatenated with self.training_samples
        # if passed to objective at this point
        print(val1, val2)
        assert (val1 < val2)

        sampler(2*ntrain_samples)[0]
        assert np.allclose(
            1+sampler.greedy_sampler.best_obj_vals[ntrain_samples-1], val1)

        assert np.allclose(
            new_samples1, sampler.training_samples[:, :ntrain_samples],
            atol=1e-12)

        assert np.allclose(
            sampler.greedy_sampler.training_samples[:, :ntrain_samples],
            new_samples1, atol=1e-12)

        A_inv = np.linalg.inv(kernel(sampler.training_samples.T))
        P = sampler.compute_P(sampler.training_samples)
        val1 = 1-np.trace(A_inv.dot(P))

        # init guess used by optimizer does not contain
        # fixed trainign points already selected so add here
        greedy_samples = np.hstack(
            [sampler.training_samples[:, :ntrain_samples],
             sampler.init_guess])
        A_inv = np.linalg.inv(kernel(greedy_samples.T))
        P = sampler.compute_P(greedy_samples)
        val2 = 1-np.trace(A_inv.dot(P))
        print(val1, val2)
        assert (val1 < val2)

        # plt.plot(sampler.training_samples[0, :],
        #          sampler.training_samples[1, :], 'o')
        # plt.plot(sampler.greedy_sampler.training_samples[0, :],
        #          sampler.greedy_sampler.training_samples[1, :], 'x')
        # plt.plot(sampler.init_guess[0, :],
        #         sampler.init_guess[1, :], '^')
        # plt.show()

    def test_greedy_gauss_quadrature_ivar_sampler_I(self):
        nvars = 2
        variables = IndependentMarginalsVariable(
            [stats.beta(20, 20)]*nvars)
        generate_random_samples = partial(
            generate_independent_random_samples, variables)

        kernel = Matern(.1, length_scale_bounds='fixed', nu=np.inf)
        np.random.seed(1)
        sampler1 = GreedyIntegratedVarianceSampler(
            nvars, 100, 10, generate_random_samples,
            variables, use_gauss_quadrature=True, econ=True)
        sampler1.set_kernel(kernel)
        np.random.seed(1)
        sampler2 = GreedyIntegratedVarianceSampler(
            nvars, 100, 10, generate_random_samples,
            variables, use_gauss_quadrature=True, econ=False)
        sampler2.set_kernel(kernel)

        obj_vals1 = sampler1.objective_vals_econ()
        obj_vals2 = sampler2.objective_vals()
        assert np.allclose(obj_vals1, obj_vals2)
        pivot1 = sampler1.refine_econ()
        pivot2 = sampler2.refine_naive()
        assert np.allclose(pivot1, pivot2)

        for nsamples in range(1, 5+1):
            # refine functions update internal variables so reset
            np.random.seed(1)
            sampler1 = GreedyIntegratedVarianceSampler(
                nvars, 50, 1000, generate_random_samples,
                variables, use_gauss_quadrature=True, econ=True)
            np.random.seed(1)
            sampler2 = GreedyIntegratedVarianceSampler(
                nvars, 50, 1000, generate_random_samples,
                variables, use_gauss_quadrature=True, econ=False)
            kernel = Matern(.1, length_scale_bounds='fixed', nu=np.inf)
            sampler1.set_kernel(kernel)
            sampler2.set_kernel(kernel)
            # print('nsamples',nsamples)
            sampler1(nsamples)
            sampler2(nsamples)
            assert np.allclose(
                sampler1.L,
                np.linalg.cholesky(sampler1.A[np.ix_(sampler1.pivots,
                                                     sampler1.pivots)]))

            obj_vals1 = sampler1.objective_vals_econ()
            obj_vals2 = sampler2.objective_vals()
            obj_vals3 = sampler1.vectorized_objective_vals_econ()
            # print(obj_vals1, obj_vals2)
            # print(obj_vals1, obj_vals3)
            assert np.allclose(obj_vals1, obj_vals2)
            assert np.allclose(obj_vals1, obj_vals3)
            pivot1 = sampler1.refine_econ()
            pivot2 = sampler2.refine_naive()
            # print(pivot1, pivot2)
            assert np.allclose(pivot1, pivot2)

    def check_greedy_monte_carlo_ivar_sampler(
            self, nvars, kernel, kernels_1d):
        variables = IndependentMarginalsVariable(
            [stats.beta(20, 20)]*nvars)
        generate_random_samples = partial(
            generate_independent_random_samples, variables)

        use_gauss_quadrature = False
        np.random.seed(1)
        sampler1 = GreedyIntegratedVarianceSampler(
            nvars, 2000, 1000, generate_random_samples,
            variables, use_gauss_quadrature=use_gauss_quadrature, econ=True,
            compute_cond_nums=True)
        sampler1.set_kernel(kernel, kernels_1d=kernels_1d)
        np.random.seed(1)
        sampler2 = GreedyIntegratedVarianceSampler(
            nvars, 2000, 1000, generate_random_samples,
            variables, use_gauss_quadrature=use_gauss_quadrature, econ=False,
            compute_cond_nums=True)
        sampler2.set_kernel(kernel, kernels_1d=kernels_1d)
        assert np.allclose(sampler1.pred_samples, sampler2.pred_samples)

        nsamples = 20
        # nsamples = 100

        t0 = time.time()
        samples1 = sampler1(nsamples)[0]
        assert np.allclose(
            sampler1.L[:nsamples, :nsamples],
            np.linalg.cholesky(kernel(sampler1.training_samples.T)))
        time1 = time.time()-t0
        print(time1)

        # samples = np.random.beta(20, 20, (nvars, 1000))
        samples = sampler1.pred_samples
        variance = gaussian_process_pointwise_variance(
            kernel, samples, sampler1.training_samples)
        assert np.allclose(variance.mean(), 1+sampler1.best_obj_vals[-1])

        t0 = time.time()
        samples2 = sampler2(nsamples)[0]
        time2 = time.time()-t0
        print(time1, time2)
        assert time1 < time2

        assert np.allclose(samples1, samples2)

        # if nvars !=2:
        #     return
        # plt.plot(samples1[0,:], samples1[1,:], 'o')
        # plt.plot(samples2[0,:], samples2[1,:], 'x')
        # plt.figure()
        # print(np.arange(len(sampler1.cond_nums))+1,sampler1.cond_nums)
        # plt.loglog(np.arange(len(sampler1.cond_nums))+1,sampler1.cond_nums)
        # plt.loglog(np.arange(len(sampler2.cond_nums))+1,sampler2.cond_nums)
        # plt.show()

    def test_greedy_monte_carlo_ivar_sampler_II(self):
        # TODO Add check to IVAR and VarofMean samplers to make sure
        # kernel and 1d_kernels are consistent
        kernel = Matern(.1, length_scale_bounds='fixed', nu=np.inf)
        kernels_1d = None
        self.check_greedy_monte_carlo_ivar_sampler(2, kernel, kernels_1d)

        kernel = Matern(.1, length_scale_bounds='fixed', nu=2.5)
        kernels_1d = None
        self.check_greedy_monte_carlo_ivar_sampler(2, kernel, kernels_1d)

    def test_greedy_variance_of_mean_sampler(self):
        nvars = 2
        variables = IndependentMarginalsVariable(
            [stats.beta(20, 20)]*nvars)
        generate_random_samples = partial(
            generate_independent_random_samples, variables)

        sampler = GreedyVarianceOfMeanSampler(
            nvars, 1000, 10, generate_random_samples,
            variables, use_gauss_quadrature=True, econ=True)
        kernel = Matern(.4, length_scale_bounds='fixed', nu=np.inf)
        sampler.set_kernel(kernel)

        sampler.nmonte_carlo_samples = 100000
        sampler.precompute_monte_carlo()
        tau_mc = sampler.tau.copy()
        sampler.nmonte_carlo_samples = 50
        sampler.precompute_gauss_quadrature()
        tau_gq = sampler.tau.copy()
        # print((tau_mc-tau_gq)/tau_mc)
        assert np.allclose(tau_mc, tau_gq, rtol=1e-2)

        kernel = Matern(.1, length_scale_bounds='fixed', nu=np.inf)

        use_gauss_quadrature = False
        nquad_samples = 10000
        # use_gauss_quadrature = True
        # nquad_samples = 50
        np.random.seed(1)
        sampler1 = GreedyVarianceOfMeanSampler(
            nvars, nquad_samples, 1000, generate_random_samples,
            variables, use_gauss_quadrature=use_gauss_quadrature, econ=True,
            compute_cond_nums=True)
        sampler1.set_kernel(kernel)

        ntrain_samples = 20
        new_samples11 = sampler1(ntrain_samples)[0]
        new_samples12 = sampler1(2*ntrain_samples)[0]

        np.random.seed(1)
        sampler2 = GreedyVarianceOfMeanSampler(
            nvars, nquad_samples, 1000, generate_random_samples,
            variables, use_gauss_quadrature=use_gauss_quadrature, econ=False,
            compute_cond_nums=True)
        sampler2.set_kernel(kernel)

        new_samples21 = sampler2(ntrain_samples)[0]
        new_samples22 = sampler2(2*ntrain_samples)[0]

        # plt.plot(sampler1.training_samples[0, :],
        #          sampler1.training_samples[1, :], 'o')
        # plt.plot(sampler2.training_samples[0, :],
        #          sampler2.training_samples[1, :], 'x')
        # plt.figure()
        # print(np.arange(len(sampler1.cond_nums))+1,sampler1.cond_nums)
        # plt.loglog(np.arange(len(sampler1.cond_nums))+1,sampler1.cond_nums)
        # plt.loglog(np.arange(len(sampler2.cond_nums))+1,sampler2.cond_nums)
        # plt.show()

        assert np.allclose(new_samples11, new_samples21)
        # Note: The sequences computed with econ on and off will diverge
        # when the sample sets produce a kernel matrix with a large condition
        # number
        assert np.allclose(new_samples12, new_samples22)

    def compare_ivar_samplers(self):
        nvars = 2
        variables = IndependentMarginalsVariable(
            [stats.beta(20, 20)]*nvars)
        generate_random_samples = partial(
            generate_independent_random_samples, variables)

        # correlation length affects ability to check gradient.
        # As kerenl matrix gets more ill conditioned then gradients get worse
        kernel = Matern(.1, length_scale_bounds='fixed', nu=np.inf)
        sampler = IVARSampler(
            nvars, 1000, 1000, generate_random_samples, variables, 'ivar')
        sampler.set_kernel(kernel)

        ntrain_samples = 10
        new_samples1 = sampler(ntrain_samples)[0]

        # new_samples2 = sampler(2*ntrain_samples)[0]

        assert np.allclose(
            sampler.training_samples[:, :ntrain_samples], new_samples1,
            atol=1e-12)

        np.random.seed(1)
        sampler2 = IVARSampler(
            nvars, 1000, 1000, generate_random_samples, variables, 'chol')
        sampler2.set_kernel(kernel)

        def weight_function(samples):
            return np.prod([variables[ii].pdf(samples[ii, :])
                            for ii in range(samples.shape[0])], axis=0)

        sampler2.set_weight_function(weight_function)

        sampler2(ntrain_samples)
        sampler2(ntrain_samples*2)

        # plt.plot(sampler.training_samples[0, :],
        #          sampler.training_samples[1, :], 'o')
        # plt.plot(sampler2.training_samples[0, :],
        #          sampler2.training_samples[1, :], 'x')
        # plt.show()


if __name__ == "__main__":
    gaussian_process_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)
    sampler_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestSamplers)
    unittest.TextTestRunner(verbosity=2).run(sampler_test_suite)
