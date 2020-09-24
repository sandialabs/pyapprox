import unittest
from pyapprox.gaussian_process import *
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
import pyapprox as pya
from scipy import stats
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
import copy


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
    quad_samples_WWXX = pya.get_all_sample_combinations(
        quad_samples, quad_samples)
    quad_weights_WWXX = pya.outer_product([quad_weights]*2)

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

    variance_of_mean = post_cov_ww_xx.dot(quad_weights_WWXX)

    mean_of_variance = (gp_vals**2+gp_std**2).dot(quad_weights) - (
        variance_of_mean+mean_of_mean**2)

    return mean_of_mean, variance_of_mean, mean_of_variance


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

    def test_gaussian_process_pointwise_variance(self):
        nvars = 1
        lb, ub = 0, 1
        ntrain_samples = 5
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        train_samples = pya.cartesian_product(
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
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)

        lb, ub = univariate_variables[0].interval(0.99999)

        ntrain_samples = 5
        # ntrain_samples = 20

        train_samples = pya.cartesian_product(
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
        kernel += WhiteKernel(noise_level=1e-5, noise_level_bounds='fixed')
        # white kernel K(x_i,x_j) is only nonzeros when x_i=x_j, i.e.
        # it is not used when calling gp.predict
        gp = GaussianProcess(kernel, n_restarts_optimizer=10)
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

        import time
        t0 = time.time()
        expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var, intermediate_quantities =\
            integrate_gaussian_process(gp, variable, return_full=True)
        print('time', time.time()-t0)

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
        xx, ww = pya.gauss_hermite_pts_wts_1D(nxx)
        xx = xx*sigma_scalar + mu_scalar
        quad_points = pya.cartesian_product([xx]*nvars)
        quad_weights = pya.outer_product([ww]*nvars)
        mean_of_mean_quad, variance_of_mean_quad, mean_of_variance_quad = \
            compute_mean_and_variance_of_gaussian_process(
                gp, length_scale, train_samples, A_inv, kernel_var, train_vals,
                quad_points, quad_weights)

        assert np.allclose(mean_of_mean_quad, expected_random_mean)
        assert np.allclose(variance_of_mean_quad, variance_random_mean)
        assert np.allclose(mean_of_variance_quad, expected_random_var)

        nsamples = 4000
        random_means, random_variances = [], []
        random_I2sq, random_I4, random_I2Isq = [], [], []
        xx, ww = pya.gauss_hermite_pts_wts_1D(nxx)
        xx = xx*sigma_scalar + mu_scalar
        quad_points = pya.cartesian_product([xx]*nvars)
        quad_weights = pya.outer_product([ww]*nvars)
        for ii in range(nsamples):
            vals = gp.predict_random_realization(quad_points)[:, 0]
            I, I2 = vals.dot(quad_weights), (vals**2).dot(quad_weights)
            random_means.append(I)
            random_variances.append(I2-I**2)
            random_I2sq.append(I2**2)
            random_I2Isq.append(I2*I**2)
            random_I4.append(I**4)

        # print('MC expected random mean', np.mean(random_means))
        # print('MC variance random mean', np.var(random_means))
        # print('MC expected random variance', np.mean(random_variances))
        # print('MC variance random variance', np.var(random_variances))
        # print('expected random mean', expected_random_mean)
        # print('variance random mean', variance_random_mean)
        # print('expected random variance', expected_random_var)
        # print('variance random variance', variance_random_var)
        assert np.allclose(
            np.mean(random_means), expected_random_mean, rtol=1e-2)
        assert np.allclose(
            np.var(random_means), variance_random_mean, rtol=2.1e-2)
        assert np.allclose(
            expected_random_var, np.mean(random_variances), rtol=1e-2)
        assert np.allclose(
            variance_random_var, np.var(random_variances), rtol=2.2e-2)

    def test_integrate_gaussian_process_uniform(self):
        np.random.seed(1)
        nvars = 1
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        ntrain_samples = 7
        train_samples = np.linspace(-1, 1, ntrain_samples)[np.newaxis, :]
        train_vals = func(train_samples)

        nu = np.inf
        kernel = Matern(length_scale_bounds=(1e-2, 10), nu=nu)
        # optimize variance
        # kernel = 1*kernel
        # optimize gp noise
        # kernel += WhiteKernel(noise_level_bounds=(1e-8, 1))
        gp = GaussianProcess(kernel, n_restarts_optimizer=1)
        gp.fit(train_samples, train_vals)

        univariate_variables = [stats.uniform(-1, 2)]
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)

        expected_random_mean, variance_random_mean, expected_random_var, \
            variance_random_var = integrate_gaussian_process(gp, variable)

        true_mean = 1/3
        true_var = 1/5-1/3**2

        print('True mean', true_mean)
        print('Expected random mean', expected_random_mean)
        std_random_mean = np.sqrt(variance_random_mean)
        print('Variance random mean', variance_random_mean)
        print('Stdev random mean', std_random_mean)
        print('Expected random mean +/- 3 stdev',
              [expected_random_mean-3*std_random_mean,
               expected_random_mean+3*std_random_mean])
        assert np.allclose(true_mean, expected_random_mean, rtol=1e-2)

        print('True var', true_var)
        print('Expected random var', expected_random_var)
        assert np.allclose(expected_random_var, true_var, rtol=1e-2)

        nsamples = 1000
        random_means = []
        xx, ww = pya.gauss_jacobi_pts_wts_1D(100, 0, 0)
        quad_points = pya.cartesian_product([xx]*nvars)
        quad_weights = pya.outer_product([ww]*nvars)
        for ii in range(nsamples):
            vals = gp.predict_random_realization(quad_points)[:, 0]
            random_means.append(vals.dot(quad_weights))

        print('MC expected random mean', np.mean(random_means))
        print('MC variance random mean', np.var(random_means))
        assert np.allclose(
            np.mean(random_means), expected_random_mean, rtol=1e-2)
        assert np.allclose(
            np.var(random_means), variance_random_mean, rtol=1e-2)

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


class TestSamplers(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_cholesky_sampler_basic_restart(self):
        nvars = 1
        variables = [stats.uniform(-1, 2)]*nvars
        sampler = CholeskySampler(nvars, 100, variables)
        kernel = pya.Matern(1, length_scale_bounds='fixed', nu=np.inf)
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
        variables = [stats.uniform(-1, 2)]*nvars
        kernel1 = pya.Matern(1, length_scale_bounds='fixed', nu=np.inf)
        kernel2 = pya.Matern(0.1, length_scale_bounds='fixed', nu=np.inf)

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
        variables = [stats.uniform(-1, 2)]*nvars
        kernel1 = pya.Matern(1, length_scale_bounds='fixed', nu=np.inf)
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
        variables = [stats.uniform(0, 1)]*nvars
        def func(samples): return np.array(
                [np.sum(samples**2, axis=0), np.sum(samples**3, axis=0)]).T

        validation_samples = np.random.uniform(0, 1, (nvars, 100))
        nsamples = 3

        kernel = pya.Matern(1, length_scale_bounds='fixed', nu=3.5)
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

    def test_RBF_gradient_wrt_sample_coordinates(self):
        nvars = 2
        lb, ub = 0, 1
        ntrain_samples_1d = 10
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        train_samples = pya.cartesian_product(
            [np.linspace(lb, ub, ntrain_samples_1d)]*nvars)
        train_vals = func(train_samples)

        length_scale = [0.1, 0.2][:nvars]
        kernel = RBF(length_scale, length_scale_bounds='fixed')

        pred_samples = np.random.uniform(0, 1, (nvars, 3))
        x0 = train_samples[:, :1]
        grad = RBF_gradient_wrt_sample_coordinates(
            x0, pred_samples, length_scale)

        fd_grad = pya.approx_jacobian(
            lambda x: kernel(x, pred_samples.T)[0, :], x0[:,0])
        assert np.allclose(grad, fd_grad, atol=1e-6)
        errors = pya.check_gradients(
            lambda x: kernel(x.T, pred_samples.T)[0, :],
            lambda x: RBF_gradient_wrt_sample_coordinates(
                x, pred_samples, length_scale), x0)
        assert errors.min()<1e-6

        jac = RBF_jacobian_wrt_sample_coordinates(
            train_samples, pred_samples, kernel)

        x0 = train_samples.flatten(order='F')
        assert np.allclose(
            train_samples, x0.reshape(train_samples.shape, order='F'))

        def func(x_flat):
            return gaussian_process_pointwise_variance(
                kernel, pred_samples, x_flat.reshape(
                    train_samples.shape, order='F'))
        fd_jac = pya.approx_jacobian(func, x0)

        # print(jac, '\n\n',f d_jac)
        # print('\n', np.absolute(jac-fd_jac).max())
        assert np.allclose(jac, fd_jac, atol=1e-5)

        errors = pya.check_gradients(
            func,
            lambda x: RBF_jacobian_wrt_sample_coordinates(
                x.reshape(nvars, x.shape[0]//nvars, order='F'),
                pred_samples, kernel), x0[:, np.newaxis])
        assert errors.min()<1e-6


    def test_RBF_gradient_wrt_sample_coordinates_subset(self):
        nvars = 2
        lb, ub = 0, 1
        ntrain_samples_1d = 10
        def func(x): return np.sum(x**2, axis=0)[:, np.newaxis]

        train_samples = pya.cartesian_product(
            [np.linspace(lb, ub, ntrain_samples_1d)]*nvars)
        train_vals = func(train_samples)

        new_samples_index = train_samples.shape[1]-10

        length_scale = [0.1, 0.2][:nvars]
        kernel = RBF(length_scale, length_scale_bounds='fixed')

        pred_samples = np.random.uniform(0, 1, (nvars, 3))
        jac = RBF_jacobian_wrt_sample_coordinates(
            train_samples, pred_samples, kernel, new_samples_index)

        x0 = train_samples.flatten(order='F')
        assert np.allclose(
            train_samples, x0.reshape(train_samples.shape, order='F'))

        def func(x_flat):
            return gaussian_process_pointwise_variance(
                kernel, pred_samples, x_flat.reshape(
                    train_samples.shape, order='F'))
        fd_jac = pya.approx_jacobian(func, x0)[:,new_samples_index*nvars:]

        # print(jac, '\n\n',f d_jac)
        # print('\n', np.absolute(jac-fd_jac).max())
        assert np.allclose(jac, fd_jac, atol=1e-5)


    def test_ivar_sampler(self):
        nvars = 2
        #variables = [stats.uniform(0, 1)]*nvars
        variables = [stats.beta(20, 20)]*nvars
        
        def generate_random_samples(nsamples):
            return np.vstack([v.rvs(size=(1, nsamples)) for v in variables])

        # correlation length affects ability to check gradient. As kerenl matrix
        # gets more ill conditioned then gradients get worse
        kernel = pya.Matern(.1, length_scale_bounds='fixed', nu=np.inf)
        sampler = IVARSampler(
            nvars, 10000, 1000, variables, generate_random_samples)
        sampler.set_kernel(copy.deepcopy(kernel))

        # nature of training samples affects ability to check gradient. As
        # training samples makes kernel matrix more ill conditioned then
        # gradients get worse
        ntrain_samples_1d = 10
        train_samples = pya.cartesian_product(
            [np.linspace(0, 1, ntrain_samples_1d)]*nvars)
        x0 = train_samples.flatten(order='F')
        #fd_grad = pya.approx_jacobian(sampler.objective, x0)
        #grad = sampler.objective_gradient(x0)
        errors = pya.check_gradients(
            sampler.objective, sampler.objective_gradient, x0[:,np.newaxis])
        assert errors.min()<1e-6

        chol_sampler = sampler.greedy_sampler

        ntrain_samples = 10
        new_samples1 = sampler(ntrain_samples)[0]

        chol_samples = chol_sampler.candidate_samples[:, chol_sampler.pivots[
            :chol_sampler.num_completed_pivots]]
        assert (sampler.objective(sampler.training_samples.flatten()) <
                sampler.objective(chol_samples.flatten()))

        # plt.plot(sampler.training_samples[0, :],
        #          sampler.training_samples[1, :], 'o', ms=15)
        # plt.plot(chol_samples[0, :], chol_samples[1, :], 'x', ms=15)

        new_samples2 = sampler(2*ntrain_samples)[0]

        assert np.allclose(
            new_samples1, sampler.training_samples[:, :ntrain_samples],
            atol=1e-12)

        chol_samples = chol_sampler.candidate_samples[:, chol_sampler.pivots[
            :chol_sampler.num_completed_pivots]]
        assert np.allclose(
            chol_samples[:, :ntrain_samples], new_samples1, atol=1e-12) 
        assert (sampler.objective(sampler.training_samples.flatten()) <
                sampler.objective(chol_samples.flatten()))

        plt.plot(sampler.training_samples[0, :],
                 sampler.training_samples[1, :], 'o')
        plt.plot(chol_samples[0, :], chol_samples[1, :], 'x')
        plt.show()


if __name__ == "__main__":
    gaussian_process_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)
    sampler_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestSamplers)
    unittest.TextTestRunner(verbosity=2).run(cholesky_sampler_test_suite)
