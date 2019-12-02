from pyapprox.nataf_transformation import *
from pyapprox.probability_measure_sampling import rejection_sampling
from scipy.stats import norm as normal_rv
from scipy.stats import beta as beta_rv
from scipy.stats import gamma as gamma_rv
from functools import partial
import unittest
from pyapprox.utilities import get_tensor_product_quadrature_rule
class TestNatafTransformation(unittest.TestCase):

    def test_independent_gaussian(self):
        num_vars = 2; num_samples = 3

        x_marginal_cdfs     = [normal_rv.cdf]*num_vars
        x_marginal_inv_cdfs = [normal_rv.ppf]*num_vars
        x_marginal_means    = np.asarray([normal_rv.mean()]*num_vars)
        x_marginal_stdevs   = np.asarray([normal_rv.std()]*num_vars)

        x_covariance = np.eye(num_vars)

        x_samples = np.random.normal(0.,1.,(num_vars,num_samples))
        u_samples = nataf_transformation(
            x_samples, x_covariance, x_marginal_cdfs,x_marginal_inv_cdfs,
            x_marginal_means, x_marginal_stdevs)
            
        assert np.allclose(u_samples,x_samples)

    def test_correlated_gaussian(self):
        num_vars = 2; num_samples = 10

        x_marginal_cdfs     = [normal_rv.cdf]*num_vars
        x_marginal_inv_cdfs = [normal_rv.ppf]*num_vars
        x_marginal_means    = np.asarray([normal_rv.mean()]*num_vars)
        x_marginal_stdevs   = np.asarray([normal_rv.std()]*num_vars)

        
        x_correlation = np.array([[1,0.5],[0.5,1]])
        x_covariance = x_correlation.copy()# because variances are 1.0
        x_covariance_chol_factor = np.linalg.cholesky(x_covariance)
        iid_x_samples = np.random.normal(0.,1.,(num_vars,num_samples))
        x_samples = np.dot(x_covariance_chol_factor,iid_x_samples)

        u_samples = nataf_transformation(
            x_samples, x_covariance, x_marginal_cdfs,x_marginal_inv_cdfs,
            x_marginal_means, x_marginal_stdevs)

        assert np.allclose(u_samples,iid_x_samples)


    def test_correlated_gamma(self):
        num_vars = 2;

        gamma_cdf = lambda x: gamma_rv.cdf(x,a=2,scale=3)
        gamma_icdf = lambda x: gamma_rv.ppf(x,a=2,scale=3)
        x_marginal_cdfs    =[gamma_cdf]*num_vars
        x_marginal_inv_cdfs=[gamma_icdf]*num_vars
        x_marginal_means   =np.asarray([gamma_rv.mean(a=2,scale=3)]*num_vars)
        x_marginal_stdevs  =np.asarray([gamma_rv.std(a=2,scale=3)]*num_vars)

        x_correlation = np.array([[1,0.7],[0.7,1]])

        quad_rule = gauss_hermite_pts_wts_1D(11)
        z_correlation = transform_correlations(
            x_correlation, x_marginal_inv_cdfs,
            x_marginal_means, x_marginal_stdevs,quad_rule)

        # from Li HongShuang et al.
        true_z_correlation = np.asarray([[1.,0.7207],[0.7207,1.]])
        assert np.allclose(z_correlation,true_z_correlation,atol=1e-4)


    def test_correlated_beta(self):

        num_vars = 2
        alpha_stat=2
        beta_stat=5
        bisection_opts = {'tol':1e-10,'max_iterations':100}
        
        beta_cdf = lambda x: beta_rv.cdf(x,a=alpha_stat,b=beta_stat)
        beta_icdf = lambda x: beta_rv.ppf(x,a=alpha_stat,b=beta_stat)
        x_marginal_cdfs    =[beta_cdf]*num_vars
        x_marginal_inv_cdfs=[beta_icdf]*num_vars
        x_marginal_means   =np.asarray(
            [beta_rv.mean(a=alpha_stat,b=beta_stat)]*num_vars)
        x_marginal_stdevs  =np.asarray(
            [beta_rv.std(a=alpha_stat,b=beta_stat)]*num_vars)
        beta_pdf = lambda x: beta_rv.pdf(x,a=alpha_stat,b=beta_stat)
        x_marginal_pdfs=[beta_pdf]*num_vars

        x_correlation = np.array([[1,0.7],[0.7,1]])

        quad_rule = gauss_hermite_pts_wts_1D(11)
        z_correlation = transform_correlations(
            x_correlation, x_marginal_inv_cdfs, x_marginal_means,
            x_marginal_stdevs, quad_rule, bisection_opts)
        assert np.allclose(z_correlation[0,1],z_correlation[1,0])

        x_correlation_recovered = \
            gaussian_copula_compute_x_correlation_from_z_correlation(
                x_marginal_inv_cdfs,x_marginal_means,x_marginal_stdevs,
                z_correlation)
        assert np.allclose(x_correlation,x_correlation_recovered)

        z_variable = multivariate_normal(
            mean=np.zeros((num_vars)), cov=z_correlation);
        z_joint_density = lambda x: z_variable.pdf(x.T)
        target_density = partial(
            nataf_joint_density,x_marginal_cdfs=x_marginal_cdfs,
            x_marginal_pdfs=x_marginal_pdfs,z_joint_density=z_joint_density)
        
        # all variances are the same so
        #true_x_covariance  = x_correlation.copy()*x_marginal_stdevs[0]**2
        true_x_covariance  = correlation_to_covariance(
            x_correlation,x_marginal_stdevs)

        def univariate_quad_rule(n):
            x,w=np.polynomial.legendre.leggauss(n)
            x = (x+1.)/2.
            w /= 2.
            return x,w
        x,w = get_tensor_product_quadrature_rule(
            100,num_vars,univariate_quad_rule)
        assert np.allclose(np.dot(target_density(x),w),1.0)

        # test covariance of computed by aplying quadrature to joint density
        mean = np.dot(x*target_density(x),w)
        x_covariance = np.empty((num_vars,num_vars))
        x_covariance[0,0] = np.dot(x[0,:]**2*target_density(x),w)-mean[0]**2
        x_covariance[1,1] = np.dot(x[1,:]**2*target_density(x),w)-mean[1]**2
        x_covariance[0,1] = np.dot(
            x[0,:]*x[1,:]*target_density(x),w)-mean[0]*mean[1]
        x_covariance[1,0]=x_covariance[0,1]
        # error is influenced by bisection_opts['tol']
        assert np.allclose(x_covariance,true_x_covariance,
                           atol=bisection_opts['tol'])

        # test samples generated using Gaussian copula are correct
        num_samples = 10000
        x_samples, true_u_samples = generate_x_samples_using_gaussian_copula(
            num_vars,z_correlation,x_marginal_inv_cdfs,num_samples)

        x_sample_covariance = np.cov(x_samples)
        assert np.allclose(true_x_covariance,x_sample_covariance,atol=1e-2)
        
        u_samples = nataf_transformation(
            x_samples, true_x_covariance, x_marginal_cdfs, x_marginal_inv_cdfs,
            x_marginal_means, x_marginal_stdevs, bisection_opts)

        assert np.allclose(u_samples, true_u_samples)

        trans_samples = inverse_nataf_transformation(
            u_samples, x_covariance, x_marginal_cdfs,x_marginal_inv_cdfs,
            x_marginal_means, x_marginal_stdevs, bisection_opts)

        assert np.allclose(x_samples, trans_samples)

        

if __name__=='__main__':
    nataf_test_suite = unittest.TestLoader().loadTestsFromTestCase(
          TestNatafTransformation)
    unittest.TextTestRunner(verbosity=2).run(nataf_test_suite)

