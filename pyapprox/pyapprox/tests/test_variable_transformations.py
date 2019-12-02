import unittest
from pyapprox.variable_transformations import *
from pyapprox.variables import IndependentMultivariateRandomVariable
from scipy.linalg import lu_factor, lu as scipy_lu
from pyapprox.tests.test_rosenblatt_transformation import rosenblatt_example_2d
from scipy.stats import beta as beta_rv
from scipy.stats import norm as normal_rv
from pyapprox.nataf_transformation import \
    gaussian_copula_compute_x_correlation_from_z_correlation,\
    generate_x_samples_using_gaussian_copula, correlation_to_covariance
from scipy.stats import norm, beta, gamma, binom, uniform

class TestVariableTransformations(unittest.TestCase):

    def test_map_hypercube_samples(self):
        num_vars = 3; num_samples = 4
        current_samples = np.random.uniform(0.,1.,(num_vars,num_samples))
        current_ranges = np.ones(2*num_vars); current_ranges[::2]=0.
        new_ranges = np.ones(2*num_vars); new_ranges[::2]=-1.
        samples = map_hypercube_samples(
            current_samples,current_ranges,new_ranges,
            active_vars=[0,2])
        true_samples = 2*current_samples-1.
        true_samples[1,:]=current_samples[1,:]
        assert np.allclose(true_samples,samples)
        
    
    def test_define_mixed_tensor_product_random_variable(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable 
        type the distribution parameters ARE NOT the same
        """
        univariate_variables = [
            uniform(-1,2),beta(1,1,-1,2),norm(-1,np.sqrt(4)),uniform(),
            uniform(-1,2),beta(2,1,-2,3)]
        var_trans = AffineRandomVariableTransformation(univariate_variables)

        # first sample is on left boundary of all bounded variables
        # and one standard deviation to left of mean for gaussian variable
        # second sample is on right boundary of all bounded variables
        # and one standard deviation to right of mean for gaussian variable
        true_user_samples = np.asarray([[-1,-1,-3,0,-1,-2],[1,1,1,1,1,1]]).T

        canonical_samples = var_trans.map_to_canonical_space(true_user_samples)
        true_canonical_samples = np.ones_like(true_user_samples)
        true_canonical_samples[:,0]=-1
        assert np.allclose(true_canonical_samples,canonical_samples)

        user_samples = var_trans.map_from_canonical_space(canonical_samples)
        assert np.allclose(user_samples,true_user_samples)

    def test_define_mixed_tensor_product_random_variable_contin_discrete(self):
        """
        Construct a multivariate random variable from the tensor-product of
        different one-dimensional variables assuming that a given variable 
        type the distribution parameters ARE NOT the same
        """
        # parameters of binomial distribution
        num_trials = 10
        prob_success = 0.5
        univariate_variables = [
            uniform(),norm(-1,np.sqrt(4)),norm(-1,np.sqrt(4)),
            binom(num_trials,prob_success),norm(-1,np.sqrt(4)),uniform(0,1),
            uniform(0,1),binom(num_trials,prob_success)]
        var_trans = AffineRandomVariableTransformation(univariate_variables)
        
        # first sample is on left boundary of all bounded variables
        # and onr standard deviation to left of mean for gaussian variables
        # second sample is on right boundary of all bounded variables
        # and one standard deviation to right of mean for gaussian variable
        true_user_samples=np.asarray(
            [[0,-3,-3,0,-3,0,0,0],[1,1,1,num_trials,1,1,1,5]]).T

        canonical_samples = var_trans.map_to_canonical_space(true_user_samples)
        true_canonical_samples = np.ones_like(true_user_samples)
        true_canonical_samples[:,0]=-1
        true_canonical_samples[5,0]=-1
        true_canonical_samples[3,:]=[0,num_trials]
        true_canonical_samples[7,:]=[0,5]
        assert np.allclose(true_canonical_samples,canonical_samples)

        user_samples = var_trans.map_from_canonical_space(canonical_samples)
        assert np.allclose(user_samples,true_user_samples)

    def test_rosenblatt_transformation(self):

        true_samples, true_canonical_samples, joint_density, limits = \
          rosenblatt_example_2d(num_samples=10)

        num_vars = 2
        opts = {'limits':limits,'num_quad_samples_1d':100}
        var_trans = RosenblattTransformation(joint_density,num_vars,opts)
        
        samples = var_trans.map_from_canonical_space(
            true_canonical_samples)
        assert np.allclose(true_samples,samples)

        canonical_samples = var_trans.map_to_canonical_space(samples)
        assert np.allclose(true_canonical_samples,canonical_samples)

    def test_nataf_transformation(self):
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

        z_correlation = np.array([[1,0.7],[0.7,1]])

        x_correlation = \
            gaussian_copula_compute_x_correlation_from_z_correlation(
                x_marginal_inv_cdfs,x_marginal_means,x_marginal_stdevs,
                z_correlation)

        x_covariance  = correlation_to_covariance(
            x_correlation,x_marginal_stdevs)
        
        var_trans = NatafTransformation(
            x_marginal_cdfs,x_marginal_inv_cdfs,x_marginal_pdfs,x_covariance,
            x_marginal_means,bisection_opts)

        assert np.allclose(var_trans.z_correlation,z_correlation)

        num_samples = 1000
        true_samples, true_canonical_samples = \
            generate_x_samples_using_gaussian_copula(
                num_vars,z_correlation,x_marginal_inv_cdfs,num_samples)
        
        canonical_samples = var_trans.map_to_canonical_space(true_samples)
        assert np.allclose(true_canonical_samples,canonical_samples)

        samples = var_trans.map_from_canonical_space(
            true_canonical_samples)
        assert np.allclose(true_samples,samples)

        
    def test_transformation_composition_I(self):

        np.random.seed(2)
        true_samples, true_canonical_samples, joint_density, limits = \
          rosenblatt_example_2d(num_samples=10)

        #  rosenblatt_example_2d is defined on [0,1] remap to [-1,1]
        true_canonical_samples=true_canonical_samples*2-1

        num_vars = 2
        opts = {'limits':limits,'num_quad_samples_1d':100}
        var_trans_1 = RosenblattTransformation(joint_density,num_vars,opts)
        var_trans_2 = define_iid_random_variable_transformation(
            uniform(0,1),num_vars)
        var_trans = TransformationComposition([var_trans_1, var_trans_2])
        
        samples = var_trans.map_from_canonical_space(
            true_canonical_samples)
        assert np.allclose(true_samples,samples)

        canonical_samples = var_trans.map_to_canonical_space(samples)
        assert np.allclose(true_canonical_samples,canonical_samples)

    def test_transformation_composition_II(self):
        num_vars = 2
        alpha_stat=5
        beta_stat=2
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

        z_correlation = -0.9*np.ones((num_vars,num_vars))
        for ii in range(num_vars):
            z_correlation[ii,ii]=1.

        x_correlation=gaussian_copula_compute_x_correlation_from_z_correlation(
            x_marginal_inv_cdfs,x_marginal_means,x_marginal_stdevs,
            z_correlation)
        x_covariance  = correlation_to_covariance(
            x_correlation,x_marginal_stdevs)

        var_trans_1 = NatafTransformation(
            x_marginal_cdfs,x_marginal_inv_cdfs,x_marginal_pdfs,x_covariance,
            x_marginal_means)

        # rosenblatt maps to [0,1] but polynomials of bounded variables
        # are in [-1,1] so add second transformation for this second mapping
        normal_cdf = lambda x: normal_rv.cdf(x)
        normal_icdf = lambda x: normal_rv.ppf(x)
        std_normal_marginal_cdfs = [normal_cdf]*num_vars
        std_normal_marginal_inv_cdfs = [normal_icdf]*num_vars
        var_trans_2 = UniformMarginalTransformation(
            std_normal_marginal_cdfs,std_normal_marginal_inv_cdfs)
        var_trans = TransformationComposition([var_trans_1, var_trans_2])

        num_samples = 1000
        true_samples, true_canonical_samples = \
            generate_x_samples_using_gaussian_copula(
                num_vars,z_correlation,x_marginal_inv_cdfs,num_samples)
        true_canonical_samples = normal_rv.cdf(true_canonical_samples)
        
        samples = var_trans.map_from_canonical_space(
            true_canonical_samples)
        assert np.allclose(true_samples,samples)

        canonical_samples = var_trans.map_to_canonical_space(samples)
        assert np.allclose(true_canonical_samples,canonical_samples)

        

    def test_pickle_rosenblatt_transformation(self):
        import pickle, os
        true_samples, true_canonical_samples, joint_density, limits = \
          rosenblatt_example_2d(num_samples=10)

        num_vars = 2
        opts = {'limits':limits,'num_quad_samples_1d':100}
        var_trans = RosenblattTransformation(joint_density,num_vars,opts)

        filename = 'rv_trans.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(var_trans,f)

        with open(filename, 'rb') as f:
            file_var_trans = pickle.load(f)

        os.remove(filename)

    def test_pickle_affine_random_variable_transformation(self):
        import pickle, os

        num_vars = 2
        alpha_stat=2
        beta_stat=10
        var_trans = define_iid_random_variable_transformation(
            beta(alpha_stat,beta_stat,0,1),num_vars)
        
        filename = 'rv_trans.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(var_trans,f)

        with open(filename, 'rb') as f:
            file_var_trans = pickle.load(f)

        os.remove(filename)
                           

if __name__== "__main__":    
    variable_transformations_test_suite = \
      unittest.TestLoader().loadTestsFromTestCase(TestVariableTransformations)
    unittest.TextTestRunner(verbosity=2).run(
        variable_transformations_test_suite)
