import numpy as np
import unittest
from scipy.stats import norm, uniform

import pyapprox as pya
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.utilities import get_tensor_product_quadrature_rule
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.bayesian_inference.laplace import \
    laplace_posterior_approximation_for_linear_models

from pyapprox_dev.bayesian_inference.markov_chain_monte_carlo import *


class LinearModel(object):
    def __init__(self, Amatrix):
        """
        Parameters
        ----------
        Amatrix : np.ndarray (nobs,nvars)
            The Jacobian of the linear model
        """
        self.Amatrix=Amatrix

    def __call__(self,samples):
        """
        Evaluate the model
        """
        assert samples.ndim==2
        vals = self.Amatrix.dot(samples).T
        return vals

from pyapprox.models.wrappers import evaluate_1darray_function_on_2d_array
class ExponentialQuarticLogLikelihoodModel(object):
    def __init__(self):
        self.a = 3.0

    def loglikelihood_function(self, x):
        value = -(0.1*x[0]**4 + 0.5*(2.*x[1]-x[0]**2)**2)
        return value

    def gradient(self, x):
        assert x.ndim==2
        assert x.shape[1]==1
        grad = -np.array([12./5.*x[0]**3-4.*x[0]*x[1],
                          4.*x[1]-2.*x[0]**2])
        return grad

    def __call__(self,x):
        return np.array([self.loglikelihood_function(x)]).T

class TestMCMC(unittest.TestCase):

    def test_linear_gaussian_inference(self):
        # set random seed, so the data is reproducible each time
        np.random.seed(1)  
        
        nobs  = 10  # number of observations
        noise_stdev = .1  # standard deviation of noise
        x = np.linspace(0., 9., nobs)
        Amatrix = np.hstack([np.ones((nobs,1)),x[:,np.newaxis]])

        univariate_variables = [norm(1,1),norm(0,4)]
        variables = IndependentMultivariateRandomVariable(
            univariate_variables)

        mtrue = 0.4  # true gradient
        ctrue = 2.   # true y-intercept
        true_sample = np.array([[ctrue, mtrue]]).T

        model = LinearModel(Amatrix)

        # make data
        data = noise_stdev*np.random.randn(nobs)+model(true_sample)[0,:]
        loglike = GaussianLogLike(model, data, noise_stdev**2)
        loglike = PYMC3LogLikeWrapper(loglike)

        # number of draws from the distribution
        ndraws = 5000
        # number of "burn-in points" (which we'll discard)
        nburn = min(1000,int(ndraws*0.1))
        # number of parallel chains
        njobs=4

        #algorithm='nuts'
        algorithm='metropolis'
        samples, effective_sample_size, map_sample = \
            run_bayesian_inference_gaussian_error_model(
                loglike,variables,ndraws,nburn,njobs,
                algorithm=algorithm,get_map=True,print_summary=False)

        prior_mean = np.asarray(
            [rv.mean() for rv in variables.all_variables()])
        prior_hessian = np.diag(
            [1./rv.var() for rv in variables.all_variables()])
        noise_covariance_inv = 1./noise_stdev**2*np.eye(nobs)
       
        exact_mean, exact_covariance = \
            laplace_posterior_approximation_for_linear_models(
                Amatrix, prior_mean, prior_hessian,
                noise_covariance_inv, data)

        print('mcmc mean error',samples.mean(axis=1)-exact_mean)
        print('mcmc cov error',np.cov(samples)-exact_covariance)
        print('MAP sample',map_sample)
        print('exact mean',exact_mean.squeeze())
        print('exact cov',exact_covariance)
        assert np.allclose(map_sample,exact_mean)
        assert np.allclose(
            exact_mean.squeeze(), samples.mean(axis=1),atol=1e-2)
        assert np.allclose(exact_covariance, np.cov(samples), atol=1e-2)


        # plot the traces
        # _ = pm.traceplot(trace)
        # plt.show()

    def test_exponential_quartic(self):
        # set random seed, so the data is reproducible each time
        np.random.seed(2)  
        
        univariate_variables = [uniform(-2,4),uniform(-2,4)]
        plot_range = np.asarray([-1,1,-1,1])*2
        variables = IndependentMultivariateRandomVariable(
            univariate_variables)

        loglike = ExponentialQuarticLogLikelihoodModel()
        loglike = PYMC3LogLikeWrapper(loglike,loglike.gradient)

        # number of draws from the distribution
        ndraws = 500
        # number of "burn-in points" (which we'll discard)
        nburn = min(1000,int(ndraws*0.1))
        # number of parallel chains
        njobs=4

        def unnormalized_posterior(x):
            # avoid use of pymc3 wrapper which only evaluates samples 1 at
            # a time
            vals = np.exp(loglike.loglike(x))
            rvs = variables.all_variables()
            for ii in range(variables.nvars):
                vals[:,0] *= rvs[ii].pdf(x[ii,:])
            return vals

        def univariate_quadrature_rule(n):
            x,w = gauss_jacobi_pts_wts_1D(n,0,0)
            x*=2
            return x,w
        x,w = get_tensor_product_quadrature_rule(
            100,variables.nvars,univariate_quadrature_rule)
        evidence = unnormalized_posterior(x)[:,0].dot(w)
        #print('evidence',evidence)

        exact_mean = ((x*unnormalized_posterior(x)[:,0]).dot(w)/evidence)
        #print(exact_mean)

        algorithm = 'nuts'
        #algorithm = 'smc'
        samples, effective_sample_size, map_sample = \
            run_bayesian_inference_gaussian_error_model(
                loglike,variables,ndraws,nburn,njobs,
                algorithm=algorithm,get_map=True,print_summary=True,
                loglike_grad = loglike.gradient, seed=2)

        # from pyapprox.visualization import get_meshgrid_function_data
        # import matplotlib
        # X,Y,Z = get_meshgrid_function_data(
        #     lambda x: unnormalized_posterior(x)/evidence, plot_range, 50)
        # plt.contourf(
        #     X, Y, Z, levels=np.linspace(Z.min(),Z.max(),30),
        #     cmap=matplotlib.cm.coolwarm)
        # plt.plot(samples[0,:],samples[1,:],'ko')
        # plt.show()
        
        print('mcmc mean error',samples.mean(axis=1)-exact_mean)
        print('MAP sample',map_sample)
        print('exact mean',exact_mean.squeeze())
        print('MCMC mean',samples.mean(axis=1))
        assert np.allclose(map_sample,np.zeros((variables.nvars,1)))
        #tolerance 3e-2 can be exceeded for certain random runs
        assert np.allclose(
            exact_mean.squeeze(), samples.mean(axis=1),atol=3e-2)



if __name__== "__main__":    
    mcmc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestMCMC)
    unittest.TextTestRunner(verbosity=2).run(mcmc_test_suite)
