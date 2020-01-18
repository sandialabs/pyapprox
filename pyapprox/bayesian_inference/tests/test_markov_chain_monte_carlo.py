import unittest
from pyapprox.bayesian_inference.markov_chain_monte_carlo import *
from functools import partial

class LinearModel(object):
    def __init__(self, x):
        """
        Parameters
        ----------
        x: (np.ndarray (nstates)
            The dependent variable (aka 'x') that our model requires
        """

        self.x=x

    def __call__(self,samples):
        """
        A straight line!
        """
        m, c = samples
        return m*self.x + c

class TestMCMC(unittest.TestCase):

    def test_linear_gaussian_model_gradient_free(self):
        nobs  = 10  # number of observations
        sigma = 1.  # standard deviation of noise
        x = np.linspace(0., 9., nobs)

        mtrue = 0.4  # true gradient
        ctrue = 2.   # true y-intercept

        m_mu,m_sigma=1,1
        c_mu,c_sigma=0,4
        model = LinearModel(x)

        # make data
        # set random seed, so the data is reproducible each time
        np.random.seed(716742)  
        data = sigma*np.random.randn(nobs) + model(np.array([mtrue, ctrue]))

        ndraws = 300  # number of draws from the distribution
        # number of "burn-in points" (which we'll discard)
        nburn = min(1000,int(ndraws*0.1))
        njobs=4

        # define model

        # create our Op
        loglike = GaussianLogLike(model, data, sigma)
        #logl = LogLike(loglike)
        logl = LogLikeWithGrad(loglike)

        # use PyMC3 to sampler from log-likelihood
        with pm.Model():
            # uniform priors on m and c
            m = pm.Normal('m', mu = m_mu, sigma=m_sigma)
            c = pm.Normal('c', mu = c_mu, sigma=c_sigma)

            # convert m and c to a tensor vector
            theta = tt.as_tensor_variable([m, c])

            # use a DensityDist (use a lamdba function to "call" the Op)
            pm.DensityDist('likelihood', lambda v: logl(v),
                           observed={'v': theta})

            trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True,
                              start={'m':mtrue,'c':ctrue},cores=njobs)

        print(pm.summary(trace))
        var_names = ['c','m']
        nvars = len(var_names)
        samples = np.empty((nvars,(ndraws-nburn)*njobs))
        for ii in range(nvars):
            samples[ii,:]=trace.get_values(
                var_names[ii],burn=nburn,chains=np.arange(njobs))
        print('mcmc mean',samples.mean(axis=1))
        print('mcmc cov',np.cov(samples))

        Amatrix = np.hstack([np.ones((nobs,1)),x[:,np.newaxis]])
        prior_mean = np.asarray([m_mu,c_mu])
        prior_hessian = np.diag([1./m_sigma**2,1./c_sigma**2])
        noise_covariance_inv = 1./sigma**2*np.eye(nobs)
        
        from pyapprox.bayesian_inference.laplace import \
                laplace_posterior_approximation_for_linear_models
        exact_laplace_mean, exact_laplace_covariance = \
            laplace_posterior_approximation_for_linear_models(
                Amatrix, prior_mean, prior_hessian,
                noise_covariance_inv, data)

        print('exact mean',exact_laplace_mean.squeeze())
        print('exact cov',exact_laplace_covariance)


        # plot the traces
        _ = pm.traceplot(trace)
        #plt.show()


if __name__== "__main__":    
    mcmc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestMCMC)
    unittest.TextTestRunner(verbosity=2).run(mcmc_test_suite)
