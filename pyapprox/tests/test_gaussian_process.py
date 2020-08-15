import unittest
from pyapprox.gaussian_process import *
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
import pyapprox as pya
from scipy import stats

class TestGaussianProcess(unittest.TestCase):
    def test_integrate_gaussian_process_gaussian(self):

        nvars=1
        func = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]

        ntrain_samples = 5
        train_samples = np.linspace(-3,3,ntrain_samples)[np.newaxis,:]
        train_vals = func(train_samples)

        nu=np.inf
        kernel = Matern(length_scale_bounds=(1e-2, 10), nu=nu)
        # optimize variance
        #kernel = 1*kernel
        # optimize gp noise
        #kernel += WhiteKernel(noise_level_bounds=(1e-8, 1))
        gp = GaussianProcess(kernel,n_restarts_optimizer=3)
        gp.fit(train_samples,train_vals)


        univariate_variables = [stats.norm(0,1)]
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)

        expected_random_mean, variance_random_mean=integrate_gaussian_process(
            gp,variable)
        print('True mean',1)
        print('Expected random mean',expected_random_mean)
        print('Stdev random mean',np.sqrt(variance_random_mean))
        print('Expected random mean +/- 2 stdev',
              [expected_random_mean-2*np.sqrt(variance_random_mean),
               expected_random_mean+2*np.sqrt(variance_random_mean)])

        #mu and sigma should match variable
        length_scale=gp.kernel_.length_scale
        Kinv_y = gp.alpha_
        mu = np.array([0]*nvars)
        sigma = np.array([1]*nvars)
        delta = length_scale**2
        dists = (train_samples-mu[:,np.newaxis])
        T = np.prod(
            np.sqrt(delta/(delta+2*sigma**2))*np.exp(
                -(dists)/(delta+2*sigma**2)),axis=0)
        print(T.shape,dists.shape)
        print(T.dot(Kinv_y))

        # xx=np.linspace(-3,3,101)
        # plt.plot(xx,func(xx[np.newaxis,:]))
        # plt.plot(xx,gp(xx[np.newaxis,:]))
        # plt.plot(train_samples[0,:],train_vals[:,0],'o')
        # plt.show()

    def test_integrate_gaussian_process_uniform(self):

        nvars=1
        func = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]

        ntrain_samples = 5
        train_samples = np.linspace(-1,1,ntrain_samples)[np.newaxis,:]
        train_vals = func(train_samples)

        nu=np.inf
        kernel = Matern(length_scale_bounds=(1e-2, 10), nu=nu)
        # optimize variance
        #kernel = 1*kernel
        # optimize gp noise
        #kernel += WhiteKernel(noise_level_bounds=(1e-8, 1))
        gp = GaussianProcess(kernel,n_restarts_optimizer=3)
        gp.fit(train_samples,train_vals)


        univariate_variables = [stats.uniform(-1,2)]
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)

        expected_random_mean, variance_random_mean=integrate_gaussian_process(
            gp,variable)
        print('True mean',1/3)
        print('Expected random mean',expected_random_mean)
        print('Stdev random mean',np.sqrt(variance_random_mean))
        print('Expected random mean +/- 2 stdev',
              [expected_random_mean-2*np.sqrt(variance_random_mean),
               expected_random_mean+2*np.sqrt(variance_random_mean)])

        # xx=np.linspace(-1,1,101)
        # plt.plot(xx,func(xx[np.newaxis,:]))
        # plt.plot(xx,gp(xx[np.newaxis,:]))
        # plt.show()

if __name__== "__main__":    
    gaussian_process_test_suite=unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)

