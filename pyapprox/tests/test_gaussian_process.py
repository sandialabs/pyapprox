import unittest
from pyapprox.gaussian_process import *
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
import pyapprox as pya
from scipy import stats

class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
    
    
    def test_integrate_gaussian_process_gaussian(self):

        nvars=2
        func = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]

        mu_scalar,sigma_scalar=3,1
        #mu_scalar,sigma_scalar=0,1

        univariate_variables = [stats.norm(mu_scalar,sigma_scalar)]*nvars
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)

        lb,ub = univariate_variables[0].interval(0.99999)

        ntrain_samples = 20
        
        train_samples = pya.cartesian_product(
            [np.linspace(lb,ub,ntrain_samples)]*nvars)
        train_vals = func(train_samples)

        #tmp=np.random.normal(mu,sigma,(nvars,10001))
        #print(func(tmp).mean())

        nu=np.inf
        nvars = train_samples.shape[0]
        length_scale = np.array([1]*nvars)
        kernel = Matern(length_scale,length_scale_bounds=(1e-2, 10), nu=nu)
        # fix kernel variance
        kernel = ConstantKernel(
            constant_value=2.,constant_value_bounds='fixed')*kernel
        # optimize kernel variance
        #kernel = ConstantKernel(
        #    constant_value=3,constant_value_bounds=(0.1,10))*kernel
        # optimize gp noise
        #kernel += WhiteKernel(noise_level_bounds=(1e-8, 1))
        # fix gp noise
        kernel += WhiteKernel(noise_level=1e-5,noise_level_bounds='fixed')
        #white kernel K(x_i,x_j) is only nonzeros when x_i=x_j, i.e.
        #it is not used when calling gp.predict
        gp = GaussianProcess(kernel,n_restarts_optimizer=10)
        gp.fit(train_samples,train_vals)
        print(gp.kernel_)

        # xx=np.linspace(lb,ub,101)
        # plt.plot(xx,func(xx[np.newaxis,:]))
        # gp_mean,gp_std = gp(xx[np.newaxis,:],return_std=True)
        # gp_mean = gp_mean[:,0]
        # plt.plot(xx,gp_mean)
        # plt.plot(train_samples[0,:],train_vals[:,0],'o')
        # plt.fill_between(xx,gp_mean-2*gp_std,gp_mean+2*gp_std,alpha=0.5)
        # plt.show()


        expected_random_mean, variance_random_mean, expected_random_var=integrate_gaussian_process(
            gp,variable)

        true_mean = nvars*(sigma_scalar**2+mu_scalar**2)
        print('True mean',true_mean)
        print('Expected random mean',expected_random_mean)
        std_random_mean = np.sqrt(variance_random_mean)
        print('Stdev random mean',std_random_mean)
        print('Expected random mean +/- 3 stdev',
              [expected_random_mean-3*std_random_mean,
               expected_random_mean+3*std_random_mean])
        

        #mu and sigma should match variable
        kernel_types = [Matern]
        kernel = extract_covariance_kernel(gp.kernel_,kernel_types)
        length_scale=np.atleast_1d(kernel.length_scale)
        constant_kernel = extract_covariance_kernel(gp.kernel_,[ConstantKernel])
        if constant_kernel is not None:
            kernel_var = constant_kernel.constant_value
        else:
            kernel_var = 1

        
        Kinv_y = gp.alpha_
        mu = np.array([mu_scalar]*nvars)[:,np.newaxis]
        sigma = np.array([sigma_scalar]*nvars)[:,np.newaxis]
        # Halyok sq exp kernel is exp(-dists/delta). Sklearn RBF kernel is
        # exp(-.5*dists/L**2)
        delta = 2*length_scale[:,np.newaxis]**2
        dists = (train_samples-mu)**2
        T = np.prod(
            np.sqrt(delta/(delta+2*sigma**2))*np.exp(
                -(dists)/(delta+2*sigma**2)),axis=0)
        #Kinv_y is inv(kernel_var*A).dot(y). Thus multiply by kernel_var to get
        #Haylock formula
        Ainv_y = Kinv_y*kernel_var
        true_expected_random_mean = T.dot(Ainv_y)
        #print(true_expected_random_mean,expected_random_mean)
        assert np.allclose(true_expected_random_mean,expected_random_mean,rtol=1e-5)

        U = np.sqrt(delta/(delta+4*sigma**2)).prod()
        from scipy.linalg import solve_triangular
        L_inv = solve_triangular(gp.L_.T,np.eye(gp.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        #K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get
        #Haylock formula
        A_inv = K_inv*kernel_var
        true_variance_random_mean = kernel_var*(U-T.dot(A_inv).dot(T.T))
        #print(true_variance_random_mean,variance_random_mean)
        assert np.allclose(true_variance_random_mean,variance_random_mean,rtol=1e-5)

        assert ((true_mean>expected_random_mean-3*std_random_mean) and 
                (true_mean<expected_random_mean+3*std_random_mean))
        #(x^2+y^2)^2=x_1^4+x_2^4+2x_1^2x_2^2
        #first term below is sum of x_i^4 terms
        #second term is um of 2x_i^2x_j^2
        #third term is mean x_i^2
        true_var = nvars*(mu_scalar**4+6*mu_scalar**2*sigma_scalar**2+3*sigma_scalar**2)+2*pya.nchoosek(nvars,2)*(mu_scalar**2+sigma_scalar**2)**2-true_mean**2
        print('True var',true_var)
        print('Expected random var',expected_random_var)
        assert np.allclose(expected_random_var,true_var,rtol=1e-3)

    def test_integrate_gaussian_process_uniform(self):
        np.random.seed(1)
        nvars=1
        func = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]

        ntrain_samples = 10
        train_samples = np.linspace(-1,1,ntrain_samples)[np.newaxis,:]
        train_vals = func(train_samples)

        nu=np.inf
        kernel = Matern(length_scale_bounds=(1e-2, 10), nu=nu)
        # optimize variance
        #kernel = 1*kernel
        # optimize gp noise
        #kernel += WhiteKernel(noise_level_bounds=(1e-8, 1))
        gp = GaussianProcess(kernel,n_restarts_optimizer=1)
        gp.fit(train_samples,train_vals)


        univariate_variables = [stats.uniform(-1,2)]
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)

        expected_random_mean, variance_random_mean, expected_random_var=integrate_gaussian_process(
            gp,variable)

        true_mean = 1/3
        true_var = 1/5-1/3**2
        
        print('True mean',true_mean)
        print('Expected random mean',expected_random_mean)
        print(variance_random_mean)
        std_random_mean = np.sqrt(variance_random_mean)
        print('Stdev random mean',std_random_mean)
        print('Expected random mean +/- 3 stdev',
              [expected_random_mean-3*std_random_mean,
               expected_random_mean+3*std_random_mean])
        assert np.allclose(true_mean,expected_random_mean,rtol=1e-4)

        assert ((true_mean>expected_random_mean-3*std_random_mean) and 
                (true_mean<expected_random_mean+3*std_random_mean))
        
        print('True var',true_var)
        print('Expected random var',expected_random_var)
        assert np.allclose(expected_random_var,true_var,rtol=1e-3)

if __name__== "__main__":    
    gaussian_process_test_suite=unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)

