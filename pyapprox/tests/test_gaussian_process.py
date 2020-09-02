import unittest
from pyapprox.gaussian_process import *
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
import pyapprox as pya
from scipy import stats

class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_integrate_gaussian_process_gaussian(self):

        nvars=1#2
        func = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]

        mu_scalar,sigma_scalar=3,1
        #mu_scalar,sigma_scalar=0,1

        univariate_variables = [stats.norm(mu_scalar,sigma_scalar)]*nvars
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)

        lb,ub = univariate_variables[0].interval(0.99999)

        ntrain_samples = 5#20
        
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
        print('set constant_value =2')
        kernel = ConstantKernel(
            constant_value=1.,constant_value_bounds='fixed')*kernel
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


        expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var, intermediate_quantities=\
                integrate_gaussian_process(gp,variable,return_full=True)

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
        T = mean_of_mean_gaussian_T(train_samples,delta,mu,sigma)
        #Kinv_y is inv(kernel_var*A).dot(y). Thus multiply by kernel_var to get
        #Haylock formula
        Ainv_y = Kinv_y*kernel_var
        true_expected_random_mean = T.dot(Ainv_y)
        #print(true_expected_random_mean,expected_random_mean)
        assert np.allclose(
            true_expected_random_mean,expected_random_mean,rtol=1e-5)

        U = mean_of_mean_gaussian_U(delta,sigma)
        from scipy.linalg import solve_triangular
        L_inv = solve_triangular(gp.L_.T,np.eye(gp.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        #K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get
        #Haylock formula
        A_inv = K_inv*kernel_var
        true_variance_random_mean = kernel_var*(U-T.dot(A_inv).dot(T.T))
        #print(true_variance_random_mean,variance_random_mean)
        assert np.allclose(
            true_variance_random_mean,variance_random_mean,rtol=1e-5)

        #assert ((true_mean>expected_random_mean-3*std_random_mean) and 
        #        (true_mean<expected_random_mean+3*std_random_mean))
        #(x^2+y^2)^2=x_1^4+x_2^4+2x_1^2x_2^2
        #first term below is sum of x_i^4 terms
        #second term is um of 2x_i^2x_j^2
        #third term is mean x_i^2
        true_var = nvars*(mu_scalar**4+6*mu_scalar**2*sigma_scalar**2+3*sigma_scalar**2)+2*pya.nchoosek(nvars,2)*(mu_scalar**2+sigma_scalar**2)**2-true_mean**2
        #print('True var',true_var)
        #print('Expected random var',expected_random_var)
        #assert np.allclose(expected_random_var,true_var,rtol=1e-3)

        nsamples_mc = 20000
        xx = np.random.normal(mu_scalar,sigma_scalar,(nvars,nsamples_mc))
        from scipy.spatial.distance import cdist
        dists = cdist(train_samples.T/length_scale,xx.T/length_scale,
                      metric='sqeuclidean')
        t = np.exp(-.5*dists)
        V_mc = (kernel_var - np.diag(t.T.dot(K_inv).dot(t))).mean()
        P_mc = (t.dot(t.T))/xx.shape[1]

        yy = np.random.normal(mu_scalar,sigma_scalar,(nvars,nsamples_mc))
        zz = np.random.normal(mu_scalar,sigma_scalar,(nvars,nsamples_mc))
        dists1a = (xx.T/length_scale-yy.T/length_scale)**2
        dists1b = cdist(xx.T/length_scale,train_samples.T/length_scale,
                        metric='sqeuclidean')
        dists1c = cdist(yy.T/length_scale,train_samples.T/length_scale,
                        metric='sqeuclidean')
        L = np.linalg.cholesky(K_inv)
        tKt1=np.sum(
            np.exp(-.5*dists1b).dot(L)*np.exp(-.5*dists1b).dot(L),axis=1)
        #assert np.allclose(np.diag(np.exp(-.5*dists1b).dot(K_inv).dot(np.exp(-.5*dists1b).T)),tKt)
        CC_tmp1 = kernel_var*np.exp(-.5*dists1a)-tKt1
        dists1d = (xx.T/length_scale-zz.T/length_scale)**2
        dists1e = cdist(zz.T/length_scale,train_samples.T/length_scale,
                        metric='sqeuclidean')
        tKt2=np.sum(
            np.exp(-.5*dists1b).dot(L)*np.exp(-.5*dists1e).dot(L),axis=1)
        #assert np.allclose(np.diag(np.exp(-.5*dists1b).dot(K_inv).dot(np.exp(-.5*dists1e).T)),tKt2)
        CC_tmp2 = kernel_var*np.exp(-.5*dists1d)-tKt2
        CC_mc = (CC_tmp1*CC_tmp2).mean()

        P_true = mean_of_mean_gaussian_P(train_samples,delta,mu,sigma)
        #print(P_true,intermediate_quantities['P'],P_mc)
        assert np.allclose(P_true,intermediate_quantities['P'])
        assert np.allclose(P_true,P_mc,rtol=3e-2)
        
        V_true = kernel_var*(1-np.trace(A_inv.dot(P_true)))
        assert np.allclose(V_true,intermediate_quantities['V'])
        assert np.allclose(V_true,V_mc,rtol=2e-2)

        CC_true = variance_of_variance_gaussian_CC(delta,sigma)
        #print(CC_true,intermediate_quantities['CC'])
        print('CC',CC_true,CC_mc)
        assert np.allclose(CC_true,intermediate_quantities['CC'])
        
        C_sq_true = variance_of_variance_gaussian_C_sq(delta,sigma)
        #print(C_sq_true,intermediate_quantities['C_sq'])
        assert np.allclose(C_sq_true,intermediate_quantities['C_sq'])

        #print(variance_random_var, intermediate_quantities)
        MC2_true=variance_of_variance_gaussian_MC2(train_samples,delta,mu,sigma)
        #print('MC2',MC2_true,intermediate_quantities['MC2'])

        C_true = variance_of_variance_gaussian_C(delta,sigma)
        assert np.allclose(C_true,intermediate_quantities['C'])
        assert np.allclose(V_mc,intermediate_quantities['V'],rtol=1e-2)

        M_sq_mc = np.mean((t.T.dot(Kinv_y))**2)
        print(M_sq_mc,intermediate_quantities['M_sq'])
        assert np.allclose(M_sq_mc,intermediate_quantities['M_sq'],rtol=2e-2)
        #assert False

        MMC3_true=variance_of_variance_gaussian_MMC3(
            train_samples,delta,mu,sigma)        
        #print(MMC3_true,'\n',intermediate_quantities['MMC3'])


        #TODO use Monte carlo to verify intermediate quantities, even ones
        #where my answer agrees with Haylock. Check that when these MC
        #quantities are used in E[I_2^2] etc. those expressions are correct.

        nsamples = 1000
        random_means, random_variances = [],[]
        random_I2sq,random_I4,random_I2Isq = [],[],[]
        xx,ww=pya.gauss_hermite_pts_wts_1D(100)
        xx = xx*sigma_scalar + mu_scalar
        quad_points = pya.cartesian_product([xx]*nvars)
        quad_weights = pya.outer_product([ww]*nvars)
        for ii in range(nsamples):
            vals = gp.predict_random_realization(quad_points)[:,0]
            I,I2 = vals.dot(quad_weights),(vals**2).dot(quad_weights)
            random_means.append(I)
            random_variances.append(I2-I**2)
            random_I2sq.append(I2**2)
            random_I2Isq.append(I2*I**2)
            random_I4.append(I**4)

        print(np.mean(random_I2sq))
        print(np.mean(random_I2Isq))
        print(np.mean(random_I4))
        print(np.mean(random_I2sq)-2*np.mean(random_I2Isq)+np.mean(random_I4)-np.mean(random_variances)**2)

        print('MC expected random mean',np.mean(random_means))
        print('MC variance random mean',np.var(random_means))
        print('MC expected random variance',np.mean(random_variances))
        print('MC variance random variance',np.var(random_variances))
        print('expected random mean',expected_random_mean)
        print('variance random mean',variance_random_mean)
        print('expected random variance',expected_random_var)
        print('variance random variance',variance_random_var)
        
        #assert np.allclose(np.mean(random_means),expected_random_mean,rtol=1e-2)
        #assert np.allclose(np.var(random_means),variance_random_mean,rtol=1e-2)
        

    def test_integrate_gaussian_process_uniform(self):
        np.random.seed(1)
        nvars=1
        func = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]

        ntrain_samples = 7
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

        expected_random_mean, variance_random_mean, expected_random_var, \
            variance_random_var=integrate_gaussian_process(gp,variable)

        true_mean = 1/3
        true_var = 1/5-1/3**2
        
        print('True mean',true_mean)
        print('Expected random mean',expected_random_mean)
        std_random_mean = np.sqrt(variance_random_mean)
        print('Variance random mean',variance_random_mean)
        print('Stdev random mean',std_random_mean)
        print('Expected random mean +/- 3 stdev',
              [expected_random_mean-3*std_random_mean,
               expected_random_mean+3*std_random_mean])
        assert np.allclose(true_mean,expected_random_mean,rtol=1e-2)

        print('True var',true_var)
        print('Expected random var',expected_random_var)
        assert np.allclose(expected_random_var,true_var,rtol=1e-2)

        nsamples = 1000
        random_means = []
        xx,ww=pya.gauss_jacobi_pts_wts_1D(100,0,0)
        quad_points = pya.cartesian_product([xx]*nvars)
        quad_weights = pya.outer_product([ww]*nvars)
        for ii in range(nsamples):
            vals = gp.predict_random_realization(quad_points)[:,0]
            random_means.append(vals.dot(quad_weights))

        print('MC expected random mean',np.mean(random_means))
        print('MC variance random mean',np.var(random_means))
        assert np.allclose(np.mean(random_means),expected_random_mean,rtol=1e-2)
        assert np.allclose(np.var(random_means),variance_random_mean,rtol=1e-2)
            
        
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
        
if __name__== "__main__":    
    gaussian_process_test_suite=unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)

