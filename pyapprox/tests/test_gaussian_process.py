import unittest
from pyapprox.gaussian_process import *
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
import pyapprox as pya
from scipy import stats
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist

def compute_mean_and_variance_of_gaussian_process(gp,quad_samples,quad_weights):
    gp_vals = gp(quad_samples)
    mean_of_mean = gp_vals.dot(quad_weights)
    variance_of_mean = gp_vals.dot(quad_weights)

    return mean_of_mean, variance_of_mean

def compute_intermediate_quantities_with_monte_carlo(mu_scalar,sigma_scalar,
                                                     length_scale,train_samples,
                                                     A_inv,kernel_var,
                                                     train_vals):
    nsamples_mc = 20000
    nvars = length_scale.shape[0]
    xx = np.random.normal(mu_scalar,sigma_scalar,(nvars,nsamples_mc))
    yy = np.random.normal(mu_scalar,sigma_scalar,(nvars,nsamples_mc))
    zz = np.random.normal(mu_scalar,sigma_scalar,(nvars,nsamples_mc))
    dists_xx_tt = cdist(train_samples.T/length_scale,xx.T/length_scale,
                        metric='sqeuclidean')
    dists_yy_tt = cdist(train_samples.T/length_scale,yy.T/length_scale,
                        metric='sqeuclidean')
    dists_zz_tt = cdist(train_samples.T/length_scale,zz.T/length_scale,
                        metric='sqeuclidean')
    dists_xx_yy = np.sum((xx.T/length_scale-yy.T/length_scale)**2,axis=1)
    dists_xx_zz = np.sum((xx.T/length_scale-zz.T/length_scale)**2,axis=1)
    t_xx = np.exp(-.5*dists_xx_tt)
    t_yy = np.exp(-.5*dists_yy_tt)
    t_zz = np.exp(-.5*dists_zz_tt)

    mean_gp_xx = t_xx.T.dot(A_inv.dot(train_vals))*kernel_var
    mean_gp_yy = t_yy.T.dot(A_inv.dot(train_vals))*kernel_var
    prior_cov_xx_xx = np.ones((xx.shape[1]))
    L = np.linalg.cholesky(A_inv)
    post_cov_xx_xx = prior_cov_xx_xx - np.sum(
        (t_xx.T.dot(L))**2,axis=1)
    prior_cov_xx_yy = np.exp(-.5*dists_xx_yy)
    post_cov_xx_yy = prior_cov_xx_yy - np.sum(
        t_xx.T.dot(L)*t_yy.T.dot(L),axis=1)
    #assert np.allclose(np.sum(
    #    t_xx.T.dot(L)*t_yy.T.dot(L),axis=1),np.diag(t_xx.T.dot(A_inv).dot(t_yy)))
    prior_cov_xx_zz = np.exp(-.5*dists_xx_zz)
    post_cov_xx_zz = prior_cov_xx_zz - np.sum(
        t_xx.T.dot(L)*t_zz.T.dot(L),axis=1)

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
    Pi_mc = np.zeros((train_samples.shape[1],train_samples.shape[1]))
    for ii in range(train_samples.shape[1]):
        for jj in range(ii,train_samples.shape[1]):
            Pi_mc[ii,jj]=(prior_cov_xx_yy*t_xx[ii,:]*t_yy[jj,:]).mean()
            Pi_mc[jj,ii]=Pi_mc[ii,jj]

    return eta_mc, varrho_mc, phi_mc, CC_mc, chi_mc, M_sq_mc, v_sq_mc, \
        varsigma_sq_mc, P_mc, lamda_mc, CC1_mc, Pi_mc

def verify_quantities(reference_quantities,quantities,tols):
    assert len(reference_quantities)==len(quantities)==len(tols)
    ii=0
    for q_ref,q,tol in zip(reference_quantities,quantities,tols):
        assert np.allclose(q_ref,q,rtol=tol),(ii,q_ref,q,tol)
        ii+=1

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
        #print(gp.kernel_)

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
        # Notes sq exp kernel is exp(-dists/delta). Sklearn RBF kernel is
        # exp(-.5*dists/L**2)
        delta = 2*length_scale[:,np.newaxis]**2
        
        #Kinv_y is inv(kernel_var*A).dot(y). Thus multiply by kernel_var to get
        #formula in notes
        Ainv_y = Kinv_y*kernel_var
        L_inv = solve_triangular(gp.L_.T,np.eye(gp.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        #K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get A_inv
        A_inv = K_inv*kernel_var

        #Verify quantities used to compute mean and variance of mean of GP
        #This is redundant know but helped to isolate incorrect terms
        #when initialing writing tests
        tau_true = gaussian_tau(train_samples,delta,mu,sigma)
        u_true = gaussian_u(delta,sigma)
        varpi_true = compute_varpi(tau_true,A_inv)
        varsigma_sq_true = compute_varsigma_sq(u_true,varpi_true)
        verify_quantities(
            [tau_true,u_true,varpi_true,varsigma_sq_true],
            intermediate_quantities[:4],
            [1e-8]*4)

        #Verify mean and variance of mean of GP
        true_expected_random_mean = tau_true.dot(Ainv_y)
        true_variance_random_mean=variance_of_mean(
            kernel_var,varsigma_sq_true)
        assert np.allclose(
            true_expected_random_mean,expected_random_mean)
        #print(true_variance_random_mean,variance_random_mean)
        assert np.allclose(
            true_variance_random_mean,variance_random_mean)

        #Verify quantities used to compute mean of variance of GP
        #This is redundant know but helped to isolate incorrect terms
        #when initialing writing tests
        P_true = gaussian_P(train_samples,delta,mu,sigma)
        v_sq_true = compute_v_sq(A_inv,P_true)
        zeta_true = compute_zeta(train_vals,A_inv,P_true)
        verify_quantities(
            [P_true,v_sq_true],
            intermediate_quantities[4:6],
            [1e-8]*2)

        true_expected_random_var = mean_of_variance(
            zeta_true,v_sq_true,true_expected_random_mean,
            true_variance_random_mean)
        assert np.allclose(true_expected_random_var,expected_random_var)

        #Verify quantities used to compute variance of variance of GP
        #This is redundant know but helped to isolate incorrect terms
        #when initialing writing tests
        nu_true = gaussian_nu(delta,sigma)
        varphi_true = compute_varphi(A_inv,P_true)
        Pi_true = gaussian_Pi(train_samples,delta,mu,sigma)
        psi_true = compute_psi(A_inv,Pi_true)
        chi_true = compute_chi(nu_true,varphi_true,psi_true)
        phi_true = compute_phi(train_vals,A_inv,Pi_true,P_true)
        lamda_true = gaussian_lamda(train_samples,delta,mu,sigma)
        varrho_true = compute_varrho(lamda_true,A_inv,train_vals,P_true,tau_true)
        xi_1_true = gaussian_xi_1(delta,sigma)
        xi_true = compute_xi(xi_1_true,lamda_true,tau_true,P_true,A_inv)
        verify_quantities(
            [zeta_true,nu_true,varphi_true,Pi_true,psi_true,chi_true,phi_true,lamda_true,varrho_true,xi_1_true,xi_true],
            intermediate_quantities[6:17],
            [1e-8]*11)


        xx,ww=pya.gauss_hermite_pts_wts_1D(100)
        xx = xx*sigma_scalar + mu_scalar
        quad_points = pya.cartesian_product([xx]*nvars)
        quad_weights = pya.outer_product([ww]*nvars)
        mean_of_mean, variance_of_mean = \
            compute_mean_and_variance_of_gaussian_process(
                gp,quad_points,quad_weights)

        assert False
        #(x^2+y^2)^2=x_1^4+x_2^4+2x_1^2x_2^2
        #first term below is sum of x_i^4 terms
        #second term is sum of 2x_i^2x_j^2
        #third term is mean x_i^2
        true_var = nvars*(mu_scalar**4+6*mu_scalar**2*sigma_scalar**2+3*sigma_scalar**2)+2*pya.nchoosek(nvars,2)*(mu_scalar**2+sigma_scalar**2)**2-true_mean**2
        #print('True var',true_var)
        #print('Expected random var',expected_random_var)
        #assert np.allclose(expected_random_var,true_var,rtol=1e-3)

        eta_mc, varrho_mc, phi_mc, CC_mc, chi_mc, M_sq_mc, v_sq_mc, \
            varsigma_sq_mc, P_mc, lamda_mc, CC1_mc, Pi_mc = \
                compute_intermediate_quantities_with_monte_carlo(
                    mu_scalar,sigma_scalar,length_scale,train_samples,
                    A_inv,kernel_var,train_vals)

        # #print(P_true,intermediate_quantities['P'],P_mc)
        # assert np.allclose(P_true,intermediate_quantities['P'])
        # assert np.allclose(P_true,P_mc,rtol=3e-2)
        
        # v_sq_true = kernel_var*(1-np.trace(A_inv.dot(P_true)))
        # assert np.allclose(v_sq_true,intermediate_quantities['v_sq'])
        # assert np.allclose(v_sq_true,v_sq_mc,rtol=2e-2)
       
        # chi_true = variance_of_variance_gaussian_chi(delta,sigma,A_inv,P_true,intermediate_quantities['Pi'])
        # print(chi_true,intermediate_quantities['chi'],chi_mc)
        # assert np.allclose(chi_true,intermediate_quantities['chi'])
        # assert np.allclose(chi_mc,intermediate_quantities['chi'],rtol=1e-2)

        # #print(variance_random_var, intermediate_quantities)
        # lamda_true=variance_of_variance_gaussian_lamda(train_samples,delta,mu,sigma)
        # #print('lamda',lamda_true,intermediate_quantities['lamda'],lamda_mc)
        # assert np.allclose(intermediate_quantities['lamda'],lamda_mc,rtol=1e-2)

        # CC1_true = variance_of_variance_gaussian_CC1(delta,sigma)
        # assert np.allclose(CC1_true,intermediate_quantities['CC1'])
        # #print('CC1',CC1_true,CC1_mc)
        # assert np.allclose(CC1_true,CC1_mc,rtol=1e-2)

        # #CC_true = variance_of_variance_gaussian_CC(delta,sigma,tau_true,P_true,CC1_true,lamda_true,A_inv)
        # CC_true = variance_of_variance_gaussian_CC(delta,sigma,tau_true,P_true,CC1_true,intermediate_quantities['lamda'],A_inv)
        # #print(CC_true,intermediate_quantities['CC'])
        # print('CC',CC_true,CC_mc,intermediate_quantities['CC'])
        # assert np.allclose(CC_true,intermediate_quantities['CC'])
        # #assert np.allclose(CC_mc,intermediate_quantities['CC'],rtol=1e-2)

        # #varrho_true = lamda_true.T.dot(Ainv_y)
        # varrho_true = variance_of_variance_varrho(
        #     intermediate_quantities['lamda'],A_inv,train_vals,P_true,tau_true)
        # print('varrho',varrho_true,intermediate_quantities['varrho'],varrho_mc)
        # #assert np.allclose(varrho_mc,intermediate_quantities['varrho'],rtol=1e-2)
        
        # varsigma_sq_true = variance_of_variance_gaussian_varsigma_sq(delta,sigma,A_inv,tau_true)
        # #assert np.allclose(varsigma_sq_true,intermediate_quantities['varsigma_sq'])
        # print(varsigma_sq_mc,intermediate_quantities['varsigma_sq'])
        # #assert np.allclose(varsigma_sq_mc,intermediate_quantities['varsigma_sq'],rtol=1e-2)

        # print('M_sq',M_sq_mc,intermediate_quantities['M_sq'])
        # assert np.allclose(M_sq_mc,intermediate_quantities['M_sq'],rtol=2e-2)
        # #assert False

        # Pi_true=variance_of_variance_gaussian_Pi(
        #     train_samples,delta,mu,sigma)        
        # #print('Pi',Pi_true,'\n',intermediate_quantities['Pi'])
        # #print('Pi',Pi_mc)
        # assert np.allclose(Pi_mc,intermediate_quantities['Pi'],rtol=2e-2)


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

