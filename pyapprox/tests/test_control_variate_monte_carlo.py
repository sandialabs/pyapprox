import unittest
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.configure_plots import *
from pyapprox.control_variate_monte_carlo import *
from scipy.stats import uniform,norm,lognorm
from functools import partial

skiptest = unittest.skipIf(
    not use_torch, reason="active_subspace package missing")


class PolynomialModelEnsemble(object):
    def __init__(self):
        self.nmodels=5
        self.nvars=1
        self.models = [self.m0,self.m1,self.m2,self.m3,self.m4]
        
        univariate_variables = [uniform(0,1)]
        self.variable=pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        self.generate_samples=partial(
            pya.generate_independent_random_samples,self.variable)

    def m0(self,samples):
        return samples.T**5
    
    def m1(self,samples):
        return samples.T**4
    
    def m2(self,samples):
        return samples.T**3
    
    def m3(self,samples):
        return samples.T**2
    
    def m4(self,samples):
        return samples.T**1

    def get_means(self):
        gauss_legendre = partial(
            pya.gauss_jacobi_pts_wts_1D,alpha_poly=0,beta_poly=0)
        x,w = gauss_legendre(10)
        #scale to [0,1]
        x = (x[np.newaxis,:]+1)/2
        nsamples = x.shape[1]
        nqoi = len(self.models)
        vals = np.empty((nsamples,nqoi))
        for ii in range(nqoi):
            vals[:,ii] = self.models[ii](x)[:,0]
        means = vals.T.dot(w)
        return means

    def get_covariance_matrix(self):
        gauss_legendre = partial(
            pya.gauss_jacobi_pts_wts_1D,alpha_poly=0,beta_poly=0)
        x,w = gauss_legendre(10)
        #scale to [0,1]
        x = (x[np.newaxis,:]+1)/2
        nsamples = x.shape[1]
        nqoi = len(self.models)
        vals = np.empty((nsamples,nqoi))
        for ii in range(nqoi):
            vals[:,ii] = self.models[ii](x)[:,0]
        cov = np.cov(vals,aweights=w,rowvar=False,ddof=0)
        return cov

class TunableModelEnsemble(object):
    
    def __init__(self,theta1,shifts=None):
        """
        Parameters
        ----------
        theta0 : float
            Angle controling 
        Notes
        -----
        The choice of A0, A1, A2 here results in unit variance for each model
        """
        self.A0 = np.sqrt(11)
        self.A1 = np.sqrt(7)
        self.A2 = np.sqrt(3)
        self.nmodels=3
        self.theta0=np.pi/2
        self.theta1=theta1
        self.theta2=np.pi/6
        assert self.theta0>self.theta1 and self.theta1>self.theta2
        self.shifts=shifts
        if self.shifts is None:
            self.shifts = [0,0]
        assert len(self.shifts)==2
        self.models = [self.m0,self.m1,self.m2]

        univariate_variables = [uniform(-1,2),uniform(-1,2)]
        self.variable=pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        self.generate_samples=partial(
            pya.generate_independent_random_samples,self.variable)

        
    def m0(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return (self.A0*(np.cos(self.theta0) * x**5 + np.sin(self.theta0) *
                         y**5))[:,np.newaxis]
    
    def m1(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return (self.A1*(np.cos(self.theta1) * x**3 + np.sin(self.theta1) *
                         y**3)+self.shifts[0])[:,np.newaxis]
    
    def m2(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return (self.A2*(np.cos(self.theta2) * x + np.sin(self.theta2) *
                         y)+self.shifts[1])[:,np.newaxis]

    def get_covariance_matrix(self):
        cov = np.eye(self.nmodels)
        cov[0, 1] = self.A0*self.A1/9*(np.sin(self.theta0)*np.sin(
            self.theta1)+np.cos(self.theta0)*np.cos(self.theta1))
        cov[1, 0] = cov[0,1]
        cov[0, 2] = self.A0*self.A2/7*(np.sin(self.theta0)*np.sin(
            self.theta2)+np.cos(self.theta0)*np.cos(self.theta2))
        cov[2, 0] = cov[0, 2]
        cov[1, 2] = self.A1*self.A2/5*(
            np.sin(self.theta1)*np.sin(self.theta2)+np.cos(
                self.theta1)*np.cos(self.theta2))
        cov[2, 1] = cov[1,2]
        return cov


class ShortColumnModelEnsemble(object):
    def __init__(self):
        self.nmodels=5
        self.nvars=5
        self.models = [self.m0,self.m1,self.m2,self.m3,self.m4]
        self.apply_lognormal=False

        univariate_variables = [
            uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
            lognorm(s=0.5,scale=np.exp(5))]
        self.variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        self.generate_samples=partial(
            pya.generate_independent_random_samples,self.variable)
    
    def extract_variables(self,samples):
        assert samples.shape[0]==5
        b = samples[0,:]
        h = samples[1,:]
        P = samples[2,:]
        M = samples[3,:]
        Y = samples[4,:]
        if self.apply_lognormal:
            Y = np.exp(Y)
        return b,h,P,M,Y

    def m0(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - 4*M/(b*(h**2)*Y) - (P/(b*h*Y))**2)[:,np.newaxis]
    
    def m1(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - 3.8*M/(b*(h**2)*Y) - (
            (P*(1 + (M-2000)/4000))/(b*h*Y))**2)[:,np.newaxis]

    def m2(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P/(b*h*Y))**2)[:,np.newaxis]

    def m3(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P*(1 + M)/(b*h*Y))**2)[:,np.newaxis]

    def m4(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P*(1 + M)/(h*Y))**2)[:,np.newaxis]

    def get_quadrature_rule(self):
        nvars = self.variable.num_vars()
        degrees=[10]*nvars
        var_trans = pya.AffineRandomVariableTransformation(self.variable)
        gauss_legendre = partial(
            pya.gauss_jacobi_pts_wts_1D,alpha_poly=0,beta_poly=0)
        univariate_quadrature_rules = [
            gauss_legendre,gauss_legendre,pya.gauss_hermite_pts_wts_1D,
            pya.gauss_hermite_pts_wts_1D,pya.gauss_hermite_pts_wts_1D]
        x,w = pya.get_tensor_product_quadrature_rule(
            degrees,self.variable.num_vars(),univariate_quadrature_rules,
            var_trans.map_from_canonical_space)
        return x,w

    def get_covariance_matrix(self):
        x,w = self.get_quadrature_rule()

        nsamples = x.shape[1]
        nqoi = len(self.models)
        vals = np.empty((nsamples,nqoi))
        for ii in range(nqoi):
            vals[:,ii] = self.models[ii](x)[:,0]
        cov = np.cov(vals,aweights=w,rowvar=False,ddof=0)
        return cov

    def get_means(self):
        x,w = self.get_quadrature_rule()
        nsamples = x.shape[1]
        nqoi = len(self.models)
        vals = np.empty((nsamples,nqoi))
        for ii in range(nqoi):
            vals[:,ii] = self.models[ii](x)[:,0]
        return vals.T.dot(w).squeeze()

def setup_check_variance_reduction_model_ensemble_short_column(
        nmodels=5,npilot_samples=None):
    example = ShortColumnModelEnsemble()
    model_ensemble = pya.ModelEnsemble(
        [example.models[ii] for ii in range(nmodels)])
    univariate_variables = [
        uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
        lognorm(s=0.5,scale=np.exp(5))]
    variable=pya.IndependentMultivariateRandomVariable(univariate_variables)
    generate_samples=partial(
        pya.generate_independent_random_samples,variable)

    if npilot_samples is not None:
        # The number of pilot samples effects ability of numerical estimate
        # of variance reduction to match theoretical value
        cov, samples, weights = pya.estimate_model_ensemble_covariance(
            npilot_samples,generate_samples,model_ensemble)
    else:
        # it is difficult to create a quadrature rule for the lognormal
        # distribution so instead define the variable as normal and then
        # apply log transform
        univariate_variables = [
            uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
            norm(loc=5,scale=0.5)]
        variable=pya.IndependentMultivariateRandomVariable(
            univariate_variables)

        example.apply_lognormal=True
        cov = example.get_covariance_matrix(variable)[:nmodels,:nmodels]
        example.apply_lognormal=False
        
    return model_ensemble, cov, generate_samples


def setup_check_variance_reduction_model_ensemble_tunable():
    example = TunableModelEnsemble(np.pi/4)
    model_ensemble = pya.ModelEnsemble(example.models)
    cov = example.get_covariance_matrix()
    return model_ensemble, cov, example.generate_samples

def setup_check_variance_reduction_model_ensemble_polynomial():
    example = PolynomialModelEnsemble()
    model_ensemble = pya.ModelEnsemble(example.models)
    cov = example.get_covariance_matrix()
    #npilot_samples=int(1e6)
    #cov, samples, weights = pya.estimate_model_ensemble_covariance(
    #    npilot_samples,generate_samples,model_ensemble)
    return model_ensemble, cov, example.generate_samples

def check_variance_reduction(allocate_samples,generate_samples_and_values,
                             get_cv_weights,get_rsquared,setup_model,
                             rtol=1e-2,ntrials=1e3,max_eval_concurrency=1):

    assert get_rsquared is not None
    model_ensemble, cov, generate_samples = setup_model()
    means, numerical_var_reduction, true_var_reduction = \
        estimate_variance_reduction(
            model_ensemble, cov, generate_samples,
            allocate_samples,generate_samples_and_values,
            get_cv_weights,get_rsquared,ntrials,max_eval_concurrency)


    #print('true',true_var_reduction,'numerical',numerical_var_reduction)
    #print(np.absolute(true_var_reduction-numerical_var_reduction),rtol*np.absolute(true_var_reduction))
    if rtol is not None:
        assert np.allclose(numerical_var_reduction,true_var_reduction,
                           rtol=rtol)


class TestCVMC(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_mlmc_sample_allocation(self):
        # The following will give mlmc with unit variance
        # and discrepancy variances 1,4,4
        target_cost = 81
        cov = np.asarray([[1.00,0.50,0.25],
                          [0.50,1.00,0.50],
                          [0.25,0.50,4.00]])
        # ensure cov is positive definite
        np.linalg.cholesky(cov)
        #print(np.linalg.inv(cov))
        costs = [6,3,1]
        nmodels = len(costs)
        nhf_samples,nsample_ratios, log10_var = pya.allocate_samples_mlmc(
            cov, costs, target_cost)
        assert np.allclose(10**log10_var,1)
        nsamples = np.concatenate([[1],nsample_ratios])*nhf_samples
        lamda = 9
        nsamples_discrepancy = 9*np.sqrt(np.asarray([1/(6+3),4/(3+1),4]))
        nsamples_true = [
            nsamples_discrepancy[0],nsamples_discrepancy[:2].sum(),
            nsamples_discrepancy[1:3].sum()]
        assert np.allclose(nsamples,nsamples_true)

    def test_standardize_sample_ratios(self):
        nhf_samples,nsample_ratios = 10,[2.19,3.32]
        std_nhf_samples, std_nsample_ratios = pya.standardize_sample_ratios(
            nhf_samples,nsample_ratios)
        assert np.allclose(std_nsample_ratios,[2.1,3.3])

    def test_generate_samples_and_values_mfmc(self):
        functions = ShortColumnModelEnsemble()
        model_ensemble = pya.ModelEnsemble(
            [functions.m0,functions.m1,functions.m2])
        univariate_variables = [
            uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
            lognorm(s=0.5,scale=np.exp(5))]
        variable=pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        generate_samples=partial(
            pya.generate_independent_random_samples,variable)
        
        nhf_samples = 10
        nsample_ratios = [2,4]
        samples,values =\
            pya.generate_samples_and_values_mfmc(
                nhf_samples,nsample_ratios,model_ensemble,generate_samples)
    
        for jj in range(1,len(samples)):
            assert samples[jj][1].shape[1]==nsample_ratios[jj-1]*nhf_samples
            idx=1
            if jj==1:
                idx=0
            assert np.allclose(samples[jj][0],samples[jj-1][idx])

    def test_rsquared_mfmc(self):
        functions = ShortColumnModelEnsemble()
        model_ensemble = pya.ModelEnsemble(
            [functions.m0,functions.m3,functions.m4])
        univariate_variables = [
            uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
            lognorm(s=0.5,scale=np.exp(5))]
        variable=pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        generate_samples=partial(
            pya.generate_independent_random_samples,variable)
        npilot_samples = int(1e4)
        pilot_samples = generate_samples(npilot_samples)
        config_vars = np.arange(model_ensemble.nmodels)[np.newaxis,:]
        pilot_samples = pya.get_all_sample_combinations(
            pilot_samples,config_vars)
        pilot_values = model_ensemble(pilot_samples)
        pilot_values = np.reshape(
            pilot_values,(npilot_samples,model_ensemble.nmodels))
        cov = np.cov(pilot_values,rowvar=False)
        
        nhf_samples = 10
        nsample_ratios = np.asarray([2,4])

        nsamples_per_model = np.concatenate(
            [[nhf_samples],nsample_ratios*nhf_samples])

        eta = pya.get_mfmc_control_variate_weights(cov)
        cor = pya.get_correlation_from_covariance(cov)
        var_mfmc = cov[0,0]/nsamples_per_model[0]
        for k in range(1,model_ensemble.nmodels):
            var_mfmc+=(1/nsamples_per_model[k-1]-1/nsamples_per_model[k])*(
                eta[k-1]**2*cov[k,k]+2*eta[k-1]*cor[0,k]*np.sqrt(
                    cov[0,0]*cov[k,k]))
            
        assert np.allclose(var_mfmc/cov[0,0]*nhf_samples,
                           1-pya.get_rsquared_mfmc(cov,nsample_ratios))

    def test_variance_reduction_acv_IS(self):
        setup_model = setup_check_variance_reduction_model_ensemble_tunable
        allocate_samples = pya.allocate_samples_mfmc
        generate_samples_and_values = generate_samples_and_values_acv_IS
        get_cv_weights = partial(
            get_approximate_control_variate_weights,
            get_discrepancy_covariances=get_discrepancy_covariances_IS)
        get_rsquared = partial(
            get_rsquared_acv,
            get_discrepancy_covariances=get_discrepancy_covariances_IS)
        check_variance_reduction(
            allocate_samples, generate_samples_and_values,
            get_cv_weights, get_rsquared, setup_model, rtol=1e-2, ntrials=4e3)

    def test_variance_reduction_acv_MF(self):
        setup_model = \
            setup_check_variance_reduction_model_ensemble_tunable
        
        allocate_samples = pya.allocate_samples_mfmc
        generate_samples_and_values = partial(
            generate_samples_and_values_mfmc, acv_modification=True)
        get_cv_weights = partial(
            get_approximate_control_variate_weights,
            get_discrepancy_covariances=get_discrepancy_covariances_MF)
        get_rsquared = partial(
            get_rsquared_acv,
            get_discrepancy_covariances=get_discrepancy_covariances_MF)
        check_variance_reduction(
            allocate_samples, generate_samples_and_values,
            get_cv_weights, get_rsquared, setup_model, ntrials=1e4,
            rtol=1e-2)
        
    def test_variance_reduction_acv_KL(self):
        KL_sets = [[4,1],[3,1],[3,2],[3,3],[2,1],[2,2]]
        # Note K,L=[nmodels-1,i], for all i<=nmodels-1, e.g. [4,0],
        # will give same result as acv_mf
        for K,L in KL_sets:
            #print(K,L)
            setup_model = \
                setup_check_variance_reduction_model_ensemble_polynomial
            allocate_samples = pya.allocate_samples_mfmc
            generate_samples_and_values = partial(
                generate_samples_and_values_acv_KL,K=K,L=L)
            get_discrepancy_covariances =  partial(
                get_discrepancy_covariances_KL,K=K,L=L)
            get_cv_weights = partial(
                get_approximate_control_variate_weights,
                get_discrepancy_covariances=get_discrepancy_covariances)
            get_rsquared = partial(
                get_rsquared_acv,
                get_discrepancy_covariances=get_discrepancy_covariances)

            # Check sizes of samples allocated to each model are correct
            model_ensemble, cov, generate_samples = setup_model()
            nmodels = cov.shape[0]
            target_cost = int(1e4)
            costs = np.asarray([100//2**ii for ii in range(nmodels)])
            nhf_samples, nsample_ratios = allocate_samples(
                cov, costs, target_cost)[:2]

            samples,values = generate_samples_and_values(
                nhf_samples,nsample_ratios,model_ensemble,generate_samples)
            for ii in range(0,K+1):
                assert samples[ii][0].shape[1]==nhf_samples
                assert values[ii][0].shape[0]==nhf_samples
            for ii in range(K+1,nmodels):
                assert samples[ii][0].shape[1]==samples[L][1].shape[1]
                assert values[ii][0].shape[0]==samples[L][1].shape[1]
            for ii in range(1,K+1):
                assert samples[ii][1].shape[1]==\
                    nsample_ratios[ii-1]*nhf_samples
                assert values[ii][1].shape[0]==\
                    nsample_ratios[ii-1]*nhf_samples
            for ii in range(K+1,nmodels):
                assert samples[ii][1].shape[1]==\
                    nsample_ratios[ii-1]*nhf_samples
                assert values[ii][1].shape[0]==samples[ii][1].shape[1]

            check_variance_reduction(
                allocate_samples, generate_samples_and_values,
                get_cv_weights, get_rsquared, setup_model, ntrials=int(3e4),
                max_eval_concurrency=1)

    def test_variance_reduction_mfmc(self):
        setup_model = \
            setup_check_variance_reduction_model_ensemble_tunable
        allocate_samples = pya.allocate_samples_mfmc
        generate_samples_and_values = generate_samples_and_values_mfmc
        get_cv_weights = get_mfmc_control_variate_weights_pool_wrapper
        get_rsquared = get_rsquared_mfmc
        check_variance_reduction(
            allocate_samples, generate_samples_and_values,
            get_cv_weights, get_rsquared, setup_model, rtol=1e-2,
            ntrials=int(1e4),max_eval_concurrency=1)

    def test_variance_reduction_mlmc(self):
        setup_model = \
            setup_check_variance_reduction_model_ensemble_polynomial 
        allocate_samples = pya.allocate_samples_mlmc
        generate_samples_and_values = generate_samples_and_values_mlmc
        get_cv_weights = get_mlmc_control_variate_weights_pool_wrapper
        get_rsquared = get_rsquared_mlmc
        check_variance_reduction(
            allocate_samples, generate_samples_and_values,
            get_cv_weights, get_rsquared, setup_model, ntrials=5e4,
            max_eval_concurrency=1)

    def test_CVMC(self):
        nhf_samples = 10
        model_ensemble, cov, generate_samples = \
            setup_check_variance_reduction_model_ensemble_polynomial()
        lf_means = PolynomialModelEnsemble().get_means()[1:]
        true_gamma = 1-get_control_variate_rsquared(cov)
        eta = get_control_variate_weights(cov)
        ntrials = int(5e3)
        means = np.empty((ntrials,2))
        for ii in range(ntrials):
            samples = generate_samples(nhf_samples)
            values  = [f(samples) for f in model_ensemble.functions] 
            # compute mean using only hf data
            hf_mean = values[0].mean()
            # compute ACV mean
            acv_mean = compute_control_variate_mean_estimate(
                eta,values,lf_means)
            means[ii,:] = hf_mean,acv_mean
        numerical_gamma=means[:,1].var(axis=0)/means[:,0].var(axis=0)
        rtol=1e-2
        #print('true',true_gamma,'numerical',numerical_gamma)
        #print(np.absolute(true_gamma-numerical_gamma),
        #      rtol*np.absolute(true_gamma))
        assert np.allclose(true_gamma,numerical_gamma,rtol=4e-2)

    def test_allocate_samples_mlmc_lagrange_formulation(self):
        cov = np.asarray([[1.00,0.50,0.25],
                          [0.50,1.00,0.50],
                          [0.25,0.50,4.00]])
        
        costs = np.array([6, 3, 1])
        
        target_cost = 81

        estimator = MLMC(cov,costs)
        estimator.use_lagrange_formulation(True)

        nhf_samples_exact, nsample_ratios_exact = allocate_samples_mlmc(
            cov,costs,target_cost,standardize=False)[:2]

        estimator_cost = nhf_samples_exact*costs[0]+(
            nsample_ratios_exact*nhf_samples_exact).dot(costs[1:])
        assert np.allclose(estimator_cost,target_cost,rtol=1e-12)
        
        lagrange_mult = pya.get_lagrange_multiplier_mlmc(
            cov,costs,nhf_samples_exact)
        #print('lagrange_mult',lagrange_mult)

        x0 = np.concatenate([[nhf_samples_exact],nsample_ratios_exact,
                             [lagrange_mult]])
        if use_torch:
            jac = estimator.jacobian(x0)
            # objective does not have lagrangian shift so account for it
            # missing here
            mlmc_var = estimator.variance_reduction(
                nsample_ratios_exact).item()*cov[0,0]/nhf_samples_exact
            jac[-1]-=mlmc_var
        else:
            jac=None
        
        estimator.use_lagrange_formulation(False)

        optim_method='SLSQP'
        #optim_method='trust-constr'
        factor=1-0.1
        initial_guess = np.concatenate([
            [x0[0]*np.random.uniform(factor,1/factor)],
            x0[1:-1]*np.random.uniform(factor,1/factor,x0.shape[0]-2)])

        nhf_samples,nsample_ratios,var=allocate_samples_acv(
            cov, costs, target_cost, estimator,
            standardize=False,initial_guess=initial_guess,
            optim_method=optim_method)

        #print(nhf_samples,nhf_samples_exact)
        #print(nsample_ratios_exact,nsample_ratios)
        assert np.allclose(nhf_samples_exact,nhf_samples)
        assert np.allclose(nsample_ratios_exact,nsample_ratios)

    def test_ACVMC_sample_allocation(self):
        np.random.seed(1)
        matr = np.random.randn(3,3)
        cov_should = np.dot(matr, matr.T)
        L = np.linalg.cholesky(cov_should)
        samp = np.dot(np.random.randn(100000, 3),L.T)
        cov = np.cov(samp, rowvar=False)

        #model_ensemble, cov, generate_samples = \
        #    setup_check_variance_reduction_model_ensemble_polynomial()
        
        costs = [4, 2, 1]
        target_cost = 20
        
        nhf_samples_init, nsample_ratios_init =  allocate_samples_mlmc(
            cov, costs, target_cost, standardize=True)[:2]
        initial_guess = np.concatenate(
            [[nhf_samples_init],nsample_ratios_init])
         
        nhf_samples,nsample_ratios,log10_var=allocate_samples_acv_best_kl(
            cov,costs,target_cost,standardize=True,
            initial_guess=initial_guess)
        print("opt = ", nhf_samples, nsample_ratios, log10_var)

        # this is a regression test to make sure optimization produces
        # answer consistently. It is hard to determine and exact solution
        regression_log10_var = np.asarray([
            0.5159013235987686,-0.2153434757601942,-0.2153434757601942])
        assert np.allclose(log10_var,regression_log10_var.min())

        gamma = 1-get_rsquared_acv_KL_best(cov,nsample_ratios)
        print(gamma)

        # To recover alexs answer use his standardization and initial guess
        # is mlmc with standardize=True')

    @skiptest
    def test_ACVMC_objective_jacobian(self):
        
        cov = np.asarray([[1.00,0.50,0.25],
                          [0.50,1.00,0.50],
                          [0.25,0.50,4.00]])

        costs = [4, 2, 1]
        
        target_cost = 20

        nhf_samples, nsample_ratios =  pya.allocate_samples_mlmc(
            cov, costs, target_cost)[:2]

        estimator = ACVMF(cov,costs)
        errors = pya.check_gradients(
            partial(acv_sample_allocation_objective,estimator),
            partial(acv_sample_allocation_jacobian,estimator),
            nsample_ratios,disp=False)
        #print(errors.min())
        assert errors.min()<1e-8

    
        
    
if __name__== "__main__":    
    cvmc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestCVMC)
    unittest.TextTestRunner(verbosity=2).run(cvmc_test_suite)

