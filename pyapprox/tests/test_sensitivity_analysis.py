import unittest
from pyapprox.sensitivity_analysis import *
from scipy.stats import uniform
import pyapprox as pya

def ishigami_function(samples,a=7,b=0.1):    
    vals = np.sin(samples[0,:])+a*np.sin(samples[1,:])**2+\
        b*samples[2,:]**4*np.sin(samples[0,:])
    return vals[:,np.newaxis]

def ishigami_function_jacobian(samples,a=7,b=0.1):
    assert samples.shape[1]==1
    nvars=3
    assert samples.shape[0]==nvars
    jac = np.empty((nvars,1))
    jac[0] = np.cos(samples[0,:]) + b*samples[2,:]**4*np.cos(samples[0,:])
    jac[1] = 2*a*np.sin(samples[1,:])*cos(samples[1,:])
    jac[2] = 4*b*samples[2,:]**3*np.sin(samples[0,:])
    return jac

def get_ishigami_funciton_statistics(a=7,b=0.1):
    """
    p_i(X_i) ~ U[-pi,pi]
    """
    mean = a/2
    variance = a**2/8+b*np.pi**4/5+b**2*np.pi**8/18+0.5
    D_1 = b*np.pi**4/5+b**2*np.pi**8/50+0.5
    D_2,D_3,D_12,D_13 = a**2/8,0,0,b**2*np.pi**8/18-b**2*np.pi**8/50
    D_23,D_123=0,0
    main_effects =  np.array([D_1,D_2,D_3])/variance
    # the following two ways of calulating the total effects are equivalent
    total_effects1 = np.array(
        [D_1+D_12+D_13+D_123,D_2+D_12+D_23+D_123,D_3+D_13+D_23+D_123])/variance
    total_effects=1-np.array([D_2+D_3+D_23,D_1+D_3+D_13,D_1+D_2+D_12])/variance
    assert np.allclose(total_effects1,total_effects)
    sobol_indices = np.array([D_1,D_2,D_3,D_12,D_13,D_23,D_123])/variance
    return mean, variance, main_effects, total_effects, sobol_indices

def sobol_g_function(coefficients,samples):
    """
    The coefficients control the sensitivity of each variable. Specifically
    they limit the range of the outputs, i.e.
    1-1/(1+a_i) <= (abs(4*x-2)+a_i)/(a_i+1) <= 1-1/(1+a_i)
    """
    nvars,nsamples = samples.shape
    assert coefficients.shape[0]==nvars
    vals = np.prod((np.absolute(4*samples-2)+coefficients[:,np.newaxis])/
                   (1+coefficients[:,np.newaxis]),axis=0)[:,np.newaxis]
    assert vals.shape[0]==nsamples
    return vals

def get_sobol_g_function_statistics(a,interaction_terms=None):
    """
    See article: Variance based sensitivity analysis of model output. 
    Design and estimator for the total sensitivity index
    """
    nvars = a.shape[0]
    mean = 1
    unnormalized_main_effects = 1/(3*(1+a)**2)
    variance = np.prod(unnormalized_main_effects+1)-1
    main_effects = unnormalized_main_effects/variance
    total_effects = np.tile(np.prod(unnormalized_main_effects+1),(nvars))
    total_effects *= unnormalized_main_effects/(unnormalized_main_effects+1)
    total_effects /= variance
    if interaction_terms is None:
        return mean,variance,main_effects,total_effects
    
    sobol_indices = np.array([
        unnormalized_main_effects[index].prod()/variance
        for index in interaction_terms])
    return mean,variance,main_effects,total_effects,sobol_indices

def morris_function(samples):
    assert samples.shape[0]==20
    beta0 = np.random.randn()
    beta_first_order = np.empty(20)
    beta_first_order[:10]=20
    beta_first_order[10:] = np.random.normal(0,1,10)
    beta_second_order = np.empty((20,20))
    beta_second_order[:6,:6]=-15
    beta_second_order[6:,6:] = np.random.normal(0,1,(14,14))
    #beta_third_order = np.zeros((20,20,20))
    #beta_third_order[:5,:5,:5]=-10
    beta_third_order=-10
    #beta_forth_order = np.zeros((20,20,20,20))
    #beta_forth_order[:4,:4,:4,:4]=5
    beta_forth_order = 5
    ww = 2*(samples-0.5)
    I = [3,5,7]
    ww[I]=2 * (1.1 * samples[I]/(samples[I]+0.1)-0.5)

    values = beta0
    values += np.sum(beta_first_order[:,np.newaxis]*ww,axis=0)

    for jj in range(20):
        for ii in range(jj):
            values += beta_second_order[ii,jj]*ww[ii]*ww[jj]

    for kk in range(5):
        for jj in range(kk):
            for ii in range(jj):
                values += beta_third_order*ww[ii]*ww[jj]*ww[kk]

    for ll in range(4):
        for kk in range(ll):
            for jj in range(kk):
                for ii in range(jj):
                    values += beta_forth_order*ww[ii]*ww[jj]*ww[kk]*ww[ll]
    return values[:,np.newaxis]

class TestSensitivityAnalysis(unittest.TestCase):
    def test_get_sobol_indices_from_pce(self):
        num_vars = 5; degree = 5
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        coefficients = np.ones((indices.shape[1],2),float)
        coefficients[:,1]*=2
        interaction_indices, interaction_values = \
            get_sobol_indices(
                coefficients,indices,max_order=num_vars)
        assert np.allclose(
            interaction_values.sum(axis=0), np.ones(2))

    def test_pce_sensitivities_of_ishigami_function(self):
        nsamples=1500
        nvars,degree = 3,18
        univariate_variables = [uniform(-np.pi,2*np.pi)]*nvars
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)

        var_trans = pya.AffineRandomVariableTransformation(variable)
        poly = pya.PolynomialChaosExpansion()
        poly_opts = pya.define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)
        indices = pya.compute_hyperbolic_indices(nvars,degree,1.0)
        poly.set_indices(indices)
        #print('No. PCE Terms',indices.shape[1])

        samples = pya.generate_independent_random_samples(
            var_trans.variable,nsamples)
        values = ishigami_function(samples)
        
        basis_matrix = poly.basis_matrix(samples)
        coef = np.linalg.lstsq(basis_matrix,values,rcond=None)[0]
        poly.set_coefficients(coef)

        nvalidation_samples = 1000
        validation_samples = pya.generate_independent_random_samples(
            var_trans.variable,nvalidation_samples)
        validation_values = ishigami_function(validation_samples)
        poly_validation_vals = poly(validation_samples)
        abs_error = np.linalg.norm(
            poly_validation_vals-validation_values)/np.sqrt(nvalidation_samples)
        #print('Abs. Error',abs_error)

        pce_main_effects,pce_total_effects=\
            pya.get_main_and_total_effect_indices_from_pce(
                poly.get_coefficients(),poly.get_indices())
        
        mean, variance, main_effects, total_effects, sobol_indices = \
            get_ishigami_funciton_statistics()
        assert np.allclose(poly.mean(),mean)
        assert np.allclose(poly.variance(),variance)
        assert np.allclose(pce_main_effects[:,0],main_effects)
        assert np.allclose(pce_total_effects[:,0],total_effects)

        interaction_terms, pce_sobol_indices = get_sobol_indices(
            poly.get_coefficients(),poly.get_indices(),max_order=3)
        assert np.allclose(pce_sobol_indices[:,0],sobol_indices)

    def test_pce_sensitivities_of_sobol_g_function(self):
        nsamples=2000
        nvars,degree = 3,8
        a = np.array([1,2,5])[:nvars]
        univariate_variables = [uniform(0,1)]*nvars
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)

        var_trans = pya.AffineRandomVariableTransformation(variable)
        poly = pya.PolynomialChaosExpansion()
        poly_opts = pya.define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)
        indices = pya.tensor_product_indices([degree]*nvars)
        poly.set_indices(indices)
        #print('No. PCE Terms',indices.shape[1])

        samples = pya.generate_independent_random_samples(
            var_trans.variable,nsamples)
        samples = (np.cos(np.random.uniform(0,np.pi,(nvars,nsamples)))+1)/2
        values = sobol_g_function(a,samples)
        
        basis_matrix = poly.basis_matrix(samples)
        weights = 1/np.sum(basis_matrix**2,axis=1)[:,np.newaxis]
        coef=np.linalg.lstsq(basis_matrix*weights,values*weights,rcond=None)[0]
        poly.set_coefficients(coef)

        nvalidation_samples = 1000
        validation_samples = pya.generate_independent_random_samples(
            var_trans.variable,nvalidation_samples)
        validation_values = sobol_g_function(a,validation_samples)

        poly_validation_vals = poly(validation_samples)
        rel_error = np.linalg.norm(
            poly_validation_vals-validation_values)/np.linalg.norm(
                validation_values)
        print('Rel. Error',rel_error)

        pce_main_effects,pce_total_effects=\
            pya.get_main_and_total_effect_indices_from_pce(
                poly.get_coefficients(),poly.get_indices())
        interaction_terms, pce_sobol_indices = get_sobol_indices(
            poly.get_coefficients(),poly.get_indices(),max_order=3)
        
        mean, variance, main_effects, total_effects, sobol_indices = \
            get_sobol_g_function_statistics(a, interaction_terms)
        assert np.allclose(poly.mean(),mean,atol=1e-2)
        #print((poly.variance(),variance))
        assert np.allclose(poly.variance(),variance,atol=1e-2)
        #print(pce_main_effects[:,0],main_effects)
        assert np.allclose(pce_main_effects[:,0],main_effects,atol=1e-2)
        #print(pce_total_effects[:,0],total_effects)
        assert np.allclose(pce_total_effects[:,0],total_effects,atol=1e-2)
        assert np.allclose(pce_sobol_indices[:,0],sobol_indices,atol=1e-2)

    def test_get_sobol_indices_from_pce_max_order(self):
        num_vars = 3; degree = 4; max_order=2
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        coefficients = np.ones((indices.shape[1],2),float)
        coefficients[:,1]*=2
        interaction_indices, interaction_values = \
            get_sobol_indices(coefficients,indices,max_order)

        assert len(interaction_indices)==6
        true_interaction_indices = [[0],[1],[2],[0,1],[0,2],[1,2]]
        for ii in range(len(interaction_indices)):
            assert np.allclose(
                true_interaction_indices[ii],interaction_indices[ii])
        
        true_variance = np.asarray(
            [indices.shape[1]-1,2**2*(indices.shape[1]-1)])

        # get the number of interactions involving variables 0 and 1
        # test problem is symmetric so number is the same for all variables
        num_pairwise_interactions = np.where(
            np.all(indices[0:2,:]>0,axis=0)&(indices[2,:]==0))[0].shape[0]
        I = np.where(np.all(indices[0:2,:]>0,axis=0))[0]
        
        true_interaction_values = np.vstack((
            np.tile(np.arange(1,3)[np.newaxis,:],
                    (num_vars,1))**2*degree/true_variance,
            np.tile(np.arange(1,3)[np.newaxis,:],
            (num_vars,1))**2*num_pairwise_interactions/true_variance))

        assert np.allclose(true_interaction_values,interaction_values)

        #plot_interaction_values( interaction_values, interaction_indices)

    def test_get_main_and_total_effect_indices_from_pce(self):
        num_vars = 3; degree = num_vars; max_order=2
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        coefficients = np.ones((indices.shape[1],2),float)
        coefficients[:,1]*=2
        main_effects, total_effects = \
            get_main_and_total_effect_indices_from_pce(coefficients,indices)
        true_variance = np.asarray(
            [indices.shape[1]-1,2**2*(indices.shape[1]-1)])
        true_main_effects = np.tile(
            np.arange(1,3)[np.newaxis,:],
            (num_vars,1))**2*degree/true_variance
        assert np.allclose(main_effects,true_main_effects)

        # get the number of interactions variable 0 is involved in
        # test problem is symmetric so number is the same for all variables
        num_interactions_per_variable = np.where(indices[0,:]>0)[0].shape[0]
        true_total_effects = np.tile(
            np.arange(1,3)[np.newaxis,:],
            (num_vars,1))**2*num_interactions_per_variable/true_variance
        assert np.allclose(true_total_effects,total_effects)

        #plot_total_effects(total_effects)
        #plot_main_effects(main_effects)

    def test_morris_elementary_effects(self):
        nvars = 20
        from functools import partial
        function = morris_function

        nvars = 6
        coefficients = np.array([78,12,0.5,2,97,33])
        function = partial(sobol_g_function,coefficients)
        
        nlevels,ncandidate_trajectories,ntrajectories = 4,40,4

        candidate_samples = get_morris_samples(
            nvars,nlevels,ncandidate_trajectories)

        samples=downselect_morris_trajectories(candidate_samples,ntrajectories)

        
        values = function(samples)
        elem_effects = get_morris_elementary_effects(samples,values)
        mu,sigma = get_morris_sensitivity_indices(elem_effects)
        print_morris_sensitivity_indices(mu,sigma)
        # ix1 = 0
        # for ii in range(ntrajectories):
        #     ix2 = ix1+nvars+1
        #     plt.plot(samples[0,ix1:ix2],samples[1,ix1:ix2],'-o')
        #     ix1=ix2
        # plt.xlim([0,1]); plt.ylim([0,1]); plt.show()
        
    def test_analyze_sensitivity_sparse_grid(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.adaptive_sparse_grid import isotropic_refinement_indicator
        benchmark = setup_benchmark("oakley")
        options = {'max_nsamples':2000}
        #'refinement_indicator':isotropic_refinement_indicator}
        res = adaptive_analyze_sensitivity(
            benchmark.fun,benchmark.variable.all_variables(),"sparse_grid",
            options=options)

        #print(res.main_effects-benchmark.main_effects)
        assert np.allclose(res.main_effects,benchmark.main_effects,atol=2e-4)


    def test_analyze_sensitivity_polynomial_chaos(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.adaptive_sparse_grid import isotropic_refinement_indicator
        benchmark = setup_benchmark("ishigami",a=7,b=0.1)
        options = {
            'approx_options':{'max_nsamples':800,'verbose':0}}
        res = adaptive_analyze_sensitivity(
            benchmark.fun,benchmark.variable.all_variables(),"polynomial_chaos",
            options=options)

        from pyapprox.approximate import compute_l2_error
        nsamples = 100
        approx=res['approx']
        error = compute_l2_error(
            approx,benchmark.fun,approx.var_trans.variable,
            nsamples)
        assert error<1e-12


        print(res.main_effects-benchmark.main_effects)
        assert np.allclose(res.main_effects,benchmark.main_effects,atol=2e-4)

    
if __name__== "__main__":    
    sensitivity_analysis_test_suite=unittest.TestLoader().loadTestsFromTestCase(
        TestSensitivityAnalysis)
    unittest.TextTestRunner(verbosity=2).run(sensitivity_analysis_test_suite)


