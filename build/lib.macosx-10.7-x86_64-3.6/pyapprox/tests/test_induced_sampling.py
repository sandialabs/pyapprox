import unittest
from pyapprox.induced_sampling import *
from pyapprox.indexing import compute_hyperbolic_indices, \
    compute_hyperbolic_level_indices
from pyapprox.variables import float_rv_discrete, \
    IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, \
    define_poly_options_from_variable_transformation
from scipy.stats import beta
from pyapprox.configure_plots import *
class TestInducedSampling(unittest.TestCase):

    def test_continous_induced_measure_ppf(self):
        degree=2
        alpha_stat,beta_stat=3,3
        ab = jacobi_recurrence(
            degree+1,alpha=beta_stat-1,beta=alpha_stat-1,probability=True)

        tol=1e-15
        var = beta(alpha_stat,beta_stat,-1,2)
        lb,ub = var.support()
        x = np.linspace(lb,ub,101)
        pdf = lambda xx: var.dist._pdf((xx+1)/2,a=alpha_stat,b=beta_stat)/2
        cdf_vals = continuous_induced_measure_cdf(pdf,ab,degree,lb,ub,tol,x)
        assert np.all(cdf_vals<=1.0)
        ppf_vals = continuous_induced_measure_ppf(
            var,ab,degree,cdf_vals,1e-10,1e-8)
        # differences caused by root finding optimization tolerance
        assert np.allclose(x,ppf_vals)
        #plt.plot(x,cdf_vals)
        #plt.plot(ppf_vals,cdf_vals,'r*',ms=2)
        #plt.show()
        
    def test_discrete_induced_sampling(self):
        degree=2
        
        nmasses1=10
        mass_locations1 = np.geomspace(1.0, 512.0, num=nmasses1)
        masses1 = np.ones(nmasses1,dtype=float)/nmasses1
        var1 = float_rv_discrete(
            name='float_rv_discrete',values=(mass_locations1,masses1))()

        nmasses2=10
        mass_locations2 = np.arange(0,nmasses2)
        # if increase from 16 unmodififed becomes ill conditioned
        masses2  = np.geomspace(1.0, 16.0, num=nmasses2)
        masses2 /= masses2.sum()
        var2 = float_rv_discrete(
            name='float_rv_discrete',values=(mass_locations2,masses2))()

        var_trans = AffineRandomVariableTransformation([var1,var2])
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(pce.num_vars(), degree, 1.0)
        pce.set_indices(indices)

        num_samples = 100
        np.random.seed(1)
        samples = generate_induced_samples(pce,num_samples)

        # np.random.seed(1)
        # basis_matrix_generator = partial(basis_matrix_generator_1d,pce,degree)
        # samples1 = discrete_induced_sampling(
        #     basis_matrix_generator,pce.indices,
        #     [np.asarray(var1.dist.xk,dtype=float),
        #      np.asarray(var2.dist.xk,dtype=float)],
        #     [var1.dist.pk,var2.dist.pk],num_samples)

        #import matplotlib.pyplot as plt
        #plt.plot(samples[0,:],samples[1,:],'o')
        #plt.plot(samples1[0,:],samples1[1,:],'*')
        #plt.show()
        
        msg = 'Compute the reference by integrating induced density'
        raise Exception(msg)

    def test_multivariate_sampling_jacobi(self):
    
        num_vars = 2; degree=2
        alph=1; bet=1.
        univ_inv = partial(idistinv_jacobi,alph=alph, bet=bet)
        num_samples = 10
        indices = np.ones((2,num_samples),dtype=int)*degree
        indices[1,:] = degree-1
        xx = np.tile(
            np.linspace(0.01,0.99,(num_samples))[np.newaxis,:],(num_vars,1))
        samples = univ_inv(xx, indices)
        
        var_trans = AffineRandomVariableTransformation(
            [beta(bet+1,alph+1,-1,2),beta(bet+1,alph+1,-1,2)])
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        pce.set_indices(indices)

        reference_samples = inverse_transform_sampling_1d(
            pce.var_trans.variable.unique_variables[0],pce.recursion_coeffs[0],
            degree,xx[0,:])
        # differences are just caused by different tolerances in optimizes
        # used to find roots of CDF
        assert np.allclose(reference_samples,samples[0,:],atol=1e-7)
        reference_samples = inverse_transform_sampling_1d(
            pce.var_trans.variable.unique_variables[0],pce.recursion_coeffs[0],
            degree-1,xx[0,:])
        assert np.allclose(reference_samples,samples[1,:],atol=1e-7)

        #num_samples = 30
        #samples = generate_induced_samples(pce,num_samples)
        #plt.plot(samples[0,:],samples[1,:],'o'); plt.show()

    def test_multivariate_migliorati_sampling_jacobi(self):
    
        num_vars = 1; degree=20
        alph=5; bet=5.
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        
        var_trans = AffineRandomVariableTransformation(
            IndependentMultivariateRandomVariable(
                [beta(alph,bet,-1,2)],[np.arange(num_vars)]))
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        pce.set_indices(indices)

        cond_tol = 1e1
        samples = generate_induced_samples_migliorati_tolerance(pce,cond_tol)
        cond = compute_preconditioned_basis_matrix_condition_number(
            pce.canonical_basis_matrix,samples)
        assert cond<cond_tol
        #plt.plot(samples[0,:],samples[1,:],'o'); plt.show()

    def test_adaptive_multivariate_sampling_jacobi(self):
    
        num_vars = 2; degree=6
        alph=5; bet=5.
        
        var_trans = AffineRandomVariableTransformation(
            IndependentMultivariateRandomVariable(
                [beta(alph,bet,-1,3)],[np.arange(num_vars)]))
        pce_opts = define_poly_options_from_variable_transformation(var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars,1,1.0)
        pce.set_indices(indices)
        cond_tol = 1e2
        samples = generate_induced_samples_migliorati_tolerance(pce,cond_tol)

        for dd in range(2,degree):
            num_prev_samples = samples.shape[1]
            new_indices = compute_hyperbolic_level_indices(num_vars, dd, 1.)
            samples = increment_induced_samples_migliorati(
                pce,cond_tol,samples,indices,new_indices)
            indices = np.hstack((indices,new_indices))
            pce.set_indices(indices)
            new_samples = samples[:,num_prev_samples:]
            prev_samples = samples[:,:num_prev_samples]
            #fig,axs = plt.subplots(1,2,figsize=(2*8,6))
            #from pyapprox.visualization import plot_2d_indices
            #axs[0].plot(prev_samples[0,:],prev_samples[1,:],'ko');
            #axs[0].plot(new_samples[0,:],new_samples[1,:],'ro');
            #plot_2d_indices(indices,other_indices=new_indices,ax=axs[1]);
            #plt.show()

        samples = var_trans.map_from_canonical_space(samples)
        cond = compute_preconditioned_basis_matrix_condition_number(
            pce.basis_matrix,samples)
        assert cond<cond_tol
            
    
if __name__== "__main__":    
    induced_sampling_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestInducedSampling)
    unittest.TextTestRunner(verbosity=2).run(induced_sampling_test_suite)

