import unittest
from functools import partial
from pyapprox.polynomial_sampling import *
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
from pyapprox.variable_transformations import \
     define_iid_random_variable_transformation, RosenblattTransformation, \
     TransformationComposition
from pyapprox.indexing import compute_hyperbolic_indices
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D,\
    clenshaw_curtis_pts_wts_1D
from pyapprox.models.genz import GenzFunction
from scipy.stats import beta as beta, uniform
from pyapprox.density import tensor_product_pdf
from pyapprox.configure_plots import *
from pyapprox.utilities import get_tensor_product_quadrature_rule
from pyapprox.tests.test_rosenblatt_transformation import rosenblatt_example_2d
class TestPolynomialSampling(unittest.TestCase):

    def test_christoffel_function(self):
        num_vars=1
        degree=2
        alpha_poly= 0
        beta_poly=0
        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            uniform(-1,2),num_vars) 
        poly.configure(
            {'alpha_poly':alpha_poly,'beta_poly':beta_poly,
             'var_trans':var_trans})
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)

        num_samples = 11
        samples = np.linspace(-1.,1.,num_samples)[np.newaxis,:]
        basis_matrix = poly.basis_matrix(samples)
        true_weights=1./np.linalg.norm(basis_matrix,axis=1)**2
        weights = 1./christoffel_function(samples,poly.basis_matrix)
        assert weights.shape[0]==num_samples
        assert np.allclose(true_weights,weights)

        # For a Gaussian quadrature rule of degree p that exactly
        # integrates all polynomials up to and including degree 2p-1
        # the quadrature weights are the christoffel function
        # evaluated at the quadrature samples
        quad_samples,quad_weights = gauss_jacobi_pts_wts_1D(
            degree,alpha_poly,beta_poly)
        quad_samples = quad_samples[np.newaxis,:]
        basis_matrix = poly.basis_matrix(quad_samples)
        weights = 1./christoffel_function(quad_samples,poly.basis_matrix)
        assert np.allclose(weights,quad_weights)

    def test_fekete_gauss_lobatto(self):
        num_vars=1
        degree=3
        num_candidate_samples = 10000
        generate_candidate_samples=lambda n: np.linspace(-1.,1.,n)[np.newaxis,:]

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            uniform(-1,2),num_vars) 
        poly.configure({'alpha_poly':0.,'beta_poly':0.,'var_trans':var_trans})
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)

        precond_func = lambda matrix, samples: 0.25*np.ones(matrix.shape[0])
        samples,_ = get_fekete_samples(
            poly.basis_matrix,generate_candidate_samples,
            num_candidate_samples,preconditioning_function=precond_func)
        assert samples.shape[1]==degree+1

        # The samples should be close to the Gauss-Lobatto samples
        gauss_lobatto_samples =  np.asarray(
            [-1.0, - 0.447213595499957939281834733746,
             0.447213595499957939281834733746, 1.0 ])
        assert np.allclose(np.sort(samples),gauss_lobatto_samples,atol=1e-1)

    def test_fekete_interpolation(self):
        num_vars=2
        degree=15

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            uniform(),num_vars) 
        poly.configure({'alpha_poly':0.,'beta_poly':0.,'var_trans':var_trans})
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)


        # candidates must be generated in canonical PCE space
        num_candidate_samples = 10000
        generate_candidate_samples=lambda n: np.cos(
            np.random.uniform(0.,np.pi,(num_vars,n)))

        # must use canonical_basis_matrix to generate basis matrix
        precond_func = lambda matrix, samples: christoffel_weights(matrix)
        samples, data_structures = get_fekete_samples(
            poly.canonical_basis_matrix,generate_candidate_samples,
            num_candidate_samples,preconditioning_function=precond_func)
        samples = var_trans.map_from_canonical_space(samples)

        assert samples.max()<=1 and samples.min()>=0.

        c = np.random.uniform(0.,1.,num_vars)
        c*=20/c.sum()
        w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
        genz_function = GenzFunction('oscillatory',num_vars,c=c,w=w)
        values = genz_function(samples)
        
        # Ensure coef produce an interpolant
        coef = interpolate_fekete_samples(samples,values,data_structures)
        poly.set_coefficients(coef)
        assert np.allclose(poly(samples),values)

        quad_w = get_quadrature_weights_from_fekete_samples(
            samples,data_structures)
        values_at_quad_x = values[:,0]
        # increase degree if want smaller atol
        assert np.allclose(
            np.dot(values_at_quad_x,quad_w),genz_function.integrate(),
            atol=1e-4)

    def test_lu_leja_interpolation(self):
        num_vars=2
        degree=15
        
        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            uniform(),num_vars) 
        poly.configure({'alpha_poly':0.,'beta_poly':0.,'var_trans':var_trans})
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)

        # candidates must be generated in canonical PCE space
        num_candidate_samples = 10000
        generate_candidate_samples=lambda n: np.cos(
            np.random.uniform(0.,np.pi,(num_vars,n)))

        # must use canonical_basis_matrix to generate basis matrix
        num_leja_samples = indices.shape[1]-1
        precond_func = lambda matrix, samples: christoffel_weights(matrix)
        samples, data_structures = get_lu_leja_samples(
            poly.canonical_basis_matrix,generate_candidate_samples,
            num_candidate_samples,num_leja_samples,
            preconditioning_function=precond_func)
        samples = var_trans.map_from_canonical_space(samples)

        assert samples.max()<=1 and samples.min()>=0.

        c = np.random.uniform(0.,1.,num_vars)
        c*=20/c.sum()
        w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
        genz_function = GenzFunction('oscillatory',num_vars,c=c,w=w)
        values = genz_function(samples)
        
        # Ensure coef produce an interpolant
        coef = interpolate_lu_leja_samples(samples,values,data_structures)
        
        # Ignore basis functions (columns) that were not considered during the
        # incomplete LU factorization
        poly.set_indices(poly.indices[:,:num_leja_samples])
        poly.set_coefficients(coef)

        assert np.allclose(poly(samples),values)

        quad_w = get_quadrature_weights_from_lu_leja_samples(
            samples,data_structures)
        values_at_quad_x = values[:,0]

        # will get closer if degree is increased
        # print (np.dot(values_at_quad_x,quad_w),genz_function.integrate())
        assert np.allclose(
            np.dot(values_at_quad_x,quad_w),genz_function.integrate(),
            atol=1e-4)


    def test_lu_leja_interpolation_with_intial_samples(self):
        num_vars=2
        degree=15
        
        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            uniform(),num_vars) 
        poly.configure({'alpha_poly':0.,'beta_poly':0.,'var_trans':var_trans})
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)

        # candidates must be generated in canonical PCE space
        num_candidate_samples = 10000
        generate_candidate_samples=lambda n: np.cos(
            np.random.uniform(0.,np.pi,(num_vars,n)))

        # enforcing lu interpolation to interpolate a set of initial points
        # before selecting best samples from candidates can cause ill conditioning
        # to avoid this issue build a leja sequence and use this as initial
        # samples and then recompute sequence with different candidates
        
        # must use canonical_basis_matrix to generate basis matrix
        num_initial_samples = 5
        initial_samples = None
        num_leja_samples = indices.shape[1]-1
        precond_func = lambda matrix, samples: christoffel_weights(matrix)
        initial_samples, data_structures = get_lu_leja_samples(
            poly.canonical_basis_matrix,generate_candidate_samples,
            num_candidate_samples,num_initial_samples,
            preconditioning_function=precond_func,
            initial_samples=initial_samples)

        samples, data_structures = get_lu_leja_samples(
            poly.canonical_basis_matrix,generate_candidate_samples,
            num_candidate_samples,num_leja_samples,
            preconditioning_function=precond_func,
            initial_samples=initial_samples)

        assert np.allclose(samples[:,:num_initial_samples],initial_samples)
        
        samples = var_trans.map_from_canonical_space(samples)

        assert samples.max()<=1 and samples.min()>=0.

        c = np.random.uniform(0.,1.,num_vars)
        c*=20/c.sum()
        w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
        genz_function = GenzFunction('oscillatory',num_vars,c=c,w=w)
        values = genz_function(samples)
        
        # Ensure coef produce an interpolant
        coef = interpolate_lu_leja_samples(samples,values,data_structures)
        
        # Ignore basis functions (columns) that were not considered during the
        # incomplete LU factorization
        poly.set_indices(poly.indices[:,:num_leja_samples])
        poly.set_coefficients(coef)

        assert np.allclose(poly(samples),values)

        quad_w = get_quadrature_weights_from_lu_leja_samples(
            samples,data_structures)
        values_at_quad_x = values[:,0]
        assert np.allclose(
            np.dot(values_at_quad_x,quad_w),genz_function.integrate(),
            atol=1e-4)

    def test_oli_leja_interpolation(self):
        num_vars=2
        degree=5
        
        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            uniform(),num_vars) 
        poly.configure({'alpha_poly':0.,'beta_poly':0.,'var_trans':var_trans})
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)

        # candidates must be generated in canonical PCE space
        num_candidate_samples = 10000
        generate_candidate_samples=lambda n: np.cos(
            np.random.uniform(0.,np.pi,(num_vars,n)))
        generate_candidate_samples=lambda n: (np.cos(
            np.random.uniform(0.,np.pi,(num_vars,n)))+1)/2.

        # must use canonical_basis_matrix to generate basis matrix
        num_leja_samples = indices.shape[1]-1
        precond_func = lambda samples: 1./christoffel_function(
            samples,poly.basis_matrix)
        samples, data_structures = get_oli_leja_samples(
            poly,generate_candidate_samples,
            num_candidate_samples,num_leja_samples,
            preconditioning_function=precond_func)
        #samples = var_trans.map_from_canonical_space(samples)

        assert samples.max()<=1 and samples.min()>=0.

        c = np.random.uniform(0.,1.,num_vars)
        c*=20/c.sum()
        w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
        genz_function = GenzFunction('oscillatory',num_vars,c=c,w=w)
        values = genz_function(samples)
        
        # Ensure we have produced an interpolant
        oli_solver = data_structures[0]
        poly = oli_solver.get_current_interpolant(samples,values)
        assert np.allclose(poly(samples),values)

        # quad_w = get_quadrature_weights_from_lu_leja_samples(
        #     samples,data_structures)
        # values_at_quad_x = values[:,0]
        # assert np.allclose(
        #     np.dot(values_at_quad_x,quad_w),genz_function.integrate())
        
    def test_random_christoffel_sampling(self):
        num_vars = 2
        degree = 20

        alpha_poly=1
        beta_poly=1

        alpha_stat = beta_poly+1
        beta_stat  = alpha_poly+1

        num_samples = 1000
        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            uniform(),num_vars) 
        poly.configure(
            {'alpha_poly':alpha_poly,'beta_poly':beta_poly,
            'var_trans':var_trans})
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)

        univariate_pdf = partial(beta.pdf,a=alpha_stat,b=beta_stat)
        probability_density = partial(
            tensor_product_pdf, univariate_pdfs=univariate_pdf)

        envelope_factor = 4e3
        generate_proposal_samples=lambda n : np.random.uniform(
            0.,1.,size=(num_vars,n))
        proposal_density = lambda x: np.ones(x.shape[1])

        # unlike fekete and leja sampling can and should use
        # pce.basis_matrix here. If use canonical_basis_matrix then
        # densities must be mapped to this space also which can be difficult
        samples = random_induced_measure_sampling(
            num_samples,num_vars,poly.basis_matrix, probability_density,
            proposal_density, generate_proposal_samples,envelope_factor)

        basis_matrix = poly.basis_matrix(samples)
        weights = np.sqrt(christoffel_weights(basis_matrix))
        #print np.linalg.cond((basis_matrix.T*weights).T)

        # plt.plot(samples[0,:],samples[1,:],'o')
        # beta_samples = np.random.beta(
        #     alpha_stat,beta_stat,(num_vars,num_samples))
        # basis_matrix = poly.basis_matrix(beta_samples)
        # print np.linalg.cond(basis_matrix)
        
        # plt.plot(beta_samples[0,:],beta_samples[1,:],'s')
        # plt.show()

    def test_fekete_rosenblatt_interpolation(self):
        np.random.seed(2)
        degree=3

        __,__,joint_density,limits = rosenblatt_example_2d(num_samples=1)
        num_vars=len(limits)//2

        rosenblatt_opts = {'limits':limits,'num_quad_samples_1d':20}
        var_trans_1 = RosenblattTransformation(
            joint_density,num_vars,rosenblatt_opts)
        # rosenblatt maps to [0,1] but polynomials of bounded variables
        # are in [-1,1] so add second transformation for this second mapping
        var_trans_2 = define_iid_random_variable_transformation(
            uniform(),num_vars)
        var_trans = TransformationComposition([var_trans_1, var_trans_2])

        poly = PolynomialChaosExpansion()
        poly.configure({'alpha_poly':0.,'beta_poly':0.,'var_trans':var_trans})
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)
        
        num_candidate_samples = 10000
        generate_candidate_samples=lambda n: np.cos(
            np.random.uniform(0.,np.pi,(num_vars,n)))

        precond_func = lambda matrix, samples: christoffel_weights(matrix)
        canonical_samples, data_structures = get_fekete_samples(
            poly.canonical_basis_matrix,generate_candidate_samples,
            num_candidate_samples,preconditioning_function=precond_func)
        samples = var_trans.map_from_canonical_space(canonical_samples)
        assert np.allclose(
            canonical_samples,var_trans.map_to_canonical_space(samples))

        assert samples.max()<=1 and samples.min()>=0.

        c = np.random.uniform(0.,1.,num_vars)
        c*=20/c.sum()
        w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
        genz_function = GenzFunction('oscillatory',num_vars,c=c,w=w)
        values = genz_function(samples)
        # function = lambda x: np.sum(x**2,axis=0)[:,np.newaxis]
        # values = function(samples)
        
        # Ensure coef produce an interpolant
        coef = interpolate_fekete_samples(
            canonical_samples,values,data_structures)
        poly.set_coefficients(coef)
        
        assert np.allclose(poly(samples),values)

        # compare mean computed using quadrature and mean computed using
        # first coefficient of expansion. This is not testing that mean
        # is correct because rosenblatt transformation introduces large error
        # which makes it hard to compute accurate mean from pce or quadrature
        quad_w = get_quadrature_weights_from_fekete_samples(
            canonical_samples,data_structures)
        values_at_quad_x = values[:,0]
        assert np.allclose(
            np.dot(values_at_quad_x,quad_w),poly.mean())
        

if __name__== "__main__":    
    polynomial_sampling_test_suite=unittest.TestLoader().loadTestsFromTestCase(
         TestPolynomialSampling)
    unittest.TextTestRunner(verbosity=2).run(polynomial_sampling_test_suite)
