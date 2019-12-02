import unittest
from scipy import special as sp
from pyapprox.multivariate_polynomials import *
from pyapprox.univariate_quadrature import gauss_hermite_pts_wts_1D, \
    gauss_jacobi_pts_wts_1D
from pyapprox.utilities import get_tensor_product_quadrature_rule, approx_fprime
from pyapprox.variable_transformations import \
     define_iid_random_variable_transformation, IdentityTransformation,\
     AffineRandomVariableTransformation
from pyapprox.density import map_to_canonical_gaussian
from pyapprox.variables import IndependentMultivariateRandomVariable,\
    float_rv_discrete
from functools import partial
from pyapprox.indexing import sort_indices_lexiographically
from scipy.stats import uniform, beta, norm, hypergeom, binom
class TestMultivariatePolynomials(unittest.TestCase):

    def test_evaluate_multivariate_orthonormal_polynomial(self):
        num_vars = 2; alpha = 0.; beta = 0.; degree = 2; deriv_order=1    
        probability_measure = True

        ab = jacobi_recurrence(
            degree+1,alpha=alpha,beta=beta,probability=probability_measure)

        x,w=np.polynomial.legendre.leggauss(degree)
        samples = cartesian_product([x]*num_vars,1)
        weights = outer_product([w]*num_vars)

        indices = compute_hyperbolic_indices(num_vars,degree,1.0)

        # sort lexographically to make testing easier
        I = np.lexsort((indices[0,:],indices[1,:], indices.sum(axis=0)))
        indices = indices[:,I]

        basis_matrix = evaluate_multivariate_orthonormal_polynomial(
            samples,indices,ab,deriv_order)

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd,:]
            exact_basis_vals_1d.append(
                np.asarray([1+0.*x,x,0.5*(3.*x**2-1)]).T)
            exact_basis_derivs_1d.append(np.asarray([0.*x,1.0+0.*x,3.*x]).T)
            exact_basis_vals_1d[-1]/=np.sqrt(1./(2*np.arange(degree+1)+1))
            exact_basis_derivs_1d[-1]/=np.sqrt(1./(2*np.arange(degree+1)+1))

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:,0],exact_basis_vals_1d[0][:,1],
                 exact_basis_vals_1d[1][:,1],exact_basis_vals_1d[0][:,2],
            exact_basis_vals_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
            exact_basis_vals_1d[1][:,2]]).T

        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,exact_basis_derivs_1d[0][:,1],0.*x,
             exact_basis_derivs_1d[0][:,2],
             exact_basis_derivs_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
             0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,0.*x,exact_basis_derivs_1d[1][:,1],0.*x,
            exact_basis_vals_1d[0][:,1]*exact_basis_derivs_1d[1][:,1],
            exact_basis_derivs_1d[1][:,2]]).T))

        func = lambda x: evaluate_multivariate_orthonormal_polynomial(
            x,indices,ab,0)
        basis_matrix_derivs = basis_matrix[samples.shape[1]:]
        basis_matrix_derivs_fd = np.empty_like(basis_matrix_derivs)
        for ii in range(samples.shape[1]):
            basis_matrix_derivs_fd[ii::samples.shape[1],:] = approx_fprime(
                samples[:,ii:ii+1],func,1e-7)
        assert np.allclose(
            exact_basis_matrix[samples.shape[1]:], basis_matrix_derivs_fd)

        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_jacobi_pce(self):
        num_vars = 2; alpha = 0.; beta = 0.; degree = 2; deriv_order=1    
        probability_measure = True

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            uniform(-1,2),num_vars) 
        poly.configure({'poly_type':'legendre','var_trans':var_trans})

        samples, weights = get_tensor_product_quadrature_rule(
            degree,num_vars,np.polynomial.legendre.leggauss)

        indices = compute_hyperbolic_indices(num_vars,degree,1.0)

        # sort lexographically to make testing easier
        I = np.lexsort((indices[0,:],indices[1,:], indices.sum(axis=0)))
        indices = indices[:,I]
        poly.set_indices(indices)

        basis_matrix = poly.basis_matrix(samples,{'deriv_order':1})

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd,:]
            exact_basis_vals_1d.append(
                np.asarray([1+0.*x,x,0.5*(3.*x**2-1)]).T)
            exact_basis_derivs_1d.append(np.asarray([0.*x,1.0+0.*x,3.*x]).T)
            exact_basis_vals_1d[-1]/=np.sqrt(1./(2*np.arange(degree+1)+1))
            exact_basis_derivs_1d[-1]/=np.sqrt(1./(2*np.arange(degree+1)+1))

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:,0],exact_basis_vals_1d[0][:,1],
                 exact_basis_vals_1d[1][:,1],exact_basis_vals_1d[0][:,2],
            exact_basis_vals_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
            exact_basis_vals_1d[1][:,2]]).T


        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,exact_basis_derivs_1d[0][:,1],0.*x,
             exact_basis_derivs_1d[0][:,2],
            exact_basis_derivs_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
            0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,0.*x,exact_basis_derivs_1d[1][:,1],0.*x,
            exact_basis_vals_1d[0][:,1]*exact_basis_derivs_1d[1][:,1],
            exact_basis_derivs_1d[1][:,2]]).T))

        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_hermite_pce(self):
        num_vars = 2; degree = 2; deriv_order=1    
        probability_measure = True

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            norm(0,1),num_vars) 
        poly.configure({'poly_type':'hermite','var_trans':var_trans})

        samples, weights = get_tensor_product_quadrature_rule(
            degree+1,num_vars,gauss_hermite_pts_wts_1D)

        indices = compute_hyperbolic_indices(num_vars,degree,1.0)

        # sort lexographically to make testing easier
        I = np.lexsort((indices[0,:],indices[1,:], indices.sum(axis=0)))
        indices = indices[:,I]
        poly.set_indices(indices)

        basis_matrix = poly.basis_matrix(samples,{'deriv_order':1})

        vals_basis_matrix = basis_matrix[:samples.shape[1],:]
        inner_products = (vals_basis_matrix.T*weights).dot(vals_basis_matrix)
        assert np.allclose(inner_products,np.eye(basis_matrix.shape[1]))

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd,:]
            exact_basis_vals_1d.append(
                np.asarray([1+0.*x,x,x**2-1]).T)
            exact_basis_derivs_1d.append(np.asarray([0.*x,1.0+0.*x,2.*x]).T)
            exact_basis_vals_1d[-1]/=np.sqrt(
                sp.factorial(np.arange(degree+1)))
            exact_basis_derivs_1d[-1]/=np.sqrt(
                sp.factorial(np.arange(degree+1)))

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:,0],exact_basis_vals_1d[0][:,1],
                 exact_basis_vals_1d[1][:,1],exact_basis_vals_1d[0][:,2],
            exact_basis_vals_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
            exact_basis_vals_1d[1][:,2]]).T

        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,exact_basis_derivs_1d[0][:,1],0.*x,
             exact_basis_derivs_1d[0][:,2],
            exact_basis_derivs_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
            0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,0.*x,exact_basis_derivs_1d[1][:,1],0.*x,
            exact_basis_vals_1d[0][:,1]*exact_basis_derivs_1d[1][:,1],
            exact_basis_derivs_1d[1][:,2]]).T))

        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_mixed_basis_pce(self):
        degree = 2; deriv_order=1    
        probability_measure = True

        gauss_mean,gauss_var = -1,4
        univariate_variables = [
            uniform(-1,2),norm(gauss_mean,np.sqrt(gauss_var)),uniform(0,3)]
        variable = IndependentMultivariateRandomVariable(univariate_variables)
        var_trans = AffineRandomVariableTransformation(variable)
        num_vars = len(univariate_variables)

        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)

        univariate_quadrature_rules = [
            partial(gauss_jacobi_pts_wts_1D,alpha_poly=0,beta_poly=0),
            gauss_hermite_pts_wts_1D,
            partial(gauss_jacobi_pts_wts_1D,alpha_poly=0,beta_poly=0)]
        samples, weights = get_tensor_product_quadrature_rule(
            degree+1,num_vars,univariate_quadrature_rules,
            var_trans.map_from_canonical_space)

        indices = compute_hyperbolic_indices(num_vars,degree,1.0)

        # sort lexographically to make testing easier
        indices = sort_indices_lexiographically(indices)
        poly.set_indices(indices)

        basis_matrix = poly.basis_matrix(samples,{'deriv_order':1})
        vals_basis_matrix = basis_matrix[:samples.shape[1],:]
        inner_products = (vals_basis_matrix.T*weights).dot(vals_basis_matrix)
        assert np.allclose(inner_products,np.eye(basis_matrix.shape[1]))

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd,:].copy()
            if dd==0 or dd==2:
                if dd==2:
                    # y = x/3
                    # z = 2*y-1=2*x/3-1=2/3*x-3/2*2/3=2/3*(x-3/2)=(x-3/2)/(3/2)
                    loc,scale=3/2,3/2
                    x = (x-loc)/scale
                exact_basis_vals_1d.append(
                    np.asarray([1+0.*x,x,0.5*(3.*x**2-1)]).T)
                exact_basis_derivs_1d.append(np.asarray([0.*x,1.0+0.*x,3.*x]).T)
                exact_basis_vals_1d[-1]/=np.sqrt(1./(2*np.arange(degree+1)+1))
                exact_basis_derivs_1d[-1]/=np.sqrt(1./(2*np.arange(degree+1)+1))
                # account for affine transformation in derivs
                if dd==2:
                    exact_basis_derivs_1d[-1]/=scale
            if dd==1:
                loc,scale=gauss_mean,np.sqrt(gauss_var)
                x = (x-loc)/scale
                exact_basis_vals_1d.append(
                    np.asarray([1+0.*x,x,x**2-1]).T)
                exact_basis_derivs_1d.append(np.asarray([0.*x,1.0+0.*x,2.*x]).T)
                exact_basis_vals_1d[-1]/=np.sqrt(
                    sp.factorial(np.arange(degree+1)))
                exact_basis_derivs_1d[-1]/=np.sqrt(
                    sp.factorial(np.arange(degree+1)))
                # account for affine transformation in derivs
                exact_basis_derivs_1d[-1]/=scale
                

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:,0],
             exact_basis_vals_1d[0][:,1],
             exact_basis_vals_1d[1][:,1],
             exact_basis_vals_1d[2][:,1],
             exact_basis_vals_1d[0][:,2],
             exact_basis_vals_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
             exact_basis_vals_1d[1][:,2],
             exact_basis_vals_1d[0][:,1]*exact_basis_vals_1d[2][:,1],
             exact_basis_vals_1d[1][:,1]*exact_basis_vals_1d[2][:,1],
             exact_basis_vals_1d[2][:,2]]).T


        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,
             exact_basis_derivs_1d[0][:,1],
             0.*x,
             0*x,
             exact_basis_derivs_1d[0][:,2],
             exact_basis_derivs_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
             0.*x,
             exact_basis_derivs_1d[0][:,1]*exact_basis_vals_1d[2][:,1],
             0.*x,
             0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,
             0.*x,
             exact_basis_derivs_1d[1][:,1],
             0.*x,
             0*x,
             exact_basis_derivs_1d[1][:,1]*exact_basis_vals_1d[0][:,1],
             exact_basis_derivs_1d[1][:,2],
             0.*x,
             exact_basis_derivs_1d[1][:,1]*exact_basis_vals_1d[2][:,1],
             0.*x]).T))

        # x3 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,
             0.*x,
             0.*x,
             exact_basis_derivs_1d[2][:,1],
             0*x,
             0*x,
             0*x,
             exact_basis_derivs_1d[2][:,1]*exact_basis_vals_1d[0][:,1],
             exact_basis_derivs_1d[2][:,1]*exact_basis_vals_1d[1][:,1],
             exact_basis_derivs_1d[2][:,2]]).T))

        func = poly.basis_matrix
        exact_basis_matrix_derivs = exact_basis_matrix[samples.shape[1]:]
        basis_matrix_derivs_fd = np.empty_like(exact_basis_matrix_derivs)
        for ii in range(samples.shape[1]):
            basis_matrix_derivs_fd[ii::samples.shape[1],:] = approx_fprime(
                samples[:,ii:ii+1],func)

        #print(np.linalg.norm(
        #    exact_basis_matrix_derivs-basis_matrix_derivs_fd,
        #    ord=np.inf))
        assert np.allclose(
            exact_basis_matrix_derivs, basis_matrix_derivs_fd,
            atol=1e-7,rtol=1e-7)
        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_monomial_pce(self):
        num_vars = 2; alpha = 0.; beta = 0.; degree = 2; deriv_order=1    
        probability_measure = True

        poly = PolynomialChaosExpansion()
        var_trans = IdentityTransformation(num_vars)
        poly.configure({'poly_type':'monomial','var_trans':var_trans})

        def univariate_quadrature_rule(nn):
            x,w=gauss_jacobi_pts_wts_1D(nn,0,0)
            x=(x+1)/2.
            return x,w
        
        samples, weights = get_tensor_product_quadrature_rule(
            degree,num_vars,univariate_quadrature_rule)

        indices = compute_hyperbolic_indices(num_vars,degree,1.0)

        # sort lexographically to make testing easier
        I = np.lexsort((indices[0,:],indices[1,:], indices.sum(axis=0)))
        indices = indices[:,I]
        poly.set_indices(indices)

        basis_matrix = poly.basis_matrix(samples,{'deriv_order':1})

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd,:]
            exact_basis_vals_1d.append(
                np.asarray([1+0.*x,x,x**2]).T)
            exact_basis_derivs_1d.append(np.asarray([0.*x,1.0+0.*x,2.*x]).T)

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:,0],exact_basis_vals_1d[0][:,1],
                 exact_basis_vals_1d[1][:,1],exact_basis_vals_1d[0][:,2],
            exact_basis_vals_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
            exact_basis_vals_1d[1][:,2]]).T


        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,exact_basis_derivs_1d[0][:,1],0.*x,
             exact_basis_derivs_1d[0][:,2],
            exact_basis_derivs_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
            0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
            [0.*x,0.*x,exact_basis_derivs_1d[1][:,1],0.*x,
            exact_basis_vals_1d[0][:,1]*exact_basis_derivs_1d[1][:,1],
            exact_basis_derivs_1d[1][:,2]]).T))

        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_mixed_basis_pce_moments(self):
        degree = 2;

        alpha_stat,beta_stat=2,3
        univariate_variables = [beta(alpha_stat,beta_stat,0,1),norm(-1,2)]
        variable = IndependentMultivariateRandomVariable(univariate_variables)
        var_trans = AffineRandomVariableTransformation(variable)
        num_vars = len(univariate_variables)

        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        poly.set_indices(indices)

        univariate_quadrature_rules = [
            partial(gauss_jacobi_pts_wts_1D,alpha_poly=beta_stat-1,
                    beta_poly=alpha_stat-1),gauss_hermite_pts_wts_1D]
        samples, weights = get_tensor_product_quadrature_rule(
            degree+1,num_vars,univariate_quadrature_rules,
            var_trans.map_from_canonical_space)

        coef = np.ones((indices.shape[1],2))
        coef[:,1]*=2
        poly.set_coefficients(coef)
        basis_matrix = poly.basis_matrix(samples)
        values = basis_matrix.dot(coef)
        true_mean = values.T.dot(weights)
        true_variance = (values.T**2).dot(weights)-true_mean**2

        assert np.allclose(poly.mean(),true_mean)
        assert np.allclose(poly.variance(),true_variance)

    def test_hahn_hypergeometric(self):
        degree = 4;
        M,n,N = 20,7,12
        rv = hypergeom(M,n,N)
        var_trans = AffineRandomVariableTransformation([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(np.arange(degree+1)[np.newaxis,:])
        xk = np.arange(0, n+1)[np.newaxis,:]
        p = poly.basis_matrix(xk)
        w = rv.pmf(xk[0,:])
        assert np.allclose(np.dot(p.T*w,p),np.eye(degree+1))

    def test_krawtchouk_binomial(self):
        degree = 4;
        n,p=10,0.5
        rv = binom(n,p)
        var_trans = AffineRandomVariableTransformation([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(np.arange(degree+1)[np.newaxis,:])
        xk = np.arange(0, n+1)[np.newaxis,:]
        p = poly.basis_matrix(xk)
        w = rv.pmf(xk[0,:])
        assert np.allclose(np.dot(p.T*w,p),np.eye(degree+1))
        
    def test_discrete_chebyshev(self):
        N,degree=10,5
        xk,pk = np.arange(N),np.ones(N)/N
        rv = float_rv_discrete(name='discrete_chebyshev',values=(xk,pk))()
        var_trans = AffineRandomVariableTransformation([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(np.arange(degree+1)[np.newaxis,:])
        p = poly.basis_matrix(xk[np.newaxis,:])
        w = pk
        assert np.allclose(np.dot(p.T*w,p),np.eye(degree+1))


    def test_float_rv_discrete_chebyshev(self):
        N,degree=10,5
        xk,pk = np.geomspace(1.0, 512.0, num=N),np.ones(N)/N
        rv = float_rv_discrete(name='float_rv_discrete',values=(xk,pk))()
        var_trans = AffineRandomVariableTransformation([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(np.arange(degree+1)[np.newaxis,:])
        p = poly.basis_matrix(xk[np.newaxis,:])
        w = pk
        assert np.allclose(np.dot(p.T*w,p),np.eye(degree+1))

    def test_conditional_moments_of_polynomial_chaos_expansion(self):
        num_vars = 3
        degree = 2
        inactive_idx = [0,2]
        np.random.seed(1)
        # keep variables on canonical domain to make constructing
        # tensor product quadrature rule, used for testing, easier
        var = [uniform(-1,2),beta(2,2,-1,2),norm(0,1)]
        quad_rules = [
            partial(gauss_jacobi_pts_wts_1D,alpha_poly=0,beta_poly=0),
            partial(gauss_jacobi_pts_wts_1D,alpha_poly=1,beta_poly=1),
            partial(gauss_hermite_pts_wts_1D)]
        var_trans = AffineRandomVariableTransformation(var)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(compute_hyperbolic_indices(num_vars,degree,1.0))
        poly.set_coefficients(
            np.arange(poly.indices.shape[1],dtype=float)[:,np.newaxis])

        fixed_samples = np.array(
            [[vv.rvs() for vv in np.array(var)[inactive_idx]]]).T
        mean, variance =  conditional_moments_of_polynomial_chaos_expansion(
            poly,fixed_samples,inactive_idx,True)

        from pyapprox.utilities import get_all_sample_combinations
        from pyapprox.probability_measure_sampling import \
            generate_independent_random_samples
        active_idx = np.setdiff1d(np.arange(num_vars),inactive_idx)
        random_samples, weights = get_tensor_product_quadrature_rule(
            [2*degree]*len(active_idx),len(active_idx),
            [quad_rules[ii] for ii in range(num_vars) if ii in active_idx])
        samples=get_all_sample_combinations(fixed_samples, random_samples)
        temp = samples[len(inactive_idx):].copy()
        samples[inactive_idx] = samples[:len(inactive_idx)]
        samples[active_idx] = temp

        true_mean = (poly(samples).T.dot(weights).T)
        true_variance = ((poly(samples)**2).T.dot(weights).T)-true_mean**2
        assert np.allclose(true_mean,mean)
        assert np.allclose(true_variance,variance)


if __name__== "__main__":    
    multivariate_polynomials_test_suite = \
 unittest.TestLoader().loadTestsFromTestCase(TestMultivariatePolynomials)
    unittest.TextTestRunner(verbosity=2).run(multivariate_polynomials_test_suite)
