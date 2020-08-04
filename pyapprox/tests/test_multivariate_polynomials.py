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
            degree-1,num_vars,np.polynomial.legendre.leggauss)

        indices = compute_hyperbolic_indices(num_vars,degree,1.0)

        # sort lexographically to make testing easier
        I = np.lexsort((indices[0,:],indices[1,:], indices.sum(axis=0)))
        indices = indices[:,I]
        # remove [0,2] index so max_level is not the same for every dimension
        # also remove [1,0] and [1,1] to make sure can handle index sets that
        # have missing univariate degrees not at the ends
        J = [1,5,4]
        reduced_indices = np.delete(indices,J,axis=1)
        poly.set_indices(reduced_indices)

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

        exact_basis_matrix = np.delete(exact_basis_matrix,J,axis=1)

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

        assert np.allclose(np.diag(poly.covariance()),poly.variance())
        assert np.allclose(poly.covariance()[0,1],coef[1:,0].dot(coef[1:,1]))

    def test_pce_jacobian(self):
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

        sample = generate_independent_random_samples(variable,1)

        coef = np.ones((indices.shape[1],2))
        coef[:,1]*=2
        poly.set_coefficients(coef)

        jac = poly.jacobian(sample)
        from pyapprox.optimization import approx_jacobian
        fd_jac = approx_jacobian(
            lambda x: poly(x[:,np.newaxis])[0,:],sample[:,0])
        assert np.allclose(jac,fd_jac)

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
        #print((np.dot(p.T*w,p),np.eye(degree+1)))
        assert np.allclose(np.dot(p.T*w,p),np.eye(degree+1))

    def test_float_rv_discrete_chebyshev(self):
        N,degree=10,5
        xk,pk = np.geomspace(1.0, 512.0, num=N),np.ones(N)/N
        rv = float_rv_discrete(name='float_rv_discrete',values=(xk,pk))()
        var_trans = AffineRandomVariableTransformation([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly_opts['numerically_generated_poly_accuracy_tolerance']=1e-9
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

    def test_compute_univariate_orthonormal_basis_products(self):
        max_degree1,max_degree2=3,2

        get_recursion_coefficients = partial(
            jacobi_recurrence, alpha=0., beta=0., probability=True)

        product_coefs = compute_univariate_orthonormal_basis_products(
            get_recursion_coefficients,max_degree1,max_degree2)


        max_degree = max_degree1+max_degree2
        x = np.linspace(-1,1,51)
        recursion_coefs = get_recursion_coefficients(max_degree+1)
        ortho_basis_matrix = evaluate_orthonormal_polynomial_1d(
            x, max_degree, recursion_coefs)

        kk=0
        for d1 in range(max_degree1+1):
            for d2 in range(min(d1+1,max_degree2+1)):
                exact_product=ortho_basis_matrix[:,d1]*ortho_basis_matrix[:,d2]

                product = ortho_basis_matrix[:,:product_coefs[kk].shape[0]].dot(
                    product_coefs[kk]).sum(axis=1)
                assert np.allclose(product,exact_product)
                kk+=1

    def test_compute_multivariate_orthonormal_basis_product(self):
        univariate_variables = [norm(),uniform()]
        variable = IndependentMultivariateRandomVariable(
            univariate_variables)

        poly1 = get_polynomial_from_variable(variable)
        poly2 = get_polynomial_from_variable(variable)

        max_degrees1, max_degrees2 = [3,3],[2,2]
        product_coefs_1d = compute_product_coeffs_1d_for_each_variable(
                poly1,max_degrees1,max_degrees2)

        for ii in range(max_degrees1[0]):
            for jj in range(max_degrees1[1]):
                poly_index_ii,poly_index_jj=np.array([ii,jj]),np.array([ii,jj])

                poly1.set_indices(poly_index_ii[:,np.newaxis])
                poly1.set_coefficients(np.ones([1,1]))
                poly2.set_indices(poly_index_jj[:,np.newaxis])
                poly2.set_coefficients(np.ones([1,1]))

                product_indices, product_coefs = \
                    compute_multivariate_orthonormal_basis_product(
                        product_coefs_1d,poly_index_ii,poly_index_jj,
                        max_degrees1,max_degrees2)

                poly_prod = get_polynomial_from_variable(variable)
                poly_prod.set_indices(product_indices)
                poly_prod.set_coefficients(product_coefs)

                samples = generate_independent_random_samples(variable,5)
                #print(poly_prod(samples),poly1(samples)*poly2(samples))
            assert np.allclose(poly_prod(samples),poly1(samples)*poly2(samples))


    def test_multiply_multivariate_orthonormal_polynomial_expansions(self):
        univariate_variables = [norm(),uniform()]
        variable = IndependentMultivariateRandomVariable(
            univariate_variables)

        degree1,degree2=3,2
        poly1 = get_polynomial_from_variable(variable)
        poly1.set_indices(compute_hyperbolic_indices(
            variable.num_vars(),degree1))
        poly1.set_coefficients(np.random.normal(0,1,(poly1.indices.shape[1],1)))
        poly2 = get_polynomial_from_variable(variable)
        poly2.set_indices(compute_hyperbolic_indices(
            variable.num_vars(),degree2))
        poly2.set_coefficients(np.random.normal(0,1,(poly2.indices.shape[1],1)))

        max_degrees1 = poly1.indices.max(axis=1)
        max_degrees2 = poly2.indices.max(axis=1)
        product_coefs_1d = compute_product_coeffs_1d_for_each_variable(
            poly1,max_degrees1,max_degrees2)

        indices,coefs=multiply_multivariate_orthonormal_polynomial_expansions(
            product_coefs_1d,poly1.get_indices(),poly1.get_coefficients(),
            poly2.get_indices(),poly2.get_coefficients())

        poly3 = get_polynomial_from_variable(variable)
        poly3.set_indices(indices)
        poly3.set_coefficients(coefs)

        samples = generate_independent_random_samples(variable,10)
        #print(poly3(samples),poly1(samples)*poly2(samples))
        assert np.allclose(poly3(samples),poly1(samples)*poly2(samples))

    def test_multiply_pce(self):
        np.random.seed(1)
        np.set_printoptions(precision=16)
        univariate_variables = [norm(),uniform()]
        variable = IndependentMultivariateRandomVariable(
            univariate_variables)
        degree1,degree2=1,2
        poly1 = get_polynomial_from_variable(variable)
        poly1.set_indices(compute_hyperbolic_indices(
            variable.num_vars(),degree1))
        poly2 = get_polynomial_from_variable(variable)
        poly2.set_indices(compute_hyperbolic_indices(
            variable.num_vars(),degree2))

        #coef1 = np.random.normal(0,1,(poly1.indices.shape[1],1))
        #coef2 = np.random.normal(0,1,(poly2.indices.shape[1],1))
        coef1 = np.arange(poly1.indices.shape[1])[:,np.newaxis]
        coef2 = np.arange(poly2.indices.shape[1])[:,np.newaxis]
        poly1.set_coefficients(coef1)
        poly2.set_coefficients(coef2)

        poly3 = poly1*poly2
        samples = generate_independent_random_samples(variable,10)
        assert np.allclose(poly3(samples),poly1(samples)*poly2(samples))

        for order in range(4):
            poly = poly1**order
            assert np.allclose(poly(samples),poly1(samples)**order)
        
    def test_add_pce(self):
        univariate_variables = [norm(),uniform()]
        variable = IndependentMultivariateRandomVariable(
            univariate_variables)
        degree1,degree2=2,3
        poly1 = get_polynomial_from_variable(variable)
        poly1.set_indices(compute_hyperbolic_indices(
            variable.num_vars(),degree1))
        poly1.set_coefficients(np.random.normal(0,1,(poly1.indices.shape[1],1)))
        poly2 = get_polynomial_from_variable(variable)
        poly2.set_indices(compute_hyperbolic_indices(
            variable.num_vars(),degree2))
        poly2.set_coefficients(np.random.normal(0,1,(poly2.indices.shape[1],1)))

        poly3 = poly1+poly2+poly2
        samples = generate_independent_random_samples(variable,10)
        #print(poly3(samples),poly1(samples)*poly2(samples))
        assert np.allclose(poly3(samples),poly1(samples)+2*poly2(samples))

        poly4 = poly1-poly2
        samples = generate_independent_random_samples(variable,10)
        #print(poly3(samples),poly1(samples)*poly2(samples))
        assert np.allclose(poly4(samples),poly1(samples)-poly2(samples))

if __name__== "__main__":    
    multivariate_polynomials_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestMultivariatePolynomials)
    unittest.TextTestRunner(verbosity=2).run(
        multivariate_polynomials_test_suite)
