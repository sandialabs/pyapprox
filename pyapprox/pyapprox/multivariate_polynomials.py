from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from pyapprox.indexing import \
     compute_hyperbolic_indices
from pyapprox.utilities import cartesian_product, outer_product
from pyapprox.orthonormal_polynomials_1d import \
     jacobi_recurrence, evaluate_orthonormal_polynomial_deriv_1d, \
     hermite_recurrence, krawtchouk_recurrence, hahn_recurrence, \
     discrete_chebyshev_recurrence, evaluate_orthonormal_polynomial_1d
from pyapprox.monomial import monomial_basis_matrix
from pyapprox.numerically_generate_orthonormal_polynomials_1d import lanczos, \
    modified_chebyshev_orthonormal

def evaluate_multivariate_orthonormal_polynomial(
        samples,indices,recursion_coeffs,deriv_order=0,
        basis_type_index_map=None):

    """
    Evaluate a multivaiate orthonormal polynomial and its s-derivatives 
    (s=1,...,num_derivs) using a three-term recurrence coefficients.

    Parameters
    ----------

    samples : np.ndarray (num_vars, num_samples)
        Samples at which to evaluate the polynomial

    indices : np.ndarray (num_vars, num_indices)
        The exponents of each polynomial term

    recursion_coeffs : np.ndarray (num_indices,2)
        The coefficients of each monomial term

    deriv_order : integer in [0,1]
       The maximum order of the derivatives to evaluate.

    Return
    ------
    values : np.ndarray (1+deriv_order*num_samples,num_indices)
        The values of the polynomials at the samples
    """
    num_vars, num_indices = indices.shape
    assert samples.shape[0]==num_vars
    assert samples.shape[1]>0
    #assert recursion_coeffs.shape[0]>indices.max()
    max_level_1d = indices.max(axis=1)
    if basis_type_index_map is None:
        basis_type_index_map = np.zeros(num_vars,dtype=int)
        recursion_coeffs = [recursion_coeffs]

    for dd in range(num_vars):
        assert (recursion_coeffs[basis_type_index_map[dd]].shape[0]>
                max_level_1d[dd])

    assert deriv_order>=0 and deriv_order<=1

    # My cython implementaion is currently slower than pure python found here
    # try:
    #     from pyapprox.cython.multivariate_polynomials import \
    #         evaluate_multivariate_orthonormal_polynomial_pyx
    #     return evaluate_multivariate_orthonormal_polynomial_pyx(
    #         samples,indices,recursion_coeffs,deriv_order)
    # except:
    #     print('evaluate_multivariate_orthonormal_polynomial extension failed')
    
    # precompute 1D basis functions for faster evaluation of
    # multivariate terms
    basis_vals_1d = []
    for dd in range(num_vars):
        basis_vals_1d_dd = evaluate_orthonormal_polynomial_deriv_1d(
            samples[dd,:],max_level_1d[dd],
            recursion_coeffs[basis_type_index_map[dd]],deriv_order)
        basis_vals_1d.append(basis_vals_1d_dd)

    num_samples = samples.shape[1]
    values = np.zeros(((1+deriv_order*num_vars)*num_samples,num_indices))

    for ii in range(num_indices):
        index = indices[:,ii]
        values[:num_samples,ii]=basis_vals_1d[0][:,index[0]]
        for dd in range(1,num_vars):
            values[:num_samples,ii]*=basis_vals_1d[dd][:,index[dd]]

    if deriv_order==0:
        return values

    for ii in range(num_indices):
        index = indices[:,ii]
        for jj in range(num_vars):
            # derivative in jj direction
            basis_vals=\
              basis_vals_1d[jj][:,(max_level_1d[jj]+1)+index[jj]].copy()
            # basis values in other directions
            for dd in range(num_vars):
                if dd!=jj:
                    basis_vals*=basis_vals_1d[dd][:,index[dd]]
            
            values[(jj+1)*num_samples:(jj+2)*num_samples,ii] = basis_vals
        
    return values

class PolynomialChaosExpansion(object):
    def __init__(self):
        self.coefficients=None
        self.indices=None
        self.recursion_coeffs=[]
        self.basis_type_index_map=None
        self.basis_type_var_indices=[]

    def configure(self, opts):
        self.config_opts=opts
        self.var_trans = opts.get('var_trans',None)
        if self.var_trans is None:
            raise Exception('must set var_trans')
        self.max_degree=-np.ones(self.num_vars(),dtype=int)

    def get_recursion_coefficients(self,opts,num_coefs):
        poly_type = opts.get('poly_type',None)
        var_type=None
        if poly_type is None:
            var_type=opts['rv_type']
        if poly_type=='legendre' or var_type=='uniform':
            recursion_coeffs = jacobi_recurrence(
                num_coefs,alpha=0,beta=0,probability=True)
        elif poly_type=='jacobi' or var_type=='beta':
            if poly_type is not None:
                alpha_poly,beta_poly = opts['alpha_poly'],opts['beta_poly']
            else:
                alpha_poly,beta_poly=opts['shapes']['b']-1,opts['shapes']['a']-1
            recursion_coeffs = jacobi_recurrence(
                num_coefs,alpha=alpha_poly,beta=beta_poly,probability=True)
        elif poly_type=='hermite' or var_type=='norm':
            recursion_coeffs = hermite_recurrence(
                num_coefs, rho=0., probability=True)
        elif poly_type=='krawtchouk' or var_type=='binom':
            if poly_type is None:
                opts = opts['shapes'] 
            n,p = opts['n'],opts['p']
            num_coefs = min(num_coefs,n)
            recursion_coeffs = krawtchouk_recurrence(
                num_coefs,n,p,probability=True)
        elif poly_type=='hahn' or var_type=='hypergeom':
            if poly_type is not None:
                apoly,bpoly = opts['alpha_poly'],opts['beta_poly']
                N=opts['N']
            else:
                M,n,N=[opts['shapes'][key] for key in ['M','n','N']]
                apoly,bpoly = -(n+1),-M-1+n
            num_coefs = min(num_coefs,N)
            recursion_coeffs = hahn_recurrence(
                num_coefs,N,apoly,bpoly,probability=True)
        elif poly_type=='discrete_chebyshev' or var_type=='discrete_chebyshev':
            if poly_type is not None:
                N = opts['N']
            else:
                N = opts['shapes']['xk'].shape[0]
                assert np.allclose(opts['shapes']['xk'],np.arange(N))
                assert np.allclose(opts['shapes']['pk'],np.ones(N)/N)
            num_coefs = min(num_coefs,N)
            recursion_coeffs = discrete_chebyshev_recurrence(
                num_coefs,N,probability=True)
        elif poly_type=='discrete_numeric' or var_type=='float_rv_discrete':
            if poly_type is None:
                opts = opts['shapes']
            xk,pk = opts['xk'],opts['pk']
            recursion_coeffs  = modified_chebyshev_orthonormal(
                num_coefs,[xk,pk],probability=True)
            p = evaluate_orthonormal_polynomial_1d(
                np.asarray(xk,dtype=float),num_coefs-1, recursion_coeffs)
            if not np.allclose((p.T*pk).dot(p),np.eye(num_coefs)):
                error = np.absolute((p.T*pk).dot(p)-np.eye(num_coefs)).max() 
                msg = f'basis created is ill conditioned. Max error: {error}'
                raise Exception(msg)
        elif poly_type=='monomial':
            recursion_coeffs=None
        else:
            if poly_type is not None:
                raise Exception('poly_type (%s) not supported'%poly_type)
            else:
                raise Exception('var_type (%s) not supported'%var_type)
        return recursion_coeffs


    def update_recursion_coefficients(self,num_coefs_per_var,opts):
        num_coefs_per_var = np.atleast_1d(num_coefs_per_var)
        initializing=False
        if self.basis_type_index_map is None:
            initializing=True
            self.basis_type_index_map = np.zeros((self.num_vars()),dtype=int)
        if 'poly_types' in opts:
            ii=0
            for key, poly_opts in opts['poly_types'].items():
                if (initializing or (
                    np.any(num_coefs_per_var[self.basis_type_var_indices[ii]]>
                        self.max_degree[self.basis_type_var_indices[ii]]+1))):
                    if initializing:
                        self.basis_type_var_indices.append(
                            poly_opts['var_nums'])
                    num_coefs=num_coefs_per_var[
                        self.basis_type_var_indices[ii]].max()
                    recursion_coeffs_ii = self.get_recursion_coefficients(
                                          poly_opts,num_coefs)
                    if recursion_coeffs_ii is None:
                        # recursion coefficients will be None is returned if
                        # monomial basis is specified. Only allow monomials to
                        # be used if all variables use monomial basis
                        assert len(opts['poly_types'])==1
                    if initializing:
                        self.recursion_coeffs.append(recursion_coeffs_ii)
                    else:
                        self.recursion_coeffs[ii] = recursion_coeffs_ii
                # extract variables indices for which basis is to be used
                self.basis_type_index_map[self.basis_type_var_indices[ii]]=ii
                ii+=1
        else:
            # when only one type of basis is assumed then allow poly_type to
            # be elevated to top level of options dictionary.
            self.recursion_coeffs=[self.get_recursion_coefficients(
                opts,num_coefs_per_var.max())]
        
        
    def set_indices(self,indices):
        #assert indices.dtype==int
        if indices.ndim==1:
            indices = indices.reshape((1,indices.shape[0]))
        self.indices=indices
        assert indices.shape[0]==self.num_vars()
        max_degree = indices.max(axis=1)
        if np.any(self.max_degree<max_degree):
            self.update_recursion_coefficients(max_degree+1,self.config_opts)
            self.max_degree=max_degree

    def basis_matrix(self,samples,opts=dict()):
        assert samples.ndim==2
        assert samples.shape[0]==self.num_vars()
        canonical_samples = self.var_trans.map_to_canonical_space(
            samples)
        basis_matrix = self.canonical_basis_matrix(canonical_samples,opts)
        deriv_order = opts.get('deriv_order',0)
        if deriv_order==1:
            basis_matrix[samples.shape[1]:,:]=\
              self.var_trans.map_derivatives_from_canonical_space(
                basis_matrix[samples.shape[1]:,:])
        return basis_matrix

    def canonical_basis_matrix(self,canonical_samples,opts=dict()):
        deriv_order = opts.get('deriv_order',0)
        if self.recursion_coeffs[0] is not None:
            basis_matrix = evaluate_multivariate_orthonormal_polynomial(
                canonical_samples,self.indices,self.recursion_coeffs,
                deriv_order,self.basis_type_index_map)
        else:
            basis_matrix = monomial_basis_matrix(
                self.indices,canonical_samples,deriv_order)
        return basis_matrix

    def set_coefficients(self,coefficients):
        assert coefficients.ndim==2
        assert coefficients.shape[0]==self.num_terms()
        
        self.coefficients = coefficients.copy()

    def get_coefficients(self):
        if self.coefficients is not None:
            return self.coefficients.copy()

    def get_indices(self):
        return self.indices.copy()

    def value(self,samples):
        basis_matrix = self.basis_matrix(samples)
        return np.dot(basis_matrix,self.coefficients)

    def num_vars(self):
        return self.var_trans.num_vars()

    def __call__(self,samples):
        return self.value(samples)

    def mean(self):
        return self.coefficients[0,:]

    def variance(self):
        var = np.sum(self.coefficients[1:,:]**2,axis=0)
        return var

    def num_terms(self):
        # truncated svd creates basis with num_terms <= num_indices
        return self.indices.shape[1]
    
from pyapprox.utilities import get_tensor_product_quadrature_rule
from functools import partial
from pyapprox.orthonormal_polynomials_1d import gauss_quadrature
def get_tensor_product_quadrature_rule_from_pce(pce,degrees):
    num_vars = pce.num_vars()
    degrees = np.atleast_1d(degrees)
    if degrees.shape[0]==1 and num_vars>1:
        degrees = np.array([degrees[0]]*num_vars)
    if np.any(pce.max_degree<degrees):
        pce.update_recursion_coefficients(degrees,pce.config_opts)
    if len(pce.recursion_coeffs)==1:
        # update_recursion_coefficients may not return coefficients
        # up to degree specified if using recursion for polynomial
        # orthogonal to a discrete variable with finite non-zero
        # probability measures
        assert pce.recursion_coeffs[0].shape[0]>=degrees.max()+1
        univariate_quadrature_rules = [
            partial(gauss_quadrature,pce.recursion_coeffs[0])]*num_vars
    else:
        univariate_quadrature_rules = []
        for dd in range(num_vars):
            # update_recursion_coefficients may not return coefficients
            # up to degree specified if using recursion for polynomial
            # orthogonal to a discrete variable with finite non-zero
            # probability measures
            assert (pce.recursion_coeffs[basis_type_index_map[dd]].shape[0]>=
                    degrees[dd]+1)
            univariate_quadrature_rules.append(
                partial(gauss_quadrature,
                        pce.recursion_coeffs[basis_type_index_map[dd]]))
            
    canonical_samples,weights = \
        get_tensor_product_quadrature_rule(
        degrees+1,num_vars,univariate_quadrature_rules)
    samples = pce.var_trans.map_from_canonical_space(
        canonical_samples)
    return samples, weights

from pyapprox.variables import get_distribution_info
def define_poly_options_from_variable_transformation(var_trans):
    pce_opts = {'var_trans':var_trans}
    basis_opts = dict()
    for ii in range(len(var_trans.variable.unique_variables)):
        var = var_trans.variable.unique_variables[ii]
        name, scales, shapes = get_distribution_info(var)
        opts = {'rv_type':name,'shapes':shapes,
                'var_nums':var_trans.variable.unique_variable_indices[ii]}
        basis_opts['basis%d'%ii]=opts
    pce_opts['poly_types']=basis_opts
    return pce_opts
    
def conditional_moments_of_polynomial_chaos_expansion(poly,samples,inactive_idx,return_variance=False):
    """
    Return mean and variance of polynomial chaos expansion with some variables
    fixed at specified values.

    Parameters
    ----------
    inactive_idx : np.ndarray
        The indices of the fixed variables

    Returns
    -------
    mean : np.ndarray
       The conditional mean (num_qoi)

    variance : np.ndarray
       The conditional variance (num_qoi). Only returned if 
       return_variance=True. Computing variance is significantly slower than
       computing mean. TODO check it is indeed slower
    """
    assert samples.shape[0] == len(inactive_idx)
    assert samples.ndim==2 and samples.shape[1]==1
    assert poly.coefficients is not None
    coef = poly.get_coefficients()
    indices = poly.get_indices()

    # precompute 1D basis functions for faster evaluation of
    # multivariate terms
    basis_vals_1d = []
    for dd in range(len(inactive_idx)):
        basis_vals_1d_dd=evaluate_orthonormal_polynomial_1d(
            samples[dd,:],indices[inactive_idx[dd],:].max(),
            poly.recursion_coeffs[poly.basis_type_index_map[inactive_idx[dd]]])
        basis_vals_1d.append(basis_vals_1d_dd)

    active_idx = np.setdiff1d(np.arange(poly.num_vars()),inactive_idx)
    mean = coef[0].copy()
    for ii in range(1,indices.shape[1]):
        index = indices[:,ii]
        coef_ii = coef[ii]# this intentionally updates the coef matrix
        for dd in range(len(inactive_idx)):
            coef_ii *= basis_vals_1d[dd][0,index[inactive_idx[dd]]]
        if index[active_idx].sum()==0:
            mean += coef_ii

    if not return_variance:
        return mean
    
    unique_indices,repeated_idx=np.unique(
        indices[active_idx,:],axis=1,return_inverse=True)
    new_coef = np.zeros((unique_indices.shape[1],coef.shape[1]))
    for ii in range(repeated_idx.shape[0]):
        new_coef[repeated_idx[ii]]+=coef[ii]
    variance = np.sum(new_coef**2,axis=0)-mean**2
    return mean, variance

def remove_variables_from_polynomial_chaos_expansion(poly,inactive_idx):
    """
    This function is not optimal. It will recreate the options
    used to configure the polynomial. Any recursion coefficients 
    calculated which are still relevant will need to be computed.
    This is probably not a large overhead though
    """
    fixed_pce = PolynomialChaosExpansion()
    opts = poly.config_opts.copy()
    opts['var_trans'] =  AffineRandomVariableTransformation(
        IndependentMultivariateRandomVariable(
            poly.var_trans.variables.all_variables()[inactive_idx]))

    if opts['poly_types'] is not None:
        for key, poly_opts in opts['poly_types'].items():
            var_nums = poly_opts['var_nums']
            poly_opts['var_nums'] = np.array(
                [var_nums[ii] for ii in range(len(var_nums))
                 if var_nums[ii] not in inactive_idx])
    #else # no need to do anything same basis is used for all variables
    
    fixed_pce.configure(opts)
    if poly.indices is not None:
        active_idx = np.setdiff1d(np.arange(poly.num_vars()),inactive_idx)
        reduced_indices = indices[active_idx,:]
    pce.set_indices(reduced_indices)
    assert pce.coefficients is None
    return fixed_pce
