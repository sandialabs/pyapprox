from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np, os
from pyapprox.orthonormal_polynomials_1d import \
     jacobi_recurrence, hermite_recurrence, gauss_quadrature
from pyapprox.utilities import beta_pdf, beta_pdf_derivative
     
def clenshaw_curtis_rule_growth(level):
    """
    The number of samples in the 1D Clenshaw-Curtis quadrature rule of a given
    level.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    if level==0:
        return 1
    else:
        return 2**level+1

def clenshaw_curtis_hierarchical_to_nodal_index(
        level,ll,ii):
    """
    Convert a 1D hierarchical index (ll,ii) to a nodal index for lookup in a
    Clenshaw-Curtis quadrature rule. 

    Given a quadrature rule of the specified max level (level)
    with indices [0,1,...,num_indices] this function can be used
    to convert a hierarchical index, e.g. of the constant function 
    (poly_index=0), to the quadrature index, e.g for poly_index=0, 
    index=num_indices/2). This allows one to take advantage of nestedness of 
    quadrature rule and only store quadrature rule for max level.

    Parameters
    ----------
    level : integer
        The maximum level of the quadrature rule

    ll : integer 
        The level of the polynomial index

    ii : integer
        The polynomial index

    Return
    ------
    nodal_index : integer
        The equivalent nodal index of (ll,ii)
    """
    num_indices = clenshaw_curtis_rule_growth(level)
    # mid point
    if ll==0:
        return num_indices/2
    # boundaries
    elif ll==1:
        if ii==0:
            return 0
        else:
            return num_indices-1
    # higher level points
    return (2*ii+1)*2**(level-ll)

def clenshaw_curtis_poly_indices_to_quad_rule_indices(level):
    """
    Convert all 1D polynomial indices of up to and including a given level
    to their equivalent nodal index for lookup in a Clenshaw-Curtis 
    quadrature rule. 

    Parameters
    ----------
    level : integer
        The maximum level of the quadrature rule

    Return
    ------
    quad_rule_indices : np.ndarray (num_vars x num_indices)
        All the quadrature rule indices
    """
    quad_rule_indices = []
    num_previous_hier_indices = 0
    for ll in range(level+1):
        num_hierarchical_indices=\
          clenshaw_curtis_rule_growth(ll)-num_previous_hier_indices
        for ii in range(num_hierarchical_indices):
            quad_index = clenshaw_curtis_hierarchical_to_nodal_index(
                level,ll,ii)
            quad_rule_indices.append(quad_index)
        num_previous_hier_indices+=num_hierarchical_indices
    return np.asarray(quad_rule_indices,dtype=int)


def clenshaw_curtis_in_polynomial_order(level,return_weights_for_all_levels=True):
    """
    Return the samples and weights of the Clenshaw-Curtis rule using 
    polynomial ordering.

    The first point will be the middle weight of the rule. The second and
    third weights will be the left and right boundary weights. All other weights
    left of mid point will come next followed by all remaining points.

    Parameters
    ----------
    level : integer
        The level of the isotropic sparse grid.

    return_weights_for_all_levels : boolean
        True  - return weights [w(0),w(1),...,w(level)]
        False - return w(level)

    Return
    ------
    ordered_samples_1d : np.ndarray (num_samples_1d)
        The reordered samples.

    ordered_weights_1d : np.ndarray (num_samples_1d)
        The reordered weights.
    """

    #w*=2. #use if want do not want to use probability formulation
   
    if return_weights_for_all_levels:
        ordered_weights_1d = []
        for ll in range(level+1):
            x,w = clenshaw_curtis_pts_wts_1D(ll)
            quad_indices = clenshaw_curtis_poly_indices_to_quad_rule_indices(ll)
            ordered_weights_1d.append(w[quad_indices])
        # ordered samples for last x
        ordered_samples_1d = x[quad_indices]
    else:
        x,w = clenshaw_curtis_pts_wts_1D(level)
        quad_indices = clenshaw_curtis_poly_indices_to_quad_rule_indices(level)
        ordered_samples_1d = x[quad_indices]
        ordered_weights_1d = w[quad_indices]
        
    return ordered_samples_1d, ordered_weights_1d


def clenshaw_curtis_pts_wts_1D(level):
    """
    Generated a nested, exponentially-growing Clenshaw-Curtis quadrature rule 
    that exactly integrates polynomials of degree 2**level+1 with respect to 
    the uniform probability measure on [-1,1].

    Parameters
    ----------
    level : integer
        The level of the nested quadrature rule. The number of samples in the
        quadrature rule will be 2**level+1

    Returns
    -------
    x : np.ndarray(num_samples)
        Quadrature samples

    w : np.ndarray(num_samples)
        Quadrature weights
    """

    try:
        from pyapprox.cython.univariate_quadrature import \
            clenshaw_curtis_pts_wts_1D_pyx
        return clenshaw_curtis_pts_wts_1D_pyx(level)
        # from pyapprox.weave.univariate_quadrature import \
        #     c_clenshaw_curtis_pts_wts_1D
        # return c_clenshaw_curtis_pts_wts_1D(level)
    except:
        print ('clenshaw_curtis_pts_wts failed')
    
    num_samples = clenshaw_curtis_rule_growth(level)
  
    wt_factor = 1./2.

    x = np.empty((num_samples))
    w = np.empty_like(x)

    if ( level == 0 ):
        x[0]=0.; w[0]=1.;
    else:
        for jj in range(num_samples):
            if ( jj == 0 ):
                x[jj] = -1.;
                w[jj] = wt_factor / float(num_samples*(num_samples -2.))
            elif ( jj == num_samples-1 ):
                x[jj] = 1.;
                w[jj] = wt_factor / float(num_samples*(num_samples-2.))
            else:
                x[jj] = -np.cos(np.pi*float(jj)/float(num_samples-1))
                mysum = 0.0
                for kk in range(1,(num_samples-3)//2+1):
                    mysum += 1. / float(4.*kk*kk-1.)*\
                      np.cos( 2.*np.pi*float(kk*jj)/float(num_samples-1.))
                w[jj] = 2./float(num_samples-1.)*(1.-np.cos(np.pi*float(jj) ) /
                          float(num_samples*(num_samples -2.))-2.*(mysum))
                w[jj] *= wt_factor;
            if ( abs( x[jj] ) < 2.*np.finfo(float).eps): x[jj] = 0.
    return x, w

def gauss_hermite_pts_wts_1D(num_samples):
    """
    Return Gauss Hermite quadrature rule that exactly integrates polynomials
    of degree 2*num_samples-1 with respect to the Gaussian probability measure
    1/sqrt(2*pi)exp(-x**2/2)

    Parameters
    ----------
    num_samples : integer
        The number of samples in the quadrature rule

    Returns
    -------
    x : np.ndarray(num_samples)
        Quadrature samples

    w : np.ndarray(num_samples)
        Quadrature weights
    """
    rho = 0.0
    ab = hermite_recurrence(
        num_samples,rho,probability=True)
    x,w = gauss_quadrature(ab,num_samples)
    return x,w

def gauss_jacobi_pts_wts_1D(num_samples,alpha_poly,beta_poly):
    """
    Return Gauss Jacobi quadrature rule that exactly integrates polynomials
    of num_samples 2*num_samples-1 with respect to the probabilty density 
    function of Beta random variables on [-1,1]

    C*(1+x)^(beta_poly)*(1-x)^alpha_poly
    
    where
    
    C = 1/(2**(alpha_poly+beta_poly)*beta_fn(beta_poly+1,alpha_poly+1))

    or equivalently
    
    C*(1+x)**(alpha_stat-1)*(1-x)**(beta_stat-1)

    where 

    C = 1/(2**(alpha_stat+beta_stat-2)*beta_fn(alpha_stat,beta_stat))

    Parameters
    ----------
    num_samples : integer
        The number of samples in the quadrature rule

    alpha_poly : float
        The Jaocbi parameter alpha = beta_stat-1

    beta_poly : float
        The Jacobi parameter beta = alpha_stat-1 

    Returns
    -------
    x : np.ndarray(num_samples)
        Quadrature samples

    w : np.ndarray(num_samples)
        Quadrature weights
    """
    ab = jacobi_recurrence(
        num_samples,alpha=alpha_poly,beta=beta_poly,probability=True)
    return gauss_quadrature(ab,num_samples)

def leja_growth_rule(level):
    """
    The number of samples in the 1D Leja quadrature rule of a given
    level.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    return level+1

from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
from pyapprox.leja_sequences import get_leja_sequence_1d,\
    get_quadrature_weights_from_samples
from functools import partial
from pyapprox.utilities import evaluate_tensor_product_function,\
     gradient_of_tensor_product_function
from scipy.stats import beta as beta_rv
def beta_leja_quadrature_rule(alpha_stat,beta_stat,level,
                              growth_rule=leja_growth_rule,
                              samples_filename=None,
                              return_weights_for_all_levels=True,
                              initial_points=None):
    """
    Return the samples and weights of the Leja quadrature rule for the beta
    probability measure. 

    By construction these rules have polynomial ordering.

    Parameters
    ----------
    level : integer
        The level of the isotropic sparse grid.

    alpha_stat : integer
        The alpha shape parameter of the Beta distribution

    beta_stat : integer
        The beta shape parameter of the Beta distribution

    samples_filename : string
         Name of file to save leja samples and weights to

    Return
    ------
    ordered_samples_1d : np.ndarray (num_samples_1d)
        The reordered samples.

    ordered_weights_1d : np.ndarray (num_samples_1d)
        The reordered weights.
    """
    num_vars = 1
    num_leja_samples = growth_rule(level)
    #print(('num_leja_samples',num_leja_samples))

    # freezing beta rv like below has huge overhead
    # it creates a docstring each time which adds up to many seconds
    # for repeated calls to pdf
    #univariate_weight_function=lambda x: beta_rv(
    #    alpha_stat,beta_stat).pdf((x+1)/2)/2
    #univariate_weight_function = lambda x: beta_rv.pdf(
    #    (x+1)/2,alpha_stat,beta_stat)/2
    univariate_weight_function = lambda x: beta_pdf(
        alpha_stat,beta_stat,(x+1)/2)/2
    univariate_weight_function_deriv = lambda x: beta_pdf_derivative(
       alpha_stat,beta_stat,(x+1)/2)/4
    
    weight_function = partial(
        evaluate_tensor_product_function,
        [univariate_weight_function]*num_vars)
    
    weight_function_deriv = partial(
        gradient_of_tensor_product_function,
        [univariate_weight_function]*num_vars,
        [univariate_weight_function_deriv]*num_vars)

    # assert np.allclose(
    #     (univariate_weight_function(0.5+1e-8)-
    #          univariate_weight_function(0.5))/1e-8,
    #     univariate_weight_function_deriv(0.5),atol=1e-6)

    poly = PolynomialChaosExpansion()
    # must be imported locally otherwise I have a circular dependency
    from pyapprox.variable_transformations import \
        define_iid_random_variable_transformation
    from scipy.stats import uniform
    var_trans = define_iid_random_variable_transformation(
        uniform(-1,2),num_vars)

    poly_opts = {'poly_type':'jacobi','alpha_poly':beta_stat-1,
                 'beta_poly':alpha_stat-1,'var_trans':var_trans}
    poly.configure(poly_opts) 

    if samples_filename is None or not os.path.exists(samples_filename):
        ranges = [-1,1]
        from scipy.stats import beta as beta_rv
        if initial_points is None:
            initial_points = np.asarray(
                [[2*beta_rv(alpha_stat,beta_stat).ppf(0.5)-1]]).T
        leja_sequence = get_leja_sequence_1d(
            num_leja_samples,initial_points,poly,
            weight_function,weight_function_deriv,ranges)
        if samples_filename is not None:
            np.savez(samples_filename,samples=leja_sequence)
    else:
        leja_sequence = np.load(samples_filename)['samples']
        #print (leja_sequence.shape[1],growth_rule(level),level)
        assert leja_sequence.shape[1]>=growth_rule(level)
        leja_sequence = leja_sequence[:,:growth_rule(level)]

    indices = np.arange(growth_rule(level))[np.newaxis,:]
    poly.set_indices(indices)
    ordered_weights_1d = get_leja_sequence_quadrature_weights(
        leja_sequence,growth_rule,poly.basis_matrix,weight_function,level,
        return_weights_for_all_levels)
    return leja_sequence[0,:], ordered_weights_1d

def gaussian_leja_quadrature_rule(level,
                                  growth_rule=leja_growth_rule,
                                  samples_filename=None,
                                  return_weights_for_all_levels=True,
                                  initial_points=None):
    """
    Return the samples and weights of the Leja quadrature rule for the beta
    probability measure. 

    By construction these rules have polynomial ordering.

    Parameters
    ----------
    level : integer
        The level of the isotropic sparse grid.

    samples_filename : string
         Name of file to save leja samples and weights to

    Return
    ------
    ordered_samples_1d : np.ndarray (num_samples_1d)
        The reordered samples.

    ordered_weights_1d : np.ndarray (num_samples_1d)
        The reordered weights.
    """
    num_vars = 1
    num_leja_samples = growth_rule(level)

    # freezing scipy gaussian rv like below has huge overhead
    # it creates a docstring each time which adds up to many seconds
    # for repeated calls to pdf
    from pyapprox.utilities import gaussian_pdf, gaussian_pdf_derivative
    univariate_weight_function = partial(gaussian_pdf,0,1)
    univariate_weight_function_deriv = partial(gaussian_pdf_derivative,0,1)
    
    weight_function = partial(
        evaluate_tensor_product_function,
        [univariate_weight_function]*num_vars)
    
    weight_function_deriv = partial(
        gradient_of_tensor_product_function,
        [univariate_weight_function]*num_vars,
        [univariate_weight_function_deriv]*num_vars)

    assert np.allclose(
        (univariate_weight_function(0.5+1e-8)-
             univariate_weight_function(0.5))/1e-8,
        univariate_weight_function_deriv(0.5),atol=1e-6)

    poly = PolynomialChaosExpansion()
    # must be imported locally otherwise I have a circular dependency
    from pyapprox.variable_transformations import \
        define_iid_random_variable_transformation
    from scipy.stats import norm
    var_trans = define_iid_random_variable_transformation(
        norm(),num_vars)
    poly_opts = {'poly_type':'hermite','var_trans':var_trans}
    poly.configure(poly_opts) 

    if samples_filename is None or not os.path.exists(samples_filename):
        ranges = [None,None]
        if initial_points is None:
            initial_points = np.asarray([[0.0]]).T
        leja_sequence = get_leja_sequence_1d(
            num_leja_samples,initial_points,poly,
            weight_function,weight_function_deriv,ranges)
        if samples_filename is not None:
            np.savez(samples_filename,samples=leja_sequence)
    else:
        leja_sequence = np.load(samples_filename)['samples']
        assert leja_sequence.shape[1]>=growth_rule(level)
        leja_sequence = leja_sequence[:,:growth_rule(level)]

    indices = np.arange(growth_rule(level))[np.newaxis,:]
    poly.set_indices(indices)
    ordered_weights_1d = get_leja_sequence_quadrature_weights(
        leja_sequence,growth_rule,poly.basis_matrix,weight_function,level,
        return_weights_for_all_levels)
    return leja_sequence[0,:], ordered_weights_1d


def get_leja_sequence_quadrature_weights(leja_sequence,growth_rule,
                                         basis_matrix_generator,
                                         weight_function,level,
                                         return_weights_for_all_levels):
    sqrt_weights = np.sqrt(weight_function(leja_sequence))
    # precondition matrix to produce better condition number
    basis_matrix = (basis_matrix_generator(leja_sequence).T*sqrt_weights).T
    if return_weights_for_all_levels:
        ordered_weights_1d = []
        for ll in range(level+1):
            basis_matrix_inv = np.linalg.inv(
                basis_matrix[:growth_rule(ll),:growth_rule(ll)])
            # make sure to adjust weights to account for preconditioning
            ordered_weights_1d.append(
                basis_matrix_inv[0,:]*sqrt_weights[:growth_rule(ll)])
    else:
        basis_matrix_inv = np.linalg.inv(
            basis_matrix[:growth_rule(level),:growth_rule(level)])
        # make sure to adjust weights to account for preconditioning
        ordered_weights_1d = basis_matrix_inv[0,:]*sqrt_weights

    return ordered_weights_1d


def uniform_leja_quadrature_rule(level,growth_rule=leja_growth_rule,
                                 samples_filename=None,
                                 return_weights_for_all_levels=True):
    return beta_leja_quadrature_rule(1,1,level,growth_rule,samples_filename,
                                     return_weights_for_all_levels)

def algebraic_growth(rate,level):
    return (level)**rate+1

def exponential_growth(level,constant=1):
    """
    The number of samples in an exponentiall growing 1D quadrature rule of 
    a given level.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    if level==0:
        return 1
    return constant*2**(level+1)-1

def exponential_growth_rule(quad_rule,level):
    return quad_rule(exponential_growth(level))

def candidate_based_leja_rule(recursion_coeffs,
                              generate_candidate_samples,
                              num_candidate_samples,
                              level,
                              initial_samples=None,
                              growth_rule=leja_growth_rule,
                              samples_filename=None,
                              return_weights_for_all_levels=True):

    from pyapprox.orthonormal_polynomials_1d import \
        evaluate_orthonormal_polynomial_1d
    from pyapprox.polynomial_sampling import get_lu_leja_samples,\
        christoffel_preconditioner, christoffel_weights
    num_leja_samples = growth_rule(level)
    generate_basis_matrix = lambda x: evaluate_orthonormal_polynomial_1d(
        x[0,:],num_leja_samples,recursion_coeffs)
    if samples_filename is None or not os.path.exists(samples_filename):
        leja_sequence,__ = get_lu_leja_samples(
            generate_basis_matrix,generate_candidate_samples,
            num_candidate_samples,num_leja_samples,
            preconditioning_function=christoffel_preconditioner,
            initial_samples=initial_samples)
        if samples_filename is not None:
            np.savez(samples_filename,samples=leja_sequence)
    else:
        leja_sequence = np.load(samples_filename)['samples']
        assert leja_sequence.shape[1]>=growth_rule(level)
        leja_sequence = leja_sequence[:,:growth_rule(level)]

    weight_function = lambda x: christoffel_weights(generate_basis_matrix(x))
    ordered_weights_1d = get_leja_sequence_quadrature_weights(
        leja_sequence,growth_rule,generate_basis_matrix,weight_function,level,
        return_weights_for_all_levels)
    
    return leja_sequence[0,:], ordered_weights_1d
    
