import numpy as np
from functools import partial

from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    gauss_quadrature, define_orthopoly_options_from_marginal
)
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence, hermite_recurrence
)
from pyapprox.surrogates.orthopoly.recursion_factory import (
    get_recursion_coefficients_from_variable
)
from pyapprox.variables.transforms import AffineTransform


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
    if level == 0:
        return 1
    else:
        return 2**level+1


def clenshaw_curtis_hierarchical_to_nodal_index(level, ll, ii):
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
    if ll == 0:
        return num_indices/2
    # boundaries
    elif ll == 1:
        if ii == 0:
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
        num_hierarchical_indices =\
            clenshaw_curtis_rule_growth(ll)-num_previous_hier_indices
        for ii in range(num_hierarchical_indices):
            quad_index = clenshaw_curtis_hierarchical_to_nodal_index(
                level, ll, ii)
            quad_rule_indices.append(quad_index)
        num_previous_hier_indices += num_hierarchical_indices
    return np.asarray(quad_rule_indices, dtype=int)


def clenshaw_curtis_in_polynomial_order(level,
                                        return_weights_for_all_levels=True):
    """
    Return the samples and weights of the Clenshaw-Curtis rule using
    polynomial ordering.

    The first point will be the middle weight of the rule. The second and
    third weights will be the left and right boundary weights. All other
    weights left of mid point will come next followed by all remaining points.

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

    # w*=2. #use if want do not want to use probability formulation

    if return_weights_for_all_levels:
        ordered_weights_1d = []
        for ll in range(level+1):
            x, w = clenshaw_curtis_pts_wts_1D(ll)
            quad_indices = clenshaw_curtis_poly_indices_to_quad_rule_indices(
                ll)
            ordered_weights_1d.append(w[quad_indices])
        # ordered samples for last x
        ordered_samples_1d = x[quad_indices]
    else:
        x, w = clenshaw_curtis_pts_wts_1D(level)
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
    except ImportError:
        print('clenshaw_curtis_pts_wts failed')

    return __clenshaw_curtis_pts_wts_1D(level)


def __clenshaw_curtis_pts_wts_1D(level):
    num_samples = clenshaw_curtis_rule_growth(level)

    wt_factor = 1./2.

    x = np.empty((num_samples))
    w = np.empty_like(x)

    if (level == 0):
        x[0] = 0.
        w[0] = 1.
    else:
        for jj in range(num_samples):
            if (jj == 0):
                x[jj] = -1.
                w[jj] = wt_factor / float(num_samples*(num_samples - 2.))
            elif (jj == num_samples-1):
                x[jj] = 1.
                w[jj] = wt_factor / float(num_samples*(num_samples-2.))
            else:
                x[jj] = -np.cos(np.pi*float(jj)/float(num_samples-1))
                mysum = 0.0
                for kk in range(1, (num_samples-3)//2+1):
                    mysum += 1. / float(4.*kk*kk-1.) *\
                        np.cos(2.*np.pi*float(kk*jj)/float(num_samples-1.))
                w[jj] = 2./float(num_samples-1.)*(
                    1.-np.cos(np.pi*float(jj)) /
                    float(num_samples*(num_samples - 2.))-2.*(mysum))
                w[jj] *= wt_factor
            if (abs(x[jj]) < 2.*np.finfo(float).eps):
                x[jj] = 0.
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
    ab = hermite_recurrence(num_samples, rho, probability=True)
    x, w = gauss_quadrature(ab, num_samples)
    return x, w


def gauss_jacobi_pts_wts_1D(num_samples, alpha_poly, beta_poly):
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
        num_samples, alpha=alpha_poly, beta=beta_poly, probability=True)
    return gauss_quadrature(ab, num_samples)


def one_point_growth_rule(level):
    """
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


def leja_growth_rule(level):
    """
    The number of samples in the 1D Leja quadrature rule of a given
    level. Most leja rules produce two point quadrature rules which
    have zero weight assigned to one point. Avoid this by skipping from
    one point rule to 3 point rule and then increment by 1.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    if level == 0:
        return 1
    return level+2


def constant_increment_growth_rule(increment, level):
    """
    The number of samples in the 1D quadrature rule where number of of points
    grow by a fixed constant at each level.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    if level == 1:
        return 3
    nsamples_1d = increment*level+1
    return nsamples_1d


def algebraic_growth(rate, level):
    return (level)**rate+1


def exponential_growth(level, constant=1):
    """
    The number of samples in an exponentially growing 1D quadrature rule of
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
    if level == 0:
        return 1
    return constant*2**(level+1)-1


def exponential_growth_rule(quad_rule, level):
    return quad_rule(exponential_growth(level))


def transformed_quadrature_rule(marginal, recursion_coeffs, nsamples):
    var_trans = AffineTransform([marginal])
    x, w = gauss_quadrature(recursion_coeffs, nsamples)
    x = var_trans.map_from_canonical(x[None, :])[0, :]
    return x, w


def get_gauss_quadrature_rule_from_marginal(
        marginal, max_nsamples, canonical=False):
    """
    Return the quadrature rule associated with the marginal.

    Parameters
    ----------
    marginal : scipy.stats.dist
        The 1D variable

    max_nsamples : integer
        The maximum number of samples that can be in the generated quadrature
        rules

    canonical : boolean
        True - the loc, and scale parameters of the marginal are
        ignored. The quadrature rules for all bounded variables will be
        defined on the interval [-1, 1].

    Returns
    -------
    quad_rule : callable
        Function that returns the quadrature samples and weights with the
        signature

        `quad_rule(nsamples) -> x, w

        where x : np.ndarray (nsamples) and w : np.ndarray (nsamples)
        are the quadrature samples and weights respectivelly.
        Note nsamples <= max_nsamples

    """
    basis_opts = define_orthopoly_options_from_marginal(marginal)
    recursion_coeffs = get_recursion_coefficients_from_variable(
        marginal, max_nsamples, basis_opts)
    univariate_quad_rule = partial(gauss_quadrature, recursion_coeffs)
    if canonical:
        return univariate_quad_rule
    return partial(transformed_quadrature_rule, marginal, recursion_coeffs)
