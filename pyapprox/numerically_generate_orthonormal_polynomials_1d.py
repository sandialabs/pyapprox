import numpy as np
import numpy.linalg as nlg

from functools import partial

from pyapprox.utilities import \
    integrate_using_univariate_gauss_legendre_quadrature_unbounded
from pyapprox.orthonormal_polynomials_1d import \
    evaluate_orthonormal_polynomial_1d, gauss_quadrature
from pyapprox.utilities import cartesian_product, outer_product


def stieltjes(nodes, weights, N):
    """
    Parameters
    ----------
    nodes : np.ndarray (nnodes)
        The locations of the probability masses

    weights : np.ndarray (nnodes)
        The weights of the probability masses

    N : integer
        The desired number of recursion coefficients
    """
    # assert weights define a probability measure. This function
    # can be applied to non-probability measures but I do not use this way
    assert abs(weights.sum()-1) < 1e-15
    nnodes = nodes.shape[0]
    assert N <= nnodes
    ab = np.empty((N, 2))
    sum_0 = np.sum(weights)
    ab[0, 0] = nodes.dot(weights)/sum_0
    ab[0, 1] = sum_0
    p1 = np.zeros(nnodes)
    p2 = np.ones((nnodes))
    for k in range(N-1):
        p0 = p1
        p1 = p2
        p2 = (nodes-ab[k, 0])*p1-ab[k, 1]*p0
        sum_1 = weights.dot(p2**2)
        sum_2 = nodes.dot(weights*p2**2)
        ab[k+1, 0] = sum_2/sum_1
        ab[k+1, 1] = sum_1/sum_0
        sum_0 = sum_1
    ab[:, 1] = np.sqrt(ab[:, 1])
    ab[0, 1] = 1
    return ab


def lanczos(nodes, weights, N):
    '''
    Parameters
    ----------
    nodes : np.ndarray (nnodes)
        The locations of the probability masses

    weights : np.ndarray (nnodes)
        The weights of the probability masses

    N : integer
        The desired number of recursion coefficients

    This algorithm was proposed in ``The numerically stable reconstruction of 
    Jacobi matrices from spectral data'',  Numer. Math. 44 (1984), 317-335.
    See Algorithm RPKW on page 328.

    \bar{\alpha} are the current estimates of the recursion coefficients alpha
    \bar{\beta}  are the current estimates of the recursion coefficients beta
    '''
    # assert weights define a probability measure. This function
    # can be applied to non-probability measures but I do not use this way
    assert abs(weights.sum()-1) < 1e-15
    nnodes = nodes.shape[0]
    assert N <= nnodes
    assert(nnodes == weights.shape[0])
    alpha, beta = np.zeros(N), np.zeros(N)
    vec = np.zeros(nnodes+1)
    vec[0] = 1
    qii = np.zeros((nnodes+1, nnodes+1))
    qii[:, 0] = vec
    sqrt_w = np.sqrt(weights)
    for ii in range(N):
        z = np.hstack(
            [vec[0]+np.sum(sqrt_w*vec[1:nnodes+1]),
             sqrt_w*vec[0]+nodes*vec[1:nnodes+1]])

        if ii > 0:
            alpha[ii-1] = vec.dot(z)

        z -= qii[:, :ii+1].dot(qii[:, :ii+1].T.dot(z))
        z -= qii[:, :ii+1].dot(qii[:, :ii+1].T.dot(z))

        if ii < N:
            znorm = np.linalg.norm(z)
            beta[ii] = znorm**2
            vec = z / znorm
            qii[:, ii+1] = vec

    alpha = np.atleast_2d(alpha[:N])
    beta = np.atleast_2d(beta[:N])
    return np.concatenate((alpha.T, np.sqrt(beta.T)), axis=1)


def lanczos_deprecated(mat, vec):
    # knobs
    symTol = 1.e-8
    # check square matrix
    assert(mat.shape[0] == mat.shape[1])
    # check symmetric matrix
    assert(np.allclose(mat, mat.T, atol=symTol))
    m, n = mat.shape
    k = n
    Q = np.empty((n, k))
    Q[:] = np.nan
    q = vec / nlg.norm(vec)
    Q[:, 0] = q
    d = np.empty(k)
    od = np.empty(k-1)
    d[:] = np.nan
    od[:] = np.nan
    # print(mat)
    for i in range(k):
        z = mat.dot(q)
        # print(z)
        d[i] = np.dot(q, z)
        z = z - np.dot(Q[:, :i+1], np.dot(Q[:, :i+1].T, z))
        z = z - np.dot(Q[:, :i+1], np.dot(Q[:, :i+1].T, z))
        if (i != k-1):
            od[i] = nlg.norm(z)
            q = z / od[i]
            Q[:, i + 1] = q
    od[0] = 1.0
    d = np.atleast_2d(d[1:])
    od = np.atleast_2d(od)
    return np.concatenate((d.T, od.T), axis=1)
    # return (d,od)


def convert_monic_to_orthonormal_recursion_coefficients(ab_monic, probability):
    assert np.all(ab_monic[:, 1] >= 0)
    ab = ab_monic.copy()
    ab[:, 1] = np.sqrt(ab[:, 1])
    if probability:
        ab[0, 1] = 1
    return ab


def evaluate_monic_polynomial_1d(x, nmax, ab):
    r"""
    Evaluate univariate monic polynomials using their
    three-term recurrence coefficients. A monic polynomial is a polynomial 
    in which the coefficient of the highest degree term is 1.

    Parameters
    ----------
    x : np.ndarray (num_samples)
       The samples at which to evaluate the polynomials

    nmax : integer
       The maximum degree of the polynomials to be evaluated

    ab : np.ndarray (num_recusion_coeffs,2)
       The recursion coefficients. num_recusion_coeffs>degree

    Returns
    -------
    p : np.ndarray (num_samples, nmax+1)
       The values of the polynomials
    """
    p = np.zeros((x.shape[0], nmax+1), dtype=float)

    p[:, 0] = 1/ab[0, 1]

    if nmax > 0:
        p[:, 1] = (x - ab[0, 0])*p[:, 0]

    for jj in range(2, nmax+1):
        p[:, jj] = (x-ab[jj-1, 0])*p[:, jj-1]-ab[jj-1, 1]*p[:, jj-2]

    return p


def modified_chebyshev_orthonormal(nterms, quadrature_rule,
                                   get_input_coefs=None, probability=True):
    """
    Use the modified Chebyshev algorithm to compute the recursion coefficients
    of the orthonormal polynomials p_i(x) orthogonal to a target measure w(x) 
    with the modified moments

    int q_i(x)w(x)dx

    where q_i are orthonormal polynomials with recursion coefficients given
    by input_coefs.

    Parameters
    ----------
    nterms : integer
        The number of desired recursion coefficients

    quadrature_rule : list [x,w]
        The quadrature points and weights used to compute the
        modified moments of the target measure. The recursion coefficients (ab)
        returned by the modified_chebyshev_orthonormal function 
        will be orthonormal to this measure.

    get_input_coefs : callable
        Function that returns the recursion coefficients of the polynomials 
        which are orthogonal to a measure close to the target measure.
        If None then the moments of monomials will be computed.
        Call signature get_input_coefs(n,probability=False). 
        Functions in this package return orthogonal polynomials which are 
        othonormal if probability=True. The modified Chebyshev algorithm
        requires monic polynomials so we must set probability=False then 
        compute the orthogonal polynomials to monic polynomials. Coefficients
        of orthonormal polynomials cannot be converted to the coefficients 
        of monic polynomials.

    probability : boolean
        If True return coefficients of orthonormal polynomials
        If False return coefficients of orthogonal polynomials

    Returns
    -------
    ab : np.ndarray (nterms)
        The recursion coefficients of the orthonormal polynomials orthogonal 
        to the target measure
    """
    quad_x, quad_w = quadrature_rule
    if get_input_coefs is None:
        moments = [np.dot(quad_x**n, quad_w) for n in range(2*nterms)]
        input_coefs = None
    else:
        input_coefs = get_input_coefs(2*nterms, probability=False)
        # convert to monic polynomials
        input_coefs[:, 1] = input_coefs[:, 1]**2
        basis_matrix = evaluate_monic_polynomial_1d(
            quad_x, 2*nterms-1, input_coefs)
        moments = [basis_matrix[:, ii].dot(quad_w)
                   for ii in range(basis_matrix.shape[1])]

    # check if the range of moments is reasonable. If to large
    # can cause numerical problems
    abs_moments = np.absolute(moments)
    #assert abs_moments.max()-abs_moments.min() < 1e16
    ab = modified_chebyshev(nterms, moments, input_coefs)
    return convert_monic_to_orthonormal_recursion_coefficients(ab, probability)


def modified_chebyshev(nterms, moments, input_coefs=None):
    """
    Use the modified Chebyshev algorithm to compute the recursion coefficients
    of the monic polynomials p_i(x) orthogonal to a target measure w(x) 
    with the modified moments

    int q_i(x)w(x)dx

    where q_i are monic polynomials with recursion coefficients given
    by input_coefs.

    Parameters
    ----------
    nterms : integer
        The number of desired recursion coefficients

    moments : np.ndarray (nmoments)
        Modified moments of the target measure. The recursion coefficients 
        returned by this function will be orthonormal to this measure.

    input_coefs : np.ndarray (nmoments,2)
        The recursion coefficients of the monic polynomials which are 
        orthogonal to a measure close to the target measure.
        Ensure nmoments>=2*nterms-1. If None then use monomials with
        input_coeff = np.zeros((nmoments,2))

    Returns
    -------
    ab : np.ndarray (nterms)
        The recursion coefficients of the monic polynomials orthogonal to the
        target measure
    """
    moments = np.asarray(moments)
    assert moments.ndim == 1
    nmoments = moments.shape[0]
    if nterms > nmoments/2:
        msg = 'nterms and nmoments are inconsistent. '
        msg += 'Not enough moments specified'
        raise Exception(msg)

    if input_coefs is None:
        # the unmodified polynomials are monomials
        input_coefs = np.zeros((nmoments, 2))

    assert nmoments == input_coefs.shape[0]

    ab = np.zeros((nterms, 2))
    ab[0, 0] = input_coefs[0, 0]+moments[1]/moments[0]
    ab[0, 1] = moments[0]

    sigma = np.zeros(2*nterms)
    sigma_m2 = np.zeros_like(sigma)
    sigma_m1 = moments[0:2*nterms].copy()
    for kk in range(1, nterms):
        sigma[:] = 0
        idx = 2*nterms-kk
        for ll in range(kk, 2*nterms-kk):
            sigma[kk:idx] = sigma_m1[kk+1:idx+1] -\
                (ab[kk-1, 0]-input_coefs[kk:idx, 0])*sigma_m1[kk:idx] -\
                ab[kk-1, 1]*sigma_m2[kk:idx]+input_coefs[kk:idx, 1] *\
                sigma_m1[kk-1:idx-1]
            # sigma[ll]=sigma_m1[ll+1]-\
            #    (ab[kk-1,0]-input_coefs[ll,0])*sigma_m1[ll] - \
            #    ab[kk-1,1]*sigma_m2[ll]+input_coefs[ll,1]*sigma_m1[ll-1]
        ab[kk, 0] = input_coefs[kk, 0]+sigma[kk+1]/sigma[kk] -\
            sigma_m1[kk]/sigma_m1[kk-1]
        ab[kk, 1] = sigma[kk]/sigma_m1[kk-1]
        sigma_m2 = sigma_m1.copy()
        sigma_m1 = sigma.copy()

    return ab


def predictor_corrector_known_scipy_pdf(nterms, rv, quad_options={}):
    lb, ub = rv.interval(1)
    if np.isfinite(lb) and np.isfinite(ub):
        interval_size = ub-lb
    else:
        interval_size = (rv.interval(0.99)[1] - rv.interval(0.99)[0])
    return predictor_corrector(
        nterms, rv.pdf, lb, ub, interval_size, quad_options)
    

def predictor_corrector(nterms, measure, lb, ub, interval_size=1,
                        quad_options={}):
    """
    Use predictor corrector method to compute the recursion coefficients
    of a univariate orthonormal polynomial

    Parameters
    ----------
    nterms : integer
        The number of coefficients requested

    measure : callable or tuple
        The function (measure) used to compute orthogonality. 
        If a discrete measure then measure = tuple(xk, pk) where
        xk are the probability masses locoation and pk are the weights
        

    lb: float
        The lower bound of the measure (can be -infinity)

    ub: float
        The upper bound of the measure (can be infinity)

    interval_size : float
        The size of the initial interval used for quadrature
        For bounded variables this should be ub-lb. For semi- or un-bounded
        variables the larger this value the larger nquad_samples should
        be set

    quad_options : dict
        Options to the numerical quadrature function with attributes

    nquad_samples : integer
        The number of samples in the Gauss quadrature rule

    Note the entry ab[-1, :] will likely be wrong when compared to analytical
    formula if they exist. This does not matter because eval_poly does not 
    use this value. If you want the correct value just request num_coef+1
    coefficients.
    """

    discrete_measure = not callable(measure)
    if discrete_measure is True:
        xk, pk = measure
        assert xk.shape[0] == pk.shape[0]
        assert nterms < xk.shape[0]
        def measure(x):
            return np.ones_like(x)
    
    ab = np.zeros((nterms, 2))
    nquad_samples = quad_options.get('nquad_samples', 100)
    quad_opts = quad_options.copy()
    if 'nquad_samples' in quad_opts:
        del quad_opts['nquad_samples']

    if np.isfinite(lb) and np.isfinite(ub):
        assert interval_size == ub-lb

    def integrate_continuous(integrand, nquad_samples, interval_size):
        return integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, **quad_opts,
            interval_size=interval_size)

    def integrate_discrete(integrand, nquad_samples, interval_size):
        return integrand(xk).dot(pk)

    if discrete_measure is True:
        integrate = integrate_discrete
    else:
        integrate = integrate_continuous

    # for probablity measures the following will always be one, but 
    # this is not true for other measures
    ab[0, 1] = np.sqrt(
        integrate(measure, nquad_samples, interval_size=interval_size))

    for ii in range(1, nterms):
        # predict
        ab[ii, 1] = ab[ii-1, 1]
        if ii > 1:
            ab[ii-1, 0] = ab[ii-2, 0]
        else:
            ab[ii-1, 0] = 0

        if np.isfinite(lb) and np.isfinite(ub) and ii > 1:
            # use previous intervals size for last degree as initial guess
            # of size needed here
            xx, __ = gauss_quadrature(ab, nterms)
            interval_size = xx.max()-xx.min()

        def integrand(measure, x):
            pvals = evaluate_orthonormal_polynomial_1d(x, ii, ab)
            return measure(x)*pvals[:, ii]*pvals[:, ii-1]        
        G_ii_iim1 = integrate(
            partial(integrand, measure), nquad_samples+ii,
            interval_size=interval_size)
        ab[ii-1, 0] += ab[ii-1, 1] * G_ii_iim1
        
        def integrand(measure, x):
            # Note eval orthogonal poly uses the new value for ab[ii, 0]
            # This is the desired behavior
            pvals = evaluate_orthonormal_polynomial_1d(x, ii, ab)
            return measure(x)*pvals[:, ii]**2
        
        G_ii_ii =  integrate(
            partial(integrand, measure), nquad_samples+ii,
            interval_size=interval_size)
        ab[ii, 1] *= np.sqrt(G_ii_ii)

    return ab


def predictor_corrector_function_of_independent_variables(
        nterms, univariate_quad_rules, fun):
    """
    Use predictor corrector method to compute the recursion coefficients
    of a univariate orthonormal polynomial orthogonal to the density
    associated with a scalar function of a set of independent 1D
    variables

    Parameters
    ----------
    nterms : integer
        The number of coefficients requested

    univariate_quad_rules : callable
        The univariate quadrature rules which include weights of
        each indendent variable

    fun : callable
        The function mapping indendent variables into a scalar variable
    """
    
    ab = np.zeros((nterms, 2))
    x_1d = [rule[0] for rule in  univariate_quad_rules]
    w_1d = [rule[1] for rule in  univariate_quad_rules]
    quad_samples = cartesian_product(x_1d, 1)
    quad_weights = outer_product(w_1d)

    # for probablity measures the following will always be one, but 
    # this is not true for other measures
    ab[0, 1] = np.sqrt(quad_weights.sum())

    for ii in range(1, nterms):
        # predict
        ab[ii, 1] = ab[ii-1, 1]
        if ii > 1:
            ab[ii-1, 0] = ab[ii-2, 0]
        else:
            ab[ii-1, 0] = 0

        def integrand(x):
            y = fun(x).squeeze()
            pvals = evaluate_orthonormal_polynomial_1d(y, ii, ab)
            # measure not included in integral because it is assumed to
            # be in the quadrature rules
            return pvals[:, ii]*pvals[:, ii-1]
            
        G_ii_iim1 = integrand(quad_samples).dot(quad_weights)
        ab[ii-1, 0] += ab[ii-1, 1] * G_ii_iim1
        
        def integrand(x):
            y = fun(x).squeeze()
            pvals = evaluate_orthonormal_polynomial_1d(y, ii, ab)
            # measure not included in integral because it is assumed to
            # be in the quadrature rules
            return pvals[:, ii]**2
        G_ii_ii =  integrand(quad_samples).dot(quad_weights)
        ab[ii, 1] *= np.sqrt(G_ii_ii)

    return ab


def predictor_corrector_product_of_functions_of_independent_variables(
        nterms, univariate_quad_rules, funs):
    nvars = len(univariate_quad_rules)
    assert len(funs) == nvars
    ab = predictor_corrector_function_of_independent_variables(
        nterms, univariate_quad_rules[:2],
        lambda x: funs[0](x[0, :])*funs[1](x[1, :]))
    for ii in range(2, nvars):
        x, w = gauss_quadrature(ab, nterms)
        ab = predictor_corrector_function_of_independent_variables(
            nterms, [(x, w), univariate_quad_rules[ii]],
            lambda x: x[0, :]*funs[ii](x[1,:]))
    return ab
        
        
