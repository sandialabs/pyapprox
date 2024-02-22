import numpy as np
import numpy.linalg as nlg
import scipy

from functools import partial

from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d, gauss_quadrature
)
from pyapprox.util.utilities import (
    cartesian_product, outer_product,
    integrate_using_univariate_gauss_legendre_quadrature_unbounded
)
from pyapprox.variables.marginals import (
    transform_scale_parameters, is_bounded_continuous_variable
)


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


def lanczos(nodes, weights, N, prob_tol=0):
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

    Lanczos is memory intensive for large numbers of nodes, e.g. from a
    set of samples, memory errors can be thrown.
    '''
    # assert weights define a probability measure. This function
    # can be applied to non-probability measures but I do not use this way
    if abs(weights.sum()-1) > prob_tol+4e-15:
        msg = f"weights sum is {weights.sum()} and so does not define "
        msg += f"a probability measure. Diff : {weights.sum()-1}"
        raise ValueError(msg)
    nnodes = nodes.shape[0]
    assert N <= nnodes
    assert(nnodes == weights.shape[0])
    alpha, beta = np.zeros(N), np.zeros(N)
    vec = np.zeros(nnodes+1)
    vec[0] = 1
    qii = np.zeros((nnodes+1, nnodes+1))
    qii[:, 0] = vec
    sqrt_w = np.sqrt(weights)
    northogonalization_steps = 2
    for ii in range(N):
        z = np.hstack(
            [vec[0]+np.sum(sqrt_w*vec[1:nnodes+1]),
             sqrt_w*vec[0]+nodes*vec[1:nnodes+1]])

        if ii > 0:
            alpha[ii-1] = vec.dot(z)

        for jj in range(northogonalization_steps):
            z -= qii[:, :ii+1].dot(qii[:, :ii+1].T.dot(z))

        if ii < N:
            znorm = np.linalg.norm(z)
            # beta[ii] = znorm**2 assume we want probability measure so
            # no need to square here then take sqrt later
            beta[ii] = znorm
            vec = z / znorm
            qii[:, ii+1] = vec

    alpha = np.atleast_2d(alpha[:N])
    # beta = np.sqrt(np.atleast_2d(beta[:N]))
    beta = np.atleast_2d(beta[:N])
    return np.concatenate((alpha.T, beta.T), axis=1)


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
    # abs_moments = np.absolute(moments)
    # assert abs_moments.max()-abs_moments.min() < 1e16
    ab = modified_chebyshev(nterms, moments, input_coefs)
    return convert_monic_to_orthonormal_recursion_coefficients(ab, probability)


def modified_chebyshev(nterms, moments, input_coefs=None):
    r"""
    Use the modified Chebyshev algorithm to compute the recursion coefficients
    of the monic polynomials p_i(x) orthogonal to a target measure w(x)
    with the modified moments

    math:: \int q_i(x)w(x)dx

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


def predictor_corrector(nterms, measure, lb, ub, quad_options={}):
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

    quad_options : dict
        Options to the numerical quadrature function with attributes

    integrate_fun : callable (optional)
        Function used to compute integrals with signature

        `integrate_fun(lb, ub integrand)`

        If not provided scipy.integrate.quad is used

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

    def integrate_discrete(integrand):
        return integrand(xk).dot(pk)

    import inspect
    sig = inspect.signature(scipy.integrate.quad)
    params = sig.parameters.values()
    for param in params:
        if param.name == "epsabs":
            default_epsabs = param.default
        if param.name == "epsrel":
            default_epsrel = param.default
            break

    def integrate_continuous(integrand):
        res = scipy.integrate.quad(
            integrand, lb, ub, **quad_options)
        atol = quad_options.get("epsabs", default_epsabs)
        rtol = quad_options.get("epsrel", default_epsrel)
        if res[1] > 2*max(atol, res[0]*rtol):
            msg = f"Desired accuracy {atol} was not reached {res[1]}. "
            msg += "Use custom integrator or change quad_options"
            raise RuntimeError(msg)
        return res[0]

    if discrete_measure is True:
        integrate = integrate_discrete
    else:
        if "integrate_fun" in quad_options:
            integrate = partial(quad_options["integrate_fun"], lb, ub)
        else:
            integrate = integrate_continuous

    # To get ab[nterms-1, 0] we must compute ab[nterms, 1]
    # we will truncate ab at the end
    ab = np.zeros((nterms+1, 2))

    # for probablity measures the following will always be one, but
    # this is not true for other measures
    ab[0, 1] = np.sqrt(integrate(measure))

    for ii in range(1, nterms+1):
        # predict
        ab[ii, 1] = ab[ii-1, 1]
        if ii > 1:
            ab[ii-1, 0] = ab[ii-2, 0]
        else:
            ab[ii-1, 0] = 0

        def integrand(measure, x):
            pvals = evaluate_orthonormal_polynomial_1d(
                np.atleast_1d(x), ii, ab)
            return (measure(x)*pvals[:, ii]*pvals[:, ii-1])

        G_ii_iim1 = integrate(partial(integrand, measure))
        ab[ii-1, 0] += ab[ii-1, 1] * G_ii_iim1

        def integrand(measure, x):
            # Note eval orthogonal poly uses the new value for ab[ii, 0]
            # This is the desired behavior
            pvals = evaluate_orthonormal_polynomial_1d(
                np.atleast_1d(x), ii, ab)
            return measure(x)*pvals[:, ii]**2

        G_ii_ii = integrate(partial(integrand, measure))
        ab[ii, 1] *= np.sqrt(G_ii_ii)

    return ab[:nterms, :]


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

    # To get ab[nterms-1, 0] we must compute ab[nterms, 1]
    # we will truncate ab at the end
    ab = np.zeros((nterms+1, 2))
    x_1d = [rule[0] for rule in univariate_quad_rules]
    w_1d = [rule[1] for rule in univariate_quad_rules]
    quad_samples = cartesian_product(x_1d, 1)
    quad_weights = outer_product(w_1d)

    # for probablity measures the following will always be one, but
    # this is not true for other measures
    ab[0, 1] = np.sqrt(quad_weights.sum())

    for ii in range(1, nterms+1):
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

        G_ii_ii = integrand(quad_samples).dot(quad_weights)
        ab[ii, 1] *= np.sqrt(G_ii_ii)

    return ab[:nterms, :]


def predictor_corrector_product_of_functions_of_independent_variables(
        nterms, univariate_quad_rules, funs, loc=0, scale=1):
    nvars = len(univariate_quad_rules)
    assert len(funs) == nvars
    ab = predictor_corrector_function_of_independent_variables(
        nterms, univariate_quad_rules[:2],
        lambda x: funs[0](x[0, :])*funs[1](x[1, :]))

    ll, ss = 0, 1
    for ii in range(2, nvars):
        x, w = gauss_quadrature(ab, nterms)
        if ii == nvars-1:
            ll, ss = loc, scale
        ab = predictor_corrector_function_of_independent_variables(
            nterms, [(x, w), univariate_quad_rules[ii]],
            lambda x: (x[0, :]*funs[ii](x[1, :])-ll)/ss)
    return ab


def apc_normalizing_constant(moments, nterms, monic_coefs):
    assert moments.shape[0] >= 2*nterms+1
    moment_mat = np.zeros((nterms+1, nterms+1))
    for ii in range(nterms+1):
        moment_mat[ii, :] = moments[ii:ii+nterms+1]
    normal_c = np.sqrt(monic_coefs.T.dot(moment_mat).dot(monic_coefs))
    return normal_c


def apc_monic_coefficients(moments, nterms):
    assert moments.shape[0] >= 2*nterms
    moment_mat = np.zeros((nterms+1, nterms+1))
    moment_mat[nterms, nterms] = 1.
    for ii in range(nterms):
        moment_mat[ii, :] = moments[ii:ii+nterms+1]
    rhs = np.zeros(nterms+1)
    rhs[nterms] = 1
    coefs = np.linalg.solve(moment_mat, rhs)
    return coefs


def arbitrary_polynomial_chaos_recursion_coefficients(moments, num_coef):
    moments = np.asarray(moments)
    monic_coefs = np.zeros((num_coef, num_coef))
    normalizing_constants = np.zeros(num_coef)
    for ii in range(num_coef):
        c = apc_monic_coefficients(moments, ii)
        normalizing_constants[ii] = apc_normalizing_constant(moments, ii, c)
        monic_coefs[0:ii+1, ii] = apc_monic_coefficients(
            moments, ii)/normalizing_constants[ii]

    ab = np.zeros((num_coef, 2))
    ab[0, 1] = normalizing_constants[0]
    ab[1, 1] = monic_coefs[0, 0]/monic_coefs[1, 1]
    ab[1, 0] = -monic_coefs[0, 1]/monic_coefs[1, 1]
    for ii in range(2, num_coef):
        ab[ii, 1] = monic_coefs[ii-1, ii-1]/monic_coefs[ii, ii]
        ab[ii, 0] = (monic_coefs[ii-2, ii-1]-ab[ii, 1] *
                     monic_coefs[ii-1, ii])/monic_coefs[ii-1, ii-1]
    return ab


def get_function_independent_vars_recursion_coefficients(opts, num_coefs):
    r"""
    Compute the recursion coefficients orthonormal to the random variable
    arising from arbitrary functions :math:`f(Z_1,\ldots, Z_D)` of
    independent random variables :math:`Z_d`. Tensor product quadrature
    is used to compute the integrals necessary for orthgonalization
    thus this function scales poorly as the number of variables increases.

    Parameters
    ----------
    opts : dictionary
        Dictionary with the following attributes

    fun : callable
        Function that maps the variables to a scalar value

    quad_rules : list (nvars)
        List of univariate quadrature rule sample, weight tuples (x, w)
        for each variable. Each quadrature rules must be in the user domain.

    Returns
    -------
    recursion_coeffs : np.ndarray (num_coefs, 2)
    """
    fun = opts['fun']
    quad_rules = opts['quad_rules']
    loc, scale = opts.get("loc", 0), opts.get("scale", 1)

    def scaled_fun(x):
        return (fun(x)-loc)/scale

    recursion_coeffs = \
        predictor_corrector_function_of_independent_variables(
            num_coefs, quad_rules, scaled_fun)
    return recursion_coeffs


def get_product_independent_vars_recursion_coefficients(opts, num_coefs):
    r"""
    Compute the recursion coefficients orthonormal to the random variable
    arising from the product of univariate functions :math:`f_d(Z_d)` of
    independent random variables :math:`Z_d` , that is

    .. math:: W = \prod_{d=1}^D f_d(Z_d)

    This function first computes recursion coefficients of
    :math:`W_{12}=f_1(Z_1)f_2(Z_2)`. Then uses this to compute a quadrature
    rule which is then used to contruct recursion coefficients for
    :math:`W_{123}=W_{12}f_3(Z_3)` and so on. The same recursion coefficients
    can be obtained using
    :func:`get_function_independent_vars_recursion_coefficients` however this
    function being documented is faster because it
    leverages the seperability of the product.

    Parameters
    ----------
    opts : dictionary
        Dictionary with the following attributes

    funs : list (nvars)
        List of univariate functions of each variable

    quad_rules : list (nvars)
        List of univariate quadrature rule sample, weight tuples (x, w)
        for each variable. Each quadrature rules must be in the user domain.
        The polynomial generated with the recursion coefficients genereated
        here will does not have a notion of canonical domain. Thus when used
        with a variable transformation set the variable index j associated with
        these recursion coefficients to use the identity map via
        var_trans.set_identity_maps([j])

    Returns
    -------
    recursion_coeffs : np.ndarray (num_coefs, 2)

    Todo
    ----
    This function can be generalized to consider compositions of functions of
    independent variable groups, e.g

    math:: W = g_2(g_1(f_{12}(Z_1, Z_2), f_{3,4}(Z_3, Z_4)), f_{5}(Z_5))

    """
    funs = opts['funs']
    quad_rules = opts['quad_rules']
    loc, scale = opts.get("loc", 0), opts.get("scale", 1)
    recursion_coeffs = \
        predictor_corrector_product_of_functions_of_independent_variables(
            num_coefs, quad_rules, funs, loc=loc, scale=scale)
    return recursion_coeffs


def ortho_polynomial_grammian_bounded_continuous_variable(
        var, ab, degree, tol, integrate_fun=None):
    """
    Compute the inner product of all polynomials up to and including
    degree. Useful for testing that the polynomials are orthonormal.
    The grammian should always be the identity (modulo errors due to
    quadrature)
    """
    if ab.shape[0] < degree+1:
        raise ValueError("Not enough recursion coefficients")

    loc, scale = transform_scale_parameters(var)
    if is_bounded_continuous_variable(var):
        can_lb, can_ub = -1, 1
    else:
        lb, ub = var.interval(1)
        can_lb = (lb-loc)/scale
        can_ub = (ub-loc)/scale

    def default_integrate(integrand):
        result = scipy.integrate.quad(
            integrand, can_lb, can_ub, epsabs=tol, epsrel=tol)
        return result[0]

    if integrate_fun is None:
        integrate = default_integrate
    else:
        integrate = partial(integrate_fun, can_lb, can_ub)

    def fun(order1, order2):
        order = max(order1, order2)

        def integrand(x):
            x = np.atleast_1d(x)
            basis_mat = evaluate_orthonormal_polynomial_1d(
                x, order, ab)
            return var.pdf(x*scale+loc)*scale*(
                basis_mat[:, order1]*basis_mat[:, order2])

        return integrate(integrand)

    vec_fun = np.vectorize(fun)
    indices = cartesian_product(
        (np.arange(degree+1), np.arange(degree+1)))
    gram_mat = vec_fun(indices[0, :], indices[1, :])
    return gram_mat.reshape((degree+1, degree+1))


def native_recursion_integrate_fun(
        interval_size, lb, ub, integrand, verbose=0, nquad_samples=50,
        max_steps=1000, tabulated_quad_rules=None, atol=1e-8, rtol=1e-8):
    """
    Parameters
    ----------
    interval_size : float
        Size of the interval used to break up the integrand. Must be in the
        canonical domain of rhe random variable
    """

    # this funciton works well for smooth unbounded variables
    # but scipy.integrate.quad works well for non smooth
    # variables
    val = \
        integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=interval_size,
            verbose=verbose, max_steps=max_steps, atol=atol, rtol=rtol,
            tabulated_quad_rules=tabulated_quad_rules)
    return val
