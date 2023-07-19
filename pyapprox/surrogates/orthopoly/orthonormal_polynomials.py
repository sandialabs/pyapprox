import numpy as np

from pyapprox.util.pya_numba import njit, gammaln_float64
from pyapprox.variables.marginals import get_distribution_info


def evaluate_orthonormal_polynomial_1d(x, nmax, ab):
    if nmax > ab.shape[0]:
        raise ValueError("Too many terms requested")

    try:
        # necessary when discrete variables are define on integers
        x = np.asarray(x, dtype=float)
        from pyapprox.cython.orthonormal_polynomials_1d import \
            evaluate_orthonormal_polynomial_1d_pyx
        return evaluate_orthonormal_polynomial_1d_pyx(x, nmax, ab)
        # from pyapprox.weave import c_evaluate_orthonormal_polynomial
        # return c_evaluate_orthonormal_polynomial_1d(x, nmax, ab)
    except ImportError:
        print('evaluate_orthornormal_polynomial_1d extension failed')
    return __evaluate_orthonormal_polynomial_1d(x, nmax, ab)


def evaluate_orthonormal_polynomial_deriv_1d(x, nmax, ab, deriv_order):
    if nmax > ab.shape[0]:
        raise ValueError("Too many terms requested")

    # filter out cython warnings.
    import warnings
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    # warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    # warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    try:
        # necessary when discrete variables are define on integers
        x = np.asarray(x, dtype=float)
        from pyapprox.cython.orthonormal_polynomials_1d import \
            evaluate_orthonormal_polynomial_deriv_1d_pyx
        return evaluate_orthonormal_polynomial_deriv_1d_pyx(
            x, nmax, ab, deriv_order)
    except (ImportError, ValueError):
        print('evaluate_orthonormal_polynomial_deriv_1d_pyx extension failed')
    return __evaluate_orthonormal_polynomial_deriv_1d(x, nmax, ab, deriv_order)


@njit(cache=True)
def numba_gammaln(x):
    return gammaln_float64(x)


@njit(cache=True)
def __evaluate_orthonormal_polynomial_1d(x, nmax, ab):
    r"""
    Evaluate univariate orthonormal polynomials using their
    three-term recurrence coefficients.

    The the degree-n orthonormal polynomial p_n(x) is associated with
    the recurrence coefficients a, b (with positive leading coefficient)
    satisfy the recurrences

    .. math:: b_{n+1} p_{n+1} = (x - a_n) p_n - \sqrt{b_n} p_{n-1}

    This assumes that the orthonormal recursion coefficients satisfy

    .. math:: b_{n+1} = \sqrt{\hat{b}_{n+1}}

    where :math:`\hat{b}_{n+1}` are the orthogonal recursion coefficients.

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
    assert ab.shape[1] == 2
    assert nmax < ab.shape[0]

    p = np.zeros((x.shape[0], nmax+1))  # must be initialized to zero

    p[:, 0] = 1/ab[0, 1]

    if nmax > 0:
        p[:, 1] = 1/ab[1, 1] * ((x - ab[0, 0])*p[:, 0])

    for jj in range(2, nmax+1):
        p[:, jj] = 1.0/ab[jj, 1]*(
            (x-ab[jj-1, 0])*p[:, jj-1]-ab[jj-1, 1]*p[:, jj-2])

    return p


@njit(cache=True)
def __evaluate_orthonormal_polynomial_deriv_1d(x, nmax, ab, deriv_order):
    r"""
    Evaluate the univariate orthonormal polynomials and its s-derivatives
    (s=1,...,num_derivs) using a three-term recurrence coefficients.

    The the degree-n orthonormal polynomial p_n(x) is associated with
    the recurrence coefficients a, b (with positive leading coefficient)
    satisfy the recurrences

    .. math:: b_{n+1} p_{n+1} = (x - a_n) p_n - \sqrt{b_n} p_{n-1}

    This assumes that the orthonormal recursion coefficients satisfy

    .. math:: b_{n+1} = \sqrt{\hat{b}_{n+1}}

    where :math:`\hat{b}_{n+1}` are the orthogonal recursion coefficients.

    Parameters
    ----------
    x : np.ndarray (num_samples)
       The samples at which to evaluate the polynomials

    nmax : integer
       The maximum degree of the polynomials to be evaluated

    ab : np.ndarray (num_recursion_coeffs,2)
       The recursion coefficients

    deriv_order : integer
       The maximum order of the derivatives to evaluate.

    Returns
    -------
    p : np.ndarray (num_samples, (deriv_num+1)*num_indices)
       The values of the 0th to s-th derivative of the polynomials
    """
    num_samples = x.shape[0]
    num_indices = nmax+1
    a = ab[:, 0]
    b = ab[:, 1]
    result = np.empty((num_samples, num_indices*(deriv_order+1)))
    p = __evaluate_orthonormal_polynomial_1d(x, nmax, ab)
    result[:, :num_indices] = p

    for deriv_num in range(1, deriv_order+1):
        pd = np.zeros((num_samples, num_indices))
        for jj in range(deriv_num, num_indices):

            if (jj == deriv_num):
                # use following expression to avoid overflow issues when
                # computing oveflow
                pd[:, jj] = np.exp(
                    # sp.gammaln(deriv_num+1)-0.5*np.sum(np.log(b[:jj+1]**2)))
                    numba_gammaln(deriv_num+1)-0.5*np.sum(np.log(b[:jj+1]**2)))
            else:

                pd[:, jj] =\
                    (x-a[jj-1])*pd[:, jj-1]-b[jj-1]*pd[:, jj-2] +\
                    deriv_num*p[:, jj-1]
                pd[:, jj] *= 1.0/b[jj]
        p = pd
        result[:, deriv_num*num_indices:(deriv_num+1)*num_indices] = p
    return result


def gauss_quadrature(recursion_coeffs, N):
    r"""Computes Gauss quadrature from recurrence coefficients

       x,w = gauss_quadrature(recursion_coeffs,N)

    Computes N Gauss quadrature nodes (x) and weights (w) from
    standard orthonormal recurrence coefficients.

    Parameters
    ----------
    recursion_coeffs : np.ndarray (num_recursion_coeffs,2)
       The recursion coefficients

    N : integer
       Then number of quadrature points

    Returns
    -------
    x : np.ndarray (N)
       The quadrature points

    w : np.ndarray (N)
       The quadrature weights
    """
    if N > recursion_coeffs.shape[0]:
        raise ValueError("Too many terms requested")

    a = recursion_coeffs[:, 0]
    b = recursion_coeffs[:, 1]

    # Form Jacobi matrix
    J = np.diag(a[:N], 0)+np.diag(b[1:N], 1)+np.diag(b[1:N], -1)
    x, eigvecs = np.linalg.eigh(J)

    w = b[0]*eigvecs[0, :]**2

    # w = evaluate_orthonormal_polynomial_1d(x, N-1, recursion_coeffs)
    # w = 1./np.sum(w**2, axis=1)

    w[~np.isfinite(w)] = 0.
    return x, w


def convert_orthonormal_polynomials_to_monomials_1d(ab, nmax):
    r"""
    Get the monomial expansion of each orthonormal basis up to a given
    degree.

    Parameters
    ----------
    ab : np.ndarray (num_recursion_coeffs,2)
       The recursion coefficients

    nmax : integer
       The maximum degree of the polynomials to be evaluated (N+1)

    Returns
    -------
    monomial_coefs : np.ndarray (nmax+1,nmax+1)
        The coefficients of :math:`x^i, i=0,...,N` for each orthonormal basis
        :math:`p_j` Each row is the coefficients of a single basis :math:`p_j`.
    """
    assert nmax < ab.shape[0]

    monomial_coefs = np.zeros((nmax+1, nmax+1))

    monomial_coefs[0, 0] = 1/ab[0, 1]

    if nmax > 0:
        monomial_coefs[1, :2] = np.array(
            [-ab[0, 0], 1])*monomial_coefs[0, 0]/ab[1, 1]

    for jj in range(2, nmax+1):
        monomial_coefs[jj, :jj] += (
            - ab[jj-1, 0]*monomial_coefs[jj-1, :jj]
            - ab[jj-1, 1]*monomial_coefs[jj-2, :jj])/ab[jj, 1]
        monomial_coefs[jj, 1:jj+1] += monomial_coefs[jj-1, :jj]/ab[jj, 1]
    return monomial_coefs


def evaluate_three_term_recurrence_polynomial_1d(abc, nmax, x):
    r"""
    Evaluate an orthogonal polynomial three recursion coefficient formulation

    .. math:: p_{n+1} = \tilde{a}_{n+1}x - \tilde{b}_np_n - \tilde{c}_n p_{n-1}

    Parameters
    ----------
    abc : np.ndarray (num_recursion_coeffs,3)
       The recursion coefficients

    nmax : integer
       The maximum degree of the polynomials to be evaluated (N+1)

    x : np.ndarray (num_samples)
       The samples at which to evaluate the polynomials

    Returns
    -------
    p : np.ndarray (num_samples, num_indices)
       The values of the polynomials at the samples
    """
    assert nmax < abc.shape[0]

    p = np.zeros((x.shape[0], nmax+1), dtype=float)

    p[:, 0] = abc[0, 0]

    if nmax > 0:
        p[:, 1] = (abc[1, 0]*x - abc[1, 1])*p[:, 0]

    for jj in range(2, nmax+1):
        p[:, jj] = (abc[jj, 0]*x-abc[jj, 1])*p[:, jj-1]-abc[jj, 2]*p[:, jj-2]

    return p


def shift_momomial_expansion(coef, shift, scale):
    assert coef.ndim == 1
    shifted_coef = np.zeros_like(coef)
    shifted_coef[0] = coef[0]
    nterms = coef.shape[0]
    for ii in range(1, nterms):
        temp = np.polynomial.polynomial.polypow([1, -shift], ii)
        shifted_coef[:ii+1] += coef[ii]*temp[::-1]/scale**ii
    return shifted_coef


def convert_orthonormal_expansion_to_monomial_expansion_1d(ortho_coef, ab,
                                                           shift, scale):
    r"""
    Convert a univariate orthonormal polynomial expansion

    .. math:: f(x)=\sum_{i=1}^N c_i\phi_i(x)

    into the equivalent monomial expansion.

    .. math:: f(x)=\sum_{i=1}^N d_ix^i

    Parameters
    ----------
    ortho_coef : np.ndarray (N)
        The expansion coeficients :math:`c_i`

    ab : np.ndarray (num_recursion_coeffs,2)
       The recursion coefficients of the polynomial family :math:`\phi_i`

    shift : float
       Parameter used to shift the orthonormal basis, which is defined on
       some canonical domain, to a desired domain

    scale : float
       Parameter used to scale the orthonormal basis, which is defined on
       some canonical domain, to a desired domain

    Returns
    -------
    mono_coefs : np.ndarray (N)
        The coefficients :math:`d_i` of the monomial basis
    """
    assert ortho_coef.ndim == 1
    # get monomial expansion of each orthonormal basis
    basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
        ab, ortho_coef.shape[0]-1)
    # scale each monomial coefficient by the corresponding orthonormal
    # expansion coefficients and collect terms
    mono_coefs = np.sum(basis_mono_coefs.T*ortho_coef, axis=1)
    # the orthonormal basis is defined on canonical domain so
    # shift to desired domain
    mono_coefs = shift_momomial_expansion(mono_coefs, shift, scale)
    return mono_coefs


def define_orthopoly_options_from_marginal(marginal):
    name, scales, shapes = get_distribution_info(marginal)
    opts = {'var': marginal, 'shapes': shapes}
    return opts
