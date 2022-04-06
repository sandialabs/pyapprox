import numpy as np

from pyapprox.surrogates.interp.indexing import (
    argsort_indices_leixographically, compute_hyperbolic_level_indices
)
from pyapprox.util.pya_numba import njit


@njit(cache=True)
def multiply_multivariate_polynomials(indices1, coeffs1, indices2, coeffs2):
    """
    TODO: instead of using dictionary to colect terms consider using

    unique_indices, repeated_idx = np.unique(
        indices[active_idx,:], axis=1, return_inverse=True)

    as is done in
    multivariate_polynomials.conditional_moments_of_polynomial_chaos_expansion.
    Choose which one is faster


    Parameters
    ----------
    index : multidimensional index
        multidimensional index specifying the polynomial degree in each
        dimension

    Returns
    -------
    """
    num_vars = indices1.shape[0]
    num_indices1 = indices1.shape[1]
    num_indices2 = indices2.shape[1]
    nqoi = coeffs1.shape[1]
    assert num_indices1 == coeffs1.shape[0]
    assert num_indices2 == coeffs2.shape[0]
    assert num_vars == indices2.shape[0]
    assert nqoi == coeffs2.shape[1]

    # using np.unique inside group like terms is much more expensive
    # than using dictionary
    # max_num_indices = num_indices1*num_indices2
    # indices = []
    # coeffs = []
    # for ii in range(num_indices1):
    #     index1 = indices1[:, ii]
    #     coeff1 = coeffs1[ii, :]
    #     for jj in range(num_indices2):
    #         indices.append(index1 + indices2[:, jj])
    #         coeffs.append(coeff1*coeffs2[jj, :])

    # return group_like_terms(np.array(coeffs), np.array(indices).T)

    indices_dict = dict()
    max_num_indices = num_indices1*num_indices2
    indices = np.empty((num_vars, max_num_indices), dtype=np.int64)
    coeffs = np.empty((max_num_indices, nqoi), dtype=np.double)
    kk = 0
    for ii in range(num_indices1):
        index1 = indices1[:, ii]
        coeff1 = coeffs1[ii]
        for jj in range(num_indices2):
            index = index1+indices2[:, jj]
            # hash_array does not work with jit
            # key = hash_array(index)
            # so use a polynomial hash
            key = 0
            for dd in range(index.shape[0]):
                key = 31*key + int(index[dd])
            coeff = coeff1*coeffs2[jj]
            if key in indices_dict:
                coeffs[indices_dict[key]] += coeff
            else:
                indices_dict[key] = kk
                indices[:, kk] = index
                coeffs[kk] = coeff
                kk += 1
    indices = indices[:, :kk]
    coeffs = coeffs[:kk]
    return indices, coeffs


@njit(cache=True)
def coeffs_of_power_of_nd_linear_monomial(num_vars, degree, linear_coeffs):
    """
    Compute the monomial (coefficients and indices) obtained by raising
    a linear multivariate monomial (no constant term) to some power.

    Parameters
    ----------
    num_vars : integer
        The number of variables

    degree : integer
        The power of the linear polynomial

    linear_coeffs: np.ndarray (num_vars)
        The coefficients of the linear polynomial

    Returns
    -------
    coeffs: np.ndarray (num_terms)
        The coefficients of the new polynomial

    indices : np.ndarray (num_vars, num_terms)
        The set of multivariate indices that define the new polynomial
    """
    assert len(linear_coeffs) == num_vars
    coeffs, indices = multinomial_coeffs_of_power_of_nd_linear_monomial(
        num_vars, degree)
    for ii in range(indices.shape[1]):
        index = indices[:, ii]
        for dd in range(num_vars):
            degree = index[dd]
            coeffs[ii] *= linear_coeffs[dd]**degree
    return coeffs, indices


def precompute_polynomial_powers(max_pow, indices, coefs, var_idx,
                                 global_var_idx, num_global_vars):
    """
    Raise a polynomial to all powers 0,..., N.
    E.g. compute 1, p(x), p(x)^2, p(x)^3, ... p(x)^N
    """
    if max_pow < 0:
        raise exception('max_pow must >= 0')

    # store input indices in global_var_idx
    assert indices.shape[0] == global_var_idx.shape[0]
    ind = np.zeros((num_global_vars, indices.shape[1]))
    ind[global_var_idx, :] = indices

    polys = [coeffs_of_power_of_monomial(ind, coefs, 0)]
    for nn in range(1, max_pow+1):
        polys.append(multiply_multivariate_polynomials(
            ind, coefs, polys[-1][0], polys[-1][1]))
    return polys


def substitute_polynomials_for_variables_in_another_polynomial(
        indices_in, coeffs_in, indices, coeffs, var_idx, global_var_idx):
    """
    Substitute multiple polynomials representing input variables of another
    polynomial. All polynomials are assumed to be monomials.

    Parameters
    ----------
    indices_in : list
        List of the polynomial indices for each input stored in a
        np.ndarray (nvars_i, nterms_i) i=0,...,len(indices_in)-1

    coeffs_in : list
        List of the polynomial coefficients for each input stored in a
        np.ndarray (nterms_i, nqoi_i) i=0,...,len(indices_in)-1

    indices : np.ndarray (nvars, nterms)
        The polynomial indices of the downstream polynomial which
        we are substituting into

    coeffs : np.ndarray (nterms, nqoi)
        The polynomial coefficients of the downstream polynomial which
        we are substituting into

    var_idx : np.ndarray (ninputs)
        The indices of the variable we are replacing by a polynomial

    global_var_idx : [np.ndarray(nvars[ii]) for ii in ninputs]
        The index of the active variables for each input.
        Note the number of parameters of the final polynomial will likely be
        greater than the number of global variables of the downstream polynomial
        E.g if y2 = y1*x3 and y1 = x1*x2 then y2 is a function of x1,x2,x3
        despite being only parameterized by two variables y1 and x3
    """
    unique_global_vars_in = np.unique(np.concatenate(tuple(global_var_idx)))
    num_global_vars = unique_global_vars_in.shape[0] + (
        indices.shape[0]-len(global_var_idx))
    num_inputs = var_idx.shape[0]
    assert num_inputs == len(indices_in)
    assert num_inputs == len(coeffs_in)
    assert var_idx.max() < num_global_vars
    assert len(global_var_idx) == num_inputs

    # precompute polynomial powers which will be used repeatedly
    input_poly_powers = []
    for jj in range(num_inputs):
        max_pow = indices[var_idx[jj], :].max()
        input_poly_powers.append(
            precompute_polynomial_powers(
                max_pow, indices_in[jj], coeffs_in[jj], var_idx[jj],
                global_var_idx[jj], num_global_vars))

    # get global indices that will not be not substituted
    mask = np.ones(num_global_vars, dtype=bool)
    mask[unique_global_vars_in] = False
    mask2 = np.ones(indices.shape[0], dtype=bool)
    mask2[np.unique(var_idx)] = False

    num_vars, num_terms = indices.shape
    new_indices = []
    new_coeffs = []
    for ii in range(num_terms):
        basis_index = indices[:, ii:ii+1]
        # The following is more memory efficient but does not reuse information
        # computed multiple times
        # ind, cf = substitute_polynomials_for_variables_in_single_basis_term(
        #   indices_in, coeffs_in, basis_index, coeffs[ii], var_idx,
        #   global_var_idx)

        degree = basis_index[var_idx[0], 0]
        ind, cf = input_poly_powers[0][degree]
        for jj in range(1, num_inputs):
            degree = basis_index[var_idx[jj], 0]
            ind2, cf2 = input_poly_powers[jj][degree]
            ind, cf = multiply_multivariate_polynomials(ind, cf, ind2, cf2)

        # multiply all terms by these remaining variables
        ind[mask, :] += basis_index[mask2]
        cf *= coeffs[ii]
        new_indices.append(ind)
        new_coeffs.append(cf)

    new_indices = np.hstack(new_indices)
    new_coeffs = np.vstack(new_coeffs)

    return group_like_terms(new_coeffs, new_indices)
    # unique_indices, repeated_idx = np.unique(
    #     new_indices, axis=1, return_inverse=True)
    # unique_coef = np.zeros((unique_indices.shape[1], new_coeffs.shape[1]))
    # for ii in range(repeated_idx.shape[0]):
    #     unique_coef[repeated_idx[ii]] += new_coeffs[ii]
    # return unique_indices, unique_coef


def substitute_polynomials_for_variables_in_single_basis_term(
        indices_in, coeffs_in, basis_index, basis_coeff, var_idx,
        global_var_idx):
    """
    Parameters
    ----------
    indices_in : np.ndarray (nvars_in, nterms_in)
        The polynomial indices for the polynomial which we will
        subsitute in

    coeffs_in : np.ndarray (nterms_in, nqoi_in)
        The polynomial coefficients for the polynomial which we will
        subsitute in

    basis_index : np.ndarray (nvars, 1)
        The degrees of the basis term we are acting on.

    basis_coeff : np.ndarray (nqoi)
        The coefficients of the basis

    var_idx : np.ndarray (nsub_vars)
        The dimensions in basis_index which will be substituted

    global_var_idx : [np.ndarray(nvars[ii]) for ii in num_inputs]
        The index of the active variables for each input.
        Note the number of parameters of the final polynomial will likely be
        greater than the number of global variables of the downstream polynomial
        E.g if y2 = y1*x3 and y1 = x1*x2 then y2 is a function of x1,x2,x3
        despite being only parameterized by two variables y1 and x3
    """
    unique_global_vars_in = np.unique(np.concatenate(tuple(global_var_idx)))
    num_global_vars = unique_global_vars_in.shape[0] + (
        basis_index.shape[0]-len(global_var_idx))
    num_inputs = var_idx.shape[0]
    assert num_inputs == len(indices_in)
    assert num_inputs == len(coeffs_in)
    assert basis_coeff.shape[0] == 1
    assert var_idx.max() < num_global_vars
    assert basis_index.shape[1] == 1
    assert len(global_var_idx) == num_inputs

    # store input indices in global_var_idx
    temp = []
    for jj in range(num_inputs):
        assert indices_in[jj].shape[0] == global_var_idx[jj].shape[0]
        ind = np.zeros((num_global_vars, indices_in[jj].shape[1]))
        ind[global_var_idx[jj], :] = indices_in[jj]
        temp.append(ind)
    indices_in = temp

    jj = 0
    degree = basis_index[var_idx[jj], 0]
    ind1, c1 = coeffs_of_power_of_monomial(
        indices_in[jj], coeffs_in[jj], degree)
    # TODO store each power of the input polynomials once before this function
    # and look up when it appears in the term considered here
    for jj in range(1, num_inputs):
        degree = basis_index[var_idx[jj], 0]
        ind2, c2 = coeffs_of_power_of_monomial(
            indices_in[jj], coeffs_in[jj], degree)
        ind1, c1 = multiply_multivariate_polynomials(ind1, c1, ind2, c2)

    # get global indices that were not substituted
    mask = np.ones(num_global_vars, dtype=bool)
    mask[unique_global_vars_in] = False

    # multiply all terms by these remaining variables
    mask2 = np.ones(basis_index.shape[0], dtype=bool)
    mask2[np.unique(var_idx)] = False
    ind1[mask, :] += basis_index[mask2]
    c1 *= basis_coeff
    return ind1, c1


@njit(cache=True)
def coeffs_of_power_of_monomial(indices, coeffs, degree):
    """
    Compute the monomial (coefficients and indices) obtained by raising
    a multivariate polynomial to some power.

    TODO: Deprecate coeffs_of_power_of_nd_linear_monomial as that function
    can be obtained as a special case of this function

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_terms)
        The indices of the multivariate polynomial

    coeffs: np.ndarray (num_terms, nqoi)
        The coefficients of the polynomial

    Returns
    -------
    coeffs: np.ndarray (num_new_terms, nqoi)
        The coefficients of the new polynomial

    indices : np.ndarray (num_vars, num_new_terms)
        The set of multivariate indices that define the new polynomial
    """
    num_vars, num_terms = indices.shape
    assert indices.shape[1] == coeffs.shape[0]
    multinomial_coeffs, multinomial_indices = \
        multinomial_coeffs_of_power_of_nd_linear_monomial(num_terms, degree)
    new_indices = np.zeros((num_vars, multinomial_indices.shape[1]))
    # new_coeffs = np.tile(multinomial_coeffs[:, np.newaxis], coeffs.shape[1])
    # numba does not support tile so replicate with repeat and reshape
    new_coeffs = multinomial_coeffs.repeat(coeffs.shape[1]).reshape(
        (-1, coeffs.shape[1]))
    for ii in range(multinomial_indices.shape[1]):
        multinomial_index = multinomial_indices[:, ii]
        for dd in range(num_terms):
            deg = multinomial_index[dd]
            new_coeffs[ii] *= coeffs[dd]**deg
            new_indices[:, ii] += indices[:, dd]*deg
    return new_indices, new_coeffs


def group_like_terms(coeffs, indices):
    if coeffs.ndim == 1:
        coeffs = coeffs[:, np.newaxis]

    unique_indices, repeated_idx = np.unique(
        indices, axis=1, return_inverse=True)

    nunique_indices = unique_indices.shape[1]
    unique_coeff = np.zeros(
        (nunique_indices, coeffs.shape[1]), dtype=np.double)
    for ii in range(repeated_idx.shape[0]):
        unique_coeff[repeated_idx[ii]] += coeffs[ii]
    return unique_indices, unique_coeff

    # num_vars, num_indices = indices.shape
    # indices_dict = {}
    # for ii in range(num_indices):
    #     key = hash_array(indices[:, ii])
    #     if not key in indices_dict:
    #         indices_dict[key] = [coeffs[ii], ii]
    #     else:
    #         indices_dict[key] = [indices_dict[key][0]+coeffs[ii], ii]

    # new_coeffs = np.empty((len(indices_dict), coeffs.shape[1]))
    # new_indices = np.empty((num_vars, len(indices_dict)), dtype=int)
    # ii = 0
    # for key, item in indices_dict.items():
    #     new_indices[:, ii] = indices[:, item[1]]
    #     new_coeffs[ii] = item[0]
    #     ii += 1
    # return new_coeffs, new_indices


@njit(cache=True)
def multinomial_coefficient(index):
    """Compute the multinomial coefficient of an index [i1,i2,...,id].

    Parameters
    ----------
    index : multidimensional index
        multidimensional index specifying the polynomial degree in each
        dimension

    Returns
    -------
    coeff : double
        the multinomial coefficient
    """
    res, ii = 1, np.sum(index)
    i0 = np.argmax(index)
    for a in np.hstack((index[:i0], index[i0+1:])):
        for jj in range(1, a+1):
            res *= ii
            res //= jj
            ii -= 1
    return res


@njit(cache=True)
def multinomial_coefficients(indices):
    coeffs = np.empty((indices.shape[1]), dtype=np.double)
    for i in range(indices.shape[1]):
        coeffs[i] = multinomial_coefficient(indices[:, i])
    return coeffs


@njit(cache=True)
def multinomial_coeffs_of_power_of_nd_linear_monomial(num_vars, degree):
    """ Compute the multinomial coefficients of the individual terms
    obtained  when taking the power of a linear polynomial
    (without constant term).

    Given a linear multivariate polynomial e.g.
    e.g. (x1+x2+x3)**2 = x1**2+2*x1*x2+2*x1*x3+2*x2**2+x2*x3+x3**2
    return the coefficients of each quadratic term, i.e.
    [1,2,2,1,2,1]

    Parameters
    ----------
    num_vars : integer
        the dimension of the multivariate polynomial
    degree : integer
        the power of the linear polynomial

    Returns
    -------
    coeffs: np.ndarray (num_terms)
        the multinomial coefficients of the polynomial obtained when
        raising the linear multivariate polynomial to the power=degree

    indices: np.ndarray (num_terms)
        the indices of the polynomial obtained when
        raising the linear multivariate polynomial to the power=degree
    """
    indices = compute_hyperbolic_level_indices(num_vars, degree, 1.0)
    coeffs = multinomial_coefficients(indices)
    return coeffs, indices


def add_polynomials(indices_list, coeffs_list):
    """
    Add many polynomials together.

    Example:
        p1 = x1**2+x2+x3, p2 = x2**2+2*x3
        p3 = p1+p2

       return the degrees of each term in the the polynomial

       p3 = x1**2+x2+3*x3+x2**2

       [2, 1, 1, 2]

       and the coefficients of each of these terms

       [1., 1., 3., 1.]


    Parameters
    ----------
    indices_list : list [np.ndarray (num_vars,num_indices_i)]
        List of polynomial indices. indices_i may be different for each
        polynomial

    coeffs_list : list [np.ndarray (num_indices_i,num_qoi)]
        List of polynomial coefficients. indices_i may be different for each
        polynomial. num_qoi must be the same for each list element.


    Returns
    -------
    indices: np.ndarray (num_vars,num_terms)
        the polynomial indices of the polynomial obtained from
        summing the polynomials. This will be the union of the indices
        of the input polynomials

    coeffs: np.ndarray (num_terms,num_qoi)
        the polynomial coefficients of the polynomial obtained from
        summing the polynomials
    """

    num_polynomials = len(indices_list)
    assert num_polynomials == len(coeffs_list)

    all_coeffs = np.vstack(coeffs_list)
    all_indices = np.hstack(indices_list)

    return group_like_terms(all_coeffs, all_indices)

    # unique_indices, repeated_idx = np.unique(
    #     all_indices, axis=1, return_inverse=True)

    # nunique_indices = unique_indices.shape[1]
    # unique_coeff = np.zeros(
    #     (nunique_indices, all_coeffs.shape[1]), dtype=np.double)
    # for ii in range(repeated_idx.shape[0]):
    #     unique_coeff[repeated_idx[ii]] += all_coeffs[ii]
    # return unique_indices, unique_coeff

    # indices_dict = dict()

    # indices = []
    # coeff = []
    # ii = 0
    # kk = 0
    # for jj in range(indices_list[ii].shape[1]):
    #     assert coeffs_list[ii].ndim == 2
    #     assert coeffs_list[ii].shape[0] == indices_list[ii].shape[1]
    #     index = indices_list[ii][:, jj]
    #     indices_dict[hash_array(index)] = kk
    #     indices.append(index)
    #     coeff.append(coeffs_list[ii][jj, :].copy())
    #     kk += 1

    # for ii in range(1, num_polynomials):
    #     # print indices_list[ii].T,num_polynomials
    #     assert coeffs_list[ii].ndim == 2
    #     assert coeffs_list[ii].shape[0] == indices_list[ii].shape[1]
    #     for jj in range(indices_list[ii].shape[1]):
    #         index = indices_list[ii][:, jj]
    #         key = hash_array(index)
    #         if key in indices_dict:
    #             nn = indices_dict[key]
    #             coeff[nn] += coeffs_list[ii][jj, :]
    #         else:
    #             indices_dict[key] = kk
    #             indices.append(index)
    #             coeff.append(coeffs_list[ii][jj, :].copy())
    #             kk += 1

    # indices = np.asarray(indices).T
    # coeff = np.asarray(coeff)

    # return indices, coeff


@njit(cache=True)
def get_indices_double_set(indices):
    """
    Given muultivariate indices

        [i1,i2,...,]

    Compute its double set by
        [i1*i1,i1*i2,...,i2*i2,i2*i3...]

    The double set will only contain unique indices

    Parameters
    ----------
    indices : np.ndarray (num_vars,num_indices)
        The initial indices

    Returns
    -------
    double_set_indices : np.ndarray (num_vars,num_indices)
        The double set of indices
    """
    dummy_coeffs = np.zeros((indices.shape[1], 1))
    double_set_indices = multiply_multivariate_polynomials(
        indices, dummy_coeffs, indices, dummy_coeffs)[0]
    return double_set_indices


def compress_and_sort_polynomial(coef, indices, tol=1e-12):
    II = np.where(np.absolute(coef) > tol)[0]
    indices = indices[:, II]
    coef = coef[II]
    JJ = argsort_indices_leixographically(indices)
    indices = indices[:, JJ]
    coef = coef[JJ, :]
    return indices, coef

# 1D versions of some of these functions can be found at
# https://docs.scipy.org/doc/numpy/reference/routines.polynomials.polynomial.html
