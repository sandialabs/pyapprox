import copy
from functools import partial
import numpy as np

from pyapprox.variables.transforms import AffineTransform
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.utilities import (
    cartesian_product, outer_product, hash_array, nchoosek
)
from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.surrogates.orthopoly.recursion_factory import (
    get_recursion_coefficients_from_variable
)
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_deriv_1d,
    evaluate_orthonormal_polynomial_1d, gauss_quadrature,
    define_orthopoly_options_from_marginal
)
from pyapprox.surrogates.orthopoly.quadrature import (
    get_gauss_quadrature_rule_from_marginal
)
from pyapprox.surrogates.interp.monomial import monomial_basis_matrix
from pyapprox.util.linalg import (
    flattened_rectangular_lower_triangular_matrix_index
)
from pyapprox.surrogates.interp.manipulate_polynomials import add_polynomials
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_level_indices
)


def make_2D_array(lis):
    """Funciton to get 2D array from a list of lists

    numba does not support lists of lists so need to compute lists to 2D array
    and store different lengths
    """
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len))

    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr, lengths


def precompute_multivariate_orthonormal_polynomial_univariate_values(
        samples, indices, recursion_coeffs, deriv_order, basis_type_index_map):
    num_vars = indices.shape[0]
    # axis keyword is not supported when usingnumba
    # max_level_1d = indices.max(axis=1)
    # so replace with
    max_level_1d = np.empty((num_vars), dtype=np.int_)
    for ii in range(num_vars):
        max_level_1d[ii] = indices[ii, :].max()

    if basis_type_index_map is None:
        # numba requires np.int_ not int or np.int
        basis_type_index_map = np.zeros((num_vars), dtype=np.int_)
        recursion_coeffs = [recursion_coeffs]

    for dd in range(num_vars):
        if (recursion_coeffs[basis_type_index_map[dd]].shape[0] <=
                max_level_1d[dd]):
            msg = f'max_level_1d for dimension {dd} exceeds number of '
            msg += 'precomputed recusion coefficients'
            raise Exception(msg)

    basis_vals_1d = np.empty(
        (num_vars, (1+deriv_order)*(max_level_1d.max()+1), samples.shape[1]),
        dtype=np.float64)
    # WARNING some entries of basis_vals_1d will remain uninitialized
    # when max_level_1d[dd]=max_level_1d.max() for directions dd
    # storing arrays of equal size allows fast vectorization based manipulation
    # in downstream functions
    for dd in range(num_vars):
        basis_vals_1d[dd, :(deriv_order+1)*(max_level_1d[dd]+1), :] = \
            evaluate_orthonormal_polynomial_deriv_1d(
                samples[dd, :], max_level_1d[dd],
                recursion_coeffs[basis_type_index_map[dd]], deriv_order).T
    return basis_vals_1d


def evaluate_multivariate_orthonormal_polynomial_values(
        indices, basis_vals_1d, num_samples):
    num_vars, num_indices = indices.shape
    temp1 = basis_vals_1d.reshape(
        (num_vars*basis_vals_1d.shape[1], num_samples))
    temp2 = temp1[indices.ravel()+np.repeat(
        np.arange(num_vars)*basis_vals_1d.shape[1], num_indices), :].reshape(
            num_vars, num_indices, num_samples)
    values = np.prod(temp2, axis=0).T
    return values


def evaluate_multivariate_orthonormal_polynomial_derivs(
        indices, max_level_1d, basis_vals_1d, num_samples, deriv_order):
    # TODO Consider combining
    # evaluate_multivariate_orthonormal_polynomial_values and derivs and
    # evaluate_multivariate_orthonormal_polynomial_derivs because they both
    # compute temp2

    assert deriv_order == 1
    num_vars, num_indices = indices.shape

    # extract derivatives
    temp1 = basis_vals_1d.reshape(
        (num_vars*basis_vals_1d.shape[1], num_samples))
    temp2 = temp1[indices.ravel()+np.repeat(
        np.arange(num_vars)*basis_vals_1d.shape[1], num_indices), :].reshape(
            num_vars, num_indices, num_samples)
    # values = np.prod(temp2, axis=0).T
    # derivs are stored immeadiately after values in basis_vals_1d
    # if max_level_1d[dd]!=max_level_1d.max() then there will be some
    # uninitialized values at the end of the array but these are never accessed
    temp3 = temp1[indices.ravel()+np.repeat(
        max_level_1d+1+np.arange(num_vars)*basis_vals_1d.shape[1],
        num_indices), :].reshape(num_vars, num_indices, num_samples)

    derivs = np.empty((num_samples*num_vars, num_indices))
    for jj in range(num_vars):
        derivs[(jj)*num_samples:(jj+1)*num_samples] = (
            np.prod(temp2[:jj], axis=0)*np.prod(
                temp2[jj+1:], axis=0)*temp3[jj]).T

    return derivs


def evaluate_multivariate_orthonormal_polynomial(
        samples, indices, recursion_coeffs, deriv_order=0,
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

    recursion_coeffs : list of np.ndarray (num_indices,2)
        The recursion coefficients for each unique polynomial

    deriv_order : integer in [0,1]
       The maximum order of the derivatives to evaluate.

    basis_type_index_map : list
        The index into recursion coeffs that points to the unique recursion
        coefficients associated with each dimension

    Return
    ------
    values : np.ndarray (1+deriv_order*num_samples,num_indices)
        The values of the polynomials at the samples
    """
    num_vars, num_indices = indices.shape
    assert samples.shape[0] == num_vars
    assert samples.shape[1] > 0
    # assert recursion_coeffs.shape[0]>indices.max()
    max_level_1d = indices.max(axis=1)

    assert deriv_order >= 0 and deriv_order <= 1

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

    precompute_values = \
        precompute_multivariate_orthonormal_polynomial_univariate_values
    compute_values = evaluate_multivariate_orthonormal_polynomial_values
    compute_derivs = evaluate_multivariate_orthonormal_polynomial_derivs

    basis_vals_1d = precompute_values(
        samples, indices, recursion_coeffs, deriv_order, basis_type_index_map)

    num_samples = samples.shape[1]
    values = compute_values(indices, basis_vals_1d, num_samples)

    if deriv_order == 0:
        return values

    derivs = compute_derivs(
        indices, max_level_1d, basis_vals_1d, num_samples, deriv_order)
    values = np.vstack([values, derivs])

    return values


class PolynomialChaosExpansion(object):
    """
    A polynomial chaos expansion for independent random variables.
    """
    def __init__(self):
        """ Constructor. """
        self.coefficients = None
        self.indices = None
        self.recursion_coeffs = []
        self.basis_type_index_map = None
        self.basis_type_var_indices = []

    def __mul__(self, other):
        if self.indices.shape[1] > other.indices.shape[1]:
            poly1 = self
            poly2 = other
        else:
            poly1 = other
            poly2 = self
        poly1 = copy.deepcopy(poly1)
        poly2 = copy.deepcopy(poly2)
        max_degrees1 = poly1.indices.max(axis=1)
        max_degrees2 = poly2.indices.max(axis=1)
        # print('###')
        # print(max_degrees1,max_degrees2)
        product_coefs_1d = compute_product_coeffs_1d_for_each_variable(
            poly1, max_degrees1, max_degrees2)
        # print(product_coefs_1d)

        indices, coefs = \
            multiply_multivariate_orthonormal_polynomial_expansions(
                product_coefs_1d, poly1.get_indices(),
                poly1.get_coefficients(),
                poly2.get_indices(), poly2.get_coefficients())
        # get_polynomial_from_variable(self.var_trans.variable)
        poly = copy.deepcopy(self)
        poly.set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def __add__(self, other):
        indices_list = [self.indices, other.indices]
        coefs_list = [self.coefficients, other.coefficients]
        indices, coefs = add_polynomials(indices_list, coefs_list)
        poly = get_polynomial_from_variable(self.var_trans.variable)
        poly.set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def __sub__(self, other):
        indices_list = [self.indices, other.indices]
        coefs_list = [self.coefficients, -other.coefficients]
        indices, coefs = add_polynomials(indices_list, coefs_list)
        poly = get_polynomial_from_variable(self.var_trans.variable)
        poly.set_indices(indices)
        poly.set_coefficients(coefs)
        return poly

    def __pow__(self, order):
        poly = get_polynomial_from_variable(self.var_trans.variable)
        if order == 0:
            poly.set_indices(np.zeros([self.num_vars(), 1], dtype=int))
            poly.set_coefficients(np.ones([1, self.coefficients.shape[1]]))
            return poly

        poly = copy.deepcopy(self)
        for ii in range(2, order+1):
            poly = poly*self
        return poly

    # def substitute(self, other):
    #     """
    #     Final polynomial will use orthonormal basis of other for any variable
    #     in other. E.g. if other is denoted z = p(x,y) and we are computing
    #     f(z)
    #     then f(z) = f(x,y) where we use basis associated with x and y and not
    #     with z.

    #     I have code to do this but it requires a transformation from an
    #     orthogonal basis into the monomial basis then into another orthogonal
    #     basis and this transformation can be ill-conditioned.
    #     """
    #     raise Exception(Not implemented)

    def configure(self, opts):
        """
        Parameters
        ----------
        var_trans : :class:`pyapprox.variables.transforms.AffineTransform`
            Variable transformation mapping user samples into the canonical
            domain of the polynomial basis

        opts : dictionary
            Options defining the configuration of the polynomial
            chaos expansion basis with the following attributes

        poly_opts : dictionary
            Options to configure each unique univariate polynomial basis
            with attibutes

        var_nums : iterable
            List of variables dimension which use the ith unique basis

        The remaining options are specific to a given basis type. See
             - :func:`pyapprox.surrogates.orthopoly.quadrature.get_recursion_coefficients_from_variable`
        """
        self.var_trans = opts["var_trans"]
        self.config_opts = opts
        self.max_degree = -np.ones(self.num_vars(), dtype=int)

    def update_recursion_coefficients(self, num_coefs_per_var):
        num_coefs_per_var = np.atleast_1d(num_coefs_per_var)
        initializing = False
        if self.basis_type_index_map is None:
            initializing = True
            self.basis_type_index_map = np.zeros((self.num_vars()), dtype=int)
        ii = 0
        for key, poly_opts in self.config_opts['poly_types'].items():
            if (initializing or (
                np.any(num_coefs_per_var[self.basis_type_var_indices[ii]] >
                       self.max_degree[self.basis_type_var_indices[ii]]+1))):
                if initializing:
                    self.basis_type_var_indices.append(poly_opts['var_nums'])
                num_coefs = num_coefs_per_var[
                    self.basis_type_var_indices[ii]].max()
                recursion_coeffs_ii = get_recursion_coefficients_from_variable(
                    poly_opts["var"], num_coefs, poly_opts)
                if recursion_coeffs_ii is None:
                    # recursion coefficients will be None is returned if
                    # monomial basis is specified. Only allow monomials to
                    # be used if all variables use monomial basis
                    assert len(self.config_opts['poly_types']) == 1
                if initializing:
                    self.recursion_coeffs.append(recursion_coeffs_ii)
                else:
                    self.recursion_coeffs[ii] = recursion_coeffs_ii

            # extract variables indices for which basis is to be used
            self.basis_type_index_map[self.basis_type_var_indices[ii]] = ii
            ii += 1
        if (np.unique(np.hstack(self.basis_type_var_indices)).shape[0] !=
                self.num_vars()):
            msg = 'poly_opts does not specify a basis for each input '
            msg += 'variable'
            raise ValueError(msg)

    def set_indices(self, indices):
        # assert indices.dtype==int
        if indices.ndim == 1:
            indices = indices.reshape((1, indices.shape[0]))

        self.indices = indices
        assert indices.shape[0] == self.num_vars()
        max_degree = indices.max(axis=1)
        if np.any(self.max_degree < max_degree):
            self.update_recursion_coefficients(max_degree+1)
            self.max_degree = max_degree

    def basis_matrix(self, samples, opts=dict()):
        assert samples.ndim == 2
        assert samples.shape[0] == self.num_vars()
        canonical_samples = self.var_trans.map_to_canonical(
            samples)
        basis_matrix = self.canonical_basis_matrix(canonical_samples, opts)
        deriv_order = opts.get('deriv_order', 0)
        if deriv_order == 1:
            basis_matrix[samples.shape[1]:, :] =\
                self.var_trans.map_derivatives_from_canonical_space(
                basis_matrix[samples.shape[1]:, :])
        return basis_matrix

    def canonical_basis_matrix(self, canonical_samples, opts=dict()):
        deriv_order = opts.get('deriv_order', 0)
        if self.recursion_coeffs[0] is not None:
            basis_matrix = evaluate_multivariate_orthonormal_polynomial(
                canonical_samples, self.indices, self.recursion_coeffs,
                deriv_order, self.basis_type_index_map)
        else:
            basis_matrix = monomial_basis_matrix(
                self.indices, canonical_samples, deriv_order)
        return basis_matrix

    def jacobian(self, sample):
        assert sample.shape[1] == 1
        derivative_matrix = self.basis_matrix(
            sample, {'deriv_order': 1})[1:]
        jac = derivative_matrix.dot(self.coefficients).T
        return jac

    def set_coefficients(self, coefficients):
        assert coefficients.ndim == 2
        assert coefficients.shape[0] == self.num_terms()

        self.coefficients = coefficients.copy()

    def get_coefficients(self):
        if self.coefficients is not None:
            return self.coefficients.copy()

    def get_indices(self):
        return self.indices.copy()

    def value(self, samples):
        basis_matrix = self.basis_matrix(samples)
        return np.dot(basis_matrix, self.coefficients)

    def num_vars(self):
        return self.var_trans.num_vars()

    def __call__(self, samples, return_grad=False):
        vals = self.value(samples)
        if not return_grad:
            return vals
        jacs = [self.jacobian(sample[:, None]) for sample in samples.T]
        if samples.shape[1] == 1:
            return vals, jacs[0]
        return vals, jacs

    def mean(self):
        """
        Compute the mean of the polynomial chaos expansion

        Returns
        -------
        mean : np.ndarray (nqoi)
            The mean of each quantitity of interest
        """
        return self.coefficients[0, :]

    def variance(self):
        """
        Compute the variance of the polynomial chaos expansion

        Returns
        -------
        var : np.ndarray (nqoi)
            The variance of each quantitity of interest
        """

        var = np.sum(self.coefficients[1:, :]**2, axis=0)
        return var

    def covariance(self):
        """
        Compute the covariance between each quantity of interest of the
        polynomial chaos expansion

        Returns
        -------
        covar : np.ndarray (nqoi)
            The covariance between each quantitity of interest
        """
        # nqoi = self.coefficients.shape[1]
        covar = self.coefficients[1:, :].T.dot(self.coefficients[1:, :])
        return covar

    def num_terms(self):
        # truncated svd creates basis with num_terms <= num_indices
        return self.indices.shape[1]


def get_univariate_quadrature_rules_from_pce(pce, degrees):
    num_vars = pce.num_vars()
    degrees = np.atleast_1d(degrees)
    if degrees.shape[0] == 1 and num_vars > 1:
        degrees = np.array([degrees[0]]*num_vars)
    if np.any(pce.max_degree < degrees):
        pce.update_recursion_coefficients(degrees+1)
    if len(pce.recursion_coeffs) == 1:
        # update_recursion_coefficients may not return coefficients
        # up to degree specified if using recursion for polynomial
        # orthogonal to a discrete variable with finite non-zero
        # probability measures
        assert pce.recursion_coeffs[0].shape[0] >= degrees.max()+1
        univariate_quadrature_rules = [
            partial(gauss_quadrature, pce.recursion_coeffs[0])]*num_vars
    else:
        univariate_quadrature_rules = []
        for dd in range(num_vars):
            # update_recursion_coefficients may not return coefficients
            # up to degree specified if using recursion for polynomial
            # orthogonal to a discrete variable with finite non-zero
            # probability measures
            assert (
                pce.recursion_coeffs[pce.basis_type_index_map[dd]].shape[0] >=
                degrees[dd]+1)
            univariate_quadrature_rules.append(
                partial(gauss_quadrature,
                        pce.recursion_coeffs[pce.basis_type_index_map[dd]]))
    return univariate_quadrature_rules


def get_tensor_product_quadrature_rule_from_pce(pce, degrees):
    univariate_quadrature_rules = get_univariate_quadrature_rules_from_pce(
        pce, degrees)
    canonical_samples, weights = \
        get_tensor_product_quadrature_rule(
            degrees+1, pce.num_vars(), univariate_quadrature_rules)
    samples = pce.var_trans.map_from_canonical(
        canonical_samples)
    return samples, weights


def define_poly_options_from_variable(variable):
    basis_opts = dict()
    for ii in range(len(variable.unique_variables)):
        var = variable.unique_variables[ii]
        opts = define_orthopoly_options_from_marginal(var)
        opts['var_nums'] = variable.unique_variable_indices[ii]
        basis_opts['basis%d' % ii] = opts
    return basis_opts


def define_poly_options_from_variable_transformation(var_trans):
    pce_opts = {'var_trans': var_trans}
    basis_opts = define_poly_options_from_variable(var_trans.variable)
    pce_opts['poly_types'] = basis_opts
    return pce_opts


def conditional_moments_of_polynomial_chaos_expansion(
        poly, samples, inactive_idx, return_variance=False):
    """
    Return mean and variance of polynomial chaos expansion with some variables
    fixed at specified values.

    Parameters
    ----------
    poly: PolynomialChaosExpansion
        The polynomial used to compute moments

    inactive_idx : np.ndarray (ninactive_vars)
        The indices of the fixed variables

    samples : np.ndarray (ninactive_vars)
        The samples of the inacive dimensions fixed when computing moments

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
    assert samples.ndim == 2 and samples.shape[1] == 1
    assert poly.coefficients is not None
    coef = poly.get_coefficients()
    indices = poly.get_indices()

    # precompute 1D basis functions for faster evaluation of
    # multivariate terms
    basis_vals_1d = []
    for dd in range(len(inactive_idx)):
        basis_vals_1d_dd = evaluate_orthonormal_polynomial_1d(
            samples[dd, :], indices[inactive_idx[dd], :].max(),
            poly.recursion_coeffs[poly.basis_type_index_map[inactive_idx[dd]]])
        basis_vals_1d.append(basis_vals_1d_dd)

    active_idx = np.setdiff1d(np.arange(poly.num_vars()), inactive_idx)
    mean = coef[0].copy()
    for ii in range(1, indices.shape[1]):
        index = indices[:, ii]
        coef_ii = coef[ii]  # this intentionally updates the coef matrix
        for dd in range(len(inactive_idx)):
            coef_ii *= basis_vals_1d[dd][0, index[inactive_idx[dd]]]
        if index[active_idx].sum() == 0:
            mean += coef_ii

    if not return_variance:
        return mean

    unique_indices, repeated_idx = np.unique(
        indices[active_idx, :], axis=1, return_inverse=True)
    new_coef = np.zeros((unique_indices.shape[1], coef.shape[1]))
    for ii in range(repeated_idx.shape[0]):
        new_coef[repeated_idx[ii]] += coef[ii]
    variance = np.sum(new_coef**2, axis=0)-mean**2
    return mean, variance


def marginalize_polynomial_chaos_expansion(poly, inactive_idx, center=True):
    """
    This function is not optimal. It will recreate the options
    used to configure the polynomial. Any recursion coefficients
    calculated which are still relevant will need to be computed.
    This is probably not a large overhead though
    """
    marginalized_pce = PolynomialChaosExpansion()
    # poly.config_opts.copy will not work
    opts = copy.deepcopy(poly.config_opts)
    all_variables = poly.var_trans.variable.marginals()
    active_idx = np.setdiff1d(np.arange(poly.num_vars()), inactive_idx)
    active_variables = IndependentMarginalsVariable(
        [all_variables[ii] for ii in active_idx])
    opts['var_trans'] = AffineTransform(active_variables)

    marginalized_var_nums = -np.ones(poly.num_vars())
    marginalized_var_nums[active_idx] = np.arange(active_idx.shape[0])
    keys_to_delete = []
    for key, poly_opts in opts['poly_types'].items():
        var_nums = poly_opts['var_nums']
        poly_opts['var_nums'] = np.array(
            [marginalized_var_nums[v] for v in var_nums
             if v in active_idx], dtype=int)
        if poly_opts['var_nums'].shape[0] == 0:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del opts['poly_types'][key]

    marginalized_pce.configure(opts)
    if poly.indices is not None:
        marginalized_array_indices = []
        for ii, index in enumerate(poly.indices.T):
            if ((index.sum() == 0 and center is False) or
                np.any(index[active_idx]) and
                    (not np.any(index[inactive_idx] > 0))):
                marginalized_array_indices.append(ii)
        marginalized_pce.set_indices(
            poly.indices[
                np.ix_(active_idx, np.array(marginalized_array_indices))])
        if poly.coefficients is not None:
            marginalized_pce.set_coefficients(
                poly.coefficients[marginalized_array_indices, :].copy())
    return marginalized_pce


def get_polynomial_from_variable(variable):
    var_trans = AffineTransform(
        variable)
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    return poly


def compute_univariate_orthonormal_basis_products(get_recursion_coefficients,
                                                  max_degree1, max_degree2):
    """
    Compute all the products of univariate orthonormal bases and re-express
    them as expansions using the orthnormal basis.
    """
    assert max_degree1 >= max_degree2
    max_degree = max_degree1+max_degree2
    num_quad_points = max_degree+1

    recursion_coefs = get_recursion_coefficients(num_quad_points)
    x_quad, w_quad = gauss_quadrature(recursion_coefs, num_quad_points)
    w_quad = w_quad[:, np.newaxis]

    # evaluate the orthonormal basis at the quadrature points. This can
    # be computed once for all degrees up to the maximum degree
    ortho_basis_matrix = evaluate_orthonormal_polynomial_1d(
        x_quad, max_degree, recursion_coefs)

    # compute coefficients of orthonormal basis using pseudo
    # spectral projection
    product_coefs = []
    for d1 in range(max_degree1+1):
        for d2 in range(min(d1+1, max_degree2+1)):
            product_vals = ortho_basis_matrix[:, d1]*ortho_basis_matrix[:, d2]
            coefs = w_quad.T.dot(
                product_vals[:, np.newaxis]*ortho_basis_matrix[:, :d1+d2+1]).T
            product_coefs.append(coefs)
    return product_coefs


def compute_product_coeffs_1d_for_each_variable(
        poly, max_degrees1, max_degrees2):
    # must ensure that poly1 and poly2 have the same basis types
    # in each dimension
    num_vars = poly.num_vars()

    def get_recursion_coefficients(N, dd):
        poly.update_recursion_coefficients([N]*num_vars)
        return poly.recursion_coeffs[poly.basis_type_index_map[dd]].copy()

    # change this to only compute this for unique 1d polys
    product_coefs_1d = [
        compute_univariate_orthonormal_basis_products(
            partial(get_recursion_coefficients, dd=dd),
            max_degrees1[dd], max_degrees2[dd])
        for dd in range(num_vars)]

    return product_coefs_1d


def compute_multivariate_orthonormal_basis_product(
        product_coefs_1d, poly_index_ii, poly_index_jj, max_degrees1,
        max_degrees2, tol=2*np.finfo(float).eps):
    """
    Compute the product of two multivariate orthonormal bases and re-express
    as an expansion using the orthnormal basis.
    """
    num_vars = poly_index_ii.shape[0]
    poly_index = poly_index_ii+poly_index_jj
    active_vars = np.where(poly_index > 0)[0]
    if active_vars.shape[0] > 0:
        coefs_1d = []
        for dd in active_vars:
            pii, pjj = poly_index_ii[dd], poly_index_jj[dd]
            if pii < pjj:
                tmp = pjj
                pjj = pii
                pii = tmp
            kk = flattened_rectangular_lower_triangular_matrix_index(
                pii, pjj, max_degrees1[dd]+1, max_degrees2[dd]+1)
            coefs_1d.append(product_coefs_1d[dd][kk][:, 0])
        indices_1d = [np.arange(poly_index[dd]+1)
                      for dd in active_vars]
        product_coefs = outer_product(coefs_1d)[:, np.newaxis]
        active_product_indices = cartesian_product(indices_1d)
        II = np.where(np.absolute(product_coefs) > tol)[0]
        active_product_indices = active_product_indices[:, II]
        product_coefs = product_coefs[II]
        product_indices = np.zeros(
            (num_vars, active_product_indices.shape[1]), dtype=int)
        product_indices[active_vars] = active_product_indices
    else:
        product_coefs = np.ones((1, 1))
        product_indices = np.zeros([num_vars, 1], dtype=int)

    return product_indices, product_coefs


def multiply_multivariate_orthonormal_polynomial_expansions(
        product_coefs_1d, poly_indices1, poly_coefficients1, poly_indices2,
        poly_coefficients2):
    num_indices1 = poly_indices1.shape[1]
    num_indices2 = poly_indices2.shape[1]
    assert num_indices2 <= num_indices1
    assert poly_coefficients1.shape[0] == num_indices1
    assert poly_coefficients2.shape[0] == num_indices2

    # following assumes the max degrees were used to create product_coefs_1d
    max_degrees1 = poly_indices1.max(axis=1)
    max_degrees2 = poly_indices2.max(axis=1)
    basis_coefs, basis_indices = [], []
    for ii in range(num_indices1):
        poly_index_ii = poly_indices1[:, ii]
        for jj in range(num_indices2):
            poly_index_jj = poly_indices2[:, jj]
            product_indices, product_coefs = \
                compute_multivariate_orthonormal_basis_product(
                    product_coefs_1d, poly_index_ii, poly_index_jj,
                    max_degrees1, max_degrees2)
            # print(ii,jj,product_coefs,poly_index_ii,poly_index_jj)
            # TODO for unique polynomials the product_coefs and indices
            # of [0,1,2] is the same as [2,1,0] so perhaps store
            # sorted active indices and look up to reuse computations
            product_coefs_iijj = product_coefs*poly_coefficients1[ii, :] *\
                poly_coefficients2[jj, :]
            basis_coefs.append(product_coefs_iijj)
            basis_indices.append(product_indices)

            assert basis_coefs[-1].shape[0] == basis_indices[-1].shape[1]

    indices, coefs = add_polynomials(basis_indices, basis_coefs)
    return indices, coefs


def get_univariate_quadrature_rules_from_variable(
        variable, max_nsamples, canonical=False):
    max_nsamples = np.atleast_1d(max_nsamples)
    if max_nsamples.shape[0] == 1:
        max_nsamples = np.ones(variable.num_vars(), dtype=int)*max_nsamples[0]
    if max_nsamples.shape[0] != variable.num_vars():
        raise ValueError(
            "max_nsamples must be an integer or specfied for each marginal")
    univariate_quad_rules = []
    for ii, marginal in enumerate(variable.marginals()):
        quad_rule = get_gauss_quadrature_rule_from_marginal(
            marginal, max_nsamples[ii], canonical=canonical)
        univariate_quad_rules.append(quad_rule)
    return univariate_quad_rules


def get_coefficients_for_plotting(pce, qoi_idx):
    """
   Get the coefficients of a
    :class:`pyapprox.polynomial_chaos.multivariate_polynomials.PolynomialChaosExpansion`
"""
    coeff = pce.get_coefficients()[:, qoi_idx]
    indices = pce.indices.copy()
    assert coeff.shape[0] == indices.shape[1]

    num_vars = pce.num_vars()
    degree = -1
    indices_dict = dict()
    max_degree = indices.sum(axis=0).max()
    for ii in range(indices.shape[1]):
        key = hash_array(indices[:, ii])
        indices_dict[key] = ii
    i = 0
    degree_breaks = []
    coeff_sorted = []
    degree_indices_set = np.empty((num_vars, 0))
    for degree in range(max_degree+1):
        nterms = nchoosek(num_vars+degree, degree)
        if nterms < 1e6:
            degree_indices = compute_hyperbolic_level_indices(
                num_vars, degree, 1.)
        else:
            'Could not plot coefficients of terms with degree >= %d' % degree
            break
        degree_indices_set = np.hstack((degree_indices_set, indices))
        for ii in range(degree_indices.shape[1]-1, -1, -1):
            index = degree_indices[:, ii]
            key = hash_array(index)
            if key in indices_dict:
                coeff_sorted.append(coeff[indices_dict[key]])
            else:
                coeff_sorted.append(0.0)
            i += 1
        degree_breaks.append(i)

    return np.array(coeff_sorted), degree_indices_set, degree_breaks


def plot_unsorted_pce_coefficients(coeffs,
                                   indices,
                                   ax,
                                   degree_breaks=None,
                                   axislabels=None,
                                   legendlabels=None,
                                   title=None,
                                   cutoffs=None,
                                   ylim=[1e-6, 1.]):
    """
    Plot the coefficients of linear (in parameters) approximation
    """

    np.set_printoptions(precision=16)

    colors = ['k', 'r', 'b', 'g', 'm', 'y', 'c']
    markers = ['s', 'o', 'd', ]
    fill_style = ['none', 'full']
    # fill_style = ['full','full']
    ax.set_xlim([1, coeffs[0].shape[0]])
    for i in range(len(coeffs)):
        coeffs_i = np.absolute(coeffs[i])
        assert coeffs_i.ndim == 1
        mfc = colors[i]
        fs = fill_style[min(i, 1)]
        if (fs == 'none'):
            mfc = 'none'
        ms = 10.-2.*i
        ax.plot(list(range(1, coeffs_i.shape[0]+1)), coeffs_i,
                marker=markers[i % 3],
                fillstyle=fs,
                markeredgecolor=colors[i],
                linestyle='None', markerfacecolor=mfc,
                markersize=ms)
        if degree_breaks is not None:
            for i in np.array(degree_breaks) + 1:
                ax.axvline(i, linestyle='--', color='k')

    if cutoffs is not None:
        i = 1
        for cutoff in cutoffs:
            plt.axhline(cutoff, linestyle='-', color=colors[i])
            i += 1

    ax.set_xscale('log')
    ax.set_yscale('log')

    if (axislabels is not None):
        ax.set_xlabel(axislabels[0], fontsize=20)
        ax.set_ylabel(axislabels[1], rotation='horizontal', fontsize=20)

    if (title is not None):
        ax.set_title(title)

    if (legendlabels is not None):
        msg = "Must provide a legend label for each filename"
        assert (len(legendlabels) >= len(coeffs)), msg
        ax.legend(legendlabels, numpoints=1)

    ax.set_ylim(ylim)


def plot_pce_coefficients(ax, pces, ylim=[1e-6, 1], qoi_idx=0):
    """
    Plot the coefficients of multiple
    :class:`pyapprox.polynomial_chaos.multivariate_polynomials.PolynomialChaosExpansion`
    """
    coeffs = []
    breaks = []
    indices_list = []
    max_num_indices = 0
    for pce in pces:
        # only plot coeff that will fit inside axis limits
        coeff, indices, degree_breaks = get_coefficients_for_plotting(
            pce, qoi_idx)
        coeffs.append(coeff)
        indices_list.append(indices)
        breaks.append(degree_breaks)
        max_num_indices = max(max_num_indices, indices.shape[1])

    for ii in range(len(indices_list)):
        nn = indices.shape[1]
        if (nn < max_num_indices):
            indices_list[ii] += [None]*(max_num_indices-nn)
            coeffs[ii] = np.resize(coeffs[ii], max_num_indices)
            coeffs[ii][nn:] = 0.

    plot_unsorted_pce_coefficients(
        coeffs, indices_list, ax, degree_breaks=breaks[0], ylim=ylim)


def _marginalize_function_1d(fun, variable, quad_degrees, ii, samples_ii,
                             qoi=0):
    assert samples_ii.ndim == 1
    all_variables = variable.marginals()
    sub_variable = IndependentMarginalsVariable(
        [all_variables[jj]
         for jj in range(variable.num_vars()) if ii != jj])
    pce = get_polynomial_from_variable(sub_variable)
    xquad, wquad = get_tensor_product_quadrature_rule_from_pce(
        pce, quad_degrees)
    samples = np.empty((xquad.shape[0]+1, xquad.shape[1]))
    indices = np.delete(np.arange(variable.num_vars()), ii)
    samples[indices, :] = xquad
    values = np.empty_like(samples_ii)
    for jj in range(samples_ii.shape[0]):
        samples[ii, :] = samples_ii[jj]
        # print(jj, samples.shape, samples_ii.shape)
        values_jj = fun(samples)[:, qoi]
        values[jj] = values_jj.dot(wquad)
    return values


def _marginalize_function_nd(fun, variable, quad_degrees, sub_indices,
                             sub_samples, qoi=0):
    assert sub_samples.ndim == 2
    assert sub_indices.shape[0] == sub_samples.shape[0]
    assert quad_degrees.shape[0] == variable.num_vars() - sub_indices.shape[0]

    if sub_indices.shape[0] == variable.num_vars():
        return fun(sub_samples)[:, qoi]

    all_variables = variable.marginals()
    sub_variable = IndependentMarginalsVariable(
        [all_variables[kk]
         for kk in range(variable.num_vars()) if kk not in sub_indices])
    pce = get_polynomial_from_variable(sub_variable)
    xquad, wquad = get_tensor_product_quadrature_rule_from_pce(
        pce, quad_degrees)
    samples = np.empty((variable.num_vars(), xquad.shape[1]))
    indices = np.delete(np.arange(variable.num_vars()), sub_indices)
    samples[indices, :] = xquad
    nsamples = sub_samples.shape[1]
    values = np.empty((nsamples, 1))
    for kk in range(nsamples):
        samples[sub_indices, :] = sub_samples[:, kk:kk+1]
        values_jj = fun(samples)[:, qoi]
        values[kk, 0] = values_jj.dot(wquad)
    return values
