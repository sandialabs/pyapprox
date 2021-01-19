from pyapprox.variable_transformations import \
    AffineRandomVariableTransformation
from pyapprox.variables import get_distribution_info
from pyapprox.utilities import get_tensor_product_quadrature_rule
from pyapprox.orthonormal_polynomials_1d import get_recursion_coefficients
from pyapprox.orthonormal_polynomials_1d import gauss_quadrature
from functools import partial
import numpy as np
from pyapprox.indexing import \
    compute_hyperbolic_indices
from pyapprox.utilities import cartesian_product, outer_product
from pyapprox.orthonormal_polynomials_1d import \
    jacobi_recurrence, evaluate_orthonormal_polynomial_deriv_1d, \
    hermite_recurrence, krawtchouk_recurrence, hahn_recurrence, \
    discrete_chebyshev_recurrence, evaluate_orthonormal_polynomial_1d
from pyapprox.monomial import monomial_basis_matrix
from pyapprox.utilities import \
    flattened_rectangular_lower_triangular_matrix_index
from pyapprox.probability_measure_sampling import \
    generate_independent_random_samples
from pyapprox.manipulate_polynomials import add_polynomials


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
    #max_level_1d = indices.max(axis=1)
    # so replace with
    max_level_1d = np.empty((num_vars), dtype=np.int)
    for ii in range(num_vars):
        max_level_1d[ii] = indices[ii, :].max()

    if basis_type_index_map is None:
        # numba requires np.int_ not int or np.int
        basis_type_index_map = np.zeros((num_vars), dtype=np.int_)
        recursion_coeffs = [recursion_coeffs]

    for dd in range(num_vars):
        assert (recursion_coeffs[basis_type_index_map[dd]].shape[0] >
                max_level_1d[dd])

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
    values = np.prod(temp2, axis=0).T
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


def precompute_multivariate_orthonormal_polynomial_univariate_values_deprecated(
        samples, indices, recursion_coeffs, deriv_order, basis_type_index_map):
    num_vars = indices.shape[0]
    max_level_1d = indices.max(axis=1)

    if basis_type_index_map is None:
        basis_type_index_map = np.zeros(num_vars, dtype=int)
        recursion_coeffs = [recursion_coeffs]

    for dd in range(num_vars):
        assert (recursion_coeffs[basis_type_index_map[dd]].shape[0] >
                max_level_1d[dd])

    basis_vals_1d = []
    for dd in range(num_vars):
        basis_vals_1d_dd = evaluate_orthonormal_polynomial_deriv_1d(
            samples[dd, :], max_level_1d[dd],
            recursion_coeffs[basis_type_index_map[dd]], deriv_order)
        basis_vals_1d.append(basis_vals_1d_dd)
    return basis_vals_1d


def evaluate_multivariate_orthonormal_polynomial_values_deprecated(
        indices, basis_vals_1d, num_samples):
    num_vars, num_indices = indices.shape
    values = np.empty((num_samples, num_indices))
    for ii in range(num_indices):
        index = indices[:, ii]
        values[:num_samples, ii] = basis_vals_1d[0][:, index[0]]
        for dd in range(1, num_vars):
            values[:num_samples, ii] *= basis_vals_1d[dd][:, index[dd]]
    return values


def evaluate_multivariate_orthonormal_polynomial_derivs_deprecated(indices, max_level_1d, basis_vals_1d, num_samples, deriv_order):

    num_vars, num_indices = indices.shape
    values = np.zeros(((deriv_order*num_vars)*num_samples, num_indices))
    for ii in range(num_indices):
        index = indices[:, ii]
        for jj in range(num_vars):
            # derivative in jj direction
            basis_vals =\
                basis_vals_1d[jj][:, (max_level_1d[jj]+1)+index[jj]].copy()
            # basis values in other directions
            for dd in range(num_vars):
                if dd != jj:
                    basis_vals *= basis_vals_1d[dd][:, index[dd]]

            values[(jj)*num_samples:(jj+1)*num_samples, ii] = basis_vals
    return values


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
    assert samples.shape[0] == num_vars
    assert samples.shape[1] > 0
    #assert recursion_coeffs.shape[0]>indices.max()
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

    # precompute_values = precompute_multivariate_orthonormal_polynomial_univariate_values_deprecated
    # compute_values =  evaluate_multivariate_orthonormal_polynomial_values_deprecated
    # compute_derivs = evaluate_multivariate_orthonormal_polynomial_derivs_deprecated

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
    Notes
    -----
    When pickling this class from a file that file must contain
    from pyapprox.variable_transformations import AffineRandomVariableTransformation
    Otherwise the following error will be thrown

    AttributeError: 'AffineRandomVariableTransformation' object has no attribute 'enforce_bounds'
    """
    def __init__(self):
        self.coefficients = None
        self.indices = None
        self.recursion_coeffs = []
        self.basis_type_index_map = None
        self.basis_type_var_indices = []
        self.numerically_generated_poly_accuracy_tolerance = None

    def __mul__(self, other):
        if self.indices.shape[1] > other.indices.shape[1]:
            poly1 = self
            poly2 = other
        else:
            poly1 = other
            poly2 = self
        import copy
        poly1 = copy.deepcopy(poly1)
        poly2 = copy.deepcopy(poly2)
        max_degrees1 = poly1.indices.max(axis=1)
        max_degrees2 = poly2.indices.max(axis=1)
        # print('###')
        # print(max_degrees1,max_degrees2)
        product_coefs_1d = compute_product_coeffs_1d_for_each_variable(
            poly1, max_degrees1, max_degrees2)
        # print(product_coefs_1d)

        indices, coefs = multiply_multivariate_orthonormal_polynomial_expansions(
            product_coefs_1d, poly1.get_indices(), poly1.get_coefficients(),
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

        import copy
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
        self.config_opts = opts
        self.var_trans = opts.get('var_trans', None)
        if self.var_trans is None:
            raise Exception('must set var_trans')
        self.max_degree = -np.ones(self.num_vars(), dtype=int)
        self.numerically_generated_poly_accuracy_tolerance = opts.get(
            'numerically_generated_poly_accuracy_tolerance', 1e-12)

    def update_recursion_coefficients(self, num_coefs_per_var, opts):
        num_coefs_per_var = np.atleast_1d(num_coefs_per_var)
        initializing = False
        if self.basis_type_index_map is None:
            initializing = True
            self.basis_type_index_map = np.zeros((self.num_vars()), dtype=int)
        if 'poly_types' in opts:
            ii = 0
            for key, poly_opts in opts['poly_types'].items():
                if (initializing or (
                    np.any(num_coefs_per_var[self.basis_type_var_indices[ii]] >
                           self.max_degree[self.basis_type_var_indices[ii]]+1))):
                    if initializing:
                        self.basis_type_var_indices.append(
                            poly_opts['var_nums'])
                    num_coefs = num_coefs_per_var[
                        self.basis_type_var_indices[ii]].max()
                    recursion_coeffs_ii = get_recursion_coefficients(
                        poly_opts, num_coefs,
                        self.numerically_generated_poly_accuracy_tolerance)
                    if recursion_coeffs_ii is None:
                        # recursion coefficients will be None is returned if
                        # monomial basis is specified. Only allow monomials to
                        # be used if all variables use monomial basis
                        assert len(opts['poly_types']) == 1
                    if initializing:
                        self.recursion_coeffs.append(recursion_coeffs_ii)
                    else:
                        self.recursion_coeffs[ii] = recursion_coeffs_ii
                # extract variables indices for which basis is to be used
                self.basis_type_index_map[self.basis_type_var_indices[ii]] = ii
                ii += 1
        else:
            # when only one type of basis is assumed then allow poly_type to
            # be elevated to top level of options dictionary.
            self.recursion_coeffs = [get_recursion_coefficients(
                opts, num_coefs_per_var.max(),
                self.numerically_generated_poly_accuracy_tolerance)]

    def set_indices(self, indices):
        #assert indices.dtype==int
        if indices.ndim == 1:
            indices = indices.reshape((1, indices.shape[0]))

        self.indices = indices
        assert indices.shape[0] == self.num_vars()
        max_degree = indices.max(axis=1)
        if np.any(self.max_degree < max_degree):
            self.update_recursion_coefficients(max_degree+1, self.config_opts)
            self.max_degree = max_degree

    def basis_matrix(self, samples, opts=dict()):
        assert samples.ndim == 2
        assert samples.shape[0] == self.num_vars()
        canonical_samples = self.var_trans.map_to_canonical_space(
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

    def __call__(self, samples):
        return self.value(samples)

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
        nqoi = self.coefficients.shape[1]
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
        pce.update_recursion_coefficients(degrees, pce.config_opts)
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


def get_univariate_quadrature_rules_from_variable(variable, degrees):
    assert len(degrees) == variable.num_vars()
    pce = get_polynomial_from_variable(variable)
    indices = []
    for ii in range(pce.num_vars()):
        indices_ii = np.zeros((pce.num_vars(), degrees[ii]+1), dtype=int)
        indices_ii[ii, :] = np.arange(degrees[ii]+1, dtype=int)
        indices.append(indices_ii)
    pce.set_indices(np.hstack(indices))
    univariate_quad_rules = get_univariate_quadrature_rules_from_pce(
        pce, degrees)
    return univariate_quad_rules, pce


def get_tensor_product_quadrature_rule_from_pce(pce, degrees):
    univariate_quadrature_rules = get_univariate_quadrature_rules_from_pce(
        pce, degrees)
    canonical_samples, weights = \
        get_tensor_product_quadrature_rule(
            degrees+1, num_vars, univariate_quadrature_rules)
    samples = pce.var_trans.map_from_canonical_space(
        canonical_samples)
    return samples, weights


def define_poly_options_from_variable(variable):
    basis_opts = dict()
    for ii in range(len(variable.unique_variables)):
        var = variable.unique_variables[ii]
        name, scales, shapes = get_distribution_info(var)
        opts = {'rv_type': name, 'shapes': shapes,
                'var_nums': variable.unique_variable_indices[ii]}
        basis_opts['basis%d' % ii] = opts
    return basis_opts


def define_poly_options_from_variable_transformation(var_trans):
    pce_opts = {'var_trans': var_trans}
    basis_opts = define_poly_options_from_variable(var_trans.variable)
    pce_opts['poly_types'] = basis_opts
    return pce_opts


def conditional_moments_of_polynomial_chaos_expansion(poly, samples, inactive_idx, return_variance=False):
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


def remove_variables_from_polynomial_chaos_expansion(poly, inactive_idx):
    """
    This function is not optimal. It will recreate the options
    used to configure the polynomial. Any recursion coefficients 
    calculated which are still relevant will need to be computed.
    This is probably not a large overhead though
    """
    fixed_pce = PolynomialChaosExpansion()
    opts = poly.config_opts.copy()
    opts['var_trans'] = AffineRandomVariableTransformation(
        IndependentMultivariateRandomVariable(
            poly.var_trans.variables.all_variables()[inactive_idx]))

    if opts['poly_types'] is not None:
        for key, poly_opts in opts['poly_types'].items():
            var_nums = poly_opts['var_nums']
            poly_opts['var_nums'] = np.array(
                [var_nums[ii] for ii in range(len(var_nums))
                 if var_nums[ii] not in inactive_idx])
    # else # no need to do anything same basis is used for all variables

    fixed_pce.configure(opts)
    if poly.indices is not None:
        active_idx = np.setdiff1d(np.arange(poly.num_vars()), inactive_idx)
        reduced_indices = indices[active_idx, :]
    pce.set_indices(reduced_indices)
    assert pce.coefficients is None
    return fixed_pce


def get_polynomial_from_variable(variable):
    var_trans = AffineRandomVariableTransformation(
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
        poly.update_recursion_coefficients([N]*num_vars, poly.config_opts)
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

    num_vars = poly_indices1.shape[0]
    num_qoi = poly_coefficients1.shape[1]
    # following assumes the max degrees were used to create product_coefs_1d
    max_degrees1 = poly_indices1.max(axis=1)
    max_degrees2 = poly_indices2.max(axis=1)
    basis_coefs, basis_indices = [], []
    for ii in range(num_indices1):
        poly_index_ii = poly_indices1[:, ii]
        active_vars_ii = np.where(poly_index_ii > 0)[0]
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
    
    
    
