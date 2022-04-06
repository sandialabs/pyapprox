import numpy as np
from functools import partial

from pyapprox.util.utilities import outer_product
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d, gauss_quadrature
)
from pyapprox.surrogates.interp.sparse_grid import (
    get_subspace_values, get_subspace_samples
)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d,
    multivariate_barycentric_lagrange_interpolation,
)
from pyapprox.surrogates.polychaos.gpc import PolynomialChaosExpansion
from pyapprox.surrogates.interp.manipulate_polynomials import add_polynomials


def convert_univariate_lagrange_basis_to_orthonormal_polynomials(
        samples_1d, get_recursion_coefficients):
    """
    Returns
    -------
    coeffs_1d : list [np.ndarray(num_terms_i,num_terms_i)]
        The coefficients of the orthonormal polynomial representation of
        each Lagrange basis. The columns are the coefficients of each
        lagrange basis. The rows are the coefficient of the degree i
        orthonormalbasis
    """
    # Get the maximum number of terms in the orthonormal polynomial that
    # are need to interpolate all the interpolation nodes in samples_1d
    max_num_terms = samples_1d[-1].shape[0]
    num_quad_points = max_num_terms+1
    # Get the recursion coefficients of the orthonormal basis
    recursion_coeffs = get_recursion_coefficients(num_quad_points)
    # compute the points and weights of the correct quadrature rule
    x_quad, w_quad = gauss_quadrature(recursion_coeffs, num_quad_points)
    # evaluate the orthonormal basis at the quadrature points. This can
    # be computed once for all degrees up to the maximum degree
    ortho_basis_matrix = evaluate_orthonormal_polynomial_1d(
        x_quad, max_num_terms, recursion_coeffs)

    # compute coefficients of orthonormal basis using pseudo spectral
    # projection
    coeffs_1d = []
    w_quad = w_quad[:, np.newaxis]
    for ll in range(len(samples_1d)):
        num_terms = samples_1d[ll].shape[0]
        # evaluate the lagrange basis at the quadrature points
        barycentric_weights_1d = [
            compute_barycentric_weights_1d(samples_1d[ll])]
        values = np.eye((num_terms), dtype=float)
        # Sometimes the following function will cause the erro
        # interpolation abscissa are not unique. This can be due to x_quad
        # not abscissa. E.g. x_quad may have points far enough outside
        # range of abscissa, e.g. abscissa are clenshaw curtis points and
        # x_quad points are Gauss-Hermite quadrature points
        lagrange_basis_vals = multivariate_barycentric_lagrange_interpolation(
            x_quad[np.newaxis, :], samples_1d[ll][np.newaxis, :],
            barycentric_weights_1d, values, np.zeros(1, dtype=int))
        # compute fourier like coefficients
        basis_coeffs = []
        for ii in range(num_terms):
            basis_coeffs.append(np.dot(
                w_quad.T,
                lagrange_basis_vals*ortho_basis_matrix[:, ii:ii+1])[0, :])
        coeffs_1d.append(np.asarray(basis_coeffs))
    return coeffs_1d


def convert_multivariate_lagrange_polys_to_orthonormal_polys(
        subspace_index, subspace_values, coeffs_1d, poly_indices,
        config_variables_idx):

    if config_variables_idx is None:
        config_variables_idx = subspace_index.shape[0]

    active_sample_vars = np.where(subspace_index[:config_variables_idx] > 0)[0]
    num_active_sample_vars = active_sample_vars.shape[0]

    if num_active_sample_vars == 0:
        coeffs = subspace_values
        return coeffs

    num_indices = poly_indices.shape[1]
    num_qoi = subspace_values.shape[1]
    coeffs = np.zeros((num_indices, num_qoi), dtype=float)
    for ii in range(num_indices):
        poly_coeffs_1d = \
            [coeffs_1d[dd][subspace_index[dd]][:, poly_indices[dd, ii]]
             for dd in active_sample_vars]
        poly_coeffs = outer_product(poly_coeffs_1d)
        coeffs += subspace_values[ii, :]*poly_coeffs[:, np.newaxis]

    return coeffs


def convert_sparse_grid_to_polynomial_chaos_expansion(sparse_grid, pce_opts,
                                                      debug=False):
    pce = PolynomialChaosExpansion()
    pce.configure(pce_opts)
    if sparse_grid.config_variables_idx is not None:
        assert pce.num_vars() == sparse_grid.config_variables_idx
    else:
        assert pce.num_vars() == sparse_grid.num_vars

    def get_recursion_coefficients(N, dd):
        pce.update_recursion_coefficients([N]*pce.num_vars())
        return pce.recursion_coeffs[pce.basis_type_index_map[dd]].copy()

    coeffs_1d = [
        convert_univariate_lagrange_basis_to_orthonormal_polynomials(
            sparse_grid.samples_1d[dd],
            partial(get_recursion_coefficients, dd=dd))
        for dd in range(pce.num_vars())]

    indices_list = []
    coeffs_list = []
    for ii in range(sparse_grid.subspace_indices.shape[1]):
        if (abs(sparse_grid.smolyak_coefficients[ii]) > np.finfo(float).eps):
            subspace_index = sparse_grid.subspace_indices[:, ii]
            poly_indices = sparse_grid.subspace_poly_indices_list[ii]
            values_indices =\
                sparse_grid.subspace_values_indices_list[ii]
            subspace_values = get_subspace_values(
                sparse_grid.values, values_indices)
            subspace_coeffs = \
                convert_multivariate_lagrange_polys_to_orthonormal_polys(
                    subspace_index, subspace_values, coeffs_1d, poly_indices,
                    sparse_grid.config_variables_idx)

            if debug:
                pce.set_indices(
                    poly_indices[:sparse_grid.config_variables_idx, :])
                pce.set_coefficients(subspace_coeffs)
                subspace_samples = get_subspace_samples(
                    subspace_index,
                    sparse_grid.subspace_poly_indices_list[ii],
                    sparse_grid.samples_1d, sparse_grid.config_variables_idx,
                    unique_samples_only=False)
                poly_values = pce(
                    subspace_samples[:sparse_grid.config_variables_idx, :])
                assert np.allclose(poly_values, subspace_values)

            coeffs_list.append(
                subspace_coeffs*sparse_grid.smolyak_coefficients[ii])
            indices_list.append(
                poly_indices[:sparse_grid.config_variables_idx, :])

    indices, coeffs = add_polynomials(indices_list, coeffs_list)
    pce.set_indices(indices)
    pce.set_coefficients(coeffs)

    return pce
