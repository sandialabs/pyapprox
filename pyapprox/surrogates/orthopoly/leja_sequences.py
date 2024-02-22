# The functions begining with __ are useful when reusing information during
# optimiaztion. For example computing the leja objective Hessian requires
# data also used when computing the value and Jacobian

import numpy as np
import os
from scipy.optimize import Bounds
from scipy.linalg import solve_triangular
from functools import partial
from warnings import warn

from pyapprox.optimization.pya_minimize import pyapprox_minimize
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d
)
from pyapprox.util.linalg import truncated_pivoted_lu_factorization


def christoffel_function(samples, basis_matrix_generator, normalize=False):
    r"""
    Evaluate the christoffel function K(x) at a set of samples x.

    Useful for preconditioning linear systems generated using
    orthonormal polynomials

    Parameters
    ----------
    normalize : boolean
        True - divide function by :math:`\sqrt(N)`
        False - Christoffel function will return Gauss quadrature weights
                if x are Gauss quadrature points
    """
    basis_matrix = basis_matrix_generator(samples)
    vals = 1./christoffel_weights(basis_matrix)
    if normalize:
        vals /= basis_matrix.shape[1]
    return vals


def christoffel_weights(basis_matrix):
    r"""
    Evaluate the 1/K(x),from a basis matrix, where K(x) is the
    Christoffel function.
    """
    return 1./np.sum(basis_matrix**2, axis=1)


def christoffel_preconditioner(basis_matrix, samples):
    return christoffel_weights(basis_matrix)


def sqrt_christoffel_function_inv_1d(basis_fun, samples, normalize=False):
    r"""
    Evaluate the inverse of the square-root of the Christoffel function
    at a set of samples. That is compute

    .. math:: \frac{1}{K(x)^{1/2}}

    where

    .. math::

       K(x) = \sum_{n=1}^N \phi_i^2(x)

    for a set of orthonormal basis function :math:`\phi_i, i=1, \ldots, N`

    Thsi function is useful for preconditioning linear systems generated using
    orthonormal polynomials

    Parameters
    ----------
    basis_fun : callable
        Evaluate the basis at a set of points.
        Function with signature

        `basis_fun(samples) -> np.ndarray(nsamples, nterms)`

    samples : np.ndarray (nvars, nsamples)

    normalize : boolean
        True - return sqrt{N/K(x)} where N is the number of basis terms
        False - return sqrt{1/K(x)}
    """
    return __sqrt_christoffel_function_inv_1d(
        basis_fun(samples[0, :]), normalize)


def get_lu_leja_samples(generate_basis_matrix, generate_candidate_samples,
                        num_candidate_samples, num_leja_samples,
                        preconditioning_function=None, initial_samples=None):
    r"""
    Generate Leja samples using LU factorization.

    Parameters
    ----------
    generate_basis_matrix : callable
        basis_matrix = generate_basis_matrix(candidate_samples)
        Function to evaluate a basis at a set of samples

    generate_candidate_samples : callable
        candidate_samples = generate_candidate_samples(num_candidate_samples)
        Function to generate candidate samples. This can siginficantly effect
        the Leja samples generated

    num_candidate_samples : integer
        The number of candidate_samples

    preconditioning_function : callable
        basis_matrix = preconditioning_function(basis_matrix)
        precondition a basis matrix to improve stability
        samples are the samples used to build the basis matrix. They must
        be in the same order as they were used to create the rows of the basis
        matrix.

    TODO unfortunately some preconditioing_functions need only basis matrix
    or samples, but cant think of a better way to generically pass in function
    here other than to require functions that use both arguments

    num_leja_samples : integer
        The number of desired leja samples. Must be <= num_indices

    initial_samples : np.ndarray (num_vars,num_initial_samples)
       Enforce that the initial samples are chosen (in the order specified)
       before any other candidate sampels are chosen. This can lead to
       ill conditioning and leja sequence terminating early

    Returns
    -------
    laja_samples : np.ndarray (num_vars, num_indices)
        The samples of the Leja sequence

    data_structures : tuple
        (Q, R, p) the QR factors and pivots. This can be useful for
        quickly building an interpolant from the samples
    """
    candidate_samples = generate_candidate_samples(num_candidate_samples)
    if initial_samples is not None:
        assert candidate_samples.shape[0] == initial_samples.shape[0]
        candidate_samples = np.hstack((initial_samples, candidate_samples))
        num_initial_rows = initial_samples.shape[1]
    else:
        num_initial_rows = 0

    basis_matrix = generate_basis_matrix(candidate_samples)
    assert num_leja_samples <= basis_matrix.shape[1]
    if preconditioning_function is not None:
        weights = np.sqrt(
            preconditioning_function(basis_matrix, candidate_samples))
        basis_matrix = (basis_matrix.T*weights).T
    else:
        weights = None
    L, U, p = truncated_pivoted_lu_factorization(
        basis_matrix, num_leja_samples, num_initial_rows)
    assert p.shape[0] == num_leja_samples, (p.shape, num_leja_samples)
    p = p[:num_leja_samples]
    leja_samples = candidate_samples[:, p]
    plot = False
    if plot and leja_samples.shape[0] == 1:
        import matplotlib.pyplot as plt
        plt.plot(candidate_samples[0, :], weights)
        plt.show()

    if plot and leja_samples.shape[0] == 2:
        import matplotlib.pyplot as plt
        print(('N:', basis_matrix.shape[1]))
        plt.plot(leja_samples[0, 0], leja_samples[1, 0], '*')
        plt.plot(leja_samples[0, :], leja_samples[1, :], 'ro', zorder=10)
        plt.scatter(candidate_samples[0, :], candidate_samples[1, :],
                    s=weights*100, color='b')
        # plt.xlim(-1,1)
        # plt.ylim(-1,1)
        # plt.title('Leja sequence and candidates')
        plt.show()

    # Ignore basis functions (columns) that were not considered during the
    # incomplete LU factorization
    L = L[:, :num_leja_samples]
    U = U[:num_leja_samples, :num_leja_samples]
    data_structures = [L, U, p, weights[p]]
    return leja_samples, data_structures


def interpolate_lu_leja_samples(leja_samples, values, data_structures):
    r"""
    Assumes ordering of values and rows of L and U are consistent.
    Typically this is done by computing leja samples then evaluating function
    at these samples.
    """
    L, U = data_structures[0], data_structures[1]
    weights = data_structures[3]
    temp = solve_triangular(L, (values.T*weights).T, lower=True)
    coef = solve_triangular(U, temp, lower=False)
    return coef


def get_quadrature_weights_from_lu_leja_samples(leja_samples, data_structures):
    L, U = data_structures[0], data_structures[1]
    precond_weights = data_structures[3]
    basis_matrix_inv = np.linalg.inv(np.dot(L, U))
    quad_weights = basis_matrix_inv[0, :]
    if precond_weights is not None:
        # Since we preconditioned, we need to "un-precondition" to get
        # the right weights. Sqrt of weights has already been applied
        # so do not do it again here
        quad_weights *= precond_weights
    return quad_weights


def __sqrt_christoffel_function_inv_1d(basis_mat, normalize):
    # avoid overflow.
    # if np.max(basis_mat) > 1e8:
    #     # note this will return zero for all rows of basis mat
    #     # when used for optimization this is fine because basis_mat will
    #     # only have one row. But if evaluating at multiple points this
    #     # will cause all values to zero if overflow occurs
    #     return np.zeros(basis_mat.shape[0])
    vals = 1./np.linalg.norm(basis_mat, axis=1)
    if normalize is True:
        vals *= np.sqrt(basis_mat.shape[1])
    return vals


def christoffel_function_inv_1d(basis_fun, samples, normalize=False):
    r"""
    Evaluate the inverse of the Christoffel function
    at a set of samples. That is compute

    .. math:: \frac{1}{K(x)}

    where

    .. math::

       K(x) = \sum_{n=1}^N \phi_i^2(x)

    for a set of orthonormal basis function :math:`\phi_i, i=1, \ldots, N`

    This function is useful for preconditioning linear systems generated using
    orthonormal polynomials

    Parameters
    ----------
    basis_fun : callable
        Evaluate the basis at a set of points.
        Function with signature

        `basis_fun(samples) -> np.ndarray(nsamples, nterms)`

    samples : np.ndarray (nvars, nsamples)

    normalize : boolean
        True - return N/K(x) where N is the number of basis terms
        False - return 1/K(x)
    """
    return __christoffel_function_inv_1d(
        basis_fun(samples[0, :]), normalize)


def __christoffel_function_inv_1d(basis_mat, normalize):
    # avoid overflow issues when computing squared l2 norm
    # which occur because optimization marching towards infinity
    # if np.max(basis_mat) > 1e8:
    #     # note this will return zero for all rows of basis mat
    #     # when used for optimization this is fine because basis_mat will
    #     # only have one row. But if evaluating at multiple points this
    #     # will cause all values to zero if overflow occurs
    #     return np.zeros(basis_mat.shape[0])
    denom_sqrt = np.linalg.norm(basis_mat, axis=1)
    # if np.any(denom_sqrt > 1e8):
    #     return np.zeros(basis_mat.shape[0])
    vals = 1/denom_sqrt**2
    if normalize is True:
        vals *= basis_mat.shape[1]
    return vals


def sqrt_christoffel_function_inv_jac_1d(basis_fun_and_jac, samples,
                                         normalize=False):
    r"""
    Return the first_derivative wrt x of the sqrt of the inverse of the
    Christoffel function at a set of samples. That is compute

    .. math::

       \frac{\partial}{\partial x}\frac{1}{K(x)^{1/2}} =
       -\frac{K^\prime(x)}{2K(x)^{3/2}}

    with

    .. math::

       K^\prime(x) = \frac{\partial K(x)}{\partial x} =
       2\sum_{n=1}^N \frac{\partial \phi_i(x)}{\partial x}\phi_i(x)

    Parameters
    ----------
    basis_fun : callable
        Evaluate the basis and its derivatives at a set of points.
        Function with signature

        `basis_fun_and_jac(samples) -> np.ndarray(nsamples, 2*nterms)`

        The first nterms columns are the values the second the derivatives.
        See :func:`pyapprox.evaluate_orthonormal_polynomial_deriv_1d`

    samples : np.ndarray (nvars, nsamples)
        The samples at which to evaluate the basis

    normalize : boolean
        True - return sqrt{nterms/K(x)} where nterms is the number of basis
               terms
        False - return \sqrt{1/K(x)}
    """
    basis_vals_and_derivs = basis_fun_and_jac(samples[0, :])
    assert basis_vals_and_derivs.shape[1] % 2 == 0
    nterms = basis_vals_and_derivs.shape[1]//2
    basis_mat = basis_vals_and_derivs[:, :nterms]
    basis_jac = basis_vals_and_derivs[:, nterms:]
    return __sqrt_christoffel_function_inv_jac_1d(
        basis_mat, basis_jac, normalize)


def __sqrt_christoffel_function_inv_jac_1d(basis_mat, basis_jac, normalize):
    vals = -2*(basis_mat*basis_jac).sum(axis=1)
    vals /= (2*np.sum(basis_mat**2, axis=1)**(1.5))
    if normalize is True:
        vals *= np.sqrt(basis_mat.shape[1])
    return vals


def christoffel_function_inv_jac_1d(basis_fun_and_jac, samples,
                                    normalize=False):
    r"""
    Return the first_derivative wrt x of the inverse of the
    Christoffel function at a set of samples. That is compute

    .. math::

       \frac{\partial}{\partial x}\frac{1}{K(x)} =
       -\frac{K^\prime(x)}{K(x)^2}

    with

    .. math::

       K^\prime(x) = \frac{\partial K(x)}{\partial x} =
       2\sum_{n=1}^N \frac{\partial \phi_i(x)}{\partial x}\phi_i(x)

    Parameters
    ----------
    basis_fun : callable
        Evaluate the basis and its derivatives at a set of points.
        Function with signature

        `basis_fun_and_jac(samples) -> np.ndarray(nsamples, 2*nterms)`

        The first nterms columns are the values the second the derivatives.
        See :func:`pyapprox.evaluate_orthonormal_polynomial_deriv_1d`

    samples : np.ndarray (nvars, nsamples)
        The samples at which to evaluate the basis

    normalize : boolean
        True - return nterms/K(x) where nterms is the number of basis
               terms
        False - return 1/K(x)
    """
    basis_vals_and_derivs = basis_fun_and_jac(samples[0, :])
    assert basis_vals_and_derivs.shape[1] % 2 == 0
    nterms = basis_vals_and_derivs.shape[1]//2
    basis_mat = basis_vals_and_derivs[:, :nterms]
    basis_jac = basis_vals_and_derivs[:, nterms:]
    return __christoffel_function_inv_jac_1d(
        basis_mat, basis_jac, normalize)


def __christoffel_function_inv_jac_1d(basis_mat, basis_jac, normalize):
    # avoid overflow issues when computing squared l2 norm
    # which occur because optimization marching towards infinity
    denom_sqrt = np.sum(basis_mat**2, axis=1)
    # if np.any(denom_sqrt > 1e8):
    #     # note this will return zero for all rows of basis mat
    #     # when used for optimization this is fine because basis_mat will
    #     # only have one row. But if evaluating at multiple points this
    #     # will cause all values to zero if overflow occurs
    #     return np.zeros(basis_mat.shape[0])
    vals = -2*(basis_mat*basis_jac).sum(axis=1)
    vals /= denom_sqrt**2
    if normalize is True:
        vals *= basis_mat.shape[1]
    return vals


def sqrt_christoffel_function_inv_hess_1d(basis_fun_jac_hess, samples,
                                          normalize=False):
    r"""
    Return the second derivative wrt x of the inverse of the square-root of the
    Christoffel function at a set of samples. That is compute

    .. math::

       \frac{\partial^2}{\partial x^2}\frac{1}{K(x)^{1/2}} =
       \frac{K^\prime(x)^2-2K(x)K^{\prime\prime}(x)}{4K(x)^{5/2}}

    with

    .. math::

       K^{\prime\prime}(x) = \frac{\partial^2 K(x)}{\partial x^2} =
       2\sum_{n=1}^N \frac{\partial^2 \phi_i(x)}{\partial x^2}\phi_i(x)+
       (\frac{\partial \phi_i(x)}{\partial x})^2

    Parameters
    ----------
    basis_fun : callable
        Evaluate the basis and its derivatives at a set of points.
        Function with signature

        `basis_fun_and_jac(samples) -> np.ndarray(nsamples, 2*nterms)`

        The first nterms columns are the values the second the derivatives.
        See :func:`pyapprox.evaluate_orthonormal_polynomial_deriv_1d`

    samples : np.ndarray (nvars, nsamples)
        The samples at which to evaluate the basis

        normalize : boolean
        True - return sqrt{nterms/K(x)} where nterms is the number of basis
               terms
        False - return \sqrt{1/K(x)}
    """
    tmp = basis_fun_jac_hess(samples[0, :])
    assert tmp.shape[1] % 3 == 0
    nterms = tmp.shape[1]//3
    basis_mat = tmp[:, :nterms]
    basis_jac = tmp[:, nterms:2*nterms]
    basis_hess = tmp[:, 2*nterms:]
    return __sqrt_christoffel_function_inv_hess_1d(
        basis_mat, basis_jac, basis_hess, normalize)


def __sqrt_christoffel_function_inv_hess_1d(basis_mat, basis_jac, basis_hess,
                                            normalize):
    k = (basis_mat**2).sum(axis=1)
    kdx1 = 2*(basis_mat*basis_jac).sum(axis=1)
    kdx2 = 2*(basis_mat*basis_hess+basis_jac**2).sum(axis=1)
    vals = (3*kdx1**2 - 2*k*kdx2)/(4*k**(2.5))
    if normalize is True:
        vals *= np.sqrt(basis_mat.shape[1])
    return vals


def christoffel_function_inv_hess_1d(basis_fun_jac_hess, samples,
                                     normalize=False):
    r"""
    Return the second derivative wrt x of the inverse of the
    Christoffel function at a set of samples. That is compute

    .. math::

       \frac{\partial^2}{\partial x^2}\frac{1}{K(x)} =
       \frac{2K^\prime(x)^2-K(x)K^{\prime\prime}(x)}{K(x)^{3}}

    with

    .. math::

       K^{\prime\prime}(x) = \frac{\partial^2 K(x)}{\partial x^2} =
       2\sum_{n=1}^N \frac{\partial^2 \phi_i(x)}{\partial x^2}\phi_i(x)+
       (\frac{\partial \phi_i(x)}{\partial x})^2

    Parameters
    ----------
    basis_fun : callable
        Evaluate the basis and its derivatives at a set of points.
        Function with signature

        `basis_fun_and_jac(samples) -> np.ndarray(nsamples, 2*nterms)`

        The first nterms columns are the values the second the derivatives.
        See :func:`pyapprox.evaluate_orthonormal_polynomial_deriv_1d`

    samples : np.ndarray (nvars, nsamples)
        The samples at which to evaluate the basis

        normalize : boolean
        True - return sqrt{nterms/K(x)} where nterms is the number of basis
               terms
        False - return \sqrt{1/K(x)}
    """
    tmp = basis_fun_jac_hess(samples[0, :])
    assert tmp.shape[1] % 3 == 0
    nterms = tmp.shape[1]//3
    basis_mat = tmp[:, :nterms]
    basis_jac = tmp[:, nterms:2*nterms]
    basis_hess = tmp[:, 2*nterms:]
    return __christoffel_function_inv_hess_1d(
        basis_mat, basis_jac, basis_hess, normalize)


def __christoffel_function_inv_hess_1d(basis_mat, basis_jac, basis_hess,
                                       normalize):
    k = (basis_mat**2).sum(axis=1)
    kdx1 = 2*(basis_mat*basis_jac).sum(axis=1)
    kdx2 = 2*(basis_mat*basis_hess+basis_jac**2).sum(axis=1)
    vals = (2*kdx1**2 - k*kdx2)/(k**3)
    if normalize is True:
        vals *= basis_mat.shape[1]
    return vals


def compute_coefficients_of_christoffel_leja_interpolant_1d(
        basis_mat, new_basis):
    # Todo replace with update of LU factorization
    assert new_basis.ndim == 2 and new_basis.shape[1] == 1
    w = __sqrt_christoffel_function_inv_1d(basis_mat, False)
    coef = np.linalg.lstsq(
        w[:, None]*basis_mat, w[:, None]*new_basis, rcond=None)[0]
    return coef


def compute_coefficients_of_pdf_weighted_leja_interpolant_1d(
        w, basis_mat, new_basis):
    # Todo replace with update of LU factorization
    assert new_basis.ndim == 2 and new_basis.shape[1] == 1
    assert w.ndim == 1
    coef = np.linalg.lstsq(
        w[:, None]*basis_mat, w[:, None]*new_basis, rcond=None)[0]
    return coef


def christoffel_leja_objective_fun_1d(basis_fun, coef, samples):
    """
    Parameters
    ----------
    samples : np.ndarray (1, nsamples)
        Samples at which values of the objective are needed.

    basis_fun : callable
        Return the values of all basis functions up to degree k+1 where
        k is the degree of the polynomial interpolating the current Leja
        sequence.
        Function with signature

        `basis_fun(samples) - > np.ndarray (nsamples, nterms)`

         In 1D nterms = k+2

    coef : np.ndarray (nterms, 1)
        Coefficients of polynomial which interpolates the new_basis at the
        samples already in the leja sequence
    """
    assert samples.ndim == 2 and samples.shape[0] == 1
    basis_vals = basis_fun(samples[0, :])
    basis_mat, new_basis = basis_vals[:, :-1], basis_vals[:, -1:]
    weights = __christoffel_function_inv_1d(
        np.hstack([basis_mat, new_basis]), False)
    return __leja_objective_fun_1d(
        weights, basis_mat, new_basis, coef)


def __leja_objective_fun_1d(weights, basis_mat, new_basis, coef):
    """
    Parameters
    ----------
    weights : np.ndarray (nsamples, 1)
        Values of the weight function at new samples x not already in the Leja
        sequence

    basis_mat : np.array (nsamples, nterms)
        Values of the basis of degree k at new samplse x not already in the
        Leja sequence. Note nterms=k+1

    new_basis : np.ndarray (nsamples, 1)
        Basis of with k+1 evaluated at x

    coef : np.ndarray (nterms, 1)
        Coefficients of polynomial which interpolates the new_basis at the
        samples already in the leja sequence
    """
    assert basis_mat.ndim == 2
    assert new_basis.ndim == 2 and new_basis.shape[1] == 1
    assert coef.ndim == 2 and coef.shape[1] == 1
    pvals = basis_mat.dot(coef)
    residual = (new_basis - pvals)
    # if np.absolute(residual).max() > 1e8:
    #     return np.inf*np.ones(residual.shape[0])
    return weights*np.sum(residual**2, axis=1)


def pdf_weighted_leja_objective_fun_1d(pdf, basis_fun, coef, samples):
    """
    Parameters
    ----------
    pdf : callable
        Weight function with signature

        `pdf(x) -> np.ndarray (nsamples)`

    where x is a 1D np.ndarray (nsamples)

    samples : np.ndarray (1, nsamples)
        Samples at which values of the objective are needed.

    basis_fun : callable
        Return the values of all basis functions up to degree k+1 where
        k is the degree of the polynomial interpolating the current Leja
        sequence.
        Function with signature

        `basis_fun(samples) - > np.ndarray (nsamples, nterms)`

         In 1D nterms = k+2

    coef : np.ndarray (nterms, 1)
        Coefficients of polynomial which interpolates the new_basis at the
        samples already in the leja sequence
    """
    assert samples.ndim == 2 and samples.shape[0] == 1
    basis_vals = basis_fun(samples[0, :])
    basis_mat, new_basis = basis_vals[:, :-1], basis_vals[:, -1:]
    weights = pdf(samples[0, :])
    return __leja_objective_fun_1d(weights, basis_mat, new_basis, coef)


def christoffel_leja_objective_jac_1d(basis_fun_jac, coef, samples):
    assert samples.ndim == 2 and samples.shape[0] == 1
    tmp = basis_fun_jac(samples[0, :])
    assert tmp.shape[1] % 2 == 0
    nterms = tmp.shape[1]//2
    basis_mat = tmp[:, :nterms]
    basis_jac = tmp[:, nterms:]
    w = __christoffel_function_inv_1d(basis_mat, False)
    wdx1 = __christoffel_function_inv_jac_1d(basis_mat, basis_jac, False)
    return __leja_objective_jac_1d(w, wdx1, basis_mat, basis_jac, coef)


def __leja_objective_jac_1d(w, wdx1, basis_mat, basis_jac, coef):
    assert basis_mat.ndim == 2
    assert basis_mat.shape == basis_jac.shape
    assert coef.ndim == 2 and coef.shape[1] == 1
    bvals = basis_mat[:, -1:]
    pvals = basis_mat[:, :-1].dot(coef)
    bderivs = basis_jac[:, -1:]
    pderivs = basis_jac[:, :-1].dot(coef)
    residual = (bvals - pvals)
    residual_jac = bderivs - pderivs
    if residual.max() > np.sqrt(np.finfo(float).max/100):
        return np.inf*np.ones(residual.shape[0])
    jac = (residual**2*wdx1 + 2*w*residual*residual_jac).sum(axis=1)
    return jac


def pdf_weighted_leja_objective_jac_1d(pdf, pdf_jac, basis_fun_jac, coef,
                                       samples):
    assert samples.ndim == 2 and samples.shape[0] == 1
    tmp = basis_fun_jac(samples[0, :])
    assert tmp.shape[1] % 2 == 0
    nterms = tmp.shape[1]//2
    basis_mat = tmp[:, :nterms]
    basis_jac = tmp[:, nterms:]
    w = pdf(samples[0, :])
    wdx1 = pdf_jac(samples[0, :])
    return __leja_objective_jac_1d(w, wdx1, basis_mat, basis_jac, coef)


def christoffel_leja_objective_hess_1d(basis_fun_jac_hess, coef, samples):
    assert samples.ndim == 2 and samples.shape[0] == 1
    tmp = basis_fun_jac_hess(samples[0, :])
    assert tmp.shape[1] % 3 == 0
    nterms = tmp.shape[1]//3
    basis_mat = tmp[:, :nterms]
    basis_jac = tmp[:, nterms:2*nterms]
    basis_hess = tmp[:, 2*nterms:3*nterms]
    w = __christoffel_function_inv_1d(basis_mat, False)
    wdx1 = __christoffel_function_inv_jac_1d(basis_mat, basis_jac, False)
    wdx2 = __christoffel_function_inv_hess_1d(
        basis_mat, basis_jac, basis_hess, False)
    return __leja_objective_hess_1d(
        w, wdx1, wdx2, basis_mat, basis_jac, basis_hess, coef)


def __leja_objective_hess_1d(w, wdx1, wdx2, basis_mat, basis_jac, basis_hess,
                             coef):
    assert basis_mat.ndim == 2
    assert basis_mat.shape == basis_jac.shape
    assert coef.ndim == 2 and coef.shape[1] == 1
    bvals = basis_mat[:, -1:]
    pvals = basis_mat[:, :-1].dot(coef)
    bderivs = basis_jac[:, -1:]
    pderivs = basis_jac[:, :-1].dot(coef)
    bhess = basis_hess[:, -1:]
    phess = basis_hess[:, :-1].dot(coef)
    residual = (bvals - pvals)
    residual_jac = bderivs - pderivs
    residual_hess = bhess - phess
    hess = (residual**2*wdx2 + 2*w*(residual*residual_hess+residual_jac**2) +
            4*wdx1*residual*residual_jac).sum(axis=1)
    return np.atleast_2d(hess)


# def pdf_weighted_leja_objective_hess_1d(
#         pdf, pdf_jac, pdf_hess, basis_fun_jac_hess, coef, samples):
#     assert samples.ndim == 2 and samples.shape[0] == 1
#     tmp = basis_fun_jac_hess(samples[0, :])
#     assert tmp.shape[1] % 3 == 0
#     nterms = tmp.shape[1]//3
#     basis_mat = tmp[:, :nterms]
#     basis_jac = tmp[:, nterms:2*nterms]
#     basis_hess = tmp[:, 2*nterms:3*nterms]
#     w = pdf(samples[0, :])
#     wdx1 = pdf_jac(samples[0, :])
#     wdx2 = pdf_hess(samples[0, :])
#     return __christoffel_leja_objective_hess_1d(
#         w, wdx1, wdx2, basis_mat, basis_jac, basis_hess, coef)


def get_initial_guesses_1d(leja_sequence, ranges):
    eps = 1e-6  # must be larger than optimization tolerance
    intervals = np.sort(leja_sequence)
    if np.isfinite(ranges[0]) and (leja_sequence.min() > ranges[0]+eps):
        intervals = np.hstack(([[ranges[0]]], intervals))
    if np.isfinite(ranges[1]) and (leja_sequence.max() < ranges[1]-eps):
        intervals = np.hstack((intervals, [[ranges[1]]]))

    if not np.isfinite(ranges[0]):
        intervals = np.hstack((
            [[min(1.1*leja_sequence.min(), -0.1)]], intervals))
    if not np.isfinite(ranges[1]):
        intervals = np.hstack((
            intervals, [[max(1.1*leja_sequence.max(), 0.1)]]))

    initial_guesses = intervals[:, :-1]+np.diff(intervals)/2.0

    # put intervals in form useful for bounding 1d optimization problems
    intervals = [intervals[0, ii] for ii in range(intervals.shape[1])]
    if not np.isfinite(ranges[0]):
        intervals[0] = -np.inf
    if not np.isfinite(ranges[1]):
        intervals[-1] = np.inf

    return initial_guesses, intervals


def get_christoffel_leja_sequence_1d(
        max_num_leja_samples, initial_points, ranges,
        basis_fun, options, callback=None):

    # def callback(leja_sequence, coef, new_samples, obj_vals,
    #              initial_guesses):
    #     import matplotlib.pyplot as plt
    #     degree = coef.shape[0]-1

    #     def plot_fun(x):
    #         return -christoffel_leja_objective_fun_1d(
    #             partial(basis_fun, nmax=degree+1, deriv_order=0), coef,
    #             x[None, :])

    #     lb = min(leja_sequence.min(), new_samples.min())
    #     ub = max(leja_sequence.max(), new_samples.max())
    #     lb = lb-0.2*abs(lb)
    #     ub = ub+0.2*abs(ub)
    #     lb, ub = -100, 100
    #     xx = np.linspace(lb, ub, 1001)
    #     plt.plot(xx, plot_fun(xx))
    #     print(leja_sequence)
    #     plt.plot(leja_sequence[0, :], plot_fun(leja_sequence[0, :]), 'o',
    #              label="Current samples", ms=15)
    #     plt.plot(new_samples[0, :], obj_vals, 's', label="New samples")
    #     plt.plot(
    #         initial_guesses[0, :], plot_fun(initial_guesses[0, :]), '*',
    #         label="Initial guess")
    #     plt.legend()
    #     plt.show()

    leja_sequence = initial_points.copy()
    nsamples = leja_sequence.shape[1]
    degree = nsamples - 2
    # row_format = "{:<12} {:<25} {:<25}"
    # print(row_format.format('# Samples', 'interp degree', 'sample'))
    while nsamples < max_num_leja_samples:
        degree += 1
        tmp = basis_fun(leja_sequence[0, :], nmax=degree+1,  deriv_order=0)
        nterms = degree+1
        basis_mat = tmp[:, :nterms]
        new_basis = tmp[:, nterms:]
        coef = compute_coefficients_of_christoffel_leja_interpolant_1d(
            basis_mat, new_basis)
        initial_guesses, intervals = get_initial_guesses_1d(
            leja_sequence, ranges)
        new_samples = np.empty((1, initial_guesses.shape[1]))
        obj_vals = np.empty((initial_guesses.shape[1]))

        def fun(x):
            # optimization passes in np.ndarray with ndim == 1
            # need to make it 2D array
            return -christoffel_leja_objective_fun_1d(
                partial(basis_fun, nmax=degree+1, deriv_order=0), coef,
                x[:, None])

        def jac(x):
            return -christoffel_leja_objective_jac_1d(
                partial(basis_fun, nmax=degree+1, deriv_order=1), coef,
                x[:, None])

        # def hess(x):
        #    return -christoffel_leja_objective_hess_1d(
        #        partial(basis_fun, nmax=degree+1, deriv_order=2), coef,
        #        x[:, None])
        hess = None

        opts = options.copy()
        if "artificial_bounds" in opts:
            artificial_bounds = options["artificial_bounds"]
            del opts["artificial_bounds"]
        else:
            artificial_bounds = (-1e3, 1e3)

        for jj in range(initial_guesses.shape[1]):
            initial_guess = initial_guesses[:, jj]
            lb = max(intervals[jj], artificial_bounds[0])
            ub = min(intervals[jj+1], artificial_bounds[1])
            # truncate bounds because christoffel weighted objective
            # will approach one as abs(x) -> oo. These truncated bounds
            # stop x getting to big. This could effect any variable
            # that is not normalized appropriately
            bounds = Bounds([lb], [ub])
            # print(jj, bounds)
            res = pyapprox_minimize(
                fun, initial_guess, jac=jac, hess=hess, bounds=bounds,
                options=opts, method='slsqp')
            new_samples[0, jj] = res.x[0]
            obj_vals[jj] = res.fun
        obj_vals[~np.isfinite(obj_vals)] = np.inf
        best_idx = np.argmin(obj_vals)
        new_sample = new_samples[:, best_idx]
        # print(new_samples, obj_vals, best_idx, new_sample, obj_vals[best_idx])
        if ((abs(new_sample[0]-artificial_bounds[0]) < 1e-8) or
                (abs(new_sample[0]-artificial_bounds[1]) < 1e-8)):
            msg = f"artificial bounds {artificial_bounds} reached. "
            msg += f"Variable should be scaled.\n Nsamples: {nsamples} "
            msg += f"Initial guess: {initial_guess}, bounds: {bounds}"
            warn(msg, UserWarning)
        # print(row_format.format(nsamples, coef.shape[0], new_sample[0]))

        if callback is not None:
            callback(
                leja_sequence, coef, new_samples, obj_vals, initial_guesses)

        leja_sequence = np.hstack([leja_sequence, new_sample[:, None]])
        nsamples += 1
    return leja_sequence


def get_christoffel_leja_quadrature_weights_1d(leja_sequence, growth_rule,
                                               basis_fun, level,
                                               return_weights_for_all_levels):
    """
    Parameters
    ----------
    basis_fun : callable
        Evaluate the basis at a set of points.
        Function with signature

        `basis_fun(samples) -> np.ndarray(nsamples, nterms)`

    samples : np.ndarray (nsamples)
    """
    weight_function = partial(
        sqrt_christoffel_function_inv_1d, basis_fun)
    # need to wrap basis_fun to allow it to be used with generic multivariate
    # function get_leja_sequence_quadrature_weights

    def __basis_fun(x):
        return basis_fun(x[0, :])

    return get_leja_sequence_quadrature_weights(
        leja_sequence, growth_rule, __basis_fun, weight_function,
        level, return_weights_for_all_levels)


def get_pdf_weighted_leja_sequence_1d(
        max_num_leja_samples, initial_points, ranges,
        basis_fun, pdf, pdf_jac, options, callback=None):
    leja_sequence = initial_points.copy()
    nsamples = leja_sequence.shape[1]
    degree = nsamples - 2
    # row_format = "{:<12} {:<25} {:<25}"
    # print(row_format.format('# Samples', 'interp degree', 'sample'))
    while nsamples < max_num_leja_samples:
        degree += 1
        tmp = basis_fun(leja_sequence[0, :], nmax=degree+1,  deriv_order=0)
        nterms = degree+1
        basis_mat = tmp[:, :nterms]
        new_basis = tmp[:, nterms:]
        w = np.sqrt(pdf(leja_sequence[0, :]))
        coef = compute_coefficients_of_pdf_weighted_leja_interpolant_1d(
            w, basis_mat, new_basis)
        initial_guesses, intervals = get_initial_guesses_1d(
            leja_sequence, ranges)
        new_samples = np.empty((1, initial_guesses.shape[1]))
        obj_vals = np.empty((initial_guesses.shape[1]))

        def fun(x):
            # optimization passes in np.ndarray with ndim == 1
            # need to make it 2D array
            return -pdf_weighted_leja_objective_fun_1d(
                pdf, partial(basis_fun, nmax=degree+1, deriv_order=0), coef,
                x[:, None])

        def jac(x):
            return -pdf_weighted_leja_objective_jac_1d(
                pdf, pdf_jac, partial(basis_fun, nmax=degree+1, deriv_order=1),
                coef, x[:, None])

        # def hess(x):
        #     return -pdf_weighted_leja_objective_hess_1d(
        #         partial(basis_fun, nmax=degree+1, deriv_order=2), coef,
        #         x[:, None])

        hess = None
        for jj in range(initial_guesses.shape[1]):
            initial_guess = initial_guesses[:, jj]
            bounds = Bounds([intervals[jj]], [intervals[jj+1]])
            res = pyapprox_minimize(
                fun, initial_guess, jac=jac, hess=hess, bounds=bounds,
                options=options, method='slsqp')
            new_samples[0, jj] = res.x[0]
            obj_vals[jj] = res.fun
        best_idx = np.argmin(obj_vals)
        new_sample = new_samples[:, best_idx]
        # print(row_format.format(nsamples, coef.shape[0], new_sample[0]))

        if callback is not None:
            callback(
                leja_sequence, coef, new_samples, obj_vals, initial_guesses)

        leja_sequence = np.hstack([leja_sequence, new_sample[:, None]])
        nsamples += 1
    return leja_sequence


def get_pdf_weighted_leja_quadrature_weights_1d(leja_sequence, growth_rule,
                                                pdf, basis_fun, level,
                                                return_weights_for_all_levels):
    """
    Parameters
    ----------
    basis_fun : callable
        Evaluate the basis at a set of points.
        Function with signature

        `basis_fun(samples) -> np.ndarray(nsamples, nterms)`

    samples : np.ndarray (nsamples)
    """
    def weight_function(x):
        return pdf(x[0, :])
    # need to wrap basis_fun to allow it to be used with generic multivariate
    # function get_leja_sequence_quadrature_weights

    def __basis_fun(x):
        return basis_fun(x[0, :])

    return get_leja_sequence_quadrature_weights(
        leja_sequence, growth_rule, __basis_fun, weight_function,
        level, return_weights_for_all_levels)


def get_leja_sequence_quadrature_weights(leja_sequence, growth_rule,
                                         basis_matrix_generator,
                                         weight_function, level,
                                         return_weights_for_all_levels):
    """
    Parameters
    ----------
    basis_matrix_generator : callable
        Evaluate the basis at a set of points.
        Function with signature

        `basis_matrix_generator(samples) -> np.ndarray(nsamples, nterms)`

    samples : np.ndarray (nvars, nsamples)
    """
    # precondition matrix to produce better condition number
    basis_matrix = basis_matrix_generator(leja_sequence)
    if return_weights_for_all_levels:
        ordered_weights_1d = []
        for ll in range(level+1):
            # Christoffel preconditioner depends on level so compute weights
            # here instead of before loop
            sqrt_weights = np.sqrt(
                weight_function(leja_sequence[:, :growth_rule(ll)]))
            basis_mat_ll = (basis_matrix[:growth_rule(ll),
                                         :growth_rule(ll)].T*sqrt_weights).T
            basis_mat_ll_inv = np.linalg.inv(basis_mat_ll)
            # make sure to adjust weights to account for preconditioning
            ordered_weights_1d.append(basis_mat_ll_inv[0, :]*sqrt_weights)
    else:
        ll = level
        sqrt_weights = np.sqrt(
            weight_function(leja_sequence[:, :growth_rule(ll)]))
        basis_mat_ll = (basis_matrix[:growth_rule(ll),
                                     :growth_rule(ll)].T*sqrt_weights).T
        basis_mat_ll_inv = np.linalg.inv(basis_mat_ll)
        # make sure to adjust weights to account for preconditioning
        ordered_weights_1d = basis_mat_ll_inv[0, :]*sqrt_weights

    return ordered_weights_1d


def get_candidate_based_christoffel_leja_sequence_1d(
        num_leja_samples, recursion_coeffs, generate_candidate_samples,
        num_candidate_samples, initial_points=None,
        samples_filename=None):
    weight_function = christoffel_preconditioner
    return get_candidate_based_leja_sequence_1d(
        num_leja_samples, recursion_coeffs, generate_candidate_samples,
        num_candidate_samples, weight_function, initial_points,
        samples_filename)


def get_candidate_based_pdf_weighted_leja_sequence_1d(
        num_leja_samples, recursion_coeffs, generate_candidate_samples,
        num_candidate_samples, pdf, initial_points=None,
        samples_filename=None):

    def weight_function(basis_matrix, samples):
        return pdf(samples[0, :])

    return get_candidate_based_leja_sequence_1d(
        num_leja_samples, recursion_coeffs, generate_candidate_samples,
        num_candidate_samples, weight_function, initial_points,
        samples_filename)


def get_candidate_based_leja_sequence_1d(
        num_leja_samples, recursion_coeffs, generate_candidate_samples,
        num_candidate_samples, weight_function, initial_points=None,
        samples_filename=None):

    def generate_basis_matrix(x):
        return evaluate_orthonormal_polynomial_1d(
            x[0, :], num_leja_samples, recursion_coeffs)

    if samples_filename is None or not os.path.exists(samples_filename):
        leja_sequence, __ = get_lu_leja_samples(
            generate_basis_matrix, generate_candidate_samples,
            num_candidate_samples, num_leja_samples,
            preconditioning_function=weight_function,
            initial_samples=initial_points)
        if samples_filename is not None:
            np.savez(samples_filename, samples=leja_sequence)
    else:
        leja_sequence = np.load(samples_filename)['samples']
        assert leja_sequence.shape[1] >= num_leja_samples
        leja_sequence = leja_sequence[:, :num_leja_samples]

    return leja_sequence
