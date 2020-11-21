#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


# The functions begining with __ are useful when reusing information during
# optimiaztion. For example computing the leja objective Hessian requires
# data also used when computing the value and Jacobian


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
        True - return N/K(x) where N is the number of basis terms
        False - return 1/K(x)
    """
    return __sqrt_christoffel_function_inv_1d(
        basis_fun(samples[0, :]), normalize)


def __sqrt_christoffel_function_inv_1d(basis_mat, normalize):
    vals = 1./np.linalg.norm(basis_mat, axis=1)
    if normalize is True:
        vals *= basis_matrix.shape[1]
    return vals


def sqrt_christoffel_function_inv_jac_1d(basis_fun_and_jac, samples,
                                         normalize=False):
    r"""
    Return the first_derivative wrt x of the inverse of the square-root of the 
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
        True - return nterms/K(x) where nterms is the number of basis terms
        False - return 1/K(x)
    """
    basis_vals_and_derivs = basis_fun_and_jac(samples[0, :])
    assert basis_vals_and_derivs.shape[1]%2 == 0
    nterms = basis_vals_and_derivs.shape[1]//2
    basis_mat = basis_vals_and_derivs[:, :nterms]
    basis_jac = basis_vals_and_derivs[:, nterms:]
    return __sqrt_christoffel_function_inv_jac_1d(
        basis_mat, basis_jac, normalize)


def __sqrt_christoffel_function_inv_jac_1d(basis_mat, basis_jac, normalize):
    vals = -2*(basis_mat*basis_jac).sum(axis=1)
    vals /= (2*np.sum(basis_mat**2, axis=1)**(1.5))
    if normalize is True:
        vals *= nterms
    return vals


def sqrt_christoffel_function_inv_hess_1d(basis_fun_jac_hess, samples,
                                          normalize=False):
    r"""
    Return the second derivative wrt x of the inverse of the square-root of the 
    Christoffel function at a set of samples. That is compute

    .. math:: 

       \frac{\partial^2}{\partial x^2}\frac{1}{K(x)^{1/2}} = 
       -\frac{K^\prime(x)^2-2K(x)K^{\prime\prime}(x)}{4K(x)^{5/2}}

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
        True - return nterms/K(x) where nterms is the number of basis terms
        False - return 1/K(x)
    """
    tmp = basis_fun_jac_hess(samples[0, :])
    assert tmp.shape[1]%3 == 0
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
        vals *= nterms
    return vals


def compute_coefficients_of_leja_interpolant_1d(basis_mat, new_basis):
    # Todo replace with update of LU factorization
    assert new_basis.ndim == 2 and new_basis.shape[1] == 1
    w = __sqrt_christoffel_function_inv_1d(basis_mat, False)
    coef = np.linalg.lstsq(
        w[:, None]*basis_mat, w[:, None]*new_basis, rcond=None)[0]
    return coef


def leja_objective_fun_1d(basis_fun, coef, samples):
    """
    Parameters
    ----------
    samples : np.ndarray (nvars, nsamples)
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
    return __leja_objective_fun_1d(basis_vals[:, :-1], basis_vals[:, -1:], coef)

    
def __leja_objective_fun_1d(basis_mat, new_basis, coef):
    """
    Parameters
    ----------
    basis_mat : np.array (nsamples, nterms)
        Values of the basis of degree k at new sample x not already in the Leja 
        sequence. Note nterms=k+1

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
    w = __sqrt_christoffel_function_inv_1d(
        np.hstack([basis_mat, new_basis]), False)
    return w*np.sum(residual**2, axis=1)


def leja_objective_jac_1d(basis_fun_jac, coef, samples):
    assert samples.ndim == 2 and samples.shape[0] == 1
    tmp = basis_fun_jac(samples[0, :])
    assert tmp.shape[1]%2 == 0
    nterms = tmp.shape[1]//2
    basis_mat = tmp[:, :nterms]
    basis_jac = tmp[:, nterms:]
    return __leja_objective_jac_1d(basis_mat, basis_jac, coef)


def __leja_objective_jac_1d(basis_mat, basis_jac, coef):
    assert basis_mat.ndim == 2
    assert basis_mat.shape == basis_jac.shape
    assert coef.ndim == 2 and coef.shape[1] == 1
    w = __sqrt_christoffel_function_inv_1d(basis_mat, False)
    wdx1 = __sqrt_christoffel_function_inv_jac_1d(basis_mat, basis_jac, False)
    bvals = basis_mat[:, -1:]
    pvals = basis_mat[:, :-1].dot(coef)
    bderivs = basis_jac[:, -1:]
    pderivs = basis_jac[:, :-1].dot(coef)
    residual = (bvals - pvals)
    residual_jac = bderivs - pderivs
    jac = (residual**2*wdx1 + 2*w*residual*residual_jac).sum(axis=1)
    return jac


def leja_objective_hess_1d(basis_fun_jac_hess, coef, samples):
    assert samples.ndim == 2 and samples.shape[0] == 1
    tmp = basis_fun_jac_hess(samples[0, :])
    assert tmp.shape[1]%3 == 0
    nterms = tmp.shape[1]//3
    basis_mat = tmp[:, :nterms]
    basis_jac = tmp[:, nterms:2*nterms]
    basis_hess = tmp[:, 2*nterms:3*nterms]
    return __leja_objective_hess_1d(basis_mat, basis_jac, basis_hess, coef)


def __leja_objective_hess_1d(basis_mat, basis_jac, basis_hess, coef):
    assert basis_mat.ndim == 2
    assert basis_mat.shape == basis_jac.shape
    assert coef.ndim == 2 and coef.shape[1] == 1
    w = __sqrt_christoffel_function_inv_1d(basis_mat, False)
    wdx1 = __sqrt_christoffel_function_inv_jac_1d(basis_mat, basis_jac, False)
    wdx2 = __sqrt_christoffel_function_inv_hess_1d(
        basis_mat, basis_jac, basis_hess, False)
    bvals = basis_mat[:, -1:]
    pvals = basis_mat[:, :-1].dot(coef)
    bderivs = basis_jac[:, -1:]
    pderivs = basis_jac[:, :-1].dot(coef)
    bhess = basis_hess[:, -1:]
    phess = basis_hess[:, :-1].dot(coef)
    residual = (bvals - pvals)
    residual_jac = bderivs - pderivs
    residual_hess = bhess - phess
    hess = (residual**2*wdx2 + 2*w*(residual*residual_hess+residual_jac**2)+
           4*wdx1*residual*residual_jac).sum(axis=1)
    return np.atleast_2d(hess)


def get_initial_guesses_1d(leja_sequence, ranges):
    eps = 1e-6 # must be larger than optimization tolerance
    intervals = np.sort(leja_sequence)
    if ranges[0] != None and (leja_sequence.min()>ranges[0]+eps):
        intervals = np.hstack(([[ranges[0]]], intervals))
    if ranges[1] != None and (leja_sequence.max()<ranges[1]-eps):
        intervals = np.hstack((intervals, [[ranges[1]]]))

    if ranges[0] is None:
        intervals = np.hstack((
            [[min(1.1*leja_sequence.min(), -0.1)]], intervals))
    if ranges[1] is None:
        intervals = np.hstack((
            intervals, [[max(1.1*leja_sequence.max(), 0.1)]]))

    diff = np.diff(intervals)
    initial_guesses = intervals[:, :-1]+np.diff(intervals)/2.0

    # put intervals in form useful for bounding 1d optimization problems
    intervals = [intervals[0, ii] for ii in range(intervals.shape[1])]
    if ranges[0] is None:
        intervals[0] = None
    if ranges[1] is None:
        intervals[-1] = None
    
    return initial_guesses, intervals


def minimize_leja_objective_1d(initial_guess, bounds, options):
    bounds = Bounds([bounds[0]], [bounds[1]])
    res = pyapprox_minimize(
        fun, initial_guess, jac, hess, bounds=bounds, options=options)
    return res


from scipy.optimize import Bounds
from pyapprox.rol_minimize import pyapprox_minimize
from functools import partial
def get_leja_sequence_1d(max_num_leja_samples, initial_points, ranges,
                         basis_fun, options, callback=None):
    leja_sequence = initial_points.copy()
    nsamples = leja_sequence.shape[1]
    degree = nsamples - 2
    row_format = "{:<12} {:<25} {:<25}"
    print(row_format.format('# Samples', 'interp degree', 'sample'))
    while nsamples < max_num_leja_samples:
        degree += 1
        tmp = basis_fun(leja_sequence[0, :], nmax=degree+1,  deriv_order=0)
        nterms = degree+1
        basis_mat = tmp[:, :nterms]
        new_basis = tmp[:, nterms:]
        coef = compute_coefficients_of_leja_interpolant_1d(
            basis_mat, new_basis)
        initial_guesses, intervals = get_initial_guesses_1d(
            leja_sequence, ranges)
        new_samples = np.empty((1, initial_guesses.shape[1]))
        obj_vals = np.empty((initial_guesses.shape[1]))
        def fun(x):
            # optimization passes in np.ndarray with ndim == 1
            # need to make it 2D array
            return -leja_objective_fun_1d(
                partial(basis_fun, nmax=degree+1, deriv_order=0), coef,
                x[:, None])
        def jac(x):
            return -leja_objective_jac_1d(
                partial(basis_fun, nmax=degree+1, deriv_order=1), coef,
                x[:, None])
        def hess(x):
            return -leja_objective_hess_1d(
                partial(basis_fun, nmax=degree+1, deriv_order=2), coef,
                x[:, None])
        for jj in range(initial_guesses.shape[1]):
            initial_guess = initial_guesses[:, jj]
            bounds = Bounds([intervals[jj]], [intervals[jj+1]])
            res = pyapprox_minimize(
                fun, initial_guess, jac=jac, hess=hess, bounds=bounds,
                options=options, method='slsqp')
            new_samples[0, jj] = res.x
            obj_vals[jj] = res.fun
        I = np.argmin(obj_vals)
        new_sample = new_samples[:, I]
        print(row_format.format(nsamples, coef.shape[0], new_sample[0]))

        if callback is not None:
            callback(
                leja_sequence, coef, new_samples, obj_vals, initial_guesses)
            
        leja_sequence = np.hstack([leja_sequence, new_sample[:, None]])
        nsamples += 1
    return leja_sequence
