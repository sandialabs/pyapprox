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
    

def leja_objective_and_gradient_1d(samples, leja_sequence, basis_fun, degree,
                                   coeff, deriv_order=0):
    """
    Evaluate the Leja objective at a set of samples.

    Parameters
    ----------
    samples : np.ndarray (num_vars, num_samples)
        The sample at which to evaluate the leja_objective

    leja_sequence : np.ndarray (num_vars, num_leja_samples)
        The sample already in the Leja sequence

    deriv_order : integer
        Flag specifiying whether to compute gradients of the objective

    new_indices : np.ndarray (num_vars, num_new_indices)
        The new indices that are considered when choosing next sample
        in the Leja sequence

    coeff : np.ndarray (num_indices, num_new_indices)
        The coefficient of the approximation that interpolates the polynomial
        terms specified by new_indices

    Return
    ------
    residuals : np.ndarray(num_new_indices,num_samples):

    objective_vals : np.ndarray (num_samples)
        The values of the objective at samples

    objective_grads : np.ndarray (num_vars,num_samples)
        The gradient of the objective at samples. Return only
        if deriv_order==1
    """
    assert samples.ndim == 2
    num_vars, num_samples = samples.shape
    assert num_samples == 1

    indices = poly.indices.copy()
    poly.set_indices(new_indices)
    basis_matrix_for_new_indices_at_samples = poly.basis_matrix(
        samples,{'deriv_order':deriv_order})
    if deriv_order==1:
        basis_deriv_matrix_for_new_indices_at_samples = \
          basis_matrix_for_new_indices_at_samples[1:,:]
    basis_matrix_for_new_indices_at_samples = \
      basis_matrix_for_new_indices_at_samples[:1,:]
    poly.set_indices(indices)

    basis_matrix_at_samples = poly.basis_matrix(
        samples[:,:1],{'deriv_order':deriv_order})
    if deriv_order==1:
        basis_deriv_matrix_at_samples = basis_matrix_at_samples[1:,:]
    basis_matrix_at_samples = basis_matrix_at_samples[:1,:]


    weights = weight_function(samples)
    # to avoid division by zero
    weights = np.maximum(weights,0)
    assert weights.ndim==1
    sqrt_weights = np.sqrt(weights)
    
    poly_vals = np.dot(basis_matrix_at_samples,coeff)

    unweighted_residual = basis_matrix_for_new_indices_at_samples-poly_vals
    residual = sqrt_weights*unweighted_residual
    
    num_residual_entries = residual.shape[1]

    if deriv_order==0:
        return (residual,)
    
    poly_derivs = np.dot(basis_deriv_matrix_at_samples,coeff)
    weight_derivs = weight_function_deriv(samples)

    unweighted_residual_derivs = \
      poly_derivs-basis_deriv_matrix_for_new_indices_at_samples
    
    jacobian = np.zeros((num_residual_entries,num_vars),dtype=float)
    I = np.where(weights>0)[0]
    for dd in range(num_vars):
        jacobian[I,dd]=(
            unweighted_residual[0,I]*weight_derivs[dd,I]/(2.0*sqrt_weights[I])-
            unweighted_residual_derivs[dd,I]*sqrt_weights[I])
    assert residual.ndim==2
    return residual, jacobian
