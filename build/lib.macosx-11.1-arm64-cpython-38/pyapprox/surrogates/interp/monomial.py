import numpy as np
from scipy.special import factorial

from pyapprox.surrogates.interp.manipulate_polynomials import (
    multiply_multivariate_polynomials
)


def univariate_monomial_basis_matrix(max_level, samples):
    assert samples.ndim == 1
    basis_matrix = samples[:, np.newaxis]**np.arange(
        max_level+1)[np.newaxis, :]
    return basis_matrix


# int x^p dx x=a..b
# 1/(p+1)*b^(p+1)-1/(p+1)*a^(p+1)

def monomial_mean_uniform_variables(indices, coeffs, bounds=[-1, 1]):
    """
    Integrate a multivaiate monomial with respect to the uniform probability
    measure on [a,b].

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    coeffs : np.ndarray (num_indices, nqoi)
        The coefficients of each monomial term

    bounds : iterable (2)
        Upper and lower bounds [a, b] of integration.
        Bounds assumed same for each variable

    Return
    ------
    integral : float
        The integral of the monomial
    """
    num_vars, num_indices = indices.shape
    assert coeffs.ndim == 2
    assert coeffs.shape[0] == num_indices
    assert len(bounds) == 2
    # vals = np.prod(((-1.0)**indices+1)/(2.0*(indices+1.0)), axis=0)
    L = (bounds[1]-bounds[0])**num_vars
    vals = np.prod(((bounds[1]**(indices+1.0)) -
                    (bounds[0]**(indices+1.0)))/(indices+1.0), axis=0)/L
    integral = np.sum(vals[:, None]*coeffs, axis=0)
    return integral


def monomial_variance_uniform_variables(indices, coeffs):
    mean = monomial_mean_uniform_variables(indices, coeffs)
    squared_indices, squared_coeffs = multiply_multivariate_polynomials(
        indices, coeffs, indices, coeffs)
    variance = monomial_mean_uniform_variables(
        squared_indices, squared_coeffs)-mean**2
    return variance


def monomial_mean_gaussian_variables(indices, coeffs):
    """
    Integrate a multivaiate monomial with respect to the Gaussian probability
    measure N(0,1).

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    coeffs : np.ndarray (num_indices, nqoi)
        The coefficients of each monomial term

    Return
    ------
    integral : float
        The integral of the monomial
    """
    num_vars, num_indices = indices.shape
    assert coeffs.ndim == 2
    assert coeffs.shape[0] == num_indices
    vals = np.prod((2)**(-indices/2)*factorial(indices)/factorial(indices/2),
                   axis=0)
    II = np.any(indices % 2 == 1, axis=0)
    vals[II] = 0
    integral = np.sum(vals[:, None]*coeffs, axis=0)
    return integral


def monomial_basis_matrix(indices, samples, deriv_order=0):
    """
    Evaluate a multivariate monomial basis at a set of samples.

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    samples : np.ndarray (num_vars, num_samples)
        Samples at which to evaluate the monomial

    deriv_order : integer in [0,1]
       The maximum order of the derivatives to evaluate.

    Return
    ------
    basis_matrix : np.ndarray (num_samples,num_indices)
        The values of the monomial basis at the samples
    """
    # weave code is slower than python version when only computing values of
    # basis. I am Not sure of timings when computing derivatives
    # return c_monomial_basis_matrix(indices,samples)

    num_vars, num_indices = indices.shape
    assert samples.shape[0] == num_vars
    num_samples = samples.shape[1]

    basis_matrix = np.empty(
        ((1+deriv_order*num_vars)*num_samples, num_indices))
    basis_vals_1d = [univariate_monomial_basis_matrix(
        indices[0, :].max(), samples[0, :])]
    basis_matrix[:num_samples, :] = basis_vals_1d[0][:, indices[0, :]]
    for dd in range(1, num_vars):
        basis_vals_1d.append(univariate_monomial_basis_matrix(
            indices[dd, :].max(), samples[dd, :]))
        basis_matrix[:num_samples, :] *= basis_vals_1d[dd][:, indices[dd, :]]

    if deriv_order > 0:
        for ii in range(num_indices):
            index = indices[:, ii]
            for jj in range(num_vars):
                # derivative in jj direction
                basis_vals = basis_vals_1d[jj][:,
                                               max(0, index[jj]-1)]*index[jj]
                # basis values in other directions
                for dd in range(num_vars):
                    if dd != jj:
                        basis_vals *= basis_vals_1d[dd][:, index[dd]]
                basis_matrix[(jj+1)*num_samples:(jj+2)*num_samples, ii] =\
                    basis_vals

    return basis_matrix


def evaluate_monomial(indices, coeffs, samples):
    """
    Evaluate a multivariate monomial at a set of samples.

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    coeffs : np.ndarray (num_indices,num_qoi)
        The coefficients of each monomial term

    samples : np.ndarray (num_vars, num_samples)
        Samples at which to evaluate the monomial

    Return
    ------
    integral : float
        The values of the monomial at the samples
    """
    if coeffs.ndim == 1:
        coeffs = coeffs[:, np.newaxis]
    assert coeffs.ndim == 2
    assert coeffs.shape[0] == indices.shape[1]

    basis_matrix = monomial_basis_matrix(indices, samples)
    values = np.dot(basis_matrix, coeffs)
    return values


def evaluate_monomial_jacobian(indices, coeffs, samples):
    """
    Evaluate the jacobian of a multivariate monomial at a set of samples.

    Parameters
    ----------
    indices : np.ndarray (num_vars, num_indices)
        The exponents of each monomial term

    coeffs : np.ndarray (num_indices,num_qoi)
        The coefficients of each monomial term

    samples : np.ndarray (num_vars, num_samples)
        Samples at which to evaluate the monomial

    Return
    ------
    integral : float
        The values of the monomial at the samples
    """
    if coeffs.ndim == 1:
        coeffs = coeffs[:, np.newaxis]
    assert coeffs.ndim == 2
    assert coeffs.shape[0] == indices.shape[1]

    nvars = indices.shape[0]
    derivs = []
    for dd in range(nvars):
        indices_dd = indices.copy()
        II = np.where(indices_dd[dd] > 0)[0]
        coeffs_dd = coeffs[II]
        indices_dd = indices_dd[:, II]
        print(coeffs.shape, coeffs_dd.shape, indices_dd.shape)
        coeffs_dd = indices_dd[dd][:, None]*coeffs_dd
        indices_dd[dd] -= 1
        basis_matrix = monomial_basis_matrix(indices_dd, samples)
        derivs_dd = np.dot(basis_matrix, coeffs_dd)
        derivs.append(derivs_dd)
    return derivs
