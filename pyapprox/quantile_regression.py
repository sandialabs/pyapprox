import numpy as np
from matplotlib import pyplot as plt
from pyapprox.cvar_regression import conditional_value_at_risk


def scale_linear_system(matrix):
    col_norms = np.linalg.norm(matrix, axis=0)
    # print(col_norms.shape, matrix.shape)
    assert col_norms.shape[0] == matrix.shape[1]
    scaled_matrix = matrix/col_norms
    # print(np.linalg.norm(scaled_matrix, axis=0))
    return scaled_matrix,col_norms


def rescale_linear_system_coefficients(coef, col_norms):
    scaled_coef = coef/col_norms
    return scaled_coef


def quantile_regression(basis_matrix, values, tau):
    from cvxopt import matrix, solvers, spmatrix, sparse
    assert basis_matrix.ndim == 2
    assert values.ndim == 1
    nsamples,nbasis = basis_matrix.shape
    assert values.shape[0] == nsamples
    
    # See https://cvxopt.org/userguide/coneprog.html
    # for documentation on tolerance parameters
    # see https://stats.stackexchange.com/questions/384909/formulating-quantile-regression-as-linear-programming-problem/384913
    solvers.options['show_progress'] = False
    #solvers.options['max_iters'] = 1000
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8

    #basis_matrix,col_norms = scale_linear_system(basis_matrix)
    #values_mean = values.mean(axis=0)
    #scaled_values = values-values_mean
    scaled_values = values
    scale = 1
    
    c_arr = np.hstack(
        (np.zeros(nbasis), tau*np.ones(nsamples),
        (1-tau)*np.ones(nsamples)))[:, np.newaxis]
    c = matrix(c_arr)
    
    Isamp  = np.identity(nsamples)
    A = sparse([[matrix(basis_matrix)], [matrix(Isamp)], [matrix(-Isamp)]])
    b = matrix(scaled_values)
    G = spmatrix(
        -1.0, nbasis+np.arange(2*nsamples), nbasis+np.arange(2*nsamples))
    h = matrix(np.zeros(nbasis+2*nsamples))
    sol = np.asarray(
        solvers.lp(c=c*scale, G=G*scale, h=h*scale, A=A*scale, b=b*scale)['x'])
    coef = sol[:nbasis]
    return coef

    # Ibasis = np.identity(nbasis)
    # Isamp  = np.identity(nsamples)
    # basis_zeros = np.zeros_like(basis_matrix)
    # samples_zeros = np.zeros_like(Isamp)
    # G_arr = np.vstack(
    #     (np.hstack((basis_matrix,Isamp,-Isamp)),
    #     np.hstack((-basis_matrix,-Isamp,Isamp)),
    #     np.hstack((basis_zeros,-Isamp,samples_zeros)),
    #     np.hstack((basis_zeros,samples_zeros,-Isamp)),
    #     ))
    # h_arr = np.hstack((scaled_values,-scaled_values,np.zeros(nsamples),np.zeros(nsamples)))
    # c = matrix(c_arr)
    # G = matrix(G_arr)
    # h = matrix(h_arr)
    # scale=1
    # sol = np.asarray(solvers.lp(c=c*scale, G=G*scale, h=h*scale)['x'])
    # coef = sol[:nbasis]
    # #coef = rescale_linear_system_coefficients(coef,col_norms)
    # #coef[0]+=values_mean
    # return coef

    
def solve_quantile_regression(tau, samples, values, eval_basis_matrix):
    basis_matrix = eval_basis_matrix(samples)
    quantile_coef = quantile_regression(basis_matrix, values.squeeze(), tau=tau)
    # assume first coefficient is for constant term
    quantile_coef[0]=0
    centered_approx_vals = basis_matrix.dot(quantile_coef)[:, 0]
    deviation = conditional_value_at_risk(
        values[:, 0]-centered_approx_vals, tau)
    quantile_coef[0] = deviation
    return quantile_coef


def solve_least_squares_regression(samples, values, eval_basis_matrix,
                                   lamda=0.):
    basis_matrix = eval_basis_matrix(samples)
    lstsq_coef = np.linalg.lstsq(basis_matrix, values, rcond=None)[0]
    lstsq_coef[0] = 0
    centered_approx_vals = basis_matrix.dot(lstsq_coef)[:, 0]
    residuals = values[:, 0]-centered_approx_vals
    deviation = residuals.mean(axis=0)+lamda*np.std(residuals, axis=0)
    lstsq_coef[0] = deviation
    return lstsq_coef
