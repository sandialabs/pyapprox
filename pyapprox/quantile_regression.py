import numpy as np
from matplotlib import pyplot as plt
def scale_linear_system(matrix):
    col_norms = np.linalg.norm(matrix,axis=0)
    print(col_norms.shape,matrix.shape)
    assert col_norms.shape[0]==matrix.shape[1]
    scaled_matrix = matrix/col_norms
    print(np.linalg.norm(scaled_matrix,axis=0))
    return scaled_matrix,col_norms

def rescale_linear_system_coefficients(coef,col_norms):
    scaled_coef = coef/col_norms
    return scaled_coef

def quantile_regression(basis_matrix, values, tau):
    from cvxopt import matrix, solvers, spmatrix, sparse
    assert basis_matrix.ndim==2
    assert values.ndim==1
    nsamples,nbasis = basis_matrix.shape
    assert values.shape[0]==nsamples
    
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
    scaled_values=values
    scale = 1
    
    c_arr = np.hstack(
        (np.zeros(nbasis),tau*np.ones(nsamples),
        (1-tau)*np.ones(nsamples)))[:,np.newaxis]
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
