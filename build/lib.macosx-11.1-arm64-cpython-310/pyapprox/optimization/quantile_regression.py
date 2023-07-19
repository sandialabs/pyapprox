import numpy as np
from pyapprox.variables.risk import conditional_value_at_risk


def scale_linear_system(matrix):
    col_norms = np.linalg.norm(matrix, axis=0)
    # print(col_norms.shape, matrix.shape)
    assert col_norms.shape[0] == matrix.shape[1]
    scaled_matrix = matrix/col_norms
    # print(np.linalg.norm(scaled_matrix, axis=0))
    return scaled_matrix, col_norms


def rescale_linear_system_coefficients(coef, col_norms):
    scaled_coef = coef/col_norms
    return scaled_coef


def quantile_regression(basis_matrix, values, tau, opts={}):
    from cvxopt import matrix, solvers, spmatrix, sparse
    assert basis_matrix.ndim == 2
    assert values.ndim == 1
    nsamples, nbasis = basis_matrix.shape
    assert values.shape[0] == nsamples

    # See https://cvxopt.org/userguide/coneprog.html
    # for documentation on tolerance parameters
    # see https://stats.stackexchange.com/questions/384909/formulating-quantile-regression-as-linear-programming-problem/384913
    solvers.options['show_progress'] = opts.get("show_progress", False)
    if "max_iters" in opts:
        solvers.options['max_iters'] = opts["max_iters"]
    solvers.options['abstol'] = opts.get("abstol", 1e-8)
    solvers.options['reltol'] = opts.get("reltol", 1e-8)
    solvers.options['feastol'] = opts.get("feastol", 1e-8)

    for key in opts.keys():
        if key not in ["abstol", "reltol", "feastol", "show_progress",
                       "max_iters"]:
            raise ValueError(f"Option {key} not supported")

    # basis_matrix, col_norms = scale_linear_system(basis_matrix)
    # values_mean = values.mean(axis=0)
    # scaled_values = values-values_mean
    scaled_values = values
    scale = 1

    c_arr = np.hstack(
        (np.zeros(nbasis), tau*np.ones(nsamples),
         (1-tau)*np.ones(nsamples)))[:, np.newaxis]
    c = matrix(c_arr)

    Isamp = np.identity(nsamples)
    A = sparse([[matrix(basis_matrix)], [matrix(Isamp)], [matrix(-Isamp)]])
    b = matrix(scaled_values)
    G = spmatrix(
        -1.0, nbasis+np.arange(2*nsamples), nbasis+np.arange(2*nsamples))
    h = matrix(np.zeros(nbasis+2*nsamples))
    # print(np.array(A).shape, np.array(G), np.array(b), np.array(h), nbasis, nsamples, Isamp.shape, A, basis_matrix.shape)
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


def solve_quantile_regression(tau, samples, values, eval_basis_matrix,
                              normalize_vals=False, opts={}):
    r"""
    Solve quantile regression problems.

    Parameters
    ----------
    tau : float
        The quantile in [0, 1)

    samples : np.ndarary (nvars, nsamples)
        The training samples

    values : np.ndarary (nsamples, 1)
        The function values at the training samples

    eval_basis_matrix : callable
        A function returning the basis evaluated at the set of samples
        with signature
        ``eval_basis_matrix(samples) -> np.ndarray (nsamples, nbasis)``

    normalize_vals : boolean
        True - normalize the training values
        False - use the raw training values
    """
    basis_matrix = eval_basis_matrix(samples)
    if basis_matrix.shape[0] < basis_matrix.shape[1]:
        raise ValueError("System is under-determined")
    if normalize_vals is True:
        factor = values[:, 0].std()
        vals = values.copy()/factor
    else:
        vals = values
    quantile_coef = quantile_regression(
        basis_matrix, vals.squeeze(), tau=tau, opts=opts)
    if normalize_vals is True:
        quantile_coef *= factor
    # assume first coefficient is for constant term
    quantile_coef[0] = 0
    centered_approx_vals = basis_matrix.dot(quantile_coef)[:, 0]
    deviation = conditional_value_at_risk(
        values[:, 0]-centered_approx_vals, tau)
    quantile_coef[0] = deviation
    return quantile_coef


def solve_least_squares_regression(samples, values, eval_basis_matrix,
                                   lamda=0., normalize_vals=True):
    """
    Solve the safety margins least squares regression problem.

    Parameters
    ----------
    samples : np.ndarary (nvars, nsamples)
        The training samples

    values : np.ndarary (nsamples, 1)
        The function values at the training samples

    eval_basis_matrix : callable
        A function returning the basis evaluated at the set of samples
        with signature
        ``eval_basis_matrix(samples) -> np.ndarray (nsamples, nbasis)``

    lambda : float
        The number [0, infty) of standard deviations used to determine
        risk averse shift

    normalize_vals : boolean
        True - normalize the training values
        False - use the raw training values
    """

    basis_matrix = eval_basis_matrix(samples)
    # assume first coefficient is for constant term
    if normalize_vals is True:
        factor = values[:, 0].std()
        vals = values.copy()/factor
    else:
        vals = values
    lstsq_coef = np.linalg.lstsq(basis_matrix, vals, rcond=None)[0]
    if normalize_vals is True:
        lstsq_coef *= factor
    lstsq_coef[0] = 0
    centered_approx_vals = basis_matrix.dot(lstsq_coef)[:, 0]
    residuals = values[:, 0]-centered_approx_vals
    deviation = residuals.mean(axis=0)+lamda*np.std(residuals, axis=0)
    lstsq_coef[0] = deviation
    return lstsq_coef
