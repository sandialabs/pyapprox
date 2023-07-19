import numpy as np
from scipy import sparse

from pyapprox.variables.risk import value_at_risk
from pyapprox.optimization.first_order_stochastic_dominance import (
    smooth_max_function_log, smooth_max_function_first_derivative_log,
    smooth_max_function_second_derivative_log
)


def conditional_value_at_risk_subgradient(
        samples, alpha, weights=None, samples_sorted=False):
    assert samples.ndim == 1 or samples.shape[1] == 1
    samples = samples.squeeze()
    num_samples = samples.shape[0]
    if weights is None:
        weights = np.ones(num_samples)/num_samples
    assert np.allclose(weights.sum(), 1)
    assert weights.ndim == 1 or weights.shape[1] == 1
    if not samples_sorted:
        II = np.argsort(samples)
        xx, ww = samples[II], weights[II]
    else:
        xx, ww = samples, weights
    VaR, index = value_at_risk(xx, alpha, ww, samples_sorted=True)
    grad = np.empty(num_samples)
    grad[:index] = 0
    grad[index] = 1/(1-alpha)*(weights[:index+1].sum()-alpha)
    grad[index+1:] = 1/(1-alpha)*weights[index+1:]
    if not samples_sorted:
        # grad is for sorted samples so revert to original ordering
        grad = grad[np.argsort(II)]
    return grad


def smooth_max_function(smoother_type, eps, x):
    if smoother_type == 0:
        return smooth_max_function_log(eps, 0, x)
    elif smoother_type == 1:
        vals = np.zeros(x.shape)
        II = np.where((x > 0) & (x < eps))  # [0]
        vals[II] = x[II]**3/eps**2*(1-x[II]/(2*eps))
        J = np.where(x >= eps)  # [0]
        vals[J] = x[J]-eps/2
        return vals
    else:
        msg = "incorrect smoother_type"
        raise Exception(msg)


def smooth_max_function_first_derivative(smoother_type, eps, x):
    if smoother_type == 0:
        return smooth_max_function_first_derivative_log(eps, 0, x)
    elif smoother_type == 1:
        vals = np.zeros(x.shape)
        II = np.where((x > 0) & (x < eps))  # [0]
        vals[II] = (x[II]**2*(3*eps-2*x[II]))/eps**3
        J = np.where(x >= eps)  # [0]
        vals[J] = 1
        return vals.T
    else:
        msg = "incorrect smoother_type"
        raise Exception(msg)


def smooth_max_function_second_derivative(smoother_type, eps, x):
    if smoother_type == 0:
        return smooth_max_function_second_derivative_log(eps, 0, x)
    elif smoother_type == 1:
        vals = np.zeros(x.shape)
        II = np.where((x > 0) & (x < eps))  # [0]
        vals[II] = 6*x[II]*(eps-x[II])/eps**3
        return vals
    else:
        msg = "incorrect smoother_type"
        raise Exception(msg)


def smooth_conditional_value_at_risk(
        smoother_type, eps, alpha, x, weights=None):
    r"""
    used for solving

    min_{x,t} t+1/(1-\alpha)E([x-t]^+)
    We need to minimize both t and x because t is a function of X
    """
    return smooth_conditional_value_at_risk_split(
        smoother_type, eps, alpha, x[:-1], x[-1], weights)


def smooth_conditional_value_at_risk_split(
        smoother_type, eps, alpha, x, t, weights=None):
    if x.ndim == 1:
        x = x[:, None]
    assert x.ndim == 2 and x.shape[1] == 1
    if weights is None:
        return t+smooth_max_function(
            smoother_type, eps, x-t)[:, 0].mean()/(1-alpha)

    assert weights.ndim == 1 and weights.shape[0] == x.shape[0]
    return t+smooth_max_function(
        smoother_type, eps, x-t)[:, 0].dot(weights)/(1-alpha)


def smooth_conditional_value_at_risk_gradient(
        smoother_type, eps, alpha, x, weights=None):
    return smooth_conditional_value_at_risk_gradient_split(
        smoother_type, eps, alpha, x[:-1], x[-1], weights)


def smooth_conditional_value_at_risk_gradient_split(
        smoother_type, eps, alpha, x, t, weights=None):
    if x.ndim == 1:
        x = x[:, None]
    assert x.ndim == 2 and x.shape[1] == 1
    if weights is None:
        weights = np.ones(x.shape[0])/(x.shape[0])
    assert weights.ndim == 1 and weights.shape[0] == x.shape[0]
    # nsamples = x.shape[0]-1
    grad = np.empty(x.shape[0]+1)
    grad[-1] = 1-smooth_max_function_first_derivative(
        smoother_type, eps, x-t).squeeze().dot(weights)/((1-alpha))
    grad[:-1] = smooth_max_function_first_derivative(
        smoother_type, eps, x-t).squeeze()/(1-alpha)*weights
    return grad


def smooth_conditional_value_at_risk_composition(
        smoother_type, eps, alpha, fun, jac, x, weights=None):
    """
    fun : callable
         A function with signature

         ``fun(x)->ndarray (nsamples,1)``

        The output values used to compute CVAR.

    jac : callable
        The jacobian of ``fun`` (with resepct to the variables ``z`` )
        with signature

        ``jac(x) -> np.ndarray (nsamples,nvars)``

        where nsamples is the number of values used to compute
        CVAR. Typically this function will be a loop evaluating
        the gradient of a function (with resepct to x) ``fun(x,z)`` at
        realizations of random variables ``z``
    """
    return smooth_conditional_value_at_risk_composition_split(
        smoother_type, eps, alpha, fun, jac, x[:-1], x[-1], weights)


def smooth_conditional_value_at_risk_composition_split(
        smoother_type, eps, alpha, fun, jac, x, t, weights=None):
    assert x.ndim == 2 and x.shape[1] == 1
    fun_vals = fun(x)
    assert fun_vals.ndim == 2 and fun_vals.shape[1] == 1
    fun_vals = fun_vals[:, 0]
    nsamples = fun_vals.shape[0]
    if weights is None:
        weights = np.ones(nsamples)/(nsamples)
    assert weights.ndim == 1 and weights.shape[0] == nsamples
    obj_val = t+smooth_max_function(
        smoother_type, eps, fun_vals-t).dot(weights)/(1-alpha)

    grad = np.empty(x.shape[0]+1)
    grad[-1] = 1-smooth_max_function_first_derivative(
        smoother_type, eps, fun_vals-t).dot(weights)/((1-alpha))
    jac_vals = jac(x)
    assert jac_vals.shape == (nsamples, x.shape[0])
    grad[:-1] = (smooth_max_function_first_derivative(
        smoother_type, eps, fun_vals-t)[:, None]*jac_vals/(1-alpha)).T.dot(weights)
    return obj_val, grad


def cvar_regression_quadrature(basis_matrix, values, alpha, nquad_intervals,
                               verbosity=1, trapezoid_rule=False,
                               solver_name='cvxopt'):
    """
    solver_name = 'cvxopt'
    solver_name='glpk'

    trapezoid works but default option is better.
    """
    from cvxopt import matrix, solvers, spmatrix
    assert alpha < 1 and alpha > 0
    basis_matrix = basis_matrix[:, 1:]
    assert basis_matrix.ndim == 2
    assert values.ndim == 1
    nsamples, nbasis = basis_matrix.shape
    assert values.shape[0] == nsamples

    if not trapezoid_rule:
        # left-hand piecewise constant quadrature rule
        beta = np.linspace(alpha, 1, nquad_intervals +
                           2)[:-1]  # quadrature points
        dx = beta[1]-beta[0]
        weights = dx*np.ones(beta.shape[0])
        nuvars = weights.shape[0]
        nvconstraints = nsamples*nuvars
        num_opt_vars = nbasis + nuvars + nvconstraints
        num_constraints = 2*nsamples*nuvars
    else:
        beta = np.linspace(alpha, 1, nquad_intervals+1)  # quadrature points
        dx = beta[1]-beta[0]
        weights = dx*np.ones(beta.shape[0])
        weights[0] /= 2
        weights[-1] /= 2
        weights = weights[:-1]  # ignore left hand side
        beta = beta[:-1]
        nuvars = weights.shape[0]
        nvconstraints = nsamples*nuvars
        num_opt_vars = nbasis + nuvars + nvconstraints + 1
        num_constraints = 2*nsamples*nuvars+nsamples

    v_coef = weights/(1-beta)*1./nsamples*1/(1-alpha)

    Iquad = np.identity(nuvars)
    Iv = np.identity(nvconstraints)

    # num_quad_point = mu
    # nsamples = nu
    # nbasis = m

    # design vars [c_1,...,c_m,u_1,...,u_{mu+1},v_1,...,v_{mu+1}nu]
    # v_ij variables ordering: loop through j fastest, e.g. v_11,v_{12} etc

    if not trapezoid_rule:
        c_arr = np.hstack((
            basis_matrix.sum(axis=0)/nsamples,
            1/(1-alpha)*weights,
            np.repeat(v_coef, nsamples)))

        # # v_ij+h'c+u_i <=y_j
        # constraints_1 = np.hstack((
        #     -np.tile(basis_matrix,(nuvars,1)),
        #     -np.repeat(Iquad,nsamples,axis=0),-Iv))
        # # v_ij >=0
        # constraints_3 = np.hstack((
        #    np.zeros((nvconstraints,nbasis+nuvars)),-Iv))

        # G_arr = np.vstack((constraints_1,constraints_3))
        # assert G_arr.shape[0]==num_constraints
        # assert G_arr.shape[1]==num_opt_vars
        # assert c_arr.shape[0]==num_opt_vars
        # I,J,data = sparse.find(G_arr)
        # G = spmatrix(data,I,J,size=G_arr.shape)

        # v_ij+h'c+u_i <=y_j
        constraints_1_shape = (nvconstraints, num_opt_vars)
        constraints_1a_I = np.repeat(np.arange(nvconstraints), nbasis)
        constraints_1a_J = np.tile(np.arange(nbasis), nvconstraints)
        constraints_1a_data = -np.tile(basis_matrix, (nquad_intervals+1, 1))

        constraints_1b_I = np.arange(nvconstraints)
        constraints_1b_J = np.repeat(
            np.arange(nquad_intervals+1), nsamples)+nbasis
        constraints_1b_data = -np.repeat(np.ones(nquad_intervals+1), nsamples)

        ii = nbasis+nquad_intervals+1
        jj = ii+nvconstraints
        constraints_1c_I = np.arange(nvconstraints)
        constraints_1c_J = np.arange(ii, jj)
        constraints_1c_data = -np.ones((nquad_intervals+1)*nsamples)

        constraints_1_data = np.hstack((
            constraints_1a_data.flatten(), constraints_1b_data,
            constraints_1c_data))
        constraints_1_I = np.hstack(
            (constraints_1a_I, constraints_1b_I, constraints_1c_I))
        constraints_1_J = np.hstack(
            (constraints_1a_J, constraints_1b_J, constraints_1c_J))

        # v_ij >=0
        constraints_3_I = np.arange(
            constraints_1_shape[0], constraints_1_shape[0]+nvconstraints)
        constraints_3_J = np.arange(
            nbasis+nquad_intervals+1, nbasis+nquad_intervals+1+nvconstraints)
        constraints_3_data = -np.ones(nvconstraints)

        constraints_shape = (num_constraints, num_opt_vars)
        constraints_I = np.hstack((constraints_1_I, constraints_3_I))
        constraints_J = np.hstack((constraints_1_J, constraints_3_J))
        constraints_data = np.hstack((constraints_1_data, constraints_3_data))
        G = spmatrix(
            constraints_data, constraints_I, constraints_J,
            size=constraints_shape)
        # assert np.allclose(np.asarray(matrix(G)),G_arr)

        h_arr = np.hstack((
            -np.tile(values, nuvars),
            np.zeros(nvconstraints)))

    else:
        c_arr = np.hstack((
            basis_matrix.sum(axis=0)/nsamples,
            1/(1-alpha)*weights,
            np.repeat(v_coef, nsamples),
            1/(nsamples*(1-alpha))*np.ones(1)))

        # v_ij+h'c+u_i <=y_j
        constraints_1 = np.hstack((
            -np.tile(basis_matrix, (nquad_intervals, 1)),
            -np.repeat(Iquad, nsamples, axis=0),
            -Iv, np.zeros((nvconstraints, 1))))

        # W+h'c<=y_j
        constraints_2 = np.hstack((
            -basis_matrix,
            np.zeros((nsamples, nuvars)),
            np.zeros((nsamples, nvconstraints)),
            -np.ones((nsamples, 1))))

        # v_ij >=0
        constraints_3 = np.hstack((
            np.zeros((nvconstraints, nbasis+nquad_intervals)), -Iv,
            np.zeros((nvconstraints, 1))))

        G_arr = np.vstack((constraints_1, constraints_2, constraints_3))

        h_arr = np.hstack((
            -np.tile(values, nuvars),
            -values,
            np.zeros(nvconstraints)))

        assert G_arr.shape[0] == num_constraints
        assert G_arr.shape[1] == num_opt_vars
        assert c_arr.shape[0] == num_opt_vars
        I, J, data = sparse.find(G_arr)
        G = spmatrix(data, I, J, size=G_arr.shape)

    c = matrix(c_arr)
    h = matrix(h_arr)
    if verbosity < 1:
        solvers.options['show_progress'] = False
    else:
        solvers.options['show_progress'] = True

    # solvers.options['abstol'] = 1e-10
    # solvers.options['reltol'] = 1e-10
    # solvers.options['feastol'] = 1e-10

    sol = np.asarray(
        solvers.lp(c=c, G=G, h=h, solver=solver_name)['x'])[:nbasis]
    residuals = values-basis_matrix.dot(sol)[:, 0]
    coef = np.append(conditional_value_at_risk(residuals, alpha), sol)
    return coef


def cvar_regression(basis_matrix, values, alpha, verbosity=1):
    """
    Solve conditional value at risk (CVaR) regression problems.

    Parameters
    ----------
    basis_matrix : np.ndarray (nsamples, nbasis)
        A basis evaluated at the set of training samples


    values : np.ndarray (nsamples, 1)
        The function values at the set of training samples

    alpha : float
        The quantile in [0, 1) defining CVaR

    verbosity : integer
        The verbosity level
    """
    from cvxopt import matrix, solvers, spmatrix
    # do not include constant basis in optimization
    assert alpha < 1 and alpha > 0
    basis_matrix = basis_matrix[:, 1:]
    assert basis_matrix.ndim == 2
    assert values.ndim == 1
    nsamples, nbasis = basis_matrix.shape
    assert values.shape[0] == nsamples

    active_index = int(np.ceil(alpha*nsamples)) - \
        1  # 0 based index 0,...,nsamples-1
    nactive_samples = nsamples-(active_index+1)
    assert nactive_samples > 0, ('no samples in alpha quantile')
    beta = np.arange(1, nsamples+1, dtype=float)/nsamples
    beta[active_index-1] = alpha


    beta_diff = np.diff(beta[active_index-1:-1])
    # print (beta_diff
    assert beta_diff.shape[0] == nactive_samples
    v_coef = np.log(1 - beta[active_index-1:-2]) - np.log(
        1 - beta[active_index:-1])
    v_coef /= nsamples*(1-alpha)

    nvconstraints = nsamples*nactive_samples
    Iv = np.identity(nvconstraints)
    Isamp = np.identity(nsamples)
    Iactsamp = np.identity(nactive_samples)

    # nactive_samples = p
    # nsamples = m
    # nbasis = n

    # design vars [c_1,...,c_n,u1,...,u_{m-p},v_1,...,v_{m-p}m,w]

    c_arr = np.hstack((
        basis_matrix.sum(axis=0)/nsamples,
        1/(1-alpha)*beta_diff,
        # np.tile(v_coef,nsamples),  # tile([1,2],2)   = [1,2,1,2]
        np.repeat(v_coef, nsamples),  # repeat([1,2],2) = [1,1,2,2]
        1./(nsamples*(1-alpha))*np.ones(1)))

    num_opt_vars = nbasis + nactive_samples + nvconstraints + 1
    # v_ij variables ordering: loop through j fastest, e.g. v_11,v_{12} etc

    # v_ij+h'c+u_i <=y_j
    constraints_1 = np.hstack((
        -np.tile(basis_matrix, (nactive_samples, 1)),
        -np.repeat(Iactsamp, nsamples, axis=0),
        -Iv,
        np.zeros((nvconstraints, 1))))

    # W+h'c<=y_j
    constraints_2 = np.hstack((
        -basis_matrix,
        np.zeros((nsamples, nactive_samples)),
        np.zeros((nsamples, nvconstraints)),
        -np.ones((nsamples, 1))))

    # v_ij >=0
    constraints_3 = np.hstack((
        np.zeros((nvconstraints, nbasis+nactive_samples)),
        -Iv, np.zeros((nvconstraints, 1))))

    # print ((constraints_1.shape, constraints_2.shape, constraints_3.shape)
    G_arr = np.vstack((constraints_1, constraints_2, constraints_3))

    h_arr = np.hstack((
        -np.tile(values, nactive_samples),
        -values, np.zeros(nvconstraints)))

    assert G_arr.shape[1] == num_opt_vars
    assert G_arr.shape[0] == h_arr.shape[0]
    assert c_arr.shape[0] == num_opt_vars

    c = matrix(c_arr)
    #G = matrix(G_arr)
    h = matrix(h_arr)

    I, J, data = sparse.find(G_arr)
    G = spmatrix(data, I, J, size=G_arr.shape)
    if verbosity < 1:
        solvers.options['show_progress'] = False
    else:
        solvers.options['show_progress'] = True

    # solvers.options['abstol'] = 1e-10
    # solvers.options['reltol'] = 1e-10
    # solvers.options['feastol'] = 1e-10

    sol = np.asarray(solvers.lp(c=c, G=G, h=h)['x'])[:nbasis]
    residuals = values-basis_matrix.dot(sol)[:, 0]
    coef = np.append(conditional_value_at_risk(residuals, alpha), sol)
    return coef
