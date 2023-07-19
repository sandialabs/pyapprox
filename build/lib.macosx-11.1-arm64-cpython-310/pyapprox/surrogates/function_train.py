import numpy as np
from functools import partial
import os
import copy
from multiprocessing import Pool
from scipy.optimize import least_squares

from pyapprox.surrogates.orthopoly.orthonormal_polynomials import \
    evaluate_orthonormal_polynomial_1d
from pyapprox.surrogates.orthopoly.orthonormal_recursions import \
    jacobi_recurrence


def core_params_per_function(core_params, core_params_map, ranks):
    """
    Copy the parameters of a core into a matrix. If somes functions
    have smaller number of params then make missing params zero.
    This speeds up evaluation of each univariate function within the core
    """
    params, nparams = [], []
    for kk in range(ranks[1]):
        for jj in range(ranks[0]):
            params.append(get_params_of_univariate_function(
                jj, kk, ranks, core_params, core_params_map))
            nparams.append(params[-1].shape[0])
    max_nparams = np.max(nparams)
    params_array = np.zeros((max_nparams, len(params)))
    for ii in range(params_array.shape[1]):
        params_array[:nparams[ii], ii] = params[ii]
    return params_array


def evaluate_core(samples, core_params, core_params_map, ranks,
                  recursion_coeffs):
    """
    Evaluate a core of the function train at a set of samples

    Parameters
    ----------
    samples : np.ndarray(1, nsamples)
        The sample at which to evaluate the function train

    univariate_params : [ np.ndarray (num_coeffs_i) ] (ranks[0]*ranks[2])
        The coeffs of each univariate function. May be of different size
        i.e. num_coeffs_i can be different for i=0,...,ranks[0]*ranks[1]

    ranks : np.ndarray (2)
        The ranks of the core [r_{k-1},r_k]

    recursion_coeffs : np.ndarray (max_degree+1)
        The recursion coefficients used to evaluate the univariate functions
        which are assumed to polynomials defined by the recursion coefficients

    Returns
    -------
    core_values : np.ndarray (nsamples, ranks[0],ranks[1])
        The values of each univariate function evaluated at each sample

    Notes
    -----
    If we assume each univariate function for variable ii is fixed
    we only need to compute basis matrix once. This is also true
    if we compute basis matrix for max degree of the univariate functions
    of the ii variable. If degree of a given univariate function is
    smaller we can just use subset of matrix. This comes at the cost of
    more storage but less computations than if vandermonde was computed
    for each different degree. We build max_degree vandermonde here.
    """
    assert samples.ndim == 2 and samples.shape[0] == 1
    if recursion_coeffs[0] is not None:
        max_degree = recursion_coeffs.shape[0]-1
        basis_matrix = evaluate_orthonormal_polynomial_1d(
            samples[0, :], max_degree, recursion_coeffs)
    else:
        from pyapprox.surrogates.interp.monomial import (
            univariate_monomial_basis_matrix)
        basis_matrix = univariate_monomial_basis_matrix(
            len(recursion_coeffs)-1, samples[0, :])
    params = core_params_per_function(core_params, core_params_map, ranks)
    core_values = basis_matrix.dot(params)
    core_values = core_values.reshape(
        (samples.shape[1], ranks[0], ranks[1]), order="F")
    # print("##", ranks)
    # print(core_params)
    # print(params)
    # print(core_values)
    return core_values


def evaluate_function_train(samples, ft_data, recursion_coeffs):
    """
    Evaluate the function train

    Parameters
    ----------
    samples : np.ndarray (num_vars, num_samples)
        The samples at which to evaluate the function train

    cores : [[[] univariate_params[ii][jj]] (ranks[ii]*ranks[ii+1]) ](num_vars)
        Parameters of the univariate function of each core.
        ii is the variable index, jj is the univariate function index within
        the ii-th core. jj=0,...,ranks[ii]*ranks[ii+1]. Univariate functions
        are assumed to be stored in column major ordering.

    ranks : np.ndarray (num_vars+1)
        The ranks of the function train cores

    recursion_coeffs : np.ndarray (max_degree+1)
        The recursion coefficients used to evaluate the univariate functions
        which are assumed to polynomials defined by the recursion coefficients

    Returns
    -------
    values : np.ndarray (num_samples,1)
        The values of the function train at the samples
    """
    ranks, ft_params, ft_params_map, ft_cores_map = ft_data
    core_params, core_params_map = get_all_univariate_params_of_core(
        ft_params, ft_params_map, ft_cores_map, 0)
    values = evaluate_core(
        samples[0:1, :], core_params, core_params_map, ranks[:2],
        recursion_coeffs)
    nvars = len(ranks)-1
    for dd in range(1, nvars):
        core_params, core_params_map = get_all_univariate_params_of_core(
            ft_params, ft_params_map, ft_cores_map, dd)
        core_values = evaluate_core(
            samples[dd:dd+1, :], core_params, core_params_map, ranks[dd:dd+2],
            recursion_coeffs)
        values = np.einsum("ijk, ikm->ijm", values, core_values)
    return values[:, :1, 0]


def evaluate_core_deprecated(sample, core_params, core_params_map, ranks,
                             recursion_coeffs):
    """
    Evaluate a core of the function train at a sample

    Parameters
    ----------
    sample : float
        The sample at which to evaluate the function train

    univariate_params : [ np.ndarray (num_coeffs_i) ] (ranks[0]*ranks[2])
        The coeffs of each univariate function. May be of different size
        i.e. num_coeffs_i can be different for i=0,...,ranks[0]*ranks[1]

    ranks : np.ndarray (2)
        The ranks of the core [r_{k-1},r_k]

    recursion_coeffs : np.ndarray (max_degree+1)
        The recursion coefficients used to evaluate the univariate functions
        which are assumed to polynomials defined by the recursion coefficients

    Returns
    -------
    core_values : np.ndarray (ranks[0],ranks[1])
        The values of each univariate function evaluated at the sample

    Notes
    -----
    If we assume each univariate function for variable ii is fixed
    we only need to compute basis matrix once. This is also true
    if we compute basis matrix for max degree of the univariate functions
    of the ii variable. If degree of a given univariate function is
    smaller we can just use subset of matrix. This comes at the cost of
    more storage but less computations than if vandermonde was computed
    for each different degree. We build max_degree vandermonde here.
    """
    try:
        from pyapprox.python.function_train import evaluate_core_pyx
        return evaluate_core_pyx(sample, core_params, core_params_map, ranks,
                                 recursion_coeffs)
        # from pyapprox.weave.function_train import c_evalute_core
        # return c_evaluate_core(sample, core_params, core_params_map, ranks,
        #                        recursion_coeffs)
    except:
        pass

    assert ranks.shape[0] == 2
    assert np.isscalar(sample)

    core_values = np.empty((ranks[0], ranks[1]), dtype=float)

    max_degree = recursion_coeffs.shape[0]-1
    basis_matrix = evaluate_orthonormal_polynomial_1d(
        np.asarray([sample]), max_degree, recursion_coeffs)
    for kk in range(ranks[1]):
        for jj in range(ranks[0]):
            params = get_params_of_univariate_function(
                jj, kk, ranks, core_params, core_params_map)
            degree = params.shape[0]-1
            assert degree < recursion_coeffs.shape[0]
            core_values[jj, kk] = np.dot(basis_matrix[:, :degree+1], params)
    return core_values


def core_grad_left(sample, core_params, core_params_map, ranks,
                   recursion_coeffs, left_vals):
    """
    Evaluate the value and intermediate derivaties, with respect to
    univariate function basis parameters, of a core of the function train at
    a sample.

    Parameters
    ----------
    sample : float
        The sample at which to evaluate the function train

    univariate_params : [ np.ndarray (num_coeffs_i) ] (ranks[0]*ranks[2])
        The params of each univariate function. May be of different size
        i.e. num_params_i can be different for i=0,...,ranks[0]*ranks[1]

    ranks : np.ndarray (2)
        The ranks of the core [r_{k-1},r_k]

    recursion_coeffs : np.ndarray (max_degree+1)
        The recursion coefficients used to evaluate the univariate functions
        which are assumed to polynomials defined by the recursion coefficients

    left_vals : np.ndarray (ranks[0])
        The values of the product of all previous cores F_1F_2...F_{k-1}.
        If None no derivatives will be computed. Setting None is useful if
        one only want function values or when computing derivatives of first
        core.

    Returns
    -------
    core_values : np.ndarray (ranks[0],ranks[1])
        The values of each univariate function evaluated at the sample

    derivs : [ [] (num_params_i) ] (ranks[0]*ranks[1])
        The derivates of the univariate function with respect to the
        basis parameters after the left pass algorithm.
        Derivs of univariate functions are in column major ordering

    Notes
    -----
    If we assume each univariate function for variable ii is fixed
    we only need to compute basis matrix once. This is also true
    if we compute basis matrix for max degree of the univariate functions
    of the ii variable. If degree of a given univariate function is
    smaller we can just use subset of matrix. This comes at the cost of
    more storage but less computations than if vandermonde was computed
    for each different degree. We build max_degree vandermonde here.
    """
    assert ranks.shape[0] == 2
    assert np.isscalar(sample)
    if left_vals is not None:
        assert left_vals.ndim == 2 and left_vals.shape[0] == 1

    core_values = np.empty((ranks[0]*ranks[1]), dtype=float)
    core_derivs = np.empty_like(core_params)

    max_degree = recursion_coeffs.shape[0]-1
    basis_matrix = evaluate_orthonormal_polynomial_1d(
        np.asarray([sample]), max_degree, recursion_coeffs)
    cnt = 0
    for kk in range(ranks[1]):
        for jj in range(ranks[0]):
            params = get_params_of_univariate_function(
                jj, kk, ranks, core_params, core_params_map)
            degree = params.shape[0]-1
            assert degree < recursion_coeffs.shape[0]
            univariate_function_num = get_univariate_function_number(
                ranks[0], jj, kk)
            core_values[univariate_function_num] = np.dot(
                basis_matrix[:, :degree+1], params)
            if left_vals is not None:
                core_derivs[cnt:cnt+params.shape[0]] = \
                    left_vals[0, jj]*basis_matrix[:, :degree+1]
            else:
                core_derivs[cnt:cnt+params.shape[0]
                            ] = basis_matrix[:, :degree+1]
            cnt += params.shape[0]
    return core_values, core_derivs


def evaluate_function_train_deprecated(samples, ft_data, recursion_coeffs):
    """
    Evaluate the function train

    Parameters
    ----------
    samples : np.ndarray (num_vars, num_samples)
        The samples at which to evaluate the function train

    cores :  [ [ [] univariate_params[ii][jj]] (ranks[ii]*ranks[ii+1]) ](num_vars)
        Parameters of the univariate function of each core.
        ii is the variable index, jj is the univariate function index within the
        ii-th core. jj=0,...,ranks[ii]*ranks[ii+1]. Univariate functions
        are assumed to be stored in column major ordering.

    ranks : np.ndarray (num_vars+1)
        The ranks of the function train cores

    recursion_coeffs : np.ndarray (max_degree+1)
        The recursion coefficients used to evaluate the univariate functions
        which are assumed to polynomials defined by the recursion coefficients

    Returns
    -------
    values : np.ndarray (num_samples,1)
        The values of the function train at the samples
    """
    ranks, ft_params, ft_params_map, ft_cores_map = ft_data

    num_vars = len(ranks)-1
    num_samples = samples.shape[1]
    assert len(ranks) == num_vars+1
    values = np.zeros((num_samples, 1), dtype=float)
    for ii in range(num_samples):
        core_params, core_params_map = get_all_univariate_params_of_core(
            ft_params, ft_params_map, ft_cores_map, 0)
        core_values = evaluate_core_deprecated(
            samples[0, ii], core_params, core_params_map, ranks[:2],
            recursion_coeffs)
        for dd in range(1, num_vars):
            core_params, core_params_map = get_all_univariate_params_of_core(
                ft_params, ft_params_map, ft_cores_map, dd)
            core_values_dd = evaluate_core_deprecated(
                samples[dd, ii], core_params, core_params_map, ranks[dd:dd+2],
                recursion_coeffs)
            core_values = np.dot(core_values, core_values_dd)
        values[ii, 0] = core_values[0, 0]
    return values


def core_grad_right(ranks, right_vals, intermediate_core_derivs,
                    core_params_map):
    """
    Evaluate the gradient of a core of the function train at a sample using
    precomputed intermediate_results

    This works because derivative of core (deriv_matrix) with respect to a
    single parameter of a single univariate function index by ii,jj is
    zero except for 1  non-zero entry at ii,jj.
    Thus theoretically we need to compute np.dot(deriv_matrix,right_vals) for
    each ii,jj we actually only need to collect all non-zero derivatives
    of each deriv_matrix into one vector and multiply this by right_vals

    Parameters
    ----------
    ranks : np.ndarray (2)
        The ranks of the core [r_{k-1},r_k]

    right_vals : np.ndarray (ranks[1])
        The values of the product of all following cores F_{k+1}F_{k+1}...F_d

    intermediate_core_derivs : [ [] (num_params_i) ] (ranks[0]*ranks[1])
        The derivates of the univariate function with respect to the
        basis parameters after the left pass algorithm.
        Derivs of univariate functions are in column major ordering

    Returns
    -------
    derivs : [ [] (num_params_i) ] (ranks[0]*ranks[1])
        The derivates of the univariate function with respect to the
        basis parameters. Derivs of univariate functions are in
        column major ordering
    """
    assert ranks.shape[0] == 2
    assert right_vals.ndim == 2 and right_vals.shape[1] == 1
    core_derivs = np.empty((0), dtype=float)
    for kk in range(ranks[1]):
        for jj in range(ranks[0]):
            lb, ub = get_index_bounds_of_univariate_function_params(
                jj, kk, ranks, core_params_map,
                intermediate_core_derivs.shape[0])
            core_derivs = np.append(
                core_derivs, intermediate_core_derivs[lb:ub]*right_vals[kk, 0])
    return core_derivs


def evaluate_function_train_grad(sample, ft_data, recursion_coeffs):
    """
    Evaluate the function train and its gradient

    Parameters
    ----------
    sample : np.ndarray (num_vars, )
        The sample1 at which to evaluate the function train gradient

    cores : [[[] univariate_params[ii][jj]] (ranks[ii]*ranks[ii+1]) ](num_vars)
        Parameters of the univariate function of each core.
        ii is the variable index, jj is the univariate function index within
        the ii-th core. jj=0,...,ranks[ii]*ranks[ii+1]. Univariate functions
        are assumed to be stored in column major ordering.

    ranks : np.ndarray (num_vars+1)
        The ranks of the function train cores

    recursion_coeffs : np.ndarray (max_degree+1)
        The recursion coefficients used to evaluate the univariate functions
        which are assumed to polynomials defined by the recursion coefficients

    Returns
    -------
    value : float
        The value of the function train at the sample

    grad : np.ndarray(num_ft_params)
        The derivative of the function train with respect to each coefficient
        of each univariate core.
        The gradient is stored for each core from first to last.
        The gradient of each core is stored using column major ordering of
        the univariate functions.
    """
    value, values_of_cores, derivs_of_cores = \
        evaluate_ft_gradient_forward_pass(sample, ft_data, recursion_coeffs)

    ranks, ft_params, ft_params_map, ft_cores_map = ft_data

    gradient = evaluate_ft_gradient_backward_pass(
        ranks, values_of_cores, derivs_of_cores, ft_params_map, ft_cores_map)

    return value, gradient


def evaluate_ft_gradient_forward_pass(sample, ft_data, recursion_coeffs):
    ranks, ft_params, ft_params_map, ft_cores_map = ft_data

    num_vars = len(ranks)-1
    assert sample.shape[1] == 1
    assert sample.shape[0] == num_vars

    core_params, core_params_map = get_all_univariate_params_of_core(
        ft_params, ft_params_map, ft_cores_map, 0)
    core_values, core_derivs = core_grad_left(
        sample[0, 0], core_params, core_params_map, ranks[:2],
        recursion_coeffs, None)
    left_vals = core_values.copy()[np.newaxis, :]
    values_of_cores = core_values.copy()
    derivs_of_cores = core_derivs.copy()
    for dd in range(1, num_vars):
        core_params, core_params_map = get_all_univariate_params_of_core(
            ft_params, ft_params_map, ft_cores_map, dd)
        core_values, core_derivs = core_grad_left(
            sample[dd, 0], core_params, core_params_map, ranks[dd:dd+2],
            recursion_coeffs, left_vals)
        left_vals = np.dot(left_vals, np.reshape(
            core_values, (ranks[dd], ranks[dd+1]), order='F'))
        values_of_cores = np.concatenate((values_of_cores, core_values))
        derivs_of_cores = np.concatenate((derivs_of_cores, core_derivs))
    value = left_vals[0]
    return value, values_of_cores, derivs_of_cores


def evaluate_ft_gradient_backward_pass(ranks, values_of_cores, derivs_of_cores,
                                       ft_params_map, ft_cores_map):
    num_vars = ranks.shape[0]-1
    num_ft_params = derivs_of_cores.shape[0]
    gradient = np.empty_like(derivs_of_cores)

    # gradient of parameters of last core
    cores_params_lb, cores_params_ub = get_index_bounds_of_core_params(
        num_vars-1, ft_cores_map, ft_params_map, num_ft_params)[:2]
    gradient[cores_params_lb:cores_params_ub] =\
        derivs_of_cores[cores_params_lb:cores_params_ub]

    values_lb, values_ub = get_index_bounds_of_core_univariate_functions(
        num_vars-1, ft_cores_map, values_of_cores.shape[0])
    right_vals = np.reshape(
        values_of_cores[values_lb:values_ub],
        (ranks[num_vars-1], ranks[num_vars]), order='F')

    # gradient of parameters of each of the middle cores
    for dd in range(num_vars-2, 0, -1):
        core_derivs, core_params_map = get_all_univariate_params_of_core(
            derivs_of_cores, ft_params_map, ft_cores_map, dd)
        core_values = get_core_values(dd, values_of_cores, ft_cores_map)

        core_params_lb, core_params_ub = get_index_bounds_of_core_params(
            dd, ft_cores_map, ft_params_map, num_ft_params)[:2]

        gradient[core_params_lb:core_params_ub] = core_grad_right(
            ranks[dd:dd+2], right_vals, core_derivs, core_params_map)

        core_values = np.reshape(
            core_values, (ranks[dd], ranks[dd+1]), order='F')
        right_vals = np.dot(core_values, right_vals)

    # gradient of parameters of first core
    core_derivs, core_params_map = get_all_univariate_params_of_core(
        derivs_of_cores, ft_params_map, ft_cores_map, 0)
    core_params_lb, core_params_ub = get_index_bounds_of_core_params(
        0, ft_cores_map, ft_params_map, num_ft_params)[:2]
    gradient[core_params_lb:core_params_ub] = core_grad_right(
        ranks[:2], right_vals, core_derivs, core_params_map)

    return gradient


def num_univariate_functions(ranks):
    """
    Compute the number of univariate function in a function train.

    Parameters
    ----------
    ranks : np.ndarray (num_vars+1)
        The ranks of the function train cores

    Returns
    -------
    num_1d_functions : integer
        The number of univariate functions
    """
    num_1d_functions = 0
    for ii in range(len(ranks)-1):
        num_1d_functions += ranks[ii]*ranks[ii+1]
    return int(num_1d_functions)


def generate_homogeneous_function_train(ranks, num_params_1d, ft_params):
    """
    Generate a function train of a specified rank using the same
    parameterization of each univariate function within a core and for all
    cores.

    Parameters
    ----------
    ranks : np.ndarray (num_vars+1)
        The ranks of the function train cores

    num_params_1d : integer
        The number of parameters of each univariate function, e.g.
        the number of parameters in a polynomial basis

    ft_params : np.ndarray (num_univariate_functions(ranks)*num_params)
        Flatten array containing the parameter values of each univariate
        function

    Returns
    -------
    cores : [[[] univariate_params[ii][jj]] (ranks[ii]*ranks[ii+1]) ](num_vars)
        Parameters of the univariate function of each core.
        ii is the variable index, jj is the univariate function index within
        the ii-th core. jj=0,...,ranks[ii]*ranks[ii+1]. Univariate functions
        are assumed to be stored in column major ordering.
    """
    num_vars = ranks.shape[0]-1
    num_1d_functions = num_univariate_functions(ranks)
    num_ft_parameters = num_params_1d*num_1d_functions
    ft_params_map = np.arange(0, num_ft_parameters, num_params_1d, dtype=int)
    assert ft_params.shape[0] == num_ft_parameters
    ft_cores_map = np.zeros((1), dtype=int)
    for ii in range(num_vars-1):
        ft_cores_map = np.append(
            ft_cores_map,
            np.asarray(ft_cores_map[-1]+ranks[ii]*ranks[ii+1], dtype=int))

    # return list so I can make assignments like
    # ft_data[1]=...
    return [ranks, ft_params, ft_params_map, ft_cores_map]


def get_index_bounds_of_core_univariate_functions(var_num, ft_cores_map,
                                                  num_univariate_functions):
    num_vars = ft_cores_map.shape[0]
    assert var_num < num_vars
    core_map_lb = ft_cores_map[var_num]
    if var_num == num_vars-1:
        core_map_ub = num_univariate_functions
    else:
        core_map_ub = ft_cores_map[var_num+1]
    return core_map_lb, core_map_ub


def get_index_bounds_of_core_params(var_num, ft_cores_map,
                                    ft_params_map, num_ft_params):
    num_vars = ft_cores_map.shape[0]
    assert var_num < num_vars
    core_map_lb, core_map_ub = get_index_bounds_of_core_univariate_functions(
        var_num, ft_cores_map, ft_params_map.shape[0])
    if var_num == num_vars-1:
        params_map_ub = num_ft_params
    else:
        params_map_ub = ft_params_map[core_map_ub]
    params_map_lb = ft_params_map[core_map_lb]
    return params_map_lb, params_map_ub, core_map_lb, core_map_ub


def get_all_univariate_params_of_core(ft_params, ft_params_map, ft_cores_map,
                                      var_num):
    params_map_lb, params_map_ub, core_map_lb, core_map_ub = \
        get_index_bounds_of_core_params(
            var_num, ft_cores_map, ft_params_map, ft_params.shape[0])

    core_params = ft_params[params_map_lb:params_map_ub]
    core_params_map = ft_params_map[core_map_lb:core_map_ub]-params_map_lb
    return core_params, core_params_map


def get_core_values(var_num, values_of_cores, ft_cores_map):
    num_ft_univariate_functions = values_of_cores.shape[0]
    lb, ub = get_index_bounds_of_core_univariate_functions(
        var_num, ft_cores_map, num_ft_univariate_functions)
    return values_of_cores[lb:ub]


def get_univariate_function_number(left_rank, ii, jj):
    # for some reason if left_rank is uint 64 and ii,jj are int
    # following operation returns a float64
    return int(jj*left_rank+ii)


def get_index_bounds_of_univariate_function_params(
        ii, jj, ranks, core_params_map, num_core_params):
    univariate_function_num = get_univariate_function_number(ranks[0], ii, jj)
    num_univariate_functions = core_params_map.shape[0]
    assert univariate_function_num < num_univariate_functions
    lb = core_params_map[univariate_function_num]
    if univariate_function_num == num_univariate_functions-1:
        ub = num_core_params
    else:
        ub = core_params_map[univariate_function_num+1]
    return lb, ub


def get_params_of_univariate_function(ii, jj, ranks, core_params,
                                      core_params_map):
    lb, ub = get_index_bounds_of_univariate_function_params(
        ii, jj, ranks, core_params_map, core_params.shape[0])
    univariate_function_params = core_params[lb:ub]
    return univariate_function_params


def add_core(univariate_functions_params, ft_params, ft_params_map,
             ft_cores_map):
    if ft_params is None:
        ft_params = np.empty((0), dtype=float)
        ft_params_map = np.empty((0), dtype=int)
        ft_cores_map = np.empty((0), dtype=int)

    params_cnt = ft_params.shape[0]
    params_map_cnt = ft_params_map.shape[0]
    ft_cores_map = np.append(ft_cores_map, params_map_cnt)

    num_univariate_functions = len(univariate_functions_params)
    ft_params = np.concatenate(
        (ft_params, np.concatenate(univariate_functions_params)))

    param_indices = np.empty((num_univariate_functions), dtype=int)
    param_indices[0] = params_cnt
    for ii in range(1, num_univariate_functions):
        params_cnt += univariate_functions_params[ii-1].shape[0]
        param_indices[ii] = params_cnt
    ft_params_map = np.concatenate((ft_params_map, param_indices))
    return ft_params, ft_params_map, ft_cores_map


def generate_additive_function_in_function_train_format(
        univariate_function_params, compress):
    """
    Generate function train representation of
    f(x) = f_1(x_1)+f_2(x_2)+...+f_d(x_d).
    An additive function in tensor train format has ranks [1,2,2,...,2,2,1]

    Parameters
    ----------
    univariate_function_params : [np.ndarray(num_params_1d_i)] (num_vars)
        The parameters of the univariate function f_i for each dimension

    compress : boolean
        True  - return compressed representation of zero and one cores,
                i.e. a coefficient vector for the univariate cores of length 1
        False - return a coefficient vector that has zeros for all 1d params
                of a univariate function. If False then the number of
                parameters of all univariate parameters must be equal
    Returns
    -------
    ranks : np.ndarray (num_vars+1)
        The ranks of the function train cores

    cores : [[[] univariate_params[ii][jj]] (ranks[ii]*ranks[ii+1]) ](num_vars)
        Parameters of the univariate function of each core.
        ii is the variable index, jj is the univariate function index within
        the ii-th core. jj=0,...,ranks[ii]*ranks[ii+1]. Univariate functions
        are assumed to be stored in column major ordering.
    """
    num_vars = len(univariate_function_params)
    ranks = ranks_vector(num_vars, 2)

    if not compress:
        num_params_1d = univariate_function_params[0].shape[0]
        for dd in range(1, num_vars):
            assert num_params_1d == univariate_function_params[dd].shape[0]
        zero = np.zeros((num_params_1d), dtype=float)
        one = zero.copy()
        one[0] = 1.
    else:
        zero = np.asarray([0.])
        one = np.asarray([1.])

    # first core is |f_1 1|
    core_univariate_functions_params = [univariate_function_params[0], one]
    ft_params, ft_params_map, ft_cores_map = add_core(
        core_univariate_functions_params, None, None, None)
    # middle cores are | 1    0|
    #                  |f_ii, 1|
    for ii in range(1, num_vars-1):
        core_univariate_functions_params = [
            one, univariate_function_params[ii], zero, one]
        ft_params, ft_params_map, ft_cores_map = add_core(
            core_univariate_functions_params, ft_params, ft_params_map,
            ft_cores_map)
    # last core is | 1 |
    #              |f_d|
    core_univariate_functions_params =\
        [one, univariate_function_params[num_vars-1]]
    ft_params, ft_params_map, ft_cores_map = add_core(
        core_univariate_functions_params, ft_params, ft_params_map,
        ft_cores_map)

    # return list so I can make assignments like
    # ft_data[1]=...
    return [ranks, ft_params, ft_params_map, ft_cores_map]


def ft_parameter_finite_difference_gradient(
        sample, ft_data, recursion_coeffs, eps=2*np.sqrt(np.finfo(float).eps)):
    ft_params = ft_data[1]
    value = evaluate_function_train(sample, ft_data, recursion_coeffs)
    num_ft_parameters = ft_params.shape[0]
    gradient = np.empty_like(ft_params)
    for ii in range(num_ft_parameters):
        perturbed_params = ft_params.copy()
        perturbed_params[ii] += eps
        ft_data = (ft_data[0], perturbed_params, ft_data[2], ft_data[3])
        perturbed_value = evaluate_function_train(
            sample, ft_data, recursion_coeffs)
        gradient[ii] = (perturbed_value-value)/eps
    return gradient


def ranks_vector(num_vars, rank):
    ranks = np.ones((num_vars+1), dtype=int)
    ranks[1:num_vars] = rank
    return ranks


def print_function_train(ft_data):
    """
    Print the parameters of the univariate functions of each core of a
    function train.
    """
    ranks, ft_params, ft_params_map, ft_cores_map = ft_data
    num_vars = len(ranks)-1
    print(('Function train d=%d' % num_vars, "rank:", ranks))
    for ii in range(num_vars):
        core_params, core_params_map = get_all_univariate_params_of_core(
            ft_params, ft_params_map, ft_cores_map, ii)
        for kk in range(ranks[ii+1]):
            for jj in range(ranks[ii]):
                params = get_params_of_univariate_function(
                    jj, kk, ranks, core_params, core_params_map)
                print(("(%d,%d,%d)" % (ii, jj, kk), params))


def function_trains_equal(ft_data_1, ft_data_2, verbose=False):
    """
    Check equality of two function trains
    """
    ranks_1, ft_params_1, ft_params_map_1, ft_cores_map_1 = ft_data_1
    ranks_2, ft_params_2, ft_params_map_2, ft_cores_map_2 = ft_data_2
    num_vars = len(ranks_1)-1
    if (num_vars != len(ranks_2)-1):
        if verbose:
            print("Inconsistent number of variables")
        return False

    if not np.allclose(ranks_1, ranks_2):
        if verbose:
            print("Inconsistent ranks")
        return False

    if not np.allclose(ft_params_1, ft_params_2):
        if verbose:
            print("Inconsistent univariate function parameters")
        return False

    if not np.allclose(ft_params_map_1, ft_params_map_2):
        if verbose:
            print("Inconsistent univariate functions parameters map")
        return False

    if not np.allclose(ft_cores_map_1, ft_cores_map_2):
        if verbose:
            print("Inconsistent cores map")
        return False

    return True


def ft_linear_least_squares_regression(samples, values, degree, perturb=None):
    num_vars, num_samples = samples.shape

    vandermonde = np.hstack((np.ones((num_samples, 1)), samples.copy().T))
    sol = np.linalg.lstsq(vandermonde, values, rcond=None)[0]
    univariate_function_params = []
    for ii in range(num_vars):
        params = np.zeros(degree+1)
        # add small constant contribution to each core
        params[0] = sol[0]/float(num_vars)
        params[1] = sol[ii+1]
        univariate_function_params.append(params)

    linear_ft_data = generate_additive_function_in_function_train_format(
        univariate_function_params, compress=False)
    if perturb is not None:
        II = np.where(linear_ft_data[1] == 0)[0]
        linear_ft_data[1][II] = perturb
    return linear_ft_data


def modify_and_evaluate_function_train(samples, ft_data, recursion_coeffs,
                                       active_indices, ft_params):
    if active_indices is not None:
        ft_data[1][active_indices] = ft_params
    else:
        ft_data[1] = ft_params

    ft_values = evaluate_function_train(samples, ft_data, recursion_coeffs)
    return ft_values


def ft_least_squares_residual(samples, values, ft_data, recursion_coeffs,
                              active_indices, ft_params):
    """
    Warning this only overwrites parameters associated with the active indices
    the rest of the parameters are taken from ft_data.
    """
    ft_values = modify_and_evaluate_function_train(
        samples, ft_data, recursion_coeffs, active_indices, ft_params)
    assert ft_values.shape == values.shape
    return (values-ft_values)[:, 0]


def ft_least_squares_jacobian(samples, values, ft_data, recursion_coeffs,
                              active_indices, ft_params):
    """
    Warning this only overwrites parameters associated with the active indices
    the rest of the parameters are taken from ft_data.
    """
    num_samples = samples.shape[1]
    num_params = ft_params.shape[0]
    if active_indices is not None:
        ft_data[1][active_indices] = ft_params
        nactive_params = active_indices.shape[0]
    else:
        ft_data[1] = ft_params
        nactive_params = num_params

    jacobian = np.empty((num_samples, nactive_params), dtype=float)
    for ii in range(num_samples):
        __, grad = evaluate_function_train_grad(
            samples[:, ii:ii+1], ft_data, recursion_coeffs)
        if active_indices is None:
            jacobian[ii, :] = -grad
        jacobian[ii, :] = -grad[active_indices]
    return jacobian


def apply_function_train_adjoint_jacobian(samples, ft_data, recursion_coeffs,
                                          perturb, ft_params, vec):
    """
    Apply the function train Jacobian to a vector. This is more memory
    efficient than simply computing Jacobain and then using a matrix-vector
    multiply to compute its action.

    Parameters
    ----------
    perturb : float
        Small positive perturbation to add to any zero valued function train
        parameters before computing Jacobian. This is useful when computing
        Jacobians for greedy regression methods that start with only a small
        number of non-zero paramters which can mean that gradient can often
        be zero unless perturbation is applied.
    """
    new_ft_data = copy.deepcopy(ft_data)
    new_ft_data[1] = ft_params.copy()
    new_ft_data[1][ft_params == 0] = perturb
    result = np.zeros((ft_params.shape[0]), dtype=float)
    num_samples = samples.shape[1]
    for ii in range(num_samples):
        __, ft_gradient = evaluate_function_train_grad(
            samples[:, ii:ii+1], new_ft_data, recursion_coeffs)
        for jj in range(ft_gradient.shape[0]):
            result[jj] += -ft_gradient[jj]*vec[ii]
        # print 'result',result
    return result


def ft_non_linear_least_squares_regression(samples, values, ft_data,
                                           recursion_coeffs,
                                           initial_guess,
                                           active_indices=None,
                                           opts=dict()):

    if active_indices is not None:
        assert active_indices.shape[0] == initial_guess.shape[0]

    assert values.ndim == 2 and values.shape[1] == 1
    residual_func = partial(
        ft_least_squares_residual, samples, values, ft_data, recursion_coeffs,
        active_indices)
    jacobian_func = partial(ft_least_squares_jacobian,
                            samples, values, ft_data, recursion_coeffs,
                            active_indices)
    # jacobian_func='2-point'

    result = least_squares(
        residual_func, initial_guess, jac=jacobian_func,
        gtol=opts.get('gtol', 1e-8), ftol=opts.get('ftol', 1e-8),
        xtol=opts.get('xtol', 1e-8))
    if opts.get('verbosity', 0) > 0:
        print('non-linear least squares output')
        print(('#fevals:', result['nfev']))
        print(('#gevals:', result['njev']))
        print(('obj:', result['cost']))
        print(('status:', result['status']))

    if active_indices is None:
        return result['x']
    else:
        ft_params = ft_data[1].copy()
        ft_params[active_indices] = result['x']
        return ft_params


def generate_random_sparse_function_train(num_vars, rank, num_params_1d,
                                          sparsity_ratio):
    ranks = rank*np.ones(num_vars+1, dtype=int)
    ranks[0] = 1
    ranks[-1] = 1
    num_1d_functions = num_univariate_functions(ranks)
    num_ft_params = num_params_1d*num_1d_functions

    sparsity = int(sparsity_ratio*num_ft_params)
    ft_params = np.random.normal(0., 1., (num_ft_params))
    II = np.random.permutation(num_ft_params)[sparsity:]
    ft_params[II] = 0.

    ft_data = generate_homogeneous_function_train(
        ranks, num_params_1d, ft_params)
    return ft_data


def get_random_compressible_vector(num_entries, decay_rate):
    vec = np.random.normal(0., 1., (num_entries))
    vec /= (np.arange(1, num_entries+1))**decay_rate
    vec = vec[np.random.permutation(num_entries)]
    return vec


def compare_compressible_vectors():
    import matplotlib.pyplot as plt
    decay_rates = [1, 2, 3, 4]
    num_entries = 1000
    II = np.arange(num_entries)
    for decay_rate in decay_rates:
        vec = get_random_compressible_vector(num_entries, decay_rate)
        plt.loglog(II, np.sort(np.absolute(vec))[
                   ::-1], 'o', label=r'$r=%1.1f$' % decay_rate)
    plt.legend()
    plt.show()


def sparsity_example(decay_rate, rank, num_vars, num_params_1d):
    num_trials = 20
    sparsity_fractions = np.arange(1, 10, dtype=float)/10.
    assert sparsity_fractions[-1] != 1  # error will always be zero

    ranks = ranks_vector(num_vars, rank)
    num_1d_functions = num_univariate_functions(ranks)
    num_ft_parameters = num_params_1d*num_1d_functions

    alpha = 0
    beta = 0
    recursion_coeffs = jacobi_recurrence(
        num_params_1d, alpha=alpha, beta=beta, probability=True)

    assert int(os.environ['OMP_NUM_THREADS']) == 1
    max_eval_concurrency = 4  # max(multiprocessing.cpu_count()-2,1)
    pool = Pool(max_eval_concurrency)
    partial_func = partial(
        sparsity_example_engine, ranks=ranks,
        num_ft_parameters=num_ft_parameters,
        decay_rate=decay_rate, recursion_coeffs=recursion_coeffs,
        sparsity_fractions=sparsity_fractions)

    filename = 'function-train-sparsity-effect-%d-%d-%d-%1.1f-%d.npz' % (
        num_vars, rank, num_params_1d, decay_rate, num_trials)
    if not os.path.exists(filename):
        result = pool.map(partial_func, [(num_params_1d)]*num_trials)
        l2_errors = np.asarray(result).T  # (num_sparsity_fractions,num_trials)
        np.savez(
            filename, l2_errors=l2_errors,
            sparsity_fractions=sparsity_fractions,
            num_ft_parameters=num_ft_parameters)
    return filename


def compress_homogeneous_function_train(ft_data, tol=1e-15):
    """
    Compress a function train such that only parameters with magnitude
    greater than a tolerance are retained.

    If all parameters of a univariate function are below tolerance
    Then a single zero is used as the parameters of that function.

    Note this function will not reduce rank, even if after compression
    the effective rank is smaller. E.g. we can have a column of all zeros
    functions.
    """
    ranks = ft_data[0]
    ft_params = ft_data[1]
    num_funcs_1d = num_univariate_functions(ranks)
    num_params_1d = ft_params.shape[0]//num_funcs_1d
    num_vars = ranks.shape[0]-1

    compressed_ft_params = None
    compressed_ft_params_map = None
    compressed_ft_cores_map = None

    cnt = 0
    for ii in range(num_vars):
        core_univariate_functions_params = []
        for jj in range(ranks[ii]):
            for kk in range(ranks[ii+1]):
                univariate_params = \
                    ft_params[cnt*num_params_1d:(cnt+1)*num_params_1d]
                II = np.where(np.absolute(univariate_params) > tol)[0]
                if II.shape[0] == 0:
                    compressed_univariate_params = np.array([0])
                else:
                    compressed_univariate_params = univariate_params[II]
                core_univariate_functions_params.append(
                    compressed_univariate_params)
                cnt += 1

        compressed_ft_params, compressed_ft_params_map, \
            compressed_ft_cores_map = add_core(
                core_univariate_functions_params, compressed_ft_params,
                compressed_ft_params_map, compressed_ft_cores_map)

    return [ranks, compressed_ft_params, compressed_ft_params_map,
            compressed_ft_cores_map]


def plot_sparsity_example():
    import matplotlib.pyplot as plt
    num_vars = 10
    num_params_1d = 10
    rank = 2
    decay_rates = [1, 2, 3, 4]
    for decay_rate in decay_rates:
        filename = sparsity_example(decay_rate, rank, num_vars, num_params_1d)
        file_data = np.load(filename)
        l2_errors = file_data['l2_errors']
        sparsity_fractions = file_data['sparsity_fractions']

        plt.semilogy(sparsity_fractions, np.median(l2_errors, axis=1))
        plt.fill_between(
            sparsity_fractions, l2_errors.min(axis=1), l2_errors.max(axis=1),
            alpha=0.5, label='$r=%1.1f$' % decay_rate)

    plt.legend()
    figname = 'function-train-sparsity-effect-%d-%d-%d.png' % (
        num_vars, rank, num_params_1d)
    plt.savefig(figname, dpi=600)
    # plt.show()


def sparsity_example_engine(num_params_1d, ranks, num_ft_parameters,
                            decay_rate, recursion_coeffs, sparsity_fractions):
    np.random.seed()
    num_vars = len(ranks)-1
    ft_params = get_random_compressible_vector(
        num_ft_parameters, decay_rate)*100
    ft_data = generate_homogeneous_function_train(
        ranks, num_params_1d, ft_params)

    num_samples = 1000
    samples = np.random.uniform(-1., 1., (num_vars, num_samples))
    values = evaluate_function_train(samples, ft_data, recursion_coeffs)

    # sort parameters in descending order
    sorted_indices = np.argsort(np.absolute(ft_params))[::-1]
    l2_errors = np.empty(len(sparsity_fractions), dtype=float)
    for jj in range(len(sparsity_fractions)):
        sparsity = int(num_ft_parameters*sparsity_fractions[jj])
        # retain only the s largest parameters
        assert sparsity <= num_ft_parameters
        sparse_params = ft_params.copy()
        sparse_params[sorted_indices[sparsity:]] = 0.
        sparse_ft_data = generate_homogeneous_function_train(
            ranks, num_params_1d, sparse_params)
        sparse_values = evaluate_function_train(
            samples, sparse_ft_data, recursion_coeffs)

        l2_error = np.linalg.norm(sparse_values-values)/np.linalg.norm(values)
        print(('num_ft_parameters', num_ft_parameters, 'sparsity', sparsity))
        print(('l2 error', l2_error))
        l2_errors[jj] = l2_error
    return l2_errors
