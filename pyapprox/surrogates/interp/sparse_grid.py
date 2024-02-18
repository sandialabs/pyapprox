import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.utilities import (
    cartesian_product, outer_product, hash_array
)
from pyapprox.surrogates.interp.indexing import (
    nchoosek, compute_hyperbolic_level_indices,
    argsort_indices_lexiographically_by_row
)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d,
    multivariate_barycentric_lagrange_interpolation,
    multivariate_hierarchical_barycentric_lagrange_interpolation
)
from pyapprox.surrogates.interp.tensorprod import (
    tensor_product_piecewise_polynomial_interpolation_with_values
)


def get_1d_samples_weights(quad_rules, growth_rules,
                           levels, config_variables_idx=None,
                           unique_rule_indices=None):
    levels = np.asarray(levels)
    num_vars = levels.shape[0]
    # [x]*n syntax, what you get is a list of n many x objects,
    # but they're all references to the same object, e.g.
    # samples_1d,weights_1d=[[]]*num_vars,[[]]*num_vars
    # instead use
    samples_1d = [[] for i in range(num_vars)]
    weights_1d = [[] for i in range(num_vars)]
    return update_1d_samples_weights(
        quad_rules, growth_rules, levels,
        samples_1d, weights_1d, config_variables_idx, unique_rule_indices)


def update_1d_samples_weights_economical(
        quad_rules, growth_rules, levels, samples_1d, weights_1d,
        config_variables_idx, unique_rule_indices):
    """
    Sometimes it is computationally time consuming to construct quadrature
    rules for each dimension, e.g. for numerically generated Leja rules.
    So only update unique rules and store result in all variable dimensions
    which use that rule.

    This function will update samples_1d of all variables
    with the same quad rule even if levels of some of those variables
    do not need to be updated (but it will only compute the rule
    once). This is in contrast to default update
    which computes the rule of every variable if and only if
    the level for that variable is insufficient ie needs to be updated.
    The former is economical in computaiontal cost the later in memory.

    TODO: ideally this function should only store samples_1d for the unique
    quadrature rules. But currently sparse grid assumes that there is a
    quadrature rule for each variable.
    """
    assert len(quad_rules) == len(growth_rules)
    assert len(quad_rules) == len(unique_rule_indices)
    cnt = 0
    levels = np.asarray(levels)
    for dd in range(len(unique_rule_indices)):
        unique_rule_indices[dd] = np.asarray(unique_rule_indices[dd])
        cnt += unique_rule_indices[dd].shape[0]

    num_vars = levels.shape[0]
    if config_variables_idx is None:
        config_variables_idx = num_vars

    if cnt != config_variables_idx:
        msg = 'unique_rule_indices inconsistent with num_random_vars '
        msg += '(config_variable_idx)'
        raise Exception(msg)

    from inspect import signature
    for dd in range(len(unique_rule_indices)):
        # use first instance of quad_rule
        # assumes samples_1d stored for every dimension not just for unique
        # quadrature rules
        index = unique_rule_indices[dd][0]
        max_level_dd = levels[unique_rule_indices[dd]].max()
        current_level = len(samples_1d[index])
        if current_level <= max_level_dd:
            sig = signature(quad_rules[dd])
            keyword_args = [p.name for p in sig.parameters.values()
                            if ((p.kind == p.POSITIONAL_OR_KEYWORD) or
                                (p.kind == p.KEYWORD_ONLY))]
            if current_level > 0 and 'initial_points' in keyword_args:
                # useful for updating Leja rules
                x, w = quad_rules[dd](
                    max_level_dd,
                    initial_points=samples_1d[index][-1][np.newaxis, :])
            else:
                x, w = quad_rules[dd](max_level_dd)
            assert x.ndim == 1 and len(w) == max_level_dd+1
            for ll in range(current_level, max_level_dd+1):
                for kk in unique_rule_indices[dd]:
                    # use weights ordered according to polynomial index ordering
                    # not typical ascending order
                    # Check if user specifies growth rule which is incompatible
                    # with quad rule.
                    assert w[ll].shape[0] == growth_rules[dd](ll)
                    weights_1d[kk].append(w[ll][:growth_rules[dd](ll)])
                    # following assumes nestedness of x
                    samples_1d[kk].append(x[:growth_rules[dd](ll)])
    return samples_1d, weights_1d


def update_1d_samples_weights(quad_rules, growth_rules,
                              levels, samples_1d, weights_1d,
                              config_variables_idx,
                              unique_rule_indices=None):
    if unique_rule_indices is not None:
        return update_1d_samples_weights_economical(
            quad_rules, growth_rules, levels, samples_1d, weights_1d,
            config_variables_idx, unique_rule_indices)

    num_vars = len(samples_1d)

    for dd in range(num_vars):
        current_level = len(samples_1d[dd])
        if current_level <= levels[dd]:
            x, w = quad_rules[dd](levels[dd])
            assert x.ndim == 1 and len(w) == levels[dd]+1
            for ll in range(current_level, levels[dd]+1):
                # Check if user specifies growth rule which is incompatible
                # with quad rule.
                assert w[ll].shape[0] == growth_rules[dd](ll)
                # use weights ordered according to polynomial index ordering
                # not typical ascending order
                weights_1d[dd].append(w[ll])
                # following assumes nestedness of x
                samples_1d[dd].append(x[:growth_rules[dd](ll)])
    return samples_1d, weights_1d


def get_hierarchical_sample_indices(subspace_index, poly_indices,
                                    samples_1d, config_variables_idx):
    """
    This function is useful for obtaining the hierarhical function values of
    a subspace

    Use this function in the following way

    hier_indices = get_hierarchical_sample_indices()
    samples = get_subspace_samples(unique_samples_only=False)
    hier_samples = samples[:,hier_indices]
    """
    num_vars, num_indices = poly_indices.shape
    if config_variables_idx is None:
        config_variables_idx = num_vars

    assert len(samples_1d) == config_variables_idx

    active_vars = np.where(subspace_index > 0)[0]
    hier_indices = np.empty((num_indices), dtype=int)
    kk = 0
    for ii in range(num_indices):
        index = poly_indices[:, ii]
        found = True
        for jj in range(len(active_vars)):
            subspace_level = subspace_index[active_vars[jj]]
            if active_vars[jj] < config_variables_idx:
                if subspace_level > 0:
                    idx = samples_1d[active_vars[jj]
                                     ][subspace_level-1].shape[0]
                else:
                    idx = 0
            else:
                idx = 0
            if index[active_vars[jj]] < idx:
                found = False
                break
        if found:
            hier_indices[kk] = ii
            kk += 1
    return hier_indices[:kk]


def get_subspace_samples(subspace_index, poly_indices, samples_1d,
                         config_variables_idx=None, unique_samples_only=False):
    """
    Compute the samples of a subspace.

    Parameters
    ----------
    unique_samples_only : boolean
        If true only return the samples that exist in this subspace
        and not in any ancestor subspace, i.e. the heirachical samples
    """
    assert len(samples_1d) == poly_indices.shape[0]
    subspace_samples = get_sparse_grid_samples(
        poly_indices, samples_1d, config_variables_idx)
    if unique_samples_only:
        I = get_hierarchical_sample_indices(
            subspace_index, poly_indices, samples_1d,
            config_variables_idx)
        subspace_samples = subspace_samples[:, I]
    return subspace_samples


def get_subspace_polynomial_indices(subspace_index, growth_rule_1d,
                                    config_variables_idx=None):
    """
    Get the polynomial indices of a tensor-product nodal subspace.

    Parameters
    ----------
    subspace index : np.ndarray (num_vars)
        The subspace index [l_1,...,l_d]

    growth_rule_1d : list of callable functions
        Function which takes a level l_i as its only argument and returns
        the number of samples in the 1D quadrature rule of the specified level.

    Return
    ------
    poly_indices : np.ndarray (num_vars x num_subspace_samples)
        The polynomial indices of the tensor-product subspace.
    """
    subspace_index = np.asarray(subspace_index)
    num_vars = subspace_index.shape[0]
    if np.all(subspace_index == 0):
        return np.zeros((num_vars, 1), dtype=int)

    if config_variables_idx is None:
        config_variables_idx = num_vars
    assert len(growth_rule_1d) == config_variables_idx

    poly_indices_1d = []
    for ii in range(num_vars):
        if ii < config_variables_idx:
            poly_indices_1d.append(
                np.arange(growth_rule_1d[ii](subspace_index[ii])))
        else:
            # for config variables just set value equal to subspace index value
            poly_indices_1d.append(np.asarray([subspace_index[ii]]))

    poly_indices = cartesian_product(poly_indices_1d, 1)
    return poly_indices


def get_subspace_weights(subspace_index, weights_1d, config_variables_idx=None):
    """
    Get the quadrature weights of a tensor-product nodal subspace.

    Parameters
    ----------
    subspace index : np.ndarray (num_vars)
        The subspace index [l_1,...,l_d]

    weights_1d : [[np.ndarray]*num_vars]
        List of quadrature weights for each level and each variable
        Each element of inner list is np.ndarray with ndim=1. which meaans only
        homogenous sparse grids are supported, i.e grids with same quadrature
        rule used in each dimension (level can be different per dimension
        though).

    Return
    ------
    subspace_weights : np.ndarray (num_subspace_samples)
        The quadrature weights of the tensor-product quadrature rule of the
        subspace.
    """
    assert subspace_index.ndim == 1
    num_vars = subspace_index.shape[0]
    if config_variables_idx is None:
        config_variables_idx = num_vars
    assert len(weights_1d) == config_variables_idx

    subspace_weights_1d = []
    constant_term = 1.
    I = np.where(subspace_index[:config_variables_idx] > 0)[0]
    subspace_weights_1d = [weights_1d[ii][subspace_index[ii]] for ii in I]

    # for all cases I have tested so far the quadrature rules weights
    # are always 1 for level 0. Using loop below takes twice as long as
    # above pythonic loop without error checks.

    # for dd in range(config_variables_idx):
    #     # integrate only over random variables. i.e. do not compute
    #     # tensor product over config variables.

    #     # only compute outer product over variables with non-zero index
    #     if subspace_index[dd]>0:
    #         # assumes level zero weight is constant
    #         subspace_weights_1d.append(weights_1d[dd][subspace_index[dd]])
    #     else:
    #         assert len(weights_1d[dd][subspace_index[dd]])==1
    #         constant_term *= weights_1d[dd][subspace_index[dd]][0]
    if len(subspace_weights_1d) > 0:
        subspace_weights = outer_product(subspace_weights_1d)*constant_term
    else:
        subspace_weights = np.ones(1)*constant_term
    return subspace_weights


def get_sparse_grid_samples(poly_indices, samples_1d, config_variables_idx=None):
    """
    Compute the unique sparse grid samples from a set of polynomial indices.

    The function assumes the sparse grid is isotropic, i.e the same level
    is used for each variable. This function can also only be used
    for nested quadrature rules.

    Parameters
    ----------
    poly_indices : np.ndarray (num_vars x num_sparse_grid_samples)
        The unique polynomial indices of the sparse grid.

    level : integer
        The level of the isotropic sparse grid.

    samples_1d : np.ndarray (num_poly_indices)
        samples of the univariate quadrature for maximum level in grid

    Return
    ------
    samples : np.ndarray (num_vars x num_sparse_grid_samples)
        The unique samples of the sparse grid.
    """
    # assume samples list for each variable has same length
    samples_1d = [samples_1d[dd][-1] for dd in range(len(samples_1d))]
    poly_indices_max = poly_indices.max(axis=1)
    for dd in range(len(samples_1d)):
        assert samples_1d[dd].shape[0] >= poly_indices_max[dd]
    num_vars, num_indices = poly_indices.shape
    if config_variables_idx is not None:
        assert num_vars > config_variables_idx
    samples = np.empty((num_vars, num_indices))
    for ii in range(num_indices):
        index = poly_indices[:, ii]
        for jj in range(num_vars):
            if config_variables_idx is None or jj < config_variables_idx:
                samples[jj, ii] = samples_1d[jj][index[jj]]
            else:
                samples[jj, ii] = index[jj]
    return samples


def get_smolyak_coefficients(subspace_indices):
    """
    Given an arbitrary set of downward close indices determine the
    smolyak coefficients.
    """
    num_vars, num_subspace_indices = subspace_indices.shape
    I = argsort_indices_lexiographically_by_row(subspace_indices)
    sorted_subspace_indices = subspace_indices[:, I]
    levels, level_change_indices = np.unique(
        sorted_subspace_indices[0, :], return_index=True)
    level_change_indices = np.append(
        level_change_indices[2:], [num_subspace_indices, num_subspace_indices])

    try:
        from pyapprox.cython.sparse_grid import get_smolyak_coefficients_pyx
        return get_smolyak_coefficients_pyx(
            sorted_subspace_indices, levels, level_change_indices)[I.argsort()]
    except:
        print('get_smolyak_coefficients extention failed')

    idx = 0
    smolyak_coeffs = np.zeros((num_subspace_indices), dtype=float)
    for ii in range(num_subspace_indices):
        index = sorted_subspace_indices[:, ii]
        if idx < levels.shape[0] and index[0] > levels[idx]:
            idx += 1
        for jj in range(ii, level_change_indices[idx]):
            diff = sorted_subspace_indices[:, jj]-index
            if diff.max() <= 1 and diff.min() >= 0:
                smolyak_coeffs[ii] += (-1.)**diff.sum()
    return smolyak_coeffs[I.argsort()]

    # try:
    #     from pyapprox.cython.sparse_grid import \
    #       get_smolyak_coefficients_without_sorting_pyx
    #     return get_smolyak_coefficients_without_sorting_pyx(subspace_indices)
    # except:
    #     print ('get_smolyak_coefficients_without_sorting extention failed')

    # num_vars, num_subspace_indices = subspace_indices.shape
    # smolyak_coeffs = np.zeros((num_subspace_indices),dtype=float)
    # for ii in range(num_subspace_indices):
    #     for jj in range(num_subspace_indices):
    #         diff = subspace_indices[:,jj]-subspace_indices[:,ii]
    #         if diff.max()<=1 and diff.min()>=0:
    #             smolyak_coeffs[ii]+=(-1.)**diff.sum()
    # return smolyak_coeffs


def get_isotropic_sparse_grid_subspace_indices(num_vars, level):
    smolyak_coefficients = np.empty((0), dtype=float)
    sparse_grid_subspace_indices = np.empty((num_vars, 0), dtype=int)
    for dd in range(min(num_vars, level+1)):
        subspace_indices_dd = compute_hyperbolic_level_indices(
            num_vars, level-dd, 1.0)
        sparse_grid_subspace_indices = np.hstack(
            (sparse_grid_subspace_indices, subspace_indices_dd))
        subspace_coefficient = (-1.0)**(dd)*nchoosek(num_vars-1, dd)
        smolyak_coefficients = np.hstack((
            smolyak_coefficients,
            subspace_coefficient*np.ones(subspace_indices_dd.shape[1])))
    return sparse_grid_subspace_indices, smolyak_coefficients


def get_sparse_grid_samples_and_weights(num_vars, level,
                                        quad_rules,
                                        growth_rules,
                                        sparse_grid_subspace_indices=None):
    """
    Compute the quadrature weights and samples of a isotropic sparse grid.

    Parameters
    ----------
    num_vars : integer
        The number of variables in (dimension of) the sparse grid

    level : integer
        The level of the isotropic sparse grid.

    quad_rules : callable univariate quadrature_rule(ll) or list
        Function used to compute univariate quadrature samples and weights
        for a given level (ll). The weights and samples must be returned
        with polynomial ordering. If list then argument is list of quadrature
        rules

    growth_rules :callable growth_rules(ll) or list
        Function that returns the number of samples in the univariate
        quadrature rule of a given level (ll). If list then argument if list
        of growth rules.

    Return
    ------
    samples : np.ndarray (num_vars x num_sparse_grid_samples)
        The unique samples of the sparse grid.

    weights : np.ndarray (num_sparse_grid_samples)
        The quadrature weights of the sparse grid.
    """
    #subspace_indices = []
    #subspace_coefficients = []

    if callable(quad_rules):
        quad_rules = [quad_rules]*num_vars
        growth_rules = [growth_rules]*num_vars

    assert len(quad_rules) == len(growth_rules)
    assert len(quad_rules) == num_vars

    samples_1d, weights_1d = get_1d_samples_weights(
        quad_rules, growth_rules, [level]*num_vars)

    poly_indices_dict = dict()
    num_sparse_grid_samples = 0
    weights = []
    poly_indices = []
    sparse_grid_subspace_poly_indices_list = []
    sparse_grid_subspace_values_indices_list = []

    if sparse_grid_subspace_indices is None:
        sparse_grid_subspace_indices, smolyak_coefficients =\
            get_isotropic_sparse_grid_subspace_indices(num_vars, level)
    else:
        smolyak_coefficients = get_smolyak_coefficients(
            sparse_grid_subspace_indices)
        II = np.where(np.absolute(smolyak_coefficients) > 1e-8)[0]
        smolyak_coefficients = smolyak_coefficients[II]
        sparse_grid_subspace_indices = sparse_grid_subspace_indices[:, II]

    for ii in range(sparse_grid_subspace_indices.shape[1]):
        subspace_index = sparse_grid_subspace_indices[:, ii]
        subspace_poly_indices = get_subspace_polynomial_indices(
            subspace_index, growth_rules)
        sparse_grid_subspace_poly_indices_list.append(subspace_poly_indices)
        subspace_weights = get_subspace_weights(
            subspace_index, weights_1d)*smolyak_coefficients[ii]
        assert subspace_weights.shape[0] == subspace_poly_indices.shape[1]
        subspace_values_indices = np.empty(
            (subspace_poly_indices.shape[1]), dtype=int)
        for jj in range(subspace_poly_indices.shape[1]):
            poly_index = subspace_poly_indices[:, jj]
            key = hash_array(poly_index)
            if key in poly_indices_dict:
                weights[poly_indices_dict[key]] += subspace_weights[jj]
                subspace_values_indices[jj] = poly_indices_dict[key]
            else:
                poly_indices.append(poly_index)
                poly_indices_dict[key] = num_sparse_grid_samples
                weights.append(subspace_weights[jj])
                subspace_values_indices[jj] = num_sparse_grid_samples
                num_sparse_grid_samples += 1
        sparse_grid_subspace_values_indices_list.append(
            subspace_values_indices)

    # get list of unique polynomial indices
    poly_indices = np.asarray(poly_indices).T
    samples = get_sparse_grid_samples(poly_indices, samples_1d)
    data_structures = [
        poly_indices_dict, poly_indices,
        sparse_grid_subspace_indices, np.asarray(smolyak_coefficients),
        sparse_grid_subspace_poly_indices_list, samples_1d, weights_1d,
        sparse_grid_subspace_values_indices_list]
    # subspace_poly_indices can be recomputed but return here to save
    # computations at the expense of more memory
    return samples, np.asarray(weights), data_structures


def get_subspace_values(values, subspace_values_indices):
    subspace_values = values[subspace_values_indices, :]
    return subspace_values


def get_subspace_values_using_dictionary(values, subspace_poly_indices,
                                         poly_indices_dict):
    num_qoi = values.shape[1]
    num_subspace_samples = subspace_poly_indices.shape[1]
    subspace_values = np.empty((num_subspace_samples, num_qoi), dtype=float)
    for jj in range(num_subspace_samples):
        poly_index = subspace_poly_indices[:, jj]
        # could reduce number of hash based lookups by simply storing
        # replicate of values for each subspace, to reduce data storage
        # I can simply store index into an array which stores the unique values
        key = hash_array(poly_index)
        subspace_values[jj, :] = values[poly_indices_dict[key], :]
    return subspace_values


def evaluate_sparse_grid_subspace(samples, subspace_index, subspace_values,
                                  samples_1d, config_variables_idx,
                                  basis_type="barycentric"):
    if config_variables_idx is None:
        config_variables_idx = samples.shape[0]
    if basis_type == "barycentric":
        return _evaluate_sparse_grid_subspace_barycentric(
            samples, subspace_index, subspace_values,
            samples_1d, config_variables_idx)

    if config_variables_idx != subspace_index.shape[0]:
        print(config_variables_idx)
        msg = "use of config values with piecwise poly basis will be "
        msg += "implemented shortly "
        raise NotImplementedError(msg)
    return tensor_product_piecewise_polynomial_interpolation_with_values(
        samples, subspace_values, subspace_index, basis_type=basis_type)


def _evaluate_sparse_grid_subspace_barycentric(
        samples, subspace_index, subspace_values,
        samples_1d, config_variables_idx):

    active_sample_vars = np.where(subspace_index[:config_variables_idx] > 0)[0]
    num_active_sample_vars = active_sample_vars.shape[0]

    abscissa_1d = []
    barycentric_weights_1d = []
    for dd in range(num_active_sample_vars):
        active_idx = active_sample_vars[dd]
        abscissa_1d.append(samples_1d[active_idx][subspace_index[active_idx]])
        interval_length = 2
        if abscissa_1d[dd].shape[0] > 1:
            interval_length = abscissa_1d[dd].max()-abscissa_1d[dd].min()
        barycentric_weights_1d.append(
            compute_barycentric_weights_1d(
                abscissa_1d[dd], interval_length=interval_length))

    if num_active_sample_vars == 0:
        return np.tile(subspace_values, (samples.shape[1], 1))
    poly_vals = multivariate_barycentric_lagrange_interpolation(
        samples, abscissa_1d, barycentric_weights_1d, subspace_values,
        active_sample_vars)
    return poly_vals


def evaluate_sparse_grid_subspace_deriv(
        eval_samples, subspace_index, subspace_values,
        samples_1d, config_variables_idx):
    if config_variables_idx is None:
        config_variables_idx = eval_samples.shape[0]

    active_vars = np.where(subspace_index[:config_variables_idx] > 0)[0]
    nactive_vars = active_vars.shape[0]

    abscissa_1d = []
    for dd in range(nactive_vars):
        active_idx = active_vars[dd]
        abscissa_1d.append(samples_1d[active_idx][subspace_index[active_idx]])

    samples = eval_samples[active_vars, :]
    nvars, nsamples = eval_samples.shape
    nqoi = subspace_values.shape[1]
    derivs = np.zeros((nsamples, nqoi, nvars))
    vals, active_derivs = tensor_product_lagrange_jacobian(
        samples, abscissa_1d, subspace_values)
    derivs[:, :, active_vars] = active_derivs
    return vals, derivs


def tensor_product_lagrange_jacobian(samples, abscissa_1d, values):
    nvars = samples.shape[0]
    nabscissa_1d = [a.shape[0] for a in abscissa_1d]
    numer = [[] for dd in range(nvars)]
    denom = [[] for dd in range(nvars)]
    samples_diff = [None for dd in range(nvars)]
    for dd in range(nvars):
        samples_diff[dd] = samples[dd][:, None]-abscissa_1d[dd][None, :]
        abscissa_diff = abscissa_1d[dd][:, None]-abscissa_1d[dd][None, :]
        for jj in range(nabscissa_1d[dd]):
            indices = np.delete(np.arange(nabscissa_1d[dd]), jj)
            numer[dd].append(samples_diff[dd][:, indices].prod(axis=1))
            denom[dd].append(abscissa_diff[jj, indices].prod(axis=0))
        numer[dd] = np.asarray(numer[dd])

    nsamples = samples.shape[1]
    derivs_1d = [np.empty((nsamples, nabscissa_1d[dd]))
                 for dd in range(nvars)]
    for dd in range(nvars):
        # sum over each 1D basis function
        for jj in range(nabscissa_1d[dd]):
            # product rule for the jth 1D basis function
            numer_deriv = 0
            for kk in range(nabscissa_1d[dd]):
                # compute deriv of kth component of product rule sum
                if jj != kk:
                    numer_deriv += np.delete(
                        samples_diff[dd], (jj, kk), axis=1).prod(axis=1)
            derivs_1d[dd][:, jj] = numer_deriv/denom[dd][jj]

    # compute dth derivative for each sample and basis
    nqoi = values.shape[1]
    vals = np.empty((nsamples, nqoi))
    derivs = np.empty((samples.shape[1], nqoi, nvars))
    # numer[dd].shape is [nabscissa_1d[dd], nsamples]
    # derivs_1d[dd].shape is [nsamples, nabscissa_1d[dd]]
    for ii in range(nsamples):
        basis_mat = outer_product(
            [numer[ss][:, ii]/denom[ss] for ss in range(nvars)])
        vals[ii, :] = basis_mat.dot(values)
    for dd in range(nvars):
        for ii in range(nsamples):
            sets = [
                numer[ss][:, ii]/denom[ss] if ss != dd else derivs_1d[dd][ii]
                for ss in range(nvars)]
            deriv_mat = outer_product(sets)
            derivs[ii, :, dd] = deriv_mat.dot(values)
    return vals, derivs


def evaluate_sparse_grid(samples, values,
                         poly_indices_dict,  # not needed with new implementation
                         sparse_grid_subspace_indices,
                         sparse_grid_subspace_poly_indices_list,
                         smolyak_coefficients, samples_1d,
                         sparse_grid_subspace_values_indices_list,
                         config_variables_idx=None, return_grad=False,
                         basis_type="barycentric"):

    num_vars, num_samples = samples.shape
    assert values.ndim == 2
    assert values.shape[0] == len(poly_indices_dict)
    assert sparse_grid_subspace_indices.shape[1] == \
        smolyak_coefficients.shape[0]

    num_qoi = values.shape[1]
    # must initialize to zero
    approx_values = np.zeros((num_samples, num_qoi), dtype=float)
    grads = 0
    for ii in range(sparse_grid_subspace_indices.shape[1]):
        if (abs(smolyak_coefficients[ii]) > np.finfo(float).eps):
            subspace_index = sparse_grid_subspace_indices[:, ii]
            subspace_values = get_subspace_values(
                values, sparse_grid_subspace_values_indices_list[ii])
            if not return_grad:
                subspace_approx_vals = evaluate_sparse_grid_subspace(
                    samples, subspace_index, subspace_values,
                    samples_1d, config_variables_idx, basis_type)
            else:
                subspace_approx_vals, subspace_grads = (
                    evaluate_sparse_grid_subspace_deriv(
                        samples, subspace_index, subspace_values,
                        samples_1d, config_variables_idx))
            approx_values += smolyak_coefficients[ii]*subspace_approx_vals
            if return_grad:
                grads += smolyak_coefficients[ii]*subspace_grads
    if not return_grad:
        return approx_values

    return approx_values, grads


def integrate_sparse_grid_subspace(subspace_index, subspace_values,
                                   weights_1d, config_variables_idx):
    subspace_weights = get_subspace_weights(
        subspace_index, weights_1d, config_variables_idx)
    mean = np.dot(subspace_weights, subspace_values)
    variance = np.dot(subspace_weights, subspace_values**2)-mean**2
    return np.vstack((mean[np.newaxis, :], variance[np.newaxis, :]))


def integrate_sparse_grid(
        values,
        poly_indices_dict,  # not needed with new implementation
        sparse_grid_subspace_indices,
        sparse_grid_subspace_poly_indices_list,
        smolyak_coefficients, weights_1d,
        sparse_grid_subspace_values_indices_list,
        config_variables_idx=None):
    assert values.ndim == 2
    assert values.shape[0] == len(poly_indices_dict)
    assert (sparse_grid_subspace_indices.shape[1] ==
            smolyak_coefficients.shape[0])

    num_qoi = values.shape[1]
    # must initialize to zero
    integral_values = np.zeros((2, num_qoi), dtype=float)
    for ii in range(sparse_grid_subspace_indices.shape[1]):
        if (abs(smolyak_coefficients[ii]) > np.finfo(float).eps):
            subspace_index = sparse_grid_subspace_indices[:, ii]
            subspace_values = get_subspace_values(
                values, sparse_grid_subspace_values_indices_list[ii])
            subspace_integral_vals = integrate_sparse_grid_subspace(
                subspace_index, subspace_values, weights_1d,
                config_variables_idx)
            integral_values += smolyak_coefficients[ii]*subspace_integral_vals
    return integral_values


def integrate_sparse_grid_from_subspace_moments(
        sparse_grid_subspace_indices,
        smolyak_coefficients, subspace_moments):
    assert sparse_grid_subspace_indices.shape[1] == \
        smolyak_coefficients.shape[0]
    assert subspace_moments.shape[0] == sparse_grid_subspace_indices.shape[1]

    num_qoi = subspace_moments.shape[1]
    # must initialize to zero
    integral_values = np.zeros((num_qoi, 2), dtype=float)
    for ii in range(sparse_grid_subspace_indices.shape[1]):
        if (abs(smolyak_coefficients[ii]) > np.finfo(float).eps):
            integral_values += smolyak_coefficients[ii]*subspace_moments[ii]
    # keep shape consistent with shape returned by integrate_sparse_grid
    return integral_values.T


def evaluate_sparse_grid_from_subspace_values(
        sparse_grid_subspace_indices,
        smolyak_coefficients, subspace_interrogation_values):
    """
    Some times you may want to evaluate a sparse grid repeatedly at the
    same set of samples. If so use this function. It avoids recomputing the
    subspace interpolants each time the sparse grid is interrogated.
    Note the reduced time complexity requires more storage
    """
    assert sparse_grid_subspace_indices.shape[1] == \
        smolyak_coefficients.shape[0]
    assert len(subspace_interrogation_values) == \
        sparse_grid_subspace_indices.shape[1]

    # must initialize to zero
    values = 0
    for ii in range(sparse_grid_subspace_indices.shape[1]):
        if (abs(smolyak_coefficients[ii]) > np.finfo(float).eps):
            values += smolyak_coefficients[ii] * \
                subspace_interrogation_values[ii]
    return values


def get_num_sparse_grid_samples(
        sparse_grid_subspace_poly_indices_list,
        smolyak_coefficients):
    """
    This only works if there are no config variables. Specifically it
    will underestimate the number of model evaluations when config variables
    are present For example, if the smolyak coefficient of subspace is 1 and the
    coefficient its backwards neighbor is -1 this function will subtract off
    the number of samples from the backward neighbor to avoid double counting.
    But if config variables are present then the backward neighbour index may
    only vary in the config variables and thus the samples in each of the
    two subspaces come from different models and thus we actually want to
    count the samples of both subspaces.
    """
    num_samples = 0
    for ii in range(smolyak_coefficients.shape[0]):
        if (abs(smolyak_coefficients[ii]) > np.finfo(float).eps):
            subspace_poly_indices = sparse_grid_subspace_poly_indices_list[ii]
            num_subspace_evals = subspace_poly_indices.shape[1]
            num_samples += smolyak_coefficients[ii]*num_subspace_evals
    return num_samples


def plot_sparse_grid_2d(samples, weights, poly_indices=None, subspace_indices=None,
                        axs=None, active_samples=None, active_subspace_indices=None,
                        config_variables_idx=None):
    """
    Plot the sparse grid samples and color the samples by their quadrature
    weight.

    Parameters
    ---------
    samples : np.ndarray (num_vars x num_sparse_grid_samples)
        The unique samples of the sparse grid.

    weights : np.ndarray (num_sparse_grid_samples)
        The quadrature weights of the sparse grid.

    poly_indices : np.ndarray (num_vars x num_sparse_grid_samples)
        The unique polynomial indices of the sparse grid.

    """
    from pyapprox.util.visualization import plot_2d_indices
    if samples.shape[0] != 2:
        return

    nplots = 1 + int(poly_indices is not None) + \
        int(subspace_indices is not None)
    if axs is None:
        fig, axs = plt.subplots(1, nplots, figsize=(nplots*8, 6))
    if type(axs) != np.ndarray:
        axs = [axs]
    assert len(axs) == nplots

    if config_variables_idx is None:
        plot = axs[0].scatter(samples[0, :], samples[1, :], s=100, c=weights,
                              cmap=plt.get_cmap('Greys'), edgecolors='black')
        plt.colorbar(plot, ax=axs[0])
        if active_samples is not None:
            axs[0].plot(active_samples[0, :], active_samples[1, :], 'ro')
    else:
        for ii in range(samples.shape[0]):
            axs[0].plot(samples[0, ii], samples[1, ii], 'ko')
        for ii in range(active_samples.shape[1]):
            axs[0].plot(active_samples[0, ii], active_samples[1, ii], 'ro')
        from matplotlib.pyplot import MaxNLocator
        ya = axs[0].get_yaxis()
        ya.set_major_locator(MaxNLocator(integer=True))
        # axs[0].set_ylabel(r'$\alpha_1$',rotation=0)
        axs[0].set_xlabel('$z_1$', rotation=0)

    ii = 1
    if poly_indices is not None:
        plot_2d_indices(poly_indices, ax=axs[ii])
        ii += 1

    if subspace_indices is not None:
        plot_2d_indices(subspace_indices, active_subspace_indices, ax=axs[ii])
        ii += 1

    return axs


def plot_sparse_grid_3d(samples, weights, poly_indices=None,
                        subspace_indices=None,
                        active_samples=None, active_subspace_indices=None):
    from pyapprox.util.visualization import plot_3d_indices
    if samples.shape[0] != 3:
        return

    nplots = 1 + int(poly_indices is not None) + \
        int(subspace_indices is not None)
    fig = plt.figure(figsize=(2*8, 6))
    axs = []
    ax = fig.add_subplot(1, nplots, 1, projection='3d')
    ax.plot(samples[0, :], samples[1, :], samples[2, :], 'ko')
    if active_samples is not None:
        ax.plot(active_samples[0, :], active_samples[1, :],
                active_samples[2, :], 'ro')
    axs.append(ax)

    angle = 45
    ax.view_init(30, angle)
    # ax.set_axis_off()
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ii = 2
    if poly_indices is not None:
        ax = fig.add_subplot(1, nplots, ii, projection='3d')
        plot_3d_indices(poly_indices, ax=ax)
        axs.append(ax)
        ii += 1

    if subspace_indices is not None:
        ax = fig.add_subplot(1, nplots, ii, projection='3d')
        plot_3d_indices(subspace_indices, ax, active_subspace_indices)
        axs.append(ax)
        ii += 1

    return axs


def evaluate_sparse_grid_subspace_hierarchically(
        samples, values, subspace_index, subspace_values_indices,
        samples_1d, subspace_poly_indices, config_variables_idx):
    if config_variables_idx is None:
        config_variables_idx = samples.shape[0]

    abscissa_1d = []
    barycentric_weights_1d = []
    hier_indices_1d = []
    active_vars = np.where(subspace_index > 0)[0]
    num_active_vars = active_vars.shape[0]

    subspace_values = values[subspace_values_indices, :]

    if num_active_vars == 0:
        return subspace_values

    for dd in range(num_active_vars):
        subspace_level = subspace_index[active_vars[dd]]
        if subspace_level > 0:
            idx1 = samples_1d[subspace_level-1].shape[0]
        else:
            idx1 = 0
        idx2 = samples_1d[subspace_level].shape[0]
        hier_indices_1d.append(np.arange(idx1, idx2))
        abscissa_1d.append(samples_1d[subspace_level])
        barycentric_weights_1d.append(
            compute_barycentric_weights_1d(abscissa_1d[dd]))

    hier_indices = get_hierarchical_sample_indices(
        subspace_index, subspace_poly_indices,
        samples_1d, config_variables_idx)

    hier_subspace_values = subspace_values[hier_indices, :]

    values = multivariate_hierarchical_barycentric_lagrange_interpolation(
        samples, abscissa_1d, barycentric_weights_1d, hier_subspace_values,
        active_vars, hier_indices_1d)

    return values


def evaluate_sparse_grid_hierarchically(
        samples, values,
        poly_indices_dict,  # not needed with new implementation
        sparse_grid_subspace_indices,
        sparse_grid_subspace_poly_indices_list,
        smolyak_coefficients, samples_1d,
        sparse_grid_subspace_values_indices_list,
        config_variables_idx=None):
    """
    This will not currently work as it requires the function argument values to
    be hierarchical surpluses not raw function values.
    """
    num_vars, num_samples = samples.shape
    assert values.ndim == 2
    num_qoi = values.shape[1]
    approx_values = np.zeros((num_samples, num_qoi), dtype=float)

    for ii in range(sparse_grid_subspace_indices.shape[1]):
        approx_values += evaluate_sparse_grid_subspace_hierarchically(
            samples, values, sparse_grid_subspace_indices[:, ii],
            sparse_grid_subspace_values_indices_list[ii], samples_1d,
            sparse_grid_subspace_poly_indices_list[ii], config_variables_idx)
    return approx_values


def get_num_model_evaluations_from_samples(samples, num_config_vars):
    config_vars_dict = dict()
    num_samples = samples.shape[1]
    sample_count = []
    unique_config_vars = []
    for ii in range(num_samples):
        config_vars = samples[-num_config_vars:, ii]
        key = hash_array(config_vars)
        if key in config_vars_dict:
            sample_count[config_vars_dict[key]] += 1
        else:
            config_vars_dict[key] = len(sample_count)
            sample_count.append(1)
            unique_config_vars.append(config_vars)
    unique_config_vars = np.array(unique_config_vars).T
    sample_count = np.array(sample_count)
    I = np.argsort(sample_count)[::-1]
    sample_count = sample_count[I]
    unique_config_vars = unique_config_vars[:, I]
    return np.vstack((sample_count[np.newaxis, :], unique_config_vars))


def get_equivalent_cost(cost_function, model_level_evals, model_ids):
    """
    Returns
    -------
    equivalent_costs : np.ndarray
        Fraction of total work. equivalent_costs.sum()=1
    """
    equivalent_costs = []
    model_costs = cost_function(model_ids)
    equivalent_costs = model_costs*model_level_evals
    total_cost = equivalent_costs.sum()
    equivalent_costs /= float(total_cost)
    return equivalent_costs, total_cost
