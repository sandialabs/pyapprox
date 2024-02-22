import numpy as np

from pyapprox.util.utilities import (
    cartesian_product, get_all_sample_combinations)
from pyapprox.interface.wrappers import WorkTrackingModel
from pyapprox.util.visualization import plt, mathrm_label


def run_convergence_study(model, variable, validation_levels,
                          get_num_degrees_of_freedom, config_var_trans,
                          num_samples=10, coarsest_levels=None,
                          finest_levels=None, reference_model=None):
    if type(model) != WorkTrackingModel:
        raise ValueError("Model must be wrapped as a work tracking model")
    validation_levels = np.asarray(validation_levels)
    assert len(validation_levels) == config_var_trans.num_vars()
    if coarsest_levels is None:
        coarsest_levels = [0]*len(validation_levels)
    if finest_levels is None:
        finest_levels = np.asarray(validation_levels)-1
    finest_levels = np.asarray(finest_levels)
    if np.any(coarsest_levels >= validation_levels):
        msg = "Entries of coarsest_levels must be smaller than "
        msg += "validation_levels"
        raise ValueError(msg)
    if np.any(finest_levels >= validation_levels):
        msg = "Entries of finest_levels must be smaller than "
        msg += "validation_levels"
        raise ValueError(msg)
    canonical_config_samples_1d = [
        np.arange(jj, kk)
        for jj, kk in zip(coarsest_levels, finest_levels+1)]
    canonical_config_samples = cartesian_product(
        canonical_config_samples_1d)
    config_samples = config_var_trans.map_from_canonical(
        canonical_config_samples)

    random_samples = variable.rvs(num_samples)
    samples = get_all_sample_combinations(random_samples, config_samples)

    reference_samples = get_all_sample_combinations(
        random_samples, config_var_trans.map_from_canonical(
            validation_levels[:, None]))

    reference_values = model(reference_samples)
    if reference_model is None:
        reference_values = model(reference_samples)
    else:
        reference_values = reference_model(
            reference_samples[:-config_var_trans.num_vars(), :])
    reference_mean = reference_values[:, 0].mean()

    if np.absolute(reference_mean) <= 1e-15:
        raise RuntimeError(
            "Cannot plot relative errors because reference mean is zero")

    values = model(samples)

    # put keys in order returned by cartesian product
    keys = sorted(model.work_tracker.costs.keys(),
                  key=lambda x: tuple(
                      config_var_trans.map_to_canonical(
                          np.asarray(x)[:, None])[::-1, 0]))
    validation_ndof = get_num_degrees_of_freedom(keys[-1])
    # remove validation key associated with validation samples
    keys = keys[:-1]
    costs, ndofs, means, errors = [], [], [], []
    for ii in range(len(keys)):
        key = keys[ii]
        costs.append(np.median(model.work_tracker.costs[key]))
        ndof = get_num_degrees_of_freedom(key)
        ndofs.append(ndof)
        means.append(np.mean(values[ii::config_samples.shape[1], 0]))
        errors.append(abs(means[-1]-reference_mean)/abs(reference_mean))

    times = costs.copy()
    # make costs relative
    # costs /= costs[-1]

    shape = tuple(np.asarray(finest_levels)-np.asarray(coarsest_levels)+1)
    indices = np.reshape(
        np.arange(len(keys), dtype=int), shape, order='F')
    costs = np.reshape(np.array(costs), shape, order='F')
    ndofs = np.reshape(np.array(ndofs), shape, order='F')
    errors = np.reshape(np.array(errors), shape, order='F')
    times = np.reshape(np.array(times), shape, order='F')

    validation_index = reference_samples[-config_var_trans.num_vars():, 0]
    validation_time = np.median(
        model.work_tracker.costs[tuple(
            reference_samples[-config_var_trans.num_vars():, 0])])
    validation_cost = validation_time  # /costs[[-1]*len(costs)]
    data = {"costs": costs, "errors": errors, "indices": indices,
            "times": times, "validation_index": validation_index,
            "validation_cost": validation_cost,
            "validation_ndof": validation_ndof,
            "validation_time": validation_time, "ndofs": ndofs,
            'canonical_config_samples_1d': canonical_config_samples_1d}

    return data


def plot_convergence_data(data, cost_type='ndof'):

    errors, costs = data['errors'], data['costs']
    config_idx = data["canonical_config_samples_1d"]

    if cost_type == 'ndof':
        costs = data['ndofs']/data['ndofs'].max()
    validation_levels = costs.shape
    nconfig_vars = len(validation_levels)
    fig, axs = plt.subplots(1, nconfig_vars,
                            figsize=(nconfig_vars*8, 6),
                            sharey=False)
    if nconfig_vars == 1:
        label = r'$(\cdot)$'
        axs.loglog(costs, errors, 'o-', label=label)
    if nconfig_vars == 2:
        for ii in range(validation_levels[1]):
            label = r'$(\cdot,%d)$' % (config_idx[1][ii])
            axs[0].loglog(costs[:, ii], errors[:, ii], 'o-', label=label)
        for ii in range(validation_levels[0]):
            label = r'$(%d,\cdot)$' % (config_idx[0][ii])
            axs[1].loglog(costs[ii, :], errors[ii, :], 'o-', label=label)
    if nconfig_vars == 3:
        for ii in range(validation_levels[1]):
            jj = costs.shape[2]-1
            label = r'$(\cdot,%d,%d)$' % (config_idx[1][ii], config_idx[2][jj])
            axs[0].loglog(costs[:, ii, jj], errors[:, ii, jj],
                          'o-', label=label)
        for ii in range(validation_levels[0]):
            jj = costs.shape[2]-1
            label = r'$(%d,\cdot,%d)$' % (config_idx[0][ii], config_idx[2][jj])
            axs[1].loglog(costs[ii, :, jj], errors[ii, :, jj],
                          'o-', label=label)
            jj = costs.shape[1]-1
            label = r'$(%d,%d,\cdot)$' % (config_idx[0][ii], config_idx[1][jj])
            axs[2].loglog(costs[ii, jj, :], errors[ii, jj, :],
                          'o-', label=label)

    for ii in range(nconfig_vars):
        axs[ii].legend()
        axs[ii].set_xlabel(mathrm_label("Work ") + r'$W_{\alpha}$')
        axs[0].set_ylabel(
            r'$\left| \mathbb{E}[f]-\mathbb{E}[f_{\alpha}]\right| / \left| \mathbb{E}[f]\right|$')
    return fig, axs
