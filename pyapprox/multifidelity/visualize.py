import numpy as np

from pyapprox.util.visualization import plt, mathrm_label
from pyapprox.multifidelity._visualize import _autolabel
from pyapprox.multifidelity.factory import (
    compute_variance_reductions, ComparisonCriteria)


def plot_model_costs(costs, model_names=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nmodels = len(costs)
    if model_names is None:
        model_names = [r"$f_{%d}$" % ii for ii in range(nmodels)]
    ax.bar(np.arange(nmodels), costs)
    ax.set_xticks(np.arange(nmodels))
    ax.set_xticklabels(model_names)


def plot_estimator_variances(optimized_estimators,
                             est_labels, ax, ylabel=None,
                             relative_id=0, cost_normalization=1,
                             criteria=ComparisonCriteria("det")):
    """
    Plot the estimator variance for a list of estimators.

    Parameters
    ----------
    optimized_estimators : list
         Each entry is a list of optimized estimators for a set of target costs

    est_labels : list (nestimators)
        String used to label each estimator

    relative_id : integer

        The model id used to normalize variance

    cost_normalization : float
        Costs are divided by this value. Useful, for example, when you
        want to plot variances as a function of the equivalent number of
        high-fidelity evaluations in which case cost_normalization=costs[0]

    criteria : callable
        A function that returns a scalar metric of the estimator covariance
        with signature

        `criteria(cov) -> float`

        where cov is an np.ndarray (nstats, nstats) is the estimator covariance
    """
    linestyles = ['-', '--', ':', '-.', (0, (5, 10)), '-']
    nestimators = len(est_labels)
    est_criteria = []
    for ii in range(nestimators):
        est_total_costs = np.array(
            [est._rounded_target_cost for est in optimized_estimators[ii]])
        est_criteria.append(np.array(
            [criteria(est._covariance_from_npartition_samples(
                est._rounded_npartition_samples), est)
             for est in optimized_estimators[ii]]))
    est_total_costs *= cost_normalization
    for ii in range(nestimators):
        ax.loglog(est_total_costs,
                  est_criteria[ii]/est_criteria[relative_id][0],
                  label=est_labels[ii], ls=linestyles[ii], marker='o')
    if ylabel is None:
        ylabel = mathrm_label("Estimator variance")
    ax.set_xlabel(mathrm_label("Target cost"))
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_estimator_variance_reductions(optimized_estimators,
                                       est_labels, ax, ylabel=None,
                                       criteria=ComparisonCriteria("det"),
                                       **bar_kawrgs):
    """
    Plot the variance reduction (relative to single model MC) for a
    list of optimized estimtors.

    Parameters
    ----------
    optimized_estimators : list
         Each entry is a list of optimized estimators for a set of target costs

    est_labels : list (nestimators)
        String used to label each estimator

    criteria : callable
        A function that returns a scalar metric of the estimator covariance
        with signature

        `criteria(cov) -> float`

        where cov is an np.ndarray (nstats, nstats) is the estimator covariance
    """
    est_labels = est_labels.copy()
    var_red, est_criterias, sf_criterias = compute_variance_reductions(
        optimized_estimators, criteria)
    rects = ax.bar(est_labels, var_red, **bar_kawrgs)
    rects = [r for r in rects]  # convert to list
    _autolabel(ax, rects, ['$%1.2f$' % (v) for v in var_red])
    if ylabel is None:
        ylabel = mathrm_label("Estimator variance reduction")
    ax.set_ylabel(ylabel)
    return var_red, est_criterias, sf_criterias


def plot_correlation_matrix(corr_matrix, ax=None, model_names=None,
                            format_string='{:1.3f}', cmap="jet", nqoi=1,
                            label_fontsize=16):
    """
    Plot a correlation matrix

    Parameters
    ----------
    corr_matrix : np.ndarray (nvars, nvars)
         The correlation between a set of random variabels
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(corr_matrix, cmap=cmap, aspect="auto")
    for (i, j), z in np.ndenumerate(corr_matrix):
        if format_string is not None:
            ax.text(j, i, format_string.format(z), ha='center', va='center',
                    fontsize=12, color='w')
    plt.colorbar(im, ax=ax)
    if model_names is None:
        nmodels = corr_matrix.shape[0]
        model_names = [r"$f_{%d}$" % ii for ii in range(nmodels)]
    ax.set_xticks(np.arange(len(model_names))*nqoi)
    ax.set_yticks(np.arange(len(model_names))*nqoi)
    ax.set_yticklabels(model_names, fontsize=label_fontsize)
    ax.set_xticklabels(model_names, rotation=60, fontsize=label_fontsize)
    return ax


def plot_estimator_sample_allocation_comparison(
        estimators, model_labels, ax, legendloc=[0.925, 0.25], xlabels=None):
    """
    Plot the number of samples allocated to each model for a set of estimators

    Parameters
    ----------
    estimators : list
       Each entry is a MonteCarlo like estimator

    model_labels : list (nestimators)
        String used to label each estimator
    """

    nestimators = len(estimators)
    xlocs = np.arange(nestimators)

    from matplotlib.pyplot import cm
    for jj, est in enumerate(estimators):
        cnt = 0
        # warning currently colors will not match if estimators use different
        # models
        colors = cm.rainbow(np.linspace(0, 1, est._nmodels))
        rects = []
        est.model_labels = model_labels
        for ii in range(est._nmodels):
            if jj == 0:
                label = est.model_labels[ii]
            else:
                label = None
            cost_ratio = (est._costs[ii]*est._rounded_nsamples_per_model[ii] /
                          est._rounded_target_cost)
            rect = ax.bar(
                xlocs[jj:jj+1], cost_ratio, bottom=cnt, edgecolor='white',
                label=label, color=colors[ii])
            rects.append(rect)
            cnt += cost_ratio
        _autolabel(ax, rects,
                   ['$%d$' % int(est._rounded_nsamples_per_model[ii])
                    for ii in range(est._nmodels)])
    ax.set_xticks(xlocs)
    # number of samples are rounded cost est_rounded cost,
    # but target cost is not rounded
    if xlabels is None:
        ax.set_xticklabels(
            ['$%1.2f$' % est._rounded_target_cost for est in estimators])
    else:
        ax.set_xticklabels(xlabels)
    ax.set_xlabel(mathrm_label("Target cost"))
    # / $N_\alpha$')
    ax.set_ylabel(
        mathrm_label("Precentage of target cost"))
    if legendloc is not None:
        ax.legend(loc=legendloc)
