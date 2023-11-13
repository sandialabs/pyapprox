import numpy as np
import networkx as nx

from pyapprox.multifidelity.multioutput_monte_carlo import determinant_variance


def _hp(G, root, width=1., vert_gap=0.2, vert_loc=0,
        xcenter=0.5, pos=None, parent=None):
    '''
    see hierarchy_pos docstring for most arguments

    pos: a dict saying where all nodes go if they have been assigned
    parent: parent of this branch. - only affects it if non-directed

    '''

    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width/len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hp(
                G, child, width=dx, vert_gap=vert_gap,
                vert_loc=vert_loc-vert_gap, xcenter=nextx,
                pos=pos, parent=root)
    return pos


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):

    '''
    Motivated by Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with
           other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        msg = 'cannot use hierarchy_pos on a graph that is not a tree'
        raise TypeError(msg)

    return _hp(G, root, width, vert_gap, vert_loc, xcenter)


def plot_model_recursion(recursion_index, ax):
    nmodels = len(recursion_index)+1
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(nmodels))
    for ii, jj in enumerate(recursion_index):
        graph.add_edge(ii+1, jj)
    pos = _hierarchy_pos(graph, 0, vert_gap=0.1, width=0.1)
    nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=[2000],
            font_size=24)


def plot_model_costs(costs, model_names=None, ax=None):
    from pyapprox.util.configure_plots import plt
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
                             criteria=determinant_variance):
    """
    Plot variance as a function of the total cost for a set of estimators.

    Parameters
    ----------
    optimized_estimators : list
         Each entry is a list of optimized estimators for a set of target costs

    est_labels : list (nestimators)
        String used to label each estimator

    relative_id the model id used to normalize variance
    """
    from pyapprox.util.configure_plots import mathrm_label
    linestyles = ['-', '--', ':', '-.', (0, (5, 10)), '-']
    nestimators = len(est_labels)
    est_criteria = []
    for ii in range(nestimators):
        est_total_costs = np.array(
            [est._rounded_target_cost for est in optimized_estimators[ii]])
        est_criteria.append(np.array(
            [criteria(est._covariance_from_npartition_samples(
                est._npartition_samples), est)
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
                                       criteria=determinant_variance,
                                       **bar_kawrgs):
    """
    Plot variance as a function of the total cost for a set of estimators.

    Parameters
    ----------
    optimized_estimators : list
         Each entry is a list of optimized estimators for a set of target costs

    est_labels : list (nestimators)
        String used to label each estimator

    """
    from pyapprox.util.configure_plots import mathrm_label
    var_red, est_criterias, sf_criterias = [], [], []
    optimized_estimators = optimized_estimators.copy()
    est_labels = est_labels.copy()
    nestimators = len(est_labels)
    for ii in range(nestimators):
        assert len(optimized_estimators[ii]) == 1
        est = optimized_estimators[ii][0]
        est_criteria = criteria(est._covariance_from_npartition_samples(
            est._npartition_samples), est)
        nhf_samples = int(est._rounded_target_cost/est._costs[0])
        sf_criteria = criteria(
            est._stat.high_fidelity_estimator_covariance(
                [nhf_samples]), est)
        var_red.append(sf_criteria/est_criteria)
        sf_criterias.append(sf_criteria)
        est_criterias.append(est_criteria)
    rects = ax.bar(est_labels, var_red, **bar_kawrgs)
    rects = [r for r in rects]  # convert to list
    from pyapprox.multifidelity.monte_carlo_estimators import _autolabel
    _autolabel(ax, rects, ['$%1.2f$' % (v) for v in var_red])
    if ylabel is None:
        ylabel = mathrm_label("Estimator variance reduction")
    ax.set_ylabel(ylabel)
    return var_red, est_criterias, sf_criterias


def plot_correlation_matrix(corr_matrix, ax=None, model_names=None,
                            format_string='{:1.3f}', cmap="jet", nqoi=1):
    """
    Plot a correlation matrix

    Parameters
    ----------
    corr_matrix : np.ndarray (nvars, nvars)
         The correlation between a set of random variabels
    """
    from pyapprox.util.configure_plots import plt
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
    ax.set_yticklabels(model_names)
    ax.set_xticklabels(model_names, rotation=60)
    return ax
