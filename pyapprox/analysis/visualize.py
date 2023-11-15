import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.variables.joint import (
    get_truncated_range, get_truncated_ranges)
from pyapprox.surrogates.interp.indexing import compute_anova_level_indices


def plot_qoi_marginals(values):
    """
    Use KDE to plot the marginals of each QoI.

    Parameters
    ----------
    values : np.ndarray (nsamples, nqoi)
        Realiations of the QoI
    """
    nqoi = values.shape[1]
    fig, axs = plt.subplots(1, nqoi, figsize=(nqoi*8, 6))
    if nqoi == 1:
        axs = [axs]
    for ii in range(nqoi):
        kde = stats.gaussian_kde(values[:, ii])
        yy = np.linspace(values[:, ii].min(), values[:, ii].max(), 101)
        axs[ii].plot(yy, kde(yy))
    return axs


def get_meshgrid_samples_from_variable(variable, num_pts_1d,
                                       logspace=False, unbounded_alpha=.99):
    plot_limits = get_truncated_ranges(variable, unbounded_alpha)
    X, Y, pts = get_meshgrid_samples(plot_limits, num_pts_1d, logspace)
    return X, Y, pts


def get_meshgrid_function_data_from_variable(
        function, variable, num_pts_1d, qoi=0,
        logspace=False, unbounded_alpha=.99):
    r"""
    Generate data from a function in the format needed for plotting.
    Samples are generated between specified lower and upper bounds
    and the function is evaluated on the tensor product of the 1d samples.

    Parameters
    ----------
    function : callable function
        The function must accept an np.ndarray of size (2, num_pts_1d**2)
        and return a np.ndarray of size (num_pts_1d,num_qoi)

    variable : :class:`pyapprox.variables.IndependentMarginalsVariable`
        Variable used to determine plotting ranges

    num_pts_1d : integer
        The number of samples in each dimension. The function is evaluated
        on the tensor product of the 1d samples

    qoi : integer
        function returns a np.ndarray of size (num_pts_1d,num_qoi) qoi
        specifies which column of the array to access.

    unbouned_alpha : float
        For any unbounded variable set the plot ranges to contain this fraction
        of the distribution, e.g. 0.99 will contain 99% of the probability

    Returns
    -------
    X : np.ndarray of size (num_pts_1d,num_pts_1d)
        The 1st coordinate of the samples

    Y : np.ndarray of size (num_pts_1d,num_pts_1d)
        The 2nd coordinate of the samples

    Z : np.ndarray of size (num_pts_1d,num_pts_1d)
        The function values at each sample
    """
    X, Y, pts = get_meshgrid_samples_from_variable(
        variable, num_pts_1d, logspace, unbounded_alpha)
    Z = function(pts)
    if (Z.ndim == 2):
        Z = Z[:, qoi]
    Z = np.reshape(Z, (X.shape[0], X.shape[1]))
    return X, Y, Z


def plot_1d_cross_section(fun, var, var_idx, nominal_sample, nsamples_1d,
                          ax, qoi, plt_kwargs):
    """
    Plot a single 1D cross section of a multivariate function.
    """
    lb, ub = get_truncated_range(var, 0.99)
    samples = np.tile(nominal_sample, (1, nsamples_1d))
    samples[var_idx, :] = np.linspace(lb, ub, nsamples_1d)
    values = fun(samples)
    ax.plot(samples[var_idx, :], values[:, qoi], **plt_kwargs)


def plot_1d_cross_sections(fun, variable, nominal_sample=None,
                           nsamples_1d=100, subplot_tuple=None, qoi=0,
                           plt_kwargs={}, axs=None):
    """
    Plot the 1D cross sections of a multivariate function.
    """
    if nominal_sample is None:
        nominal_sample = variable.get_statistics("mean")

    if subplot_tuple is None:
        nfig_rows, nfig_cols = 1, variable.num_vars()
    else:
        nfig_rows, nfig_cols = subplot_tuple

    if nfig_rows*nfig_cols < variable.num_vars():
        raise ValueError("Number of subplots is insufficient")

    if axs is None:
        fig, axs = plt.subplots(
            nfig_rows, nfig_cols, figsize=(nfig_cols*8, nfig_rows*6))
        if variable.num_vars() == 1:
            axs = [axs]
        else:
            axs = axs.flatten()
    all_variables = variable.marginals()
    for ii, var in enumerate(all_variables):
        axs[ii].set_title(r"$Z_{%d}$" % (ii+1))
        plot_1d_cross_section(
            fun, var, ii, nominal_sample, nsamples_1d, axs[ii], qoi,
            plt_kwargs)

    for ii in range(variable.num_vars(), nfig_rows*nfig_cols):
        axs[ii].axis("off")

    return axs


def setup_2d_cross_section_axes(variable, variable_pairs, subplot_tuple):
    if variable_pairs is None:
        variable_pairs = np.array(
            compute_anova_level_indices(variable.num_vars(), 2))
        # make first column values vary fastest so we plot lower triangular
        # matrix of subplots
        variable_pairs[:, 0], variable_pairs[:, 1] = \
            variable_pairs[:, 1].copy(), variable_pairs[:, 0].copy()

    if variable_pairs.shape[1] != 2:
        raise ValueError("Variable pairs has the wrong shape")

    if subplot_tuple is None:
        nfig_rows, nfig_cols = variable.num_vars(), variable.num_vars()
    else:
        nfig_rows, nfig_cols = subplot_tuple

    if nfig_rows*nfig_cols < len(variable_pairs):
        raise ValueError("Number of subplots is insufficient")

    fig, axs = plt.subplots(
        nfig_rows, nfig_cols, sharex="col")
    #, figsize=(nfig_cols*8, nfig_rows*6))
    return fig, axs, variable_pairs


def plot_2d_cross_sections(fun, variable, nominal_sample=None,
                           nsamples_1d=100, variable_pairs=None,
                           subplot_tuple=None, qoi=0, num_contour_levels=20,
                           plot_samples=None, marginals=False):
    """
    Plot the 2D cross sections of a multivariate function.
    """
    if nominal_sample is None:
        nominal_sample = variable.get_statistics("mean")

    fig, axs, variable_pairs = setup_2d_cross_section_axes(
        variable, variable_pairs, subplot_tuple)

    all_variables = variable.marginals()

    if plot_samples is not None and type(plot_samples) == np.ndarray:
        plot_samples = [[plot_samples, 'ko']]

    for ii, var in enumerate(all_variables):
        lb, ub = get_truncated_range(var)
        if not marginals:
            samples = np.tile(nominal_sample, (1, nsamples_1d))
            samples[ii, :] = np.linspace(lb, ub, nsamples_1d)
            values = fun(samples)
            axs[ii][ii].plot(samples[ii, :], values[:, qoi])
        else:
            quad_degrees = np.array([20]*(variable.num_vars()-1))
            samples_ii = np.linspace(lb, ub, nsamples_1d)
            from pyapprox.surrogates.polychaos.gpc import (
                _marginalize_function_1d, _marginalize_function_nd)
            values = _marginalize_function_1d(
                fun, variable, quad_degrees, ii, samples_ii, qoi=0)
            axs[ii][ii].plot(samples_ii, values)

        if plot_samples is not None:
            for s in plot_samples:
                axs[ii][ii].plot(s[0][ii, :], s[0][ii, :]*0, s[1])

    for ii, pair in enumerate(variable_pairs):
        # use pair[1] for x and pair[0] for y because we reverse
        # pairs above
        var1, var2 = all_variables[pair[1]], all_variables[pair[0]]
        axs[pair[1], pair[0]].axis("off")
        lb1, ub1 = get_truncated_range(var1)
        lb2, ub2 = get_truncated_range(var2)
        X, Y, samples_2d = get_meshgrid_samples(
            [lb1, ub1, lb2, ub2], nsamples_1d)
        if marginals:
            quad_degrees = np.array([10]*(variable.num_vars()-2))
            values = _marginalize_function_nd(
                fun, variable, quad_degrees, np.array([pair[1], pair[0]]),
                samples_2d, qoi=qoi)
        else:
            samples = np.tile(nominal_sample, (1, samples_2d.shape[1]))
            samples[[pair[1], pair[0]], :] = samples_2d
            values = fun(samples)[:, qoi]
        Z = np.reshape(values, (X.shape[0], X.shape[1]))
        ax = axs[pair[0]][pair[1]]
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, r"$(\mathrm{%d, %d})$" % (pair[1], pair[0]),
                transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.contourf(
            X, Y, Z, levels=np.linspace(Z.min(), Z.max(), num_contour_levels),
            cmap='jet')
        if plot_samples is not None:
            for s in plot_samples:
                # use pair[1] for x and pair[0] for y because we reverse
                # pairs above
                axs[pair[0]][pair[1]].plot(
                    s[0][pair[1], :], s[0][pair[0], :], s[1])

    return fig, axs


def plot_discrete_measure_1d(samples, weights, ax=None):
    """
    Plot a discrete measure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plt.plot(samples, weights, 'o')
    for s, w in zip(samples, weights):
        ax.vlines(x=s, ymin=0, ymax=w)
    return ax


def plot_discrete_distribution_surface_2d(rv1, rv2, ax=None):
    """
    Plot the probability masses of a 2D discrete random variable.

    Only works if rv1 and rv2 are defined on consecutive integers
    """
    from matplotlib import cm
    from pyapprox.util.utilities import cartesian_product, outer_product
    from pyapprox.variables.marginals import get_probability_masses

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    x_1d = [get_probability_masses(rv)[0] for rv in [rv1, rv2]]
    w_1d = [get_probability_masses(rv)[1] for rv in [rv1, rv2]]
    samples = cartesian_product(x_1d)
    weights = outer_product(w_1d)

    dz = weights
    cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
    max_height = np.max(dz)    # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz]
    # Only works if rv1 and rv2 are defined on consecutive integers
    dx, dy = 1, 1
    ax.bar3d(samples[0, :], samples[1, :], 0, dx, dy, dz, color=rgba,
             zsort='average')

    angle = 45
    ax.view_init(10, angle)
    ax.set_axis_off()


def plot_discrete_distribution_heatmap_2d(rv1, rv2, ax=None, zero_tol=1e-4):
    """
    Plot the probability masses of a 2D discrete random variable.

    Only works if rv1 and rv2 are defined on consecutive integers
    """
    import copy
    from pyapprox.util.utilities import outer_product
    from pyapprox.variables.marginals import get_probability_masses

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    x_1d = [get_probability_masses(rv)[0] for rv in [rv1, rv2]]
    w_1d = [get_probability_masses(rv)[1] for rv in [rv1, rv2]]
    weights = outer_product(w_1d)

    Z = np.reshape(weights, (len(x_1d[0]), len(x_1d[1])), order='F')
    Z[Z < zero_tol] = np.inf
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad('gray', 1)
    xx = np.hstack((x_1d[0], x_1d[0].max()+1))-0.5
    yy = np.hstack((x_1d[1], x_1d[1].max()+1))-0.5
    p = ax.pcolormesh(xx, yy, Z.T, cmap=cmap)
    plt.colorbar(p, ax=ax)
    # xticks = ax.get_xticks()
    # xticklabels = ax.get_xticklabels()
    # print(xticklabels, xticks)
    # yticks = ax.get_yticks()
    # yticklabels = ax.get_yticklabels()
    # print(yticklabels, yticks)
    # ax.set_xticks((xx[:-1]+xx[1:])/2)
    # ax.set_xticklabels([f"${x}$" for x in x_1d[0]])
    # ax.set_yticks((yy[:-1]+yy[1:])/2)
    # ax.set_yticklabels([f"${x}$" for x in x_1d[1]])
    
