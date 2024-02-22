import numpy as np
import os
import matplotlib.pyplot as plt

from pyapprox.variables.joint import get_truncated_ranges
from pyapprox.variables.transforms import map_hypercube_samples
from pyapprox.variables.density import map_from_canonical_gaussian


def get_parameter_sweeps_samples_using_rotation(
        rotation_matrix, get_sweep_bounds, num_samples_per_sweep=50,
        random_samples=False):
    """
    Get the samples of parameter sweeps through  directions
    d-dimensional hypercube on [a_0,b_0] x ... x [a_d,b_d]. Directions are
    specified by the rows of rotation matrix

    Parameters
    ----------
    rotation_matrix : np.ndarray (num_vars,num_sweeps)
        Each row contains the rotation vector of each sweep.

    get_sweep_bounds : callable
        lb,ub = get_sweep_uppper_and_lower_bounds(W)
        Function to compute the upper and lower bounds of the parameter sweep

    num_samples_per_sweep : integer
        The number of samples in each of the parameter sweeps

    random_samples : boolean
        True  - generate uniform random samples within bounds of sweep
        False - generate equidistant samples within bounds of sweep

    Returns
    -------
    samples : np.ndarray (num_vars, num_samples_per_sweep*num_sweeps)
        The samples in the d-dimensional space. Each sweep is listed
        consecutivelty. That is num_samples_per_sweep for first sweep
        are the first rows, then the second sweep are the next set of
        num_samples_per_sweep rows, and so on.

    active_samples : np.ndarray (num_sweeps, num_samples_per_sweep)
        The univariate samples of the parameter sweeps. These samples are
        for normalized hypercubes [-1,1]^d.
    """
    # num_sweeps, num_vars = rotation_matrix.shape
    num_vars, num_sweeps = rotation_matrix.shape

    samples = np.empty((num_vars, num_samples_per_sweep*num_sweeps))
    active_samples = np.empty((num_sweeps, num_samples_per_sweep))
    for i in range(num_sweeps):
        W = rotation_matrix[:, i:i+1]
        # find approximate upper and lower bounds for active variable
        y_lb, y_ub = get_sweep_bounds(W)
        # define samples in sweep inside approximate upper and lower bounds
        if num_samples_per_sweep == 1:
            y = np.asarray([[(y_lb+y_ub)/2.]])
        else:
            if random_samples:
                y = np.random.uniform(y_lb, y_ub, (1, num_samples_per_sweep))
            else:
                y = np.asarray(
                    [np.linspace(y_lb, y_ub, num_samples_per_sweep)])
        x = np.dot(W, y)
        active_samples[i, :] = y[0, :]
        samples[:, i*num_samples_per_sweep:(i+1)*num_samples_per_sweep] = x

    return samples, active_samples


def get_parameter_sweeps_samples(
        num_vars, get_sweep_bounds, num_samples_per_sweep=50, num_sweeps=1,
        random_samples=False, sweep_rotation_matrix=None):
    """
    Get the samples of parameter sweeps through random directions of a
    d-dimensional hypercube on [a_0,b_0] x ... x [a_d,b_d]

    Parameters
    ----------
    num_vars : integer
        The number of variables d

    get_sweep_bounds : callable
        lb,ub = get_sweep_uppper_and_lower_bounds(W)
        Function to compute the upper and lower bounds of the parameter sweep

    num_samples_per_sweep : integer
        The number of samples in each of the parameter sweeps

    num_sweeps : integer
        The number of sweeps

    random_samples : boolean
        True  - return random samples along sweep
        False - return equidistanly spaced samples

    sweep_rotation_matrix : np.ndarray (num_vars, num_sweeps)
        Precomputed directions along which to compute the parameter sweeps.
        If matrices is zeros except a value of one in each column
        the parameter sweeps will be along the axial directions

    Returns
    -------
    samples : np.ndarray (num_vars, num_samples_per_sweep*num_sweeps)
        The samples in the d-dimensional space. Each sweep is listed
        consecutivelty. That is num_samples_per_sweep for first sweep
        are the first rows, then the second sweep are the next set of
        num_samples_per_sweep rows, and so on.

    active_samples : np.ndarray (num_sweeps, num_samples_per_sweep)
        The univariate samples of the parameter sweeps. These samples are
        for normalized hypercubes [-1,1]^d.

    rotation_matrix : np.ndarray (num_sweeps, num_vars)
        Each row contains the rotation vector of each sweep.
    """
    samples = np.empty((num_vars, num_samples_per_sweep*num_sweeps))
    active_samples = np.empty((num_sweeps, num_samples_per_sweep))

    if sweep_rotation_matrix is not None:
        assert num_vars == sweep_rotation_matrix.shape[0]
        assert num_sweeps == sweep_rotation_matrix.shape[1]
        samples, active_samples = get_parameter_sweeps_samples_using_rotation(
            sweep_rotation_matrix, get_sweep_bounds, num_samples_per_sweep,
            random_samples)
        return samples, active_samples, sweep_rotation_matrix

    # can only generate a maximum of num_var sweeps for each rotation matrix
    # to generate more create different rotation matrices until enough
    # sweeps are generated
    ii = 0
    samples, active_samples, rot_mats = [], [], []
    while ii*num_vars < num_sweeps:
        A = np.random.normal(0, 1, (num_vars, num_sweeps))
        Q, R = np.linalg.qr(A)
        sweep_rotation_matrix_ii = Q[:, :min(num_vars, num_sweeps-ii*num_vars)]

        samples_ii, active_samples_ii = \
            get_parameter_sweeps_samples_using_rotation(
                sweep_rotation_matrix_ii, get_sweep_bounds,
                num_samples_per_sweep, random_samples)
        samples.append(samples_ii)
        active_samples.append(active_samples_ii)
        rot_mats.append(sweep_rotation_matrix_ii)
        ii += 1

    return np.hstack(samples), np.vstack(active_samples), rot_mats


def get_hypercube_sweep_bounds(W):
    num_vars = W.shape[0]
    maxdist = np.sqrt(num_vars*4)
    y = np.asarray([np.linspace(-maxdist/2., maxdist/2., 1000)])
    x = np.dot(W, y)
    II = np.where(np.all(x >= -1, axis=0) & np.all(x <= 1, axis=0))[0]
    y_lb = y[0, II[0]]
    y_ub = y[0, II[-1]]
    return y_lb, y_ub


def get_hypercube_parameter_sweeps_samples(
        ranges, num_samples_per_sweep=50, num_sweeps=1,
        sweep_rotation_matrix=None):
    """
    Get the samples of parameter sweeps through random directions of a
    d-dimensional hypercube on [a_0,b_0] x ... x [a_d,b_d]

    Parameters
    ----------
    ranges : np.ndarray (2*num_vars)
        lower and upper bounds for each of the d random variables
        [lb_1,ub_1,...,lb_d,ub_d]

    num_samples_per_sweep : integer
        The number of samples in each of the parameter sweeps

    num_sweeps : integer
        The number of sweeps

    Returns
    -------
    samples : np.ndarray (num_vars, num_samples_per_sweep*num_sweeps)
        The samples in the d-dimensional space. Each sweep is listed
        consecutivelty. That is num_samples_per_sweep for first sweep
        are the first rows, then the second sweep are the next set of
        num_samples_per_sweep rows, and so on.

    active_samples : np.ndarray (num_sweeps, num_samples_per_sweep)
        The univariate samples of the parameter sweeps. These samples are
        for normalized hypercubes [-1,1]^d.
    """

    num_vars = ranges.shape[0]//2

    samples, active_samples, W = get_parameter_sweeps_samples(
        num_vars, get_hypercube_sweep_bounds,
        num_samples_per_sweep, num_sweeps,
        sweep_rotation_matrix=sweep_rotation_matrix)

    canonical_ranges = np.ones((num_vars*2), dtype=float)
    canonical_ranges[::2] = -1
    samples = map_hypercube_samples(
        samples, canonical_ranges, ranges)
    return samples, active_samples, W


def get_gaussian_parameter_sweeps_samples(
        mean, covariance=None, covariance_sqrt=None,
        sweep_radius=1, num_samples_per_sweep=50, num_sweeps=1,
        sweep_rotation_matrix=None):
    """
    Get the samples of parameter sweeps through random directions of a
    zero-mean multivariate Gaussian

    One and only one of covariance and covariance_sqrt must be
    not None.

    Parameters
    ----------
    mean : np.ndarray (num_vars)
        The mean of the multivariate Gaussian

    covariance : np.ndarray (num_vars, num_vars)
        The covariance of the multivariate Gaussian

    covariance_sqrt : callable
        correlated_samples = covariance_sqrt(stdnormal_samples)
        An operator that applies the sqrt of the Gaussian covariance to a set
        of vectors. Useful for large scale applications.

    sweep_radius : float
        The radius of the parameter sweep as a multiple of
        one standard deviation of the standard normal

    num_samples_per_sweep : integer
        The number of samples in each of the parameter sweeps

    num_sweeps : integer
        The number of sweeps

    Returns
    -------
    samples : np.ndarray (num_vars, num_samples_per_sweep*num_sweeps)
        The samples in the d-dimensional space. Each sweep is listed
        consecutivelty. That is num_samples_per_sweep for first sweep
        are the first rows, then the second sweep are the next set of
        num_samples_per_sweep rows, and so on.

    active_samples : np.ndarray (num_sweeps, num_samples_per_sweep)
        The univariate samples of the parameter sweeps. These samples are
        for normalized hypercubes [-1,1]^d.
    """

    def get_gaussian_sweep_bounds(W):
        z = sweep_radius  # number of standard deviations
        return np.asarray([-z, z])

    num_vars = mean.shape[0]
    samples, active_samples, W = get_parameter_sweeps_samples(
        num_vars, get_gaussian_sweep_bounds, num_samples_per_sweep, num_sweeps,
        random_samples=False, sweep_rotation_matrix=sweep_rotation_matrix)

    if covariance is not None:
        assert covariance_sqrt is None
        covariance_chol_factor = np.linalg.cholesky(covariance)
    else:
        covariance_chol_factor = None

    samples = map_from_canonical_gaussian(
        samples, mean, covariance_chol_factor=covariance_chol_factor,
        covariance_sqrt=covariance_sqrt)

    return samples, active_samples, W


def plot_parameter_sweep_single_qoi(active_samples, vals, num_sweeps,
                                    num_samples_per_sweep, label_opts=dict(),
                                    axs=plt, markers='o', colors=None,
                                    alpha=1):
    if type(markers) == str:
        markers = [markers]*num_sweeps
    assert len(markers) == num_sweeps

    if type(colors) == str or colors is None:
        colors = [colors]*num_sweeps
    assert len(colors) == num_sweeps

    for i in range(num_sweeps):
        # scale y to [-1,1]
        lb = active_samples[i, :].min()
        ub = active_samples[i, :].max()
        y = (active_samples[i, :]-lb)/(ub-lb)*2-1.
        sweep_label = label_opts.get('sweep_label', r'$\mathrm{Sweep}$')
        axs.plot(y, vals[i*num_samples_per_sweep:(i+1)*num_samples_per_sweep],
                 markers[i], lw=2, label=sweep_label+r' $%d$' % i,
                 color=colors[i], alpha=alpha)
    try:
        if 'title' in label_opts:
            axs.title(label_opts['title'])
        xlabel = label_opts.get('xlabel',  r"$\mathrm{Sweep\;variable}$")
        if xlabel is not None:
            axs.set_xlabel(xlabel)
        axs.ylabel(label_opts.get('ylabel', r"$\mathrm{Function\;value}$"))
    except:
        # needed if axs is of type  subplot
        if 'title' in label_opts:
            axs.set_title(label_opts['title'])
        xlabel = label_opts.get('xlabel', r"$\mathrm{Sweep\;variable}$")
        if xlabel is not None:
            axs.set_xlabel(xlabel)
        axs.set_ylabel(label_opts.get('ylabel', r"$\mathrm{Function\;value}$"))
    axs.legend(loc=0, numpoints=1)


def plot_parameter_sweeps(active_samples, vals, fig_basename=None,
                          qoi_indices=None, show=False, axs=None,
                          axes_label_opts=None, markers='o', colors=None,
                          alpha=1):
    num_sweeps, num_samples_per_sweep = active_samples.shape
    assert vals.shape[0] == num_sweeps*num_samples_per_sweep
    if qoi_indices is None:
        qoi_indices = np.arange(vals.shape[1])
    assert np.all(qoi_indices < vals.shape[1])

    if axs is None:
        fig, axs = plt.subplots(1, len(qoi_indices),
                                figsize=(8*len(qoi_indices), 6), sharey=True)
        if len(qoi_indices) == 1:
            axs = [axs]

    if axes_label_opts is None:
        axes_label_opts = [dict() for ii in range(len(qoi_indices))]

    if type(axes_label_opts) == dict:
        axes_label_opts = [axes_label_opts for ii in range(len(qoi_indices))]

    for j in range(len(qoi_indices)):
        label_opts = axes_label_opts[j]
        if 'title' not in label_opts:
            label_opts['title'] = r"$\mathrm{QoI\;%d}$" % qoi_indices[j]
        plot_parameter_sweep_single_qoi(
            active_samples, vals[:, j], num_sweeps, num_samples_per_sweep,
            label_opts, axs[j], markers, colors, alpha)
    if fig_basename is not None:
        plt.savefig(fig_basename+'.pdf', bbox_inches='tight')
    if show:
        plt.show()
    return axs


def generate_parameter_sweeps_samples(opts, sweep_type, num_samples_per_sweep,
                                      num_sweeps, sweep_rotation_matrix):
    if sweep_type == 'hypercube':
        ranges = opts['ranges']
        samples, active_samples, W = \
            get_hypercube_parameter_sweeps_samples(
                ranges, num_samples_per_sweep, num_sweeps,
                sweep_rotation_matrix)
        # print(samples.T)
    elif sweep_type == 'gaussian':
        mean = opts['mean']
        covariance = opts.get('covariance', None)
        covariance_sqrt = opts.get('covariance_sqrt', None)
        sweep_radius = opts['sweep_radius']
        samples, active_samples, W = get_gaussian_parameter_sweeps_samples(
            mean, covariance=covariance, covariance_sqrt=covariance_sqrt,
            sweep_radius=sweep_radius,
            num_samples_per_sweep=num_samples_per_sweep,
            num_sweeps=num_sweeps,
            sweep_rotation_matrix=sweep_rotation_matrix)
    else:
        raise Exception('incorrect sweep_type : %s' % sweep_type)
    return samples, active_samples, W


def generate_parameter_sweeps(opts, sweep_type, num_samples_per_sweep,
                              num_sweeps, sweep_rotation_matrix, model):
    samples, active_samples, W = generate_parameter_sweeps_samples(
        opts, sweep_type, num_samples_per_sweep, num_sweeps,
        sweep_rotation_matrix)
    vals = model(samples)
    assert vals.ndim == 2
    assert vals.shape[0] == samples.shape[1]
    return samples, active_samples, W, vals


def generate_parameter_sweeps_and_plot(
        model, opts, filename, sweep_type, num_samples_per_sweep=50,
        num_sweeps=2, qoi_indices=None, show=False,
        sweep_rotation_matrix=None, axes_label_opts=None, axs=None):

    if filename is None or not os.path.exists(filename):
        samples, active_samples, W, vals = generate_parameter_sweeps(
            opts, sweep_type, num_samples_per_sweep, num_sweeps,
            sweep_rotation_matrix, model)

        if filename is not None:
            path = os.path.split(filename)[0]
            if len(path) > 0 and not os.path.exists(path):
                os.makedirs(path)
            np.savez(filename, samples=samples, vals=vals,
                     active_samples=active_samples)
    else:
        data = np.load(filename)
        samples = data['samples']
        vals = data['vals']
        active_samples = data['active_samples']

    if filename is not None:
        if filename[-4] == '.':
            figbasename = filename[:-4]
        else:
            figbasename = filename
    else:
        figbasename = None

    return plot_parameter_sweeps(
        active_samples, vals, figbasename, qoi_indices,
        show, axes_label_opts=axes_label_opts, axs=axs)


def generate_parameter_sweeps_and_plot_from_variable(
        model, variable, filename=None, num_samples_per_sweep=50,
        num_sweeps=2, qoi_indices=None, show=False,
        axes_label_opts=None, axs=None):
    """
    Plot parameter sweeps of a function.

    Parameters
    ----------
    model : callable
        Function with the signature

        `model(samples) -> np.ndarray(nsamples, nqoi)`

        where samples : np.ndarray (nvars, nsamples)

    variable : :class:`pyapprox.variables.IndependentMarginalsVariable`
        Random variable

    filename : string
        Name of file to store parameter sweeps. If None no file is written

    num_samples_per_sweep : interger
        The number of samples in a parameter sweep

    num_samples : integer
        The number of parameter sweeps

    qoi_indices : iterable
        The column indices in the values outputed by model which will be
        plotted. A separate subplot will be used for each QoI

    show : boolean
        True - plt.show() is called

    axes_label_opts : list of dict
        Dictionary specifying plt kwargs for each axes

    axs : list of :class:`matplotlib.pyplot.axes`
        If provided will be used to plot each QoI
        Otherwise axes will be created
    """
    opts = {"ranges": get_truncated_ranges(variable)}
    return generate_parameter_sweeps_and_plot(
        model, opts, filename, "hypercube", num_samples_per_sweep,
        num_sweeps, qoi_indices, show, None, axes_label_opts, axs)
