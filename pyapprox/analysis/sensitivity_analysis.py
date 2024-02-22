import numpy as np
from scipy.optimize import OptimizeResult
from scipy.spatial.distance import cdist
from itertools import combinations
from functools import partial

from pyapprox.surrogates.interp.indexing import (
    hash_array, argsort_indices_leixographically, compute_hyperbolic_indices
)
from pyapprox.util.utilities import nchoosek
from pyapprox.expdesign.low_discrepancy_sequences import (
    sobol_sequence, halton_sequence
)
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    _compute_expected_sobol_indices, generate_gp_realizations,
    extract_gaussian_process_attributes_for_integration, GaussianProcess
)
from pyapprox.surrogates.polychaos.gpc import (
    define_poly_options_from_variable_transformation, PolynomialChaosExpansion
)
from pyapprox.surrogates.polychaos.sparse_grid_to_gpc import (
    convert_sparse_grid_to_polynomial_chaos_expansion
)


def get_main_and_total_effect_indices_from_pce(coefficients, indices):
    r"""
    Assume basis is orthonormal
    Assume first coefficient is the coefficient of the constant basis. Remove
    this assumption by extracting array index of constant term from indices

    Returns
    -------
    main_effects : np.ndarray(num_vars)
        Contribution to variance of each variable acting alone

    total_effects : np.ndarray(num_vars)
        Contribution to variance of each variable acting alone or with
        other variables

    """
    num_vars = indices.shape[0]
    num_terms, num_qoi = coefficients.shape
    assert num_terms == indices.shape[1]

    main_effects = np.zeros((num_vars, num_qoi), np.double)
    total_effects = np.zeros((num_vars, num_qoi), np.double)
    variance = np.zeros(num_qoi)

    for ii in range(num_terms):
        index = indices[:, ii]

        # calculate contribution to variance of the index
        var_contribution = coefficients[ii, :]**2

        # get number of dimensions involved in interaction, also known
        # as order
        non_constant_vars = np.where(index > 0)[0]
        order = non_constant_vars.shape[0]

        if order > 0:
            variance += var_contribution

        # update main effects
        if (order == 1):
            var = non_constant_vars[0]
            main_effects[var, :] += var_contribution

        # update total effects
        for ii in range(order):
            var = non_constant_vars[ii]
            total_effects[var, :] += var_contribution

    assert np.all(np.isfinite(variance))
    assert np.all(variance > 0)
    main_effects /= variance
    total_effects /= variance
    return main_effects, total_effects


def get_sobol_indices(coefficients, indices, max_order=2):
    num_terms, num_qoi = coefficients.shape
    variance = np.zeros(num_qoi)
    assert num_terms == indices.shape[1]
    interactions = dict()
    interaction_values = []
    interaction_terms = []
    kk = 0
    for ii in range(num_terms):
        index = indices[:, ii]
        var_contribution = coefficients[ii, :]**2
        non_constant_vars = np.where(index > 0)[0]
        key = hash_array(non_constant_vars)

        if len(non_constant_vars) > 0:
            variance += var_contribution

        if len(non_constant_vars) > 0 and len(non_constant_vars) <= max_order:
            if key in interactions:
                interaction_values[interactions[key]] += var_contribution
            else:
                interactions[key] = kk
                interaction_values.append(var_contribution)
                interaction_terms.append(non_constant_vars)
                kk += 1

    # interaction_terms = np.asarray(interaction_terms).T
    interaction_values = np.asarray(interaction_values)

    return interaction_terms, interaction_values/variance


def plot_main_effects(main_effects, ax, truncation_pct=0.95,
                      max_slices=5, rv='z', qoi=0):
    r"""
    Plot the main effects in a pie chart showing relative size.

    Parameters
    ----------
    main_effects : np.ndarray (nvars,nqoi)
        The variance based main effect sensitivity indices

    ax : :class:`matplotlib.pyplot.axes.Axes`
        Axes that will be used for plotting

    truncation_pct : float
        The proportion :math:`0<p\le 1` of the sensitivity indices
        effects to plot

    max_slices : integer
        The maximum number of slices in the pie-chart. Will only
        be active if the turncation_pct gives more than max_slices

    rv : string
        The name of the random variables when creating labels

    qoi : integer
        The index 0<qoi<nqoi of the quantitiy of interest to plot
    """
    main_effects = main_effects[:, qoi]
    assert main_effects.sum() <= 1.+np.finfo(float).eps
    main_effects_sum = main_effects.sum()

    # sort main_effects in descending order
    II = np.argsort(main_effects)[::-1]
    main_effects = main_effects[II]

    labels = []
    partial_sum = 0.
    for i in range(II.shape[0]):
        if partial_sum/main_effects_sum < truncation_pct and i < max_slices:
            labels.append('$%s_{%d}$' % (rv, II[i]+1))
            partial_sum += main_effects[i]
        else:
            break

    main_effects.resize(i + 1)
    if abs(partial_sum - main_effects_sum) > 0.5:
        explode = np.zeros(main_effects.shape[0])
        labels.append(r'$\mathrm{other}$')
        main_effects[-1] = main_effects_sum - partial_sum
        explode[-1] = 0.1
    else:
        main_effects.resize(i)
        explode = np.zeros(main_effects.shape[0])

    p = ax.pie(main_effects, labels=labels, autopct='%1.1f%%',
               shadow=True, explode=explode)
    return p


def plot_sensitivity_indices_with_confidence_intervals(
        labels, ax, sa_indices_median, sa_indices_q1, sa_indices_q3,
        sa_indices_min, sa_indices_max, reference_values=None, fliers=None):
    nindices = len(sa_indices_median)
    assert len(labels) == nindices
    if reference_values is not None:
        assert len(reference_values) == nindices
    stats = [dict() for nn in range(nindices)]
    for nn in range(nindices):
        # use boxplot stats mean entry to store reference values.
        if reference_values is not None:
            stats[nn]['mean'] = reference_values[nn]
        stats[nn]['med'] = sa_indices_median[nn]
        stats[nn]['q1'] = sa_indices_q1[nn]
        stats[nn]['q3'] = sa_indices_q3[nn]
        stats[nn]['label'] = labels[nn]
        # use whiskers for min and max instead of fliers
        stats[nn]['whislo'] = sa_indices_min[nn]
        stats[nn]['whishi'] = sa_indices_max[nn]
        if fliers is not None:
            stats[nn]['fliers'] = fliers[nn]

    if reference_values is not None:
        showmeans = True
    else:
        showmeans = False

    if fliers is not None:
        showfliers = True
    else:
        showfliers = False

    bp = ax.bxp(stats, showfliers=showfliers, showmeans=showmeans,
                patch_artist=True,
                meanprops=dict(marker='o', markerfacecolor='blue',
                               markeredgecolor='blue', markersize=12),
                medianprops=dict(color='red'))
    ax.tick_params(axis='x', labelrotation=45)

    colors = ['gray']*nindices
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    colors = ['red']*nindices
    return bp


def plot_total_effects(total_effects, ax, truncation_pct=0.95,
                       rv='z', qoi=0):
    r"""
    Plot the total effects in a bar chart showing relative size.

    Parameters
    ----------
    total_effects : np.ndarray (nvars,nqoi)
        The variance based total effect sensitivity indices

    ax : :class:`matplotlib.pyplot.axes.Axes`
        Axes that will be used for plotting

    truncation_pct : float
        The proportion :math:`0<p\le 1` of the sensitivity indices
        effects to plot

    rv : string
        The name of the random variables when creating labels

    qoi : integer
        The index 0<qoi<nqoi of the quantitiy of interest to plot
    """

    total_effects = total_effects[:, qoi]

    width = .95
    locations = np.arange(total_effects.shape[0])
    p = ax.bar(locations-width/2, total_effects, width, align='edge')
    labels = ['$%s_{%d}$' % (rv, ii+1) for ii in range(total_effects.shape[0])]
    ax.set_xticks(locations)
    ax.set_xticklabels(labels, rotation=0)
    return p


def plot_interaction_values(interaction_values, interaction_terms, ax,
                            truncation_pct=0.95, max_slices=5, rv='z', qoi=0):
    r"""
    Plot sobol indices in a pie chart showing relative size.

    Parameters
    ----------
    interaction_values : np.ndarray (nvars,nqoi)
        The variance based Sobol indices

    interaction_terms : nlist (nchoosek(nvars+max_order,nvars))
        Indices np.ndarrays of varying size specifying the variables in each
        interaction in ``interaction_indices``

    ax : :class:`matplotlib.pyplot.axes.Axes`
        Axes that will be used for plotting

    truncation_pct : float
        The proportion :math:`0<p\le 1` of the sensitivity indices
        effects to plot

    max_slices : integer
        The maximum number of slices in the pie-chart. Will only
        be active if the turncation_pct gives more than max_slices

    rv : string
        The name of the random variables when creating labels

    qoi : integer
        The index 0<qoi<nqoi of the quantitiy of interest to plot
    """

    assert interaction_values.shape[0] == len(interaction_terms)
    interaction_values = interaction_values[:, qoi]

    II = np.argsort(interaction_values)[::-1]
    interaction_values = interaction_values[II]
    interaction_terms = [interaction_terms[ii] for ii in II]

    labels = []
    partial_sum = 0.
    for i in range(interaction_values.shape[0]):
        if partial_sum < truncation_pct and i < max_slices:
            label = '($'
            for j in range(len(interaction_terms[i])-1):
                label += '%s_{%d},' % (rv, interaction_terms[i][j]+1)
            label += '%s_{%d}$)' % (rv, interaction_terms[i][-1]+1)
            labels.append(label)
            partial_sum += interaction_values[i]
        else:
            break

    interaction_values = interaction_values[:i]
    if abs(partial_sum - 1.) > 10 * np.finfo(np.double).eps:
        labels.append(r'$\mathrm{other}$')
        interaction_values = np.concatenate(
            [interaction_values, [1.-partial_sum]])

    explode = np.zeros(interaction_values.shape[0])
    explode[-1] = 0.1
    assert interaction_values.shape[0] == len(labels)
    p = ax.pie(interaction_values, labels=labels, autopct='%1.1f%%',
               shadow=True, explode=explode)
    return p


def get_morris_trajectory(nvars, nlevels, eps=0):
    r"""
    Compute a morris trajectory used to compute elementary effects

    Parameters
    ----------
    nvars : integer
        The number of variables

    nlevels : integer
        The number of levels used for to define the morris grid.

    eps : float
        Set grid used defining the morris trajectory to [eps,1-eps].
        This is needed when mapping the morris trajectories using inverse
        CDFs of unbounded variables

    Returns
    -------
    trajectory : np.ndarray (nvars,nvars+1)
        The Morris trajectory which consists of nvars+1 samples
    """
    assert nlevels % 2 == 0
    delta = nlevels/((nlevels-1)*2)
    samples_1d = np.linspace(eps, 1-eps, nlevels)

    initial_point = np.random.choice(samples_1d, nvars)
    shifts = np.diag(np.random.choice([-delta, delta], nvars))
    trajectory = np.empty((nvars, nvars+1))
    trajectory[:, 0] = initial_point
    for ii in range(nvars):
        trajectory[:, ii+1] = trajectory[:, ii].copy()
        if (trajectory[ii, ii]-delta) >= 0 and (trajectory[ii, ii]+delta) <= 1:
            trajectory[ii, ii+1] += shifts[ii]
        elif (trajectory[ii, ii]-delta) >= 0:
            trajectory[ii, ii+1] -= delta
        elif (trajectory[ii, ii]+delta) <= 1:
            trajectory[ii, ii+1] += delta
        else:
            raise Exception('This should not happen')
    return trajectory


def get_morris_samples(nvars, nlevels, ntrajectories, eps=0, icdfs=None):
    r"""
    Compute a set of Morris trajectories used to compute elementary effects

    Notes
    -----
    The choice of nlevels must be linked to the choice of ntrajectories.
    For example, if a large number of possible levels is used ntrajectories
    must also be high, otherwise if ntrajectories is small effort will be
    wasted because many levels will not be explored. nlevels=4 and
    ntrajectories=10 is often considered reasonable.

    Parameters
    ----------
    nvars : integer
        The number of variables

    nlevels : integer
        The number of levels used for to define the morris grid.

    ntrajectories : integer
        The number of Morris trajectories requested

    eps : float
        Set grid used defining the Morris trajectory to [eps,1-eps].
        This is needed when mapping the morris trajectories using inverse
        CDFs of unbounded variables

    icdfs : list (nvars)
        List of inverse CDFs functions for each variable

    Returns
    -------
    trajectories : np.ndarray (nvars,ntrajectories*(nvars+1))
        The Morris trajectories
    """
    if icdfs is None:
        icdfs = [lambda x: x]*nvars
    assert len(icdfs) == nvars

    trajectories = np.hstack([get_morris_trajectory(nvars, nlevels, eps)
                              for n in range(ntrajectories)])
    for ii in range(nvars):
        trajectories[ii, :] = icdfs[ii](trajectories[ii, :])
    return trajectories


def get_morris_elementary_effects(samples, values):
    r"""
    Get the Morris elementary effects from a set of trajectories.

    Parameters
    ----------
    samples : np.ndarray (nvars,ntrajectories*(nvars+1))
        The morris trajectories

    values : np.ndarray (ntrajectories*(nvars+1),nqoi)
        The values of the vecto-valued target function with nqoi quantities
        of interest (QoI)

    Returns
    -------
    elem_effects : np.ndarray(nvars,ntrajectories,nqoi)
        The elementary effects of each variable for each trajectory and QoI
    """
    nvars = samples.shape[0]
    nqoi = values.shape[1]
    assert samples.shape[1] % (nvars+1) == 0
    assert samples.shape[1] == values.shape[0]
    ntrajectories = samples.shape[1]//(nvars+1)
    elem_effects = np.empty((nvars, ntrajectories, nqoi))
    ix1 = 0
    for ii in range(ntrajectories):
        ix2 = ix1+nvars
        delta = np.diff(samples[:, ix1+1:ix2+1]-samples[:, ix1:ix2]).max()
        assert delta > 0
        elem_effects[:, ii] = (values[ix1+1:ix2+1]-values[ix1:ix2])/delta
        ix1 = ix2+1
    return elem_effects


def get_morris_sensitivity_indices(elem_effects):
    r"""
    Compute the Morris sensitivity indices mu and sigma from the elementary
    effects computed for a set of trajectories.

    Mu is the mu^\star from Campolongo et al.

    Parameters
    ----------
    elem_effects : np.ndarray(nvars,ntrajectories,nqoi)
        The elementary effects of each variable for each trajectory and
        quantity of interest (QoI)

    Returns
    -------
    mu : np.ndarray (nvars, nqoi)
        The sensitivity of each output to each input. Larger mu corresponds to
        higher sensitivity

    sigma: np.ndarray(nvars, nqoi)
        A measure of the non-linearity and/or interaction effects of each input
        for each output. Low values suggest a linear realationship between
        the input and output. Larger values suggest a that the output is
        nonlinearly dependent on the input and/or the input interacts with
        other inputs
    """
    mu = np.absolute(elem_effects).mean(axis=1)
    assert mu.shape == (elem_effects.shape[0], elem_effects.shape[2])
    sigma = np.std(elem_effects, axis=1)
    return mu, sigma


def print_morris_sensitivity_indices(mu, sigma, qoi=0):
    str_format = "{:<3} {:>10} {:>10}"
    print(str_format.format(" ", "mu*", "sigma"))
    str_format = "{:<3} {:10.5f} {:10.5f}"
    for ii in range(mu.shape[0]):
        print(str_format.format(f'Z_{ii+1}', mu[ii, qoi], sigma[ii, qoi]))


def downselect_morris_trajectories(samples, ntrajectories):
    nvars = samples.shape[0]
    assert samples.shape[1] % (nvars+1) == 0
    ncandidate_trajectories = samples.shape[1]//(nvars+1)
    # assert 10*ntrajectories<=ncandidate_trajectories

    trajectories = np.reshape(
        samples, (nvars, nvars+1, ncandidate_trajectories), order='F')

    distances = np.zeros((ncandidate_trajectories, ncandidate_trajectories))
    for ii in range(ncandidate_trajectories):
        for jj in range(ii+1):
            distances[ii, jj] = cdist(
                trajectories[:, :, ii].T, trajectories[:, :, jj].T).sum()
            distances[jj, ii] = distances[ii, jj]

    get_combinations = combinations(
        np.arange(ncandidate_trajectories), ntrajectories)
    ncombinations = nchoosek(ncandidate_trajectories, ntrajectories)
    print('ncombinations', ncombinations)
    # values = np.empty(ncombinations)
    best_index = None
    best_value = -np.inf
    for ii, index in enumerate(get_combinations):
        value = np.sqrt(np.sum(
            [distances[ix[0], ix[1]]**2 for ix in combinations(index, 2)]))
        if value > best_value:
            best_value = value
            best_index = index

    samples = trajectories[:, :, best_index].reshape(
        nvars, ntrajectories*(nvars+1), order='F')
    return samples


class SensitivityResult(OptimizeResult):
    pass


def morris_sensitivities(fun, variable, ntrajectories,
                         nlevels=4):
    r"""
    Compute sensitivity indices by constructing an adaptive polynomial chaos
    expansion.

    Parameters
    ----------
    fun : callable
        The function being analyzed

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

    variable : :py:class:`pyapprox.variables.IndependentMarginalsVariable`
         Object containing information of the joint density of the inputs z
         which is the tensor product of independent and identically distributed
         uniform variables.

    ntrajectories : integer
        The number of Morris trajectories requested


    nlevels : integer
        The number of levels used for to define the morris grid.

    Returns
    -------
    result : :class:`pyapprox.analysis.sensitivity_analysis.SensitivityResult`
         Result object with the following attributes

    mu : np.ndarray (nvars,nqoi)
        The sensitivity of each output to each input. Larger mu corresponds to
        higher sensitivity

    sigma: np.ndarray (nvars,nqoi)
        A measure of the non-linearity and/or interaction effects of each input
        for each output. Low values suggest a linear realationship between
        the input and output. Larger values suggest a that the output is
        nonlinearly dependent on the input and/or the input interacts with
        other inputs

    samples : np.ndarray(nvars,ntrajectories*(nvars+1))
        The coordinates of each morris trajectory

    values : np.ndarray(nvars,nqoi)
        The values of ``fun`` at each sample in ``samples``
    """

    nvars = variable.num_vars()
    icdfs = [v.ppf for v in variable.marginals()]
    samples = get_morris_samples(nvars, nlevels, ntrajectories, icdfs=icdfs)
    values = fun(samples)
    elem_effects = get_morris_elementary_effects(samples, values)
    mu, sigma = get_morris_sensitivity_indices(elem_effects)

    return SensitivityResult(
        {'morris_mu': mu,
         'morris_sigma': sigma,
         'samples': samples, 'values': values})


def sparse_grid_sobol_sensitivities(sparse_grid, max_order=2):
    r"""
    Compute sensitivity indices from a sparse grid
    by converting it to a polynomial chaos expansion

    Parameters
    ----------
    sparse_grid :class:`pyapprox.adaptive_sparse_grid:CombinationSparseGrid`
       The sparse grid

    max_order : integer
        The maximum interaction order of Sobol indices to compute. A value
        of 2 will compute all pairwise interactions, a value of 3 will
        compute indices for all interactions involving 3 variables. The number
        of indices returned will be nchoosek(nvars+max_order,nvars). Warning
        when nvars is high the number of indices will increase rapidly with
        max_order.

    Returns
    -------
    result : :class:`pyapprox.analysis.sensitivity_analysis.SensitivityResult`
         Result object with the following attributes

    main_effects : np.ndarray (nvars)
        The variance based main effect sensitivity indices

    total_effects : np.ndarray (nvars)
        The variance based total effect sensitivity indices

    sobol_indices : np.ndarray (nchoosek(nvars+max_order,nvars),nqoi)
        The variance based Sobol sensitivity indices

    sobol_interaction_indices : np.ndarray(nvars,nchoosek(nvars+max_order,nvars))
        Indices specifying the variables in each interaction in
        ``sobol_indices``

    pce : :class:`multivariate_polynomials.PolynomialChaosExpansion`
       The pce respresentation of the sparse grid ``approx``
    """
    pce_opts = define_poly_options_from_variable_transformation(
        sparse_grid.var_trans)
    pce = convert_sparse_grid_to_polynomial_chaos_expansion(
        sparse_grid, pce_opts)
    pce_main_effects, pce_total_effects =\
        get_main_and_total_effect_indices_from_pce(
            pce.get_coefficients(), pce.get_indices())

    interaction_terms, pce_sobol_indices = get_sobol_indices(
        pce.get_coefficients(), pce.get_indices(), max_order=max_order)

    return SensitivityResult(
        {'main_effects': pce_main_effects,
         'total_effects': pce_total_effects,
         'sobol_indices': pce_sobol_indices,
         'sobol_interaction_indices': interaction_terms,
         'pce': pce})


def gpc_sobol_sensitivities(pce, variable, max_order=2):
    r"""
    Compute variance based sensitivity metrics from a polynomial chaos
    expansion

    Parameters
    ----------
    pce :class:`pyapprox.surrogates.polychaos.gpc.PolynomialChaosExpansion`
       The polynomial chaos expansion

    max_order : integer
        The maximum interaction order of Sobol indices to compute. A value
        of 2 will compute all pairwise interactions, a value of 3 will
        compute indices for all interactions involving 3 variables. The number
        of indices returned will be nchoosek(nvars+max_order,nvars). Warning
        when nvars is high the number of indices will increase rapidly with
        max_order.

    Returns
    -------
    result : :class:`pyapprox.analysis.sensitivity_analysis.SensitivityResult`
         Result object with the following attributes

    main_effects : np.ndarray (nvars, nqoi)
        The variance based main effect sensitivity indices

    total_effects : np.ndarray (nvars, nqoi)
        The variance based total effect sensitivity indices

    sobol_indices : np.ndarray (nchoosek(nvars+max_order,nvars), nqoi)
        The variance based Sobol sensitivity indices

    sobol_interaction_indices : np.ndarray(nvars, nchoosek(nvars+max_order, nvars))
        Indices specifying the variables in each interaction in
        ``sobol_indices``
    """
    if not issubclass(pce.__class__, PolynomialChaosExpansion):
        raise ValueError("Must provide a PCE")

    if variable.marginals() != pce.var_trans.variable.marginals():
        msg = "variable is inconsistent with PCE. "
        msg += "Can only compute sensitivities with respect to variable "
        msg += "used to build the PCE"
        raise ValueError(msg)

    pce_main_effects, pce_total_effects =\
        get_main_and_total_effect_indices_from_pce(
            pce.get_coefficients(), pce.get_indices())

    interaction_terms, pce_sobol_indices = get_sobol_indices(
        pce.get_coefficients(), pce.get_indices(),
        max_order=max_order)

    return SensitivityResult(
        {'main_effects': pce_main_effects,
         'total_effects': pce_total_effects,
         'sobol_indices': pce_sobol_indices,
         'sobol_interaction_indices': interaction_terms})


def generate_sobol_index_sample_sets(samplesA, samplesB, index):
    """
    Given two sample sets A and B generate the sets :math:`A_B^{I}` from

    The rows of A_B^I are all from A except for the rows with non zero entries
    in the index I. When A and B are QMC samples it is best to change as few
    rows as possible

    See

    Variance based sensitivity analysis of model output. Design and estimator
    for the total sensitivity index
    """
    nvars = samplesA.shape[0]
    II = np.arange(nvars)
    mask = np.asarray(index, dtype=bool)
    samples = np.vstack([samplesA[~mask], samplesB[mask]])
    JJ = np.hstack([II[~mask], II[mask]])
    samples = samples[np.argsort(JJ), :]
    return samples


def get_AB_sample_sets_for_sobol_sensitivity_analysis(
        variables, nsamples, method, qmc_start_index=0):
    if method == 'random':
        samplesA = generate_independent_random_samples(variables, nsamples)
        samplesB = generate_independent_random_samples(variables, nsamples)
    elif method == 'halton' or 'sobol':
        nvars = variables.num_vars()
        if method == 'halton':
            qmc_samples = halton_sequence(
                2*nvars, nsamples, qmc_start_index)
        else:
            qmc_samples = sobol_sequence(2*nvars, nsamples, qmc_start_index)
        samplesA = qmc_samples[:nvars, :]
        samplesB = qmc_samples[nvars:, :]
        for ii, rv in enumerate(variables.marginals()):
            lb, ub = rv.interval(1)
            # transformation is undefined at [0,1] for unbouned
            # random variables
            # create bounds for unbounded interval that exclude 1e-8
            # of the total probability
            t1, t2 = rv.interval(1-1e-8)
            nlb, nub = rv.cdf([t1, t2])
            if not np.isfinite(lb):
                samplesA[ii, samplesA[ii, :] == 0] = nlb
                samplesB[ii, samplesB[ii, :] == 0] = nlb
            if not np.isfinite(ub):
                samplesA[ii, samplesA[ii, :] == 1] = nub
                samplesB[ii, samplesB[ii, :] == 1] = nub
            samplesA[ii, :] = rv.ppf(samplesA[ii, :])
            samplesB[ii, :] = rv.ppf(samplesB[ii, :])
    else:
        raise Exception(f'Sampling method {method} not supported')
    return samplesA, samplesB


def sampling_based_sobol_indices(
        fun, variables, interaction_terms, nsamples, sampling_method='sobol',
        qmc_start_index=0):
    """
    See I.M. Sobol. Mathematics and Computers in Simulation 55 (2001) 271–280

    and

    Saltelli, Annoni et. al, Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index. 2010.
    https://doi.org/10.1016/j.cpc.2009.09.018

    Parameters
    ----------
    interaction_terms : np.ndarray (nvars, nterms)
        Index defining the active terms in each interaction. If the
        ith  variable is active interaction_terms[i] == 1 and zero otherwise
        This index must be downward closed due to way sobol indices are
        computed
    """
    nvars = interaction_terms.shape[0]
    nterms = interaction_terms.shape[1]
    samplesA, samplesB = get_AB_sample_sets_for_sobol_sensitivity_analysis(
        variables, nsamples, sampling_method, qmc_start_index)
    assert nvars == samplesA.shape[0]
    valuesA = fun(samplesA)
    valuesB = fun(samplesB)
    mean = valuesA.mean(axis=0)
    variance = valuesA.var(axis=0)
    interaction_values = np.empty((nterms, valuesA.shape[1]))
    total_effect_values = [None for ii in range(nvars)]
    interaction_values_dict = dict()
    for ii in range(nterms):
        index = interaction_terms[:, ii]
        assert index.sum() > 0
        samplesAB = generate_sobol_index_sample_sets(
            samplesA, samplesB, index)
        valuesAB = fun(samplesAB)
        # entry b in Table 2 of Saltelli, Annoni et. al
        interaction_values[ii, :] = \
            (valuesB*(valuesAB-valuesA)).mean(axis=0)/variance
        interaction_values_dict[tuple(np.where(index > 0)[0])] = ii
        if index.sum() == 1:
            dd = np.where(index == 1)[0][0]
            # entry f in Table 2 of Saltelli, Annoni et. al
            total_effect_values[dd] = 0.5 * \
                np.mean((valuesA-valuesAB)**2, axis=0)/variance

    # must substract of contributions from lower-dimensional terms from
    # each interaction value For example, let R_ij be interaction_values
    # the sobol index S_ij satisfies R_ij = S_i + S_j + S_ij
    II = argsort_indices_leixographically(interaction_terms)
    sobol_indices = interaction_values.copy()
    sobol_indices_dict = dict()
    for ii in range(II.shape[0]):
        index = interaction_terms[:, II[ii]]
        active_vars = np.where(index > 0)[0]
        nactive_vars = index.sum()
        sobol_indices_dict[tuple(active_vars)] = II[ii]
        if nactive_vars > 1:
            for jj in range(nactive_vars-1):
                indices = combinations(active_vars, jj+1)
                for key in indices:
                    sobol_indices[II[ii]] -= \
                        sobol_indices[sobol_indices_dict[key]]

    total_effect_values = np.asarray(total_effect_values)
    assert np.all(variance >= 0)
    # main_effects = sobol_indices[interaction_terms.sum(axis=0)==1, :]
    # We cannot guarantee that the main_effects will be <= 1. Because
    # variance and each interaction_index are computed with different sample
    # sets. Consider function of two variables which is constant in one
    # variable
    # then interaction_index[0] should equal variance. But with different
    # sample
    # sets interaction_index could be smaller or larger than the variance.
    # assert np.all(main_effects<=1)
    # Similarly we cannot even guarantee main effects will be non-negative
    # assert np.all(main_effects>=0)
    # We also cannot guarantee that the sobol indices will be non-negative.
    # assert np.all(total_effect_values>=0)
    # assert np.all(sobol_indices>=0)
    return sobol_indices, total_effect_values, variance, mean


def _repeat_sampling_based_sobol_indices(
        fun, variable, interaction_terms=None,
        nsamples=1000,
        sampling_method="random",
        nsobol_realizations=10,
        qmc_start_index=1):
    if interaction_terms is None:
        interaction_terms = get_isotropic_anova_indices(
            variable.num_vars(), 2)

    means, variances, sobol_values,  total_values = [], [], [], []
    # qmc_start_index = 0
    for ii in range(nsobol_realizations):
        sv, tv, vr, me = sampling_based_sobol_indices(
            fun, variable, interaction_terms, nsamples,
            sampling_method='sobol', qmc_start_index=qmc_start_index)
        means.append(me)
        variances.append(vr)
        sobol_values.append(sv)
        total_values.append(tv)
        qmc_start_index += nsamples
    means = np.asarray(means)
    variances = np.asarray(variances)
    sobol_values = np.asarray(sobol_values)
    total_values = np.asarray(total_values)

    interaction_terms = [
        np.where(index > 0)[0]
        for index in interaction_terms.T]
    return sobol_values, total_values, variances, means


def repeat_sampling_based_sobol_indices(
        fun, variable, interaction_terms=None,
        nsamples=1000,
        sampling_method="random",
        nsobol_realizations=10,
        summary_stats=["mean", "median", "min", "max", "quantile-0.25",
                       "quantile-0.75"],
        qmc_start_index=1):
    """
    Compute sobol indices for different sample sets. This allows estimation
    of error due to finite sample sizes. This function requires evaluting
    the function at nsobol_realizations * N, where N is the
    number of samples required by sampling_based_sobol_indices. Thus
    This function is useful when applid to a random
    realization of a Gaussian process requires the Cholesky decomposition
    of a nsamples x nsamples matrix which becomes to costly for nsamples >1000
    """
    sobol_values, total_values, variances, means = (
        _repeat_sampling_based_sobol_indices(
            fun, variable, interaction_terms,
            nsamples, sampling_method, nsobol_realizations,
            qmc_start_index))

    stat_functions = _get_stats_functions(summary_stats)
    result = dict()
    result["sobol_interaction_indices"] = interaction_terms
    data = [sobol_values, total_values, variances, means]
    data_names = ['sobol_indices', 'total_effects', 'variance', 'mean']
    for item, name in zip(data, data_names):
        subdict = dict()
        for ii, sfun in enumerate(stat_functions):
            subdict[sfun.__name__] = sfun(item, axis=(0))
        subdict['values'] = item
        result[name] = subdict

    return result


def _get_stats_functions(summary_stats):
    quantile_stats = []
    for q in [0.25, 0.75]:
        sfun = partial(np.quantile, q=q)
        sfun.__name__ = f'quantile-{q}'
        quantile_stats.append(sfun)
    stat_functions_dict = {"mean": np.mean,
                           "median": np.median,
                           "min": np.min,
                           "max": np.max,
                           "std": np.std,
                           "quantile-0.25": quantile_stats[0],
                           "quantile-0.75": quantile_stats[1]}
    stat_functions_dict["min"].__name__ = "amin"
    stat_functions_dict["max"].__name__ = "amax"
    for name in summary_stats:
        if name not in stat_functions_dict:
            msg = f"Summary stats {name} not supported\n"
            msg += f"Select from {list(stat_functions_dict.keys())}"
            raise ValueError(msg)
    stat_functions = [stat_functions_dict[name] for name in summary_stats]
    return stat_functions


def analytic_sobol_indices_from_gaussian_process(
        gp, variable, interaction_terms=None, ngp_realizations=1000,
        summary_stats=["mean", "median", "min", "max", "quantile-0.25",
                       "quantile-0.75"],
        ninterpolation_samples=None, nvalidation_samples=100,
        ncandidate_samples=1000, nquad_samples=50, use_cholesky=True,
        alpha=1e-8, verbosity=0):

    if ninterpolation_samples is None:
        ninterpolation_samples = min(
            5*gp.num_training_samples(), ncandidate_samples)

    if not issubclass(gp.__class__, GaussianProcess):
        raise ValueError("Argument gp must be a Gaussian process")

    stat_functions = _get_stats_functions(summary_stats)

    if interaction_terms is None:
        interaction_terms = get_isotropic_anova_indices(
            variable.num_vars(), 2)

    x_train, y_train, K_inv, lscale, kernel_var, transform_quad_rules = \
        extract_gaussian_process_attributes_for_integration(gp)

    if ngp_realizations > 0:
        gp_realizations = generate_gp_realizations(
            gp, ngp_realizations, ninterpolation_samples, nvalidation_samples,
            ncandidate_samples, variable, use_cholesky, alpha, verbosity)

        # Check how accurate realizations
        validation_samples = generate_independent_random_samples(
            variable, 1000)
        mean_vals, std = gp(validation_samples, return_std=True)
        realization_vals = gp_realizations(validation_samples)
        # print(mean_vals[:, 0].mean())
        # print(std,realization_vals.std(axis=1))
        # this checks the accuracy of the number of realizations.
        # error will decrease with number of samples
        if verbosity > 0:
            print('std of realizations error',
                  np.linalg.norm(std-realization_vals.std(axis=1)) /
                  np.linalg.norm(std))
            print('var of realizations error',
                  np.linalg.norm(std**2-realization_vals.var(axis=1)) /
                  np.linalg.norm(std**2))
            print('mean interpolation error',
                  np.linalg.norm((mean_vals[:, 0]-realization_vals[:, -1])) /
                  np.linalg.norm(mean_vals[:, 0]))

        x_train = gp_realizations.selected_canonical_samples
        # gp_realizations.train_vals is normalized so unnormalize
        y_train = gp._y_train_std*gp_realizations.train_vals
        # kernel_var has already been adjusted by call to
        # extract_gaussian_process_attributes_for_integration
        K_inv = np.linalg.inv(gp_realizations.L.dot(gp_realizations.L.T))
        K_inv /= gp._y_train_std**2

    sobol_values, total_values, means, variances = \
        _compute_expected_sobol_indices(
            gp, variable, interaction_terms, nquad_samples,
            x_train, y_train, K_inv, lscale, kernel_var,
            transform_quad_rules, gp._y_train_mean)
    sobol_values = sobol_values.T
    total_values = total_values.T

    result = dict()
    interaction_terms = [
        np.where(index > 0)[0]
        for index in interaction_terms.T]

    result["sobol_interaction_indices"] = interaction_terms
    data = [sobol_values, total_values, variances, means]
    data_names = ['sobol_indices', 'total_effects', 'variance', 'mean']
    for item, name in zip(data, data_names):
        subdict = dict()
        for ii, sfun in enumerate(stat_functions):
            subdict[sfun.__name__] = sfun(item, axis=(0))
        subdict['values'] = item
        result[name] = subdict
    return result


def sampling_based_sobol_indices_from_gaussian_process(
    gp, variables, interaction_terms, nsamples, sampling_method='sobol',
        ngp_realizations=1, normalize=True, nsobol_realizations=1,
        stat_functions=(np.mean, np.median, np.min, np.max),
        ninterpolation_samples=500, nvalidation_samples=100,
        ncandidate_samples=1000, use_cholesky=True, alpha=0):
    """
    Compute sobol indices from Gaussian process using sampling.
    This function returns the mean and variance of these values with
    respect to the variability in the GP (i.e. its function error)

    Following Kennedy and O'hagan we evaluate random realizations of each
    GP at a discrete set of points. To predict at larger sample sizes we
    interpolate these points and use the resulting approximation to make any
    subsequent predictions. This introduces an error but the error can be
    made arbitrarily small by setting ninterpolation_samples large enough.
    The geometry of the interpolation samples can effect accuracy of the
    interpolants. Consequently we use Pivoted cholesky algorithm in
    Harbrecht et al for choosing the interpolation samples.

    Parameters
    ----------
    ngp_realizations : integer
        The number of random realizations of the Gaussian process
        if ngp_realizations == 0 then the sensitivity indices will
        only be computed using the mean of the GP.

    nsobol_realizations : integer
        The number of random realizations of the random samples used to
        compute the sobol indices. This number should be similar to
        ngp_realizations, as mean and stdev are taken over both these
        random values.

    stat_functions : list
        List of callable functions with signature fun(np.ndarray)
        E.g. np.mean. If fun has arguments then we must wrap then with partial
        and set a meaniningful __name__, e.g. fun = partial(np.quantile, q=0.5)
        fun.__name__ == 'quantile-0.25'.
        Note: np.min and np.min names are amin, amax

    ninterpolation_samples : integer
        The number of samples used to interpolate the discrete random
        realizations of a Gaussian Process

    nvalidation_samples : integer
        The number of samples used to assess the accuracy of the interpolants
        of the random realizations

    ncandidate_samples : integer
        The number of candidate samples selected from when building the
        interpolants of the random realizations

    Returns
    -------
    result : dictionary
        Result containing the numpy functions in stat_funtions applied
        to the mean, variance, sobol_indices and total_effects of the Gaussian
        process. To access the data associated with a fun in stat_function
        use the key fun.__name__, For example  if the stat_function is np.mean
        the mean sobol indices are accessed via result['sobol_indices']['mean']
        The raw values of each iteration are stored in
        result['sobol_indices]['values']
    """
    assert nsobol_realizations > 0

    if ngp_realizations > 0:
        assert ncandidate_samples > ninterpolation_samples
        gp_realizations = generate_gp_realizations(
            gp, ngp_realizations, ninterpolation_samples, nvalidation_samples,
            ncandidate_samples, variables, use_cholesky, alpha)
        fun = gp_realizations
    else:
        fun = gp

    sobol_values, total_values, variances, means = \
        _repeat_sampling_based_sobol_indices(
            fun, variables, interaction_terms, nsamples,
            sampling_method, nsobol_realizations)

    result = dict()
    data = [sobol_values, total_values, variances, means]
    data_names = ['sobol_indices', 'total_effects', 'variance', 'mean']
    for item, name in zip(data, data_names):
        subdict = dict()
        for ii, sfun in enumerate(stat_functions):
            # have to deal with averaging over axis = (0, 1) and axis = (0, 2)
            # for mean, variance and sobol_indices, total_effects respectively
            subdict[sfun.__name__] = sfun(item, axis=(0, -1))
        subdict['values'] = item
        result[name] = subdict
    return result


def _borgonovo_estimation(
        samples, values, marginal_icdfs, nbins=None):

    nvars, nsamples = samples.shape
    nqoi = values.shape[1]
    assert values.shape[0] == nsamples

    if nbins is None:
        nbins = max(2, int(1/3*(nsamples)**(1./3)))
        print('Number of bins', nbins)
    mean = values.mean(axis=0)
    variance = values.var(axis=0)
    sa_indices = np.zeros((nvars, nqoi))
    for ii in range(nvars):
        bin_bounds = marginal_icdfs[ii](np.linspace(0, 1, nbins+1))
        for jj in range(nbins):
            inds = np.where((samples[ii, :] >= bin_bounds[jj]) &
                            (samples[ii, :] < bin_bounds[jj+1]))[0]
            nsamples_ii = inds.shape[0]
            sa_indices[ii] += nsamples_ii/nsamples*(
                values[inds, :].mean()-mean)**2
        sa_indices[ii] /= variance
    return sa_indices


def bootstrapped_borgonovo_sensivities(
        fun, variable, nsamples, nbins=None, nbootstraps=10):
    """
    Compute main-effect sensitivity indices for a model using
    algorithm from [BHPRA2016]_

    Parameters
    ----------
        fun : callable
        The function to be approximated

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars, nsamples) and the
        output is a 2D np.ndarray with shape (nsamples, nqoi)

    variable : pya.IndependentMarginalsVariable
        Object containing information of the joint density of the inputs z.
        This is used to generate random samples from this join density

    nsamples : integer
        The number of samples used to compute the sensitivity indices

    nbins : integer
        The number of bins used to divide the domain of each marginal variable

    nbootstraps : integer
        The number of bootstraps used to obtain estimates of error in the
        sensitivity indices

    Returns
    -------
    result : SensitivityResult
       Object containing the sensitivity indices

    References
    ----------
    `Borgonovo, E., Hazen, G. and Plischke, E. A Common Rationale for Global Sensitivity Measures and Their Estimation. 36(10):1871-1895, 2016. <https://doi.org/10.1111/risa.12555>`_
    """
    nvars = variable.num_vars()
    samples = variable.rvs(nsamples)
    values = fun(samples)
    nqoi = values.shape[1]
    sa_indices = np.zeros((nbootstraps+1, nvars, nqoi))
    marginal_icdfs = [v.ppf for v in variable.marginals()]
    sa_indices[0, :] = _borgonovo_estimation(
        samples, values, marginal_icdfs, nbins)
    for kk in range(1, nbootstraps+1):
        permuted_inds = np.random.choice(np.arange(nsamples), nsamples)
        psamples = samples[:, permuted_inds]
        pvalues = values[permuted_inds, :]
        sa_indices[kk, :] = _borgonovo_estimation(
            psamples, pvalues, marginal_icdfs, nbins)
    return sa_indices


def run_sensitivity_analysis(method, fun, variable, *args, **kwargs):
    """
    Compute sensitivity indices for a model.

    Parameters
    ----------
    method : string
        The name of the sensitivity method

    fun : callable
        The function to be approximated

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars, nsamples) and the
        output is a 2D np.ndarray with shape (nsamples, nqoi)


    variable : pya.IndependentMarginalsVariable
        Object containing information of the joint density of the inputs z.
        This is used to generate random samples from this join density

    args: kwargs
        optional keyword arguments

    kwargs: kwargs
        optional keyword arguments

    For more details on method specfici args, kwargs and results attributes see

        - :func:`pyapprox.analysis.sensitivity_analysis.sampling_based_sobol_indices`

        - :func:`pyapprox.analysis.sensitivity_analysis.bootstrapped_borgonovo_sensivities`

        - :func:`pyapprox.analysis.sensitivity_analysis.morris_sensitivities`

        - :func:`pyapprox.analysis.sensitivity_analysis.gpc_sobol_sensitivities`

        - :func:`pyapprox.analysis.sensitivity_analysis.analytic_sobol_indices_from_gaussian_process`

        - : func:`pyapprox.analysis.sensitivity_analysis.sparse_grid_sobol_sensitivities`

    Returns
    -------
    result : SensitivityResult
       Object containing the sensitivity indices
    """
    from pyapprox.surrogates.polychaos.adaptive_polynomial_chaos import (
        AdaptiveInducedPCE)
    from pyapprox.surrogates.interp.adaptive_sparse_grid import (
        CombinationSparseGrid)
    if method == "surrogate_sobol":
        if issubclass(type(fun), GaussianProcess):
            method = "gp_sobol"
        elif issubclass(type(fun), AdaptiveInducedPCE):
            method = "pce_sobol"
            fun = fun.pce
        elif type(fun) == PolynomialChaosExpansion:
            method = "pce_sobol"
        elif type(fun) == CombinationSparseGrid:
            method = "sg_sobol"
        else:
            msg = "Surrogate specific computation requested, but fun"
            msg += "is not a type of surrogate supported"
            raise ValueError(msg)

    methods = {
        "sobol": repeat_sampling_based_sobol_indices,
        "bin_sobol": bootstrapped_borgonovo_sensivities,
        "morris": morris_sensitivities,
        "pce_sobol": gpc_sobol_sensitivities,
        "gp_sobol": analytic_sobol_indices_from_gaussian_process,
        "sg_sobol": sparse_grid_sobol_sensitivities}

    if method not in methods:
        msg = f'Method "{method}" not found.\n Available methods are:\n'
        for key in methods.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)

    return methods[method](fun, variable, *args, **kwargs)


def get_isotropic_anova_indices(nvars, order):
    interaction_terms = compute_hyperbolic_indices(nvars, order)
    interaction_terms = interaction_terms[:, np.where(
        interaction_terms.max(axis=0) == 1)[0]]
    return interaction_terms


def _plot_sensitivity_indices(labels, ax, sa_indices):
    nindices = len(sa_indices)
    assert len(labels) == nindices
    locations = np.arange(sa_indices.shape[0])
    bp = ax.bar(locations, sa_indices[:, 0])
    ax.set_xticks(locations)
    ax.set_xticklabels(labels, rotation=45)
    return bp


def _get_sobol_indices_labels(result):
    interaction_terms = result["sobol_interaction_indices"]
    rv = 'z'
    labels = []
    for ii in range(len(interaction_terms)):
        ll = '($'
        for jj in range(len(interaction_terms[ii])-1):
            ll += '%s_{%d},' % (rv, interaction_terms[ii][jj]+1)
        ll += '%s_{%d}$)' % (rv, interaction_terms[ii][-1]+1)
        labels.append(ll)
    return labels


def plot_sensitivity_indices(result, axs=None, include_vars=None):
    import matplotlib.pyplot as plt
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(3*8, 6), sharey=True)

    rv = "z"

    if type(result["sobol_indices"]) == np.ndarray:
        nvars = len(result['total_effects'])
        if include_vars is None:
            include_vars = np.arange(nvars, dtype=int)

        labels = [r'$%s_{%d}$' % (rv, ii+1) for ii in include_vars]
        im0 = _plot_sensitivity_indices(
            labels, axs[0], result["sobol_indices"][:nvars][include_vars])
        im1 = _plot_sensitivity_indices(
            labels, axs[1], result["total_effects"][include_vars])
        II = np.argsort(result["sobol_indices"][:, 0])[-10:][::-1]
        labels = _get_sobol_indices_labels(result)
        labels = [labels[ii] for ii in II]
        im2 = _plot_sensitivity_indices(
            labels, axs[2], result["sobol_indices"][II])
        return [im0, im1, im2], axs

    nvars = len(result['total_effects']['median'])
    if include_vars is None:
        include_vars = np.arange(nvars, dtype=int)

    im0 = plot_sensitivity_indices_with_confidence_intervals(
        [r'$%s_{%d}$' % (rv, ii+1) for ii in include_vars], axs[0],
        result["sobol_indices"]["median"][:nvars][include_vars],
        result["sobol_indices"]["quantile-0.25"][:nvars][include_vars],
        result["sobol_indices"]["quantile-0.75"][:nvars][include_vars],
        result["sobol_indices"]["amin"][:nvars][include_vars],
        result["sobol_indices"]["amax"][:nvars][include_vars])

    im1 = plot_sensitivity_indices_with_confidence_intervals(
        [r'$%s_{%d}$' % (rv, ii+1) for ii in include_vars], axs[1],
        result["total_effects"]["median"][include_vars],
        result["total_effects"]["quantile-0.25"][include_vars],
        result["total_effects"]["quantile-0.75"][include_vars],
        result["total_effects"]["amin"][include_vars],
        result["total_effects"]["amax"][include_vars])

    # sort sobol indices largest to smallest values left to right
    II = np.argsort(result["sobol_indices"]["median"].squeeze())[-10:][::-1]
    labels = _get_sobol_indices_labels(result)
    labels = [labels[ii] for ii in II]
    median_sobol_indices = result["sobol_indices"]["median"][II]
    q1_sobol_indices = result["sobol_indices"]["quantile-0.25"][II]
    q3_sobol_indices = result["sobol_indices"]["quantile-0.75"][II]
    min_sobol_indices = result["sobol_indices"]["amin"][II]
    max_sobol_indices = result["sobol_indices"]["amax"][II]
    im2 = plot_sensitivity_indices_with_confidence_intervals(
        labels, axs[2], median_sobol_indices, q1_sobol_indices,
        q3_sobol_indices, min_sobol_indices, max_sobol_indices)
    return [im0, im1, im2], axs
