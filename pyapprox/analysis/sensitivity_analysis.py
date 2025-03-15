from itertools import combinations
from functools import partial
from typing import List, Dict

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.spatial.distance import cdist


from abc import ABC, abstractmethod
from typing import Tuple
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.bases.multiindex import HyperbolicIndexGenerator
from pyapprox.util.misc import argsort_indices_leixographically
from pyapprox.util.sys_utilities import hash_array

from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.expdesign.sequences import SobolSequence, HaltonSequence
from pyapprox.surrogates.bases.univariate.orthopoly import GaussQuadratureRule
from pyapprox.surrogates.bases.basis import TensorProductQuadratureRule
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.sparsegrids.combination import (
    CombinationSparseGrid,
    SparseGridToOrthonormalPolynomialChaosExpansionConverter,
)
from pyapprox.surrogates.autogp.stats import (
    ExactGaussianProcess,
    GaussianProcessStatistics,
    EnsembleGaussianProcessStatistics,
)


class VarianceBasedSensitivityAnalysis(ABC):
    def __init__(self, nvars: int, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._nvars = nvars
        self._bkd = backend

    def nvars(self) -> int:
        return self._nvars

    def isotropic_interaction_terms(self, order: int) -> Array:
        gen = HyperbolicIndexGenerator(
            self.nvars(), order, 1.0, backend=self._bkd
        )
        interaction_terms = gen.get_indices()
        interaction_terms = interaction_terms[
            :, self._bkd.where(interaction_terms.max(axis=0) == 1)[0]
        ]
        return interaction_terms

    def _default_interaction_terms(self) -> Array:
        return self.isotropic_interaction_terms(2)

    def set_interaction_terms_of_interest(self, interaction_terms: Array):
        """
        Parameters
        ----------
        interaction_terms : Array (nvars, nterms)
        Index defining the active terms in each interaction. If the
        ith  variable is active interaction_terms[i] == 1 and zero otherwise
        This index must be downward closed due to way sobol indices are
        computed
        """
        main_effect_indices = interaction_terms[
            :, interaction_terms.sum(axis=0) == 1
        ]
        if main_effect_indices.shape[1] != self.nvars():
            # This is not required by computation of sobol indices
            # but only to ensure all main effects are computed
            raise ValueError(
                "interaction_terms must contain all main effect indices"
            )
        self._interaction_terms = interaction_terms

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    def _correct_interaction_variance_ratios(
        self, interaction_variances: Array
    ):
        # must substract of contributions from lower-dimensional terms from
        # each interaction value For example, let R_ij be interaction_variances
        # the sobol index S_ij satisfies R_ij = S_i + S_j + S_ij
        idx = argsort_indices_leixographically(self._interaction_terms)
        sobol_indices = interaction_variances.copy()
        sobol_indices_dict = dict()
        for ii in range(idx.shape[0]):
            index = self._interaction_terms[:, idx[ii]]
            active_vars = self._bkd.where(index > 0)[0]
            nactive_vars = index.sum()
            sobol_indices_dict[tuple(active_vars)] = idx[ii]
            if nactive_vars > 1:
                for jj in range(nactive_vars - 1):
                    indices = combinations(active_vars, jj + 1)
                    for key in indices:
                        sobol_indices[idx[ii]] -= sobol_indices[
                            sobol_indices_dict[key]
                        ]
        return sobol_indices


class PolynomialChaosSensivitityAnalysis(VarianceBasedSensitivityAnalysis):
    def _compute_main_and_total_effects(self):
        r"""
        Assume basis is orthonormal
        Assume first coefficient is the coefficient of the constant basis.
        Remove this assumption by extracting array index of constant term
        from indices

        Returns
        -------
        main_effects : Array(num_vars)
            Contribution to variance of each variable acting alone

        total_effects : Array(num_vars)
            Contribution to variance of each variable acting alone or with
            other variables

        """
        main_effects = self._bkd.zeros((self._pce.nvars(), self._pce.nqoi()))
        total_effects = self._bkd.zeros((self._pce.nvars(), self._pce.nqoi()))
        variance = self._bkd.zeros(self._pce.nqoi())

        for ii in range(self._pce.basis().nterms()):
            index = self._pce.basis().get_indices()[:, ii]

            # calculate contribution to variance of the index
            var_contribution = self._pce.get_coefficients()[ii, :] ** 2

            # get number of dimensions involved in interaction, also known
            # as order
            non_constant_vars = self._bkd.where(index > 0)[0]
            order = non_constant_vars.shape[0]

            if order > 0:
                variance += var_contribution

            # update main effects
            if order == 1:
                var = non_constant_vars[0]
                main_effects[var, :] += var_contribution

            # update total effects
            for ii in range(order):
                var = non_constant_vars[ii]
                total_effects[var, :] += var_contribution

        if not self._bkd.all(self._bkd.isfinite(variance)):
            raise RuntimeError("Variance was not finite")
        if self._bkd.any(variance <= 0):
            raise RuntimeError("Variance was not positive")
        main_effects /= variance
        total_effects /= variance
        return main_effects, total_effects

    def _compute_sobol_indices(self):
        variance = self._bkd.zeros(self._pce.nqoi())
        interaction_variances = self._bkd.zeros(
            (self._interaction_terms.shape[1], self._pce.nqoi())
        )
        interaction_terms_dict = dict(
            zip(
                [hash_array(index) for index in self._interaction_terms.T],
                self._bkd.arange(self._interaction_terms.shape[1], dtype=int),
            )
        )
        for ii in range(self._pce.nterms()):
            basis_index = self._pce.basis().get_indices()[:, ii]
            var_contribution = self._pce.get_coefficients()[ii, :] ** 2
            non_constant_vars = self._bkd.where(basis_index > 0)[0]
            index = self._bkd.zeros((self.nvars(),), dtype=int)
            index[non_constant_vars] = 1
            key = hash_array(index)
            if len(non_constant_vars) > 0:
                variance += var_contribution
            if key in interaction_terms_dict:
                interaction_variances[
                    interaction_terms_dict[key]
                ] += var_contribution
        interaction_variances = self._bkd.asarray(interaction_variances)
        return interaction_variances / variance

    def compute(self, pce: PolynomialChaosExpansion):
        self._pce = pce
        if not hasattr(self, "_interaction_terms"):
            self.set_interaction_terms_of_interest(
                self._default_interaction_terms()
            )

        self._main_effects, self._total_effects = (
            self._compute_main_and_total_effects()
        )
        self._sobol_indices = self._compute_sobol_indices()

    def mean(self) -> Array:
        return self._pce.mean()

    def variance(self) -> Array:
        return self._pce.variance()

    def main_effects(self) -> Array:
        return self._main_effects

    def total_effects(self) -> Array:
        return self._total_effects

    def sobol_indices(self) -> Array:
        return self._sobol_indices


class LagrangeSparseGridSensitivityAnalysis(
    PolynomialChaosSensivitityAnalysis
):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
    ):
        self._variable = variable
        super().__init__(self._variable.nvars(), variable._bkd)

    def compute(self, sg: CombinationSparseGrid):
        pce_quad_rule = TensorProductQuadratureRule(
            self._variable.nvars(),
            [
                GaussQuadratureRule(marginal)
                for marginal in self._variable.marginals()
            ],
        )
        converter = SparseGridToOrthonormalPolynomialChaosExpansionConverter(
            pce_quad_rule
        )
        pce = converter.convert(sg)
        return super().compute(pce)


# TODO consider making this a member function of variance based sensivitity
# classes. Similarly for plotting total effects and sobol indices.
# Use EnsembleGaussianProcessSensivitityAnalysis as an example
def plot_main_effects(
    main_effects, ax, truncation_pct=0.95, max_slices=5, rv="z", qoi=0
):
    r"""
    Plot the main effects in a pie chart showing relative size.

    Parameters
    ----------
    main_effects : Array (nvars,nqoi)
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
    if main_effects.ndim == 1:
        main_effects = main_effects[:, None]
    main_effects = main_effects[:, qoi]
    if main_effects.sum() > 1.0 + np.finfo(float).eps:
        raise ValueError("main_effects sum was greater than 1")
    main_effects_sum = main_effects.sum()

    # sort main_effects in descending order
    II = np.argsort(main_effects)[::-1]
    main_effects = main_effects[II]

    labels = []
    partial_sum = 0.0
    for i in range(II.shape[0]):
        if partial_sum / main_effects_sum < truncation_pct and i < max_slices:
            labels.append("$%s_{%d}$" % (rv, II[i] + 1))
            partial_sum += main_effects[i]
        else:
            break

    if abs(partial_sum - main_effects_sum) > 0.5:
        main_effects.resize(i + 1)
        explode = np.zeros(main_effects.shape[0])
        labels.append(r"$\mathrm{other}$")
        main_effects[-1] = main_effects_sum - partial_sum
        explode[-1] = 0.1
    else:
        main_effects.resize(i)
        labels = labels[:i]
        explode = np.zeros(main_effects.shape[0])

    p = ax.pie(
        main_effects,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        explode=explode,
    )
    return p


def plot_sensitivity_indices_with_confidence_intervals(
    labels,
    ax,
    sa_indices_median,
    sa_indices_q1,
    sa_indices_q3,
    sa_indices_min,
    sa_indices_max,
    reference_values=None,
    fliers=None,
):
    nindices = len(sa_indices_median)
    assert len(labels) == nindices
    if reference_values is not None:
        assert len(reference_values) == nindices
    stats = [dict() for nn in range(nindices)]
    for nn in range(nindices):
        # use boxplot stats mean entry to store reference values.
        if reference_values is not None:
            stats[nn]["mean"] = reference_values[nn]
        stats[nn]["med"] = sa_indices_median[nn]
        stats[nn]["q1"] = sa_indices_q1[nn]
        stats[nn]["q3"] = sa_indices_q3[nn]
        stats[nn]["label"] = labels[nn]
        # use whiskers for min and max instead of fliers
        stats[nn]["whislo"] = sa_indices_min[nn]
        stats[nn]["whishi"] = sa_indices_max[nn]
        if fliers is not None:
            stats[nn]["fliers"] = fliers[nn]

    if reference_values is not None:
        showmeans = True
    else:
        showmeans = False

    if fliers is not None:
        showfliers = True
    else:
        showfliers = False

    bp = ax.bxp(
        stats,
        showfliers=showfliers,
        showmeans=showmeans,
        patch_artist=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="blue",
            markeredgecolor="blue",
            markersize=12,
        ),
        medianprops=dict(color="red"),
    )
    ax.tick_params(axis="x", labelrotation=45)

    colors = ["gray"] * nindices
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    colors = ["red"] * nindices
    return bp


def plot_total_effects(total_effects, ax, truncation_pct=0.95, rv="z", qoi=0):
    r"""
    Plot the total effects in a bar chart showing relative size.

    Parameters
    ----------
    total_effects : Array (nvars,nqoi)
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

    if total_effects.ndim == 1:
        total_effects = total_effects[:, None]
    total_effects = total_effects[:, qoi]

    width = 0.95
    locations = np.arange(total_effects.shape[0])
    p = ax.bar(locations - width / 2, total_effects, width, align="edge")
    labels = [
        "$%s_{%d}$" % (rv, ii + 1) for ii in range(total_effects.shape[0])
    ]
    ax.set_xticks(locations)
    ax.set_xticklabels(labels, rotation=0)
    return p


def plot_interaction_values(
    interaction_values,
    interaction_terms,
    ax,
    truncation_pct=0.95,
    max_slices=5,
    rv="z",
    qoi=0,
):
    r"""
    Plot sobol indices in a pie chart showing relative size.

    Parameters
    ----------
    interaction_values : Array (nvars,nqoi)
        The variance based Sobol indices

    interaction_terms : nlist (nchoosek(nvars+max_order,nvars))
        Indices Arrays of varying size specifying the variables in each
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

    if interaction_values.shape[0] != interaction_terms.shape[1]:
        print(interaction_values.shape, interaction_terms.shape)
        raise ValueError(
            "interaction_values and interaction_terms are inconsistent"
        )
    interaction_values = interaction_values[:, qoi]

    II = np.argsort(interaction_values)[::-1]
    interaction_values = interaction_values[II]
    interaction_terms = [interaction_terms[:, ii] for ii in II]

    labels = []
    partial_sum = 0.0
    for i in range(interaction_values.shape[0]):
        if partial_sum < truncation_pct and i < max_slices:
            label = "($"
            for j in range(len(interaction_terms[i]) - 1):
                label += "%s_{%d}," % (rv, interaction_terms[i][j] + 1)
            label += "%s_{%d}$)" % (rv, interaction_terms[i][-1] + 1)
            labels.append(label)
            partial_sum += interaction_values[i]
        else:
            break

    interaction_values = interaction_values[:i]
    if abs(partial_sum - 1.0) > 10 * np.finfo(np.double).eps:
        labels.append(r"$\mathrm{other}$")
        interaction_values = np.concatenate(
            [interaction_values, [1.0 - partial_sum]]
        )

    explode = np.zeros(interaction_values.shape[0])
    explode[-1] = 0.1
    assert interaction_values.shape[0] == len(labels)
    p = ax.pie(
        interaction_values,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        explode=explode,
    )
    return p


class MorrisSensitivityAnalysis:
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        nlevels: int,
        eps: float = 0,
    ):
        """
        Parameters
        ----------
        nvars : integer
            The number of variables

        nlevels : int
            The number of levels used for to define the morris grid.

        eps : float
            Set grid used defining the morris trajectory to [eps,1-eps].
            This is needed when mapping the morris trajectories using inverse
            CDFs of unbounded variables
        """
        self._variable = variable
        self._nvars = self._variable.nvars()
        if nlevels % 2 != 0:
            raise ValueError("nlevels must be an even integer")
        self._nlevels = nlevels
        self._eps = eps
        self._bkd = variable._bkd

    def _get_trajectory(self) -> Array:
        r"""
        Compute a morris trajectory used to compute elementary effects

        Returns
        -------
        trajectory : Array (nvars, nvars+1)
            The Morris trajectory which consists of nvars+1 samples
        """

        delta = self._nlevels / ((self._nlevels - 1) * 2)
        samples_1d = self._bkd.linspace(
            self._eps, 1 - self._eps, self._nlevels
        )

        initial_point = self._bkd.asarray(
            np.random.choice(samples_1d, self._nvars)
        )
        shifts = self._bkd.diag(
            self._bkd.asarray(np.random.choice([-delta, delta], self._nvars))
        )
        trajectory = self._bkd.empty((self._nvars, self._nvars + 1))
        trajectory[:, 0] = initial_point
        for ii in range(self._nvars):
            trajectory[:, ii + 1] = trajectory[:, ii].copy()
            if (trajectory[ii, ii] - delta) >= 0 and (
                trajectory[ii, ii] + delta
            ) <= 1:
                trajectory[ii, ii + 1] += shifts[ii]
            elif (trajectory[ii, ii] - delta) >= 0:
                trajectory[ii, ii + 1] -= delta
            elif (trajectory[ii, ii] + delta) <= 1:
                trajectory[ii, ii + 1] += delta
            else:
                raise Exception("This should not happen")
        return trajectory

    def _downselect_trajectories(self, candidate_samples: Array) -> Array:
        """Find trajectories that best 'fill the space'"""
        distances = self._bkd.zeros(
            (self._ncandidate_trajectories, self._ncandidate_trajectories)
        )
        for ii in range(self._ncandidate_trajectories):
            for jj in range(ii + 1):
                distances[ii, jj] = cdist(
                    candidate_samples[ii].T,
                    candidate_samples[jj].T,
                ).sum()
                distances[jj, ii] = distances[ii, jj]

        get_combinations = combinations(
            self._bkd.arange(self._ncandidate_trajectories),
            self._ntrajectories,
        )
        best_index = None
        best_value = -np.inf
        for ii, index in enumerate(get_combinations):
            value = self._bkd.sqrt(
                self._bkd.sum(
                    [
                        distances[ix[0], ix[1]] ** 2
                        for ix in combinations(index, 2)
                    ]
                )
            )
            if value > best_value:
                best_value = value
                best_index = index

        samples = self._bkd.hstack(
            [candidate_samples[ii, :, :] for ii in best_index]
        )
        return samples

    def generate_samples(
        self, ntrajectories: int, ncandidate_trajectories: int = None
    ) -> Array:
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
        ntrajectories : integer
            The number of Morris trajectories requested

        Returns
        -------
        samples : Array (nvars, ntrajectories * (nvars + 1))
            The samples of the Morris trajectories
        """
        if (
            ncandidate_trajectories is not None
            and ncandidate_trajectories <= ntrajectories
        ):
            raise ValueError("ncandidate_trajectories msut be > ntrajectories")
        self._ncandidate_trajectories = ncandidate_trajectories
        if self._ncandidate_trajectories is None:
            ncandidate_trajectories = ntrajectories
        else:
            ncandidate_trajectories = self._ncandidate_trajectories
        self._ntrajectories = ntrajectories
        candidate_samples = self._bkd.stack(
            [self._get_trajectory() for n in range(ncandidate_trajectories)],
            axis=0,
        )
        if self._ncandidate_trajectories is not None:
            self._samples = self._downselect_trajectories(candidate_samples)
        else:
            self._samples = self._bkd.hstack(
                [
                    candidate_samples[ii]
                    for ii in range(ncandidate_trajectories)
                ]
            )
        for ii, marginal in enumerate(self._variable.marginals()):
            self._samples[ii, :] = marginal.ppf(self._samples[ii, :])
        return self._samples

    def _compute_elementary_effects(self, values: Array):
        r"""
        Get the Morris elementary effects from a set of trajectories.

        Parameters
        ----------
        samples : Array (nvars,ntrajectories*(nvars+1))
            The morris trajectories

        values : Array (ntrajectories*(nvars+1),nqoi)
            The values of the vecto-valued target function with nqoi quantities
            of interest (QoI)

        Returns
        -------
        elem_effects : Array(nvars,ntrajectories,nqoi)
            The elementary effects of each variable for each trajectory and QoI
        """
        nqoi = values.shape[1]
        if self._samples.shape[1] != values.shape[0]:
            raise ValueError("Samples and values are inconsistent")
        ntrajectories = self._samples.shape[1] // (self._nvars + 1)
        self._elem_effects = self._bkd.empty(
            (self._nvars, ntrajectories, nqoi)
        )
        abs_delta = self._nlevels / ((self._nlevels - 1) * 2)
        for ii in range(ntrajectories):
            # Use delta (d) in [0, 1]. Not clear from literature
            # if this should be scaled to true variable distributions
            # when dividing (f(x)-f(x+d)) / d
            trajectory_samples = self._samples[
                :, ii * (self._nvars + 1) : (ii + 1) * (self._nvars + 1)
            ]
            delta = (
                abs_delta
                * self._bkd.sign(
                    self._bkd.diag(
                        trajectory_samples[:, 1:] - trajectory_samples[:, :-1]
                    )
                )[:, None]
            )
            self._elem_effects[:, ii] = (
                self._bkd.diff(
                    values[
                        ii * (self._nvars + 1) : (ii + 1) * (self._nvars + 1)
                    ].T
                ).T
                / delta
            )
        self._elem_effects

    def compute(self, values: Array):
        self._compute_elementary_effects(values)
        self._compute_sensitivity_indices()

    def _compute_sensitivity_indices(self):
        self._mu = self._elem_effects.mean(axis=1)
        self._mu_star = self._bkd.abs(self._elem_effects).mean(axis=1)
        assert self._mu.shape == (
            self._elem_effects.shape[0],
            self._elem_effects.shape[2],
        )
        self._sigma = self._bkd.std(self._elem_effects, ddof=1, axis=1)

    def mu(self) -> Array:
        r"""
        Return the Morris sensitivity indices mu_star

        Mu_star is the mu^\star from Campolongo et al.

        Returns
        -------
        mu : Array (nvars, nqoi)
            The sensitivity of each output to each input. Larger mu corresponds
            to higher sensitivity
        """
        return self._mu

    def mu_star(self) -> Array:
        r"""
        Return the Morris sensitivity indices mu_star

        Mu_star is the mu^\star from Campolongo et al.

        Returns
        -------
        mu : Array (nvars, nqoi)
            The sensitivity of each output to each input. Larger mu corresponds
            to higher sensitivity
        """
        return self._mu_star

    def sigma(self) -> Array:
        r"""
        Return the Morris sensitivity indices sigma

        Returns
        -------
         sigma: Array (nvars, nqoi)
            A measure of the non-linearity and/or interaction effects of each
            input for each output.
            Low values suggest a linear realationship between
            the input and output. Larger values suggest a that the output is
            nonlinearly dependent on the input and/or the input interacts with
            other inputs
        """
        return self._sigma

    def print_sensitivity_indices(self, qoi: int = 0):
        str_format = "{:<3} {:>10} {:>10} {:>10}"
        print(str_format.format(" ", "mu", "mu*", "sigma"))
        str_format = "{:<3} {:10.5f} {:10.5f} {:10.5f}"
        for ii in range(self._mu.shape[0]):
            print(
                str_format.format(
                    f"Z_{ii+1}",
                    self._mu[ii, qoi],
                    self._mu_star[ii, qoi],
                    self._sigma[ii, qoi],
                )
            )


class SensitivityResult(OptimizeResult):
    pass


# def _repeat_sampling_based_sobol_indices(
#     fun,
#     variable,
#     interaction_terms=None,
#     nsamples=1000,
#     sampling_method="random",
#     nsobol_realizations=10,
#     qmc_start_index=1,
# ):
#     if interaction_terms is None:
#         interaction_terms = get_isotropic_anova_indices(variable.num_vars(), 2)

#     means, variances, sobol_values, total_values = [], [], [], []
#     # qmc_start_index = 0
#     for ii in range(nsobol_realizations):
#         sv, tv, vr, me = sampling_based_sobol_indices(
#             fun,
#             variable,
#             interaction_terms,
#             nsamples,
#             sampling_method="sobol",
#             qmc_start_index=qmc_start_index,
#         )
#         means.append(me)
#         variances.append(vr)
#         sobol_values.append(sv)
#         total_values.append(tv)
#         qmc_start_index += nsamples
#     means = np.asarray(means)
#     variances = np.asarray(variances)
#     sobol_values = np.asarray(sobol_values)
#     total_values = np.asarray(total_values)

#     interaction_terms = [
#         np.where(index > 0)[0] for index in interaction_terms.T
#     ]
#     return sobol_values, total_values, variances, means


class SampleBasedSensivitityAnalysis(VarianceBasedSensitivityAnalysis):
    """
    See I.M. Sobol. Mathematics and Computers in Simulation 55 (2001) 271–280

    and

    Saltelli, Annoni et. al, Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index. 2010.
    https://doi.org/10.1016/j.cpc.2009.09.018
    """

    def __init__(
        self,
        variable: IndependentMarginalsVariable,
    ):
        super().__init__(variable.nvars(), variable._bkd)
        self._variable = variable
        self._all_idx = self._bkd.arange(self._variable.nvars(), dtype=int)

    @abstractmethod
    def _get_AB_samples(self, nsamples: int) -> Tuple[Array, Array]:
        raise NotImplementedError

    def _sobol_index_samples(self, sobol_index: Array) -> Array:
        """
        Given two sample sets A and B generate the sets :math:`A_B^{I}` from
        The rows of A_B^I are all from A except for the rows with non zero
        entries in the index I.
        When A and B are QMC samples it is best to change as few rows
        as possible
        """
        mask = self._bkd.asarray(sobol_index, dtype=bool)
        samples = np.vstack([self._samplesA[~mask], self._samplesB[mask]])
        idx = np.hstack([self._all_idx[~mask], self._all_idx[mask]])
        samples = samples[self._bkd.argsort(idx), :]
        return samples

    def generate_samples(self, nsamples: int) -> Array:
        if not hasattr(self, "_interaction_terms"):
            self.set_interaction_terms_of_interest(
                self._default_interaction_terms()
            )
        self._samplesA, self._samplesB = self._get_AB_samples(nsamples)
        self._samplesAB = []
        for ii in range(self._interaction_terms.shape[1]):
            sobol_index = self._interaction_terms[:, ii]
            self._samplesAB.append(self._sobol_index_samples(sobol_index))
        return self._bkd.hstack(
            [self._samplesA, self._samplesB] + self._samplesAB
        )

    def _unpack_values(self, values: Array) -> Array:
        cnt = 0
        valuesA = values[cnt : cnt + self._samplesA.shape[1]]
        cnt += self._samplesA.shape[1]
        valuesB = values[cnt : cnt + self._samplesB.shape[1]]
        cnt += self._samplesB.shape[1]
        valuesAB = []
        for ii in range(self._interaction_terms.shape[1]):
            valuesAB.append(values[cnt : cnt + self._samplesAB[ii].shape[1]])
            cnt += self._samplesAB[ii].shape[1]
        return valuesA, valuesB, valuesAB

    def compute(self, values: Array):
        # We cannot guarantee that the main_effects will be <= 1. Because
        # variance and each interaction_index are computed with different
        # sample sets.
        # Consider function of two variables which is constant in one variable
        # then interaction_index[0] should equal variance.
        # But with different sample
        # sets interaction_index could be smaller or larger than the variance.
        # Similarly we cannot even guarantee main effects will be non-negative
        # We also cannot guarantee that the sobol indices will be non-negative.

        valuesA, valuesB, valuesAB = self._unpack_values(values)
        self._mean = self._bkd.mean(valuesA, axis=0)
        self._variance = self._bkd.var(valuesA, axis=0)
        nterms = self._interaction_terms.shape[1]
        nvars = self._variable.nvars()
        interaction_values = self._bkd.empty((nterms, valuesA.shape[1]))
        self._total_effects = self._bkd.empty((nvars, valuesA.shape[1]))
        for ii in range(nterms):
            sobol_index = self._interaction_terms[:, ii]
            interaction_values[ii, :] = (
                self._bkd.mean(valuesB * (valuesAB[ii] - valuesA), axis=0)
                / self._variance
            )
            if sobol_index.sum() == 1:
                idx = self._bkd.where(sobol_index == 1)[0][0]
                # entry f in Table 2 of Saltelli, Annoni et. al
                self._total_effects[idx] = (
                    0.5
                    * self._bkd.mean((valuesA - valuesAB[ii]) ** 2, axis=0)
                    / self._variance
                )
        self._sobol_indices = self._correct_interaction_variance_ratios(
            interaction_values
        )
        self._main_effects = self._sobol_indices[
            self._interaction_terms.sum(axis=0) == 1, :
        ]

    def main_effects(self) -> Array:
        return self._main_effects

    def mean(self) -> Array:
        return self._mean

    def variance(self) -> Array:
        return self._variance

    def total_effects(self) -> Array:
        return self._total_effects

    def sobol_indices(self) -> Array:
        return self._sobol_indices

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class MonteCarloBasedSensitivityAnalysis(SampleBasedSensivitityAnalysis):
    def _get_AB_samples(self, nsamples: int) -> Tuple[Array, Array]:
        return self._variable.rvs(nsamples), self._variable.rvs(nsamples)


class LowDiscrepancySequenceBasedSensitivityAnalysis(
    SampleBasedSensivitityAnalysis
):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        seq_start_idx: int = 0,
    ):
        super().__init__(variable)
        # need to create sobol sequence that is twice the dimenion of
        # the variable
        self._seq_variable = IndependentMarginalsVariable(
            self._variable.marginals() + self._variable.marginals()
        )
        self._set_sequence(seq_start_idx)

    @abstractmethod
    def _set_sequence(self, seq_start_idx: int):
        raise NotImplementedError

    def _get_AB_samples(self, nsamples: int) -> Tuple[Array, Array]:
        samples = self._seq.rvs(nsamples)
        nvars = self._variable.nvars()
        return samples[:nvars, :], samples[nvars:, :]


class SobolSequenceBasedSensitivityAnalysis(
    LowDiscrepancySequenceBasedSensitivityAnalysis
):
    def _set_sequence(self, seq_start_idx: int):
        self._seq = SobolSequence(
            self._seq_variable.nvars(),
            seq_start_idx,
            self._seq_variable,
            self._bkd,
        )


class HaltonSequenceBasedSensitivityAnalysis(
    LowDiscrepancySequenceBasedSensitivityAnalysis
):
    def _set_sequence(self, seq_start_idx: int):
        self._seq = HaltonSequence(
            self._seq_variable.nvars(),
            seq_start_idx,
            self._seq_variable,
            self._bkd,
        )


# def repeat_sampling_based_sobol_indices(
#     fun,
#     variable,
#     interaction_terms=None,
#     nsamples=1000,
#     sampling_method="random",
#     nsobol_realizations=10,
#     summary_stats=[
#         "mean",
#         "median",
#         "min",
#         "max",
#         "quantile-0.25",
#         "quantile-0.75",
#     ],
#     qmc_start_index=1,
# ):
#     """
#     Compute sobol indices for different sample sets. This allows estimation
#     of error due to finite sample sizes. This function requires evaluting
#     the function at nsobol_realizations * N, where N is the
#     number of samples required by sampling_based_sobol_indices. Thus
#     This function is useful when applid to a random
#     realization of a Gaussian process requires the Cholesky decomposition
#     of a nsamples x nsamples matrix which becomes to costly for nsamples >1000
#     """
#     sobol_values, total_values, variances, means = (
#         _repeat_sampling_based_sobol_indices(
#             fun,
#             variable,
#             interaction_terms,
#             nsamples,
#             sampling_method,
#             nsobol_realizations,
#             qmc_start_index,
#         )
#     )

#     stat_functions = _get_stats_functions(summary_stats)
#     result = dict()
#     result["sobol_interaction_indices"] = interaction_terms
#     data = [sobol_values, total_values, variances, means]
#     data_names = ["sobol_indices", "total_effects", "variance", "mean"]
#     for item, name in zip(data, data_names):
#         subdict = dict()
#         for ii, sfun in enumerate(stat_functions):
#             subdict[sfun.__name__] = sfun(item, axis=(0))
#         subdict["values"] = item
#         result[name] = subdict

#     return result


def _get_stats_functions(summary_stats):
    quantile_stats = []
    for q in [0.25, 0.75]:
        sfun = partial(np.quantile, q=q)
        sfun.__name__ = f"quantile-{q}"
        quantile_stats.append(sfun)
    stat_functions_dict = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "std": np.std,
        "quantile-0.25": quantile_stats[0],
        "quantile-0.75": quantile_stats[1],
    }
    stat_functions_dict["min"].__name__ = "amin"
    stat_functions_dict["max"].__name__ = "amax"
    for name in summary_stats:
        if name not in stat_functions_dict:
            msg = f"Summary stats {name} not supported\n"
            msg += f"Select from {list(stat_functions_dict.keys())}"
            raise ValueError(msg)
    stat_functions = [stat_functions_dict[name] for name in summary_stats]
    return stat_functions


class BinBasedVarianceSensitivityAnalysis:
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        nbins: int = None,
        eps: float = 0.0,
    ):
        """
        Compute main-effect sensitivity indices for a model using
        algorithm from [BHPRA2016]_

        Parameters
        ----------
        variable : pya.IndependentMarginalsVariable
            Object containing information of the joint density of the inputs z.
            This is used to generate random samples from this join density

        nbins : integer
            The number of bins used to divide the domain of each marginal
            variable

        eps: float
            Tolerance when used to map [eps,1-eps] to variable domain using
            inverse transform sampling.
            Transform is not defined when eps is 0 for unbouned variables

        References
        ----------
        `Borgonovo, E., Hazen, G. and Plischke, E. A Common Rationale for Global Sensitivity Measures and Their Estimation. 36(10):1871-1895, 2016. <https://doi.org/10.1111/risa.12555>`_
        """
        self._variable = variable
        self._bkd = variable._bkd
        self._nbins = None
        self._eps = eps

    def _compute(self, samples: Array, values: Array) -> Array:
        # TODO currently only supports main effects.
        # Make 2D and 3D binning o compute 2d and 3d interactions
        nvars, nsamples = samples.shape
        nqoi = values.shape[1]
        assert values.shape[0] == nsamples

        if self._nbins is None:
            nbins = max(2, int(1 / 3 * (nsamples) ** (1.0 / 3)))
        else:
            nbins = self._nbins

        mean = self._bkd.mean(values, axis=0)
        variance = self._bkd.var(values, axis=0)
        main_effects = self._bkd.zeros((nvars, nqoi))
        for ii, marginal in enumerate(self._variable.marginals()):
            bin_bounds = marginal.ppf(
                self._bkd.linspace(self._eps, 1 - self._eps, nbins + 1)
            )
            for jj in range(nbins):
                inds = self._bkd.where(
                    (samples[ii, :] >= bin_bounds[jj])
                    & (samples[ii, :] < bin_bounds[jj + 1])
                )[0]
                nsamples_ii = inds.shape[0]
                main_effects[ii] += (
                    nsamples_ii
                    / nsamples
                    * (values[inds, :].mean() - mean) ** 2
                )
            main_effects[ii] /= variance
        return main_effects

    def compute(self, samples: Array, values: Array) -> Array:
        self._main_effects = self._compute(samples, values)

    def main_effects(self) -> Array:
        return self._main_effects

    def bootstrap(
        self, samples: Array, values: Array, nbootstraps: int = 10
    ) -> Array:
        """

        Parameters
        ----------
        nbootstraps : integer
            The number of bootstraps used to obtain estimates of error in the
            sensitivity indices
        """
        nsamples, nqoi = values.shape
        main_effects = self._bkd.zeros(
            (nbootstraps + 1, self._variable.nvars(), nqoi)
        )
        main_effects[0, :] = self._compute_borgonovo_estimation(
            samples, values
        )
        for kk in range(1, nbootstraps + 1):
            # sample with replacement
            permuted_inds = self._bkd.array(
                np.random.choice(self._bkd.arange(nsamples), nsamples)
            )
            psamples = samples[:, permuted_inds]
            pvalues = values[permuted_inds, :]
            main_effects[kk, :] = self._compute(psamples, pvalues)

        bootstrapped_stats = SensitivityResult(
            {
                "median": self._bkd.median(main_effects, axis=0),
                "min": self._bkd.max(main_effects, axis=0),
                "max": self._bkd.max(main_effects, axis=0),
                "quantile-0.25": self._bkd.quantile(
                    main_effects, q=0.25, axis=0
                ),
                "quantile-0.75": self._bkd.quantile(
                    main_effects, q=0.75, axis=0
                ),
            }
        )
        return bootstrapped_stats


def get_isotropic_anova_indices(nvars, order):
    interaction_terms = compute_hyperbolic_indices(nvars, order)
    interaction_terms = interaction_terms[
        :, np.where(interaction_terms.max(axis=0) == 1)[0]
    ]
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
    rv = "z"
    labels = []
    for ii in range(len(interaction_terms)):
        ll = "($"
        for jj in range(len(interaction_terms[ii]) - 1):
            ll += "%s_{%d}," % (rv, interaction_terms[ii][jj] + 1)
        ll += "%s_{%d}$)" % (rv, interaction_terms[ii][-1] + 1)
        labels.append(ll)
    return labels


def plot_sensitivity_indices(result, axs=None, include_vars=None):
    import matplotlib.pyplot as plt

    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(3 * 8, 6), sharey=True)

    rv = "z"

    if type(result["sobol_indices"]) == Array:
        nvars = len(result["total_effects"])
        if include_vars is None:
            include_vars = np.arange(nvars, dtype=int)

        labels = [r"$%s_{%d}$" % (rv, ii + 1) for ii in include_vars]
        im0 = _plot_sensitivity_indices(
            labels, axs[0], result["sobol_indices"][:nvars][include_vars]
        )
        im1 = _plot_sensitivity_indices(
            labels, axs[1], result["total_effects"][include_vars]
        )
        II = np.argsort(result["sobol_indices"][:, 0])[-10:][::-1]
        labels = _get_sobol_indices_labels(result)
        labels = [labels[ii] for ii in II]
        im2 = _plot_sensitivity_indices(
            labels, axs[2], result["sobol_indices"][II]
        )
        return [im0, im1, im2], axs

    nvars = len(result["total_effects"]["median"])
    if include_vars is None:
        include_vars = np.arange(nvars, dtype=int)

    im0 = plot_sensitivity_indices_with_confidence_intervals(
        [r"$%s_{%d}$" % (rv, ii + 1) for ii in include_vars],
        axs[0],
        result["sobol_indices"]["median"][:nvars][include_vars],
        result["sobol_indices"]["quantile-0.25"][:nvars][include_vars],
        result["sobol_indices"]["quantile-0.75"][:nvars][include_vars],
        result["sobol_indices"]["amin"][:nvars][include_vars],
        result["sobol_indices"]["amax"][:nvars][include_vars],
    )

    im1 = plot_sensitivity_indices_with_confidence_intervals(
        [r"$%s_{%d}$" % (rv, ii + 1) for ii in include_vars],
        axs[1],
        result["total_effects"]["median"][include_vars],
        result["total_effects"]["quantile-0.25"][include_vars],
        result["total_effects"]["quantile-0.75"][include_vars],
        result["total_effects"]["amin"][include_vars],
        result["total_effects"]["amax"][include_vars],
    )

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
        labels,
        axs[2],
        median_sobol_indices,
        q1_sobol_indices,
        q3_sobol_indices,
        min_sobol_indices,
        max_sobol_indices,
    )
    return [im0, im1, im2], axs


class FixedGaussianProcess(ExactGaussianProcess):
    def __init__(self, gp: ExactGaussianProcess, alpha: float = 0):
        self._gp = gp
        self._bkd = gp._bkd
        self._alpha = alpha

    def generate_training_values(self, samples: Array, nrealizations: int):
        rand_noise = self._bkd.asarray(
            np.random.normal(0, 1, (int(nrealizations), samples.shape[1])).T
        )
        # make last sample mean of gaussian process
        rand_noise = self._bkd.hstack(
            (self._rand_noise, self._bkd.zeros((rand_noise.shape[0])))
        )
        return self._gp._predict_random_realizations_from_rand_noise(
            samples, rand_noise
        )


class GaussianProcessSensivitityAnalysis(VarianceBasedSensitivityAnalysis):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
    ):
        self._variable = variable
        super().__init__(self._variable.nvars(), variable._bkd)

    def _set_gp(self, gp: ExactGaussianProcess):
        self._gp = gp
        self._stat = GaussianProcessStatistics(self._gp, self._variable)
        self._nrealizations = 1

    def _conditional_variance(self, index: Array) -> Array:
        return self._stat.conditional_variance(index)

    def _mean(self) -> Array:
        return self._stat.expectation_of_mean()

    def _variance(self) -> Array:
        return self._stat.expectation_of_variance()

    def compute(self, gp: ExactGaussianProcess):
        self._set_gp(gp)
        self._mean = self._mean()
        self._variance = self._variance()
        if not hasattr(self, "_interaction_terms"):
            self.set_interaction_terms_of_interest(
                self._default_interaction_terms()
            )

        self._sobol_indices = self._compute_sobol_indices()
        self._main_effects = self._compute_main_effects()
        self._total_effects = self._compute_total_effects()

    def _compute_sobol_indices(self):
        interaction_variances = self._bkd.zeros(
            (self._interaction_terms.shape[1], self._nrealizations)
        )
        for ii in range(self._interaction_terms.shape[1]):
            index = self._interaction_terms[:, ii]
            interaction_variances[ii] = self._conditional_variance(index)
        return self._correct_interaction_variance_ratios(
            interaction_variances / self._variance
        )

    def _main_effect_idx(self):
        return self._bkd.where(self._interaction_terms.sum(axis=0) == 1)[0]

    def _compute_main_effects(self):
        return self._sobol_indices[self._main_effect_idx()]

    def _compute_total_effects(self):
        total_effect_interaction_variances = self._bkd.zeros(
            (self._gp.nvars(), self._nrealizations)
        )
        total_effect_interaction_terms = self._bkd.ones(
            (self._gp.nvars(), self._gp.nvars()), dtype=int
        ) - np.eye(self._gp.nvars(), dtype=int)
        for ii in range(self._gp.nvars()):
            index = total_effect_interaction_terms[:, ii]
            total_effect_interaction_variances[ii] = (
                self._conditional_variance(index)
            )
        return 1 - total_effect_interaction_variances / self._variance

    def mean(self) -> Array:
        return self._mean

    def variance(self) -> Array:
        return self._variance

    def main_effects(self) -> Array:
        return self._main_effects

    def total_effects(self) -> Array:
        return self._total_effects

    def sobol_indices(self) -> Array:
        return self._sobol_indices


class EnsembleGaussianProcessSensivitityAnalysis(
    GaussianProcessSensivitityAnalysis
):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        nrealizations: int = 100,
    ):
        self._nrealizations = nrealizations
        super().__init__(variable)

    def _set_gp(self, gp: ExactGaussianProcess):
        self._gp = gp
        self._stat = EnsembleGaussianProcessStatistics(
            self._gp, self._variable
        )

    def _mean(self) -> Array:
        return self._stat.means_of_realizations(self._nrealizations)

    def _variance(self) -> Array:
        return self._stat.variances_of_realizations(self._nrealizations)

    def _conditional_variance(self, index: Array) -> Array:
        return self._stat.conditional_variances_of_realizations(
            index, self._nrealizations
        )

    def _get_stat_function(self, statname: str) -> callable:
        if "quantile" in statname:
            if len(statname) != 10:
                raise ValueError(
                    "statname must have two digits indicating quantile"
                    "e.g. statname='quantile95' returns the 95th quantile"
                )
            quantile = int(statname[-2]) / 100
            statname = statname[:-2]
            print(quantile)
        else:
            quantile = None
        stats = {
            "mean": self._bkd.mean,
            "median": self._bkd.mean,
            "min": self._bkd.min,
            "max": self._bkd.max,
            "quantile": partial(self._bkd.quantile, q=quantile),
        }
        return stats[statname]

    def mean(self, statname: str = "mean") -> Array:
        stat_fun = self._get_stat_function(statname)
        return self._bkd.atleast1d(stat_fun(self._mean))

    def variance(self, statname: str = "mean") -> Array:
        stat_fun = self._get_stat_function(statname)
        return self._bkd.atleast1d(stat_fun(self._variance))

    def main_effects(self, statname: str = "mean") -> Array:
        stat_fun = self._get_stat_function(statname)
        return stat_fun(self._main_effects, axis=1)[:, None]

    def total_effects(self, statname: str = "mean") -> Array:
        stat_fun = self._get_stat_function(statname)
        return stat_fun(self._total_effects, axis=1)[:, None]

    def sobol_indices(self, statname: str = "mean") -> Array:
        stat_fun = self._get_stat_function(statname)
        return stat_fun(self._sobol_indices, axis=1)[:, None]

    def _prepare_boxplot_stats(
        self, sensitivity_indices: Array, labels: List[str]
    ) -> List[Dict]:
        nindices = sensitivity_indices.shape[0]
        if len(labels) != nindices:
            raise ValueError("must provide lable for each index")
        stats = [dict() for nn in range(nindices)]
        med = self._get_stat_function("quantile50")(
            sensitivity_indices, axis=1
        )
        q1 = self._get_stat_function("quantile25")(sensitivity_indices, axis=1)
        q3 = self._get_stat_function("quantile75")(sensitivity_indices, axis=1)
        whislo = self._get_stat_function("min")(sensitivity_indices, axis=1)
        whishi = self._get_stat_function("max")(sensitivity_indices, axis=1)
        for nn in range(nindices):
            stats[nn]["med"] = med[nn]
            stats[nn]["q1"] = q1[nn]
            stats[nn]["q3"] = q3[nn]
            # use whiskers for min and max instead of fliers
            stats[nn]["whislo"] = whislo[nn]
            stats[nn]["whishi"] = whishi[nn]
            stats[nn]["label"] = labels[nn]
        return stats

    def _plot_indices(self, ax, sensitivity_indices: Array, labels: List[str]):
        stats = self._prepare_boxplot_stats(sensitivity_indices, labels)
        bp = ax.bxp(
            stats,
            showfliers=False,
            showmeans=False,
            patch_artist=True,
            meanprops=dict(
                marker="o",
                markerfacecolor="blue",
                markeredgecolor="blue",
                markersize=12,
            ),
            medianprops=dict(color="red"),
        )
        ax.tick_params(axis="x", labelrotation=45)
        nindices = sensitivity_indices.shape[0]
        colors = ["gray"] * nindices
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        colors = ["red"] * nindices
        return bp

    def plot_main_effects(self, ax, labels: List[str] = None, nindices=None):
        if nindices is None:
            nindices = min(self._gp.nvars(), 5)
        if nindices > self._gp.nvars():
            raise ValueError("You cannot plot that many indices")
        # sort lagest to smallest by mean value of indices
        idx = self._bkd.flip(
            self._bkd.argsort(self.main_effects("mean")[:, 0])
        )
        main_effects = self._main_effects[idx, :]
        if labels is None:
            main_effect_indices = self._bkd.where(
                self._interaction_terms[:, self._main_effect_idx()] == 1
            )[0]
            # add 1 to use base 1 indexing of variables for plotting
            labels = [f"$z_{ii+1}$" for ii in main_effect_indices]
        return self._plot_indices(ax, main_effects, labels)

    def plot_total_effects(self, ax, labels: List[str] = None, nindices=None):
        if nindices is None:
            nindices = min(self._gp.nvars(), 5)
        if nindices > self._gp.nvars():
            raise ValueError("You cannot plot that many indices")
        # sort largest to smallest by mean value of indices
        idx = self._bkd.flip(
            self._bkd.argsort(self.total_effects("mean")[:, 0])
        )
        total_effects = self._total_effects[idx, :]
        if labels is None:
            # total effects are presorted by increasing variable index
            # so just get value from idx
            labels = [f"$z_{ii+1}$" for ii in idx]
        return self._plot_indices(ax, total_effects, labels)

    def _sobol_index_label(self, index: Array) -> str:
        active_varables = self._bkd.where(index > 0)[0]
        label = "({0})".format(",".join([str(idx) for idx in active_varables]))
        return label

    def plot_sobol_indices(self, ax, labels: List[str] = None, nindices=None):
        if nindices is None:
            nindices = min(self._interaction_terms.shape[1], 5)
        if nindices > self._interaction_terms.shape[1]:
            raise ValueError("You cannot plot that many indices")
        # sort largest to smallest by mean value of indices
        idx = self._bkd.flip(
            self._bkd.argsort(self.sobol_indices("mean")[:, 0])
        )
        sobol_indices = self._sobol_indices[idx, :]
        if labels is None:
            labels = []
            for ii in idx:
                labels.append(
                    self._sobol_index_label(self._interaction_terms[:, ii])
                )
        return self._plot_indices(ax, sobol_indices, labels)
