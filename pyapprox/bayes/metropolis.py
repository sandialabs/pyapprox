from typing import Tuple, Union, List
import math

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.variables.joint import JointVariable
from pyapprox.analysis.visualize import setup_2d_cross_section_axes
from pyapprox.bayes.hmc import hmc
from pyapprox.bayes.likelihood import LogLikelihood, LogUnNormalizedPosterior
from pyapprox.util.backends.template import Array
from pyapprox.util.backends.numpy import NumpyMixin


def update_mean_and_covariance(new_draw, mean, cov, ndraws, sd, nugget, bkd):
    assert new_draw.ndim == 2 and new_draw.shape[1] == 1
    assert mean.ndim == 2 and mean.shape[1] == 1
    # cov already contains s_d*eps*I and is scaled by sd
    updated_mean = (ndraws * mean + new_draw) / (ndraws + 1)
    nvars = new_draw.shape[0]
    updated_cov = (ndraws - 1) / ndraws * cov
    updated_cov += sd * nugget * bkd.eye(nvars) / ndraws
    delta = new_draw - mean  # use mean and not updated mean
    updated_cov += sd * delta @ delta.T / (ndraws + 1)
    return updated_mean, updated_cov


def compute_mvn_cholesky_based_data(C, bkd) -> Tuple[Array, Array, Array]:
    L = bkd.cholesky(C)
    L_inv = bkd.solve_triangular(L, bkd.eye(C.shape[0]), lower=True)
    logdet = 2 * bkd.log(bkd.diag(L)).sum()
    return L, L_inv, logdet


def mvn_log_pdf(
    xx: Array, m: Array, C: Union[Tuple[Array, Array, Array], Array], bkd
) -> Array:
    if m.ndim == 1:
        m = m[:, None]
    if xx.ndim == 1:
        xx = xx[:, None]
        assert xx.ndim == 2 and m.ndim == 2
    assert m.shape[1] == 1

    if not isinstance(C, tuple):
        L, L_inv, logdet = compute_mvn_cholesky_based_data(C, bkd)
    else:
        L, L_inv, logdet = C

    nvars = xx.shape[0]
    log2pi = math.log(2 * np.pi)
    tmp = L_inv @ (xx - m)
    MDsq = bkd.sum(tmp * tmp, axis=0)
    log_pdf = -0.5 * (MDsq + nvars * log2pi + logdet)
    return log_pdf


def mvnrnd(
    m: Array,
    C: Array,
    nsamples: int = 1,
    is_chol_factor: bool = False,
    bkd=NumpyMixin,
) -> Array:
    if is_chol_factor is False:
        L = bkd.cholesky(C)
    else:
        L = C
    if m.ndim == 1:
        m = m[:, None]
    nvars = m.shape[0]
    white_noise = bkd.asarray(np.random.normal(0.0, 1.0, (nvars, nsamples)))
    return L @ white_noise + m


def delayed_rejection(
    y0: Array,
    proposal_chol_tuple: Tuple[Array, Array, Array],
    logpost_fun,
    cov_scaling: float,
    log_f_y0: Array,
    prior_bounds: Array,
    bkd,
):

    L = proposal_chol_tuple[0]
    while True:
        y1 = mvnrnd(y0, L, is_chol_factor=True, bkd=bkd)
        if bkd.all(y1[:, 0] >= prior_bounds[:, 0]) and bkd.all(
            y1[:, 0] <= prior_bounds[:, 1]
        ):
            break
    if log_f_y0 is None:
        log_f_y0 = logpost_fun(y0)

    log_f_y1 = logpost_fun(y1)
    log_q1_y0_y1 = mvn_log_pdf(y0, y1, proposal_chol_tuple, bkd)
    log_q1_y1_y0 = mvn_log_pdf(y1, y0, proposal_chol_tuple, bkd)

    # the following is true because we are using a symmetric proposal
    assert bkd.allclose(log_q1_y0_y1, log_q1_y1_y0, atol=1e-15)

    log_numer_1 = log_f_y1 + log_q1_y0_y1
    log_denom_1 = log_f_y0 + log_q1_y1_y0
    # acceptance probability
    log_alpha_1_y0_y1 = log_numer_1 - log_denom_1
    randu_1 = np.random.uniform(0, 1)
    if np.log(randu_1) < log_alpha_1_y0_y1:  # acceptance
        return y1, 1, log_f_y1

    # 1st rejection
    nvars = proposal_chol_tuple[0].shape[0]
    cov_scaling = bkd.asarray(cov_scaling)
    scaled_proposal_chol_tuple = (
        proposal_chol_tuple[0] * bkd.sqrt(cov_scaling),
        proposal_chol_tuple[1] / bkd.sqrt(cov_scaling),
        proposal_chol_tuple[2] + nvars * bkd.log(cov_scaling),
    )

    # y2 = mvnrnd(y0, cov_scaling*proposal_cov)
    # q2 is an aribtraty proposal which is a function of y0 and y1
    # Below we ignore y1 and take proposal centered at y0
    # but could equally ignore y0 and take proposal centered at y1
    while True:
        y2 = mvnrnd(
            y0, scaled_proposal_chol_tuple[0], is_chol_factor=True, bkd=bkd
        )
        if bkd.all(y2[:, 0] >= prior_bounds[:, 0]) and bkd.all(
            y2[:, 0] <= prior_bounds[:, 1]
        ):
            break

    log_f_y2 = logpost_fun(y2)
    log_q1_y1_y2 = mvn_log_pdf(y1, y2, proposal_chol_tuple, bkd)
    log_q1_y2_y1 = mvn_log_pdf(y2, y1, proposal_chol_tuple, bkd)

    # the following is true because we are using a symmetric proposal
    assert bkd.allclose(log_q1_y2_y1, log_q1_y1_y2, atol=1e-15)

    log_q2_y2_y0 = mvn_log_pdf(y2, y0, scaled_proposal_chol_tuple, bkd)
    log_q2_y0_y2 = mvn_log_pdf(y0, y2, scaled_proposal_chol_tuple, bkd)

    # the following is true because we are using a symmetric proposal
    assert bkd.allclose(log_q2_y2_y0, log_q2_y0_y2, atol=1e-15)

    log_alpha_1_y1_y2 = (log_f_y1 + log_q1_y2_y1) - (log_f_y2 + log_q1_y1_y2)
    if math.log(1) <= log_alpha_1_y1_y2:
        return y0, 0, log_f_y0
    log_numer_2 = (
        log_f_y2
        + log_q1_y1_y2
        + log_q2_y0_y2
        + bkd.log(1.0 - min(1.0, bkd.exp(log_alpha_1_y1_y2)))
    )
    log_denom_2 = (
        log_f_y0
        + log_q1_y1_y0
        + log_q2_y2_y0
        + bkd.log(1 - min(1, bkd.exp(log_alpha_1_y0_y1)))
    )

    # acceptance probability
    log_alpha_2_y0_y2 = log_numer_2 - log_denom_2
    if np.log(np.random.uniform(0, 1)) < log_alpha_2_y0_y2:  # acceptance
        return y2, 1, log_f_y2

    return y0, 0, log_f_y0


def DRAM(
    logpost_fun,
    init_sample,
    proposal_cov,
    nsamples,
    prior_bounds,
    ndraws_init_update=20,
    nugget=1e-6,
    cov_scaling=1e-2,
    verbosity=0,
    sd=None,
    bkd=NumpyMixin,
):
    nvars = init_sample.shape[0]

    samples = bkd.empty((nvars, nsamples))
    if sd is None:
        sd = (
            2.4**2 / nvars
        )  # page 341 of Haario. Stat Comput (2006) 16:339–354
    # do not scale initial proposal cov by sd.
    # Assume user provides a good guess
    proposal_cov = proposal_cov  # * sd
    proposal_chol_tuple = compute_mvn_cholesky_based_data(proposal_cov, bkd)
    logpost = logpost_fun(init_sample)
    sample_mean = init_sample
    sample_cov = bkd.zeros((nvars, nvars))

    # use init samples as first sample in chain
    ndraws = 0
    sample = init_sample
    samples[:, 0] = sample[:, 0]
    accepted = bkd.empty(nsamples)
    accepted[ndraws] = 1
    ndraws = +1

    # if verbosity > 0:
    #     pbar = tqdm(total=nsamples)
    #     pbar.update(1)
    while ndraws < nsamples:
        if ndraws % ndraws_init_update == 0:
            proposal_chol_tuple = compute_mvn_cholesky_based_data(
                sample_cov, bkd
            )
        sample, acc, logpost = delayed_rejection(
            sample,
            proposal_chol_tuple,
            logpost_fun,
            cov_scaling,
            logpost,
            prior_bounds,
            bkd,
        )
        samples[:, ndraws] = sample.squeeze()
        sample_mean, sample_cov = update_mean_and_covariance(
            sample, sample_mean, sample_cov, ndraws, sd, nugget, bkd
        )
        accepted[ndraws] = acc
        ndraws += 1
        # if verbosity > 0:
        #     pbar.update(1)
    return samples, accepted, sample_cov


def auto_correlation(time_series, lag):
    assert time_series.ndim == 1
    X_t = time_series[:-lag]
    assert lag < time_series.shape[0] - 1
    X_tpk = time_series[lag:]
    corr = np.corrcoef(X_t, y=X_tpk)
    return corr[1, 0]


def plot_auto_correlations(timeseries, maxlag=100):
    nvars, ndata = timeseries.shape
    maxlag = min(maxlag, ndata - 1)
    fig, axs = plt.subplots(1, nvars, figsize=(nvars * 8, 6))
    lags = np.arange(0, maxlag + 1)
    for ii in range(nvars):
        auto_correlations = [1]
        for lag in lags[1:]:
            auto_correlations.append(auto_correlation(timeseries[ii], lag))
        axs[ii].bar(lags, np.array(auto_correlations))


class MetropolisMCMCVariable(JointVariable):
    def __init__(
        self,
        prior: JointVariable,
        loglike: LogLikelihood,
        burn_fraction: float = 0.1,
        nsamples_per_tuning: int = 100,
        algorithm: str = "DRAM",
        method_opts: dict = {},
        verbosity: int = 1,
        init_proposal_cov: Array = None,
    ):
        super().__init__(prior._bkd)
        self._prior = prior
        self._loglike = loglike
        self._burn_fraction = burn_fraction
        self._algorithm = algorithm
        self._nsamples_per_tuning = nsamples_per_tuning
        self._method_opts = method_opts
        self._verbosity = verbosity
        self._init_proposal_cov = init_proposal_cov
        self._log_unnormalized_post = LogUnNormalizedPosterior(
            self._loglike, self._prior
        )

        # will be set each time self.rvs is called
        self._acceptance_rate = None

    def nvars(self) -> int:
        return self._prior.nvars()

    def _rvs_dram(
        self, init_sample, init_proposal_cov, nsamples, nburn_samples
    ):
        nugget = self._method_opts.get("nugget", 1e-6)
        cov_scaling = self._method_opts.get("cov_scaling", 1e-2)
        sd = self._method_opts.get("sd", None)
        # hack remove self._sample_covariance
        samples, accepted, self._sample_covariance = DRAM(
            self._log_unnormalized_post,
            init_sample,
            init_proposal_cov,
            nsamples + nburn_samples,
            self._prior.interval(1),
            self._nsamples_per_tuning,
            nugget,
            cov_scaling,
            verbosity=self._verbosity,
            sd=sd,
            bkd=self._bkd,
        )
        # print(accepted)
        # print(accepted[nburn_samples:].sum(), nsamples, nburn_samples)
        acceptance_rate = accepted[nburn_samples:].sum() / nsamples
        return samples, acceptance_rate

    def maximum_aposteriori_point(self, iterate: Array = None):
        return self._log_unnormalized_post.maximum_aposteriori_point(iterate)

    def _log_bayes_numerator_hmc(self, sample):
        val = self._log_unnormalized_post(sample[:, None])
        grad = self._log_unnormalized_post.jacobian(sample[:, None])[0]
        return val, grad

    def _rvs_hmc(self, init_sample, init_momentum, nsamples):
        hmc_opts = self._method_opts.copy()
        hmc_opts["nsamples"] = nsamples
        result = hmc(
            init_sample[:, 0],
            init_momentum[:, 0],
            self._log_bayes_numerator_hmc,
            [],
            hmc_opts,
            np.random,
        )
        acceptance_rate = (result["nacc"]) / (result["nacc"] + result["nrej"])
        return result["samples"].T, acceptance_rate

    def rvs(self, nsamples, init_sample=None):
        if init_sample is None:
            init_sample = self.maximum_aposteriori_point()
        if self._init_proposal_cov is None:
            init_proposal_cov = self._prior.covariance()
        if init_proposal_cov.shape[0] != self._prior.nvars():
            raise ValueError("init_proposal_cov specified has wrong shape")
        # print(nsamples, self._burn_fraction)
        nburn_samples = int(np.ceil(nsamples * self._burn_fraction))
        if self._algorithm == "DRAM":
            samples, acceptance_rate = self._rvs_dram(
                init_sample, init_proposal_cov, nsamples, nburn_samples
            )
        elif self._algorithm == "hmc":
            raise NotImplementedError("Tests not passing")
            init_momentum = np.random.normal(0, 1, (init_sample.shape[0], 1))
            samples, acceptance_rate = self._rvs_hmc(
                init_sample, init_momentum, nsamples
            )
            # L, eps = self._method_opts["L"], self._method_opts["eps"]
            # return hamiltonian_monte_carlo(
            #     L, eps, self._log_bayes_numerator, init_sample, nsamples)
        else:
            # TODO allow all combinations of adaptive and delayed
            # rejection metropolis
            raise ValueError(f"Algorithm {self._algorithm} not supported")
        self._acceptance_rate = acceptance_rate
        return samples[:, nburn_samples:]

    @staticmethod
    def plot_traces(samples):
        nvars = samples.shape[0]
        fig, axs = plt.subplots(1, nvars, figsize=(nvars * 8, 6))
        for ii in range(nvars):
            axs[ii].plot(
                np.arange(1, samples.shape[1] + 1), samples[ii, :], lw=0.5
            )

    @staticmethod
    def plot_auto_correlations(samples):
        return plot_auto_correlations(samples)

    def plot_2d_marginals(
        self,
        samples: Array,
        variable_pairs: List = None,
        subplot_tuple: Tuple = None,
        unbounded_alpha: float = 0.99,
        map_sample: Array = None,
        true_sample: Array = None,
        nsamples_per_bin: int = 100,
    ):
        if samples.shape[0] != self._prior.nvars():
            raise ValueError(
                f"{samples.shape=} does not match "
                "prior.nvars()={self._prior.nvars()}"
            )
        fig, axs, variable_pairs = setup_2d_cross_section_axes(
            self._prior, variable_pairs, subplot_tuple
        )
        all_variables = self._prior.marginals()

        # for ii, var in enumerate(all_variables):
        #     axs[ii, ii].axis("off")
        if samples.shape[1] / nsamples_per_bin < 10:
            raise ValueError("Not enough samples to create at least 10 bins")
        for ii, pair in enumerate(variable_pairs):
            if pair[0] == pair[1]:
                axs[pair[0]][pair[1]].hist(
                    samples[pair[0]],
                    bins=int(samples.shape[1] / nsamples_per_bin),
                )
                continue

            # use pair[1] for x and pair[0] for y because we reverse
            # pairs above
            var1, var2 = all_variables[pair[1]], all_variables[pair[0]]
            axs[pair[1], pair[0]].axis("off")
            lb1, ub1 = var1.truncated_range(unbounded_alpha)
            lb2, ub2 = var2.truncated_range(unbounded_alpha)
            ax = axs[pair[0]][pair[1]]
            # ax.set_xlim(lb1, ub1)
            # ax.set_ylim(lb2, ub2)
            # print(samples.shape, pair)
            ax.plot(samples[pair[1], :], samples[pair[0], :], "ko", alpha=0.4)
            if map_sample is not None:
                ax.plot(
                    map_sample[pair[1], 0],
                    map_sample[pair[0], 0],
                    marker="X",
                    color="r",
                )
            if true_sample is not None:
                ax.plot(
                    true_sample[pair[1], 0],
                    true_sample[pair[0], 0],
                    marker="D",
                    color="g",
                )
        return fig, axs

    def __repr__(self) -> int:
        return "{0}(nvars={1}, likelihood={2}, burn_fraction={3})".format(
            self.__class__.__name__,
            self.nvars(),
            self._loglike,
            self._burn_fraction,
        )


def leapfrog(loglikefun, theta, momentum, eps):
    updated_momentum = (
        momentum + eps / 2 * loglikefun(theta, return_grad=True)[1]
    )
    updated_theta = theta + eps * updated_momentum
    updated_momentum += (
        eps / 2 * loglikefun(updated_theta, return_grad=True)[1]
    )
    return updated_theta, updated_momentum


def hamiltonian_monte_carlo(L, eps, logpost_fun, init_sample, nsamples):
    nvars = init_sample.shape[0]
    samples = np.empty((nvars, nsamples))
    sample = init_sample
    samples[:, 0] = sample[:, 0]
    accepted = np.empty(nsamples, dtype=bool)
    for ii in range(nsamples):
        momentum = np.random.normal(0, 1, (nvars, 1))
        log_denom = logpost_fun(sample) - momentum.T @ momentum / 2
        new_sample = sample
        for jj in range(L):
            new_sample, momentum = leapfrog(
                logpost_fun, new_sample, momentum, eps
            )
        log_numer = logpost_fun(new_sample) - momentum.T @ momentum / 2
        alpha = min(1, np.exp(log_numer) / np.exp(log_denom))
        u = np.random.uniform(0, 1)
        if u < alpha:
            samples[:, ii] = new_sample
            accepted[ii] = True
            sample = new_sample
        else:
            samples[:, ii] = sample
            accepted[ii] = False

    return samples


# class HamiltonianMonteCarlo(MetropolisMCMCVariable):
#     def rvs(self, nsamples, init_sample=None):
#         L, eps = self._method_opts["L"], self._method_opts["eps"]
#         return hamiltonian_monte_carlo(
#             L, eps, self._log_bayes_numerator, init_sample, nsamples)
