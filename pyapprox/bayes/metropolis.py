import numpy as np
# from tqdm import tqdm
from scipy.linalg import solve_triangular
from scipy.optimize import (
    minimize, differential_evolution)
import matplotlib.pyplot as plt
from functools import partial

from pyapprox.variables.marginals import get_distribution_info
from pyapprox.variables.joint import JointVariable
from pyapprox.analysis.visualize import setup_2d_cross_section_axes
from pyapprox.variables.joint import get_truncated_range
from pyapprox.bayes.hmc import hmc


def update_mean_and_covariance(
        new_draw, mean, cov, ndraws, sd, nugget):
    assert new_draw.ndim == 2 and new_draw.shape[1] == 1
    assert mean.ndim == 2 and mean.shape[1] == 1
    # cov already contains s_d*eps*I and is scaled by sd
    updated_mean = (ndraws*mean + new_draw)/(ndraws+1)
    nvars = new_draw.shape[0]
    updated_cov = (ndraws-1)/ndraws*cov
    updated_cov += sd*nugget*np.eye(nvars)/ndraws
    delta = new_draw-mean  # use mean and not updated mean
    updated_cov += sd*delta.dot(delta.T)/(ndraws+1)
    return updated_mean, updated_cov


def compute_mvn_cholesky_based_data(C):
    L = np.linalg.cholesky(C)
    L_inv = solve_triangular(L, np.eye(C.shape[0]), lower=True)
    logdet = 2*np.log(np.diag(L)).sum()
    return L, L_inv, logdet


def mvn_log_pdf(xx, m, C):
    if m.ndim == 1:
        m = m[:, None]
    if xx.ndim == 1:
        xx = xx[:, None]
        assert xx.ndim == 2 and m.ndim == 2
    assert m.shape[1] == 1

    if type(C) != tuple:
        L, L_inv, logdet = compute_mvn_cholesky_based_data(C)
    else:
        L, L_inv, logdet = C

    nvars = xx.shape[0]
    log2pi = np.log(2*np.pi)
    tmp = np.dot(L_inv, xx-m)
    MDsq = np.sum(tmp*tmp, axis=0)
    log_pdf = -0.5*(MDsq + nvars*log2pi + logdet)
    return log_pdf


def mvnrnd(m, C, nsamples=1, is_chol_factor=False):
    if is_chol_factor is False:
        L = np.linalg.cholesky(C)
    else:
        L = C
    if m.ndim == 1:
        m = m[:, None]
    nvars = m.shape[0]
    white_noise = np.random.normal(0, 1, (nvars, nsamples))
    return np.dot(L, white_noise) + m


def delayed_rejection(y0, proposal_chol_tuple, logpost_fun, cov_scaling,
                      log_f_y0):

    L = proposal_chol_tuple[0]
    y1 = mvnrnd(y0, L, is_chol_factor=True)
    if log_f_y0 is None:
        log_f_y0 = logpost_fun(y0)

    log_f_y1 = logpost_fun(y1)
    log_q1_y0_y1 = mvn_log_pdf(y0, y1, proposal_chol_tuple)
    log_q1_y1_y0 = mvn_log_pdf(y1, y0, proposal_chol_tuple)

    # the following is true because we are using a symmetric proposal
    assert np.allclose(log_q1_y0_y1, log_q1_y1_y0, atol=1e-15)

    log_numer_1 = log_f_y1 + log_q1_y0_y1
    log_denom_1 = log_f_y0 + log_q1_y1_y0
    # acceptance probability
    log_alpha_1_y0_y1 = log_numer_1 - log_denom_1
    randu_1 = np.random.uniform(0, 1)
    if (np.log(randu_1) < log_alpha_1_y0_y1):  # acceptance
        return y1, 1, log_f_y1

    # 1st rejection
    nvars = proposal_chol_tuple[0].shape[0]
    scaled_proposal_chol_tuple = (
        proposal_chol_tuple[0]*np.sqrt(cov_scaling),
        proposal_chol_tuple[1]/np.sqrt(cov_scaling),
        proposal_chol_tuple[2]+nvars*np.log(cov_scaling))

    # y2 = mvnrnd(y0, cov_scaling*proposal_cov)
    # q2 is an aribtraty proposal which is a function of y0 and y1
    # Below we ignore y1 and take proposal centered at y0
    # but could equally ignore y0 and take proposal centered at y1
    y2 = mvnrnd(y0, scaled_proposal_chol_tuple[0], is_chol_factor=True)

    log_f_y2 = logpost_fun(y2)
    log_q1_y1_y2 = mvn_log_pdf(y1, y2, proposal_chol_tuple)
    log_q1_y2_y1 = mvn_log_pdf(y2, y1, proposal_chol_tuple)

    # the following is true because we are using a symmetric proposal
    assert np.allclose(log_q1_y2_y1, log_q1_y1_y2, atol=1e-15)

    log_q2_y2_y0 = mvn_log_pdf(y2, y0, scaled_proposal_chol_tuple)
    log_q2_y0_y2 = mvn_log_pdf(y0, y2, scaled_proposal_chol_tuple)

    # the following is true because we are using a symmetric proposal
    assert np.allclose(log_q2_y2_y0, log_q2_y0_y2, atol=1e-15)

    log_alpha_1_y1_y2 = (log_f_y1 + log_q1_y2_y1) - (log_f_y2 + log_q1_y1_y2)
    if (np.log(1) <= log_alpha_1_y1_y2):
        return y0, 0, log_f_y0

    log_numer_2 = log_f_y2 + log_q1_y1_y2 + log_q2_y0_y2 + np.log(
        1-min(1, np.exp(log_alpha_1_y1_y2)))
    log_denom_2 = log_f_y0 + log_q1_y1_y0 + log_q2_y2_y0 + np.log(
        1-min(1, np.exp(log_alpha_1_y0_y1)))

    # acceptance probability
    log_alpha_2_y0_y2 = (log_numer_2 - log_denom_2)
    if (np.log(np.random.uniform(0, 1)) < log_alpha_2_y0_y2):  # acceptance
        return y2, 1, log_f_y2

    return y0, 0, log_f_y0


def DRAM(logpost_fun, init_sample, proposal_cov, nsamples,
         ndraws_init_update=20, nugget=1e-6, cov_scaling=1e-2,
         verbosity=0, sd=None):
    nvars = init_sample.shape[0]

    samples = np.empty((nvars, nsamples))
    if sd is None:
        sd = 2.4**2 / nvars  # page 341 of Haario. Stat Comput (2006) 16:339–354
    # do not scale initial proposal cov by sd.
    # Assume user provides a good guess
    proposal_cov = proposal_cov  # * sd
    proposal_chol_tuple = compute_mvn_cholesky_based_data(proposal_cov)
    logpost = logpost_fun(init_sample)
    sample_mean = init_sample
    sample_cov = np.zeros((nvars, nvars))

    # use init samples as first sample in chain
    ndraws = 0
    sample = init_sample
    samples[:, 0] = sample[:, 0]
    accepted = np.empty(nsamples)
    accepted[ndraws] = 1
    ndraws = +1

    # if verbosity > 0:
    #     pbar = tqdm(total=nsamples)
    #     pbar.update(1)
    while ndraws < nsamples:
        if ndraws % ndraws_init_update == 0:
            proposal_chol_tuple = compute_mvn_cholesky_based_data(sample_cov)
        sample, acc, logpost = delayed_rejection(
            sample, proposal_chol_tuple, logpost_fun, cov_scaling, logpost)
        samples[:, ndraws] = sample.squeeze()
        sample_mean, sample_cov = update_mean_and_covariance(
            sample, sample_mean, sample_cov, ndraws, sd, nugget)
        accepted[ndraws] = acc
        ndraws += 1
        # if verbosity > 0:
        #     pbar.update(1)
    return samples, accepted, sample_cov


def auto_correlation(time_series, lag):
    assert time_series.ndim == 1
    X_t = time_series[:-lag]
    assert lag < time_series.shape[0]-1
    X_tpk = time_series[lag:]
    corr = np.corrcoef(X_t, y=X_tpk)
    return corr[1, 0]


def plot_auto_correlations(timeseries, maxlag=100):
    nvars, ndata = timeseries.shape
    maxlag = min(maxlag, ndata-1)
    fig, axs = plt.subplots(1, nvars, figsize=(nvars*8, 6))
    lags = np.arange(0, maxlag+1)
    for ii in range(nvars):
        auto_correlations = [1]
        for lag in lags[1:]:
            auto_correlations.append(auto_correlation(timeseries[ii], lag))
        axs[ii].bar(lags, np.array(auto_correlations))


class MetropolisMCMCVariable(JointVariable):
    def __init__(self, variable, loglike, burn_fraction=0.1,
                 nsamples_per_tuning=100, algorithm="DRAM",
                 method_opts={}, verbosity=1, init_proposal_cov=None):
        self._variable = variable
        self._loglike = loglike
        self._burn_fraction = burn_fraction
        self._algorithm = algorithm
        self._nsamples_per_tuning = nsamples_per_tuning
        self._method_opts = method_opts
        self._verbosity = verbosity
        self._init_proposal_cov = init_proposal_cov

        # will be set each time self.rvs is called
        self._acceptance_rate = None

    @staticmethod
    def _univariate_logprior_grad(rv, x):
        if rv.dist.name == "beta":
            # todo only extract this info once
            scales, shapes = get_distribution_info(rv)[1:]
            loc, scale = scales["loc"], scales["scale"]
            a, b = shapes["a"], shapes["b"]
            val = (loc*(-(a+b-2))+x*(a+b-2)-a*scale+scale)/(
                (loc-x)*(loc+scale-x))
            return val
        if rv.dist.name == "uniform":
            return 0
        if rv.dist.name == "norm":
            mu, sigma = rv.args
            return -(x-mu)/sigma**2
        raise ValueError(f"rv type {rv.name} is not supported")

    def _logprior_grad(self, sample):
        grad = np.zeros_like(sample)
        for ii, rv in enumerate(self._variable.marginals()):
            grad[ii] += self._univariate_logprior_grad(rv, sample[ii, 0])
        return grad.T

    def _log_bayes_numerator(self, sample, return_grad=False):
        assert sample.ndim == 2 and sample.shape[1] == 1
        if not return_grad:
            loglike = self._loglike(sample)
        else:
            loglike, loglike_grad = self._loglike(sample, return_grad)
        if type(loglike) == np.ndarray:
            loglike = loglike.squeeze()
        # _pdf is faster but requires mapping of x to canonical domain
        logprior = self._variable.pdf(sample, log=True)[0, 0]
        if not return_grad:
            return loglike+logprior

        logprior_grad = self._logprior_grad(sample)
        return loglike+logprior, loglike_grad + logprior_grad

    def _rvs_dram(self, init_sample, init_proposal_cov, num_samples,
                  nburn_samples):
        nugget = self._method_opts.get("nugget", 1e-6)
        cov_scaling = self._method_opts.get("cov_scaling", 1e-2)
        sd = self._method_opts.get("sd", None)
        # hack remove self._sample_covariance
        samples, accepted, self._sample_covariance = DRAM(
            self._log_bayes_numerator, init_sample, init_proposal_cov,
            num_samples+nburn_samples, self._nsamples_per_tuning, nugget,
            cov_scaling, verbosity=self._verbosity, sd=sd)
        # print(accepted)
        # print(accepted[nburn_samples:].sum(), num_samples, nburn_samples)
        acceptance_rate = accepted[nburn_samples:].sum()/num_samples
        return samples, acceptance_rate

    def _log_bayes_numerator_hmc(self, sample):
        val, grad = self._log_bayes_numerator(
            sample[:, None], return_grad=True)
        return val, grad[0]

    def _rvs_hmc(self, init_sample, init_momentum, num_samples):
        hmc_opts = self._method_opts.copy()
        hmc_opts["num_samples"] = num_samples
        result = hmc(
            init_sample[:, 0], init_momentum[:, 0],
            self._log_bayes_numerator_hmc, [],
            hmc_opts, np.random)
        acceptance_rate = (result["num_acc"])/(
            result["num_acc"]+result["num_rej"])
        return result["samples"].T, acceptance_rate

    def rvs(self, num_samples, init_sample=None):
        if init_sample is None:
            # init_sample = self._variable.get_statistics("mean")
            init_sample = self.maximum_aposteriori_point()
        if self._init_proposal_cov is None:
            init_proposal_cov = np.diag(
                self._variable.get_statistics("std")[:, 0]**2)
        if init_proposal_cov.shape[0] != self._variable.num_vars():
            raise ValueError("init_proposal_cov specified has wrong shape")
        # print(num_samples, self._burn_fraction)
        nburn_samples = int(np.ceil(num_samples*self._burn_fraction))
        if self._algorithm == "DRAM":
            samples, acceptance_rate = self._rvs_dram(
                init_sample, init_proposal_cov, num_samples, nburn_samples)
        elif self._algorithm == "hmc":
            init_momentum = np.random.normal(0, 1, (init_sample.shape[0], 1))
            samples, acceptance_rate = self._rvs_hmc(
                init_sample, init_momentum, num_samples)
            # L, eps = self._method_opts["L"], self._method_opts["eps"]
            # return hamiltonian_monte_carlo(
            #     L, eps, self._log_bayes_numerator, init_sample, num_samples)
        else:
            # TODO allow all combinations of adaptive and delayed
            # rejection metropolis
            raise ValueError(f"Algorithm {self._algorithm} not supported")
        self._acceptance_rate = acceptance_rate
        return samples[:, nburn_samples:]

    def maximum_aposteriori_point(self, init_guess=None):
        import inspect
        def obj(x):
            return -self._log_bayes_numerator(x[:, None])
        return_grad = None

        if init_guess is None:
            bounds = self._variable.get_statistics("interval", 1)
            trunc_bounds = self._variable.get_statistics(
                "interval", 1-1e-6)
            for ii in range(bounds.shape[0]):
                if bounds[ii, 0] == -np.inf:
                    bounds[ii, 0] = trunc_bounds[ii, 0]
                if bounds[ii, 1] == np.inf:
                    bounds[ii, 1] = trunc_bounds[ii, 1]
            bounds = [(bb[0], bb[1]) for bb in bounds]
            res = differential_evolution(
                obj, bounds, maxiter=100, popsize=15)
            # res = dual_annealing(
            #     obj, bounds, maxiter=100, maxfun=10000)
            init_guess = res.x[:, None]
            # init_guess = self._variable.get_statistics("mean")

        if "return_grad" in inspect.getfullargspec(self._loglike).args:
            def obj(x):
                vals, grad = self._log_bayes_numerator(x[:, None], True)
                return -vals, -grad[0, :]
            return_grad = True

        assert init_guess.ndim == 2 and init_guess.shape[1] == 1
        assert init_guess.shape[0] == self._variable.num_vars()
        init_guess = init_guess[:, 0]
        bounds = self._variable.get_statistics("interval", 1)
        res = minimize(obj, init_guess, jac=return_grad, method="l-bfgs-b",
                       options={"disp": False}, bounds=bounds)
        MAP = res.x[:, None]
        return MAP

    @staticmethod
    def plot_traces(samples):
        nvars = samples.shape[0]
        fig, axs = plt.subplots(1, nvars, figsize=(nvars*8, 6))
        for ii in range(nvars):
            axs[ii].plot(
                np.arange(1, samples.shape[1]+1), samples[ii, :], lw=0.5)

    @staticmethod
    def plot_auto_correlations(samples):
        return plot_auto_correlations(samples)

    def plot_2d_marginals(self, samples, variable_pairs=None,
                          subplot_tuple=None, unbounded_alpha=0.99,
                          map_sample=None, true_sample=None):
        fig, axs, variable_pairs = setup_2d_cross_section_axes(
            self._variable, variable_pairs, subplot_tuple)
        all_variables = self._variable.marginals()
        nsamples = samples.shape[1]

        for ii, var in enumerate(all_variables):
            axs[ii, ii].axis("off")
        #     lb, ub = get_truncated_range(var, unbounded_alpha)
        #     # axs[ii][ii].set_xlim(lb, ub)
        #     axs[ii][ii].hist(samples[ii, :], bins=max(10, nsamples//100))
        #     if map_sample is not None:
        #         axs[ii][ii].plot(map_sample[ii, 0], 0, 'ro')
        #     if true_sample is not None:
        #         axs[ii][ii].plot(true_sample[ii, 0], 0, 'gD')

        for ii, pair in enumerate(variable_pairs):
            # use pair[1] for x and pair[0] for y because we reverse
            # pairs above
            var1, var2 = all_variables[pair[1]], all_variables[pair[0]]
            axs[pair[1], pair[0]].axis("off")
            lb1, ub1 = get_truncated_range(var1, unbounded_alpha)
            lb2, ub2 = get_truncated_range(var2, unbounded_alpha)
            ax = axs[pair[0]][pair[1]]
            # ax.set_xlim(lb1, ub1)
            # ax.set_ylim(lb2, ub2)
            ax.plot(samples[pair[1], :], samples[pair[0], :],
                    'ko', alpha=0.4)
            if map_sample is not None:
                ax.plot(map_sample[pair[1], 0], map_sample[pair[0], 0],
                        marker='X', color='r')
            if true_sample is not None:
                ax.plot(true_sample[pair[1], 0], true_sample[pair[0], 0],
                        marker='D', color='g')
        return fig, axs


def leapfrog(loglikefun, theta, momentum, eps):
    updated_momentum = momentum + eps/2 * loglikefun(
        theta, return_grad=True)[1]
    updated_theta = theta+eps*updated_momentum
    updated_momentum += eps/2 * loglikefun(updated_theta, return_grad=True)[1]
    return updated_theta, updated_momentum


def hamiltonian_monte_carlo(L, eps, logpost_fun, init_sample, nsamples):
    nvars = init_sample.shape[0]
    samples = np.empty((nvars, nsamples))
    sample = init_sample
    samples[:, 0] = sample[:, 0]
    accepted = np.empty(nsamples, dtype=bool)
    for ii in range(nsamples):
        momentum = np.random.normal(0, 1, (nvars, 1))
        log_denom = logpost_fun(sample)-momentum.T.dot(momentum)/2
        new_sample = sample
        for jj in range(L):
            new_sample, momentum = leapfrog(
                logpost_fun, new_sample, momentum, eps)
        log_numer = logpost_fun(new_sample)-momentum.T.dot(momentum)/2
        alpha = min(1, np.exp(log_numer)/np.exp(log_denom))
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


def _unnormalized_pdf_for_marginalization(
        variable, loglike, sub_indices, samples):
    marginal_pdf_vals = variable.evaluate("pdf", samples)
    sub_pdf_vals = marginal_pdf_vals[sub_indices, :].prod(axis=0)
    nll_vals = loglike(samples).squeeze()
    # only use sub_pdf_vals. The other vals will be accounted for
    # with quadrature rule used to marginalize
    return np.exp(nll_vals+np.log(sub_pdf_vals))[:, None]


def plot_unnormalized_2d_marginals(
        variable, loglike, nsamples_1d=100, variable_pairs=None,
        subplot_tuple=None, qoi=0, num_contour_levels=20,
        plot_samples=None, unbounded_alpha=0.995):
    from pyapprox.variables.joint import get_truncated_range
    from pyapprox.surrogates.interp.indexing import (
        compute_anova_level_indices)
    from pyapprox.util.visualization import get_meshgrid_samples

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
        nfig_rows, nfig_cols = (
            variable.num_vars(), variable.num_vars())
    else:
        nfig_rows, nfig_cols = subplot_tuple

    if nfig_rows*nfig_cols < len(variable_pairs):
        raise ValueError("Number of subplots is insufficient")

    fig, axs = plt.subplots(
        nfig_rows, nfig_cols, figsize=(nfig_cols*8, nfig_rows*6))
    all_variables = variable.marginals()

    # if plot_samples is not None and type(plot_samples) == np.ndarray:
    #     plot_samples = [
    #         [plot_samples, {"c": "k", "marker": "o", "alpha": 0.4}]]

    for ii, var in enumerate(all_variables):
        lb, ub = get_truncated_range(var, unbounded_alpha=unbounded_alpha)
        quad_degrees = np.array([20]*(variable.num_vars()-1))
        # quad_degrees = np.array([10]*(variable.num_vars()-1))
        samples_ii = np.linspace(lb, ub, nsamples_1d)
        from pyapprox.surrogates.polychaos.gpc import (
            _marginalize_function_1d, _marginalize_function_nd)
        values = _marginalize_function_1d(
            partial(_unnormalized_pdf_for_marginalization,
                    variable, loglike, np.array([ii])),
            variable, quad_degrees, ii, samples_ii, qoi=0)
        axs[ii][ii].plot(samples_ii, values)
        if plot_samples is not None:
            for s in plot_samples:
                axs[ii][ii].scatter(s[0][ii, :], s[0][ii, :]*0, **s[1])
        axs[ii][ii].set_yticks([])

    for ii, pair in enumerate(variable_pairs):
        # use pair[1] for x and pair[0] for y because we reverse
        # pairs above
        var1, var2 = all_variables[pair[1]], all_variables[pair[0]]
        axs[pair[1], pair[0]].axis("off")
        lb1, ub1 = get_truncated_range(
            var1, unbounded_alpha=unbounded_alpha)
        lb2, ub2 = get_truncated_range(
            var2, unbounded_alpha=unbounded_alpha)
        X, Y, samples_2d = get_meshgrid_samples(
            [lb1, ub1, lb2, ub2], nsamples_1d)
        quad_degrees = np.array([10]*(variable.num_vars()-2))
        if variable.num_vars() > 2:
            values = _marginalize_function_nd(
                partial(_unnormalized_pdf_for_marginalization,
                        variable, loglike, np.array([pair[1], pair[0]])),
                variable, quad_degrees, np.array([pair[1], pair[0]]),
                samples_2d, qoi=qoi)
        else:
            values = _unnormalized_pdf_for_marginalization(
                variable, loglike, np.array([pair[1], pair[0]]), samples_2d)
        Z = np.reshape(values, (X.shape[0], X.shape[1]))
        ax = axs[pair[0]][pair[1]]
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, r"$(\mathrm{%d, %d})$" % (pair[1], pair[0]),
                transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.contourf(
            X, Y, Z, levels=np.linspace(Z.min(), Z.max(),
                                        num_contour_levels),
            cmap='jet')
        if plot_samples is not None:
            for s in plot_samples:
                # use pair[1] for x and pair[0] for y because we reverse
                # pairs above
                axs[pair[0]][pair[1]].scatter(
                    s[0][pair[1], :], s[0][pair[0], :], **s[1])

    return fig, axs


class GaussianLogLike(object):
    r"""
    A Gaussian log-likelihood function for a model with parameters given in
    sample
    """

    def __init__(self, model, data, noise_covar, model_jac=None):
        r"""
        Initialise the Op with various things that our log-likelihood
        function requires.

        Parameters
        ----------
        model : callable
            The model relating the data and noise

        data : np.ndarray (nobs)
            The "observed" data

        noise_covar : float, np.ndarray (nobs), np.ndarray (nobs,nobs)
            The noise covariance

        model_jac : callable
            The Jacobian of the model with respect to the parameters
        """
        self.model = model
        self.model_jac = model_jac
        self.data = data
        assert self.data.ndim == 1
        self.ndata = data.shape[0]
        self.noise_covar_inv, self.log_noise_covar_det = (
            self.noise_covariance_inverse(noise_covar))

    def noise_covariance_inverse(self, noise_covar):
        if np.isscalar(noise_covar):
            return 1/noise_covar, np.log(noise_covar)
        if noise_covar.ndim == 1:
            assert noise_covar.shape[0] == self.data.shape[0]
            return 1/noise_covar, np.log(np.prod(noise_covar))
        elif noise_covar.ndim == 2:
            assert noise_covar.shape == (self.ndata, self.ndata)
            return np.linalg.inv(noise_covar), np.log(
                np.linalg.det(noise_covar))
        raise ValueError("noise_covar has the wrong shape")

    # def noise_covariance_determinant(self, noise_covar):
    #     r"""The determinant is only necessary in log likelihood if the noise
    #     covariance has a hyper-parameter which is being inferred which is
    #     not currently supported"""
    #     if np.isscalar(noise_covar):
    #         determinant = noise_covar**self.ndata
    #     elif noise_covar.ndim==1:
    #         determinant = np.prod(noise_covar)
    #     else:
    #         determinant = np.linalg.det(noise_covar)
    #     return determinant

    def __call__(self, samples, return_grad=False):
        nsamples = samples.shape[1]
        model_vals = self.model(samples)
        assert model_vals.ndim == 2
        assert model_vals.shape[1] == self.ndata
        vals = np.empty((nsamples, 1))
        for ii in range(nsamples):
            residual = self.data - model_vals[ii, :]
            if np.isscalar(self.noise_covar_inv):
                tmp = self.noise_covar_inv*residual
            elif self.noise_covar_inv.ndim == 1:
                tmp = self.noise_covar_inv[:, None]*residual
                # vals[ii] = (residual.T*self.noise_covar_inv).dot(residual)
            else:
                tmp = self.noise_covar_inv.dot(residual)
                # vals[ii] = residual.T.dot(self.noise_covar_inv).dot(residual)
            vals[ii] = residual.dot(tmp)
        vals += self.ndata*np.log(2*np.pi) + self.log_noise_covar_det
        vals *= -0.5
        vals = np.atleast_2d(vals)
        if not return_grad:
            return vals

        if nsamples != 1:
            raise ValueError("nsamples must be 1 when return_grad is True")
        if self.model_jac is None:
            raise ValueError("model_jac is none but gradient requested")
        grad = tmp.dot(self.model_jac(samples))
        return vals, grad


def loglike_from_negloglike(negloglike, samples, jac=False):
    if not jac:
        return -negloglike(samples)
    vals, grads = negloglike(samples, jac=jac)
    return -vals, -grads
