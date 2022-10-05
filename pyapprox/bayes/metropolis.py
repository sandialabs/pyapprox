import numpy as np
from tqdm import tqdm
from scipy.linalg import solve_triangular

from pyapprox.variables.joint import JointVariable


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
         verbosity=0):
    nvars = init_sample.shape[0]

    samples = np.empty((nvars, nsamples))
    sd = 2.4**2 / nvars  # page 341 of Haario. Stat Comput (2006) 16:339–354
    # do not scale initial proposal cov by sd.
    # Assume user provides a good guess
    proposal_cov = proposal_cov  # * sd
    proposal_chol_tuple = compute_mvn_cholesky_based_data(proposal_cov)
    logpost = logpost_fun(init_sample)
    sample_mean = init_sample
    sample_cov = np.zeros((nvars, nvars))

    # use init samples as first sample in chain
    ndraws = 1
    sample = init_sample
    samples[:, 0] = sample[:, 0]
    accepted = np.empty(nsamples)
    accepted[ndraws] = 1
    
    if verbosity > 0:
        pbar = tqdm(total=nsamples)
        pbar.update(1)
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
        if verbosity > 0:
            pbar.update(1)
    return samples, accepted


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

    def _log_bayes_numerator(self, sample):
        assert sample.ndim == 2 and sample.shape[1] == 1
        loglike = self._loglike(sample)
        if type(loglike) == np.ndarray:
            loglike = loglike.squeeze()
        logprior = self._variable._pdf(sample, log=True)[0, 0]
        return loglike+logprior

    def rvs(self, num_samples):
        init_sample = self._variable.get_statistics("mean")
        if self._init_proposal_cov is None:
            init_proposal_cov = np.diag(
                self._variable.get_statistics("std")[:, 0]**2)
        if init_proposal_cov.shape[0] != self._variable.num_vars():
            raise ValueError("init_proposal_cov specified has wrong shape")
        nburn_samples = int(np.ceil(num_samples*self._burn_fraction))
        if self._algorithm == "DRAM":
            nugget = self._method_opts.get("nugget", 1e-6)
            cov_scaling = self._method_opts.get("cov_scaling", 1e-2)
            samples, accepted = DRAM(
                self._log_bayes_numerator, init_sample, init_proposal_cov,
                num_samples+nburn_samples, self._nsamples_per_tuning, nugget,
                cov_scaling, verbosity=self._verbosity)
            acceptance_rate = accepted[nburn_samples:].sum()/num_samples
        else:
            # TODO allow all combinations of adaptive and delayed rejection metropolis
            raise ValueError(f"Algorithm {self._algorithm} not supported")
        self._acceptance_rate = acceptance_rate
        return samples[:, nburn_samples:]
