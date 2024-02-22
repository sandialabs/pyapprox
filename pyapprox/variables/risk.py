import numpy as np
from functools import partial
import scipy
from scipy import stats
from scipy.special import erfinv, gamma as gamma_fn, gammainc
from pyapprox.util.pya_numba import njit

from pyapprox.variables.algebra import invert_monotone_function


def entropic_risk_measure(samples, weights=None):
    """
    Compute the entropic risk measure
    from univariate samples of multiple quantities of interest

    Parameters
    ----------
    samples : np.ndarray (nsamples, nqoi)

    weights : np.ndarray (nsamples, 1)
        Weights associated with each sample

    Returns
    -------
    risk_measure_vals : np.ndarray (nqoi, 1)
    """
    if weights is None:
        weights = np.ones((samples.shape[0], 1))/samples.shape[0]
    assert weights.ndim == 2 and weights.shape[1] == 1
    assert weights.shape[0] == samples.shape[0]
    risk_measure_vals = np.log(
        np.sum(np.exp(samples)*weights, axis=0))[:, None]
    return risk_measure_vals


def deviation_measure(risk_measure_fun, samples):
    """
    Compute the deviation measure associated with a risk measure
    from univariate samples of multiple quantities of interest

    Parameters
    ----------
    risk_measure_fun : callable
        Callable function with signature

        `risk_measure_fun(samples) -> np.ndarray (nqoi, 1)`

        where samples : np.ndarray (nsamples, nqoi)

    samples : np.ndarray (nsamples, nqoi)
        Realizations of a univariate random variable

    Returns
    -------
    deviation_measure_vals : np.ndarray (nqoi, 1)
    """
    return risk_measure_fun(samples) - np.mean(samples)


def weighted_quantiles(samples, weights, qq, samples_sorted=False, prob=True):
    """
    Compute a set of quantiles from a weighted set of samples
    Parameters
    ----------
    samples : np.ndarray (nsamples, 1)
        Realizations of a univariate random variable

    weights : np.ndarray (nsamples, 1)
        Weights associated with each sample

    qq : np.ndarray (nquantiles, 1)
        The quantiles in [0,1]

    samples_sorted : boolean
        True - samples are already sorted which reduces computational cost
        False - samples are sorted internally

    prob : boolean
        True - weights must sum to 1
        False - no constraint on the weights

    Returns
    -------
    quantile_vals : np.ndarray (nquantiles, 1)
        The values of the random variable at each quantile
    """
    assert samples.shape == weights.shape
    assert samples.ndim == 1 and weights.ndim == 1
    if prob:
        assert np.allclose(weights.sum(), 1), weights.sum()
    if not samples_sorted:
        II = np.argsort(samples)
        xx, ww = samples[II], weights[II]
    else:
        xx, ww = samples, weights
    ecdf = ww.cumsum()-0.5*weights
    return np.interp(qq, ecdf, xx)


def value_at_risk(samples, alpha, weights=None, samples_sorted=False,
                  prob=True):
    """
    Compute the value at risk of a variable Y using a set of samples.

    Parameters
    ----------
    samples : np.ndarray (num_samples)
        Samples of the random variable Y

    alpha : integer
        The superquantile parameter

    weights : np.ndarray (num_samples)
        Importance weights associated with each sample. If samples Y are drawn
        from biasing distribution g(y) but we wish to compute VaR with respect
        to measure f(y) then weights are the ratio f(y_i)/g(y_i) for
        i=1,...,num_samples.

    Returns
    -------
    var : float
        The value at risk of the random variable Y
    """
    assert alpha >= 0 and alpha < 1
    assert samples.ndim == 1
    num_samples = samples.shape[0]
    if weights is None:
        weights = np.ones(num_samples)/num_samples
    if prob:
        factor = 1
        assert np.allclose(weights.sum(), 1)
    else:
        factor = weights.sum()

    assert weights.ndim == 1 or weights.shape[1] == 1
    assert samples.ndim == 1 or samples.shape[1] == 1
    if not samples_sorted:
        # TODO only need to find largest k entries. k is determined by
        # ecdf>=alpha
        II = np.argsort(samples)
        xx, ww = samples[II], weights[II]
    else:
        xx, ww = samples, weights
    ecdf = ww.cumsum()/factor
    index = np.arange(num_samples)[ecdf >= alpha][0]
    VaR = xx[index]
    if not samples_sorted:
        index = II[index]
        # assert samples[index]==VaR
    return VaR, index


@njit(cache=True)
def conditional_value_at_risk_vectorized(samples, alpha, weights=None,
                                         samples_sorted=False):
    nvars, nsamples = samples.shape
    if weights is None:
        weights = np.full((nvars, nsamples), 1/nsamples)
    assert weights.shape == samples.shape
    samples_sort = np.empty((nsamples))
    weights_sort = np.empty_like(samples_sort)
    if not samples_sorted:
        for ii in range(nvars):
            sorted_idx = np.argsort(samples[ii, :])
            for kk in range(nsamples):
                samples_sort[kk] = samples[ii, sorted_idx[kk]]
                weights_sort[kk] = weights[ii, sorted_idx[kk]]
            samples[ii, :] = samples_sort.copy()
            weights[ii, :] = weights_sort.copy()
    indices = np.empty(nvars, dtype=np.int64)
    quantiles = np.empty(nvars, dtype=samples.dtype)
    CVaR = np.empty_like(quantiles)
    for ii in range(nvars):
        # effective sample size
        # print((weights[ii].sum())**2/(weights[ii]**2).sum(), "w",
        #       nsamples, 'ii', ii)
        ecdf = weights[ii].cumsum()
        ecdf /= ecdf[-1]
        indices[ii] = np.argmax(ecdf >= alpha)
        quantiles[ii] = samples[ii, indices[ii]]
        CVaR[ii] = quantiles[ii]+1/((1-alpha))*np.sum(
            (samples[ii, indices[ii]+1:]-quantiles[ii]) *
            weights[ii, indices[ii]+1:])
    return CVaR


def value_at_risk_np_vectorized(samples, alpha, weights=None,
                                samples_sorted=False):
    nvars, nsamples = samples.shape
    if weights is None:
        weights = np.full((nvars, nsamples), 1/nsamples)
    assert weights.shape == samples.shape
    II = np.arange(nvars).reshape(nvars, 1)
    if not samples_sorted:
        sorted_idx = np.argsort(samples, axis=1)
        samples, weights = samples[II, sorted_idx], weights[II, sorted_idx]
    ecdf = weights.cumsum(axis=1)
    ecdf /= ecdf[:, -1:]
    indices = np.argmax(ecdf >= alpha, axis=1)
    quantiles = samples[II[:, 0], indices]
    if not samples_sorted:
        indices = sorted_idx[indices]
    return quantiles, indices


def conditional_value_at_risk_np_vectorized(samples, alpha, weights=None,
                                            samples_sorted=False,
                                            return_var=False):
    nvars, nsamples = samples.shape
    if weights is None:
        weights = np.full((nvars, nsamples), 1/nsamples)
    assert weights.shape == samples.shape
    if not samples_sorted:
        # argsort does not work with numba
        # sorted_idx = np.argsort(samples, axis=1)
        sorted_idx = np.empty(samples.shape, dtype=np.int64)
        for ii in range(nvars):
            sorted_idx[ii, :] = np.argsort(samples[ii, :])
        II = np.arange(nvars).reshape(nvars, 1)
        samples, weights = samples[II, sorted_idx], weights[II, sorted_idx]
    VaR, indices = value_at_risk_np_vectorized(
        samples, alpha, weights, samples_sorted=True)
    # quantile indices will be different for each row
    # only need to sum above quantiles. So ignore all entries of
    # samples and weights < imin. Then set to zero any entries > imin
    # but less than indices[row] for each row
    imin = indices.min()
    mask = (np.arange(imin, nsamples) < indices.reshape(indices.shape[0], 1))
    samples[:, imin:][mask], weights[:, imin:][mask] = 0.0, 0.0
    CVaR = VaR+1/((1-alpha))*np.sum(
        (samples[:, imin+1:]-VaR.reshape(nvars, 1))*weights[:, imin+1:],
        axis=1)
    if return_var:
        return CVaR, VaR
    return CVaR


def conditional_value_at_risk(samples, alpha, weights=None,
                              samples_sorted=False, return_var=False,
                              prob=True):
    """
    Compute conditional value at risk of a variable Y using a set of samples.

    Note accuracy of Monte Carlo Estimate of CVaR is dependent on alpha.
    As alpha increases more samples will be needed to achieve a fixed
    level of accruracy.

    Parameters
    ----------
    samples : np.ndarray (num_samples)
        Samples of the random variable Y

    alpha : integer
        The superquantile parameter

    weights : np.ndarray (num_samples)
        Importance weights associated with each sample. If samples Y are drawn
        from biasing distribution g(y) but we wish to compute VaR with respect
        to measure f(y) then weights are the ratio f(y_i)/g(y_i) for
        i=1,...,num_samples.

    Returns
    -------
    cvar : float
        The conditional value at risk of the random variable Y
    """
    assert samples.ndim == 1 or samples.shape[1] == 1
    samples = samples.squeeze()
    num_samples = samples.shape[0]
    if weights is None:
        weights = np.ones(num_samples)/num_samples
    if prob:
        assert np.allclose(weights.sum(), 1), (weights.sum())
    assert weights.ndim == 1
    if not samples_sorted:
        II = np.argsort(samples)
        xx, ww = samples[II], weights[II]
    else:
        xx, ww = samples, weights
    VaR, index = value_at_risk(xx, alpha, ww, samples_sorted=True, prob=prob)
    CVaR = VaR+1/((1-alpha))*np.sum((xx[index+1:]-VaR)*ww[index+1:])
    # The above one line can be used instead of the following
    # # number of support points above VaR
    # n_plus = num_samples-index-1
    # if n_plus==0:
    #     CVaR=VaR
    # else:
    #     # evaluate CDF at VaR
    #     cdf_at_var = (index+1)/num_samples
    #     lamda = (cdf_at_var-alpha)/(1-alpha)
    #     # Compute E[X|X>VaR(beta)]
    #     CVaR_plus = xx[index+1:].dot(ww[index+1:])/n_plus
    #     CVaR=lamda*VaR+(1-lamda)*CVaR_plus
    if not return_var:
        return CVaR
    else:
        return CVaR, VaR


def cvar_importance_sampling_biasing_density(pdf, function, beta, VaR, tau, x):
    """
    Evalute the biasing density used to compute CVaR of the variable
    Y=f(X), for some function f, vector X and scalar Y.

    The PDF of the biasing density is

    q(x) = [ beta/alpha p(x)         if f(x)>=VaR
           [ (1-beta)/(1-alpha) p(x) otherwise


    See https://link.springer.com/article/10.1007/s10287-014-0225-7

    Parameters
    ==========
    pdf: callable
        The probability density function p(x) of x

    function : callable
        Call signature f(x), where x is a 1D np.ndarray.


    VaR : float
        The value-at-risk associated above which to compute conditional value
        at risk

    tau : float
        The quantile of interest. 100*tau% percent of data will fall below this
        value

    beta: float
        Tunable parameter that controls shape of biasing density. As beta=0
        all samples will have values above VaR. If beta=tau, then biasing
        density will just be density of X p(x).

    x : np.ndarray (nsamples)
        The samples used to evaluate the biasing density.

    Returns
    =======
    vals: np.ndarray (nsamples)
        The values of the biasing density at x
    """
    if np.isscalar(x):
        x = np.array([[x]])
    assert x.ndim == 2
    vals = np.atleast_1d(pdf(x))
    assert vals.ndim == 1 or vals.shape[1] == 1
    y = function(x)
    assert y.ndim == 1 or y.shape[1] == 1
    II = np.where(y < VaR)[0]
    JJ = np.where(y >= VaR)[0]
    vals[II] *= beta/tau
    vals[JJ] *= (1-beta)/(1-tau)
    return vals


def generate_samples_from_cvar_importance_sampling_biasing_density(
        function, beta, VaR, generate_candidate_samples, nsamples):
    """
    Draw samples from the biasing density used to compute CVaR of the variable
    Y=f(X), for some function f, vector X and scalar Y.

    The PDF of the biasing density is

    q(x) = [ beta/alpha p(x)         if f(x)>=VaR
           [ (1-beta)/(1-alpha) p(x) otherwise

    See https://link.springer.com/article/10.1007/s10287-014-0225-7

    Parameters
    ==========
    function : callable
        Call signature f(x), where x is a 1D np.ndarray.

    beta: float
        Tunable parameter that controls shape of biasing density. As beta=0
        all samples will have values above VaR.  If beta=tau, then biasing
        density will just be density of X p(x). Best value of beta is problem
        dependent, but 0.2 has often been a reasonable choice.

    VaR : float
        The value-at-risk associated above which to compute conditional value
        at risk

    generate_candidate_samples : callable
        Function used to draw samples of X from pdf(x)
        Callable signature generate_canidate_samples(n) for some integer n

    nsamples : integer
        The numebr of samples desired

    Returns
    =======
    samples: np.ndarray (nvars,nsamples)
        Samples from the biasing density
    """
    candidate_samples = generate_candidate_samples(nsamples)
    nvars = candidate_samples.shape[0]
    samples = np.empty((nvars, nsamples))
    r = np.random.uniform(0, 1, nsamples)
    Ir = np.where(r < beta)[0]
    Jr = np.where(r >= beta)[0]
    Icnt = 0
    Jcnt = 0
    while True:
        vals = function(candidate_samples)
        assert vals.ndim == 1 or vals.shape[1] == 1
        II = np.where(vals < VaR)[0]
        JJ = np.where(vals >= VaR)[0]
        Iend = min(II.shape[0], Ir.shape[0]-Icnt)
        Jend = min(JJ.shape[0], Jr.shape[0]-Jcnt)
        samples[:, Ir[Icnt:Icnt+Iend]] = candidate_samples[:, II[:Iend]]
        samples[:, Jr[Jcnt:Jcnt+Jend]] = candidate_samples[:, JJ[:Jend]]
        Icnt += Iend
        Jcnt += Jend
        if Icnt == Ir.shape[0] and Jcnt == Jr.shape[0]:
            break
        candidate_samples = generate_candidate_samples(nsamples)
    assert Icnt+Jcnt == nsamples
    return samples


def compute_conditional_expectations(
        eta, samples, disutility_formulation=True):
    r"""
    Compute the conditional expectation of :math:`Y`
    .. math::
      \mathbb{E}\left[\max(0,\eta-Y)\right]

    or of :math:`-Y` (disutility form)
    .. math::
      \mathbb{E}\left[\max(0,Y-\eta)\right]

    where \math:`\eta\in Y' in the domain of :math:`Y'

    The conditional expectation is convex non-negative and non-decreasing.

    Parameters
    ==========
    eta : np.ndarray (num_eta)
        values of :math:`\eta`

    samples : np.ndarray (nsamples)
        The samples of :math:`Y`

    disutility_formulation : boolean
        True  - evaluate \mathbb{E}\left[\max(0,\eta-Y)\right]
        False - evaluate \mathbb{E}\left[\max(0,Y-\eta)\right]

    Returns
    =======
    values : np.ndarray (num_eta)
        The conditional expectations
    """
    assert samples.ndim == 1
    assert eta.ndim == 1
    if disutility_formulation:
        values = np.maximum(
            0, samples[:, np.newaxis]+eta[np.newaxis, :]).mean(axis=0)
    else:
        values = np.maximum(
            0, eta[np.newaxis, :]-samples[:, np.newaxis]).mean(axis=0)
    return values


def univariate_cdf_continuous_variable(pdf, lb, ub, x, quad_opts={}):
    x = np.atleast_1d(x)
    assert x.ndim == 1
    assert x.min() >= lb and x.max() <= ub
    vals = np.empty_like(x, dtype=float)
    for jj in range(x.shape[0]):
        integral, err = scipy.integrate.quad(pdf, lb, x[jj], **quad_opts)
        vals[jj] = integral
        if vals[jj] > 1 and vals[jj]-1 < quad_opts.get("epsabs", 1.49e-8):
            vals[jj] = 1.
    return vals


def univariate_quantile_continuous_variable(pdf, bounds, beta, opt_tol=1e-8,
                                            quad_opts={}):
    if quad_opts.get("epsabs", 1.49e-8) > opt_tol:
        raise ValueError("epsabs must be smaller than opt_tol")
    func = partial(univariate_cdf_continuous_variable,
                   pdf, bounds[0], bounds[1], quad_opts=quad_opts)
    method = 'bisect'
    quantile = invert_monotone_function(
        func, bounds, np.array([beta]), method, opt_tol)
    return quantile


def _univariate_cvar_continuous_variable_integrand(pdf, x):
    return x*pdf(x)


def univariate_cvar_continuous_variable(pdf, bounds, beta, opt_tol=1e-8,
                                        quad_opts={}, return_quantile=False):
    quantile = univariate_quantile_continuous_variable(
        pdf, bounds, beta, opt_tol, quad_opts)

    integrand = partial(_univariate_cvar_continuous_variable_integrand, pdf)
    integral, err = scipy.integrate.quad(
        integrand, quantile, bounds[1], **quad_opts)
    cvar = integral*1/(1-beta)
    if not return_quantile:
        return cvar
    return cvar, quantile


def lognormal_mean(mu, sigma_sq):
    """
    Compute the mean of a univariate lognormal variable
    """
    return np.exp(mu+sigma_sq/2)


def lognormal_cvar(p, mu, sigma_sq):
    """
    Compute the conditional value at risk of a univariate lognormal
    variable
    """
    mean = lognormal_mean(mu, sigma_sq)
    if p == 0:
        return mean
    # import warnings
    # warnings.filterwarnings("error")
    if sigma_sq < 0 and sigma_sq > -1e-16:
        sigma_sq = 0
    sigma = np.sqrt(sigma_sq)
    quantile = np.exp(mu+sigma*np.sqrt(2)*erfinv(2*p-1))
    if sigma == 0:
        print("Warning: sigma is zero", quantile)
        return quantile
    cvar = mean*stats.norm.cdf(
        (mu+sigma_sq-np.log(quantile))/sigma)/(1-p)
    return cvar


def lognormal_cvar_deviation(p, mu, sigma_sq):
    """
    Compute the conditional value at risk deviation of a univariate lognormal
    variable
    """
    mean = lognormal_mean(mu, sigma_sq)
    cvar = lognormal_cvar(p, mu, sigma_sq)
    return cvar-mean


def gaussian_entropic_risk(mu, sigma):
    return sigma**2/2+mu


def gaussian_entropic_deviation(sigma):
    return sigma**2/2


def lognormal_variance(mu, sigma_sq):
    """
    Compute the variance of a univariate lognormal variable
    """
    return (np.exp(sigma_sq)-1)*np.exp(2*mu+sigma_sq)


def beta_entropic_risk(alpha, beta):
    # x in [0, 1]
    from mpmath import hyp1f1
    return np.log(float(hyp1f1(alpha, alpha+beta, 1)))


def chi_squared_cvar(k, quantile):
    """
    Compute the conditional value at risk of a univariate Chi-squared variable
    """
    def upper_gammainc(a, b):
        return gamma_fn(a)*(1 - gammainc(a, b))
    VaR = stats.chi2.ppf(quantile, k)
    cvar = 2*upper_gammainc(1+k/2, VaR/2)/gamma_fn(k/2)/(1-quantile)
    return cvar


def gaussian_cvar(mu, sigma, quantile):
    """
    Compute the conditional value at risk of a univariate Gaussian variable
    See https://doi.org/10.1007/s10479-019-03373-1
    Calculating CVaR and bPOE for common probability
    distributions with application to portfolio optimization
    and density estimation.
    """
    val = mu+sigma*stats.norm.pdf(stats.norm.ppf(quantile))/(1-quantile)
    return val


def lognormal_kl_divergence(mu1, sigma1, mu2, sigma2):
    """Compute the KL divergence between two univariate lognormal variables

    That is compute :math:`KL(LN(mu_1, sigma_1)||LN(mu_2, sigma_2))` where
    :math:`mu_i,` :math:`sigma_i` are the mean and standard devition of their
    Gaussian distribution associated with each lognormal variable.

    Notes
    -----
    The Kullback-Leibler between two loggormals is the same as the pdf between
    the corresponding Gaussians
    """
    kl_div = (1/(2*sigma2**2)*((mu1-mu2)**2+sigma1**2-sigma2**2) +
              np.log(sigma2/sigma1))
    return kl_div


def gaussian_kl_divergence(mean1, cov1, mean2, cov2):
    r"""
    Compute KL( N(mean1, cov1) || N(mean2, cov2) )

    :math:`\int p_1(x)\log\left(\frac{p_1(x)}{p_2(x)}\right)dx`

    :math:`p_2(x)` must dominate :math:`p_1(x)`, e.g. for Bayesian inference
    the :math:`p_2(x)` is the posterior and :math:`p_1(x)` is the prior
    """
    if mean1.ndim != 2 or mean2.ndim != 2:
        raise ValueError("means must have shape (nvars, 1)")
    nvars = mean1.shape[0]
    cov2_inv = np.linalg.inv(cov2)
    val = np.log(np.linalg.det(cov2)/np.linalg.det(cov1))-float(nvars)
    val += np.trace(cov2_inv.dot(cov1))
    val += (mean2-mean1).T.dot(cov2_inv.dot(mean2-mean1))
    return 0.5*val.item()

# Useful thesis with derivations of KL and Renyi divergences for a number
# of canonical distributions
# Manuel Gil. 2011. ON RÉNYI DIVERGENCE MEASURES FOR CONTINUOUS ALPHABET
# SOURCES. https://mast.queensu.ca/~communications/Papers/gil-msc11.pdf


def compute_f_divergence(density1, density2, quad_rule, div_type,
                         normalize=False):
    r"""
    Compute f divergence between two densities

    .. math:: \int_\Gamma f\left(\frac{p(z)}{q(z)}\right)q(x)\,dx

    Parameters
    ----------
    density1 : callable
        The density p(z)

    density2 : callable
        The density q(z)

    normalize : boolean
        True  - normalize the densities
        False - Check that densities are normalized, i.e. integrate to 1

    quad_rule : tuple
        x,w - quadrature points and weights
        x : np.ndarray (num_vars,num_samples)
        w : np.ndarray (num_samples)

    div_type : string
        The type of f divergence (KL,TV,hellinger).
        KL - Kullback-Leibler :math:`f(t)=t\log t`
        TV - total variation  :math:`f(t)=\frac{1}{2}\lvert t-1\rvert`
        hellinger - squared Hellinger :math:`f(t)=(\sqrt(t)-1)^2`
    """
    x, w = quad_rule
    assert w.ndim == 1

    density1_vals = density1(x).squeeze()
    const1 = density1_vals.dot(w)
    density2_vals = density2(x).squeeze()
    const2 = density2_vals.dot(w)
    if normalize:
        density1_vals /= const1
        density2_vals /= const2
    else:
        tol = 1e-14
        # print(const1)
        # print(const2)
        assert np.allclose(const1, 1.0, atol=tol)
        assert np.allclose(const2, 1.0, atol=tol)
        const1, const2 = 1.0, 1.0

    # normalize densities. May be needed if density is
    # Unnormalized Bayesian Posterior
    def d1(x): return density1(x)/const1
    def d2(x): return density2(x)/const2

    if div_type == 'KL':
        # Kullback-Leibler
        def f(t): return t*np.log(t)
    elif div_type == 'TV':
        # Total variation
        def f(t): return 0.5*np.absolute(t-1)
    elif div_type == 'hellinger':
        # Squared hellinger int (p(z)**0.5-q(z)**0.5)**2 dz
        # Note some formulations use 0.5 times above integral. We do not
        # do that here
        def f(t): return (np.sqrt(t)-1)**2
    else:
        raise Exception(f'Divergence type {div_type} not supported')

    d1_vals, d2_vals = d1(x), d2(x)
    II = np.where(d2_vals > 1e-15)[0]
    ratios = np.zeros_like(d2_vals)+1e-15
    ratios[II] = d1_vals[II]/d2_vals[II]
    if not np.all(np.isfinite(ratios)):
        print(d1_vals[II], d2_vals[II])
        msg = 'Densities are not absolutely continuous. '
        msg += 'Ensure that density2(z)=0 implies density1(z)=0'
        raise Exception(msg)

    divergence_integrand = f(ratios)*d2_vals

    return divergence_integrand.dot(w)
