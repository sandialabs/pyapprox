import numpy as np
from functools import partial
import scipy

from pyapprox.random_variable_algebra import invert_monotone_function


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


def value_at_risk(samples, alpha, weights=None, samples_sorted=False):
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
    assert np.allclose(weights.sum(), 1)
    assert weights.ndim == 1 or weights.shape[1] == 1
    assert samples.ndim == 1 or samples.shape[1] == 1
    if not samples_sorted:
        # TODO only need to find largest k entries. k is determined by
        # ecdf>=alpha
        II = np.argsort(samples)
        xx, ww = samples[II], weights[II]
    else:
        xx, ww = samples, weights
    ecdf = ww.cumsum()
    index = np.arange(num_samples)[ecdf >= alpha][0]
    VaR = xx[index]
    if not samples_sorted:
        index = II[index]
        # assert samples[index]==VaR
    return VaR, index


def conditional_value_at_risk(samples, alpha, weights=None,
                              samples_sorted=False, return_var=False):
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
    assert np.allclose(weights.sum(), 1), (weights.sum())
    assert weights.ndim == 1 or weights.shape[1] == 1
    if not samples_sorted:
        II = np.argsort(samples)
        xx, ww = samples[II], weights[II]
    else:
        xx, ww = samples, weights
    VaR, index = value_at_risk(xx, alpha, ww, samples_sorted=True)
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


def univariate_cvar_continuous_variable(pdf, bounds, beta, opt_tol=1e-8,
                                        quad_opts={}):
    quantile = univariate_quantile_continuous_variable(
        pdf, bounds, beta, opt_tol, quad_opts)

    def integrand(x): return x*pdf(x)
    return 1/(1-beta)*scipy.integrate.quad(
        integrand, quantile, bounds[1], **quad_opts)[0]
