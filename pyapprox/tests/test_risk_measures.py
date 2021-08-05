import unittest
from scipy import stats
from scipy.special import erf, erfinv, factorial, betainc, beta as beta_fn, \
    gamma as gamma_fn
from pyapprox.risk_measures import *
from pyapprox.variables import get_distribution_info
from scipy import integrate
from functools import partial
from scipy.optimize import minimize


def cvar_univariate_integrand(f, pdf, t, x):
    x = np.atleast_2d(x)
    assert x.shape[0] == 1
    pdf_vals = pdf(x[0, :])
    I = np.where(pdf_vals > 0)[0]
    vals = np.zeros(x.shape[1])
    if I.shape[0] > 0:
        vals[I] = np.maximum(f(x[:, I])[:, 0]-t, 0)*pdf_vals[I]
    return vals


def compute_cvar_objective_from_univariate_function_quadpack(
        f, pdf, lbx, ubx, alpha, t, tol=4*np.finfo(float).eps):
    import warnings
    # warnings.simplefilter("ignore")
    integral, err = integrate.quad(
        partial(cvar_univariate_integrand, f, pdf, t), lbx, ubx,
        epsrel=tol, epsabs=tol, limit=100)
    # warnings.simplefilter("default")
    #assert err<1e-13, err
    val = t+1./(1.-alpha)*integral
    return val


def compute_cvar_from_univariate_function(f, pdf, lbx, ubx, alpha, init_guess,
                                          tol=1e-7):
    # tolerance used to compute integral should be more accurate than
    # optimization tolerance
    obj = partial(compute_cvar_objective_from_univariate_function_quadpack,
                  f, pdf, lbx, ubx, alpha)
    method = 'L-BFGS-B'
    options = {'disp': False, 'gtol': tol, 'ftol': tol}
    result = minimize(obj, init_guess, method=method,
                      options=options)
    value_at_risk = result['x']
    cvar = result['fun']
    return value_at_risk, cvar


def cvar_beta_variable(rv, beta):
    """
    cvar of Beta variable on [0,1]
    """
    qq = rv.ppf(beta)
    scales, shapes = get_distribution_info(rv)[1:]
    qq = (qq-scales["loc"])/scales["scale"]
    g1, g2 = shapes["a"], shapes["b"]
    cvar01 = ((1-betainc(1+g1, g2, qq))*gamma_fn(1+g1)*gamma_fn(g2) /
              gamma_fn(1+g1+g2))/(beta_fn(g1, g2)*(1-beta))
    return cvar01*scales["scale"] + scales["loc"]


def triangle_quantile(u, c, loc, scale):
    """
    Also known as inverse CDF
    """
    if np.isscalar(u):
        u = np.asarray([u])
    assert u.ndim == 1
    lb = loc
    mid = loc+c*scale
    ub = loc+scale
    I = np.where((u >= 0) & (u < (mid-lb)/(ub-lb)))[0]
    J = np.where((u <= 1) & (u >= (mid-lb)/(ub-lb)))[0]
    if I.shape[0]+J.shape[0] != u.shape[0]:
        raise Exception('Ensure u in [0,1] and loc<=loc+c*scale <=loc+scale')

    quantiles = np.empty_like(u)
    quantiles[I] = lb + np.sqrt((ub-lb)*(mid-lb)*u[I])
    quantiles[J] = ub - np.sqrt((ub-lb)*(ub-mid)*(1-u[J]))
    return quantiles


def triangle_superquantile(u, c, loc, scale):
    lb = loc
    mid = loc+c*scale
    ub = loc+scale

    if np.isscalar(u):
        u = np.asarray([u])
    assert u.ndim == 1

    def left_integral(u): return 2./3*u*np.sqrt(u*(lb - ub)*(lb - mid)) + lb*u
    def right_integral(u): return ub*u-2./3*(u-1) * \
        np.sqrt((u-1)*(lb-ub)*(ub-mid))

    I = np.where((u >= 0) & (u < (mid-lb)/(ub-lb)))[0]
    J = np.where((u < 1) & (u >= (mid-lb)/(ub-lb)))[0]
    K = np.where((u == 1))[0]

    if I.shape[0]+J.shape[0]+K.shape[0] != u.shape[0]:
        raise Exception('Ensure u in [0,1] and loc<=loc+c*scale <=loc+scale')

    superquantiles = np.empty_like(u)
    superquantiles[I] = (
        left_integral((mid-lb)/(ub-lb))-left_integral(u[I]) +
        right_integral(1)-right_integral((mid-lb)/(ub-lb)))/(1-u[I])
    superquantiles[J] = (right_integral(1)-right_integral(u[J]))/(1-u[J])
    superquantiles[K] = stats.triang.interval(1, c, loc, scale)[1]
    return superquantiles


def get_lognormal_example_exact_quantities(mu, sigma):
    # print('mu,sigma',mu,sigma)
    def f(x): return np.exp(x).T

    mean = np.exp(mu+sigma**2/2)

    def f_cdf(y):
        y = np.atleast_1d(y)
        vals = np.zeros_like(y)
        II = np.where(y > 0)[0]
        vals[II] = stats.norm.cdf((np.log(y[II])-mu)/sigma)
        return vals

    # PDF of output variable (lognormal PDF)
    def f_pdf(y):
        vals = np.zeros_like(y, dtype=float)
        II = np.where(y > 0)[0]
        print(II)
        vals[II] = np.exp(-(np.log(y[II])-mu)**2/(2*sigma**2))/(
            sigma*np.sqrt(2*np.pi)*y[II])
        return vals

    # Analytic VaR of model output
    def VaR(p): return np.exp(mu+sigma*np.sqrt(2)*erfinv(2*p-1))

    # Analytic VaR of model output
    def CVaR(p): return mean*stats.norm.cdf(
        (mu+sigma**2-np.log(VaR(p)))/sigma)/(1-p)

    def cond_exp_le_eta(y):
        vals = np.zeros_like(y, dtype=float)
        II = np.where(y > 0)[0]
        vals[II] = mean*stats.norm.cdf((np.log(y[II])-mu-sigma**2)/sigma)/f_cdf(
            y[II])
        return vals

    def ssd(y): return f_cdf(y)*(y-cond_exp_le_eta(y))

    def cond_exp_y_ge_eta(y):
        vals = np.ones_like(y, dtype=float)*mean
        II = np.where(y > 0)[0]
        vals[II] = mean*stats.norm.cdf(
            (mu+sigma**2-np.log(y[II]))/sigma)/(1-f_cdf(y[II]))
        return vals

    def ssd_disutil(eta): return (1-f_cdf(-eta))*(eta+cond_exp_y_ge_eta(-eta))
    return f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil


def get_truncated_lognormal_example_exact_quantities(lb, ub, mu, sigma):
    def f(x): return np.exp(x).T

    # lb,ub passed to truncnorm_rv are defined for standard normal.
    # Adjust for mu and sigma using
    alpha, beta = (lb-mu)/sigma, (ub-mu)/sigma

    denom = stats.norm.cdf(beta)-stats.norm.cdf(alpha)
    # truncated_normal_cdf = lambda x: (
    #    stats.norm.cdf((x-mu)/sigma)-stats.norm.cdf(alpha))/denom

    def truncated_normal_cdf(x): return truncnorm_rv.cdf(
        x, alpha, beta, loc=mu, scale=sigma)

    def truncated_normal_pdf(x): return truncnorm_rv.pdf(
        x, alpha, beta, loc=mu, scale=sigma)

    def truncated_normal_ppf(p): return truncnorm_rv.ppf(
        p, alpha, beta, loc=mu, scale=sigma)

    # CDF of output variable (log truncated normal PDF)
    def f_cdf(y):
        vals = np.zeros_like(y)
        II = np.where((y > np.exp(lb)) & (y < np.exp(ub)))[0]
        vals[II] = truncated_normal_cdf(np.log(y[II]))
        JJ = np.where((y >= np.exp(ub)))[0]
        vals[JJ] = 1.
        return vals

    # PDF of output variable (log truncated normal PDF)
    def f_pdf(y):
        vals = np.zeros_like(y)
        II = np.where((y > np.exp(lb)) & (y < np.exp(ub)))[0]
        vals[II] = truncated_normal_pdf(np.log(y[II]))/y[II]
        return vals

    # Analytic VaR of model output
    def VaR(p): return np.exp(truncated_normal_ppf(p))

    const = np.exp(mu+sigma**2/2)

    # Analytic VaR of model output
    def CVaR(p): return -0.5/denom*const/(1-p)*(
        erf((mu+sigma**2-ub)/(np.sqrt(2)*sigma))
        - erf((mu+sigma**2-np.log(VaR(p)))/(np.sqrt(2)*sigma)))

    def cond_exp_le_eta(y):
        vals = np.zeros_like(y)
        II = np.where((y > np.exp(lb)) & (y < np.exp(ub)))[0]
        vals[II] = -0.5/denom*const*(
            erf((mu+sigma**2-np.log(y[II]))/(np.sqrt(2)*sigma))
            - erf((mu+sigma**2-lb)/(np.sqrt(2)*sigma)))/f_cdf(y[II])
        JJ = np.where((y >= np.exp(ub)))[0]
        vals[JJ] = mean
        return vals

    def ssd(y): return f_cdf(y)*(y-cond_exp_le_eta(y))

    mean = CVaR(np.zeros(1))

    def cond_exp_y_ge_eta(y):
        vals = np.ones_like(y)*mean
        II = np.where((y > np.exp(lb)) & (y < np.exp(ub)))[0]
        vals[II] = -0.5/denom*const*(
            erf((mu+sigma**2-ub)/(np.sqrt(2)*sigma))
            - erf((mu+sigma**2-np.log(y[II]))/(np.sqrt(2)*sigma)))/(
                1-f_cdf(y[II]))
        JJ = np.where((y > np.exp(ub)))[0]
        vals[JJ] = 0
        return vals

    def ssd_disutil(eta): return (1-f_cdf(-eta))*(eta+cond_exp_y_ge_eta(-eta))

    return f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil


def plot_truncated_lognormal_example_exact_quantities(
        num_samples=int(1e5), plot=False, mu=0, sigma=1):
    if plot:
        assert num_samples <= 1e5
    num_vars = 1.
    lb, ub = -1, 3

    f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
        get_truncated_lognormal_example_exact_quantities(lb, ub, mu, sigma)
    # lb,ub passed to truncnorm_rv are defined for standard normal.
    # Adjust for mu and sigma using
    alpha, beta = (lb-mu)/sigma, (ub-mu)/sigma
    samples = truncnorm_rv.rvs(
        alpha, beta, mu, sigma, size=num_samples)[np.newaxis, :]
    values = f(samples)[:, 0]

    fig, axs = plt.subplots(1, 6, sharey=False, figsize=(16, 6))

    from pyapprox.density import EmpiricalCDF
    ygrid = np.linspace(np.exp(lb)-1, np.exp(ub)*1.1, 100)
    ecdf = EmpiricalCDF(values)
    if plot:
        axs[0].plot(ygrid, ecdf(ygrid), '-')
        axs[0].plot(ygrid, f_cdf(ygrid), '--')
        axs[0].set_xlim(ygrid.min(), ygrid.max())
        axs[0].set_title('CDF')

    if plot:
        ygrid = np.linspace(np.exp(lb)-1, np.exp(ub)*1.1, 100)
        axs[1].hist(values, bins='auto', density=True)
        axs[1].plot(ygrid, f_pdf(ygrid), '--')
        axs[1].set_xlim(ygrid.min(), ygrid.max())
        axs[1].set_title('PDF')

    pgrid = np.linspace(0.01, 1-1e-2, 10)
    evar = np.array([value_at_risk(values, p)[0] for p in pgrid]).squeeze()
    if plot:
        axs[2].plot(pgrid, evar, '-')
        axs[2].plot(pgrid, VaR(pgrid), '--')
        axs[2].set_title('VaR')
    else:
        assert np.allclose(evar, VaR(pgrid), rtol=2e-2)

    pgrid = np.linspace(0, 1-1e-2, 100)
    ecvar = np.array([conditional_value_at_risk(values, p) for p in pgrid])
    # CVaR for alpha=0 should be the mean
    assert np.allclose(ecvar[0], values.mean())
    if plot:
        axs[3].plot(pgrid, ecvar, '-')
        axs[3].plot(pgrid, CVaR(pgrid), '--')
        axs[3].set_xlim(pgrid.min(), pgrid.max())
        axs[3].set_title('CVaR')
    else:
        assert np.allclose(ecvar.squeeze(), CVaR(pgrid).squeeze(), rtol=1e-2)

    ygrid = np.linspace(np.exp(lb)-10, np.exp(ub)+1, 100)
    essd = compute_conditional_expectations(ygrid, values, False)
    if plot:
        axs[4].plot(ygrid, essd, '-')
        axs[4].plot(ygrid, ssd(ygrid), '--')
        axs[4].set_xlim(ygrid.min(), ygrid.max())
        axs[4].set_title(r'$E[(\eta-Y)^+]$')
        axs[5].set_xlabel(r'$\eta$')
    else:
        assert np.allclose(essd.squeeze(), ssd(ygrid), rtol=2e-2)

    # zoom into ygrid over high probability region of -Y
    ygrid = -ygrid[::-1]
    disutil_essd = compute_conditional_expectations(ygrid, values, True)
    assert np.allclose(disutil_essd, compute_conditional_expectations(
        ygrid, -values, False))
    # print(np.linalg.norm(disutil_essd.squeeze()-ssd_disutil(ygrid),ord=np.inf))
    if plot:
        axs[5].plot(ygrid, disutil_essd, '-', label='Empirical')
        axs[5].plot(ygrid, ssd_disutil(ygrid), '--', label='Exact')
        axs[5].set_xlim(ygrid.min(), ygrid.max())
        axs[5].set_title(r'$E[(\eta-(-Y))^+]$')
        axs[5].set_xlabel(r'$\eta$')
        axs[5].legend()

        plt.show()


def plot_lognormal_example_exact_quantities(num_samples=int(2e5), plot=False,
                                            mu=0, sigma=1):
    num_vars = 1
    if plot:
        assert num_samples <= 1e5
    else:
        assert num_samples >= 1e4

    f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
        get_lognormal_example_exact_quantities(mu, sigma)
    from pyapprox.low_discrepancy_sequences import transformed_halton_sequence
    #samples = np.random.normal(mu,sigma,(num_vars,num_samples))
    #values = f(samples)[:,0]
    samples = transformed_halton_sequence(
        [partial(stats.norm.ppf, loc=mu, scale=sigma)], num_vars, num_samples)
    values = f(samples)[:, 0]

    fig, axs = plt.subplots(1, 6, sharey=False, figsize=(16, 6))

    from pyapprox.density import EmpiricalCDF
    if plot:
        ygrid = np.linspace(-1, 5, 100)
        #ecdf = EmpiricalCDF(values)
        # axs[0].plot(ygrid,ecdf(ygrid),'-')
        axs[0].plot(ygrid, f_cdf(ygrid), '--')
        # axs[0].set_xlim(ygrid.min(),ygrid.max())
        axs[0].set_title('CDF')
        #ecdf = EmpiricalCDF(-values)
        # axs[0].plot(-ygrid,ecdf(-ygrid),'-')

        ygrid = np.linspace(-1, 20, 100)
        # axs[1].hist(values,bins='auto',density=True)
        axs[1].plot(ygrid, f_pdf(ygrid), '--')
        axs[1].set_xlim(ygrid.min(), ygrid.max())
        axs[1].set_title('PDF')

    pgrid = np.linspace(1e-2, 1-1e-2, 100)
    evar = np.array([value_at_risk(values, p)[0] for p in pgrid])
    # print(np.linalg.norm(evar.squeeze()-VaR(pgrid),ord=np.inf))
    if plot:
        axs[2].plot(pgrid, evar, '-')
        axs[2].plot(pgrid, VaR(pgrid), '--')
        axs[2].set_title('VaR')
    else:
        assert np.allclose(evar.squeeze(), VaR(pgrid), atol=2e-1)

    pgrid = np.linspace(1e-2, 1-1e-2, 100)
    ecvar = np.array([conditional_value_at_risk(values, y) for y in pgrid])
    # print(np.linalg.norm(ecvar.squeeze()-CVaR(pgrid).squeeze(),ord=np.inf))
    print(CVaR(0.8))
    if plot:
        axs[3].plot(pgrid, ecvar, '-')
        axs[3].plot(pgrid, CVaR(pgrid), '--')
        axs[3].set_xlim(pgrid.min(), pgrid.max())
        axs[3].set_title('CVaR')
    else:
        assert np.allclose(ecvar.squeeze(), CVaR(pgrid).squeeze(), rtol=4e-2)

    #ygrid = np.linspace(-1,10,100)
    ygrid = np.linspace(
        logstats.norm.ppf(0.0, np.exp(mu), sigma),
        logstats.norm.ppf(0.9, np.exp(mu), sigma), 101)
    essd = compute_conditional_expectations(ygrid, values, False)
    # print(np.linalg.norm(essd.squeeze()-ssd(ygrid),ord=np.inf))
    if plot:
        axs[4].plot(ygrid, essd, '-')
        axs[4].plot(ygrid, ssd(ygrid), '--')
        axs[4].set_xlim(ygrid.min(), ygrid.max())
        axs[4].set_title(r'$E[(\eta-Y)^+]$')
        axs[4].set_xlabel(r'$\eta$')
    else:
        assert np.allclose(essd.squeeze(), ssd(ygrid), atol=1e-3)

    # zoom into ygrid over high probability region of -Y
    ygrid = -ygrid[::-1]
    disutil_essd = compute_conditional_expectations(ygrid, values, True)
    assert np.allclose(disutil_essd, compute_conditional_expectations(
        ygrid, -values, False))
    # print(np.linalg.norm(disutil_essd.squeeze()-ssd_disutil(ygrid),ord=np.inf))
    if plot:
        axs[5].plot(ygrid, disutil_essd, '-', label='Empirical')
        axs[5].plot(ygrid, ssd_disutil(ygrid), '--', label='Exact')
        axs[5].set_xlim((ygrid).min(), (ygrid).max())
        axs[5].set_title(r'$E[(\eta-(-Y))^+]$')
        axs[5].set_xlabel(r'$\eta$')
        axs[5].plot([0], [np.exp(mu+sigma**2/2)], 'o')
        axs[5].legend()
        plt.show()
    else:
        assert np.allclose(disutil_essd.squeeze(),
                           ssd_disutil(ygrid), atol=1e-3)


class TestRiskMeasures(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_triangle_quantile(self):
        rv_1 = stats.triang(0.5, loc=-0.5, scale=2)

        alpha = np.array([0.3, 0.75])
        assert np.allclose(
            rv_1.ppf(alpha), triangle_quantile(alpha, 0.5, -0.5, 2))

    def test_triangle_superquantile(self):
        c = 0.5
        loc = -0.5
        scale = 2
        u = np.asarray([0.3, 0.75])
        cvar = triangle_superquantile(u, c, loc, scale)
        samples = triangle_quantile(
            np.random.uniform(0, 1, (100000)), c, loc, scale)
        for ii in range(len(u)):
            mc_cvar = conditional_value_at_risk(samples, u[ii])
        assert abs(cvar[ii]-mc_cvar) < 1e-2

    def test_value_at_risk_normal(self):
        weights = None
        alpha = 0.8
        num_samples = int(1e2)
        samples = np.random.normal(0, 1, num_samples)
        # [np.random.permutation(num_samples)]
        samples = np.arange(1, num_samples+1)
        #xx = np.sort(samples)
        #VaR,VaR_index = value_at_risk(xx,alpha,weights,samples_sorted=True)
        xx = samples
        VaR, VaR_index = value_at_risk(
            xx, alpha, weights, samples_sorted=False)
        sorted_index = int(np.ceil(alpha*num_samples)-1)
        I = np.argsort(samples)
        index = I[sorted_index]
        print(index, VaR_index)
        assert np.allclose(VaR_index, index)
        assert np.allclose(VaR, xx[index])

    def test_value_at_risk_lognormal(self):
        mu, sigma = 0, 1

        def f(x): return np.exp(x).T
        def VaR(p): return np.exp(mu+sigma*np.sqrt(2)*erfinv(2*p-1))
        mean = np.exp(mu+sigma**2/2)
        def CVaR(p): return mean*stats.norm.cdf(
            (mu+sigma**2-np.log(VaR(p)))/sigma)/(1-p)

        weights = None
        alpha = 0.8
        num_samples = int(1e6)
        samples = f(np.random.normal(0, 1, num_samples))
        xx = np.sort(samples)
        empirical_VaR, __ = value_at_risk(
            xx, alpha, weights, samples_sorted=True)
        # print(VaR(alpha),empirical_VaR)
        assert np.allclose(VaR(alpha), empirical_VaR, 1e-2)

    def test_weighted_value_at_risk_normal(self):
        mu, sigma = 1, 1
        # bias_mu,bias_sigma=1.0,1
        bias_mu, bias_sigma = mu, sigma

        from scipy.special import erf
        def VaR(alpha): return stats.norm.ppf(alpha, loc=mu, scale=sigma)

        def CVaR(alpha):
            vals = 0.5*mu
            vals -= 0.5*mu*erf((VaR(alpha)-mu)/(np.sqrt(2)*sigma)) -\
                sigma*np.exp(-(mu-VaR(alpha))**2/(2*sigma**2))/np.sqrt(2*np.pi)
            vals /= (1-alpha)
            return vals

        alpha = 0.8
        num_samples = int(1e5)
        samples = np.random.normal(bias_mu, bias_sigma, num_samples)
        target_pdf_vals = stats.norm.pdf(samples, loc=mu, scale=sigma)
        bias_pdf_vals = stats.norm.pdf(samples, loc=bias_mu, scale=bias_sigma)
        I = np.where(bias_pdf_vals < np.finfo(float).eps)[0]
        assert np.all(target_pdf_vals[I] < np.finfo(float).eps)
        J = np.where(bias_pdf_vals >= np.finfo(float).eps)[0]
        weights = np.zeros_like(target_pdf_vals)
        weights[J] = target_pdf_vals[J]/bias_pdf_vals[J]
        weights /= weights.sum()

        empirical_VaR, __ = value_at_risk(
            samples, alpha, weights, samples_sorted=False)
        # print('VaR',VaR(alpha),empirical_VaR)
        assert np.allclose(VaR(alpha), empirical_VaR, rtol=1e-2)

        empirical_CVaR = conditional_value_at_risk(
            samples, alpha, weights)
        # print('CVaR',CVaR(alpha),empirical_CVaR)
        assert np.allclose(CVaR(alpha), empirical_CVaR, rtol=1e-2)

    def test_equivalent_formulations_of_cvar(self):
        mu, sigma = 0, 1
        f, f_cdf, f_pdf, VaR, CVaR, ssd, ssd_disutil = \
            get_lognormal_example_exact_quantities(mu, sigma)

        eps = 1e-15
        lbx, ubx = stats.norm.ppf([eps, 1-eps], loc=mu, scale=sigma)
        lbx, ubx = -np.inf, np.inf
        tol = 4*np.finfo(float).eps
        alpha = .8
        t = VaR(alpha)
        #fun1 = lambda x: np.maximum(f(x)-t,0)*stats.norm.pdf(x,loc=mu,scale=sigma)

        def fun1(x):
            x = np.atleast_1d(x)
            pdf_vals = stats.norm.pdf(x, loc=mu, scale=sigma)
            # Next line avoids problems caused by f(x) being to large or nan
            I = np.where(pdf_vals > 0)[0]
            vals = np.zeros_like(x)
            vals[I] = np.maximum(f(x[I])-t, 0)*pdf_vals[I]
            return vals
        cvar1 = t+1./(1.-alpha)*integrate.quad(
            fun1, lbx, ubx, epsabs=tol, epsrel=tol, full_output=True)[0]

        #fun2 = lambda x: f(x)*stats.norm.pdf(x,loc=mu,scale=sigma)
        def fun2(x):
            x = np.atleast_1d(x)
            pdf_vals = stats.norm.pdf(x, loc=mu, scale=sigma)
            # Next line avoids problems caused by f(x) being to large or nan
            I = np.where(pdf_vals > 0)[0]
            vals = np.zeros_like(x)
            vals[I] = f(x[I])*pdf_vals[I]
            return vals
        x_cdf = partial(stats.norm.cdf, loc=mu, scale=sigma)
        integral = integrate.quad(
            fun2, np.log(t), ubx, epsabs=tol, epsrel=tol, full_output=True)[0]
        cvar2 = t+1./(1.-alpha)*(integral-t*(1-x_cdf(np.log(t))))
        #fun3=lambda x: np.absolute(f(x)-t)*stats.norm.pdf(x,loc=mu,scale=sigma)

        def fun3(x):
            x = np.atleast_1d(x)
            pdf_vals = stats.norm.pdf(x, loc=mu, scale=sigma)
            # Next line avoids problems caused by f(x) being to large or nan
            I = np.where(pdf_vals > 0)[0]
            vals = np.zeros_like(x)
            vals[I] = np.absolute(f(x[I])-t)*pdf_vals[I]
            return vals

        mean = CVaR(0)
        integral = integrate.quad(
            fun3, lbx, ubx, epsabs=tol, epsrel=tol, full_output=True)[0]
        cvar3 = t+0.5/(1.-alpha)*(integral+mean-t)

        lbx, ubx = - np.inf, np.inf
        value_at_risk4, cvar4 = compute_cvar_from_univariate_function(
            f, partial(stats.norm.pdf, loc=mu, scale=sigma), lbx, ubx, alpha, 5,
            tol=1e-7)

        # print(abs(cvar1-CVaR(alpha)))
        # print(abs(cvar2-CVaR(alpha)))
        # print(abs(cvar3-CVaR(alpha)))
        # print(abs(cvar4-CVaR(alpha)))
        assert np.allclose(cvar1, CVaR(alpha))
        assert np.allclose(cvar2, CVaR(alpha))
        assert np.allclose(cvar3, CVaR(alpha))
        assert np.allclose(cvar4, CVaR(alpha))

    def test_conditional_value_at_risk(self):
        N = 6
        p = np.ones(N)/N
        X = np.random.normal(0, 1, N)
        #X = np.arange(1,N+1)
        X = np.sort(X)
        beta = 7/12
        i_beta_exact = 3
        VaR_exact = X[i_beta_exact]
        cvar_exact = 1/5*VaR_exact+2/5*(np.sort(X)[i_beta_exact+1:]).sum()
        ecvar, evar = conditional_value_at_risk(
            X, beta, return_var=True)
        # print(cvar_exact,ecvar)
        assert np.allclose(cvar_exact, ecvar)

    def test_conditional_value_at_risk_using_opitmization_formula(self):
        """
        Compare value obtained via optimization and analytical formula
        """
        plot = False
        num_samples = 5
        alpha = np.array([1./3., 0.5, 0.85])
        samples = np.random.normal(0, 1, (num_samples))
        for ii in range(alpha.shape[0]):
            ecvar, evar = conditional_value_at_risk(
                samples, alpha[ii], return_var=True)

            def objective(tt): return np.asarray([t+1/(1-alpha[ii])*np.maximum(
                0, samples-t).mean() for t in tt])

            tol = 1e-8
            method = 'L-BFGS-B'
            options = {'disp': False, 'gtol': tol, 'ftol': tol}
            init_guess = 0
            result = minimize(objective, init_guess, method=method,
                              options=options)
            value_at_risk = result['x']
            cvar = result['fun']
            print(alpha[ii], value_at_risk, cvar, ecvar)
            assert np.allclose(ecvar, cvar)
            assert np.allclose(evar, value_at_risk)

            if plot:
                rv = stats.norm
                lb, ub = rv.interval(.99)
                xx = np.linspace(lb, ub, 101)
                obj_vals = objective(xx)
                plt.plot(xx, obj_vals)
                plt.plot(value_at_risk, cvar, 'ro')
                plt.show()

    def test_compute_conditional_expectations(self):
        num_samples = 5
        samples = np.random.normal(0, 1, (num_samples))
        eta = samples

        values = compute_conditional_expectations(eta, samples, True)
        values1 = [np.maximum(0, eta[ii]+samples).mean()
                   for ii in range(eta.shape[0])]
        assert np.allclose(values, values1)

        values = compute_conditional_expectations(eta, samples, False)
        values1 = [np.maximum(0, eta[ii]-samples).mean()
                   for ii in range(eta.shape[0])]
        assert np.allclose(values, values1)

    def test_univariate_cvar_continuous_variable(self):
        beta = 0.8
        var = stats.beta(3, 5)
        quantile = univariate_quantile_continuous_variable(
            var.pdf, var.support(), beta, 1e-6)
        # print((quantile, var.ppf(beta)))
        assert np.allclose(quantile, var.ppf(beta))

        cvar = univariate_cvar_continuous_variable(
            var.pdf, var.support(), beta, 1e-6)
        assert np.allclose(cvar, cvar_beta_variable(var, beta))

        var = stats.beta(3, 5, loc=1, scale=2)
        cvar = univariate_cvar_continuous_variable(
            var.pdf, var.support(), beta, 1e-8, {"epsabs": 1e-10})
        assert np.allclose(cvar, cvar_beta_variable(var, beta))


if __name__ == "__main__":
    risk_measures_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestRiskMeasures)
    unittest.TextTestRunner(verbosity=2).run(risk_measures_test_suite)
