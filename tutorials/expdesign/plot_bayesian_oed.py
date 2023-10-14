r"""
Bayesian Optimal Experimental Design
====================================
"""
#%%
#Load modules
import os
from scipy import stats
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from pyapprox.util.visualization import (
    get_meshgrid_function_data, create_3d_axis, plot_surface,
    _turn_off_3d_axes)
from pyapprox.util.configure_plots import mathrm_label
from pyapprox.variables.joint import (
    IndependentMarginalsVariable, get_truncated_range)
from pyapprox.variables.marginals import get_pdf
from pyapprox.variables.risk import (
    univariate_cvar_continuous_variable, univariate_cdf_continuous_variable,
    conditional_value_at_risk)
from pyapprox.bayes.metropolis import GaussianLogLike
from pyapprox.surrogates.integrate import integrate
from multiprocessing import Pool
from pyapprox.surrogates.interp.tensorprod import (
    TensorProductInterpolant, get_univariate_interpolation_basis)
savefigs = False
max_eval_concurrency = 1
np.random.seed(1)


def horizontal_subplots(ny):
    return plt.subplots(1, ny, figsize=(ny*8, 1*6), sharey=True)


def vertical_subplots(nx):
    return plt.subplots(nx, 1, figsize=(1*8, nx*6), sharex=True)


horizontal = True
subplots = horizontal_subplots

#%%
#Configure latex for plotting
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True  # use latex for all text handling
mpl.rcParams['text.latex.preamble'] = (
    r'\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}')

#%%
#Setup prior and noise variables

# # prior_rvs = [stats.beta(2, 2, loc=-1, scale=2)]
# prior_rvs = [stats.norm(0.0, 0.25)]
# prior_variable = IndependentMarginalsVariable(prior_rvs)
# noise_std = 0.5
# noise_rv = stats.norm(0, noise_std)

# def obs_fun(design, params):
#     assert design.ndim == 2 and params.ndim == 2
#     assert design.shape[0] == 1 and params.shape[0] == 1
#     power = 2
#     # when lb, ub = 0, 1
#     # vals = (((params[0]*2-1)+design[0])**power)[:, None]
#     # when lb, ub = -1, 1
#     vals = ((params[0]+0.5*design[0])**power)[:, None]
#     # vals = ((1 + params[0]*design[0])**power)[:, None]
#     return vals


# def qoi_fun(pred, params):
#     assert pred.ndim == 2 and params.ndim == 2
#     assert pred.shape[0] == 1 and params.shape[0] == 1
#     # keep values in [0, 1] so can keep qq the sa
#     # power = 2
#     # vals = np.exp(((1+params.T*(pred[0]+1.1))**power))
#     # vals = np.exp(3-30/8*params.T*(1.01+pred[0])+pred[0]**4*35/8)
#     factor = (2+np.cos(2*np.pi*pred[0]))
#     vals = np.exp(params.T)*factor
#     return vals


#prior_rvs = [stats.norm(0.0, 0.25) for ii in range(2)]
prior_rvs = [stats.beta(2, 2, loc=-1, scale=2) for ii in range(2)]
prior_variable = IndependentMarginalsVariable(prior_rvs)
noise_std = 0.25
noise_rv = stats.norm(0, noise_std)


def obs_fun(design, params):
    assert design.ndim == 2 and params.ndim == 2
    assert design.shape[0] == 1
    nparams = params.shape[0]
    # vals = params[0][:, None]+(
    #     (design.T**np.arange(1, nparams+1)[None, :]) @ params).T
    vals = ((((design.T**np.arange(nparams)[None, :]) @ params).T)+0.5*design)**2
    return vals


def qoi_fun(pred, params):
    vals = np.exp(obs_fun(pred, params))
    return vals


#%%
#Plot prior
if prior_variable.num_vars() == 1:
    lb, ub = get_truncated_range(prior_rvs[0])
    prior_xx = np.linspace(lb, ub, 101)
    plt.plot(prior_xx, prior_rvs[0].pdf(prior_xx))
    plt.fill_between(
        prior_xx, 0, prior_rvs[0].pdf(prior_xx), alpha=0.4, label=r"$\pi(\theta)$")
    plt.legend()
    if savefigs:
        plt.savefig("oed-workflow-prior.pdf")


#%%
#Define observation operator

def get_noisy_data(design_pt, samples):
    true_samples = samples[:1, :]
    noises = samples[1, :]
    noiseless_data = obs_fun(design_pt, true_samples)[:, 0]
    data = noiseless_data + noises
    return data


#%%
#Plot observations as a function of the ra variable for each design
def plot_data_surface_ribbons(design_pts, design_symbs, ax):
    normalized_y = np.linspace(0, 1, len(design_pts))
    ribbon_width = 1/(len(design_pts))
    plot_data = []
    for ii, design_pt in enumerate(design_pts):
        X, Y, Z = get_meshgrid_function_data(
            lambda x: obs_fun(x[:1], x[1:2]),
            [-1, 1, design_pt[0, 0], design_pt[0, 0]], [50, 1])
        X = np.vstack([X, X])
        # Y = np.vstack([Y, Y+ribbon_width])
        Y = np.vstack(
            [np.full(Y.shape, normalized_y[ii]-ribbon_width/2),
             np.full(Y.shape, normalized_y[ii]+ribbon_width/2)])
        Z = np.vstack([Z, Z])
        plot_data.append([X, Y, Z])

    vmin = np.min([data[2].min() for data in plot_data])
    vmax = np.max([data[2].max() for data in plot_data])
    for data in plot_data:
        X, Y, Z = data
        # plot_surface(X, Y, Z, ax, plot_axes=False)
        ax.plot_surface(X, Y, Z, cmap="coolwarm", vmin=vmin, vmax=vmax)
    _turn_off_3d_axes(ax)
    # ax.set_yticks([pt[0, 0] for pt in design_pts])
    ax.set_yticks(normalized_y)
    ax.set_yticklabels([r"$%s$" % s for s in design_symbs])
    # ax.set_ylabel(mathrm_label("Design"))
    # plot prior on right wall
    # ax.plot(xx, np.ones(xx.shape[0])+ribbon_width/2, prior_rv.pdf(xx),
    #               zorder=-1)
    ax.set_ylim([-ribbon_width/2, 1+ribbon_width/2])
    ax.set_xlabel(mathrm_label("Parameter")+r" $\theta$")


# design_pts = [np.array([[-1]]), np.array([[0.5]]), np.array([[1]])]
design_pts = [np.array([[0.25]]), np.array([[0.5]]), np.array([[1]])]
design_symbs = [r"\xi", r"\xi^\prime", r"\xi^\dagger"]

if prior_variable.num_vars() == 1:
    ax_ribbon = create_3d_axis()
    plot_data_surface_ribbons(design_pts, design_symbs, ax_ribbon)
    if savefigs:
        plt.savefig("oed-workflow-data-ribbons.pdf")


#%%
#Plot the response surface of the observation function as a function of
#the random parameter and noise
def plot_data_surfaces(
        design_pts, lb, ub, noise_plot_bounds, design_symbs, axs_data,
        horizontal=False):
    plot_data = []
    for ii, design_pt in enumerate(design_pts):
        X, Y, Z = get_meshgrid_function_data(
            partial(get_noisy_data, design_pt),
            [lb, ub, noise_plot_bounds[0], noise_plot_bounds[1]], 50)
        plot_data.append([X, Y, Z])

    vmin = np.min([data[2].min() for data in plot_data])
    vmax = np.max([data[2].max() for data in plot_data])

    for ii, data in enumerate(plot_data):
        X, Y, Z = data
        im = axs_data[ii].contourf(
            X, Y, Z, levels=np.linspace(vmin, vmax, 21), cmap="coolwarm")
        plt.colorbar(im, ax=axs_data[ii])

    if not horizontal:
        for ii, data in enumerate(plot_data):
            axs_data[ii].set_ylabel(mathrm_label("Noise")+r" $\eta$")
            axs_data[ii].set_title(
                mathrm_label(
                    "Possible data")+r" $y=M(\theta, %s)+\eta$" % (
                        design_symbs[ii]))
            axs_data[-1].set_xlabel(mathrm_label("Parameter")+r" $\theta$")
        return

    for ii, data in enumerate(plot_data):
        axs_data[ii].set_xlabel(mathrm_label("Parameter")+r" $\theta$")
        axs_data[ii].set_title(
            mathrm_label(
                "Possible data")+r" $y=M(\theta, %s)+\eta$" % (
                    design_symbs[ii]))
    axs_data[0].set_ylabel(mathrm_label("Noise")+r" $\eta$")


if prior_variable.num_vars() == 1:
    unbounded_alpha = 1-1e-3  # controls how wide plot range is
    noise_plot_bounds = get_truncated_range(
        noise_rv, bounded_alpha=unbounded_alpha)
    fig_data, axs_data = subplots(len(design_pts))
    plot_data_surfaces(
        design_pts, lb, ub, noise_plot_bounds, design_symbs, axs_data,
        horizontal=horizontal)
    if savefigs:
        fig_data.savefig("oed-workflow-data-surfaces.pdf")


#%%
#Plot three realizations of the observational data and the associated
#posterior and its pushforward PDF and CDF
def get_loglike(samples, design_pt, noise_std, ii):
    # true_sample = quad_data[0][:1, ii:ii+1]
    # noise = quad_data[0][1, ii]
    samples = samples[:, ii:ii+1]
    data = get_noisy_data(design_pt, samples)
    loglike = GaussianLogLike(
        partial(obs_fun, design_pt), data, noise_std**2)
    return loglike


def eval_post_pdf(loglike, prior_pdf, evidence, x):
    x = np.atleast_2d(x)
    return np.exp(loglike(x)[:, 0])*prior_pdf(x[0])/evidence


def plot_push_cdf(pdf, bounds, cvar_p, qq, color, label, ax, ls):
    cvar, quantile = univariate_cvar_continuous_variable(
        pdf, bounds, cvar_p, opt_tol=1e-7, return_quantile=True)
    func = partial(univariate_cdf_continuous_variable,
                   pdf, *bounds, quad_opts={})
    cdf_vals = func(qq)
    ax.plot(qq, cdf_vals, c=color, label=label, ls=ls)
    ax.fill_between(
        qq[qq > quantile], cdf_vals[qq > quantile], 1, color=color,
        alpha=0.3)


def plot_posteriors(
        prior_quad_xx, prior_quad_ww, prior_rv, lb, ub, design_pts,
        cvar_p, joint_prior_noise_variable, noise_std,
        data_markers, data_latex_markers, design_symbs,
        axs_pdf, axs_data, horizontal=False):
    xx = np.linspace(lb, ub, 101)
    prior_quad_xx, prior_quad_ww = prior_quad_xx[0], prior_quad_ww[:, 0]
    prior_pdf = get_pdf(prior_rv)
    prior_noise_samples = joint_prior_noise_variable.rvs(len(data_markers))
    linestyles = ["--", "-.", ":"]
    for ii, design_pt in enumerate(design_pts):
        im, = axs_pdf[ii].plot(xx, prior_pdf(xx))
        axs_pdf[ii].fill_between(
            xx, 0, prior_pdf(xx), alpha=0.4, label=r"$\pi(\theta)$")
        color = im.get_color()
        for jj in range(len(data_markers)):
            ls = linestyles[jj]
            prior_noise_sample = prior_noise_samples[:, jj:jj+1]
            loglike = get_loglike(prior_noise_sample, design_pt, noise_std, 0)
            like_vals = np.exp(loglike(prior_quad_xx[None, :])[:, 0])
            # prior factored into quadrature rule weights ww
            evidence = like_vals.dot(prior_quad_ww)
            post_pdf = partial(eval_post_pdf, loglike, prior_pdf, evidence)
            im, = axs_pdf[ii].plot(
                xx, post_pdf(xx), label=r"$\pi(\theta\mid y^{%s},%s)$" % (
                    data_latex_markers[jj], design_symbs[ii]), ls=ls)
            color = im.get_color()
            axs_data[ii].plot(
                *prior_noise_sample, marker=data_markers[jj],
                markersize=20, c='k')
        axs_pdf[ii].legend()
        if horizontal:
            axs_pdf[ii].set_xlabel(r"$\mathrm{Variable}\;\theta$")
    if not horizontal:
        axs_pdf[-1].set_xlabel(r"$\mathrm{Variable}\;\theta$")
    return prior_noise_samples


cvar_p1 = 0.9
cvar_p2 = 0.2
data_markers = ["X", "s", "o"]
data_latex_markers = [r"\times", r"\square", r"\circ"]
joint_prior_noise_variable = IndependentMarginalsVariable(
    prior_rvs + [noise_rv])
if prior_variable.num_vars() == 1:
    prior_quad_xx, prior_quad_ww = integrate(
        "tensorproduct", prior_variable, rule="quadratic", levels=100)
else:
    prior_quad_xx, prior_quad_ww = integrate(
        "quasimontecarlo", prior_variable, rule="halton", nsamples=1000)

# plot data surfaces again now with random observations on it
if prior_variable.num_vars() == 1:
    fig_data, axs_data = subplots(len(design_pts))
    plot_data_surfaces(
        design_pts, lb, ub, noise_plot_bounds, design_symbs, axs_data,
        horizontal=horizontal)
    fig_pdf, axs_pdf = subplots(len(design_pts))
    plot_posteriors(
        prior_quad_xx, prior_quad_ww, prior_rvs[0], lb, ub, design_pts,
        cvar_p1, joint_prior_noise_variable, noise_std,
        data_markers, data_latex_markers, design_symbs,
        axs_pdf, axs_data, horizontal=horizontal)
    if savefigs:
        fig_pdf.savefig("oed-workflow-pdfs.pdf")
        fig_data.savefig("oed-workflow-data-surfaces-update.pdf")


#%%
# Plot posterior push forward
def small_noise_likelihood(vals, noise_std, Q):
    rv = stats.norm(0, noise_std)
    pdf = get_pdf(rv)
    pdf_vals = pdf(vals-Q)
    return pdf_vals


def estimate_density(vals, quad_w, noise_std, qvals):
    assert vals.ndim == 2 and vals.shape[1] == 1
    assert quad_w.ndim == 2 and quad_w.shape[1] == 1
    like_vals = small_noise_likelihood(vals, noise_std, qvals)
    return like_vals.T.dot(quad_w)


def pushforward_pdf(qoi_fun, quad_xx, quad_ww, small_noise_std, qq):
    zz = qoi_fun(quad_xx)
    density_vals = estimate_density(
        zz, quad_ww, small_noise_std, qq)
    return density_vals[:, 0]


def posterior_pdf(loglike, prior_pdf, evidence, x):
    assert isinstance(x, float)
    return np.exp(
        loglike(np.atleast_2d(x))[0, 0])*prior_pdf(x)/evidence


def get_posterior_weights(design_pt, prior_noise_quad_xx, noise_std, prior_quad_xx,
                          prior_quad_ww, qoi_fun, ii):
    loglike = get_loglike(prior_noise_quad_xx, design_pt, noise_std, ii)
    like_vals = np.exp(loglike(prior_quad_xx)[:, 0])
    # prior factored into quadrature rule weights ww
    zz = qoi_fun(prior_quad_xx)
    assert zz.shape[1] == 1, zz.shape
    zz = zz[:, 0]
    evidence = like_vals.dot(prior_quad_ww)
    ww_post = like_vals*prior_quad_ww/evidence
    return ww_post, evidence, loglike, zz, like_vals


def get_posterior_and_pushforward_functions(
        design_pt, prior_noise_quad_xx, noise_std,
        prior_quad_xx, prior_quad_ww, prior_rv, qoi_fun, prior_variable,
        prior_density_quad_xx, prior_density_quad_ww, ii):
    ww_post, evidence, loglike, zz, like_vals = get_posterior_weights(
        design_pt, prior_noise_quad_xx, noise_std, prior_quad_xx,
        prior_quad_ww, qoi_fun, ii)
    print(prior_density_quad_xx.shape)
    if prior_variable.num_vars() == 1:
        prior_pdf = get_pdf(prior_rv[0])
    else:
        prior_pdf = prior_variable._pdf
    post_pdf = partial(posterior_pdf, loglike, prior_pdf, evidence)

    post_density_quad_xx = prior_density_quad_xx
    post_density_quad_ww = prior_density_quad_ww*np.exp(
        loglike(prior_density_quad_xx))/evidence
    # ndensity_quad_samples = prior_density_quad_xx.shape[1]
    # tol = 1/(ndensity_quad_samples/100)
    # tol = 1e-3
    tol = 8e-4
    post_push_pdf = partial(
        pushforward_pdf, qoi_fun, post_density_quad_xx, post_density_quad_ww,
        tol)
    return post_pdf, post_push_pdf, zz, ww_post, evidence, like_vals


def plot_pushforwards(prior_variable, qq, qoi_fun, design_pts, cvar_p,
                      joint_prior_noise_variable, prior_quad_xx, prior_quad_ww,
                      noise_std, prior_rv, qbounds,
                      data_markers, data_latex_markers,
                      design_symbs, axs_pdf, axs_cdf, horizontal=False):
    if prior_variable.num_vars() == 1:
        ndensity_quad_samples = int(1e4)+1
        prior_density_quad_xx, prior_density_quad_ww = integrate(
            "tensorproduct", prior_variable, levels=ndensity_quad_samples,
            rule="quadratic")
        tol = 8e-4
    else:
        tol = 8e-3
        ndensity_quad_samples = int(1e4)+1# int(1e6)+1
        prior_density_quad_xx, prior_density_quad_ww = integrate(
            "quasimontecarlo", prior_variable, nsamples=ndensity_quad_samples)
    # tol = 1/(ndensity_quad_samples/100)

    prior_push_pdf = partial(
        pushforward_pdf, qoi_fun, prior_density_quad_xx,
        prior_density_quad_ww, tol)
    prior_pdf_vals = prior_push_pdf(qq)
    linestyles = ["-", "--", ":"]
    for ii, design_pt in enumerate(design_pts):
        im, = axs_pdf[ii].plot(qq, prior_pdf_vals)
        axs_pdf[ii].fill_between(
            qq, 0, prior_pdf_vals, alpha=0.4, label=r"$\pi(q(\theta))$")
        color = im.get_color()
        # CDFS are expensive so turn off for now
        # plot_push_cdf(prior_push_pdf, qbounds, cvar_p, qq, color,
        #               r"$\mathbb{P}(q(\theta))$", axs_cdf[ii], "-")

        for jj in range(len(data_markers)):
            ls = linestyles[jj]
            # todo prior_noise_sample the same as in plot_posteriors
            prior_noise_sample = joint_prior_noise_variable.rvs(1)
            post_push_pdf = get_posterior_and_pushforward_functions(
                design_pt, prior_noise_sample, noise_std,
                prior_quad_xx, prior_quad_ww, prior_rv, qoi_fun,
                prior_variable, prior_density_quad_xx, prior_density_quad_ww,
                0)[1]
            im, = axs_pdf[ii].plot(
                qq, post_push_pdf(qq),
                label=r"$\pi(q(\theta)\mid y^{%s},%s)$" % (
                    data_latex_markers[jj], design_symbs[ii]))
            color = im.get_color()
            label = r"$\mathbb{P}(q(\theta)\mid y^{%s},%s)$" % (
                data_latex_markers[jj], design_symbs[ii])
            # plot_push_cdf(post_push_pdf, qbounds, cvar_p, qq, color,
            #               label, axs_cdf[ii], ls)
        axs_pdf[ii].legend()
        axs_cdf[ii].legend()
        if horizontal:
            axs_pdf[ii].set_xlabel(mathrm_label("QoI")+r" $q(\theta)$")
            axs_cdf[ii].set_xlabel(mathrm_label("QoI")+r" $q(\theta)$")
    if not horizontal:
        axs_pdf[ii].set_xlabel(mathrm_label("QoI")+r" $q(\theta)$")
        axs_cdf[-1].set_xlabel(mathrm_label("QoI")+r" $q(\theta)$")


npred_pts = 17
pred_rv = stats.uniform(-1, 2)
pred_variable = IndependentMarginalsVariable([pred_rv])
pred_pts, pred_wts = integrate(
    "tensorproduct", pred_variable, rule="quadratic", levels=npred_pts-1)
pred_idx = npred_pts//4*3
assert pred_pts[0, pred_idx] == 0.5

# assumes monotonic increasing function in uncertain parameter
# qbounds = np.stack(
#     [qoi_fun(x[None, :], np.array([[lb, ub]]))[:, 0] for x in pred_pts.T])
qoi_vals = qoi_fun(pred_pts, prior_variable.rvs(10000))
qbounds = np.stack([[q.min(), q.max()] for q in qoi_vals.T])

# turn off temporarily because it is expensive
fig_push_pdf_horz, axs_push_pdf_horz = plt.subplots(
    1, len(design_pts), figsize=(len(design_pts)*8, 1*6), sharey=True)
fig_push_cdf_horz, axs_push_cdf_horz = plt.subplots(
    1, len(design_pts), figsize=(len(design_pts)*8, 1*6), sharey=True)
qq = np.linspace(*qbounds[pred_idx], 51)
plot_pushforwards(
    prior_variable, qq, partial(qoi_fun, pred_pts[:, pred_idx:pred_idx+1]),
    design_pts, cvar_p1, joint_prior_noise_variable,
    prior_quad_xx, prior_quad_ww,
    noise_std, prior_rvs, qbounds[pred_idx], data_markers, data_latex_markers,
    design_symbs, axs_push_pdf_horz, axs_push_cdf_horz, horizontal=horizontal)
if savefigs:
    fig_push_pdf_horz.savefig("oed-workflow-push-pdfs.pdf")
    fig_push_cdf_horz.savefig("oed-workflow-push-cdfs.pdf")


#%%
#Compute deviations for each design and prediction point
def _compute_deviations(design_pt, prior_noise_quad_xx, noise_std,
                        prior_quad_xx, prior_quad_ww,
                        cvar_p1, cvar_p2, qoi_fun, prior_variable,
                        nonlinear_qoi, ii):
    assert prior_quad_ww.shape[1] == 1
    prior_quad_ww = prior_quad_ww[:, 0]

    ww_post, evidence, loglike, zz, like_vals = get_posterior_weights(
        design_pt, prior_noise_quad_xx, noise_std, prior_quad_xx,
        prior_quad_ww, qoi_fun, ii)

    mean = zz.dot(ww_post)
    # The following is numerically unstable
    # stdev = np.sqrt((zz**2).dot(ww_post) - mean**2)
    stdev = np.sqrt(((zz-mean)**2).dot(ww_post))
    entropic_dev = np.log(np.exp(zz).dot(ww_post)) - mean
    assert stdev >= 0, (stdev)
    assert entropic_dev >= 0, (
        entropic_dev, np.exp(zz).max(), np.exp(zz).min())
    # entropic_dev1 = np.log(np.exp(zz-mean).dot(ww_post))
    # assert np.allclose(entropic_dev, entropic_dev1)

    kl_div = (np.log(like_vals)-np.log(evidence)).dot(ww_post)
    assert kl_div > 0

    cvar, quantile = conditional_value_at_risk(
        zz, cvar_p1, ww_post, return_var=True)
    cvar_dev = cvar - mean
    assert cvar_dev >= 0, (ii, cvar_dev)

    cvar1, quantile1 = conditional_value_at_risk(
        zz, cvar_p2, ww_post, return_var=True)
    cvar_dev1 = cvar1 - mean
    assert cvar_dev1 >= 0, (ii, cvar_dev1)

    return stdev, entropic_dev, cvar_dev, cvar_dev1, kl_div


def compute_deviations(design_pt, prior_noise_quad_data, noise_std, xx, ww,
                       cvar_p1, cvar_p2, qoi_fun, prior_variable,
                       nonlinear_qoi):
    nsamples = prior_noise_quad_data[0].shape[1]

    if max_eval_concurrency > 1:
        pool = Pool(max_eval_concurrency)
        deviations = pool.map(
            partial(_compute_deviations,
                    design_pt, prior_noise_quad_data[0], noise_std, xx, ww,
                    cvar_p1, cvar_p2, qoi_fun, prior_variable,
                    nonlinear_qoi),
            list(range(nsamples)))
        pool.close()
        return deviations

    deviations = []
    for ii in range(nsamples):
        devs = _compute_deviations(
            design_pt, prior_noise_quad_data[0], noise_std, xx, ww,
            cvar_p1, cvar_p2, qoi_fun, prior_variable,
            nonlinear_qoi, ii)
        deviations.append(devs)
    return deviations


nonlinear_qoi = True

if prior_variable.num_vars() == 2:
    levels = np.array([10 for ii in range(prior_variable.num_vars()+1)])
else:
    levels = np.array([20 for ii in range(prior_variable.num_vars()+1)])
nsamples_1d = levels+1
# basis_type = "quadratic"
basis_type = "linear"
prior_noise_quad_data = integrate(
    "tensorproduct", joint_prior_noise_variable,
    rule=basis_type, levels=levels)

deviations_filename = (
    f"oed-workflow-deviations-{levels}-{nonlinear_qoi}-{npred_pts}.npz")
if not os.path.exists(deviations_filename):
    print(f"Creating {deviations_filename}")
    deviations = []
    for ii, design_pt in enumerate(design_pts):
        deviations_ii = []
        for jj, pred_pt in enumerate(pred_pts.T):
            print(ii, jj, "@@@")
            deviations_ii.append(compute_deviations(
                design_pt, prior_noise_quad_data, noise_std,
                prior_quad_xx, prior_quad_ww,
                cvar_p1, cvar_p2, partial(qoi_fun, pred_pt[None, :]),
                prior_variable, nonlinear_qoi))
        deviations.append(deviations_ii)
    deviations = np.array(deviations)
    assert np.all(deviations > 0), (deviations.min())
    np.savez(deviations_filename, deviations=deviations)
else:
    print(f"Loading {deviations_filename}")
    deviations = np.load(deviations_filename)["deviations"]

# exclude entropic deviation
deviations = deviations[..., [0, 2, 3, 4]]
deviation_symbs = [
    r"\sigma", r"\mathrm{AVaRD}_{%1.1f}" % cvar_p1,
    r"\mathrm{AVaRD}_{%1.1f}" % cvar_p2, r"\mathrm{KL}"]


#%%
#Plot the KL divergence for a single prediction point as a function of the data
pred_idx = npred_pts//4*3
assert pred_pts[0, pred_idx] == 0.5
dev_idx = 3  # 3 corresponds to KL


def interpolate_deviation(nsamples_1d, basis_type, quad_data, deviations,
                          samples):
    # assumes same samples for each dimension
    abscissa_1d = [quad_data[0][0, :nsamples_1d[0]],
                   quad_data[0][1, ::nsamples_1d[0]]]
    assert deviations.ndim == 1
    interp = TensorProductInterpolant(
        [get_univariate_interpolation_basis(basis_type) for ii in range(2)])
    interp.fit(abscissa_1d, deviations)
    vals = interp(samples)
    return vals


def plot_deviation_surfaces(
        deviations, prior_noise_quad_data, lb, ub, noise_plot_bounds,
        design_symbs, deviation_symb, axs, basis_type, nsamples_1d,
        horizontal=False, pf_based=True):
    Z_vals, Z_bounds = [], []
    for ii in range(deviations.shape[0]):
        interp = partial(
            interpolate_deviation, nsamples_1d, basis_type,
            prior_noise_quad_data, deviations[ii, :])
        X, Y, Z = get_meshgrid_function_data(
            interp, [lb, ub, noise_plot_bounds[0], noise_plot_bounds[1]], 50)
        Z_vals.append(Z)
        Z_bounds.append([Z.min(), Z.max()])
    Z_bounds = np.asarray(Z_bounds)

    for ii in range(len(Z_vals)):
        Z_min = Z_bounds[:, 0].min(axis=0)
        Z_max = Z_bounds[:, 1].max(axis=0)
        im = axs[ii].contourf(
            X, Y, Z_vals[ii], levels=np.linspace(Z_min, Z_max, 21),
            cmap="coolwarm")
        if pf_based:
            axs[ii].set_title(r" $%s[\pi(q(\theta)\mid y,%s)]$" % (
                deviation_symb, design_symbs[ii]))
        else:
            axs[ii].set_title(r" $%s[\pi(\theta\mid y,%s)]$" % (
                deviation_symb, design_symbs[ii]))
        plt.colorbar(im, ax=axs[ii])
        if horizontal:
            axs[ii].set_xlabel(mathrm_label("Parameter")+r" $\theta$")
        else:
            axs[ii].set_ylabel(mathrm_label("Noise")+r" $\eta$")
    if not horizontal:
        axs[-1].set_xlabel(mathrm_label("Parameter")+r" $\theta$")
    else:
        axs[0].set_ylabel(mathrm_label("Noise")+r" $\eta$")


if prior_variable.num_vars() == 1:
    fig_kl_surf, axs_kl_surf = subplots(len(design_pts))
    plot_deviation_surfaces(
        deviations[:, pred_idx, ..., dev_idx], prior_noise_quad_data, lb, ub,
        noise_plot_bounds, design_symbs, deviation_symbs[dev_idx], axs_kl_surf,
        basis_type, nsamples_1d, True, False)
    if savefigs:
        fig_kl_surf.savefig("oed-workflow-kl-div-surfaces.pdf")


#%%
#Plot CVaR deviation surfaces
if prior_variable.num_vars() == 1:
    fig_dev_surf, axs_dev_surf = subplots(len(design_pts))
    dev_idx = 1  # 1 corresponds to cvar_p1
    plot_deviation_surfaces(
        deviations[:, pred_idx, ..., dev_idx], prior_noise_quad_data, lb, ub,
        noise_plot_bounds, design_symbs, deviation_symbs[dev_idx], axs_dev_surf,
        basis_type, nsamples_1d, True)
    if savefigs:
        fig_dev_surf.savefig("oed-workflow-avar-dev-surfaces.pdf")

#%%
#Plot the PDF of the KL divergence as a function of the noise

# use small noise approximation which needs QMC or MC samples
njoint_xx = int(1e5)
joint_qmc_xx, joint_qmc_ww = integrate(
    "quasimontecarlo", joint_prior_noise_variable, nsamples=njoint_xx,
    rule="halton")


def plot_deviation_pdf(prior_noise_quad_data, deviations, joint_xx,
                       joint_ww, deviation_symb, design_symb, ax,
                       basis_type, nsamples_1d, pf_based, **kwargs):
    interp = partial(
        interpolate_deviation, nsamples_1d, basis_type, prior_noise_quad_data,
        deviations)
    interp_vals = interp(joint_xx)
    assert np.all(interp_vals > 0), (interp_vals.min(), interp_vals.max(),
                                     deviations.min(), deviations.max())
    qq = np.linspace(0, interp_vals.max(), 401)
    small_noise_std = 0.0075
    joint_vals = interp(joint_xx)
    density_vals = estimate_density(
        joint_vals, joint_ww, small_noise_std, qq)
    if pf_based:
        ax.plot(qq, density_vals, label=r"$\pi(%s[q(\theta) \mid y,%s])$" % (
            deviation_symb, design_symb), **kwargs)
    else:
        ax.plot(qq, density_vals, label=r"$\pi(%s[\theta \mid y,%s])$" % (
            deviation_symb, design_symb), **kwargs)
    ax.fill_between(qq, 0, density_vals[:, 0], alpha=0.3)
    ub = max(ax.get_xlim()[1],
             qq[density_vals[:, 0] > 1e-2*density_vals[:, 0].max()][-1])
    ax.set_xlim(0, ub)


fig_kl_pdf, axs_kl_pdf = subplots(1)
dev_idx = 3
for ii, design_pt in enumerate(design_pts):
    plot_deviation_pdf(
        prior_noise_quad_data, deviations[ii, pred_idx, :, dev_idx],
        joint_qmc_xx, joint_qmc_ww, deviation_symbs[dev_idx], design_symbs[ii],
        axs_kl_pdf, basis_type, nsamples_1d, False)
axs_kl_pdf.legend()
axs_kl_pdf.set_xlabel(mathrm_label("KL Divergence"))
if savefigs:
    fig_kl_pdf.savefig("oed-workflow-kl-div-pdfs.pdf")


#%%
#Plot the PDF of the CVaR deviations
fig_cvar_pdf, axs_cvar_pdf = subplots(1)
dev_idx = 1
for ii, design_pt in enumerate(design_pts):
    plot_deviation_pdf(
        prior_noise_quad_data, deviations[ii, pred_idx, :, dev_idx],
        joint_qmc_xx, joint_qmc_ww, deviation_symbs[dev_idx], design_symbs[ii],
        axs_cvar_pdf, basis_type, nsamples_1d, True)
axs_cvar_pdf.legend()
axs_cvar_pdf.set_xlabel(mathrm_label("Deviation"))
if savefigs:
    fig_cvar_pdf.savefig("oed-workflow-cvar-dev-pdfs.pdf")

#%%
#Compute the expected utilities for all design pts and deviation measures
utilities = []
for ii, design_pt in enumerate(design_pts):
    utilities.append(
        np.sum(prior_noise_quad_data[1]*deviations[ii, pred_idx], axis=0))
utilities = np.asarray(utilities)


#%%
#Plot the expected KL-based expected utility
def plot_utilities(utilities, design_symbs, deviation_symbs, ax):
    bar_width = 0.25
    multiplier = 0
    index = np.arange(utilities.shape[1])
    colors = plt.cm.coolwarm(np.linspace(0, 0.5, utilities.shape[0]))
    for ii in range(utilities.shape[0]):
        offset = bar_width*multiplier
        rects = ax.bar(
            index+offset, utilities[ii], bar_width,
            label="$%s$" % design_symbs[ii], color=colors[ii])
        ax.bar_label(rects, fmt=r"$%1.3f$", padding=1)
        multiplier += 1.0
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels([r"$%s$" % s for s in deviation_symbs])
    ax.legend(loc=0, ncol=utilities.shape[0], columnspacing=0.5)
    ax.set_ylabel(r"$-\psi(\xi)$")


fig_kl_utility, axs_kl_utility = subplots(1)
plot_utilities(utilities[..., -1:], design_symbs,
               deviation_symbs[-1:], axs_kl_utility)
bar_width = 0.25
index = np.arange(len(design_symbs))*bar_width
axs_kl_utility.set_xticks(index)
axs_kl_utility.set_xticklabels([r"$%s$" % s for s in design_symbs])
axs_kl_utility.set_xlabel(r"$\mathrm{Design}\; \xi$")
axs_kl_utility.get_legend().remove()
if savefigs:
    fig_kl_utility.savefig("oed-workflow-expected-kl-utilities.pdf")


#%%
#Plot the divergence based expected utilities

fig_utility, axs_utility = plt.subplots(1)
utilities = np.asarray(utilities)
plot_utilities(utilities[..., :-1], design_symbs, deviation_symbs[:-1],
               axs_utility)
if savefigs:
    fig_utility.savefig("oed-workflow-expected-utilities.pdf")


#%%
#Plot prediction deviation surfaces
def plot_prediction_deviation_surface(deviations, axs, indices, pred_pts):
    Z_list = []
    Zmin, Zmax = np.inf, -np.inf
    for ii in range(deviations.shape[0]):
        Z_list_ii = []
        for idx in indices:
            interp = partial(
                interpolate_deviation, nsamples_1d, basis_type,
                prior_noise_quad_data, deviations[ii, idx])
            X, Y, Z = get_meshgrid_function_data(
                interp, [lb, ub, noise_plot_bounds[0], noise_plot_bounds[1]],
                50)
            Zmax = max(Z.max(), Zmax)
            Zmin = min(Z.min(), Zmin)
            Z_list_ii.append(Z)
        Z_list.append(Z_list_ii)

    stride = 4/(len(indices)-1)  # be on [0, 4] on z-axis
    for ii in range(deviations.shape[0]):
        cnt = 0
        for idx, Z in zip(indices, Z_list[ii]):
            offset = cnt*stride
            axs[ii].contourf(
                X, Y, Z, levels=np.linspace(Zmin, Zmax, 21), cmap="coolwarm",
                offset=offset)
            cnt += 1
        _turn_off_3d_axes(axs[ii])


if prior_variable.num_vars() == 1:
    indices = np.hstack((np.arange(npred_pts)[::3], npred_pts-1))
    axs = plt.subplots(1, 3, figsize=(3*8, 6), subplot_kw={"projection": "3d"})[1]
    dev_idx = 1
    plot_prediction_deviation_surface(
        deviations[..., dev_idx], axs, indices, pred_pts)

#%%
#Plot expected deviation over prediction space
def plot_risk_prediction_deviation_surface(
        prior_noise_quad_data, deviations, joint_xx,
        joint_ww, deviation_symb, design_symb, ax,
        basis_type, nsamples_1d, pred_wts, pred_pts, **kwargs):
    expected_deviations = (deviations @ prior_noise_quad_data[1])[:, 0]
    xx = np.linspace(pred_pts[0, 0], pred_pts[0, -1], 101)[None, :]
    interp = TensorProductInterpolant(
        [get_univariate_interpolation_basis(basis_type)])
    interp.fit(pred_pts, expected_deviations)
    vals = interp(xx)
    ax.plot(xx[0], vals)


dev_idx = 1
fig, axs = subplots(3)
for ii, design_pt in enumerate(design_pts):
    plot_risk_prediction_deviation_surface(
        prior_noise_quad_data, deviations[ii, ..., dev_idx],
        joint_qmc_xx, joint_qmc_ww, deviation_symbs[dev_idx], design_symbs[ii],
        axs[ii], basis_type, nsamples_1d, pred_wts, pred_pts)
plt.show()
assert False


#%%
#Plot the densities of the expected (over the prediction space) deviation
def plot_risk_prediction_deviation_pdf(
        prior_noise_quad_data, deviations, joint_xx,
        joint_ww, deviation_symb, design_symb, ax,
        basis_type, nsamples_1d, pred_wts, **kwargs):

    risk_deviations = (deviations.T @ pred_wts)[:, 0]

    risk_deviations = []
    cvar_p3 = 0.5
    risk_label = r"\mathcal{R}_{%1.1f}" % cvar_p3
    for ii in range(deviations.shape[1]):
        cvar, quantile = conditional_value_at_risk(
            deviations[:, ii], cvar_p3, pred_wts[:, 0], return_var=True)
        risk_deviations.append(cvar)
    risk_deviations = np.array(risk_deviations)

    interp = partial(
        interpolate_deviation, nsamples_1d, basis_type, prior_noise_quad_data,
        risk_deviations)
    interp_vals = interp(joint_xx)
    assert np.all(interp_vals > 0), (interp_vals.min(), interp_vals.max(),
                                     deviations.min(), deviations.max())
    qq = np.linspace(0, interp_vals.max(), 201)
    small_noise_std = 0.0075
    joint_vals = interp(joint_xx)
    density_vals = estimate_density(
        joint_vals, joint_ww, small_noise_std, qq)
    ax.plot(qq, density_vals, label=r"$\pi(%s[%s[q(\theta) \mid y,%s]])$" % (
        risk_label, deviation_symb, design_symb), **kwargs)
    ax.fill_between(qq, 0, density_vals[:, 0], alpha=0.3)


fig_kl_pred_pdf, axs_kl_pred_pdf = subplots(1)
dev_idx = 0
for ii, design_pt in enumerate(design_pts):
    plot_risk_prediction_deviation_pdf(
        prior_noise_quad_data, deviations[ii, ..., dev_idx],
        joint_qmc_xx, joint_qmc_ww, deviation_symbs[dev_idx], design_symbs[ii],
        axs_kl_pred_pdf, basis_type, nsamples_1d, pred_wts)
axs_kl_pred_pdf.legend()
axs_kl_pred_pdf.set_xlabel(mathrm_label("Divergence") + r" $\phi$")
if savefigs:
    fig_kl_pred_pdf.savefig("oed-workflow-kl-pred-div-pdfs.pdf")

plt.show()
