r"""
Bayesian Inference
==================
This tutorial describes how to use Bayesian inference condition estimates of uncertainty on observational data.

Background
----------
When observational data are available, that data should be used to inform prior assumptions of model uncertainties. This so-called inverse problem that seeks to estimate uncertain parameters from measurements or observations is usually ill-posed. Many different realizations of parameter values may be consistent with the data. The lack of a unique solution can be due to the non-convexity of the parameter-to-QoI map, lack of data, and model structure and measurement errors.

Deterministic model calibration is an inverse problem that seeks to find a single parameter set that minimizes the misfit between the measurements and model predictions. A unique solution is found by simultaneously minimising the misfit and a regularization term which penalises certain characteristics of the model parameters.

In the presence of uncertainty we typically do not want a single optimal solution, but rather a probabilistic description of the extent to which different realizations of parameters are consistent with the observations.
Bayesian inference [KAIPO2005]_ can be used to define a posterior density for the model parameters :math:`\rv` given
observational data :math:`\V{y}=(y_1,\ldots,y_{n_y})`:

Bayes Rule
^^^^^^^^^^
Given a model :math:`\mathcal{M}(\rv)` parameterized by a set of parameters :math:`\rv`, our goal is to infer the parameter :math:`\rv` from data :math:`d`.

Bayes Theorem describes the probability of the parameters :math:`\rv` conditioned on the data :math:`d` is proportional to the conditional probability of observing the data given the parameters multiplied by the probability of observing the data, that is

.. math::

  \pi (\rv\mid d)&=\frac{\pi (d\mid \rv)\,\pi (\rv)}{\pi(d)}\\
  &=\frac{\pi (d\mid \rv)\,\pi (\rv)}{\int_{\mathbb{R}^d} \pi (d\mid \rv)\,\pi (\rv)\,d\rv}

The density :math:`\pi (\rv\mid d)` is referred to as the posterior density.

Prior
^^^^^
To find the posterior density we must first quantify our prior belief of the possible values
of the parameter that can give the data. We do this by specifying the probability of 
observing the parameter independently of observing the data. 

Here we specify the prior distribution to be Normally distributed, e.g

.. math:: \pi\sim N(m_\text{prior},\Sigma_\text{prior})

Likelihood
^^^^^^^^^^
Next we must specify the likelihood :math:`\pi(d\mid \rv)` of observing the data given a realizations of the parameter :math:`\rv`
The likelihood answers the question: what is the distribution of the data assuming that :math:`\rv` are the exact parameters?

The form of the likelihood is derived from an assumed relationship between the model and the
data.

It is often assumed that

.. math :: d=\mathcal{M}(\rv)+\eta

where :math:`\eta\sim N(0,\Sigma_\text{noise})` is normally distributed noise with zero mean and covariance :math:`\Sigma_\text{noise}`.

In this case the likelihood is

.. math:: \pi(d|\rv)=\frac{1}{\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma_\text{noise} }}}|}\exp \left(-{\frac {1}{2}}(\mathcal{M}(\rv)-d)^{\mathrm {T} }{\boldsymbol {\Sigma_\text{noise} }}^{-1}(\mathcal{M}(\rv)-d)\right)

where :math:`|\Sigma_\text{noise}|=\det \Sigma_\text{noise}` is the determinant of :math:`\Sigma_\text{noise}`

Exact Linear-Gaussian Inference
-------------------------------
In the following we will generate data at a truth parameter :math:`\rv_\text{truth}` and use Bayesian inference
to estimate the probability of any model parameter :math:`\rv` conditioned on the observations we generated.
Firstly assume  :math:`\mathcal{M}` is a linear model, i.e.

.. math:: \mathcal{M}(\rv)=A\rv+b,

and as above assume that

.. math:: d=\mathcal{M}(\rv)+\eta

Now define the prior probability of the parameters to be

.. math:: \pi(\rv)\sim N(m_\text{prior},\Sigma_\text{prior})

Under these assumptions, the marginal density (integrating over the prior of :math:`\rv`) of the data and parameters is

.. math:: \pi(d)\sim N(m_\text{noise}+Am_\text{prior},\Sigma_\text{noise}+ A\Sigma_\text{prior} A^T)=N(m_\text{data},\Sigma_\text{data})

and the joint density of the parameters and data is

.. math:: \pi(\rv,d)\sim N(m_\text{joint},\Sigma_\text{joint})

where

.. math:: 

   \boldsymbol m_\text{joint}=\begin{bmatrix} \boldsymbol m_\text{prior} \\ \boldsymbol m_\text{data}\end{bmatrix},\quad
   \boldsymbol \Sigma_\text{joint}=\begin{bmatrix} \boldsymbol\Sigma_\text{prior} & \boldsymbol\Sigma_\text{prior,data} \\ \boldsymbol\Sigma_\text{prior,data} & \boldsymbol\Sigma_\text{data}\end{bmatrix}

and
 
.. math:: \Sigma_\text{prior,data}=A\Sigma_\text{prior}

is the covariance between the parameters and data.

Now let us setup this problem in Python
"""

from pyapprox.bayes.metropolis import MetropolisMCMCVariable
from pyapprox.bayes.tests.test_metropolis import (
    ExponentialQuarticLogLikelihoodModel,
)
from pyapprox.variables.gaussian import DenseMatrixMultivariateGaussian
from scipy import stats
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import matplotlib as mpl

np.random.seed(1)

A = np.array([[0.5]])
b = 0.0
# define the prior
prior_covariance = np.atleast_2d(1.0)
prior = DenseMatrixMultivariateGaussian(np.atleast_2d(0.0), prior_covariance)
# define the noise
noise_covariance = np.atleast_2d(0.1)
noise = DenseMatrixMultivariateGaussian(np.atleast_2d(0.0), noise_covariance)
# compute the covariance between the prior and data
C_12 = A @ prior_covariance
# define the data marginal distribution
data_covariance = noise_covariance + C_12 @ A.T
data = DenseMatrixMultivariateGaussian(A @ prior.mean() + b, data_covariance)
# define the covariance of the joint distribution of the prior and data


def form_normal_joint_covariance(C_11, C_22, C_12):
    nvars1 = C_11.shape[0]
    nvars2 = C_22.shape[0]
    joint_covariance = np.empty((nvars1 + nvars2, nvars1 + nvars2))
    joint_covariance[:nvars1, :nvars1] = C_11
    joint_covariance[nvars1:, nvars1:] = C_22
    joint_covariance[:nvars1, nvars1:] = C_12
    joint_covariance[nvars1:, :nvars1] = C_12
    return joint_covariance


joint_mean = np.vstack((prior.mean(), data.mean()))
joint_covariance = form_normal_joint_covariance(
    prior_covariance, data_covariance, C_12
)
joint = DenseMatrixMultivariateGaussian(joint_mean, joint_covariance)

# %%
# Now we can plot the joint distribution and some samples from that distribution
# and print the sample covariance of the joint distribution

nsamples = 10000
theta_samples = prior.rvs(nsamples)
noise_samples = noise.rvs(nsamples)
data_samples = A @ theta_samples + b + noise_samples
plot_limits = [
    theta_samples.min(),
    theta_samples.max(),
    data_samples.min(),
    data_samples.max(),
]
joint.plot_pdf(
    joint.get_plot_axis()[1],
    plot_limits=plot_limits,
    cmap=mpl.cm.coolwarm,
    levels=20,
)
_ = plt.plot(theta_samples[0, :100], data_samples[0, :100], "o")

# %%
# Conditional probability of multivariate Gaussians
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For multivariate Gaussians the dsitribution of :math:`x` conditional on observing the data :math:`d^\star=d_\text{truth}+\eta^\star`, :math:`\pi(x\mid d=d^\star)\sim N(m_\text{post},\Sigma_\text{post})` is a multivariate Gaussian with mean and covariance
#
# .. math::
#
#  \boldsymbol{m}_\text{post}&=\boldsymbol m_\text{prior} + \boldsymbol\Sigma_\text{prior,data} \boldsymbol\Sigma_{data}^{-1}\left( \mathbf{d}^\star - \boldsymbol m_\text{data}\right),\\
#  \boldsymbol \Sigma_\text{post}&=\boldsymbol \Sigma_\text{prior}-\boldsymbol \Sigma_\text{prior,data}\boldsymbol\Sigma_\text{data}^{-1}\boldsymbol\Sigma_\text{data,prior}^T.
#
# where :math:`\eta^\star` is a random sample from the noise variable :math:`\eta`. In the case of one parameter and one QoI we have
#
# .. math:: \pi(x\mid d=d^\star) \sim\ N\left(m_\text{prior}+\frac{\sigma_\text{prior}}{\sigma_\text{data}}\rho( d^\star - m_\text{data}),\, (1-\rho^2)\sigma_\text{prior}^2\right).
#
# where the correlation coefficient between the parameter and data is
#
# .. math: \rho=\frac{\Sigma_\text{prior,data}}{\sigma_\text{prior}\sigma_\text{data}}
#
# Lets use this formula to update the prior when one data :math:`d^\star` becomes available.


def condition_normal_on_data(mean, covariance, fixed_indices, values):
    indices = set(np.arange(mean.shape[0]))
    ind_remain = list(indices.difference(set(fixed_indices)))
    new_mean = mean[ind_remain]
    diff = values - mean[fixed_indices]
    sigma_11 = np.array(covariance[ind_remain, ind_remain], ndmin=2)
    sigma_12 = np.array(covariance[ind_remain, fixed_indices], ndmin=2)
    sigma_22 = np.array(covariance[fixed_indices, fixed_indices], ndmin=2)
    update = (sigma_12 @ np.linalg.solve(sigma_22, diff)).flatten()
    new_mean = new_mean + update
    new_cov = sigma_11 - (sigma_12 @ np.linalg.solve(sigma_22, sigma_12.T))
    return new_mean, new_cov


x_truth = np.atleast_2d(0.2)
data_obs = A @ x_truth + b + noise.rvs(1)

new_mean, new_cov = condition_normal_on_data(
    joint_mean, joint_covariance, [1], data_obs
)
posterior = DenseMatrixMultivariateGaussian(new_mean, new_cov)

# %%
# Now lets plot the prior and posterior of the parameters as well as the joint distribution and the data.

f, axs = plt.subplots(1, 2, figsize=(16, 6))
prior_plot_limits = [-3, 3]
prior.plot_pdf(axs[0], prior_plot_limits, label="prior")
axs[0].axvline(x=x_truth, lw=3, label=r"$x_\text{truth}$")
posterior.plot_pdf(
    plot_limits=prior_plot_limits, ls="-", label="posterior", ax=axs[0]
)
print(joint)
joint.plot_pdf(axs[1], [-3, 3, -3, 3], cmap=mpl.cm.coolwarm, levels=20)
axhline = axs[1].axhline(y=data_obs, color="k")
axplot = axs[1].plot(x_truth, data_obs, "ok", ms=10)

# %%
# Lets also plot the joint distribution and marginals in a 3d plot


def data_obs_limit_state(samples, vals, data_obs):
    idx = np.where(samples[1, :] <= data_obs[0, 0])[0]
    return idx, 0.0


from pyapprox.util.visualization import _turn_off_3d_axes

npts_1d = 100
limit_state = partial(data_obs_limit_state, data_obs=data_obs)
ax = joint.get_plot_axis(surface=True)[1]
joint.plot_pdf(
    ax,
    [-3, 3, -3, 3],
    npts_1d=101,
    cmap=mpl.cm.coolwarm,
    zorder=3,
    alpha=0.5,
)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
_turn_off_3d_axes(ax)
offset = -0.5
samples = np.hstack([[x_truth, data_obs, np.atleast_2d(offset)]]).T
joint.plot_pdf(
    ax,
    [-3, 3, -3, 3],
    offset=offset,
    zorder=-1,
    cmap=mpl.cm.coolwarm,
    levels=30,
    zdir="z",
)
X, Y, pts = joint.meshgrid_samples([-3, 3, -3, 3])
Z_prior = prior.pdf(pts[0]).reshape(X.shape)
ax.contour(X, Y, Z_prior, zdir="y", offset=Y.max(), cmap=mpl.cm.coolwarm)
Z_data = data.pdf(pts[1]).reshape(Y.shape)
ax.contour(X, Y, Z_data, zdir="x", offset=X.min(), cmap=mpl.cm.coolwarm)
x = np.linspace(X.min(), X.max(), npts_1d)
y = np.full((npts_1d,), data_obs[0, 0])
z = np.full((npts_1d,), offset)
ax.plot(x, y, z, zorder=100, color="k")
_ = ax.plot(
    [x_truth], [data_obs[0, 0]], [offset], zorder=100, color="k", marker="o"
)

# %%
# Now lets assume another piece of observational data becomes available we can use the posterior as a new prior.

nobs = 5
posteriors = [None] * nobs
posteriors[0] = posterior
for ii in range(1, nobs):
    new_prior = posteriors[ii - 1]
    data_obs = A @ x_truth + b + noise.rvs(1)
    C_12 = A @ new_prior.covariance()
    new_joint_covariance = form_normal_joint_covariance(
        new_prior.covariance(), data.covariance(), C_12
    )
    new_joint = DenseMatrixMultivariateGaussian(
        np.vstack((new_prior.mean(), data.mean())), new_joint_covariance
    )
    new_mean, new_cov = condition_normal_on_data(
        new_joint.mean(),
        new_joint.covariance(),
        np.arange(new_prior.nvars(), new_prior.nvars() + data.nvars()),
        data_obs,
    )
    posteriors[ii] = DenseMatrixMultivariateGaussian(new_mean, new_cov)

# %%
# And now lets again plot the joint density before the last data was added and final posterior and the intermediate priors.

f, axs = plt.subplots(1, 2, figsize=(16, 6))
prior.plot_pdf(axs[0], plot_limits=prior_plot_limits, label="prior")
axs[0].axvline(x=x_truth, lw=3, label=r"$x_\text{truth}$")
for ii in range(nobs):
    posteriors[ii].plot_pdf(
        plot_limits=prior_plot_limits, ls="-", label="posterior", ax=axs[0]
    )

new_joint.plot_pdf(ax=axs[1], plot_limits=[-3, 3, -3, 3], levels=20)
axhline = axs[1].axhline(y=data_obs, color="k")
axplot = axs[1].plot(x_truth, data_obs, "ok", ms=10)

# %%
# As you can see the variance of the joint density decreases as more data is added. The posterior variance also decreases and the posterior will converge to a Dirac-delta function as the number of observations tends to infinity. Currently the mean of the posterior is not near the true parameter value (the horizontal line). Try increasing ``nobs1`` to see what happens.

# %%
# Inexact Inference using Markov Chain Monte Carlo
# ------------------------------------------------
#
# When using non-linear or non-Gaussian priors, a functional representation of the posterior distribution :math:`\pi_\text{post}` cannot be computed analytically. Instead the the posterior is characterized by samples drawn from the posterior using Markov-chain Monte Carlo (MCMC) sampling methods.
#
# Lets consider non-linear model with two uncertain parameters with independent uniform priors on [-2,2] and the negative log likelihood function
#
# .. math:: -\log\left(\pi(d\mid\rv)\right)=\frac{1}{10}\rv_1^4 + \frac{1}{2}(2\rv_2-\rv_1^2)^2
#
# We can sample the posterior using Delayed Rejection Adaptive Metropolis (DRAM) Markov Chain Monte Carlo using the following code.
np.random.seed(1)

from pyapprox.variables.joint import IndependentMarginalsVariable

prior_marginals = [stats.uniform(-2, 4), stats.uniform(-2, 4)]
plot_range = np.asarray([-1, 1, -1, 1]) * 2
prior = IndependentMarginalsVariable(prior_marginals)

loglike = ExponentialQuarticLogLikelihoodModel()
mcmc_variable = MetropolisMCMCVariable(prior, loglike)

# number of draws from the distribution
ndraws = 500
# number of "burn-in points" (which we'll discard) as a fraction of ndraws
burn_fraction = 0.1
map_sample = mcmc_variable.maximum_aposteriori_point()
samples = mcmc_variable.rvs(ndraws)

print("MAP sample", map_sample.squeeze())

# %%
# Lets plot the posterior distribution and the MCMC samples. First we must compute the evidence

from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.surrogates.bases.basis import (
    setup_tensor_product_gauss_quadrature_rule,
)
from pyapprox.bayes.likelihood import LogUnNormalizedPosterior

log_unnormalized_posterior = LogUnNormalizedPosterior(loglike, prior)


marginals = [stats.uniform(-2, 4)] * 2
quad_rule = setup_tensor_product_gauss_quadrature_rule(
    IndependentMarginalsVariable(marginals)
)
xquad, wquad = quad_rule([100, 100])
wquad *= 2.0  # adjust for Lebsesque integration
print(xquad.shape)
evidence = np.exp(log_unnormalized_posterior(xquad))[:, 0] @ wquad[:, 0]
print("evidence", evidence)

from pyapprox.util.visualization import get_meshgrid_function_data

plt.figure()
X, Y, Z = get_meshgrid_function_data(
    lambda x: np.exp(log_unnormalized_posterior(x)) / evidence, plot_range, 50
)
plt.contourf(
    X,
    Y,
    Z,
    levels=np.linspace(Z.min(), Z.max(), 30),
    cmap=matplotlib.cm.coolwarm,
)
_ = plt.plot(samples[0, :], samples[1, :], "ko")

# %%
# Now lets compute the mean of the posterior using a highly accurate quadrature rule and compars this to the mean estimated using MCMC samples.

exact_mean = (xquad * np.exp(log_unnormalized_posterior(xquad))[:, 0]).dot(
    wquad
) / evidence
print("mcmc mean", samples.mean(axis=1))
print("exact mean", exact_mean.squeeze())

# %%
# References
# ^^^^^^^^^^
# .. [KAIPO2005] `J. Kaipio and E. Somersalo. Statistical and Computational Inverse Problems. 2005 <https://link.springer.com/book/10.1007/b138659>`_
