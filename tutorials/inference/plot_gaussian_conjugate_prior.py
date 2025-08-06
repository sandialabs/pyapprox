r"""
Bayesian Inference with Gaussian Conjugate Priors and Gaussian Observations
===========================================================================

Goal
----
This tutorial demonstrates how to perform Bayesian inference using a Gaussian conjugate prior and Gaussian observations. Specifically, we will use Bayesian inference to estimate the mean of a multivariate Gaussian distribution with a known covariance. For example, we can estimate the mean age and density, and porosity of rocks in a region using :math:`n`-experiments of the vector containing the portions of three types of rocks in the regions, when our measurement device introduces an additive mean-zero Gaussian noise, which is independent on past and future observations.

Prerequisites
-------------
Understanding of Bayesian Concepts: Familiarity with concepts like prior, posterior, likelihood, and evidence is essential.

Define the prior
----------------
First, we encapulate prior knowledge of the values of the unknown vector-valued mean :math:`\vec{\nu}\in\reals^{k}` by representing it as a linear function of :math:`d=2` the portions of two rock types, such that

.. math:: \vec{\nu}=\mat{A}\rvv,

where :math:`\mat{A}\in\reals^{k\times d}` is a known matrix.
We then place a prior on the three traits

.. math:: p(\rvv) = f(\rvv) = \frac{1}{\sqrt{(2\pi)^d |\mat{\Sigma}|}} \exp\left(-\frac{1}{2} (\rvv - \vec{\mu})^\top \mat{\Sigma}^{-1} (\rvv - \vec{\mu})\right),

where :math:`\vec{\mu}` is the mean of the prior and :math:`\mat{\Sigma}` is its covariance.
Note the prior mean is different to the mean :math:`\vec{\nu}` we want to estimate.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.variables.gaussian import IndependentMultivariateGaussian
from pyapprox.bayes.likelihood import (
    ModelBasedIndependentGaussianLogLikelihood,
)
from pyapprox.bayes.laplace import DenseMatrixLaplacePosteriorApproximation

# Set the seed for reproducibility
np.random.seed(1)

# Define the prior shape parameters
nvars = 2
prior_mean = bkd.zeros((nvars, 1))
prior_covariance_diag = bkd.ones((nvars))

# Define prior distribution using independent Beta marginals
prior = IndependentMultivariateGaussian(prior_mean, prior_covariance_diag)


# %%
# Define the likelihood function
# ------------------------------
# We assume that we are given :math:`n` experiments :math:`\vec{y}_i=\mat{A}\rvv+\vec{\epsilon}_i\in\reals^{k}, i=1,\ldots,n` of a :math:`k`-dimensional vector containing the rock age, density, and porosity corrupted by additive mean-zero Gausian noise :math:`\vec{\epsilon}_i\sim\mathcal{N}(\vec{0}_{k},\Gamma)`. This leads to the likelihood
#
# .. math:: p(\obsv_1,\ldots,\obsv_n\mid \rvv) = \prod_{i=1}^n \frac{1}{\sqrt{(2\pi)^k |\mat{\Gamma}|}} \exp\left(-\frac{1}{2} (\obsv_i - \mat{A}\rvv)^\top \mat{\Gamma}^{-1} (\obsv_i -\mat{A}\rvv)\right),
#
# where :math:`\mat{\Gamma}` is the covariance of the noise, which for this example we assume is a diagonal matrix, with the same value along the diagonal, e.g. the noise is independent and homoscedastic. Here, :math:`\mat{A}\rvv=\mathbb{E}[\mat{A}\rvv+\vec{\epsilon}]=\mathbb{E}[\obsv]` is the unknown mean of the data which we are estimating (well we are estimating the parameters of the mean).

# Specify the number of experiments n
nexperiments = 3
# Specify the dimension of each observation k
nobs = 3
# Define the noise covariance
noise_cov_diag = bkd.full((nobs, 1), 0.1)
# Define the model that maps the model parameters to the observations,
# e.g. the traits to weight and height.
# We will just randomly generate the A matrix for convenience
obs_mat = bkd.array(np.random.uniform(-1, 1, (nobs, nvars)))
model = DenseMatrixLinearModel(obs_mat, backend=bkd)
# Setup the loglikelihood function
loglike = ModelBasedIndependentGaussianLogLikelihood(model, noise_cov_diag)
true_samples = prior.rvs(1)
obs = loglike.rvs(true_samples)
loglike.set_observations(obs)

# %%
# Speed Up Evaluation of the Log Likelihood
# -----------------------------------------
# The Gaussian log likelihood is often used in many applications, however computing it naively for many samples of the parameters :math:`\rVv=[\rvv^{(1)},\ldots,\rvv^{(s)}]\in\reals^{d\times s}` can be inefficient. We can compute the log posterior efficiently by first observing that the trace of a scalar is itself so we can write
#
# .. math::
#     \begin{align}
#     \log p(\obsv_1,\ldots,\obsv_n\mid \rvv) &\propto \sum_{j=1}^s\sum_{i=1}^n \left(-\frac{1}{2} (\obsv_i - \mat{A}\rvv^{(s)})^\top \mat{\Gamma}^{-1} (\obsv_i^{(s)} -\mat{A}\rvv^{(s)})\right)\\
#     &=-\frac{1}{2} \sum_{i=1}^n\text{Trace}\left[(\obsv_i - \mat{A}\rVv)^\top \mat{\Gamma}^{-1} (\obsv_i -\mat{A}\rVv)\right]
#     \end{align}
#
# Then we use the identity
#
# .. math:: \text{Trace}(\mat{A} \mat{B}^\top) = \sum_i \sum_j (\mat{A}\circ \mat{B})_{ij}
#
# for matrices :math:`\mat{A}\in\reals^{m\times n}`, :math:`\mat{B}\in\reals^{m\times n}`
# to write
#
# .. math::
#     \begin{align}
#     \log p(\obsv_1,\ldots,\obsv_n\mid \rvv)
#     &\propto -\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^k \sum_{l=1}^s\left((\obsv_i - \mat{A}\rVv)^\top \circ \left(\mat{\Gamma}^{-1} (\obsv_i -\mat{A}\rVv)\right)^\top\right)_{jl}
#     \end{align}
#
# Compute the Posterior Using Conjugacy
# -------------------------------------
# The Gaussian distribution is the conjugate prior for the liklihood. This is true even when using a linear transformation of the uncertain variables :math:`\rvv` because a linear transformation of a Gaussian variable is still a Gaussian variable.
#
# To compute the posterior distribution we first compute observe that the negative log PDF of the joint density of the prior and likelihood can be written as
#
# .. math::
#     -\log p(\obsv,\rvv) \propto
#     \begin{bmatrix}(\vec{\obsv}-\vec{\nu})^\top &(\rvv-\vec{\mu})^\top\end{bmatrix}
#     \begin{bmatrix} \mat{\Sigma}_{yy}& \mat{\Sigma}_{\obs\rv}\\ \mat{\Sigma}_{\obs\rv}^\top &\mat{\Sigma}\end{bmatrix}^{-1}
#     \begin{bmatrix}\obsv-\vec{\nu} \\ \rvv-\vec{\mu}\end{bmatrix}
#
# Here :math:`\vec{\nu}=\mathbb{E}[\obsv]=\mat{A}\mathbb{E}[\rvv]=\mat{A}\vec{\mu}` is the mean of the data, :math:`\mat{\Sigma}_{\obs\obs}=\mathbb{C}\text{ov}[\obsv,\obsv]=\mathbb{C}\text{ov}[\mat{A}\vec{z}+\vec{\epsilon}, \mat{A}\vec{z}+\vec{\epsilon}]=\mat{A}\Sigma\mat{A}^\top+\mat{\Gamma}` is the covariance of the data, and :math:`\mat{\Sigma}_{\obs\rv}=\mathbb{C}\text{ov}[\obsv,\rvv]=\mat{A}\mat{\Sigma}` is the covariance between the data :math:`\obsv` and the random variables :math:`\rvv`.
#
# We can then condition the joint density on the observations using the identity for conditioning multivariate Gaussians derived in another tutorial. Using this identity lets us write the mean and variance of the Gaussian posterior
#
# .. math::
#     \begin{align}
#     \vec{\mu}^\star&=\vec{\mu} + \mat{\Sigma}\mat{A}^\top\left(\mat{A}\mat{\Sigma}\mat{A}^\top+\mat{\Gamma}\right)^{-1}(\vec{y}-\mat{A}\vec{\mu}) =\mat{\Sigma}^\star\left(\mat{A}^\top\mat{\Gamma}^{-1}\vec{y}+\mat{\Sigma}^{-1}\vec{\mu}\right)=\vec{\mu}+\mat{\Sigma}^\star\mat{A}^\top\mat{\Gamma}^{-1}(\vec{y}-\mat{A}\vec{\mu}),\\
#     \mat{\Sigma}^\star&=\mat{\Sigma}-\mat{\Sigma}\mat{A}^\top\left(\mat{A}\mat{\Sigma}\mat{A}^\top+\mat{\Gamma}\right)^{-1}\mat{A}\mat{\Sigma}=\left(\mat{\Sigma}^{-1}+\mat{A}^\top\mat{\Gamma}^{-1}\mat{A}\right)^{-1}.
#     \end{align}
#
# The first equalities for the mean and covariance follow from the Woodbury Identity.
# The last equality for the mean follows from observing
#
# .. math:: \mat{\Sigma}^\star=\left(\mat{\Sigma}^{-1}+\mat{A}^\top\mat{\Gamma}^{-1}\mat{A}\right)^{-1}
#
# So
#
# .. math::
#     \begin{align}
#     \mat{I} &=\mat{\Sigma}^\star\left(\mat{\Sigma}^{-1}+\mat{A}^\top\mat{\Gamma}^{-1}\mat{A}\right)\\
#     &=\mat{\Sigma}^\star\mat{\Sigma}^{-1}+\mat{\Sigma}^\star\mat{A}^\top\mat{\Gamma}^{-1}\mat{A}
#     \end{align}
#
# So
#
# .. math:: \mat{\Sigma}^\star\mat{\Sigma}^{-1} = \mat{I}-\mat{\Sigma}^\star\mat{A}^\top\mat{\Gamma}^{-1}\mat{A}
#
# So
#
# .. math:: \mat{\Sigma}^\star\mat{\Sigma}^{-1}\vec{\mu}=\left(\mat{I}-\mat{\Sigma}^\star\mat{A}^\top\mat{\Gamma}^{-1}\mat{A}\right)\vec{\mu}
#
#
# Finally, add :math:`\mat{A}^\top\mat{\Gamma}^{-1}\vec{y}` to both sides
#
# .. math:: \mat{\Sigma}^\star\mat{\Sigma}^{-1}\vec{\mu} + \mat{A}^\top\mat{\Gamma}^{-1}\vec{y} = \left(\mat{I}-\mat{\Sigma}^\star\mat{A}^\top\mat{\Gamma}^{-1}\mat{A}\right)\vec{\mu} + \mat{A}^\top\mat{\Gamma}^{-1}\vec{y}
#
# The following code computes the posterior and compares it with the prior.

# Compute the posterior using the anlaytical formula
gaussian_post = DenseMatrixLaplacePosteriorApproximation(
    obs_mat,
    prior.mean(),
    prior.covariance(),
    bkd.diag(noise_cov_diag[:, 0]),
    backend=bkd,
)
gaussian_post.compute(obs)
posterior = gaussian_post.posterior_variable()

# Plot the prior PDF
axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
axs[0].set_title("Prior PDF")
prior.plot_pdf(axs[0], [-3, 3, -3, 3], levels=31, cmap="coolwarm")

# Plot the posterior distribution
axs[1].set_title("Posterior PDF")
posterior.plot_pdf(axs[1], [-3, 3, -3, 3], levels=31, cmap="coolwarm")
# Plot the sample of the model parameters that generated the observations
axs[1].plot(*true_samples, "ko", ms=20)
plt.show()

# %%
# Verify the Posterior Numerically
# --------------------------------
# We can verify the closed-form expression from the prior with Monte Carlo (MC) integration. The evidence (marginal likelihood) is computed as the mean of the likelihood over prior samples
#
# .. math:: \int_{[0,1]}p(\obs_1,\ldots,\obs_n\mid \rv)p(\rv)\, d\rv \approx \sum_{j=1}^M p(\obs_1,\ldots,\obs_n\mid \rv^{(i)})
#
# where the samples :math:`\rv^{(i)}` are drawn randomly from the prior.

# Draw samples from the prior distribution.
prior_samples = prior.rvs(100000)
evidence = bkd.exp(loglike(prior_samples)).mean()
print("Exact evidence", gaussian_post.evidence())
print("MC evidence", evidence)

# %%
# Note that the MC estimate of the evidence is not exact but its accuracy can be improved by increasing the number of samples from the prior it uses.
#
# We can also numerically check the mean and variance of the posterior.
# The posterior mean and variance are computed using the prior samples and likelihood.

mean = (prior_samples * bkd.exp(loglike(prior_samples))[:, 0] / evidence).mean(
    axis=1
)

print("Exact posterior mean", posterior.mean()[:, 0])
print("MC posterior mean", mean)

covariance = (
    (prior_samples * bkd.exp(loglike(prior_samples))[:, 0]) @ prior_samples.T
) / (prior_samples.shape[1] * evidence) - mean[:, None] ** 2
print("Exact posterior covariance", posterior.covariance())
print("MC posterior covariance", covariance)


# %%
# Compute the Posterior Using Repeated Observations
# -------------------------------------------------
# The expressions for the mean and covariance of the Gaussian posterior we introduced in this tutorial are general and can be used for a single observation vector or :math:`n` independent repititions of the same :math:`k`-dimensional observations vector (and hybrid cases). In the latter case we can make some simplifications. Specifically, letting :math:`\mat{A}_k\in\reals^{k\times d}` be the observation matrix for a single experiment and :math:`\mat{\Gamma}_k\in\reals^{k\times k}` be its covariance, then
#
# .. math::
#     \begin{align}\mat{A}=
#     \begin{bmatrix}
#     \mat{A}_k\\
#     \vdots\\
#     \mat{A}_k
#     \end{bmatrix}\in\reals^{nk\times d} &&
#     \mat{\Gamma}=
#     \begin{bmatrix}
#     \mat{\Gamma}_k & \mat{0}_{k\times k} & \cdots & \mat{0}_{k\times k}\\
#     \mat{0}_{k\times k} & \mat{\Gamma}_k & \cdots & \mat{0}_{k\times k}\\
#     \vdots & \vdots & \ddots & \vdots \\
#     \mat{0}_{k\times k} & \cdots & \mat{0}_{k\times k} & \mat{\Gamma}_k
#     \end{bmatrix}\in\reals^{nk\times nk}
#     \end{align}
#
# Plugging these into the general expression for the posterior covariance yields
#
# .. math::
#     \begin{align}
#     \left(\mat{\Sigma}^{-1}+\mat{A}^\top\mat{\Gamma}^{-1}\mat{A}\right)^{-1}&=\left(\mat{\Sigma}^{-1}+\begin{bmatrix}\mat{A_k}^\top&\cdots & \mat{A_k}^\top\end{bmatrix}
#     \begin{bmatrix}
#     \mat{\Gamma}_k & \mat{0}_{k\times k} & \cdots & \mat{0}_{k\times k}\\
#     \mat{0}_{k\times k} & \mat{\Gamma}_k & \cdots & \mat{0}_{k\times k}\\
#     \vdots & \vdots & \ddots & \vdots \\
#     \mat{0}_{k\times k} & \cdots & \mat{0}_{k\times k} & \mat{\Gamma}_k
#     \end{bmatrix}^{-1}\begin{bmatrix}\mat{A}_k\\\vdots\\\mat{A}_k\end{bmatrix}\right)^{-1}\\
#     &=\left(\mat{\Sigma}^{-1}+n\mat{A}_k^\top\mat{\Gamma}_k\mat{A}_k\right)^{-1}
#     \end{align}
#
# Similarly for the posterior mean
#
# .. math::
#     \begin{align}
#     \vec{\mu}^\star&=\mat{\Sigma}^\star\left(\mat{A}^\top\mat{\Gamma}^{-1}\vec{y}+\mat{\Sigma}^{-1}\vec{\mu}\right)\\
#     &=\mat{\Sigma}^\star\left(\begin{bmatrix}\mat{A_k}^\top&\cdots & \mat{A_k}^\top\end{bmatrix}
#     \begin{bmatrix}
#     \mat{\Gamma}_k & \mat{0}_{k\times k} & \cdots & \mat{0}_{k\times k}\\
#     \mat{0}_{k\times k} & \mat{\Gamma}_k & \cdots & \mat{0}_{k\times k}\\
#     \vdots & \vdots & \ddots & \vdots \\
#     \mat{0}_{k\times k} & \cdots & \mat{0}_{k\times k} & \mat{\Gamma}_k
#     \end{bmatrix}^{-1}\begin{bmatrix}\obsv_k^{(1)}\\\vdots\\\obsv_k^{(n)}\end{bmatrix}+\mat{\Sigma}^{-1}\vec{\mu}\right)\\
#     &=\mat{\Sigma}^\star\left(n\mat{A}_k^\top\mat{\Gamma}_k^{-1}\frac{1}{n}\sum_{i=1}^n\vec{y}_k^{(i)}+\mat{\Sigma}^{-1}\vec{\mu}\right)
#     \end{align}
#
# where :math:`\obsv_k^{(i)}, i=1\ldots,n` is the ith realization of the data from the ith experiment.
#
# Summary
# -------
# This tutorial demonstrates how to perform Bayesian inference using a multivariate Gaussian conjugate prior and Gaussian observations. This tutorial is useful for developing unit tests for variational inference because we can specify the parameterized class of posteriors that includes the true posterior.
