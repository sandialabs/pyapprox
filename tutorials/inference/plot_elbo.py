r"""
Evidence Lower Bound
====================

Want to find a distribution in a family of distributions that is close to the posterior :math:`p(z\mid x)`

.. math:: q(z) = \text{arg} \min_{q(z)\in\mathcal{Q}} D_{\text{KL}}[q(z)\lVert p(z\mid x)]

.. math::

    D_{\text{KL}}[q(z) \lVert p(z | x)] &= \int q(z) \log \frac{q(z)}{p(z\mid x)} \;\mathrm{d}z\\
    &= \int q(z) \log q(z) - q(z) \log {p(z\mid x)} \;\mathrm{d}z\\
    & = \mathbb{E}_{q(z)}\left[ \log q(z)\right] - \mathbb{E}_{q(z)}\left[\log {p(z \mid x)} \right]\\
    & = \mathbb{E}_{q(z)}\left[ \log q(z)\right] - \mathbb{E}_{q(z)}\left[\log \frac{p(x, z)}{p(x)} \right]\\
    & = \mathbb{E}_{q(z)}\left[ \log q(z)\right] - \mathbb{E}_{q(z)}\left[\log p(x, z)-\log p(x) \right]\\
    &= \underbrace{\mathbb{E}_{q(z)}[ \log q(z)] - \mathbb{E}_{q(z)}[\log p(z, x)]}_{-\text{ELBO}(q)} + \log p(x).

Thus

.. math:: \mathcal{L}^A(q) = \text{ELBO}(q) =\mathbb{E}_{q(z)}[\log p(z, x)]-\mathbb{E}_{q(z)}[ \log q(z)]

and

.. math:: \log p(x) = \text{ELBO}(q) + D_{\text{KL}}[q(z) \lVert p(z | x)]

So ELBO stands for evidence lower bound because it bounds the evidence from below, i.e.

.. math::  \text{ELBO}(q) \ge \log p(x)


Jensen's inequality can also be used to derive the ELBO. Jensen's inequality states for any random variable X

.. math:: \log \mathbb{E}[X] \ge \mathbb{E}[\log X]

Thus

.. math::

   \log p(x) &= \log \int p(z, x) \;\mathrm{d}z\\
             & = \log \int p(z, x) \frac{q(z)}{q(z)} \;\mathrm{d}z\\
             & = \log \mathbb{E}_{q(z)}\left[ \frac{p(z, x)}{q(z)} \right]\\
             & \ge \underbrace{\mathbb{E}_{q(z)} \log \left[\frac{p(z, x)}{q(z)}\right]}_{\text{ELBO}(q)}


Kingma actually presents two estimators, which he denotes :math:`\mathcal{L}^A` and :math:`\mathcal{L}^B`. We have presnted the former, however often the ELBO is presented in a different form by performing the following manipulations

.. math::

    \mathcal{L}^B(q) = \text{ELBO}(q) &=  \mathbb{E}_{q(z)}[\log p(z, x)]-\mathbb{E}_{q(z)}[ \log q(z)]\\
            &=  \mathbb{E}_{q(z)}[\log p(x\mid z)p(z)]-\mathbb{E}_{q(z)}[ \log q(z)]\\
            &=  \mathbb{E}_{q(z)}[\log p(x\mid z)]+\mathbb{E}_{q(z)}[\log p(z) - \log q(z)]\\
            &=  \mathbb{E}_{q(z)}[\log p(x\mid z)]+\int {q(z)}\log \frac{p(z)}{q(z)}\;\mathrm{d}z\\
            &=  \mathbb{E}_{q(z)}[\log p(x\mid z)]-D_{\text{KL}}[q(z) \lVert p(z)]

Here we used

.. math:: D_{\text{KL}}[q(z) \lVert p(z)] = \int {q(z)}[\log \frac{q(z)}{p(z)}\;\mathrm{d}z = -\int {q(z)}\log \frac{p(z)}{q(z)}\;\mathrm{d}z\\


Reparameterization Trick
------------------------
The reparameterization trick is used to we can compute the gradient of the ELBO with respect to :math:`\theta`

Let

.. math::
    \epsilon &\sim p(\epsilon)\\
    z &= g_\theta(\epsilon, x)

Then letting :math:`x^{(i)}` denote the ith sample of the data and  :math:`z^{(i)}` denote the ith sample of the random variable 

.. math::
    \mathbb{E}_{p_{\theta}(z)}[f(z^{(i)})] = \mathbb{E}_{p(\epsilon)} [f(g_{\theta}(\epsilon, x^{(i)}))]

The gradient of a scalar function :math:`f`, is

.. math::

    \nabla_{\theta} \mathbb{E}_{p_{\theta}(z)}[f(z^{(i)})] &= \nabla_{\theta} \mathbb{E}_{p(\epsilon)} [f(g_{\theta}(\epsilon, x^{(i)}))]\\
    &= \mathbb{E}_{p(\epsilon)} [\nabla_{\theta} f(g_{\theta}(\epsilon, x^{(i)}))]\\
    &\approx \frac{1}{L} \sum_{l=1}^{L} \nabla_{\theta} f(g_{\theta}(\epsilon^{(l)}, x^{(i)}))

Thus

.. math::

    \nabla_{\theta, \phi} \mathcal{L}^B = \nabla_{\theta, \phi} \underbrace{\left[ \frac{1}{L} \sum_{l=1}^{L} \left( \log p_{\boldsymbol{\theta}}(x^{(i)} \mid z^{(l)}) \right)\right]}_{\text{Estimate with Monte Carlo}} - \nabla_{\theta, \phi} \underbrace{\left[\text{KL}[q_{\phi}(z) \lVert p_{\theta}(z)]\right]}_{\text{Analytically evaluate}}

Setup the Problem
-----------------
"""

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.backends.torch import TorchMixin as bkd
from pyapprox.variables.marginals import BetaMarginal
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    update_pdf_contourf_plots,
)
from pyapprox.bayes.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.interface.model import (
    DenseMatrixLinearModel,
    ModelFromSingleSampleCallable,
)
from pyapprox.surrogates.affine.basis import TensorProductQuadratureRule
from pyapprox.surrogates.univariate.orthopoly import GaussQuadratureRule
from pyapprox.bayes.variational.elbo import (
    IndependentBetaVariationalPosterior,
    VariationalInverseProblem,
)

# Set the seed for reproducibility
np.random.seed(2025)

# Define the number of observations and latent variables
nobs = 2
nvars = 2

# Define the noise covariance matrix
noise_std = 0.2
noise_cov = bkd.eye(nobs) * noise_std**2

# Define the observational model
obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))
# obs_mat = bkd.diag(bkd.asarray(np.random.normal(0.0, 1.0, (nobs,))))
obs_model = DenseMatrixLinearModel(obs_mat, backend=bkd)

# Define the loglikelihood
loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)

# Generate data for an artificial truth
true_sample = bkd.asarray(np.random.uniform(0, 1, (nvars, 1)))
obs = loglike.rvs(true_sample)
loglike.set_observations(obs)

# %%
# Define the Priors
# -----------------

# Define the uniformative Uniform priors
a1, b1 = 1, 1
a2, b2 = 1, 1
bounds = [0, 1]

# Create the prior object
prior = IndependentMarginalsVariable(
    [
        BetaMarginal(a1, b1, *bounds, backend=bkd),
        BetaMarginal(a2, b2, *bounds, backend=bkd),
    ],
    backend=bkd,
)

# %%
# Define the Variational Posterior
# --------------------------------

# define the number of latent samples used to compute loss during training
nlatent_samples = 1000  # 1000


# Specify the initial shape parameters passed to fit
# Make them different to prior shape parameters
ashapes = [prior.marginals()[i]._a + 1.0 for i in range(nvars)]
bshapes = [prior.marginals()[i]._b + 1.0 for i in range(nvars)]

# Define the variational posterior
variational_posterior = IndependentBetaVariationalPosterior(
    prior,
    nlatent_samples,
    ashapes,
    bshapes,
    prior.interval(1),
    ashape_bounds=(1, 100),
    bshape_bounds=(1, 100),
    backend=bkd,
)


# %%
# Fit the Variational Posterior
# -----------------------------

# Define the variational problem
vi = VariationalInverseProblem(prior, loglike, variational_posterior)
vi.set_optimizer(
    vi.default_optimizer(verbosity=2, local_method="trust-constr")
)
iterate = vi._neg_elbo.hyp_list().get_active_opt_params()[:, None]
# iterate = bkd.array(
#     [
#         2.4410330600690187,
#         2.3931614971069486,
#         1.7506675102073170,
#         1.7021390304400792,
#     ]
# )[:, None]
# np.random.seed(2025)
# errors = vi._neg_elbo.check_apply_jacobian(iterate, disp=True)
# np.random.seed(2025)
# print(vi._neg_elbo.jacobian(iterate))

# Optimize the variational posterior
import time

t0 = time.time()
vi.fit()
print(time.time() - t0)
# assert False

# %%
# Plot the Results
# ----------------
# Plot the approximate posterior distribution
axs = plt.subplots(1, 2)[1]
im_vi = variational_posterior._variable.plot_pdf(
    axs[0], prior.interval(1.0).flatten(), levels=31
)
axs[0].set_xlabel(r"Marginal $z_1$")
axs[0].set_ylabel(r"Marginal $z_2$")
axs[0].set_title("Variational Posterior Distribution")

print(variational_posterior)


# Define the unnormalized posterior
def posterior_numerator(x):
    return bkd.exp(loglike(x) + prior.logpdf(x))


# Compute the evidence to normalize the posterior numerator
quad_rule = TensorProductQuadratureRule(
    2,
    [
        GaussQuadratureRule(marginal, backend=bkd)
        for marginal in prior.marginals()
    ],
)
quadx, quadw = quad_rule([100, 100])
print(quadx.min(), prior.marginals()[0].interval(1))
evidence = posterior_numerator(quadx)[:, 0] @ quadw[:, 0]
print(evidence, "evidence")

# Construct a wrapper to evaluate the true posterior PDF
posterior_pdf = ModelFromSingleSampleCallable(
    1, 2, lambda x: posterior_numerator(x) / evidence, backend=bkd
)

# Plot the true posterior
im_true = posterior_pdf.plot_contours(axs[1], [0, 1, 0, 1], levels=30)
axs[1].set_xlabel(r"Marginal $z_1$")
axs[1].set_ylabel(r"Marginal $z_2$")
axs[1].set_title("True Posterior Distribution")

# Plot the true sample
axs[0].plot(*true_sample, "ro", ms=20)
axs[1].plot(*true_sample, "ro", ms=20)

# Adjust the color limits after the plots are created
update_pdf_contourf_plots(im_vi, im_true, axs[0], axs[1])
_ = plt.tight_layout()
plt.show()
