r"""Constructing A Gaussian Process with Inducing Points and Variational Inference
==================================================================================

Gaussian Processes (GPs) are powerful tools for modeling functions in a probabilistic framework. They are widely used in regression, classification, and optimization tasks. However, standard GPs scale poorly with the number of data points :math:`n`, as their computational complexity is :math:`\mathcal{O}(n^3)`. To address this, **inducing points** are introduced as a sparse approximation method to reduce computational costs.

This tutorial demonstrates how to construct a GP with inducing points and variational inference using the provided `InducingSamples` and `InducingGaussianProcess` classes. We will cover the mathematical foundations, demonstrate how to use the classes, and show how to optimize the GP model.

Gaussian Processes
------------------

A Gaussian Process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. It is fully specified by a mean function :math:`m(\mathbf{x})` and a covariance function :math:`k(\mathbf{x}, \mathbf{x}')`:

.. math::

    \mathbf{f} \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')).

In regression tasks, the GP models the relationship between inputs :math:`\mathbf{X}` and outputs :math:`\mathbf{y}`. The posterior distribution of the GP can be used to make predictions at new input points.


Sparse Approximation with Inducing Points
-----------------------------------------

Inducing points are a small set of pseudo-inputs :math:`\mathbf{u}` that summarize the information in the full dataset. The covariance matrix of the GP is approximated using these inducing points, reducing the computational complexity to :math:`\mathcal{O}(nm^2 + m^3)`, where :math:`m` is the number of inducing points (:math:`m \ll n`).

The covariance matrix is approximated using the Nyström method:

.. math::

    \mathbf{K}_{\text{approx}} = \mathbf{K}_{nm} \mathbf{K}_{mm}^{-1} \mathbf{K}_{nm}^\top,

where:

- :math:`\mathbf{K}_{nm}` is the covariance matrix between the data points (:math:`\mathbf{X}`) and the inducing points (:math:`\mathbf{U}`),
- :math:`\mathbf{K}_{mm}` is the covariance matrix between the inducing points.


Variational Inference
---------------------

Variational inference is used to approximate the posterior distribution of the inducing points. The variational posterior :math:`q(\mathbf{u})` is defined as:

.. math::

    q(\mathbf{u}) = \mathcal{N}(\mathbf{u} | \mathbf{m}, \mathbf{S}),

where:

- :math:`\mathbf{m}` is the mean vector of the inducing points,
- :math:`\mathbf{S}` is the covariance matrix of the inducing points.

The Evidence Lower Bound (ELBO) is maximized to optimize the GP model:

.. math::

    \mathcal{L} = \mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} | \mathbf{f})] - \text{KL}(q(\mathbf{u}) \| p(\mathbf{u})),

where:

- The first term is the likelihood term used to train and exact Gaussian process.
- The second term is the KL divergence between the variational posterior :math:`q(\mathbf{u})` and the prior :math:`p(\mathbf{u}).` If we use all the training points as the inducing points this term will be zero



Compute the Likelihood Term
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The likelihood term is computed using the Nyström approximation:

.. math::

    \mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} | \mathbf{f})] = -\frac{1}{2} \left[ \mathbf{y}^\top \mathbf{K}_{\text{noisy}}^{-1} \mathbf{y} + \log |\mathbf{K}_{\text{noisy}}| + n \log(2\pi) \right],

where:

- :math:`\mathbf{K}_{\text{noisy}} = \mathbf{K}_{\text{approx}} + \sigma^2 \mathbf{I}_n`,
- :math:`\mathbf{K}_{\text{approx}} = \mathbf{K}_{nm} \mathbf{K}_{mm}^{-1} \mathbf{K}_{nm}^\top`.

---

Compute the KL Divergence Term
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The KL divergence term is:

.. math::

    \text{KL}(q(\mathbf{u}) \| p(\mathbf{u})) = \frac{1}{2} \left[ \text{tr}(\mathbf{K}_{mm}^{-1} \mathbf{S}) + \mathbf{m}^\top \mathbf{K}_{mm}^{-1} \mathbf{m} - \log |\mathbf{S}| + \log |\mathbf{K}_{mm}| - m \right].

This term measures the difference between the variational posterior :math:`q(\mathbf{u})` and the prior :math:`p(\mathbf{u})`.



Steps to Construct a GP with Inducing Points
--------------------------------------------

This tutorial now demonstrates how to construct and optimize a GP with inducing points and variational inference using the provided classes.
"""

# %%
# Step 1: Define the Kernel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The kernel function defines the covariance structure of the GP. For example, we can use the **Matern Kernel**:
#
# .. math::
#
#    k(\mathbf{x}, \mathbf{x}') = \sigma^2 \left(1 + \frac{\sqrt{5} \|\mathbf{x} - \mathbf{x}'\|}{\ell} + \frac{5 \|\mathbf{x} - \mathbf{x}'\|^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5} \|\mathbf{x} - \mathbf{x}'\|}{\ell}\right),
#
# where:
#
# - :math:`\sigma^2` is the variance,
# - :math:`\ell` is the length scale.


import numpy as np
from scipy import stats
from pyapprox.surrogates.kernels import MaternKernel
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.backends.torch import TorchMixin as bkd

nvars = 2  # Number of input variables
kernel = MaternKernel(np.inf, 0.1, [1e-1, 1], nvars, backend=bkd)


# %%
# Step 2: Initialize Inducing Samples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `InducingSamples` class manages the inducing points and noise hyperparameters. You can specify the number of inducing points and their bounds.

from pyapprox.surrogates.gaussianprocess.variationalgp import InducingSamples

ninducing_samples = 10  # Number of inducing points
inducing_samples = InducingSamples(
    nvars,
    ninducing_samples,
    bkd,
    inducing_sample_bounds=bkd.asarray([-1.0, 1.0]),
    noise_std=1e-2,  # Initial noise value
)


# %%
# Step 3: Setup the GP
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `InducingGaussianProcess` class constructs the GP model using the kernel and inducing samples.


from pyapprox.surrogates.gaussianprocess.variationalgp import (
    InducingGaussianProcess,
)

vi_gp = InducingGaussianProcess(
    nvars=nvars,
    kernel=kernel,
    inducing_samples=inducing_samples,
)

# The GP model is optimized by maximizing the ELBO using gradient-based methods.

from pyapprox.optimization.scipy import (
    DifferentialEvolutionScipyConstrainedGlobalLocalOptimizer,
)

optimizer = DifferentialEvolutionScipyConstrainedGlobalLocalOptimizer()
optimizer.set_verbosity(3)
optimizer.global_optimizer().set_options(maxiter=10)
vi_gp.set_optimizer(optimizer)


# %%
# Step : Train the GP
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The GP model is trained at training points


from pyapprox.interface.model import ModelFromVectorizedCallable

prior = IndependentMarginalsVariable(
    [stats.uniform(-1, 2) for ii in range(nvars)], backend=bkd
)


def fun(x):
    return (bkd.sin(np.pi * x[0, :]) + bkd.cos(np.pi * x[1, :]))[:, None]


model = ModelFromVectorizedCallable(1, nvars, fun, backend=bkd)

np.random.seed(0)
ntrain_samples = 50
X_train = prior.rvs(ntrain_samples)
y_train = fun(X_train)

vi_gp.fit(X_train, y_train)


# %%
# Step 5: Evaluate the GP
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The GP model is evaluated at test points to compute the posterior mean and
# standard deviation.

ntest_samples = 20
X_test = prior.rvs(ntest_samples)
y_test = fun(X_test)
vi_gp_mean = vi_gp(X_test)

print("Test values:\n", y_test)
print("Predicted mean:\n", vi_gp_mean)

# %%
# Compare the Variational GP to the exact GP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from pyapprox.surrogates.gaussianprocess.exactgp import ExactGaussianProcess
import matplotlib.pyplot as plt

exact_gp = ExactGaussianProcess(
    nvars,
    kernel,
    trend=None,
    kernel_reg=0,
)
exact_gp.set_optimizer(optimizer)
exact_gp.fit(X_train, y_train)
exact_gp_mean = exact_gp(X_test)

fig, axs = plt.subplots(1, 3, figsize=(3 * 8, 6))
model.plot(axs[0], prior.interval(1).flatten(), levels=31)
exact_gp.plot(axs[1], prior.interval(1).flatten(), levels=31)
vi_gp.plot(axs[2], prior.interval(1).flatten(), levels=31)

# %%
# Conclusion
# ----------
#
# This tutorial demonstrated how to construct a Gaussian Process with inducing points and variational inference using the provided classes. By leveraging sparse approximations and variational inference, the GP model scales efficiently to large datasets while maintaining accuracy.
