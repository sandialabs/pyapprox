r"""
Gaussian processes
==================
Gaussian processes (GPs) [RW2006]_ are an extremely popular tool for approximating multivariate functions from limited data. A GP is a distribution over a set of functions. Given a prior distribution on the class of admissible functions an approximation of a deterministic function is obtained by conditioning the GP on available observations of the function.

Constructing a GP requires specifying a prior mean :math:`m(\rv)` and covariance kernel :math:`C(\rv, \rv^\star)`. The GP leverages the correlation between training samples to approximate the residuals between the training data and the mean function. In the following we set the mean to zero. The covariance kernel should be tailored to the smoothness of the class of functions under consideration. The matern kernel with hyper-parameters :math:`\theta=[\sigma^2,\ell^\top]^\top` is a common choice.

.. math::
   C_\nu(\rv, \rv^\star; \theta)=\sigma^2 \frac{2^{1-\nu}}{\mathsf{\Gamma}(\nu)}\left(\frac{\sqrt{2\nu}d(\rv,\rv^\star; \ell)}{\ell}\right)^{\nu}K_{\nu}\left(\frac{\sqrt{2\nu}d(\rv,\rv^\star; \ell)}{\ell}\right).

Here :math:`d(\rv,\rv^\star; \ell)` is the weighted Euclidean distance between two points parameterized by the  vector hyper-parameters :math:`\ell=[\ell_1,\ldots,\ell_d]^\top`. The variance of the kernel is determined by :math:`\sigma^2` and :math:`K_{\nu}` is the modified Bessel function of the second
kind of order :math:`\nu` and :math:`\mathsf{\Gamma}` is the gamma function.
Note that the parameter :math:`\nu` dictates for the smoothness of the
kernel function. The analytic squared-exponential kernel can be obtained as
:math:`\nu\to\infty`.

Given a kernel and mean function, a Gaussian process approximation assumes that the joint prior distribution of :math:`f`, conditional on kernel hyper-parameters :math:`\theta=[\sigma^2,\ell^\top]^\top`,  is multivariate normal such that

.. math::

   f(\cdot) \mid \theta \sim \mathcal{N}\left(m(\cdot),C(\cdot,\cdot;\theta)+\epsilon^2I\right)

where :math:`\epsilon^2` is the variance of the mean zero white noise in the observations.

The following plots realizations from the prior distribution of a Gaussian process at a set :math:`\mathcal{Z}` of values of :math:`\rv`. Random realizations are drawn by taking the singular value decomposition of the kernel evaluated at the set of points :math:`\mathcal{Z}`, such that

.. math:: USV = C(\mathcal{Z}, \mathcal{Z}),

where :math:`U, V` are the left and right singular vectors and :math:`S` are the singular values. The left singular vectors and singular values are then used to generate random realizations :math:`y` using independent and identically distributed draws :math:`X` from the multivariate standard Normal distribution :math:`\mathcal{N}(0, \V{I}_N)`, where :math:`\V{I}_N` is the identity matrix of size :math:`N`, and :math:`N` is the number of samples in :math:`\mathcal{Z}`. Specifically

.. math:: y = US^{\frac{1}{2}}X.

Note the Cholesky decomposition could also be used instead of the singular value decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.kernels.kernels import MaternKernel
from pyapprox.interface.model import ModelFromVectorizedCallable

np.random.seed(1)
nvars = 1
kernel = MaternKernel(np.inf, 0.5, (1e-1, 1e1), nvars)
gp = ExactGaussianProcess(nvars, kernel)

lb, ub = -1, 1
xx = np.linspace(lb, ub, 101)[None, :]
nsamples = 40
for ii in range(nsamples):
    plt.plot(xx[0], gp.predict_random_realizations(xx, nsamples))
# %%
# The length scale effects the variability of the GP realizations. Try chaning the length_scale to see effect on the GP realizations.
#
# The type of kernel used by the GP also effects the GP realizations. The squared-exponential kernel above is very useful for smooth functions. For less smooth functions a Matern kernel with a smaller :math:`\nu` hyper-parameter may be more effective,

other_kernel = MaternKernel(0.5, 0.5, (1e-1, 1e1), nvars)
other_gp = ExactGaussianProcess(nvars, other_kernel)
for ii in range(nsamples):
    plt.plot(xx[0], other_gp.predict_random_realizations(xx, nsamples))

# %%
# Given a set of training samples :math:`\mathcal{Z}=\{\rv^{(m)}\}_{m=1}^M` and associated values :math:`y=[y^{(1)}, \ldots, y^{(M)}]^\top` the posterior distribution of the GP is
#
# .. math::  f(\cdot) \mid \theta,y \sim \mathcal{N}\left(m^\star(\cdot),C^\star(\cdot,\cdot;\theta)+\epsilon^2I\right)
#
# where
#
# .. math::  m^\star(\rv)=t(\rv)^\top A^{-1}_\mathcal{Z}y \quad\quad C^\star(\rv,\rv^\prime)=C(\rv,\rv^\prime)-t(\rv)^\top A^{-1}_\mathcal{Z}t(\rv^\prime)
#
# with
#
# .. math::      t(\rv)=[C(\rv,\rv^{(1)}),\ldots,C(\rv,\rv^{(N)})]^\top
#
# and :math:`A_\mathcal{Z}` is a matrix with with elements :math:`(A_\mathcal{Z})_{ij}=C(\rv^{(i)},\rv^{(j)})` for :math:`i,j=1,\ldots,M`. Here we dropped the dependence on the hyper-parameters :math:`\theta` for convenience.
#
# Consider the univariate Runge function
#
# .. math:: f(\rv) = \frac{1}{1+25\rv^2}, \quad \rv\in[-1,1]
#
# Lets construct a GP with a fixed set of training samples and associated values we can train the Gaussian process. But first lets plot the true function and prior GP mean and plus/minus 2 standard deviations using the prior covariance


def fun(x):
    return 1 / (1 + 25 * x[0, :] ** 2)[:, np.newaxis]


model = ModelFromVectorizedCallable(1, 1, fun)


validation_samples = np.linspace(lb, ub, 101)[None, :]
validation_values = model(validation_samples)
plt.plot(
    validation_samples[0, :], validation_values[:, 0], "r-", label="Exact"
)
gp_vals, gp_std = gp.evaluate(validation_samples, return_std=True)
plt.plot(validation_samples[0, :], gp_vals[:, 0], "b-", label="GP prior mean")
_ = plt.fill_between(
    validation_samples[0, :],
    gp_vals[:, 0] - 2 * gp_std[:, 0],
    gp_vals[:, 0] + 2 * gp_std[:, 0],
    alpha=0.2,
    color="blue",
    label="GP prior uncertainty",
)

# %%
# Now lets train the GP using a small number of evaluations and plot
# the posterior mean and variance.
ntrain_samples = 5
train_samples = np.linspace(lb, ub, ntrain_samples)[None, :]
train_values = model(train_samples)
gp.fit(train_samples, train_values)
print(gp.kernel())
gp_vals, gp_std = gp.evaluate(validation_samples, return_std=True)
plt.plot(
    validation_samples[0, :], validation_values[:, 0], "r-", label="Exact"
)
plt.plot(train_samples[0, :], train_values[:, 0], "or")
plt.plot(
    validation_samples[0, :], gp_vals[:, 0], "-k", label="GP posterior mean"
)
plt.fill_between(
    validation_samples[0, :],
    gp_vals[:, 0] - 2 * gp_std[:, 0],
    gp_vals[:, 0] + 2 * gp_std[:, 0],
    alpha=0.5,
    color="gray",
    label="GP posterior uncertainty",
)

_ = plt.legend()

# %% As we add more training data the posterior uncertainty will decrease and the mean will become a more accurate estimate of the true function.


# %%
# Constrained Gaussian processes
# ------------------------------
# For certain application is it desirable for GPs to satisfy certain properties, e.g. positivity or monotonicity. See [SGFSJ2020]_ for a review of methods to construct constrained GPs.

# %%
# References
# ^^^^^^^^^^
# .. [RW2006] `C.E. Rasmussen and C. WIlliams. Gaussian Processes for Machine Learning. MIT Press (2006) <http://www.gaussianprocess.org/gpml/>`_
#
# .. [SGFSJ2020] `L.P. Swiler, M. Gulian, A. Frankel, C. Safta, J.D. Jakeman. A Survey of Constrained Gaussian Process Regression: Approaches and Implementation Challenges. Journal of Machine Learning for Modeling and Computing (2020) <http://www.dl.begellhouse.com/journals/558048804a15188a,2cbcbe11139f18e5,0776649265326db4.html>`_
