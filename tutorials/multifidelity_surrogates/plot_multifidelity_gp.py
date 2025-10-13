r"""
Multi-fidelity Gaussian Processes
=================================
This tutorial describes how to implement and deploy multi-level Gaussian processes built using the output of a high-fidelity model and evaluations of a set of lower-fidelity models of lower accuracy and cost [KOB2000]_. This tutorial assumes understanding of the concepts in :ref:`sphx_glr_auto_tutorials_surrogates_plot_gaussian_processes.py`

Multilevel GPs assume that all the available models :math:`\{f_m\}_{m=1}^M` can be ordered into a hierarchy of increasing cost and accuracy, where :math:`m=1` denotes the lowest fidelity model and :math:`m=M` denotes the hightest-fidelity model.
We model the output :math:`y_m` from the :math:`m`-th level code as :math:`y_m = f_m(\rv)`
and assume the models satisfy the  hierarchical relationship

.. math:: f_m(\rv)=\rho_{m-1}f_{m-1}(\rv)+\delta_m(\rv), \quad m=2,\ldots,M.

with :math:`f_1(\rv)=\delta_1(\rv)`. We assume that the prior distributions on :math:`\delta_m(\cdot)\sim\mathcal{N}(0, C_m(\cdot,\cdot))` are independent.

Just like traditional GPs the posterior mean and variance of the multi-fidelity GP are given by

.. math::  m^\star(\rv)=t(\rv)^\top C(\mathcal{Z}, \mathcal{Z})^{-1}y \quad\quad C^\star(\rv,\rv^\prime)=C(\rv,\rv^\prime)-t(\rv)^\top C(\mathcal{Z}, \mathcal{Z})^{-1}t(\rv^\prime),

where :math:`C(\mathcal{Z}, \mathcal{Z})` is the prior covariance evaluated at the training data :math:`\mathcal{Z}`.

The difference comes from the definitions of the prior covariance :math:`C` and the vector :math:`t(\rv)`.

Two models
----------
Given data :math:`\mathcal{Z}=[\mathcal{Z}_1, \mathcal{Z}_2]` from two models of differing fidelity, the data covariance of the multi-fidelity GP consisting of two models can be expressed in block form

.. math::
    C(\mathcal{Z},\mathcal{Z})=\begin{bmatrix}
    \covar{f_1(\mathcal{Z}_1)}{f_1(\mathcal{Z}_1)} & \covar{f_1(\mathcal{Z}_1)}{f_2(\mathcal{Z}_2)}\\
   \covar{f_2(\mathcal{Z}_2)}{f_1(\mathcal{Z}_1)} & \covar{f_2(\mathcal{Z}_2)}{f_2(\mathcal{Z}_2)}
    \end{bmatrix}

The upper-diagonal block is given by

.. math::

    \covar{f_1(\mathcal{Z}_1)}{f_1(\mathcal{Z}_1)} = \covar{\delta_1(\mathcal{Z}_1)}{\delta_1(\mathcal{Z}_1)} = C_1(\mathcal{Z}_1, \mathcal{Z}_1)

The lower-diagonal block is given by

.. math::

    \covar{f_2(\mathcal{Z}_2)}{f_2(\mathcal{Z}_2)} &= \covar{\rho_1f_1(\mathcal{Z}_2)+\delta_2(\mathcal{Z}_2)}{\rho_1f_1(\mathcal{Z}_2)+\delta_2(\mathcal{Z}_2)}
 \\ &= \covar{\rho_1\delta_2(\mathcal{Z}_2)+\delta_2(\mathcal{Z}_2)}{\rho_1\delta_1(\mathcal{Z}_2)+\delta_2(\mathcal{Z}_2)} \\ &= \covar{\rho_1\delta_2(\mathcal{Z}_1)}{\rho_1\delta_1(\mathcal{Z}_2)}+\covar{\delta_2(\mathcal{Z}_2)}{\delta_2(\mathcal{Z}_2)}\\ &= \rho_1^2C_1(\mathcal{Z}_2, \mathcal{Z}_2) + C_2(\mathcal{Z}_2, \mathcal{Z}_2)


Where on the second last line we used that :math:`\delta_1` and :math:`\delta_2` are independent.

The upper-right block is given by

.. math::

    \covar{f_1(\mathcal{Z}_1)}{f_2(\mathcal{Z}_2)} &= \covar{\delta_1(\mathcal{Z}_1)}{\rho_1\delta_1(\mathcal{Z}_2)+\delta_2(\mathcal{Z}_2)} \\ &= \covar{\delta_1(\mathcal{Z}_1)}{\rho_1\delta_1(\mathcal{Z}_2)} = \rho_1 C_1(\mathcal{Z}_1, \mathcal{Z}_2)


and :math:`\covar{f_2(\mathcal{Z}_2)}{f_1(\mathcal{Z}_1)}=\covar{f_1(\mathcal{Z}_2)}{f_2(\mathcal{Z}_2)}^\top.`


Combining yields

.. math::
    C(\mathcal{Z},\mathcal{Z})=\begin{bmatrix}
    C_1(\mathcal{Z}_1, \mathcal{Z}_1) & \rho_1 C_1(\mathcal{Z}_1, \mathcal{Z}_2)\\
    \rho_1C_1(\mathcal{Z}_2, \mathcal{Z}_1) & \rho_1^2C_1(\mathcal{Z}_2, \mathcal{Z}_2) + C_2(\mathcal{Z}_2, \mathcal{Z}_2)
    \end{bmatrix}

In this tutorial we assume :math:`\rho_m` are scalars. However PyApprox supports polynomial versions :math:`\rho_m(\rv)`. The above formulas must be slightly modified in this case.

Similary we have

.. math::
    t_m(\rv;\mathcal{Z})^\top=\left[\covar{f_m(\rv)}{f_m(\mathcal{Z}_1)}, \covar{f_m(\rv)}{f_m(\mathcal{Z}_2)}\right]

where

.. math::
   t_1(\rv;\mathcal{Z})^\top=\left[C_1(\rv, \mathcal{Z}_1), \rho_1C_1(\rv, \mathcal{Z}_2)\right]^\top

.. math::
   t_2(\rv;\mathcal{Z})^\top=\left[\rho_1 C_1(\rv, \mathcal{Z}_1), \rho_1^2C_1(\rv, \mathcal{Z}_2) + C_2(\rv, \mathcal{Z}_2)\right]


M models
--------
The diagonal covariance blocks of the prior covariance :math:`C` for :math:`m>1` satisfy

.. math:: C_m(\mathcal{Z}_m,\mathcal{Z}_m)+\rho_{m-1}^2C_{m-1}(\mathcal{Z}_m,\mathcal{Z}_m)+\cdots+\prod_{i=1}^{m-1}\rho_i^2C_1(\mathcal{Z}_m,\mathcal{Z}_m).

The off-diagonal covariance blocks for :math:`m<n` satisfy

.. math::

    C_{m,n}(\mathcal{Z}_m,\mathcal{Z}_n)=&\prod_{i=m}^{n-1}\rho_iC_m(\mathcal{Z}_m,\mathcal{Z}_n)+\rho_{m-1}\prod_{i=m-1}^{n-1}\rho_i\,C_{m-1}(\mathcal{Z}_m,\mathcal{Z}_n)+\\ &\prod_{j=m-2}^{m-1}\rho_{j}\,\prod_{i=m-2}^{n-1}\rho_i\,C_{m-2}(\mathcal{Z}_m,\mathcal{Z}_n)+\cdots+ \prod_{j=2}^{m-1}\rho_{j}\,\prod_{i=2}^{n-1}\rho_i\,C_{2}(\mathcal{Z}_m,\mathcal{Z}_n)+\\ &\prod_{j=1}^{m-1}\rho_{j}\,\prod_{i=1}^{n-1}\rho_i\,C_{1}(\mathcal{Z}_m,\mathcal{Z}_n)



For example, the covariance matrix for :math:`M=3` models is

.. math::
    C=\begin{bmatrix}
    C_1(\mathcal{Z}_1,\mathcal{Z}_1) & \rho_1C_1(\mathcal{Z}_1,\mathcal{Z}_2) & \rho_1\rho_2C_1(\mathcal{Z}_1,\mathcal{Z}_3)\\
    \rho_1C_1(\mathcal{Z}_2,\mathcal{Z}_1) & \rho_1^2C_1(\mathcal{Z}_2,\mathcal{Z}_2)+C_2(\mathcal{Z}_2,\mathcal{Z}_2) & \rho_1^2\rho_2C_1(\mathcal{Z}_2,\mathcal{Z}_3) + \rho_2C_2(\mathcal{Z}_2,\mathcal{Z}_3) \\
    \rho_1\rho_2C_1(\mathcal{Z}_3,\mathcal{Z}_1) & \rho_1^2\rho_2C_1(\mathcal{Z}_3,\mathcal{Z}_2)+\rho_2C_2(\mathcal{Z}_3,\mathcal{Z}_2) & \rho_1^2\rho_2^2C_1(\mathcal{Z}_3,\mathcal{Z}_3)+\rho_2^2C_2(\mathcal{Z}_3,\mathcal{Z}_3)+C_3(\mathcal{Z}_3,\mathcal{Z}_3)
    \end{bmatrix}

and :math:`t_m` is the mth row of :math:`C` that replaces the first argument of each covariance kernel with :math:`\rv`, for example

.. math:: t_3(\rv)^\top = \left[\rho_1\rho_2C_1(\rv,\mathcal{Z}_1), \rho_1^2\rho_2C_1(\rv,\mathcal{Z}_2)+\rho_2C_2(\rv,\mathcal{Z}_2), \rho_1^2\rho_2^2C_1(\rv,\mathcal{Z}_3)+\rho_2^2C_2(\rv,\mathcal{Z}_3)+C_3(\rv,\mathcal{Z}_3)\right]

Now lets plot the structure of the multi-fidelity covariance matrix when using two models and scalar scaling :math:`\rho`. The blocks correspond to the four blocks of the two model data covariance matrix.
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.gaussianprocess.exactgp import (
    SequentialMultiLevelGaussianProcess,
    MOExactGaussianProcess,
    ExactGaussianProcess,
)
from pyapprox.util.hyperparameter import LogHyperParameterTransform
from pyapprox.surrogates.kernels.kernels import MaternKernel, ConstantKernel
from pyapprox.surrogates.gaussianprocess.mokernels import MultiLevelKernel
from pyapprox.surrogates.affine.basisexp import MonomialExpansion
from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.surrogates.affine.basis import MultiIndexBasis
from pyapprox.expdesign.sequences import HaltonSequence
from pyapprox.util.backends.torch import TorchMixin as bkd

np.random.seed(1)
nvars = 1
variable = IndependentMarginalsVariable(
    [stats.uniform(0, 1) for ii in range(nvars)], backend=bkd
)

nmodels = 2
sigmas = [1.0, 0.1]
length_scale = np.hstack([np.full(nvars, 0.3)] * nmodels)
kernels = [
    ConstantKernel(
        sigmas[nn],
        (1e-3, 1e1),
        transform=LogHyperParameterTransform(backend=bkd),
        backend=bkd,
    )
    * MaternKernel(np.inf, length_scale[nn], (0.1, 1.0), nvars, backend=bkd)
    for nn in range(nmodels)
]

rho = bkd.array([[1.0]])
nterms_1d = 1
basis = MultiIndexBasis([Monomial1D(backend=bkd) for ii in range(nvars)])
basis.set_tensor_product_indices([nterms_1d] * nvars)
kernel_scalings = [MonomialExpansion(basis) for nn in range(nmodels - 1)]
[scaling.set_coefficients(rho) for scaling in kernel_scalings]
kernel = MultiLevelKernel(kernels, kernel_scalings)

# sort for plotting covariance matrices
nsamples_per_model = [100, 50]
train_samples_per_model = [
    bkd.sort(
        HaltonSequence(nvars, start_idx=1000 * nn + 1, variable=variable).rvs(
            nsamples_per_model[nn]
        )
    )
    for nn in range(nmodels)
]

ax = plt.subplots(1, 1, figsize=(1 * 8, 6))[1]
Kmat = kernel(train_samples_per_model, train_samples_per_model)
im = ax.imshow(Kmat)
plt.colorbar(im, ax=ax)

# %%
# Now lets plot the structure of the multi-fidelity covariance matrix when using two models a linear polynomial scaling :math:`\rho(x)=a+bx`. In comparison to a scalar scaling the correlation between the low and high-fidelity models changes as a function of the input :math:`\rv`.
ax = plt.subplots(1, 1, figsize=(1 * 8, 6))[1]
rho_2 = bkd.array([0.0, 1.0])[:, None]  # [a, b]
basis_2 = MultiIndexBasis([Monomial1D() for ii in range(nvars)])
basis_2.set_tensor_product_indices([2] * nvars)
kernel_scalings_2 = [MonomialExpansion(basis_2) for nn in range(nmodels - 1)]
[scaling.set_coefficients(rho_2) for scaling in kernel_scalings_2]
kernel_2 = MultiLevelKernel(kernels, kernel_scalings_2)
Kmat2 = kernel_2(train_samples_per_model)
im = ax.imshow(Kmat2)


# %%
# Now lets define two functions with multilevel structure
def scale(degree, x, rho, kk):
    if degree == 0:
        return rho[kk]
    if x.shape[0] == 1:
        return rho[2 * kk] + x.T * rho[2 * kk + 1]
    return rho[3 * kk] + x[:1].T * rho[3 * kk + 1] + x[1:2].T * rho[3 * kk + 2]


def f0(x):
    # y = x.sum(axis=0)[:, None]
    y = x[0:1].T
    return ((y * 3 - 1) ** 2 + 1) * bkd.sin((y * 10 - 2)) / 5


def f1(degree, rho, x):
    # y = x.sum(axis=0)[:, None]
    y = x[0:1].T
    return scale(degree, x, rho, 0) * f0(x) + ((y - 0.5)) / 2


# %%
# Now build a GP with 8 samples from the low-fidelity model and four samples from the high-fidelity model
nsamples_per_model = [8, 4]
train_samples_per_model = [
    bkd.sort(
        HaltonSequence(nvars, start_idx=1000 * nn + 1, variable=variable).rvs(
            nsamples_per_model[nn]
        ),
        axis=1,
    )
    for nn in range(nmodels)
]
degree = 0

true_rho = bkd.full(((nmodels - 1) * degree + 1,), 0.9)
models = [f0, partial(f1, degree, true_rho)]
train_values_per_model = [
    f(x) for f, x in zip(models, train_samples_per_model)
]

gp = MOExactGaussianProcess(nvars, kernel)
gp.set_optimizer(ncandidates=3)
gp.fit(train_samples_per_model, train_values_per_model)
prior_kwargs = {"color": "gray", "alpha": 0.3}
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.plot(train_samples_per_model[0][0], train_values_per_model[0], "ko")
ax.plot(train_samples_per_model[1][0], train_values_per_model[1], "rs")
xx = bkd.linspace(0, 1, 101)[None, :]
ax.plot(xx[0], models[0](xx), "k-", label=r"$f_0$")
ax.plot(xx[0], models[1](xx), "r-", label=r"$f_1$")
gp.plot_1d(
    ax,
    [0, 1],
    0,
    101,
    plt_kwargs={"ls": "--", "label": "MF0", "c": "b"},
    fill_kwargs={"color": "b", "alpha": 0.3},
)
gp.plot_1d(
    ax,
    [0, 1],
    1,
    101,
    plt_kwargs={"ls": ":", "label": "MF1", "c": "g"},
)
_ = ax.legend()

# %%
# The low-fidelity model is very well approximated and the high-fidelity is very good even away from the high-fidelity training data. The low-fidelity data is being used to extrapolate.
#
# Now lets compare the multi-fidelity GP to a single fidelity GP built using only the high-fidelity data
sf_kernel = ConstantKernel(
    sigmas[0],
    (1e-3, 1e1),
    transform=LogHyperParameterTransform(backend=bkd),
    backend=bkd,
) * MaternKernel(bkd.inf(), length_scale[0], (1e-1, 1.0), nvars, backend=bkd)
sf_gp = ExactGaussianProcess(nvars, sf_kernel)
sf_gp.set_optimizer(ncandidates=3)
sf_gp.fit(train_samples_per_model[1], train_values_per_model[1])

ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.plot(xx[0], models[1](xx), "r-", label=r"$f_1$")
ax.plot(train_samples_per_model[1][0], train_values_per_model[1], "rs")
ax.plot(train_samples_per_model[0][0], train_values_per_model[0], "ko")
sf_gp.plot_1d(
    ax,
    [0, 1],
    101,
    plt_kwargs={"label": "SF", "c": "gray"},
    fill_kwargs={"color": "gray", "alpha": 0.3},
)
gp.plot_1d(
    ax,
    [0, 1],
    1,
    101,
    plt_kwargs={"ls": "--", "label": "MF1", "c": "b"},
    fill_kwargs={"color": "b", "alpha": 0.3},
)
gp.plot_1d(
    ax,
    [0, 1],
    0,
    101,
    plt_kwargs={"ls": ":", "label": "MF0", "c": "g"},
)
_ = ax.legend()

# %%
# The single fidelity approximation of the high-fidelity model is much worse than the multi-fidelity approximation.

# %%
# Sequential Construction
# -----------------------
# The computational cost of the multi-level GP presented grows cubically with the number of model evaluations :math:`N_m` for all models, that is the number of operations is
#
# .. math:: O\left(\left(\sum_{m=1}^M N_m)\right)^3\right).
#
# When the training data is nested, e.g. :math:`\mathcal{Z_m}\subseteq \mathcal{Z}_{m+1}` and noiseless, the algorithm in [LGIJUQ2014]_ can be used to construct a MF GP in
#
# .. math:: O\left(\sum_{m=1}^M N_m^3\right)
#
# operations.
#
# Another popular approach used to build MF GPs with either nested or unnested noisy data is to construct a single fidelity GP of the lowest model, then fix that GP and consruct a GP of the difference. This sequential algorithm is also much cheaper than the algorithm presented here, but the mean of the sequentially constructed GP is often less accurate and the posterior variance over- or under-estimated.
#
# The following plot compares a sequential multi-level GP with the exact co-kriging on the toy problem introduced earlier.

sml_kernels = [
    MaternKernel(bkd.inf(), 0.1, (1e-10, 1), nvars, backend=bkd)
    for ii in range(nmodels)
]
sml_scaling_basis = MultiIndexBasis(
    [Monomial1D(backend=bkd) for ii in range(nvars)]
)
sml_scaling_basis.set_tensor_product_indices([1] * nvars)
sml_scalings = [
    MonomialExpansion(sml_scaling_basis) for nn in range(nmodels - 1)
]
[scaling.set_coefficients(bkd.array([[1.0]])) for scaling in sml_scalings]
sml_gp = SequentialMultiLevelGaussianProcess(sml_kernels, sml_scalings)
sml_gp.fit(train_samples_per_model, train_values_per_model)
# gps = sml_gp.gaussian_processes_per_level()

ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.plot(xx[0], models[0](xx), "k-", label=r"$f_0$")
ax.plot(xx[0], models[1](xx), "r-", label=r"$f_1$")
ax.plot(train_samples_per_model[1][0], train_values_per_model[1], "rs")
gp.plot_1d(
    ax,
    [0, 1],
    1,
    101,
    plt_kwargs={"ls": "--", "label": "Co-kriging HF", "c": "b"},
    fill_kwargs={"color": "b", "alpha": 0.3},
)
_ = sml_gp.plot_1d(
    ax,
    [0, 1],
    1,
    101,
    plt_kwargs={"ls": "--", "label": "Sequential HF", "c": "b"},
    fill_kwargs={"color": "g", "alpha": 0.3},
)

_ = sml_gp.plot_1d(
    ax,
    [0, 1],
    0,
    101,
    plt_kwargs={"ls": "--", "label": "Sequential LF", "c": "g"},
    fill_kwargs={"color": "g", "alpha": 0.3},
)
ax.plot(train_samples_per_model[0][0], train_values_per_model[0], "ko")
plt.show()


# %%
# Remarks
# -------
# Some approaches train the GPs of each sequentially, that is train a GP of the lowest-fidelity model. The lowest-fidelity GP is then fixed and data from the next lowest fidelity model is then used to train the GP associated with that data, and so on. However this approach typically produces less accurate approximations (GP means) and does not provide a way to estimate the correct posterior uncertainty of the multilevel GP.

# %%
# Multifidelity Deep Gaussian Processes
# -------------------------------------
# The aforementioned algorithms assumed that the hierarchy of models are linearly related. However, for some model ensembles this may be inefficient and a nonlinear relationship may be more appropriate, e.g.
#
# .. math:: f_m(\rv)=\rho_{m-1}g(f_{m-1}(\rv))+\delta_m(\rv),
#
# where :math:`g` is a nonlinear function. Setting :math:`g` to be Gaussian process leads to multi-fidelity deep Gaussian processes [KPDL2019]_, [PRDLK2017]_.
#
# The following figures (generated using Emukit [EMUKIT]_) demonstrate that the nonlinear formulation works better for the bi-fidelity model enesmble
#
# .. math:: f_1^\text{NL}(\rv)=\sin(8\pi\rv), \qquad f_2^\text{NL}(\rv)=(\rv-\sqrt{2})f_1^\text{NL}(\rv)^2.
#
# .. list-table::
#
#   * -
#       .. _linear-mf-gp-nonlinear-model:
#
#       .. figure:: ../../figures/linear-mf-gp-nonlinear-model.png
#          :width: 100%
#          :align: center
#
#          Linear MF GP
#
#     -
#       .. _nonlinear-mf-gp-nonlinear-model:
#
#       .. figure:: ../../figures/nonlinear-mf-gp-nonlinear-model.png
#          :width: 100%
#          :align: center
#
#          Nonlinear MF GP
#
# The following plots the correlations between the models used previously in this tutorial with the non-linear ensemble used here
axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
nonlinear_models = [
    lambda xx: bkd.sin(np.pi * 8 * xx.T),
    lambda xx: (xx.T - np.sqrt(2)) * bkd.sin(np.pi * 8 * xx.T) ** 2,
]
axs[0].plot(models[0](xx), models[1](xx))
axs[1].plot(nonlinear_models[0](xx), nonlinear_models[1](xx))
# axs[0].legend(['HF-LF Correlation'])
# axs[1].legend(['HF-LF Correlation'])
axs[0].set_xlabel(r"$f_1(z)$")
axs[0].set_ylabel(r"$f_2(z)$")
axs[1].set_xlabel(r"$f_1^{\mathrm{NL}}(z)$")
_ = axs[1].set_ylabel(r"$f_2^{\mathrm{NL}}(z)$")


# %%
# No exact statements can be made about a problem from such plots, however according to [PRDLK2017]_, the additional complexity of the models with the non-linear relationship often indicates that a linear MF GP will perform poorly.

# %%
# References
# ^^^^^^^^^^
# .. [LGIJUQ2014]	`L. Le Gratiet and J. Garnier Recursive co-kriging model for design of computer experiments with multiple levels of fidelity. International Journal for Uncertainty Quantification, 4(5), 365--386, 2014. <http://dx.doi.org/10.1615/Int.J.UncertaintyQuantification.2014006914>`_
#
# .. [KOB2000] `M. C. Kennedy and A. O'Hagan. Predicting the Output from a Complex Computer Code When Fast Approximations Are Available. Biometrika, 87(1), 1-13, 2000. <http://www.jstor.org/stable/2673557>`_
#
# .. [KPDL2019] `K. Cutajar et al. Deep Gaussian Processes for Multi-fidelity Modeling. 2019. <https://doi.org/10.48550/arXiv.1903.07320>`_
#
# .. [PRDLK2017] `P. Perdikaris, et al. Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling. Proceedings of the Royal Society of London A. 2017. <http://rspa.royalsocietypublishing.org/content/473/2198/20160751>`_
#
# .. [EMUKIT] `A. Paleyes et al. Emulation of physical processes with Emukit. econd Workshop on Machine Learning and the Physical Sciences, NeurIPS. <https://github.com/EmuKit/emukit>`_
