r"""
Experimental Design for Multi-fidelity Gaussian Processes
=========================================================
 As with single fidelity GPs the location of the training data significantly impacts the accuracy of a multi-fidelity GP. The remainder of this tutorial discusses xtensions of the experimental design methods used for single-fidelity GPs in :ref:`sphx_glr_auto_tutorials_surrogates_plot_gaussian_processes.py`.

 The following code demonstrates the additional complexity faced when desigining experimental designs for multi-fidelity GPs, specifically one must not only choose what input to sample but also what model to sample.
"""

from functools import partial

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.gaussianprocess.exactgp import (
    MOExactGaussianProcess,
    ExactGaussianProcess,
)
from pyapprox.surrogates.kernels.kernels import (
    MaternKernel,
    ConstantKernel,
    LogHyperParameterTransform,
)
from pyapprox.surrogates.gaussianprocess.mokernels import (
    MultiLevelKernel,
    construct_tensor_product_monomial_scaling,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.interface.model import ModelListCostFunction
from pyapprox.surrogates.gaussianprocess.activelearning import (
    MultiOutputMonteCarloGreedyIntegratedVarianceSampler,
    MonteCarloGreedyIntegratedVarianceSampler,
)

bkd = NumpyMixin
np.random.seed(1)
nvars = 1
nmodels = 2
variable = IndependentMarginalsVariable(
    [stats.uniform(0, 1)] * nvars, backend=bkd
)
degree = 0
# fix hyperparams so fit does not adjust them, otherwise comparison
# of posterior variance will not be correct
kernels = [
    MaternKernel(np.inf, 0.3, [1e-1, 1], nvars, fixed=True, backend=bkd)
    for nn in range(nmodels)
]
# Changing the default value of scaling will change how points are placed
# As will changing relative model costs. Decreasing scaling will mean
# more high-fidelity points are added for a given cost ratio. Reducing the high-fdielity model cost will also have the same affect.
scalings = [
    construct_tensor_product_monomial_scaling(
        nvars, [degree + 1] * nvars, 1, [0.8, 1], fixed=True, bkd=bkd
    )
]
kernel = MultiLevelKernel(kernels, scalings)

# build GP using only one low-fidelity data point. The values of the function
# do not matter as we are just going to plot the pointwise variance of the GP.
gp = MOExactGaussianProcess(nvars, kernel)
gp.fit(
    [np.full((nvars, 1), 0.5), np.empty((nvars, 0))],
    [np.zeros((1, 1)), np.empty((0, 1))],
)

# plot the variance of the multi-fidelity GP approximation of the
# high-fidelity model
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
gp.plot_1d(
    ax,
    [0, 1],
    nmodels - 1,
    npts_1d=101,
    plt_kwargs={"ls": "--", "label": "1 LF data", "c": "gray"},
    fill_kwargs={"color": "g", "alpha": 0.3},
)
ax.plot(0.5, 0, "rs", ms=15)


# build GP using only one high-fidelity data point
gp.fit(
    [np.empty((nvars, 0)), np.full((nvars, 1), 0.5)],
    [np.empty((0, 1)), np.zeros((1, 1))],
)
gp.plot_1d(
    ax,
    [0, 1],
    nmodels - 1,
    npts_1d=101,
    plt_kwargs={"ls": "--", "label": "1 HF data", "c": "b"},
    fill_kwargs={"color": "b", "alpha": 0.3},
)
ax.plot(0.5, 0, "ko")
_ = ax.legend()

# %%
# As expected the high-fidelity data point reduces the high-fidelity GP variance the most.
#
# The following shows that two low-fidelity points may produce smaller variance than a single high-fidelity point.


# plot one point high-fidelity GP against two low-fidelity points
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
gp.plot_1d(
    ax,
    [0, 1],
    nmodels - 1,
    101,
    plt_kwargs={"ls": "--", "label": "1 HF data", "c": "b"},
    fill_kwargs={"color": "b", "alpha": 0.3},
)
ax.plot(0.5, 0, "ko")

# build GP using twp low-fidelity data points.
gp.fit(
    [np.array([[0.3, 0.7]]), np.empty((nvars, 0))],
    [np.zeros((2, 1)), np.empty((0, 1))],
)
gp.plot_1d(
    ax,
    [0, 1],
    nmodels - 1,
    101,
    plt_kwargs={"ls": "--", "label": "2 LF data", "c": "gray"},
    fill_kwargs={"color": "g", "alpha": 0.3},
)
ax.plot([0.3, 0.7], [0, 0], "rs", ms=15)
_ = ax.legend()

# %%
# Analagous to single-fidelity GPs we will now use integrated variance to produce good experimental designs. Two model, multi-fidelity, integrated variance designs, as they are often called, find a set of samples :math:`\mathcal{Z}\subset\Omega\subset\rvdom` from a set of candidate samples :math:`\Omega=\Omega_1\cup\Omega_2` by greedily selecting a new point :math:`\rv^{(n+1)}` to add to an existing set :math:`\mathcal{Z}_n` according to
#
# .. math:: \rv^{(n+1)} = \argmin_{\mathcal{Z}_n\cup \rv \subset\Omega\subset\rvdom} W(z) \int_{\rvdom} C^\star(\rv, \rv\mid \mathcal{Z})\pdf(\rv)d\rv.
#
# Here :math:`\Omega_m` denotes the candidate samples for the mth model, :math:`W(\rv)` is the cost of evaluating the candidate which is typically constant for all inputs for a given model, but different between models.
#
# Now generate a sample set
nquad_samples = 10000
cost_function = ModelListCostFunction(bkd.array([1, 3]))
sampler = MultiOutputMonteCarloGreedyIntegratedVarianceSampler(
    variable, cost_function
)
sampler.set_surrogate(gp)
model_costs = np.array([1, 3])
ncandidates_per_model = 101
candidate_samples = [
    bkd.linspace(0, 1, ncandidates_per_model)[None, :] for nn in range(nmodels)
]
sampler.set_candidate_samples(candidate_samples)
# init_pivots refer to index in array that concatenates all candidates
# following adds the midpoint of the highest indexed output
init_pivots = bkd.array(
    [ncandidates_per_model * (nmodels - 1) + ncandidates_per_model // 2],
    dtype=int,
)
sampler.set_initial_pivots(init_pivots)

# %%
# First plot the variance of the GP. The values of the function
# do not matter as we are just going to plot the pointwise variance of the GP.

nsamples = 4
axs = plt.subplots(1, nsamples, sharey=True, figsize=(nsamples * 8, 6))[1]
for ii in range(nsamples):
    new_samples = sampler(ii + 1)
    if ii == 0:
        samples = new_samples
    else:
        samples = [np.hstack((s, n)) for s, n in zip(samples, new_samples)]
    gp.fit([s for s in samples], [s[0][:, None] * 0 for s in samples])
    gp.plot_1d(
        axs[ii],
        [0, 1],
        nmodels - 1,
        101,
        plt_kwargs={"ls": "--", "c": "gray"},
        fill_kwargs={"color": "g", "alpha": 0.3},
    )
    axs[ii].set_ylim(-3, 3)
    axs[ii].scatter(
        samples[0], 0 * samples[0], marker="s", c="r", s=15**2, label="LF"
    )
    axs[ii].scatter(
        samples[1], 0 * samples[1], marker="o", c="k", s=15**2, label="HF"
    )
    _ = ax.legend()


# %%
# Now generate more samples and fit a GP with the selected samples
nsamples = 7
new_samples = sampler(nsamples)
train_samples = [np.hstack((s, n)) for s, n in zip(samples, new_samples)]


# Activate all hyperparams so they can be optimized
gp.hyp_list().set_all_active()
# must recall set_optimizer so that optimizer knows that parameters
# are now active
gp.set_optimizer(ncandidates=20)


def scale(degree, x, rho, kk):
    if degree == 0:
        return rho[kk]
    if x.shape[0] == 1:
        return rho[2 * kk] + x.T * rho[2 * kk + 1]
    return rho[3 * kk] + x[:1].T * rho[3 * kk + 1] + x[1:2].T * rho[3 * kk + 2]


def f0(x):
    # y = x.sum(axis=0)[:, None]
    y = x[0:1].T
    return ((y * 3 - 1) ** 2 + 1) * np.sin((y * 10 - 2)) / 5


def f1(degree, rho, x):
    # y = x.sum(axis=0)[:, None]
    y = x[0:1].T
    return scale(degree, x, rho, 0) * f0(x) + ((y - 0.5)) / 2


true_rho = np.full((nmodels - 1) * degree + 1, 0.9)
models = [f0, partial(f1, degree, true_rho)]
costs = sampler.ntrain_samples_per_output() * cost_function.cost_per_model()
total_cost = costs.sum()
print("NSAMPLES PER MODEL", sampler.ntrain_samples_per_output())
print("TOTAL COST", total_cost)
train_values = [f(x) for f, x in zip(models, train_samples)]
gp.fit(train_samples, train_values)


# %%
# Now build a single-fidelity GP using IVAR. Fix the variance of the SF GP to be that of the variance of the highest fidelity model so comparison is fair
assert degree == 0
# only works for degree == 0
sf_prior_var = scalings[0].get_coefficients()[0] ** 2 + 1
constant_kernel = ConstantKernel(
    sf_prior_var,
    (1e-3, 1e1),
    transform=LogHyperParameterTransform(backend=bkd),
    fixed=True,
    backend=bkd,
)
sf_kernel = constant_kernel * MaternKernel(
    np.inf, 0.3, [1e-1, 1], nvars, backend=bkd
)
sf_gp = ExactGaussianProcess(nvars, sf_kernel)
sf_sampler = MonteCarloGreedyIntegratedVarianceSampler(variable)
sf_sampler.set_surrogate(sf_gp)
sf_sampler.set_candidate_samples(candidate_samples[nmodels - 1])
init_pivots = bkd.array([ncandidates_per_model // 2], dtype=int)
sf_sampler.set_initial_pivots(init_pivots)
nsf_train_samples = int(total_cost // model_costs[1])
print("NSAMPLES SF", nsf_train_samples)
sf_train_samples = sf_sampler(nsf_train_samples)
sf_train_values = models[nmodels - 1](sf_train_samples)
sf_gp.fit(sf_train_samples, sf_train_values)

# %%
# Compare the multi- and single-fidelity GPs.
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
xx = np.linspace(0, 1, 101)[None, :]
ax.plot(xx[0], models[1](xx), "r-", label=r"$f_2$")
ax.scatter(
    train_samples[1][0], train_values[1], c="r", marker="s", label="MF2"
)
ax.scatter(sf_train_samples[0], sf_train_values, c="k", marker="X", label="SF")
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
    nmodels - 1,
    101,
    plt_kwargs={"ls": "--", "label": "MF2", "c": "b"},
    fill_kwargs={"color": "b", "alpha": 0.3},
)
# turn off plot of low fidelity plots to reduce complexity of image
# ax.plot(xx[0], models[0](xx), "g-", label=r"$f_1$")
# ax.scatter(
#     train_samples[0][0], train_values[0], c="g", marker="D", label="MF1"
# )
# gp.plot_1d(
#     ax,
#     [0, 1],
#     0,
#     101,
#     plt_kwargs={"ls": ":", "label": "MF1", "c": "g"},
# )
_ = ax.legend()

# %%
# The multi-fidelity GP is again clearly superior.
#
# Note the selected samples depend on hyperparameter values, change the hyper-parameters and see how the design changes. The relative cost of evaluating each model also impacts the design. For fixed hyper-parameters a larger ratio will typically result in more low-fidelity samples.

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
    lambda xx: np.sin(np.pi * 8 * xx.T),
    lambda xx: (xx.T - np.sqrt(2)) * np.sin(np.pi * 8 * xx.T) ** 2,
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
