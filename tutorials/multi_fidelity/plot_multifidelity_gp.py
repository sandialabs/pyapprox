r"""
Multifidelity Gaussian processes
================================
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.gaussianprocess.multilevel import (
    GreedyMultifidelityIntegratedVarianceSampler, MultifidelityGaussianProcess,
    GaussianProcess)
from pyapprox.surrogates.gaussianprocess.kernels import (
    RBF, MultifidelityPeerKernel, MonomialScaling)
from pyapprox.surrogates.integrate import integrate


def scale(x, rho, kk):
    if degree == 0:
        return rho[kk]
    if x.shape[0] == 1:
        return rho[2*kk] + x.T*rho[2*kk+1]
    return rho[3*kk] + x[:1].T*rho[3*kk+1]+x[1:2].T*rho[3*kk+2]


def f0(x):
    # y = x.sum(axis=0)[:, None]
    y = x[0:1].T
    return ((y*3-1)**2+1)*np.sin((y*10-2))/5


def f1(x):
    # y = x.sum(axis=0)[:, None]
    y = x[0:1].T
    return scale(x, rho, 0)*f0(x)+((y-0.5))/2


np.random.seed(1)
nvars = 1
variable = IndependentMarginalsVariable(
            [stats.uniform(0, 1) for ii in range(nvars)])

nmodels = 2
kernels = [RBF for nn in range(nmodels)]
sigma = [1, 0.1]
length_scale = np.hstack([np.full(nvars, 0.3)]*nmodels)

rho = [1.0]
degree = 0
kernel_scalings = [
    MonomialScaling(nvars, degree) for nn in range(nmodels-1)]
kernel = MultifidelityPeerKernel(
    nvars, kernels, kernel_scalings, length_scale=length_scale,
    length_scale_bounds=(1e-10, 1), rho=rho,
    rho_bounds=(1e-3, 1), sigma=sigma,
    sigma_bounds=(1e-3, 2))

# sort for plotting covariance matrices
nsamples_per_model = [100, 50]
train_samples = [
    np.sort(integrate("quasimontecarlo", variable,
                      nsamples=nsamples_per_model[nn],
                      startindex=1000*nn+1)[0], axis=1)
    for nn in range(nmodels)]

ax = plt.subplots(1, 1, figsize=(1*8, 6))[1]
kernel.set_nsamples_per_model(nsamples_per_model)
Kmat = kernel(np.hstack(train_samples).T)
im = ax.imshow(Kmat)
plt.colorbar(im, ax=ax)

ax = plt.subplots(1, 1, figsize=(1*8, 6))[1]
rho_2 = [0., 1.0]
kernel_scalings_2 = [
    MonomialScaling(nvars, 1) for nn in range(nmodels-1)]
kernel_2 = MultifidelityPeerKernel(
    nvars, kernels, kernel_scalings_2, length_scale=length_scale,
    length_scale_bounds=(1e-10, 1), rho=rho_2,
    rho_bounds=(1e-3, 1), sigma=sigma,
    sigma_bounds=(1e-3, 1))
kernel_2.set_nsamples_per_model(nsamples_per_model)
Kmat2 = kernel_2(np.hstack(train_samples).T)
im = ax.imshow(Kmat2)


nsamples_per_model = [8, 4]
train_samples = [
    np.sort(integrate("quasimontecarlo", variable,
                      nsamples=nsamples_per_model[nn],
                      startindex=1000*nn+1)[0], axis=1)
    for nn in range(nmodels)]
true_rho = np.full((nmodels-1)*degree+1, 0.9)
models = [f0, f1]
train_values = [f(x) for f, x in zip(models, train_samples)]

gp = MultifidelityGaussianProcess(kernel, n_restarts_optimizer=20)
gp.set_data(train_samples, train_values)
gp.fit()
prior_kwargs = {"color": "gray", "alpha": 0.3}
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.plot(train_samples[0][0], train_values[0], 'ko')
ax.plot(train_samples[1][0], train_values[1], 'rs')
xx = np.linspace(0, 1, 101)[None, :]
ax.plot(xx[0], models[0](xx), 'k-', label=r"$f_0$")
ax.plot(xx[0], models[1](xx), 'r-', label=r"$f_1$")
gp.plot_1d(101, [0, 1], plt_kwargs={"ls": "--", "label": "MF1", "c": "b"},
           ax=ax, fill_kwargs={"color": "b", "alpha": 0.3})
gp.plot_1d(101, [0, 1], plt_kwargs={"ls": ":", "label": "MF0", "c": "g"},
           ax=ax, model_eval_id=0)
print(gp.kernel_)
ax.legend()
# plt.show()

sf_kernel = 1.0*RBF(length_scale[0], (1e-3, 1))
sf_gp = GaussianProcess(sf_kernel)
sf_gp.fit(train_samples[1], train_values[1])

ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.plot(xx[0], models[1](xx), 'r-', label=r"$f_1$")
ax.plot(train_samples[1][0], train_values[1], 'rs')
sf_gp.plot_1d(101, [0, 1], plt_kwargs={"label": "SF", "c": "gray"}, ax=ax,
              fill_kwargs={"color": "gray", "alpha": 0.3})
gp.plot_1d(101, [0, 1], plt_kwargs={"ls": "--", "label": "MF1", "c": "b"},
           ax=ax, fill_kwargs={"color": "b", "alpha": 0.3})
gp.plot_1d(101, [0, 1], plt_kwargs={"ls": ":", "label": "MF0", "c": "g"},
           ax=ax, model_eval_id=0)
ax.legend()
# plt.show()


axs = plt.subplots(1, 2, figsize=(2*8, 6))[1]
kernel = MultifidelityPeerKernel(
    nvars, kernels, kernel_scalings, length_scale=length_scale,
    length_scale_bounds="fixed", rho=rho,
    rho_bounds="fixed", sigma=sigma,
    sigma_bounds="fixed")
gp_2 = MultifidelityGaussianProcess(kernel)
gp_2.set_data([np.full((nvars, 1), 0.5), np.empty((nvars, 0))],
              [np.zeros((1, 1)), np.empty((0, 1))])
gp_2.fit()
gp_2.plot_1d(101, [0, 1], plt_kwargs={"ls": "--", "label": "MF1", "c": "gray"},
             ax=axs[0], fill_kwargs={"color": "g", "alpha": 0.3})
gp_2.set_data([np.empty((nvars, 0)), np.full((nvars, 1), 0.5)],
              [np.empty((0, 1)), np.zeros((1, 1))])
axs[0].plot(0.5, 0, 'o')
gp_2.fit()
gp_2.plot_1d(101, [0, 1], plt_kwargs={"ls": "--", "label": "MF1", "c": "b"},
             ax=axs[0], fill_kwargs={"color": "b", "alpha": 0.3})
axs[1].plot(0.5, 0, 'ko')

gp_2.plot_1d(101, [0, 1], plt_kwargs={"ls": "--", "label": "MF1", "c": "b"},
             ax=axs[1], fill_kwargs={"color": "b", "alpha": 0.3})
gp_2.set_data([np.array([[0.3, 0.7]]), np.empty((nvars, 0))],
              [np.zeros((2, 1)), np.empty((0, 1))])
gp_2.fit()
gp_2.plot_1d(101, [0, 1], plt_kwargs={"ls": "--", "label": "MF1", "c": "gray"},
             ax=axs[1], fill_kwargs={"color": "g", "alpha": 0.3})

plt.show()

nquad_samples = 1000
model_costs = np.array([1, 3])
ncandidate_samples_per_model = 101
sampler = GreedyMultifidelityIntegratedVarianceSampler(
    nmodels, nvars, nquad_samples, ncandidate_samples_per_model,
    variable.rvs, variable, econ=True,
    compute_cond_nums=False, nugget=0, model_costs=model_costs)

integrate_args = ["tensorproduct", variable]
integrate_kwargs = {"rule": "gauss", "levels": 100}
sampler.set_kernel(kernel, *integrate_args, **integrate_kwargs)

nsamples = 10
samples = sampler(nsamples)[0]
train_samples = np.split(
    samples, np.cumsum(sampler.nsamples_per_model).astype(int)[:-1],
    axis=1)
nsamples_per_model = np.array([s.shape[1] for s in train_samples])
costs = nsamples_per_model*model_costs
total_cost = costs.sum()
print("NSAMPLES", nsamples_per_model)
print("TOTAL COST", total_cost)
train_values = [f(x) for f, x in zip(models, train_samples)]
gp.set_data(train_samples, train_values)
gp.fit()

# Note samples depend on hyperparameter values
from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    GreedyIntegratedVarianceSampler)
sf_sampler = GreedyIntegratedVarianceSampler(
    nvars, 2000, 1000, variable.rvs,
    variable, use_gauss_quadrature=True, econ=True,
    compute_cond_nums=False)
sf_sampler.set_kernel(sf_kernel)
nsf_train_samples = int(total_cost//model_costs[1])
print("NSAMPLES SF", nsf_train_samples)
sf_train_samples = sf_sampler(nsf_train_samples)[0]
sf_kernel = 1.0*RBF(length_scale[0], (1e-3, 1))
sf_gp = GaussianProcess(sf_kernel)
sf_train_values = models[1](sf_train_samples)
sf_gp.fit(sf_train_samples, sf_train_values)

ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.plot(xx[0], models[1](xx), 'r-', label=r"$f_1$")
ax.plot(train_samples[1][0], train_values[1], 'rs')
sf_gp.plot_1d(101, [0, 1], plt_kwargs={"label": "SF", "c": "gray"}, ax=ax,
              fill_kwargs={"color": "gray", "alpha": 0.3})
gp.plot_1d(101, [0, 1], plt_kwargs={"ls": "--", "label": "MF1", "c": "b"},
           ax=ax, fill_kwargs={"color": "b", "alpha": 0.3})
gp.plot_1d(101, [0, 1], plt_kwargs={"ls": ":", "label": "MF0", "c": "g"},
           ax=ax, model_eval_id=0)
ax.legend()
plt.show()



