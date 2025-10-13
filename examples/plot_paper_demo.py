r"""
End-to-End Model Analysis
=========================
This tutorial describes how to use each of the major model analyses in Pyapprox
following the exposition in [PYAPPROX2023]_.

First lets load all the necessary modules and set the random seeds for reproducibility.
"""

import time

from scipy import stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from pyapprox.util.visualization import mathrm_label
from pyapprox.variables import IndependentMarginalsVariable, AffineTransform
from pyapprox.benchmarks import (
    PyApproxPaperAdvectionDiffusionKLEInversionBenchmark,
)
from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.analysis.sensitivity_analysis import (
    EnsembleGaussianProcessSensitivityAnalysis,
)
from pyapprox.surrogates.kernels import MaternKernel, ConstantKernel
from pyapprox.surrogates.gaussianprocess.activelearning import (
    CholeskySampler,
    AdaptiveGaussianProcess,
    SamplingScheduleFromList,
)
from pyapprox.bayes.metropolis import MetropolisMCMCVariable
from pyapprox.expdesign.optbayes import (
    BruteForceKLBayesianOED,
    BayesianOEDDataGenerator,
    IndependentGaussianOEDInnerLoopLogLikelihood,
)
from pyapprox.util.backends.torch import TorchMixin as bkd
from pyapprox.bayes.likelihood import LogLikelihoodFromModel
from pyapprox.multifidelity.factory import multioutput_stats
from pyapprox import multifidelity as mf

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(2023)
_ = torch.manual_seed(2023)

# %%
# The tutorial can save the figures to file if desired. If you do want the plots
# set savefig=True

# savefig = True
savefig = False

# %%
# The following code shows how to create and sample from two independent uniform random variables defined on :math:`[-2, 2]`. We use uniform variables here, but any marginal from the scipy.stats module can be used.
nsamples = 30
univariate_variables = [stats.uniform(-2, 4), stats.uniform(-2, 4)]
variable = IndependentMarginalsVariable(univariate_variables, backend=bkd)
samples = variable.rvs(nsamples)

# %%
# PyApprox supports various types of variable transformations. The following code
# shows how to use an affinte transformation to map samples from variables
# to samples from the variable's canonical form.
var_trans = AffineTransform(variable)
canonical_samples = var_trans.map_to_canonical(samples)


# %%
# Pyapprox provides many utilities for interfacing with complex numerical codes.
# The following shows how to wrap a model and store the wall time required
# to evaluate each sample in a set. First define a function
# with a random execution time that takes in one sample at a time, i.e. a
# 1D array. Then wrap that model so that multiple samples can be evaluated at
# once.
def fun_pause_1(sample):
    assert sample.ndim == 1
    time.sleep(np.random.uniform(0, 0.05))
    return bkd.atleast1d(bkd.sum(sample**2))


# Create the model
model = ModelFromSingleSampleCallable(
    1, variable.nvars(), fun_pause_1, sample_ndim=1, values_ndim=1, backend=bkd
)
# Activate timing of the model evaluations
model.activate_model_data_base()

# %%
# Run the model and print the computational cost of the evaluations

# Run the model
values = model(samples)
# Print the number of evaluations and average time
print(model.work_tracker())

# %%
# Other wrappers available in PyApprox include those for running multiple models
# at once, useful for multi-fidelity methods, wrappers that fix a subset of inputs
# to user specified values, wrappers that only return a subset of all
# possible model ouputs, and wrappers for evaluating samples in parallel.

# %%
# Pyapprox provide numerous benchmarks for verifying, validating and comparing
# model analysis algorithms. The following list the names of all benchmarks and
# then creates a benchmark that can be used to test the creation of surrogates,
# Bayesian inference, and optimal experimental design. This benchmark requires
# determining the true coefficients of the Karhunene Loeve expansion (KLE)
# used to characterize the uncertain diffusivity field of an advection
# diffusion equation.
# This benchmark is slightly different to that documented in [PYAPPROX2023]_.
# Here we solve the conservative advection diffusion equations and adjust
# the source slightly.
# See documentation of the benchmark for more details.
noise_stdev = 1  # 1e-1
inv_benchmark = PyApproxPaperAdvectionDiffusionKLEInversionBenchmark(
    backend=bkd
)
print(inv_benchmark)

# %%
# The following plots the modes of the KLE
eigvecs = inv_benchmark.diffusion_function().eigenfunctions()
fig, axs = plt.subplots(1, len(eigvecs), figsize=(8 * len(eigvecs), 6))
for ii in range(len(eigvecs)):
    eigvecs[ii].plot(axs[ii], cmap=plt.cm.coolwarm, levels=50)

# %%
# PyApprox provides many popular methods for constructing surrogates
# that once constructed can be evaluated in place of a computaionally
# expensive simulation model in model analyses. The following code creates
# a Gaussian process (GP) surrogate. The function used to construct the surrogate
# takes a callback which is evaluated each time the adaptive surrogate is refined.
# Here we use to compute the error of the surrogate as it is constructed using
# validation data. Uncomment the code to use a polynomial based surrogate instead
# of a GP. The user does not have to change any subsequent code
validation_samples = inv_benchmark.prior().rvs(100)
validation_values = inv_benchmark.loglike()(validation_samples)

# fig, axs = plt.subplots(1, 4, figsize=(4 * 8, 6))
# # plot a single solution to the PDE before overlaying the designs

# debug_samples = bkd.array(
#     [
#         [-3.2905267314918945, -2.6324213851935125],
#         [-0.6581053462983659, -1.9743160388951302],
#         [0.0, 0.0],
#     ]
# )

# sample = inv_benchmark._true_kle_params
# sol_array = inv_benchmark.model().forward_solve(sample)
# print(sol_array.min())
# print(sol_array.max())
# sol = inv_benchmark.model().physics().solution_from_array(sol_array)
# im = sol.plot(axs[0], 50, levels=30)
# plt.colorbar(im, ax=axs[0])
# im = inv_benchmark.model().physics()._diffusion.plot(axs[1])
# plt.colorbar(im, ax=axs[1])
# sample = debug_samples[:, :1]
# sol_array = inv_benchmark.model().forward_solve(sample)
# print(sol_array.min())
# print(sol_array.max())
# sol = inv_benchmark.model().physics().solution_from_array(sol_array)
# im = sol.plot(axs[2], 50, levels=30)
# plt.colorbar(im, ax=axs[2])
# im = inv_benchmark.model().physics()._diffusion.plot(axs[3])
# plt.colorbar(im, ax=axs[3])
# plt.show()

# inv_benchmark.model().activate_model_data_base()
# inv_benchmark.model().plot_cross_sections(
#     inv_benchmark.prior().mean(),
#     inv_benchmark.prior().truncated_ranges(1 - 1e-3),
#     npts_1d=21,
#     levels=30,
# )
print(inv_benchmark.loglike().work_tracker())


class Callback:
    def __init__(self, backend):
        self._nsamples, self._errors = [], []
        self._bkd = backend

    def __call__(self, approx):
        self._nsamples.append(approx.ntrain_samples())
        error = self._bkd.norm(
            approx(validation_samples) - validation_values, axis=0
        )
        error /= self._bkd.norm(validation_values, axis=0)
        self._errors.append(error)

    def nsamples(self):
        print(self._nsamples)
        return self._bkd.array(self._nsamples)

    def errors(self):
        return self._bkd.hstack(self._errors)


kernel_variance = 1.0
kernel = ConstantKernel(
    kernel_variance, fixed=True, backend=bkd
) * MaternKernel(
    np.inf, 1.0, [1e-1, 1], inv_benchmark.prior().nvars(), backend=bkd
)
sampling_schedule = SamplingScheduleFromList([10, 10, 10, 10, 10], backend=bkd)
sampler = CholeskySampler(inv_benchmark.prior())
gp = AdaptiveGaussianProcess(
    inv_benchmark.prior().nvars(),
    kernel,
    sampling_schedule=sampling_schedule,
)
# unlike paper example approximate loglikelihood not negative loglikelihood
gp.set_sampler(sampler)
callback = Callback(bkd)
while gp.step(inv_benchmark.loglike()):
    callback(gp)

# %%
# We can plot the errors obtained from the callback with
ax = plt.subplots(figsize=(8, 6))[1]
ax.loglog(callback.nsamples(), callback.errors(), "o-")
ax.set_xlabel(mathrm_label("No. Samples"))
ax.set_ylabel(mathrm_label("Error"))
ax.set_xticks([10, 25, 50])
ax.set_yticks([0.3, 0.75, 1.5])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.minorticks_off()
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
# ax.tick_params(axis='y', which='minor', bottom=False)
if savefig:
    plt.savefig("gp-error-plot.pdf")

# %%
# Now we will perform a sensitivity analysis. Specifically we compute
# variance based sensitivity indices that measure the impact of each KLE mode
# on the mismatch between the observed data and the model predictions.
# We use the negative log likelihood to characterize this mismatch.
# Here we have used the surrogate to speed up the computation of the sensitivity
# indices. Uncomment the commented code to use the numerical model. Note
# the drastic increase in computational cost. Warning: using the numerical model
# will take many minutes. The plots in the figure, generated from
# left to right are: main effect, largest Sobol indices and total effect indices.


analyzer = EnsembleGaussianProcessSensitivityAnalysis(inv_benchmark.prior())
analyzer.set_interaction_terms_of_interest(
    inv_benchmark.sobol_interaction_indices()
)
analyzer.compute(gp)
axs = plt.subplots(1, 3, figsize=(3 * 8, 6), sharey=True)[1]
analyzer.plot_main_effects(axs[0])
analyzer.plot_total_effects(axs[1])
analyzer.plot_sobol_indices(axs[2])
if savefig:
    plt.savefig("gp-sa-indices.pdf", bbox_inches="tight")

# %%
# Now we will use the surrogate with Bayesian inference to learn the
# coefficients of the KL. Specifically we will draw a set of samples from
# the posterior distribution of the KLE given the observed data provided
# in the benchmark.
#
# But First we will improve the accuracy of the surrogate
# and print out the error which can be compared to the errors previously plotted.
# The error of the original surrogate was kept low to demonstrate the ability
# to quantify error in the sensitivity indices from using a surrogate.


sampling_schedule.update([100, 200])
while gp.step(inv_benchmark.loglike()):
    callback(gp)
print(callback.nsamples())
print(callback.errors())
# %%
# Now create a MCMCVariable to sample from the posterior. The benchmark
# has already formulated the negative log likelihood that is needed. Here
# we will use PyApprox's native delayed rejection adaptive metropolis (DRAM)
# sampler.
#
# Uncomment the commented code to use the numerical model instead of the surrogate
# with the MCMC algorithm. Again note the significant increase in computational
# time
npost_samples = 1000
loglike = LogLikelihoodFromModel(gp)
# loglike = inv_benchmark.loglike()
mcmc_variable = MetropolisMCMCVariable(
    inv_benchmark.prior(), loglike, method_opts={"cov_scaling": 1}
)
print(mcmc_variable)
map_sample = mcmc_variable.maximum_aposteriori_point()
print("Computed Map Point", map_sample[:, 0])
post_samples = mcmc_variable.rvs(npost_samples)
print("Acceptance rate", mcmc_variable._acceptance_rate)

# %%
# Now plot the posterior samples with the 2D Marginals of the posterior. Note
# do not do this with the numerical model as this would take an eternity due
# to the cost of evaluating the numerical model, which is much higher relative
# to the cost of running the surrogate.
fig, axs = mcmc_variable.plot_2d_marginals(post_samples, unbounded_alpha=0.999)
if savefig:
    plt.savefig("posterior-samples.pdf", bbox_inches="tight")

# %%
# In the Bayesian inference above we used a fixed number of observations
# at randomly chosen spatial locations. However choosing observation locations
# is usually a poor idea. Not all observations can reduce the uncertainty
# in the parameters equally. Here we use Bayesian optimal experimental design
# to choose the 3 best design locations from the previously observed 10 pretending
# that we do not know the value of the observations.
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# plot a single solution to the PDE before overlaying the designs
sol_array = inv_benchmark.observation_model().forward_solve(map_sample)
sol = (
    inv_benchmark.observation_model().physics().solution_from_array(sol_array)
)
sol.plot(ax, 50, levels=30)

ndesign = 3
inloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
    bkd.full((inv_benchmark.nobservations(),), noise_stdev**2)[:, None],
    backend=bkd,
)
brute_oed = BruteForceKLBayesianOED(inloop_loglike)
oed_data_gen = BayesianOEDDataGenerator(bkd)
(
    outloop_samples,
    outloop_quad_weights,
    inloop_samples,
    inloop_quad_weights,
) = oed_data_gen.prepare_simulation_inputs(
    brute_oed, inv_benchmark.prior(), "MC", 100, "MC", 100
)

brute_oed.set_data_from_model(
    inv_benchmark.observation_model(),
    inv_benchmark.prior(),
    outloop_samples,
    outloop_quad_weights,
    inloop_samples,
    inloop_quad_weights,
)
opt_design = brute_oed.compute(ndesign)


design_candidates = inv_benchmark.observation_design()
selected_candidates = design_candidates[:, opt_design]
print(selected_candidates)
ax.plot(design_candidates[0, :], design_candidates[1, :], "rs", ms=16)
ax.plot(selected_candidates[0, :], selected_candidates[1, :], "ko")
if savefig:
    plt.savefig("oed-selected-design.pdf")


# %%
# Note that typically optimal
# experimental design (OED) would be used before conducting Bayesian inference.
# However, because understanding of Bayesian inference is needed to understand
# Bayesian OED we reversed the order. OED is much more expensive than a single
# Bayesian calibration because it requires solving many calibration problems.
# So typically we do not solve the calibration problems in the OED procedure
# to the same degree of accuracy as a final calibration. The accuracy of the
# calibrations used by OED must only be sufficient to distinguish between designs.
# This accuracy is typically much lower than the accuracy required in
# estimates of uncertainty in the parameters or predictions needed for decision making tasks such as risk assessment.

# %%
# Here we will set up a related benchmark to the one we have been using,
# which can be used to demonstrate the forward propagation of uncertainty.
# This benchmark uses the steady state solution of the advection diffusion,
# obtained with a constant addition of a tracer into the domain at a single
# source model as initial condition. A pump at another locations is then activated
# to extract the tracer from the domain. The benchmark quantity of interest
# measures the change of the tracer concentration in a subomain.
# The benchmark provides models of varying cost
# and accuracy that use different discretizations of the spatial PDE mesh
# and number of time steps which can be used with multi-fideilty methods.
# To setup the benchmark use the following
qoi_models, qoi_model_names = inv_benchmark.qoi_models()
# turn on work tracking
for model in qoi_models:
    model.activate_model_data_base()

# %%
# Here we will use Multi-fidelity statistical estimation to compute the
# mean value of the QoI to account for the uncertainty in the KLE cofficients.
# So first we must compute the covariance between the QoI returned by
# each of our models. We use samples from the prior. But try using samples from
# the posterior
# pilot_samples = post_samples[:, :20]
npilot_samples = 20
pilot_samples = inv_benchmark.prior().rvs(npilot_samples)
pilot_values_per_model = [model(pilot_samples) for model in qoi_models]
stat = multioutput_stats["mean"](qoi_models[0].nqoi(), backend=bkd)
pilot_quantities = stat.compute_pilot_quantities(pilot_values_per_model)
stat.set_pilot_quantities(*pilot_quantities)

# %%
# By using a WorkTrackingModel we can extract the median costs
# of evaluatin each model which is needed to predict the
# error of the multi-fidelity estimate of the mean which we can
# compare to a prediction of the single fidelity estimate that only uses
# the highest fidelity model.
model_costs = bkd.array(
    [model.work_tracker().average_wall_time("val") for model in qoi_models]
)
# make costs in terms of fraction of cost of high-fidelity evaluation
print(model_costs)
model_costs = model_costs / model_costs[0]
print(model_costs)

# %%
# Now visualize the correlation between the models and their computational
# cost relative to the highest-fidelity model cost
fig, axs = plt.subplots(1, 2, figsize=(2 * 8, 6))
mf.plot_correlation_matrix(
    mf.covariance_to_correlation(stat.pilot_covariance(), bkd=bkd),
    ax=axs[0],
    model_names=qoi_model_names,
)
mf.plot_model_costs(model_costs, ax=axs[1])
axs[0].set_title(mathrm_label("Model covariances"))
_ = axs[1].set_title(mathrm_label("Relative model costs"))

# %%
# Now find the best multi-fidelity estimator among all available options
# Note, the exact predicted variance will change from run to run even with the
# same seed because the computational time measured will change slightly
# for each run
best_est = mf.get_estimator(
    "gmf", stat, model_costs, tree_depth=3, allow_failures=True, max_nmodels=3
)

target_cost = 1e2
best_est.allocate_samples(target_cost)
print(
    "Predicted variance",
    best_est._covariance_from_npartition_samples(
        best_est._rounded_npartition_samples
    ),
)

# %%
# Now we can plot the relative performance of the single and multi-fidelity
# estimates of the mean before requiring any additional model evaluations
fig, axs = plt.subplots(1, 2, figsize=(2 * 8, 6))
print(best_est)
mf.plot_estimator_variance_reductions([best_est], ["Best"], axs[0])
mf.plot_estimator_sample_allocation_comparison(
    [best_est], qoi_model_names, axs[1]
)
if savefig:
    plt.savefig("acv-variance-reduction.pdf")

# %%
# It is clear that the multi-fidelity estimator will be more computationally
# efficient. Once the user
# is ready to actually estimate the mean QoI they can use
samples_per_model = best_est.generate_samples_per_model(
    inv_benchmark.prior().rvs
)
best_models = [qoi_models[idx] for idx in best_est._best_model_indices]
values_per_model = [
    fun(samples) for fun, samples in zip(best_models, samples_per_model)
]
mean = best_est(values_per_model)
print("Mean QoI", mean)

# %%
# References
# ^^^^^^^^^^
# .. [PYAPPROX2023] `Jakeman J.D., PyApprox: A software package for sensitivity analysis, Bayesian inference, optimal experimental design, and multi-fidelity uncertainty quantification and surrogate modeling. (2023) <https://doi.org/10.1016/j.envsoft.2023.105825>`_
