r"""
End-to-End Model Analysis
=========================
This tutorial describes how to use each of the major model analyses in Pyapprox
following the exposition in [PYAPPROX2022]_.

First lets load all the necessary modules and set the random seeds for reproducibility.
"""
from scipy import stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
import torch
import time
from pyapprox.util.visualization import mathrm_label, mathrm_labels
from pyapprox.variables import (
    IndependentMarginalsVariable, print_statistics, AffineTransform)
from pyapprox.benchmarks import setup_benchmark, list_benchmarks
from pyapprox.interface.wrappers import (
    ModelEnsemble, TimerModel, WorkTrackingModel,
    evaluate_1darray_function_on_2d_array)
from pyapprox.surrogates import adaptive_approximate
from pyapprox.analysis.sensitivity_analysis import (
    run_sensitivity_analysis, plot_sensitivity_indices)
from pyapprox.bayes.metropolis import (
    loglike_from_negloglike, plot_unnormalized_2d_marginals)
from pyapprox.bayes.metropolis import MetropolisMCMCVariable
from pyapprox.expdesign.bayesian_oed import get_bayesian_oed_optimizer
from pyapprox import multifidelity
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(2023)
torch.manual_seed(2023)

#%%
#The tutorial can save the figures to file if desired. If you do want the plots
#set savefig=True
# savefig = True
savefig = False

#%%
#The following code shows how to create and sample from two independent uniform random variables defined on :math:`[-2, 2]`. We use uniform variables here, but any marginal from the scipy.stats module can be used.
nsamples = 30
univariate_variables = [stats.uniform(-2, 4), stats.uniform(-2, 4)]
variable = IndependentMarginalsVariable(univariate_variables)
samples = variable.rvs(nsamples)
print_statistics(samples)

#%%
#PyApprox supports various types of variable transformations. The following code
#shows how to use an affinte transformation to map samples from variables
#to samples from the variable's canonical form.
var_trans = AffineTransform(variable)
canonical_samples = var_trans.map_to_canonical(samples)
print_statistics(canonical_samples)

#%%
#Pyapprox provides many utilities for interfacing with complex numerical codes.
#The following shows how to wrap a model and store the wall time required
#to evaluate each sample in a set. First define a function
#with a random execution time that takes in one sample at a time, i.e. a
#1D array. Then wrap that model so that multiple samples can be evaluated at
#once.
def fun_pause_1(sample):
    assert sample.ndim == 1
    time.sleep(np.random.uniform(0, .05))
    return np.sum(sample**2)


def pyapprox_fun_1(samples):
    return evaluate_1darray_function_on_2d_array(fun_pause_1, samples)


#%%
#Now wrap the latter function and run it while tracking
#their execution times. The last print statement
#prints the median execution time of the model.
timer_model = TimerModel(pyapprox_fun_1)
model = WorkTrackingModel(timer_model)
values = model(samples)
print(model.work_tracker())

#%%
#Other wrappers available in PyApprox include those for running multiple models
#at once, useful for multi-fidelity methods, wrappers that fix a subset of inputs
#to user specified values, wrappers that only return a subset of all
#possible model ouputs, and wrappers for evaluating samples in parallel.

#%%
#Pyapprox provide numerous benchmarks for verifying, validating and comparing
#model analysis algorithms. The following list the names of all benchmarks and
#then creates a benchmark that can be used to test the creation of surrogates,
#Bayesian inference, and optimal experimental design. This benchmark requires
#determining the true coefficients of the Karhunene Loeve expansion (KLE)
#used to characterize the uncertain diffusivity field of an advection
#diffusion equation. See documentation of the benchmark for more details).
print(list_benchmarks())
noise_stdev = 1 #1e-1
inv_benchmark = setup_benchmark(
    "advection_diffusion_kle_inversion", kle_nvars=3,
    noise_stdev=noise_stdev, nobs=5, kle_length_scale=0.5)
print(inv_benchmark.keys())

#%%
#The following plots the modes of the KLE
fig, axs = plt.subplots(
    1, inv_benchmark.KLE.nterms, figsize=(8*inv_benchmark.KLE.nterms, 6))
for ii in range(inv_benchmark.KLE.nterms):
    inv_benchmark.mesh.plot(inv_benchmark.KLE.eig_vecs[:, ii:ii+1], 50,
                            ax=axs[ii])

#%%
#PyApprox provides many popular methods for constructing surrogates
#that once constructed can be evaluated in place of a computaionally
#expensive simulation model in model analyses. The following code creates
#a Gaussian process (GP) surrogate. The function used to construct the surrogate
#takes a callback which is evaluated each time the adaptive surrogate is refined.
#Here we use to compute the error of the surrogate as it is constructed using
#validation data. Uncomment the code to use a polynomial based surrogate instead
#of a GP. The user does not have to change any subsequent code
validation_samples = inv_benchmark.variable.rvs(100)
validation_values = inv_benchmark.negloglike(validation_samples)
nsamples, errors = [], []
def callback(approx):
    nsamples.append(approx.num_training_samples())
    error = np.linalg.norm(
        approx(validation_samples)-validation_values, axis=0)
    error /= np.linalg.norm(validation_values, axis=0)
    errors.append(error)

approx_result = adaptive_approximate(
    inv_benchmark.negloglike, inv_benchmark.variable, "gaussian_process",
    {"max_nsamples": 50, "ncandidate_samples": 2e3, "verbose": 0,
     "callback": callback, "kernel_variance": 400})

# approx_result = adaptive_approximate(
#     inv_benchmark.negloglike, inv_benchmark.variable, "polynomial_chaos",
#     {"method": "leja", "options": {
#         "max_nsamples": 100, "ncandidate_samples": 3e3, "verbose": 0,
#         "callback": callback}})
approx = approx_result.approx

#%%
#We can plot the errors obtained from the callback with
ax = plt.subplots(figsize=(8, 6))[1]
ax.loglog(nsamples, errors, "o-")
ax.set_xlabel(mathrm_label("No. Samples"))
ax.set_ylabel(mathrm_label("Error"))
ax.set_xticks([10, 25, 50])
ax.set_yticks([0.3, 0.75, 1.5])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.minorticks_off()
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
#ax.tick_params(axis='y', which='minor', bottom=False)
if savefig:
    plt.savefig("gp-error-plot.pdf")

#%%
#Now we will perform a sensitivity analysis. Specifically we compute
#variance based sensitivity indices that measure the impact of each KLE mode
#on the mismatch between the observed data and the model predictions.
#We use the negative log likelihood to characterize this mismatch.
#Here we have used the surrogate to speed up the computation of the sensitivity
#indices. Uncomment the commented code to use the numerical model. Note
#the drastic increase in computational cost. Warning: using the numerical model
#will take many minutes. The plots in the figure, generated from
#left to right are: main effect, largest Sobol indices and total effect indices.
sa_result = run_sensitivity_analysis(
    "surrogate_sobol", approx, inv_benchmark.variable)
# sa_result = run_sensitivity_analysis(
#     "sobol", benchmark.negloglike, inv_benchmark.variable)
axs = plot_sensitivity_indices(
    sa_result)[1]
if savefig:
    plt.savefig("gp-sa-indices.pdf", bbox_inches="tight")

#%%
#Now we will use the surrogate with Bayesian inference to learn the
#coefficients of the KL. Specifically we will draw a set of samples from
#the posterior distribution of the KLE given the observed data provided
#in the benchmark.
#
#But First we will improve the accuracy of the surrogate
#and print out the error which can be compared to the errors previously plotted.
#The error of the original surrogate was kept low to demonstrate the ability
#to quantify error in the sensitivity indices from using a surrogate.
approx.refine(100)
error = np.linalg.norm(
    approx(validation_samples)-validation_values, axis=0)
error /= np.linalg.norm(validation_values, axis=0)
print("Surrogate", error)

#%%
#Now create a MCMCVariable to sample from the posterior. The benchmark
#has already formulated the negative log likelihood that is needed. Here
#we will use PyApprox's native delayed rejection adaptive metropolis (DRAM)
#sampler.
#
#Uncomment the commented code to use the numerical model instead of the surrogate
#with the MCMC algorithm. Again note the significant increase in computational
#time
npost_samples = 200
loglike = partial(loglike_from_negloglike, approx)
# loglike = partial(loglike_from_negloglike, inv_benchmark.negloglike)
mcmc_variable = MetropolisMCMCVariable(
    inv_benchmark.variable, loglike, method_opts={"cov_scaling": 1})
print(mcmc_variable)
map_sample = mcmc_variable.maximum_aposteriori_point()
print("Computed Map Point", map_sample[:, 0])
post_samples = mcmc_variable.rvs(npost_samples)
print("Acceptance rate", mcmc_variable._acceptance_rate)

#%%
#Now plot the posterior samples with the 2D Marginals of the posterior. Note
#do not do this with the numerical model as this would take an eternity due
#to the cost of evaluating the numerical model, which is much higher relative
#to the cost of running the surrogate.
plot_unnormalized_2d_marginals(
    mcmc_variable._variable, mcmc_variable._loglike, nsamples_1d=50,
    plot_samples=[
        [post_samples, {"alpha": 0.3, "c": "orange"}],
        [map_sample, {"c": "k", "marker": "X", "s": 100}]],
    unbounded_alpha=0.999)
if savefig:
    plt.savefig("posterior-samples.pdf", bbox_inches="tight")

#%%
#In the Bayesian inference above we used a fixed number of observations
#at randomly chosen spatial locations. However choosing observation locations
#is usually a poor idea. Not all observations can reduce the uncertainty
#in the parameters equally. Here we use Bayesian optimal experimental design
#to choose the 3 best design locations from the previously observed 10 pretending
#that we do not know the value of the observations.
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#plot a single solution to the PDE before overlaying the designs
inv_benchmark.mesh.plot(
    inv_benchmark.obs_fun._fwd_solver.solve()[:, None], 50, ax=ax)

ndesign = 3
design_candidates = inv_benchmark.mesh.mesh_pts[:, inv_benchmark.obs_indices]
ndesign_candidates = design_candidates.shape[1]
oed = get_bayesian_oed_optimizer(
    "kl_params", ndesign_candidates, inv_benchmark.obs_fun, noise_stdev,
    inv_benchmark.variable, max_ncollected_obs=ndesign)
oed_results = []
for step in range(ndesign):
    results_step = oed.update_design()[1]
    oed_results.append(results_step)
selected_candidates = design_candidates[:, np.hstack(oed_results)]
print(selected_candidates)
ax.plot(design_candidates[0, :], design_candidates[1, :], "rs")
ax.plot(selected_candidates[0, :], selected_candidates[1, :], "ko")
if savefig:
    plt.savefig("oed-selected-design.pdf")


#%%
#Note that typically optimal
#experimental design (OED) would be used before conducting Bayesian inference.
#However, because understanding of Bayesian inference is needed to understand
#Bayesian OED we reversed the order. OED is much more expensive than a single
#Bayesian calibration because it requires solving many calibration problems.
#So typically we do not solve the calibration problems in the OED procedure
#to the same degree of accuracy as a final calibration. The accuracy of the
#calibrations used by OED must only be sufficient to distinguish between designs.
#This accuracy is typically much lower than the accuracy required in
#estimates of uncertainty in the parameters or predictions needed for decision making tasks such as risk assessment.

#%%
#Here we will set up a related benchmark to the one we have been using,
#which can be used to demonstrate the forward propagation of uncertainty.
#This benchmark uses the steady state solution of the advection diffusion,
#obtained with a constant addition of a tracer into the domain at a single
#source model as initial condition. A pump at another locations is then activated
#to extract the tracer from the domain. The benchmark quantity of interest
#measures the change of the tracer concentration in a subomain.
#The benchmark provides models of varying cost
#and accuracy that use different discretizations of the spatial PDE mesh
#and number of time steps which can be used with multi-fideilty methods.
#To setup the benchmark use the following
fwd_benchmark = setup_benchmark(
    "multi_index_advection_diffusion",
    kle_nvars=inv_benchmark.variable.num_vars(), kle_length_scale=0.5,
    time_scenario=True)
model = WorkTrackingModel(
    TimerModel(fwd_benchmark.model_ensemble), num_config_vars=1)

#%%
#Here we will use Multi-fidelity statistical estimation to compute the
#mean value of the QoI to account for the uncertainty in the KLE cofficients.
#So first we must compute the covariance between the QoI returned by
#each of our models. We use samples from the posterior. But uncommenting
#the code below will use samples from the prior.
npilot_samples = 20
generate_samples = inv_benchmark.variable.rvs # for sampling from prior
# generate_samples = post_samples 
cov = multifidelity.estimate_model_ensemble_covariance(
    npilot_samples, generate_samples, model,
    fwd_benchmark.model_ensemble.nmodels)[0]
print(cov)

#%%
#By using a WorkTrackingModel we can extract the median costs
#of evaluatin each model which is needed to predict the
#error of the multi-fidelity estimate of the mean which we can
#compare to a prediction of the single fidelity estimate that only uses
#the highest fidelity model.
model_ids = np.asarray([np.arange(fwd_benchmark.model_ensemble.nmodels)])
model_costs = model.work_tracker(model_ids)
# make costs in terms of fraction of cost of high-fidelity evaluation
model_costs /= model_costs[0]

#%%
#Now visualize the correlation between the models and their computational
#cost relative to the highest-fidelity model cost
fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
multifidelity.plot_correlation_matrix(
    multifidelity.get_correlation_from_covariance(cov), ax=axs[0])
multifidelity.plot_model_costs(model_costs, ax=axs[1])
axs[0].set_title(mathrm_label("Model covariances"))
axs[1].set_title(mathrm_label("Relative model costs"))

#%%
#Now find the best multi-fidelity estimator among all available option
#Note, the exact predicted variance will change from run to run even with the
#same seed because the computational time measured will change slightly
#for each run
best_est, best_model_indices = (
    multifidelity.get_best_models_for_acv_estimator(
        "acvgmfb", cov, model_costs, inv_benchmark.variable, 1e2, max_nmodels=3,
        init_kwargs={"tree_depth": 4}))
target_cost = 1000
best_est.allocate_samples(target_cost)
print("Predicted variance", best_est.optimized_variance)

#%%
#Now we can plot the relative performance of the single and multi-fidelity
#estimates of the mean before requiring any additional model evaluations
hf_cov, hf_cost = cov[:1, :1], model_costs[:1]
estimators = [
    multifidelity.get_estimator(
        "mc", hf_cov, hf_cost, inv_benchmark.variable),
    best_est]
target_costs = np.array([1e1, 1e2, 1e3, 1e4], dtype=int)
optimized_estimators = multifidelity.compare_estimator_variances(
    target_costs, estimators)
est_labels = mathrm_labels(["MC", "ACV"])
fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
multifidelity.plot_estimator_variances(
    optimized_estimators, est_labels, axs[0],
    ylabel=mathrm_label("Relative Estimator Variance"))
axs[0].set_xlim(target_costs.min(), target_costs.max())
nmodels = cov.shape[0]
model_labels = [
    r"$f_{%d}$" % ii for ii in np.arange(nmodels)[best_model_indices]]
multifidelity.plot_acv_sample_allocation_comparison(
    optimized_estimators[1], model_labels, axs[1])
if savefig:
    plt.savefig("acv-variance-reduction.pdf")

#%%
#It is clear that the multi-fidelity estimator will be more computationally
#efficient for multiple computational budgets (target costs). Once the user
#is ready to actually estimate the mean QoI they can use
target_cost = 10
best_est.allocate_samples(target_cost)
best_model_ensemble = ModelEnsemble(
    [fwd_benchmark.model_ensemble.functions[ii] for ii in best_model_indices])
samples, values = best_est.generate_data(best_model_ensemble)
mean = best_est(values)
print("Mean QoI", mean)

plt.show()

#%%
#References
#^^^^^^^^^^
#.. [PYAPPROX2022] `Jakeman J.D., PyApprox: Enabling efficient model analysis. (2022) <https://www.osti.gov/biblio/1879614>`_
