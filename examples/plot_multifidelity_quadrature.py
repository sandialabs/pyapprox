"""
Multi-fidelity Quadrature
-------------------------
The following provides an example of how to use multi-fidelity quadrature, e.g. multilevel Monte Carlo, control variates to estimate the mean of a high-fidelity model from an ensemble of related models of varying cost and accuracy. A set of detailed tutorials on this subject can be found in the tutorials section, e.g. :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_monte_carlo.py`.
"""
#%%
#Load the necessary modules
import numpy as np
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox import multifidelity
from pyapprox import interface
# set seed for reproducibility
np.random.seed(1)

#%%
#First define an ensemble of models using :py:class:`~pyapprox.benchmarks.setup_benchmark`, see  :mod:`pyapprox.benchmarks`.
benchmark = setup_benchmark(
    "tunable_model_ensemble", theta1=np.pi/2*.95, shifts=[.1, .2])
model_ensemble = interface.ModelEnsemble(benchmark.funs)
hf_mean = benchmark.mean[0]

#%%
#Initialize a multifidelity estimator. This requires an estimate of the covariance between the models and the model costs and the random variable representing the model inputs

# generate pilot samples to estimate correlation
npilot_samples = int(1e2)
# The models are trivial to evaluate so make up model costs
model_costs = 10.**(-np.arange(3))

est_name = "mlblue"
stat_name = "mean"
cov = multifidelity.estimate_model_ensemble_covariance(
    npilot_samples, benchmark.variable.rvs, model_ensemble,
    model_ensemble.nmodels)[0]
est = multifidelity.get_estimator(
    est_name, stat_name, 1, model_costs, cov)

#%%
#Define a target cost and determine the optimal number of samples to allocate to each model
target_cost = 1000
est.allocate_samples(target_cost)
args = [benchmark.variable] if est_name == "mlblue" else []
samples_per_model = est.generate_samples_per_model(
    benchmark.variable.rvs)
best_models = [benchmark.funs[idx] for idx in est._best_model_indices]
values_per_model = [
    fun(samples) for fun, samples in zip(best_models, samples_per_model)]
mf_mean = est(values_per_model)

print("Multi-fidelity mean", mf_mean)
print("Exact high-fidelity mean", hf_mean)
print("Multi-fidelity estimator variance",
      est._covariance_from_npartition_samples(est._rounded_npartition_samples))

#%%
#Questions
#^^^^^^^^^
#Compare the multi-fidelity mean to the single-fidelity means using only one model
#
#Increase the target cost
#
#Change the correlation between the models by varying the theta1 argument to setup benchmarks
#
#Change the estimator (via est_name). Names of the available estimators can be printed via

print(multifidelity.factory.multioutput_estimators.keys())

#Change the statistic computed (via stat_name). Names of the implemented statistics can be printed via

print(multifidelity.factory.multioutput_stats.keys())
