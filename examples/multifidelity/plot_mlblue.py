"""
Multi-fidelity Statistical Estimation
-------------------------------------
The following provides an example of how to use multi-fidelity quadrature, e.g. multilevel Monte Carlo, control variates to estimate the mean of a high-fidelity model from an ensemble of related models of varying cost and accuracy. A set of detailed tutorials on this subject can be found in the tutorials section, e.g. :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_monte_carlo.py`.
"""

# %%
# Load the necessary modules
import numpy as np
from pyapprox.benchmarks.multifidelity_benchmarks import (
    TunableModelEnsembleBenchmark,
)
from pyapprox.multifidelity.factory import (
    multioutput_stats,
    multioutput_estimators,
)
from pyapprox.multifidelity.groupacv import MLBLUEEstimator
from pyapprox.util.backends.numpy import NumpyMixin

# set seed for reproducibility
np.random.seed(1)
bkd = NumpyMixin

# %%
# First define an ensemble of models using using a benchmark, see  :mod:`pyapprox.benchmarks`.
benchmark = TunableModelEnsembleBenchmark(
    theta1=np.pi / 2 * 0.95, backend=bkd, shifts=[0.1, 0.2]
)

# %%
# Initialize a multifidelity estimator. This requires an estimate of the covariance between the models and the model costs and the random variable representing the model inputs

# generate pilot samples to estimate correlation
npilot_samples = int(1e2)

stat = multioutput_stats["mean"](benchmark.nqoi(), bkd)
pilot_samples = benchmark.prior().rvs(npilot_samples)
pilot_values = [model(pilot_samples) for model in benchmark.models()]
stat.set_pilot_quantities(*stat.compute_pilot_quantities(pilot_values))


# %%
# Define a target cost and determine the optimal number of samples to allocate to each model
target_cost = 1000
mlb_est = MLBLUEEstimator(stat, benchmark.costs(), reg_blue=0)
mlb_est.allocate_samples(target_cost, stat.min_nsamples())


# %%
# Construct the estimator
samples_per_model = mlb_est.generate_samples_per_model(benchmark.prior().rvs)
values_per_model = [
    model(samples)
    for model, samples in zip(benchmark.models(), samples_per_model)
]
mlb_mean = mlb_est(values_per_model)
hf_mean = benchmark.mean()[0]
print("Multi-fidelity mean", mlb_mean)
print("Exact high-fidelity mean", hf_mean)
print("Multi-fidelity estimator variance", mlb_est.optimized_covariance())

# %%
# Excercises
# ^^^^^^^^^^
# Compare the multi-fidelity mean to the single-fidelity means using only one model
#
# Increase the target cost
#
# Change the correlation between the models by varying the theta1 argument to setup benchmarks
#
# Change the estimator (via est_name). Names of the available estimators can be printed via

print(multioutput_estimators.keys())

# Change the statistic computed (via stat_name). Names of the implemented statistics can be printed via

print(multioutput_stats.keys())
