"""
Bayesian Risk-Aware Optimal Experimental Design
===============================================

This tutorial demonstrates how to judiciously select how to collect experimental data using risk-aware Bayesian optimal experimental design (OED) for prediction.

Load the Modules Needed
-----------------------
"""

# Load modules from dependencies
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load modules from pyapprox
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.benchmarks import LotkaVolterraOEDBenchmark
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.expdesign.optbayes import (
    BayesianOEDForPrediction,
    IndependentGaussianOEDInnerLoopLogLikelihood,
    OEDStandardDeviationMeasure,
    OEDEntropicDeviationMeasure,
    NoiseStatistic,
)
from pyapprox.optimization.minimize import (
    SampleAverageMean,
    SampleAverageMeanPlusStdev,
    SampleAverageEntropicRisk,
)

# %% Setup the Model to Generate the Simulation Data
# -------------------------------------------------
# Users can load in their own simulation data to create an OED.
# Here we show how to construct the data needed. First we must
# setup the simulation model


# Set the random seed for reproducibility
np.random.seed(2)
# Setup the benchmark
benchmark = LotkaVolterraOEDBenchmark(backend=bkd)
# Extract the model from the benchmark
obs_model = benchmark.model()
# Extract the prediction model from the benchmark
pred_model = benchmark.prediction_model()
# Extract the random variable
prior = benchmark.variable()

# %%
# Plot the ODE states for a Nominal Simulation
# --------------------------------------------
# Plot the ODE evolution at a single relization of the random model parameters.
# Also plot the noisless observations which we collect at two of the three states.

# Run the simulation
sample = prior.rvs(1)
model_obs_sol = obs_model.forward_solve(sample)[0]
# Extract out the noiseless observations
observations = obs_model(sample).reshape((2, -1))
# Plot the evolution of each of the three ODE states
axs = plt.subplots(1, 4, figsize=(4 * 8, 6), sharey=True)[1]
axs[0].plot(benchmark.solution_times(), model_obs_sol.T)
# Plot the observations
axs[0].plot(
    benchmark.observation_times().T, observations.T, "o", label="Observations"
)
# Plot the predictions of the unobserved state
predictions = pred_model(sample)
axs[0].plot(
    benchmark.prediction_times().T, predictions.T, "x", label="Predictions"
)
axs[0].legend()
# Plot different trajectories of each state at various samples
for sample in prior.rvs(100).T:
    model_obs_sol = obs_model.forward_solve(sample[:, None])[0]
    for ii in range(3):
        axs[ii + 1].plot(benchmark.solution_times(), model_obs_sol[ii])
# plt.show()

# %%
# Define the likelihood
# ---------------------
# Use a Gaussian likelihood with a known digonal covariance, i.e. the
# observations have independent additive zero-mean Gaussian noise

# Define the diagonal of the noise covariance,
# We assume we have the option to sample two of three three states at
# one or more times.
noise_std = 0.2
noise_cov_diag = bkd.full((obs_model.nqoi(), 1), noise_std**2)
# Initialize the likelihood function. We must use a likelihood tailored
# to OED that can handle inner and outerloop calculations needed
# to compute the utility of a design
innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
    noise_cov_diag, backend=bkd
)

# %%
# Initialize the OED Problem
# --------------------------

# Define the OED problem
pred_oed = BayesianOEDForPrediction(innerloop_loglike)

# %% Generate The Simulation Data
# -------------------------------
# Generate the simulation data needed to perform OED. Here we use a model of a predator-prey system.
# We must create observations for the outerloop and simulation data for the innerloop.

# Generate the outerloop observations
nouterloop_samples = 5000
prior_data_variable = IndependentMarginalsVariable(
    prior.marginals()
    + [stats.norm(0, bkd.sqrt(variance)) for variance in noise_cov_diag[:, 0]],
    backend=bkd,
)

# Generate the outerloop samples needed to compute the outerloop observations
outerloop_samples = prior_data_variable.rvs(nouterloop_samples)
outerloop_quad_weights = bkd.full(
    (nouterloop_samples, 1), 1.0 / nouterloop_samples
)

# Generate the shapes of the likelihood, e.g. the model predictions,
# for the inner OED loop
ninnerloop_samples = int(math.sqrt(nouterloop_samples))
# ninnerloop_samples = nouterloop_samples
# Generate the samples to evaluate the model
innerloop_samples = prior.rvs(ninnerloop_samples)
innerloop_quad_weights = bkd.full(
    (ninnerloop_samples, 1), 1.0 / ninnerloop_samples
)

# Assume MC quadrature for prediction space
qoi_quad_weights = bkd.full((pred_model.nqoi(), 1), 1.0 / pred_model.nqoi())

# Specify the noise statistic taken over all realizations of the data.
# We will use the expectation
noise_stat = NoiseStatistic(SampleAverageMean(bkd))
# Specify the inner deviation measure
deviation_measure = OEDStandardDeviationMeasure(pred_model.nqoi(), bkd)
# Specify the risk measure over the prediction space
risk_measure = SampleAverageMeanPlusStdev(1, bkd)

# Pass the data to the OED problem
pred_oed.set_data_from_model(
    obs_model,
    pred_model,
    prior,
    outerloop_samples,
    outerloop_quad_weights,
    innerloop_samples,
    innerloop_quad_weights,
    qoi_quad_weights,
    deviation_measure,
    risk_measure,
    noise_stat,
)
pred_oed.set_optimizer(pred_oed.default_optimizer(verbosity=3))

# %%
# Compute the OED
# ---------------
# Use the double loop algorithm to construct the OED

design_weights = pred_oed.compute()

# %%
# Plot the OED
# -----------
fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
# Reshape weights so that each row correspond to one of the OED states
design_weights = design_weights.reshape((2, -1))
axs[0].bar(benchmark.observation_times()[0], design_weights[0])
axs[0].set_title("State 1")
axs[0].set_ylabel("Design Weight")
axs[1].bar(benchmark.observation_times()[1], design_weights[1])
axs[1].set_title("State 3")
[ax.set_xlabel("Time (seconds)") for ax in axs]
fig.suptitle("Entropic Deviation")

# %%
# Compute an OED that encodes different risk preferences

# Change the deviation measure
deviation_measure = OEDEntropicDeviationMeasure(pred_model.nqoi(), 1.0, bkd)
# Specify the risk measure over the prediction space
# Initialize the OED problem
pred_oed_2 = BayesianOEDForPrediction(innerloop_loglike)
# Pass the data to the OED problem
pred_oed_2.set_data_from_model(
    obs_model,
    pred_model,
    prior,
    outerloop_samples,
    outerloop_quad_weights,
    innerloop_samples,
    innerloop_quad_weights,
    qoi_quad_weights,
    deviation_measure,
    risk_measure,
    noise_stat,
)
pred_oed_2.set_optimizer(pred_oed.default_optimizer(verbosity=3))
design_weights_2 = pred_oed_2.compute()
fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
# Reshape weights so that each row correspond to one of the OED states
design_weights_2 = design_weights_2.reshape((2, -1))
axs[0].bar(benchmark.observation_times()[0], design_weights_2[0])
axs[0].set_title("State 1")
axs[0].set_ylabel("Design Weight")
axs[1].bar(benchmark.observation_times()[1], design_weights_2[1])
axs[1].set_title("State 3")
[ax.set_xlabel("Time (seconds)") for ax in axs]
fig.suptitle("Entropic Deviation")

design_weights = design_weights.reshape((-1, 1))
design_weights_2 = design_weights_2.reshape((-1, 1))
print(pred_oed.objective()(design_weights))
print(pred_oed.objective()(design_weights_2))

print(pred_oed_2.objective()(design_weights))
print(pred_oed_2.objective()(design_weights_2))

plt.show()
