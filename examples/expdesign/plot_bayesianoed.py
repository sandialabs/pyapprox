"""
Bayesian Optimal Experimental Design
====================================

This tutorial demonstrates how to judiciously select how to collect experimental data using Bayesian optimal experimental design (OED).

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
    KLBayesianOED,
    IndependentGaussianOEDInnerLoopLogLikelihood,
)

# Set the seed for reproducibility
np.random.seed(1)

# %% Setup the Model to Generate the Simulation Data
# -------------------------------------------------
# Users can load in their own simulation data to create an OED.
# Here we show how to construct the data needed. First we must
# setup the simulation model


# Set the random seed for reproducibility
np.random.seed = 2
# Setup the benchmark
benchmark = LotkaVolterraOEDBenchmark(backend=bkd)
# Extract the model from the benchmark
obs_model = benchmark.model()
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
plt.plot(benchmark.solution_times(), model_obs_sol.T)
# Plot the observations
plt.plot(benchmark.observation_times().T, observations.T, "o")
# plt.show()

# %%
# Define the likelihood
# ---------------------
# Use a Gaussian likelihood with a known digonal covariance, i.e. the
# observations have independent additive zero-mean Gaussian noise

# Define the diagonal of the noise covariance,
# We assume we have the option to sample two of three three states at
# one or more times.
noise_std = 0.1
noise_cov_diag = bkd.full((obs_model.nqoi(), 1), noise_std**2)
# Initialize the likelihood function. We must use a likelihood tailored
# to OED that can handle inner and outerloop calculations needed
# to compute the utility of a design
innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
    noise_cov_diag, backend=bkd
)

# %% Generate The Simulation Data
# -------------------------------
# Generate the simulation data needed to perform OED. Here we use a model of a predator-prey system.
# We must create observations for the outerloop and simulation data for the innerloop.

# Generate the outerloop observations
nouterloop_samples = 100000
# Define the data distribution, e.g. the joint distribution of the prior
# and the noise
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
# Generate the noiseless observations using the numerical model
# We must discard the samples of the noise here when evaluating the model
outerloop_shapes_samples = outerloop_samples[: prior.nvars()]
outerloop_shapes = obs_model(outerloop_shapes_samples).T
# Generate the observations from the liklihood, e.g. add the noise
outerloop_loglike = innerloop_loglike.outerloop_loglike()
obs = outerloop_loglike.rvs_from_shapes(outerloop_shapes)
outerloop_loglike.set_observations_and_shapes(obs, outerloop_shapes)

# Generate the shapes of the likelihood, e.g. the model predictions,
# for the inner OED loop
ninnerloop_samples = int(math.sqrt(nouterloop_samples))
# Generate the samples to evaluate the model
innerloop_samples = prior.rvs(ninnerloop_samples)
innerloop_quad_weights = bkd.full(
    (ninnerloop_samples, 1), 1.0 / ninnerloop_samples
)
# Simulate the model and pass them to the inner likelihood
innerloop_shapes = obs_model(innerloop_samples).T
innerloop_loglike.set_observations_and_shapes(obs, innerloop_shapes)


# %%
# Initialize the OED Problem
# --------------------------
# Setup the OED problem using the data provided.

# Define the OED problem
kl_oed = KLBayesianOED(innerloop_loglike)
# Dass the data to the OED problem
kl_oed.set_data(
    innerloop_loglike.outerloop_loglike().shapes(),
    outerloop_samples,
    outerloop_quad_weights,
    innerloop_loglike.shapes(),
    innerloop_quad_weights,
)

# %%
# Compute the OED
# ---------------
# Use the double loop algorithm to construct the OED

design_weights = kl_oed.compute()

# %%
# Plot the OED
# -----------
axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)[1]
# Reshape weights so that each row correspond to one of the OED states
design_weights = design_weights.reshape((2, -1))
axs[0].bar(benchmark.observation_times()[0], design_weights[0])
axs[0].set_title("State 1")
axs[0].set_ylabel("Design Weight")
axs[1].bar(benchmark.observation_times()[1], design_weights[1])
axs[1].set_title("State 3")
[ax.set_xlabel("Time (seconds)") for ax in axs]
# plt.show()
