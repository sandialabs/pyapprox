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
from pyapprox.expdesign.bayesoed import (
    KLBayesianOED,
    IndependentGaussianOEDInnerLoopLogLikelihood,
)

# %% Setup the Model to Generate the Simulation Data
# -------------------------------------------------
# Users can load in their own simulation data to create an OED.
# Here we show how to construct the data needed. First we must
# setup the simulation model


# Set the random seed for reproducibility
np.random.seed(1)
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
axs = plt.subplots(1, 4, figsize=(4 * 8, 6), sharey=True)[1]
axs[0].plot(benchmark.solution_times(), model_obs_sol.T)
# Plot the observations
axs[0].plot(
    benchmark.observation_times().T, observations.T, "o", label="Observations"
)
# Plot the predictions of the unobserved state
axs[0].legend()
# Plot different trajectories of each state at various samples
for sample in prior.rvs(100).T:
    model_obs_sol = obs_model.forward_solve(sample[:, None])[0]
    for ii in range(3):
        axs[ii + 1].plot(benchmark.solution_times(), model_obs_sol[ii])
plt.show()

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

# %%
# Initialize the OED Problem
# --------------------------

# Define the OED problem
kl_oed = KLBayesianOED(innerloop_loglike)

# %% Generate The Simulation Data
# -------------------------------
# Generate the simulation data needed to perform OED. Here we use a model of a predator-prey system.
# We must create observations for the outerloop and simulation data for the innerloop.

# Generate the outerloop observations
nouterloop_samples = 10000
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
ninnerloop_samples = int(math.sqrt(nouterloop_samples))
# Generate the samples to evaluate the model
innerloop_samples = prior.rvs(ninnerloop_samples)
innerloop_quad_weights = bkd.full(
    (ninnerloop_samples, 1), 1.0 / ninnerloop_samples
)

# Use the sample to evaluate the models and pass the data to the OED problem
kl_oed.set_data_from_model(
    obs_model,
    prior,
    outerloop_samples,
    outerloop_quad_weights,
    innerloop_samples,
    innerloop_quad_weights,
)
kl_oed.set_optimizer(kl_oed.default_optimizer(verbosity=0))

# %%
# Compute the OED
# ---------------
# Use the double loop algorithm to construct the OED

design_weights = kl_oed.compute()

# %%
# Plot the OED
# ------------
axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)[1]
# Reshape weights so that each row correspond to one of the OED states
design_weights = design_weights.reshape((2, -1))
axs[0].bar(benchmark.observation_times()[0], design_weights[0])
axs[0].set_title("State 1")
axs[0].set_ylabel("Design Weight")
axs[1].bar(benchmark.observation_times()[1], design_weights[1])
axs[1].set_title("State 3")
[ax.set_xlabel("Time (seconds)") for ax in axs]
plt.show()
