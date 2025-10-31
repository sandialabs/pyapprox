"""
Geenrating Data for Bayesian Optimal Experimental Design
========================================================

This tutorial demonstrates how to generate and save simulation data that can be used for Bayesian optimal experimental design (OED). The tutorial plot_bayesoed_from_data will show how to use this data to generate an OED/

Load the Modules Needed
-----------------------
"""

# Load modules from dependencies
import os
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load modules from pyapprox
from pyapprox.util.backends.torch import TorchMixin as bkd
from pyapprox.benchmarks import ObstructedAdvectionDiffusionOEDBenchmark
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.expdesign.bayesoed import (
    BayesianOEDForPrediction,
    IndependentGaussianOEDInnerLoopLogLikelihood,
    OEDStandardDeviationMeasure,
    OEDEntropicDeviationMeasure,
    NoiseStatistic,
    OEDDataManager,
    BayesianOEDDataGenerator,
)

# %% Setup the Model to Generate the Simulation Data
# -------------------------------------------------
# Users can load in their own simulation data to create an OED.
# Here we show how to construct the data needed. First we must
# setup the simulation model


# Set the random seed for reproducibility
np.random.seed(1)
# Setup the benchmark
benchmark = ObstructedAdvectionDiffusionOEDBenchmark(backend=bkd)
# Extract the model from the benchmark
obs_model = benchmark.observation_model()
# Extract the prediction model from the benchmark
pred_model = benchmark.prediction_model()
# Extract the random prior
prior = benchmark.prior()

# We use a small number of samples here only so large files are not created
# during nightly testing. Increase to the amount you desire.
noutloop_samples = 10
ninloop_samples = 10
quad_type = "Halton"
# quad_type = "MC"

# %% Generate The Simulation Data
# -------------------------------
# Generate the simulation data needed to perform OED. Here we use a model of a predator-prey system.
# We must create observations for the outerloop and simulation data for the innerloop.

# Generate the outerloop observations

prior_data_variable = IndependentMarginalsVariable(
    prior.marginals()
    + [stats.norm(0, 1.0) for ii in range(benchmark.nobservations())],
    backend=bkd,
)

# # Generate the outerloop samples needed to compute the outerloop observations
# outloop_samples = prior_data_variable.rvs(noutloop_samples)
# outloop_quad_weights = bkd.full(
#     (noutloop_samples, 1), 1.0 / noutloop_samples
# )

# # Generate the shapes of the likelihood, e.g. the model predictions,
# # for the inner OED loop
# # Generate the samples to evaluate the model
# inloop_samples = prior.rvs(ninloop_samples)
# inloop_quad_weights = bkd.full((ninloop_samples, 1), 1.0 / ninloop_samples)

data_generator = BayesianOEDDataGenerator(bkd, 1e-8)
outloop_samples, outloop_quad_weights = data_generator.setup_quadrature_data(
    quad_type, prior_data_variable, noutloop_samples, "outer"
)
inloop_samples, inloop_quad_weights = data_generator.setup_quadrature_data(
    quad_type, prior, ninloop_samples, "inner"
)

# Assume MC quadrature for prediction space
qoi_quad_weights = bkd.full((pred_model.nqoi(), 1), 1.0 / pred_model.nqoi())

# Building docs with sphinx gallery does not define __file__. However,
# this variable is defined when running the script from the command line.
# Luckily when buildings docs execution is done is the same directory as
# the script so we can set __file__ using os.getcwd
if "__file__" not in globals():
    # assumes script is run in the directory this file is contained
    file_dir = os.getcwd()
else:
    file_dir = os.path.abspath(__file__)
data_dir = os.path.join(os.path.dirname(file_dir), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_filename = "obstructed_advec_diff_oed_data_n_{0}_m_{1}_{2}.pkl".format(
    noutloop_samples, ninloop_samples, quad_type
)
data_filename = os.path.join(data_dir, data_filename)

original_oed_data_manager = OEDDataManager(bkd)
if not os.path.exists(data_filename):
    print("Generating Data")
    outloop_shapes_samples = outloop_samples[: prior.nvars()]
    outloop_shapes = obs_model(outloop_shapes_samples).T
    inloop_shapes = obs_model(inloop_samples).T
    qoi_vals = pred_model(inloop_samples)

    original_oed_data_manager.save_data(
        data_filename,
        outloop_samples,
        outloop_shapes,
        outloop_quad_weights,
        benchmark.observation_locations(),
        inloop_samples,
        inloop_shapes,
        inloop_quad_weights,
        qoi_vals,
        qoi_quad_weights,
    )


# Preprocess the data
# -------------------
# Extract subsets of the simulation data and observations for the OED problem.

original_oed_data_manager.load_data(data_filename)

noutloop_samples = 10  # Number of outer loop samples to use
ninloop_samples = 10  # Number of inner loop samples to use
nobs = 100  # Number of observations to use
# The active observational indices. Each location has three time observations
# So extract observation at final time from each location
active_obs_indices = bkd.arange(original_oed_data_manager.nobservations())[
    2::3
][:nobs]
active_obs_location_indices = bkd.arange(nobs)

oed_data_manager = original_oed_data_manager.extract_data_subset(
    active_obs_indices,
    active_obs_location_indices,
    noutloop_samples,
    ninloop_samples,
)


# %%
# Visualize the Observation Locations
# -----------------
# Plot the optimal experimental design locations for the full and reduced data set.

fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

# Plot the domain boundaries
benchmark = ObstructedAdvectionDiffusionOEDBenchmark(backend=bkd)
fe_model = benchmark.observation_model().fe_model()
fe_model.stokes_model().plot_domain_boundaries(ax)

# Plot the orginal observation locations
ax.scatter(
    *original_oed_data_manager.get("observation_locations"), s=20, marker="o"
)

# Plot the desired subset of observation locations
ax.scatter(*oed_data_manager.get("observation_locations"), s=10, marker="x")
