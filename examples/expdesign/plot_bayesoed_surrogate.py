"""
Accelerating Bayesian Optimal Experimental Design with Surrogates
=================================================================

This tutorial demonstrates how to construct surrogates (using sparse grids) to speed up Bayesian optimal experimental design (OED).
We will walk through the process of setting up the simulation model, building surrogates, preprocessing data, and solving the OED problem.

Steps:
------
1. Load necessary modules and dependencies.
2. Set up the simulation model and benchmark.
3. Build surrogates for the observation and prediction models.
4. Preprocess the surrogate models for OED.
5. Set up the Bayesian OED problem.
6. Generate simulation data and compute the optimal experimental design.
7. Visualize the results.

"""

# %%
# Load Modules and Dependencies
# -----------------------------
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
from pyapprox.optimization.minimize import SampleAverageMean

# Build a surrogate of the observation model to reduce computation cost of OED
from pyapprox.surrogates.sparsegrids.combination import (
    LejaLagrangeAdaptiveCombinationSparseGrid,
    MaxNSamplesSparseGridSubspaceAdmissibilityCriteria,
)

# %%
# Setup the Simulation Model and Benchmark
# ----------------------------------------
# Users can load their own simulation data to create an OED.
# Here, we demonstrate how to construct the necessary data using a predefined benchmark.

# Set the random seed for reproducibility
np.random.seed(1)

# Initialize the benchmark
benchmark = ObstructedAdvectionDiffusionOEDBenchmark(backend=bkd)

# Extract components from the benchmark
obs_model = benchmark.observation_model()  # Observation model
pred_model = benchmark.prediction_model()  # Prediction model
prior = benchmark.prior()  # Random prior

noutloop_samples = 10
ninloop_samples = 10

# %%
# Build Surrogates for Observation and Prediction Models
# ------------------------------------------------------
# Sparse grids are used to build surrogates for the observation and prediction models,
# reducing the computational cost of OED.


def build_surrogate(filename, model, ntrain_samples, ntest_samples):
    """
    Build a sparse grid surrogate for the given model.

    Parameters:
    -----------
    filename : str
        Filename to save the surrogate.
    model : pyapprox.model
        The model to approximate.
    ntrain_samples : int
        Number of training samples for the sparse grid.
    ntest_samples : int
        Number of test samples for validation.

    Returns:
    --------
    tuple
        Sparse grid surrogate, test samples, and test values.
    """
    sg = LejaLagrangeAdaptiveCombinationSparseGrid(
        benchmark.prior(), model.nqoi()
    )
    admissibility_criteria = (
        MaxNSamplesSparseGridSubspaceAdmissibilityCriteria(ntrain_samples)
    )
    sg.setup(admissibility_criteria)
    model.work_tracker().set_active(True)
    sg.build(model)
    test_samples = benchmark.prior().rvs(ntest_samples)
    test_values = model(test_samples)
    return sg, test_samples, test_values


# Define filenames for surrogates
ntrain_samples = 100
obs_sg_filename = f"obstructed_advec_diff_obs_model_sg_n_{ntrain_samples}.pkl"
pred_sg_filename = (
    f"obstructed_advec_diff_pred_model_sg_n_{ntrain_samples}.pkl"
)

# Build and save surrogates if they don't already exist
if not os.path.exists(obs_sg_filename):
    ntest_samples = 20
    obs_sg, obs_test_samples, obs_test_vals = build_surrogate(
        obs_sg_filename, obs_model, ntrain_samples, ntest_samples
    )
    obs_sg_dict = {
        "sg": obs_sg,
        "test_samples": obs_test_samples,
        "test_vals": obs_test_vals,
        "work_tracker": obs_model.work_tracker(),
    }
    pickle.dump(obs_sg_dict, open(obs_sg_filename, "wb"))

    pred_sg, pred_test_samples, pred_test_vals = build_surrogate(
        pred_sg_filename, pred_model, ntrain_samples, ntest_samples
    )
    pred_sg_dict = {
        "sg": pred_sg,
        "test_samples": pred_test_samples,
        "test_vals": pred_test_vals,
        "work_tracker": pred_model.work_tracker(),
    }
    pickle.dump(pred_sg_dict, open(pred_sg_filename, "wb"))

# Load surrogates
print(f"Loading Sparse Grids From {obs_sg_filename}")
obs_sg_dict = pickle.load(open(obs_sg_filename, "rb"))
pred_sg_dict = pickle.load(open(pred_sg_filename, "rb"))
obs_sg, obs_test_samples, obs_test_vals = (
    obs_sg_dict["sg"],
    obs_sg_dict["test_samples"],
    obs_sg_dict["test_vals"],
)
pred_sg, pred_test_samples, pred_test_vals = (
    pred_sg_dict["sg"],
    pred_sg_dict["test_samples"],
    pred_sg_dict["test_vals"],
)

# Validate surrogates
print("Observation Model Work Tracker:", obs_sg_dict["work_tracker"])
print("Prediction Model Work Tracker:", pred_sg_dict["work_tracker"])
print("Observation Test Values Shape:", obs_test_vals.shape)
print("Observation Test Values Mean:", obs_test_vals.mean(axis=0))
print(
    "Obs SG Error:",
    bkd.norm(obs_sg(obs_test_samples) - obs_test_vals)
    / bkd.norm(obs_test_vals),
)
print(
    "Pred SG Error:",
    bkd.norm(pred_sg(pred_test_samples) - pred_test_vals)
    / bkd.norm(pred_test_vals),
)

# Update models to use surrogates
obs_model = obs_sg
pred_model = pred_sg

# %%
# Preprocess Sparse Grid for OED
# ------------------------------
# Use the surrogate on a subset of the data by creating an active set model.

from pyapprox.interface.wrappers import create_active_set_qoi_model

nobs_locations = 50
obs_model = create_active_set_qoi_model(
    obs_sg, bkd.arange(obs_sg.nqoi())[2::3][:nobs_locations]
)

# %%
# Set up the Bayesian OED Problem
# -------------------------------
# Define the inner loop log-likelihood and initialize the Bayesian OED object.

noise_std = 1.0  # Standard deviation of noise
noise_cov_diag = bkd.full((nobs_locations, 1), noise_std**2)
innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
    noise_cov_diag, backend=bkd
)
pred_oed = BayesianOEDForPrediction(innerloop_loglike)

# Specify the noise statistic taken over all realizations of the data.
noise_stat = NoiseStatistic(SampleAverageMean(bkd))
# Specify the inner deviation measure
nqoi = pred_model.nqoi()
deviation_measure = OEDStandardDeviationMeasure(nqoi, bkd)
# Specify the risk measure over the prediction space
risk_measure = SampleAverageMean(bkd)

# %%
# Generate Simulation Data and Compute the OED
# --------------------------------------------
data_generator = BayesianOEDDataGenerator(bkd)
(
    outerloop_samples,
    outerloop_quad_weights,
    innerloop_samples,
    innerloop_quad_weights,
) = data_generator.prepare_simulation_inputs(
    pred_oed, prior, "MC", 100, "MC", 100
)

# Set the data for the OED problem
qoi_quad_weights = bkd.full((nqoi, 1), 1.0 / nqoi)
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

# Compute the optimal experimental design
pred_oed.set_optimizer(
    pred_oed.default_optimizer(global_search=True, verbosity=0, gtol=1.0e-6)
)
design_weights = pred_oed.compute()

# %%
# Visualize the Results
# ---------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

# Plot the domain boundaries
benchmark = ObstructedAdvectionDiffusionOEDBenchmark(backend=bkd)
fe_model = benchmark.observation_model().fe_model()
fe_model.stokes_model().plot_domain_boundaries(ax)

# Plot the observation locations
observation_locations = benchmark.observation_locations()[:, :nobs_locations]
ax.scatter(
    *observation_locations,
    c=design_weights[:, 0],
    s=design_weights[:, 0] / design_weights.max() * 50,
)
