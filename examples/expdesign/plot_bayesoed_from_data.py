"""
Optimal Experimental Design from Pre-computed Simulation Data
=============================================================

This tutorial demonstrates how to create an optimal experimental design (OED) using pre-computed simulation data.
We will load the data, preprocess it, set up the OED problem, compute the optimal design, and visualize the results.

Prerequisites:
--------------
- Python
- Required libraries: `pickle`, `numpy`, `matplotlib`, and any specific backend library (e.g., `bkd`).
- Pre-computed simulation data files: `outerloop_data_filename` and `innerloop_data_filename`.

Steps:
------
1. Load the pre-computed simulation data.
2. Preprocess the data to extract relevant subsets.
3. Set up the OED problem.
4. Compute the optimal experimental design using a double-loop algorithm.
5. Visualize the results.

"""

# %%
# Load the pre-computed simulation data
# -------------------------------------
# The simulation data is stored in two files: `outerloop_data_filename` and `innerloop_data_filename`.
# We will load the data using the `pickle` module.

import os
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.util.backends.torch import TorchMixin as bkd
from pyapprox.optimization.minimize import SampleAverageMean
from pyapprox.expdesign.bayesoed import (
    BayesianOEDForPrediction,
    IndependentGaussianOEDInnerLoopLogLikelihood,
    OEDStandardDeviationMeasure,
    NoiseStatistic,
    OEDDataManager,
)
from pyapprox.benchmarks import ObstructedAdvectionDiffusionOEDBenchmark

# Define file paths for the pre-computed data
# We use a small number of samples here only so large files are not created
# during nightly testing. Increase to the amount you desire.
nouterloop_samples = 10
ninnerloop_samples = 10
# nouterloop_samples = 10000
# ninnerloop_samples = 10000

filename = "obstructed_advec_diff_oed_data_n_{0}_m_{1}.pkl".format(
    nouterloop_samples, ninnerloop_samples
)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
filename = os.path.join(data_dir, filename)
print(filename)
original_oed_data_manager = OEDDataManager(bkd)
original_oed_data_manager.load_data(filename)

# %%
# Preprocess the data
# -------------------
# Extract subsets of the simulation data and observations for the OED problem.

noutloop_samples = 9  # 100  # Number of outer loop samples to use
ninloop_samples = 9  # 100  # Number of inner loop samples to use
# noutloop_samples = 100
# ninloop_samples = 100
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
outerloop_samples = oed_data_manager.get("outloop_samples")
outerloop_shapes = oed_data_manager.get("outloop_shapes")
outerloop_quad_weights = oed_data_manager.get("outloop_quad_weights")
observation_locations = oed_data_manager.get("observation_locations")

innerloop_samples = oed_data_manager.get("inloop_samples")
innerloop_shapes = oed_data_manager.get("inloop_shapes")
innerloop_quad_weights = oed_data_manager.get("inloop_quad_weights")

qoi_vals = oed_data_manager.get("qoi_vals")
qoi_quad_weights = oed_data_manager.get("qoi_quad_weights")
nqoi = qoi_vals.shape[1]

# %%
# Set up the OED problem
# ----------------------
# Define the inner loop log-likelihood and initialize the Bayesian OED object.

noise_std = 1.0  # Standard deviation of noise
noise_cov_diag = bkd.full((nobs, 1), noise_std**2)
innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
    noise_cov_diag, backend=bkd
)
pred_oed = BayesianOEDForPrediction(innerloop_loglike)

# Specify the noise statistic taken over all realizations of the data.
# We will use the expectation
noise_stat = NoiseStatistic(SampleAverageMean(bkd))
# Specify the inner deviation measure
deviation_measure = OEDStandardDeviationMeasure(nqoi, bkd)
# Specify the risk measure over the prediction space
risk_measure = SampleAverageMean(bkd)

# Set the data for the OED problem
pred_oed.set_data(
    outerloop_shapes,
    outerloop_samples,
    outerloop_quad_weights,
    innerloop_shapes,
    innerloop_quad_weights,
    qoi_vals,
    qoi_quad_weights,
    deviation_measure,
    risk_measure,
    noise_stat,
)

# %%
# Compute the OED
# ---------------
# Use the double-loop algorithm to compute the optimal experimental design.

pred_oed.set_optimizer(
    pred_oed.default_optimizer(global_search=True, verbosity=3, gtol=1.0e-6)
)
design_weights = pred_oed.compute()

# %%
# Visualize the OED
# -----------------
# Plot the optimal experimental design and compare it with random designs.

fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

# Plot the domain boundaries
benchmark = ObstructedAdvectionDiffusionOEDBenchmark(backend=bkd)
fe_model = benchmark.observation_model().fe_model()
fe_model.stokes_model().plot_domain_boundaries(ax)

# Plot the observation locations
ax.scatter(
    *observation_locations,
    c=design_weights[:, 0],
    s=design_weights[:, 0] / design_weights.max() * 50,
)

# Generate random designs and compute objective values
nrandom_designs = 100
obj_vals = bkd.empty(nrandom_designs)

for ii in range(nrandom_designs):
    random_design_weights = bkd.asarray(
        np.random.uniform(0.0, 1.0, design_weights.shape)
    )
    random_design_weights /= bkd.sum(random_design_weights)
    obj_vals[ii] = pred_oed.objective()(random_design_weights)

# Print minimum and maximum objective values
print("Min objective value:", obj_vals.min())
print("Max objective value:", obj_vals.max())

# Plot histogram of objective values
plt.figure(figsize=(8, 6))
plt.hist(
    obj_vals, bins=10, color="blue", alpha=0.7, edgecolor="black", density=True
)

# Add vertical line for optimal design objective value
optimal_design_obj_val = pred_oed.objective()(design_weights)
plt.axvline(
    optimal_design_obj_val,
    color="orange",
    linestyle="--",
    linewidth=2,
    label="Optimal Design Obj Val",
)
plt.xlabel("Objective Values")
plt.ylabel("Density")
plt.title("Histogram of Objective Values")
plt.legend()
plt.show()
