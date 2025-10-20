"""
Computing the Convergence Rate of Numerical Estimates for Risk-Aware Bayesian Experimental Design
===============================================================================================

This tutorial builds upon the analytical expressions derived in the previous tutorial, which quantified the posterior mean, covariance, and risk measures such as entropic risk and Average Value at Risk (AVaR) in Bayesian linear regression. Here, we focus on numerically verifying the accuracy of these expressions by computing the convergence rate of numerical estimates using different quadrature methods and sample sizes.

The convergence rate analysis is critical for validating the numerical implementation of Bayesian Optimal Experimental Design (BOED) software, ensuring that the computed risk-aware measures align with theoretical expectations. Specifically, we examine how the mean squared error (MSE) of numerical estimates decreases as the number of samples increases, providing insights into the efficiency and reliability of the computational methods.

"""

# %%
# Section 1: Import Libraries
# ---------------------------
# Import the necessary libraries and modules for the tutorial.
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.expdesign.optbayes_benchmarks import (
    LinearGaussianBayesianOEDForPredictionBenchmark,
    BayesianOEDForPredictionDiagnostics,
    ConjugateGaussianPriorOEDForLinearPredictionAVaRDeviation,
    ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation,
)
from pyapprox.expdesign.optbayes import (
    NoiseStatistic,
    SampleAverageMean,
    OEDAVaRDeviationMeasure,
    OEDStandardDeviationMeasure,
)

# %%
# Section 2: Problem Setup
# ------------------------
# Define the problem parameters for Bayesian experimental design.
np.random.seed(1)  # Set the seed for reproducibility
nobs = 2  # Number of observations
min_degree = 0  # Minimum polynomial degree
degree = 3  # Maximum polynomial degree
noise_std = 0.125 * 4  # Noise standard deviation
prior_std = 0.5  # Prior standard deviation
nqoi = 1  # Number of quantities of interest (adjust as needed)

# Create the problem instance
problem = LinearGaussianBayesianOEDForPredictionBenchmark(
    nobs=nobs,
    min_degree=min_degree,
    degree=degree,
    noise_std=noise_std,
    prior_std=prior_std,
    nqoi=nqoi,
    backend=bkd,
)
# Define the sampling parameters for the quadrature methods and sample sizes.
design_weights = (
    bkd.ones((nobs, 1)) / nobs
)  # Specify a design for which we will compue the utility

# %%
# Section 3: Define Risk and Deviation Measures
# ---------------------------------------------
# Set up the risk and deviation measures for the analysis.
noise_stat = NoiseStatistic(SampleAverageMean(bkd))
risk_measure = SampleAverageMean(bkd)
# deviation_measure = OEDAVaRDeviationMeasure(nqoi, 0.5, 1000000, bkd)
# utility_cls = ConjugateGaussianPriorOEDForLinearPredictionAVaRDeviation
deviation_measure = OEDStandardDeviationMeasure(nqoi, bkd)
utility_cls = ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation

# Create the diagnostics instance
oed_diagnostic = BayesianOEDForPredictionDiagnostics(
    problem,
    utility_cls,
    deviation_measure,
    risk_measure,
    noise_stat,
)

# %%
# Section 4: Define the MC Quadrature Rules For Computing the Utility
# -------------------------------------------------------------------
quadtype = "MC"  # Quadrature type: Monte Carlo (MC)
# Set the number of realizations and sample counts. Must be one for a deterministic sequence
nrealizations = 100

outerloop_sample_counts = [100, 500]  # Number of outer loop samples
innerloop_sample_counts = [500, 1000, 5000]  # Inner loop sample sizes

# %%
# Section 5: Compute and Plot Mean Squared Error (MSE)
# ----------------------------------------------------
# Compute the MSE for different sample combinations and plot the results.
fig, axes = plt.subplots(1, 3, figsize=(3 * 8, 6), sharex=True, sharey=True)

values = oed_diagnostic.compute_mse_for_sample_combinations(
    outerloop_sample_counts,
    innerloop_sample_counts,
    nrealizations,
    design_weights,
    quadtype,
    quadtype,
)
oed_diagnostic.plot_mse_vs_innerloop_samples(
    axes, outerloop_sample_counts, innerloop_sample_counts, values
)

# %%
# The plots show that for linear OED problems, such as this one, the MSE not depend on the number of outerloop iterations. However, this is not true for nonlinear OED problems. Note, the variation in the horizontal lines is only caused by using a finite number of trials to compute the MSE.
#
# Section 6: Compute Convergence Rate
# -----------------------------------
# Compute the convergence rate with respect to inner loop samples.
for ii, nin in enumerate(outerloop_sample_counts):
    convergence_rate = oed_diagnostic.compute_convergence_rate(
        innerloop_sample_counts, bkd.vstack(values["mse"])[:, ii]
    )
    print(
        f"MC convergence rate (ninner_loop_samples={nin}): {convergence_rate:.8f}"
    )

# %%
# Compute the MSE and Convergence Rate using Quasi-Monte Carlo
# ------------------------------------------------------------
# We can make the MSE converge faster by using Quasi Monte Carlo methods. For example, here we use Halton sequences

quadtype = "Halton"  # Quadrature type: Halton Sequence
# Set the number of realizations and sample counts
nrealizations = 10

outerloop_sample_counts = [100, 500, 1000]  # Number of outer loop samples
innerloop_sample_counts = [500, 1000, 5000]  # Inner loop sample sizes


# Compute the MSE for different sample combinations and plot the results.
fig, axes = plt.subplots(1, 3, figsize=(3 * 8, 6), sharex=True, sharey=True)

oed_diagnostic = BayesianOEDForPredictionDiagnostics(
    problem,
    utility_cls,
    deviation_measure,
    risk_measure,
    noise_stat,
)

values = oed_diagnostic.compute_mse_for_sample_combinations(
    outerloop_sample_counts,
    innerloop_sample_counts,
    nrealizations,
    design_weights,
    quadtype,
    quadtype,
)
oed_diagnostic.plot_mse_vs_innerloop_samples(
    axes, outerloop_sample_counts, innerloop_sample_counts, values
)


# Compute the convergence rate with respect to inner loop samples.
print(len(innerloop_sample_counts), bkd.vstack(values["mse"]).shape)
for ii, nin in enumerate(outerloop_sample_counts):
    convergence_rate = oed_diagnostic.compute_convergence_rate(
        innerloop_sample_counts, bkd.vstack(values["mse"])[:, ii]
    )
    print(
        f"Halton convergence rate (ninner_loop_samples={nin}): {convergence_rate:.8f}"
    )
print(f"Convergence rate: {convergence_rate:.8f}")
plt.show()

# %%
# Section 8: Final Remarks
# ------------------------
# Summarize the results and their significance.
# In this tutorial, we have demonstrated how to compute the convergence rate of numerical estimates for risk-aware Bayesian experimental design. The results validate the numerical implementation of BOED software and ensure that the computed risk measures, such as AVaR deviation, align with theoretical expectations. By analyzing the convergence rate, we can assess the efficiency and reliability of different quadrature methods and sampling strategies, paving the way for robust risk-aware experimental design in Bayesian inference.
