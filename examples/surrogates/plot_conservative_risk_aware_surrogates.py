r"""
Conservative Risk-Aware Surrogates
==================================

Risk measures combine objective probabilities with the subjective values of a decision maker to quantify anticipated outcomes. In this tutorial, we will construct a surrogate model that produces estimates of risk measures that are always greater than their empirical approximations obtained from training data. These conservative surrogates limit over-confidence in reliability and safety assessments and produce estimates of risk measures that converge faster to the true value than purely sample-based estimates.

Load the Modules Needed
-----------------------
"""

# Load modules from dependencies
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load modules from pyapprox
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.affine.linearsystemsolvers import (
    ConservativeQuantileRegressionSolver,
    ConservativeLstSqSolver,
)
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.variables.joint import IndependentMarginalsVariable


# %%
# Step 1: Define the Distribution of the model inputs
# ---------------------------------------------------
# Define a uniform marginal distribution over the interval [0, 1] to
# represents the uncertainty in the input variable.
# Use Legendre polynomials to construct an polynomial basis orthonoral to the input variable.
# This basis will be used to approximate the response variable.


marginals = [stats.uniform(0, 1)]
variable = IndependentMarginalsVariable(marginals, backend=bkd)
bases_1d = [
    setup_univariate_orthogonal_polynomial_from_marginal(marginal, backend=bkd)
    for marginal in marginals
]
basis = OrthonormalPolynomialBasis(bases_1d)
# initialize a linear expansion
basis.set_tensor_product_indices([2])

# %%
# Step 2: Generate Training Data
# ------------------------------
# Generate training samples from the variable distribution and compute the corresponding response values using the polynomial basis.
# Noise is added to simulate real-world data.

ntrain_samples = 1000
train_samples = variable.rvs(ntrain_samples)
exact_coef = bkd.ones((2, 1))
train_values = basis(train_samples) @ exact_coef
train_values += bkd.asarray(np.random.normal(0, 1, train_values.shape))

# %%
# Step 3: Solve The Conservative Regression Problem
# -------------------------------------------------
# Solve the conservative quantile regression problem for a quantile level of :math:`\tau = 0.8`.
# Solve the conservative risk margins regression problem for a strength of 1.0.

quantile = 0.8
quantile_solver = ConservativeQuantileRegressionSolver(quantile, backend=bkd)
quantile_pce = PolynomialChaosExpansion(basis, solver=quantile_solver, nqoi=1)
quantile_pce.fit(train_samples, train_values)

strength = 1.0
lstsq_solver = ConservativeLstSqSolver(strength, backend=bkd)
risk_margins_pce = PolynomialChaosExpansion(basis, solver=lstsq_solver, nqoi=1)
risk_margins_pce.fit(train_samples, train_values)

# %%
# Step 4: Visualize the Results
# -----------------------------
# Plot the training data, the surrogate model, and the true function to compare their behavior.

plt.plot(train_samples[0], train_values[:, 0], "ob", label="Training data")
test_samples = bkd.linspace(0, 1, 101)[None, :]
plt.plot(
    test_samples[0],
    quantile_pce(test_samples)[:, 0],
    "k",
    lw=3,
    label="Quantile Surrogate",
)
plt.plot(
    test_samples[0],
    risk_margins_pce(test_samples)[:, 0],
    "g",
    lw=3,
    label="Risk Margins Surrogate",
)
plt.plot(
    test_samples[0],
    basis(test_samples) @ exact_coef,
    "r",
    lw=3,
    label="True function",
)
plt.legend()
plt.xlabel("X")
_ = plt.ylabel("y")


# %%
# The plot shows:
#
# Training Data: Blue points represent the noisy training data.
# True Function: The red line represents the true function used to generate the data.
# Quantile Surrogate Model: The black line represents the conservative surrogate model learned using quantile regression.
# Saftey Margins Surrogate Model: The black line represents the conservative surrogate model learned using saftey margins regression.
#

# %%
# Step 5: Constrast The Surrogate- and Data-Based Measures of Risk
# ----------------------------------------------------------------
# The following checks that the surrogate model predicts measurs of risk that are greater than the training data, ensuring conservative estimates of risk measures.
quantile_solver.risk_measure().set_samples(quantile_pce(train_samples).T)
print(
    "Quantile Surrogate Average Value at Risk (AVaR)",
    quantile_solver.risk_measure()(),
)
quantile_solver.risk_measure().set_samples(train_values.T)
print(
    "Training Data AVaR",
    quantile_solver.risk_measure()(),
)
lstsq_solver.risk_measure().set_samples(risk_margins_pce(train_samples).T)
print(
    "LstSq Surrogate Saftey Margins Risk Measure",
    lstsq_solver.risk_measure()(),
)
lstsq_solver.risk_measure().set_samples(train_values.T),
print(
    "Training Data Saftey Margins Risk Measure",
    lstsq_solver.risk_measure()(),
)

# %%
# Conclusion
# ----------
# In this tutorial, we constructed a conservative risk-aware surrogate model using quantile regression. These surrogates are valuable in reliability and safety assessments, where over-confidence can lead to underestimating risks. By ensuring that the surrogate model's predictions are always greater than empirical approximations, we can ensure we are less likely to underestimate risk.
