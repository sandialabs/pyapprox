"""
Polynomial Chaos Regression
===========================

This tutorial demonstrates how to build a polynomial chaos expansion (PCE) using least squares regression.
We will use the Genz benchmark to illustrate the process, including model setup, basis initialization,
training data generation, cross-validation, and estimation of the PCE accuracy with test data.

Load the Modules Needed
-----------------------
"""

# Load modules from dependencies
import numpy as np

# Load modules from pyapprox
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.benchmarks.genz import GenzBenchmark
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.affine.linearsystemsolvers import LstSqSolver
from pyapprox.surrogates.affine.crossvalidation import (
    KFoldCrossValidation,
    CrossValidationStructureSearch,
    PolynomialDegreeIterator,
)

# %%
# Setup the Model and Variables
# -----------------------------
# In this section, we set up the benchmark model and define its random input variables.


# Set the random seed for reproducibility
np.random.seed(1)
# Load in the genz benchmark which creates the model
# and defines its random input variables
benchmark = GenzBenchmark("oscillatory", 2, backend=bkd)
# Extract the model from the benchmark
model = benchmark.model()
# Extract the random variable
variable = benchmark.prior()

# %%
# Initialize the Orthonormal Basis
# --------------------------------
# Each basis in a polynomial chaos expansions is constructed from a tensor-product of univariate orthonormal polynomial bases.
# Here, we initialize the basis for the random input variables.

# Define the distribution of the mode
# Define univariate orthonormal bases for each variable
polys_1d = [
    setup_univariate_orthogonal_polynomial_from_marginal(marginal, backend=bkd)
    for marginal in benchmark.prior().marginals()
]
# Now, create the tensor product basis
basis = OrthonormalPolynomialBasis(polys_1d)
# Set initial basis index set (a linear total degree basis)
basis.set_hyperbolic_indices(1, 1.0)

# %%
# Instantiate the PCE
# -------------------
# Initialize the PCE by specifying the solver and associating it with the basis.

# Specify the solver used to compute the PCE coefficients
solver = LstSqSolver(backend=bkd)
# Create the PCE
pce = PolynomialChaosExpansion(basis, solver=solver, nqoi=model.nqoi())


# %%
# Generate the Training Data
# --------------------------
# Generate training data by sampling the random variables and evaluating the model.
ntrain_samples = 100
# Generate random samples of the model inputs
train_samples = variable.rvs(ntrain_samples)
# Evaluate the model at the samples
train_values = model(train_samples)

# %%
# Run Cross Validation to Choose Best Polynomial Degree
# -----------------------------------------------------
# Use K-fold cross-validation to select the optimal polynomial degree for the PCE.

# Initialize the K-fold algorithm
kcv = KFoldCrossValidation(train_samples, train_values, pce)
# Define the degrees and strengths of the hyperbolic index set to considers
search = PolynomialDegreeIterator([1, 2, 3, 4, 5], [1.0])
# Initialize the hyper-parameter search
cv_search = CrossValidationStructureSearch(kcv, search)
# Run the cross validation
cv_search.run()
# Print the learned PCE structure
print(pce)
# Print the hyper-parameter search results
print(cv_search)

# %%
# Quantify the True Accuracy of the PCE
# -------------------------------------
# Compute the test error using unseen data to evaluate the accuracy of the PCE.

# Generate the test data
ntest_samples = 1000
test_samples = variable.rvs(ntest_samples)
test_values = model(test_samples)
# Estimate the absolute root-mean-squared-error (RMSE) error
rmse = bkd.norm(test_values - pce(test_samples))
# Print the error
print("The RMSE error of the PCE is", rmse)
# Compare the RMSE error to the cross-validation error
print("The CV error for the PCE is", bkd.sqrt(cv_search.best_cv_score()))
