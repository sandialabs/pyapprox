r"""
Polynomial Chaos Expansions for Dependent Variables
===================================================

Polynomial chaos expansions (PCE) are frequently used to build surrogates of models parameterized by independent random variables. However, when the variables are **dependent**, constructing a PCE becomes more challenging. This tutorial explains how to construct a polynomial basis for dependent variables using numerical orthogonalization.

Introduction
------------

PCE for dependent variables are constructed numerically from a set of multivariate linearly independent basis functions. Specifically, a quadrature rule form the dependent measure is used to approximate an **inner product** that accounts for the dependency between variables and is used to produce the new orthogonal basis.

Problem Setup
-------------

Given a set of dependent variables :math:`\vec{z} = [z_1, z_2, \dots, z_n]`, we aim to construct a set of orthogonal polynomial basis functions :math:`\{P_0, P_1, \dots, P_m\}` such that:

1. The basis functions are orthogonal with respect to a given inner product:

   .. math::

      \langle P_i, P_j \rangle = \int P_i(\vec{z}) P_j(\vec{z}) w(\vec{z}) \, d\vec{z} = 0 \quad \text{for } i \neq j

   where :math:`w(\vec{z})` is a weight function that reflects the dependency between variables.

2. The basis functions span the space of polynomials up to a given degree :math:`m`.

Steps to Construct the Basis
-----------------------------

Step 1: Import the necessary modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

# Load modules from dependencies
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load modules from pyapprox
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.benchmarks.genz import GenzBenchmark
from pyapprox.surrogates.univariate.orthopoly import (
    LegendrePolynomial1D,
    AffineMarginalTransform,
)
from pyapprox.variables.marginals import ContinuousScipyMarginal
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables import GaussCopulaVariable
from pyapprox.surrogates.affine.basis import (
    QRBasedRotatedOrthonormalPolynomialBasis,
)
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.affine.linearsystemsolvers import LstSqSolver

# %%
# Step 2: Setup the Model and Variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now set up the benchmark model and define its random input variables.


# Set the random seed for reproducibility
np.random.seed(1)
# Load in the genz benchmark which creates the model
benchmark = GenzBenchmark("oscillatory", 2, cfactor=5, backend=bkd)
# Extract the model from the benchmark
model = benchmark.model()
# Defines the random input variable.
# TODO Change to dependent variable based on copula
x_correlation = bkd.array([[1, 0.9], [0.9, 1]])
scipy_marginals = [stats.beta(a=2, b=5), stats.beta(a=5, b=2)]
marginals = [ContinuousScipyMarginal(marginal) for marginal in scipy_marginals]
variable = GaussCopulaVariable(marginals, x_correlation, backend=bkd)


# %%
# Step 3: Define the Inner Product
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The inner product between two functions :math:`f(\vec{z})` and :math:`g(\vec{z})` is defined as:
#
# .. math::
#
#    \langle f, g \rangle = \int f(\vec{z}) g(\vec{z}) w(\vec{z}) \, d\vec{z}
#
# where:
#
# - :math:`w(\vec{z})` is the weight function.
# - The integral is computed over the domain of :math:`\vec{z}`.
#
# For discrete data, the inner product can be approximated as:
#
# .. math::
#
#    \langle f, g \rangle \approx \sum_{i=1}^N f(\vec{z}_i) g(\vec{z}_i) w(\vec{z}_i)

nquad_samples = int(1e6)
quad_samples = variable.rvs(nquad_samples)
quad_weights = bkd.full((nquad_samples, 1), 1.0 / nquad_samples)

# %%
# Step 4: Start with a Set of Candidate Polynomials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Begin with a set of candidate polynomials :math:`\{Q_0, Q_1, \dots, Q_m\}`, where :math:`Q_k(\vec{z})` is a polynomial of degree :math:`k`. For example:
#
# .. math::
#
#    Q_0(\vec{z}) = 1, \quad Q_1(\vec{z}) = z_1, \quad Q_2(\vec{z}) = z_2, \quad Q_3(\vec{z}) = z_1^2, \dots

marginals = [stats.uniform(0, 1) for ii in range(model.nvars())]
transforms = [
    AffineMarginalTransform(marginal, enforce_bounds=True, backend=bkd)
    for marginal in marginals
]
# Define univariate polynomials that to construct the multivariate polynoimials Q
polys_1d = [
    LegendrePolynomial1D(trans=transforms[ii], backend=bkd)
    for ii in range(model.nvars())
]

# %%
# Step 5: Apply Gram-Schmidt Orthogonalization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Gram-Schmidt process constructs orthogonal polynomials :math:`\{P_0, P_1, \dots, P_m\}` from the candidate polynomials :math:`\{Q_0, Q_1, \dots, Q_m\}`.

# Define the basis
rotated_basis = QRBasedRotatedOrthonormalPolynomialBasis(polys_1d)
# Set the basis indices to compute the rotation
rotated_basis.set_tensor_product_indices([9 for ii in range(model.nvars())])
rotated_basis.set_quadrature_rule_tuple(quad_samples, quad_weights)


# %%
# Step 6: Instantiate the PCE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Initialize the PCE by specifying the solver and associating it with the basis.

# Specify the solver used to compute the PCE coefficients
solver = LstSqSolver(backend=bkd)
# Create the PCE
pce = PolynomialChaosExpansion(rotated_basis, solver=solver, nqoi=model.nqoi())


# %%
# Step 7: Generate the Training Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Generate training data by sampling the random variables and evaluating the model.

ntrain_samples = 100
# Generate random samples of the model inputs
train_samples = variable.rvs(ntrain_samples)
# Evaluate the model at the samples
train_values = model(train_samples)


# %%
# Step 8: Build and plot the PCE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Build the PCE using the training data and comapre it to the true function

# Build the PCE
pce.fit(train_samples, train_values)

axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
# Plot the true function
model.plot_contours(axs[0], [0, 1, 0, 1], levels=31)
# Plot the PCE
pce.plot_contours(axs[1], [0, 1, 0, 1], levels=31)
# Plot the ntraining samples
_ = axs[1].plot(*train_samples, "ok")

# %%
# Quantify the True Accuracy of the PCE
# -------------------------------------
# Compute the test error using unseen data to evaluate the accuracy of the PCE.

# Generate the test data
ntest_samples = 1000
test_samples = variable.rvs(ntest_samples)
test_values = model(test_samples)
# Estimate the relative root-mean-squared-error (RMSE) error
rmse = bkd.norm(test_values - pce(test_samples)) / bkd.norm(test_values)
# Print the error
print("The RMSE error of the PCE is", rmse)

# %%
# Notice how the PCE is only accurate in the regions of higher-probability.
