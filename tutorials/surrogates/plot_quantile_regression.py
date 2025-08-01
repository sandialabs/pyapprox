r"""
Quantile Regression using Linear Programming
============================================

Introduction
------------

Quantile regression is a statistical technique used to estimate the conditional quantiles of a surrogate given its inputs. Unlike ordinary least squares regression, which minimizes the sum of squared residuals, quantile regression minimizes the sum of weighted absolute residuals. This makes it particularly useful for modeling the tails of a distribution.

Quantile regression can be formulated as a **linear program** (LP), making it computationally efficient for linear models. In this tutorial, we will discuss how to implement quantile regression as an LP.

Quantile Regression Formulation
-------------------------------

Problem Definition
~~~~~~~~~~~~~~~~~~

Given a dataset with predictor variables :math:`X` and response variable :math:`y`, quantile regression estimates the coefficients :math:`\beta` that minimize the following objective function:

.. math::

    \min_{\beta} \sum_{i=1}^n \rho_\tau(r_i)

where:

- :math:`r_i = y_i - X_i \beta` are the residuals,
- :math:`\tau` is the quantile level (e.g., :math:`\tau = 0.5` for the median),
- :math:`\rho_\tau(r)` is the quantile loss function defined as:

.. math::

    \rho_\tau(r) = 
    \begin{cases} 
    \tau \cdot r & \text{if } r \geq 0, \\
    (1 - \tau) \cdot |r| & \text{if } r < 0.
    \end{cases}.

Linear Programming Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quantile regression problem can be reformulated as a linear program:

1. Introduce auxiliary variables :math:`u_i` and :math:`v_i` to represent positive and negative residuals, respectively:

   .. math::

       r_i = u_i - v_i, \quad u_i, v_i \geq 0.

2. The objective function becomes:

   .. math::

       \min \sum_{i=1}^n \left( \tau \cdot u_i + (1 - \tau) \cdot v_i \right).

3. Subject to the constraints:

   .. math::

       y_i - X_i \beta = u_i - v_i, \quad u_i, v_i \geq 0.

This formulation is linear in :math:`\beta`, :math:`u_i`, and :math:`v_i`, making it solvable using standard LP solvers.

The linear program can be expressed in matrix form as:

.. math::

    \begin{align}
    \min_{\beta, u, v} \quad & \begin{bmatrix} \mathbf{0}^\top & \tau \cdot \mathbf{1}^\top & (1 - \tau) \cdot \mathbf{1}^\top \end{bmatrix}
    \begin{bmatrix} \beta \\ u \\ v \end{bmatrix} \\
    \text{subject to} \quad & 
    \begin{bmatrix} X & I & -I \end{bmatrix}
    \begin{bmatrix} \beta \\ u \\ v \end{bmatrix} = y, \\
    & u \geq 0, \quad v \geq 0.
    \end{align}

Numerical Example
-----------------
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.surrogates.affine.linearsystemsolvers import (
    QuantileRegressionSolver,
)
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.variables.joint import IndependentMarginalsVariable

# Define a linear expansion using Legendre polynomials
marginals = [stats.uniform(0, 1)]
variable = IndependentMarginalsVariable(marginals, backend=bkd)
bases_1d = [
    setup_univariate_orthogonal_polynomial_from_marginal(marginal, backend=bkd)
    for marginal in marginals
]
basis = OrthonormalPolynomialBasis(bases_1d)
basis.set_tensor_product_indices([2])

# Generate the training data
ntrain_samples = 1000
train_samples = variable.rvs(ntrain_samples)
exact_coef = bkd.ones((2, 1))
train_values = basis(train_samples) @ exact_coef
train_values += bkd.asarray(np.random.normal(0, 1, train_values.shape))

# Solve the quantile regression problem
quantile = 0.8
solver = QuantileRegressionSolver(quantile, backend=bkd)
coef = solver.solve(basis(train_samples), train_values)


# Plot the training data
plt.plot(train_samples[0], train_values[:, 0], "ob", label="Training data")
# Plot the surrogate learned
test_samples = bkd.linspace(0, 1, 101)[None, :]
plt.plot(
    test_samples[0], basis(test_samples) @ coef, "k", lw=3, label="Surrogate"
)
# Plot the true function
plt.plot(
    test_samples[0],
    basis(test_samples) @ exact_coef,
    "r",
    lw=3,
    label="True function",
)
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.show()
