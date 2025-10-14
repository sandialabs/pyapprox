r"""
Basis Pursuit Densoising
========================
Basis Pursuit Denoising (BPDN) is a popular optimization problem in signal processing and surrogate modeling. It seeks to find a sparse solution to an underdetermined linear system while allowing for some noise in the measurements. This tutorial explains how to formulate BPDN as a quadratic program (QP).

Problem Statement: Basis Pursuit Denoising
------------------------------------------
The BPDN problem is typically written as:

.. math::
    \argmin⁡_\vec{x} \lVert \vec{x}\rVert_1 \text{  subject to  } \lVert\mat{A}\vec{x}−\vec{y}\rVert_2^2\le \epsilon.

Where:

- :math:`\mat{A} \in \mathbb{R}^{m \times n}`: A matrix representing the linear system.
- :math:`\vec{y}\in \mathbb{R}^m`: A vector of measurements.
- :math:`\vec{x} \in \mathbb{R}^n`: The sparse solution we want to find.
- :math:`\vec{\epsilon}`: A tolerance for the noise level in the measurements.
- :math:`\|\vec{x}\|_1`: The :math:`\ell_1`-norm of :math:`\vec{x}`, which promotes sparsity.
- :math:`\|\mat{A}\vec{x} - \vec{b}\|_2`: :math:`\ell_2`-norm of the residual, which measures the error between the measurements and the reconstruction.


Reformulating BPDN as a Quadratic Program
-----------------------------------------
To write BPDN as a quadratic program, we need to:

Replace the :math:`\ell_1`-norm with linear constraints.
Represent the :math:`\ell_2`-norm constraint as a quadratic constraint or penalty.


Step 1: Replace :math:`\|\vec{x}\|_1` with Linear Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :math:`\ell_1`-norm of :math:`\vec{x}` can be expressed as:

.. math:: \|\vec{x}\|_1 = \sum_{i=1}^n |x_i|

To handle the absolute values, introduce auxiliary variables :math:`u_i \ge 0` for each :math:`x_i`, and enforce:

.. math:: x_i \leq u_i \quad \text{and} \quad -x_i \leq u_i.

​This reformulates :math:`\|\vec{x}\|_1` as:

.. math::\|\vec{x}\|_1 = \sum_{i=1}^n u_i

​Thus, minimizing :math:`\|\vec{x}\|_1` becomes minimizing :math:`\sum_{i=1}^n u_i`, subject to the constraints:

.. math:: -x_i \leq u_i \quad \text{and} \quad x_i \leq u_i \quad \forall i

Step 2: Represent :math:`\|\mat{A}\vec{x} - \vec{y}\|_2 \leq \epsilon`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constraint :math:`\|\mat{A}\vec{x} - \vec{y}\|_2 \leq \epsilon` can be rewritten as:

.. math:: (\mat{A}\vec{x} - \vec{y})^\top (\mat{A}\vec{x} - \vec{y}) \leq \epsilon^2

This is a quadratic constraint.
Alternatively, if we want to penalize the residual instead of constraining it, we can add a quadratic term to the objective:

.. math:: \|\mat{A}\vec{x} - \vec{y}\|_2^2 = (\mat{A}\vec{x} - \vec{y})^\top (\mat{A}\vec{x} - \vec{y})

Step 3: Combine the Objective and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The BPDN problem can now be written as a quadratic program:

.. math:: \min_{\vec{x}, \vec{u}} \quad \frac{1}{2} (\mat{A}\vec{x} - \vec{y})^\top (\mat{A}\vec{x} - \vec{y}) + \lambda \sum_{i=1}^n u_i

Subject to:

.. math:: -x_i \leq u_i \quad \text{and} \quad x_i \leq u_i \quad \forall i

Where:

 - :math:`\lambda > 0`: \mat{A} regularization parameter that balances the sparsity (:math:`\ell_1`-norm) and the residual (:math:`\ell_2`-norm).



Final Quadratic Program Formulation
-----------------------------------
Noting that :math:`\frac{1}{2} (\mat{A}\vec{x} - \vec{y})^\top (\mat{A}\vec{x} - \vec{y}) = \frac{1}{2} \vec{x}^\top \mat{A}^\top \mat{A} \vec{x} - b^\top \mat{A} \vec{x}  + \frac{1}{2} \vec{y}^\top \vec{y}` and that
:math:`\frac{1}{2} \vec{y}^\top \vec{y}` does not effect the optimal solution so we can drop it from the optimization,
the quadratic program is:

Objective:

.. math:: \min_{\vec{x}, \vec{u}} \quad \frac{1}{2} \vec{x}^\top \mat{A}^\top \mat{A} \vec{x} - b^\top \mat{A} \vec{x} + \lambda \sum_{i=1}^n u_i

Constraints:

.. math:: -x_i \leq u_i \quad \text{and} \quad x_i \leq u_i \quad \forall i

Solving the Quadratic Program
-----------------------------
"""

# Import the necessary modules
import numpy as np
from scipy import stats
import copy
import matplotlib.pyplot as plt
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.affine.linearsystemsolvers import (
    BasisPursuitDenoisingCVXRegressionSolver,
)
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.affine.basis import (
    OrthonormalPolynomialBasis,
    FixedGaussianTensorProductQuadratureRuleFromVariable,
)
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.affine.induced import (
    GaussQuadratureBasedDiscreteInducedPolySampler,
)
from pyapprox.interface.model import ModelFromVectorizedCallable

# Set the random seed for reproducibility
np.random.seed(1)
# Setup an orthonormal basis
nvars = 2
marginals = [stats.uniform(0, 1)] * nvars
variable = IndependentMarginalsVariable(marginals, backend=bkd)
bases_1d = [
    setup_univariate_orthogonal_polynomial_from_marginal(marginal, backend=bkd)
    for marginal in marginals
]
basis = OrthonormalPolynomialBasis(bases_1d)
# Set the degree of the multivariate basis
degree = 10
basis.set_hyperbolic_indices(degree, 1.0)
# Setup the surrogate
bexp = PolynomialChaosExpansion(basis)
# Initialize class to sample from the optimal induced measure
sampler = GaussQuadratureBasedDiscreteInducedPolySampler(
    bexp, variable, [500] * nvars
)
# Generate the training data
model = ModelFromVectorizedCallable(
    1, nvars, lambda x: np.cos(2 * np.pi * x).sum(axis=0)[:, None]
)
nsamples = 50
train_samples = sampler(nsamples)
train_values = model(train_samples)
solver = BasisPursuitDenoisingCVXRegressionSolver(0.001, backend=bkd)
solver.set_options({"abstol": 1e-14, "reltol": 1e-14, "feastol": 1e-14})
solver.set_weights(sampler.christoffel_function(train_samples))
bexp.set_solver(solver)
bexp.fit(train_samples, train_values)

# Plot the true and approximate models
axs = plt.subplots(1, 2, figsize=(2 * 8, 6), sharey=True)[1]
bexp.plot_contours(axs[0], [0, 1, 0, 1], npts_1d=101, levels=31)
model.plot_contours(axs[1], [0, 1, 0, 1], npts_1d=101, levels=31)
# plot the recovered coefficients
ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)[1]
# Only even terms have non-zero coefficients
ax.semilogy(
    bkd.arange(basis.nterms()), bkd.abs(bexp.get_coefficients()[:, 0]), "o"
)

# Compute highly accurate coefficients using L2 projection
ref_bexp = copy.deepcopy(bexp)
quad_rule = FixedGaussianTensorProductQuadratureRuleFromVariable(
    variable, [30, 30]
)
quad_samples, quad_weights = quad_rule()
values = model(quad_samples)
projection_coefs = ref_bexp.coefficients_from_projection(
    quad_samples, quad_weights, values
)
ref_bexp.set_coefficients(projection_coefs)
# Plots its coefficients
ax.semilogy(
    bkd.arange(ref_bexp.basis().nterms()),
    bkd.abs(ref_bexp.get_coefficients()[:, 0]),
    "x",
)

# Compare the accuracy of the reference polynomial and the one computed using BPDN
test_samples = variable.rvs(1000)
print(
    "BPDN Error",
    bkd.norm(model(test_samples) - bexp(test_samples))
    / bkd.norm(model(test_samples)),
)

print(
    "Reference Error",
    bkd.norm(model(test_samples) - ref_bexp(test_samples))
    / bkd.norm(model(test_samples)),
)

# %%
# The reference error is above machine precision because we are using a truncated basis. The reference error is dominated by the error from this truncation which is substantially larger than the pfojection error because we use a very high-order Gaussian quadrature rule.

# %%
# Key Notes:
# ---------
#
# Regularization Parameter (:math:`\lambda`):
#
# Controls the trade-off between sparsity and fitting the measurements.
# Larger :math:`\lambda` promotes sparsity, while smaller :math:`\lambda` prioritizes minimizing the residual.
#
# Constraint vs Penalty:
#
# If you want to enforce :math:`\|\mat{A}\vec{x} - b\|_2 \leq \epsilon` strictly, use then :math:`\|\mat{A}\vec{x} - b\|_2^2`  as a constraint.
#
# Scalability:
#
# Quadratic programs are computationally efficient for moderate-sized problems, but for very large problems, specialized solvers or approximations may be needed.
#
# Accuracy
# The sparse solution returned by BPDN produces an accurate approximation of the target function despite using amount of training data that was smaller than the number of unknowns.

# Exercise: Comparing th BPDN solution with the solution obtained using underdetermined least-squares using the pseudo inverse as implemented by LstSqSolver
#
# Conclusion
# ----------
# By reformulating BPDN as a quadratic program, we can leverage efficient QP solvers to find sparse solutions to noisy linear systems. This approach is widely used in signal processing, compressed sensing, and machine learning applications.
