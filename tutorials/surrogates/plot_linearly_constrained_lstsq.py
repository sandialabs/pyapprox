r"""
Linearly Constrained Linear Least Squares
=========================================
Linearly Constrained Linear Least Squares is useful when we want to build a surrogate that enforces certain linear prioerties.
This tutorial explains how to derive and solve this problem step by step.

Linear Least Squares
--------------------
The unconstrained linear least squares problem is formulated as:

.. math:: \min_\vec{x} \|\mat{A}\vec{x} - \vec{y}\|_2^2

where

 - :math:`\mat{A} \in \mathbb{R}^{m \times n}`: A matrix representing the linear system.
 - :math:`\vec{y} \in \mathbb{R}^m`: A vector of observations or measurements.
 - :math:`\vec{x} \in \mathbb{R}^n`: The vector of unknowns to be solved.

The goal is to minimize the squared residual :math:`\|\mat{A}\vec{x} - \vec{y}\|_2^2`, which measures the error between the observed data :math:`\vec{y}` and the model :math:`\mat{A}\vec{x}`.

Adding Linear Constraints
-------------------------
In the linearly constrained version, we impose additional constraints expressed as:

.. math::\mat{C}\vec{x}=\vec{d}

Where:

 - :math:`\mat{C} \in \mathbb{R}^{p \times n}`: A matrix representing the linear constraints.
 - :math:`\vec{d} \in \mathbb{R}^p`: A vector specifying the values of the constraints.

The problem becomes:

.. math:: \|\mat{A}\vec{x} - \vec{y}\|_2^2 \quad \text{subject to} \quad \mat{C}\vec{x} = \vec{d}

Derivation of the Solution
--------------------------
The method of Lagrange multipliers can be used to to incorporate the constraints into the unconstrained least squares objective function.

Step 1: Define the Lagrangian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Lagrangian combines the objective function and the constraints:

.. math:: \mathcal{L}(\vec{x}, \vec{\lambda}) = \|\mat{A}\vec{x} - \vec{y}\|_2^2 + \vec{\lambda}^\top (\mat{C}\vec{x} - \vec{d})

Where:

 - :math:`\vec{\lambda} \in \mathbb{R}^p`: Lagrange multipliers associated with the constraints :math:`\mat{C}\vec{x}=\vec{d}`.


Step 2: Expand the Objective Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The squared residual can be expanded as:

.. math:: \|\mat{A}\vec{x} - \vec{y}\|_2^2 = (\mat{A}\vec{x} - \vec{y})^\top (\mat{A}\vec{x} - \vec{y}) = \vec{x}^\top \mat{A}^\top \mat{A} \vec{x} - 2\vec{y}^\top \mat{A} \vec{x} + \vec{y}^\top \vec{y}

Thus, the Lagrangian becomes:

.. math:: \mathcal{L}(\vec{x}, \vec{\lambda}) = \vec{x}^\top \mat{A}^\top \mat{A} \vec{x} - 2\vec{y}^\top \mat{A} \vec{x} + \vec{y}^\top \vec{y} + \vec{\lambda}^\top (\mat{C}\vec{x} - \vec{d})

Step 3: Take Derivatives and Solve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To find the optimal solution, we take the derivative of the lagrangian with respect to :math:`\vec{x}` and :math:`\vec{\lambda}`, and set them to zero.

Derivative with respect to :math:`\vec{x}`:

.. math:: \frac{\partial \mathcal{L}}{\partial \vec{x}} = 2\mat{A}^\top \mat{A} \vec{x} - 2\mat{A}^\top \vec{y} + \mat{C}^\top \vec{\lambda} = 0

which simplifies to:

.. math:: \mat{A}^\top \mat{A} \vec{x} - \mat{A}^\top \vec{y} + \frac{1}{2} \mat{C}^\top \vec{\lambda} = 0

Derivative with respect to :math:`\vec{\lambda}`:

.. math:: \frac{\partial \mathcal{L}}{\partial \vec{\lambda}} = \mat{C}\vec{x} - \vec{d} = 0


Step 4: Solve the System of Equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The two equations can be written as a block system:

.. math::

    \begin{bmatrix}
    \mat{A}^\top \mat{A} & \frac{1}{2}\mat{C}^\top \\
    \mat{C} & \mat{0}
    \end{bmatrix}
    \begin{bmatrix}
    \vec{x} \\
    \vec{\lambda}
    \end{bmatrix}
    =
    \begin{bmatrix}
    \mat{A}^\top \vec{y} \\
    \vec{d}
    \end{bmatrix}

Where:

- :math:`\mat{A}^\top \mat{A}`: The Hessian matrix of the least squares problem.
- :math:`\mat{0}`: A zero matrix of size :math:`p \times p`.

First, solve for :math:`\vec{x}` using the equation in the first row:

.. math:: \vec{x} = (\mat{A}^\top \mat{A})^{-1}\left(\mat{A}^\top \vec{y}-\frac{1}{2}\mat{C}^\top\right).

Second, substitute the above equation into the constraint:

.. math:: \mat{C}(\mat{A}^\top \mat{A})^{-1}\left(\mat{A}^\top \vec{y}-\frac{1}{2}\mat{C}^\top\vec{\lambda}\right) = \vec{d}.

Third, solve for the lagrange multipliers:

.. math:: \frac{1}{2}\vec{\lambda} = \left(\mat{C}(\mat{A}^\top \mat{A})^{-1} \mat{C}^\top \right)^{-1}\left(\mat{C}(\mat{A}^\top \mat{A})^{-1}\mat{A}^\top\vec{y} -\vec{d}\right)

Finally, substitute :math:`\vec{\lambda}` into the expression for :math:`\vec{x}`:

.. math:: \vec{x} = (\mat{A}^\top \mat{A})^{-1}\left(\mat{A}^\top \vec{y}-\mat{C}^\top \left(\mat{C}(\mat{A}^\top \mat{A})^{-1} \mat{C}^\top \right)^{-1}\left(\mat{C}(\mat{A}^\top \mat{A})^{-1}\mat{A}^\top\vec{y} -\vec{d}\right)\right)

The solution can be written in terms of the uncontrained least squares solution :math:`\vec{\gamma}=(\mat{A}^\top \mat{A})^{-1}\mat{A}^\top \vec{y}`:

.. math:: \vec{x} = \vec{\gamma}+(\mat{A}^\top \mat{A})^{-1}\left(\mat{C}^\top \left(\mat{C}(\mat{A}^\top \mat{A})^{-1} \mat{C}^\top \right)^{-1}\left(\vec{d}-\mat{C}\vec{\gamma}\right)\right)

Solving the constrained problem
-------------------------------
"""

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.util.backends.numpy import NumpyMixin as bkd
from pyapprox.surrogates.affine.linearsystemsolvers import (
    LstSqSolver,
    LinearlyConstrainedLstSqSolver,
)
from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.surrogates.affine.basisexp import MonomialExpansion
from pyapprox.surrogates.affine.basis import MultiIndexBasis

# Set seed for reporducibility
np.random.seed(3)

# Define the problem
nsamples, nbasis, nconstraints = 10, 5, 2

# Generate the training data
train_samples = bkd.asarray(np.random.uniform(0.0, 1.0, (1, nsamples)))
true_coef = bkd.asarray(np.random.normal(0.0, 1.0, (nbasis, 1)))

true_basis = MultiIndexBasis([Monomial1D(backend=bkd)])
true_basis.set_tensor_product_indices([nbasis])
true_basisexp = MonomialExpansion(true_basis, solver=None, nqoi=1)
true_basisexp.set_coefficients(true_coef)
train_values = true_basisexp(train_samples)
# add noise so surrogate will not interpolate data exactly
# except where constraints are enforved
train_values += bkd.asarray(np.random.normal(0, 0.1, train_values.shape))

# Define the constraints
constraint_samples = bkd.array([[0, 1]])
C = true_basis(constraint_samples)
d = C @ true_coef

# Solve the constrained least-squares problem
constrained_basis = MultiIndexBasis([Monomial1D(backend=bkd)])
constrained_basis.set_tensor_product_indices([nbasis])
constrained_basisexp = MonomialExpansion(
    constrained_basis, solver=None, nqoi=1
)
solver = LinearlyConstrainedLstSqSolver(C, d, backend=bkd)
constrained_basisexp.set_solver(solver)
constrained_basisexp.fit(train_samples, train_values)

# Solve the unconstrainted least-squares problem
unconstrained_basis = MultiIndexBasis([Monomial1D(backend=bkd)])
unconstrained_basis.set_tensor_product_indices([nbasis])
unconstrained_basisexp = MonomialExpansion(
    unconstrained_basis, solver=None, nqoi=1
)
solver = LstSqSolver(backend=bkd)
unconstrained_basisexp.set_solver(solver)
unconstrained_basisexp.fit(train_samples, train_values)

# Plot the surrogates and the data
plot_samples = bkd.linspace(0, 1, 101)[None, :]
ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.plot(train_samples[0], train_values[:, 0], "o", label="Training Data")
constrained_basisexp.plot_surface(ax, [0, 1], 101, label="Constrained LstSq")
unconstrained_basisexp.plot_surface(
    ax, [0, 1], 101, label="Unconstrained LstSq"
)
true_basisexp.plot_surface(ax, [0, 1], 101, label="Noiseless Function")
ax.legend()
ax.set_xlabel(r"$z$")
_ = ax.set_ylabel(r"$f(z$)")

# %%
# Key Notes:
# ---------
#
# Block Matrix:
#
# The block matrix formulation ensures that both the least squares objective and the linear constraints are satisfied.
#
# Numerical Stability:
#
# Solving the block system directly can be computationally expensive for large problems. Consider using iterative solvers for scalability.
#
#
# Applications:
#
# Linearly constrained least squares is widely used in optimization, control systems, and machine learning.
#
# Conclusion
# ----------
# The linearly constrained linear least squares problem extends the standard least squares problem by incorporating linear constraints. By using the method of Lagrange multipliers, we derive a block system that can be solved efficiently using numerical linear algebra techniques. This formulation is powerful for solving constrained optimization problems in various fields.
