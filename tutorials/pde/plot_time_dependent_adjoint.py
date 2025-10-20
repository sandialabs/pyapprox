r"""
Computing Gradients of ODEs Using Adjoints
==========================================

In this tutorial, we will explore the computation of gradients using adjoints for time-dependent problems. For linear time-dependent residuals, the adjoint method can be formulated using linear algebra, and solving adjoints becomes analogous to integrating an ODE backward in time.

This tutorial assumes familarity with the introductory tutorial on computing gradients using adjoints for arbitrary functions.

Intuition
---------

Solving adjoints for time-dependent problems is akin to integrating an ODE backward in time. For example, consider the ODE parameterized by optimization variable :math:`\theta`

.. math:: \dydx{y}{t} = f(y, \theta)

Here we focus on computing the gradient of an objective function that depends on :math:`y` and :math:`\theta` by solving adjoints of the discretized ODE. However, it must be noted that this is not equivalent to solving adjoint equation of an ODE. The gradients obtained by these two different approaches will only coincide, when the time step asymptotes to zero. Up to numerical precision issues the approach described here will return the gradients of the discretized ODE that would also be obtained using automatic differentiation. However, the approach used here will be faster and more memory efficient as it takes advantage of the structure of ODE time stepping schemes.

Using using backward Euler to evolve a linear version of this ODE for two time-steps from an initial condition looks like solving the set of simultaneous equations:

.. math::

   c(y, \theta) =
   \begin{bmatrix}
     A_{00}(t_0) & 0 & 0\\
     A_{10}(t_0) & A_{11}(t_1) & 0\\
     0 & A_{21}(t_1) & A_{22}(t_2)
   \end{bmatrix}
   \begin{bmatrix}
   y_0\\ y_1\\ y_2
   \end{bmatrix}
   -
   \begin{bmatrix}
   b_0\\ b_1\\ b_2
   \end{bmatrix}

We can apply the three gradient steps to formulate and solve the time-dependent discretized adjoint equation:

.. math::

   \begin{bmatrix}
     A_{00}(t_0)^\top & A_{10}(t_0)^\top & 0\\
     0 & A_{11}(t_1)^\top & A_{21}(t_1)^\top\\
     0 & 0 & A_{22}(t_2)^\top
   \end{bmatrix}
   \begin{bmatrix}
   \lambda_0\\ \lambda_1\\ \lambda_2
   \end{bmatrix}
   =
   \begin{bmatrix}
   \nabla_{y_0} J(y_0, \theta)\\ \nabla_{y_1} J(y_1, \theta)\\ \nabla_{y_2} J(y_2, \theta)
   \end{bmatrix}

The gradient can then be computed using the usual formula

.. math:: \nabla_\theta J(\theta) = \nabla_\theta J(y(\theta), \theta) + \lambda^\top \nabla_\theta c(y(\theta), \theta)

Example: Linear ODE
-------------------

Let :math:`\theta = [a, b]^\top` and consider:

.. math:: \frac{\mathrm{d}y}{\mathrm{d}t} = by \quad y(0) = a

With the exact solution:

.. math:: y(t, \theta) = a \exp(bt)

First, focus on computing the gradient of the functional:

.. math:: J(\theta) = \int_0^T y(t, \theta) \, \mathrm{d}t = \int_0^T a \exp(bt) \, \mathrm{d}t = \frac{a}{b}(\exp(bT) - 1)

i.e.,

.. math:: \frac{\mathrm{d}J}{\mathrm{d}\theta} = 
   \begin{bmatrix}
   \frac{1}{b}(\exp(bT) - 1) \\
   \frac{aT}{b} \exp(bT) - \frac{a}{b^2}(\exp(bT) - 1)
   \end{bmatrix}




Step 1: Compute Forward Solution with Backward Euler
----------------------------------------------------

.. math::
   y = \begin{bmatrix}
   y^{(0)} \\ y^{(1)} \\ y^{(2)}
   \end{bmatrix}
   =
   \begin{bmatrix}
   a \\
   \frac{y^{(0)}}{\Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1} \\
   \frac{y^{(1)}}{\Delta^{(2)} b^{2} \left(t_{2} + 2\right) + 1}
   \end{bmatrix}

.. math::
   y = \begin{bmatrix}
   a \\
   \frac{a}{\Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1} \\
   \frac{a}{\left(\Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1\right) \left(\Delta^{(2)} b^{2} \left(t_{2} + 2\right) + 1\right)}
   \end{bmatrix}

Step 2: Compute Adjoint Solution
--------------------------------

.. math::
   \lambda = -\left(\nabla_y C(y(\theta), \theta)^\top\right)^{-1} \nabla_y J(y(\theta), \theta)^\top

Define Objective
----------------
Backward Euler integration corresponds to right-sided piecewise constant quadrature in time so that

.. math:: J(y,\theta)=\int_0^T y(t,\theta) \dx{t} \approx\sum_{n=1}^N \Delta^{(n)} y^{(n)}

Here the sum starts from 1 and not 0 because the piecewise constant quadrature is right sided.


Thus, for our 2 step (and initial condition)

.. math::
   J(y(\theta), \theta) = \Delta^{(1)} y^{(1)} + \Delta^{(2)} y^{(2)}

Compute Necessary Quantities
----------------------------

.. math::
   C(y(\theta), \theta) =
   \begin{bmatrix}
   - a + y^{(0)} \\
   \Delta^{(1)} b^{2} y^{(1)} \left(t_{1} + 2\right) - y^{(0)} + y^{(1)} \\
   \Delta^{(2)} b^{2} y^{(2)} \left(t_{2} + 2\right) - y^{(1)} + y^{(2)}
   \end{bmatrix}
   = 0

.. math::
   \nabla_y C(y(\theta), \theta) =
   \begin{bmatrix}
   1 & 0 & 0 \\
   -1 & \Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1 & 0 \\
   0 & -1 & \Delta^{(2)} b^{2} \left(t_{2} + 2\right) + 1
   \end{bmatrix}

.. math::
   \nabla_y J(y(\theta), \theta) =
   \begin{bmatrix}
   0 & \Delta^{(1)} & \Delta^{(2)}
   \end{bmatrix}

Compute via Backward Substitution
---------------------------------

Solve the adjoint equation using backward substitution, i.e., time step backward in time:

.. math::
   \begin{bmatrix}
   1 & -1 & 0 \\
   0 & \Delta^{(1)} b^{2} \left(t_{1} + 2\right) + 1 & -1 \\
   0 & 0 & \Delta^{(2)} b^{2} \left(t_{2} + 2\right) + 1
   \end{bmatrix}
   \lambda =
   -\begin{bmatrix}
   0 \\ \Delta^{(1)} \\ \Delta^{(2)}
   \end{bmatrix}

.. math::
   \lambda =
   \begin{bmatrix}
   \frac{- \Delta^{(1)} \Delta^{(2)} b^{2} t_{2} - 2 \Delta^{(1)} \Delta^{(2)} b^{2} - \Delta^{(1)} - \Delta^{(2)}}{\Delta^{(1)} \Delta^{(2)} b^{4} t_{1} t_{2} + 2 \Delta^{(1)} \Delta^{(2)} b^{4} t_{1} + 2 \Delta^{(1)} \Delta^{(2)} b^{4} t_{2} + 4 \Delta^{(1)} \Delta^{(2)} b^{4} + \Delta^{(1)} b^{2} t_{1} + 2 \Delta^{(1)} b^{2} + \Delta^{(2)} b^{2} t_{2} + 2 \Delta^{(2)} b^{2} + 1} \\
   \frac{- \Delta^{(1)} \Delta^{(2)} b^{2} t_{2} - 2 \Delta^{(1)} \Delta^{(2)} b^{2} - \Delta^{(1)} - \Delta^{(2)}}{\Delta^{(1)} \Delta^{(2)} b^{4} t_{1} t_{2} + 2 \Delta^{(1)} \Delta^{(2)} b^{4} t_{1} + 2 \Delta^{(1)} \Delta^{(2)} b^{4} t_{2} + 4 \Delta^{(1)} \Delta^{(2)} b^{4} + \Delta^{(1)} b^{2} t_{1} + 2 \Delta^{(1)} b^{2} + \Delta^{(2)} b^{2} t_{2} + 2 \Delta^{(2)} b^{2} + 1} \\
   - \frac{\Delta^{(2)}}{\Delta^{(2)} b^{2} t_{2} + 2 \Delta^{(2)} b^{2} + 1}
   \end{bmatrix}


Let :math:`y` be the concatenation of all states of the discretized ODE, that is:

.. math::
   y = [y^{(0)}, y^{(1)}, \ldots, y^{(N)}]^\top

Then the linearity of the ODE leads to the following matrix equation for :math:`N=3` (including the initial condition) computed using backward Euler time-stepping:

.. math::
   c(y, \theta) =
   \begin{bmatrix}
     1 & 0 & 0\\
     -1 & (1+\Delta^{(1)} b^2) & 0\\
     0 & -1 & (1+\Delta^{(2)} b^2)
   \end{bmatrix}
   \begin{bmatrix}
     y^{(0)}\\ y^{(1)}\\ y^{(2)}
   \end{bmatrix}
   -
   \begin{bmatrix}
     a\\ 0\\ 0
   \end{bmatrix}
   = 0

This is a lower-diagonal matrix, which can be solved by forward substitution, equivalent to the time-stepping scheme laid out previously.

Gradient Computation
--------------------

The gradient computation involves the following matrices:

.. math::
   \nabla_y c(y, \theta) =
   \begin{bmatrix}
     1 & 0 & 0\\
     -1 & (1+\Delta^{(1)} b^2) & 0\\
     0 & -1 & (1+\Delta^{(2)} b^2)
   \end{bmatrix}, \quad
   \nabla_y J(y, \theta) =
   \begin{bmatrix}
   0 & \Delta^{(1)} & \Delta^{(2)}
   \end{bmatrix}

.. math::
   \nabla_\theta c(y, \theta) =
   \begin{bmatrix}
   0 & -1\\
   2\Delta^{(1)} b y^{(1)} & 0\\
   2\Delta^{(2)} b y^{(2)} & 0
   \end{bmatrix}

.. math::
   \nabla_\theta J(y, \theta) =
   \begin{bmatrix}
   0 & 0
   \end{bmatrix}


Adjoint Equation
----------------

Solve the adjoint equation using backward substitution:

.. math::
   -\nabla_y c(y(\theta), \theta)^\top \lambda = \nabla_y J(y(\theta), \theta)^\top

This is equivalent to time-stepping backward in time from the final solution:

.. math::
   \begin{align*}
   \lambda^{(2)} &= -\Delta^{(2)}(1+\Delta^{(2)} b^2)^{-1} \\
   -\lambda^{(1)}(1+\Delta^{(1)} b^2) + \lambda^{(2)} &= -\Delta^{(1)} \\
   \lambda^{(1)} &= (-\Delta^{(2)}(1+\Delta^{(2)} b^2)^{-1} - \Delta^{(1)})(1+\Delta^{(1)} b^2)^{-1} \\
   &= (-\Delta^{(2)} - \Delta^{(1)}(1+\Delta^{(2)} b^2))(1+\Delta^{(1)} b^2)^{-1}(1+\Delta^{(2)} b^2)^{-1} \\
   \lambda^{(0)} &= \lambda^{(1)}
   \end{align*}


Final Gradient Computation
--------------------------

Finally, compute the gradient:

.. math::
   \nabla_\theta J(\theta) = \nabla_\theta J(y(\theta), \theta) + \lambda^\top \nabla_\theta c(y(\theta), \theta)

Substituting the computed values:

.. math::
   \nabla_\theta J(\theta) =
   \begin{bmatrix}
   0 \\ 0
   \end{bmatrix} +
   \begin{bmatrix}
   \frac{2 b a \left(\Delta^{(1)} \Delta^{(2)} \left(2 \Delta^{(2)} b^2+1\right)+\left(\Delta^{(1)} \Delta^{(2)} b^2+\Delta^{(1)}\right)^2+(\Delta^{(2)})^2\right)}{\left(1+{\Delta^{(1)}} b^2\right)^2 \left(1+{\Delta^{(2)}} b^2\right)^2} \\
   \frac{{\Delta^{(1)}} {\Delta^{(2)}} b^2+{\Delta^{(1)}}+{\Delta^{(2)}}}{\left(1+{\Delta^{(1)}} b^2\right) \left(1+{\Delta^{(2)}} b^2\right)}
   \end{bmatrix}

We can verify these gradients by taking the gradient of the objective directly:

.. math::
   J(y, \theta) = \int_0^T y(t, \theta) dt \approx \sum_{n=1}^N \Delta^{(n)} y^{(n)}

Taking the gradient with respect to :math:`a`:

.. math::
   \nabla_a J(y, \theta) = \Delta^{(1)}(1+\Delta^{(1)} b)^{-1} + \Delta^{(2)}(1+\Delta^{(2)} b)^{-1}(1+\Delta^{(1)} b)^{-1}

This matches the derived expression for :math:`\nabla_\theta J(\theta)` above.
"""
