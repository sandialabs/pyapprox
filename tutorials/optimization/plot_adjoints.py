r"""
Adjoint Method
==============

In this tutorial, we will explore how to compute gradients using the adjoint method. The adjoint method is a powerful technique for efficiently computing gradients of an objective function with respect to design parameters, especially in optimization problems constrained by partial differential equations (PDEs).

Definition
----------

Adopt the convention that the Jacobian of a vector-valued function :math:`f(x)\in\mathbb{R}^{M}` with respect to the input :math:`x\in\mathbb{R}^{N}` is:

.. math::
   \nabla_x f = \begin{bmatrix}
   \frac{\partial f_1}{\partial x_1} &  \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_N}\\
   \frac{\partial f_2}{\partial x_1} &  \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_N}\\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial f_M}{\partial x_1} &  \frac{\partial f_M}{\partial x_2} & \cdots & \frac{\partial f_M}{\partial x_N}
   \end{bmatrix} \in \mathbb{R}^{M \times N}

Let :math:`J(y, \theta)` be the objective function, where:

- :math:`\theta` are the design parameters,
- :math:`y` is the solution that satisfies the constraint:

.. math:: c(y, \theta) = 0

We want to minimize:

.. math:: \mathcal{L}(y, \theta) = J(y, \theta) + \lambda^\top c(y, \theta)

Setting the derivative of :math:`\mathcal{L}` with respect to :math:`y` to zero yields:

.. math::
   \begin{align*}
   \frac{\partial J}{\partial y} + \lambda^\top \frac{\partial c}{\partial y} &= 0 \\
   \left(\frac{\partial c}{\partial y}\right)^\top \lambda &= -\frac{\partial J}{\partial y}^\top \\
   \lambda &= -\left(\frac{\partial c}{\partial y}\right)^{-\top} \frac{\partial J}{\partial y}^\top
   \end{align*}

The last expression is the adjoint equation, and :math:`\lambda` is the adjoint solution.

Taking the derivative of :math:`J` with respect to :math:`\theta` yields:

.. math::
   \frac{d J}{d \theta} = \frac{\partial J}{\partial \theta} + \frac{\partial J}{\partial y} \frac{\partial y}{\partial \theta} + \lambda^\top \left( \frac{\partial c}{\partial \theta} + \frac{\partial c}{\partial y} \frac{\partial y}{\partial \theta} \right)

Substituting the definition of the adjoint solution :math:`\lambda` simplifies this to:

.. math::
   \frac{d J}{d \theta} = \frac{\partial J}{\partial \theta} + \lambda^\top \frac{\partial c}{\partial \theta}


Gradient Computation
--------------------

To solve the optimization problem:

.. math:: \min_\theta J(y(\theta), \theta) \quad \text{s.t.} \quad c(y(\theta), \theta) = 0

The gradient :math:`\nabla_\theta J` can be computed in three steps:

**Step 1**: Solve the forward equation:

.. math:: c(y(\theta), \theta) = 0

where:

- :math:`c(y(\theta), \theta) \in \mathbb{R}^{M \times 1}`,
- :math:`\theta \in \mathbb{R}^{N \times 1}`.

**Step 2**: Solve the adjoint equation for :math:`\lambda`:

.. math:: \nabla_y c(y(\theta), \theta)^\top \lambda = -\nabla_y J(y(\theta), \theta)^\top

where:

- :math:`\nabla_y c(y(\theta), \theta) \in \mathbb{R}^{M \times M}`,
- :math:`\lambda \in \mathbb{R}^{M \times 1}`,
- :math:`\nabla_y J(y(\theta), \theta) \in \mathbb{R}^{1 \times M}`.

**Step 3**: Compute:

.. math:: \nabla_\theta J(\theta) = \nabla_\theta J(y(\theta), \theta) + \lambda^\top \nabla_\theta c(y(\theta), \theta)

where:

- :math:`\nabla_\theta c(y(\theta), \theta) \in \mathbb{R}^{M \times N}`,
- :math:`\nabla_\theta J(y(\theta), \theta) \in \mathbb{R}^{1 \times N}`,
- :math:`\nabla_\theta J(\theta) \in \mathbb{R}^{1 \times N}`.

Hessian-Vector Computation
--------------------------

The Hessian multiplied by a vector :math:`v \in \mathbb{R}^N` can be computed in 5 steps.

First, define:

.. math:: L(y(\theta), \theta, \lambda) = J(y(\theta), \theta) + \lambda^\top c(y(\theta), \theta)

**Step 1**: Solve the forward equation (same as for gradient computation):

.. math:: c(y(\theta), \theta) = 0

**Step 2**: Solve the adjoint equation (same as for gradient computation):

.. math:: \nabla_y c(y(\theta), \theta)^\top \lambda = -\nabla_y J(y(\theta), \theta)^\top

**Step 3**: Solve for :math:`w`:

.. math:: \nabla_y c(y(\theta), \theta) w = \nabla_\theta c(y(\theta), \theta) v

**Step 4**: Solve for :math:`p`:

.. math:: \nabla_y c(y(\theta), \theta)^\top p = \nabla_{yy} L(y(\theta), \theta, \lambda) w - \nabla_{y\theta} L(y(\theta), \theta, \lambda) v

**Step 5**: Compute the Hessian-vector product:

.. math:: \nabla_{\theta\theta} J v = \nabla_\theta c(y(\theta), \theta)^\top p - \nabla_{\theta y} L(y(\theta), \theta, \lambda) w + \nabla_{\theta\theta} L(y(\theta), \theta, \lambda) v

These steps allow us to compute the Hessian-vector product efficiently without explicitly forming the Hessian matrix, which is particularly useful for high-dimensional problems.
"""
