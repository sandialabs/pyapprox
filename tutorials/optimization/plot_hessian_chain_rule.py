r"""
Hessian Chain Rule
==================

In this tutorial, we will explore the Hessian chain rule, which is used to compute second-order derivatives for composite functions. The Hessian chain rule involves tensor contractions, which are higher-dimensional generalizations of matrix products.

Mathematical Formulation
------------------------

Consider the functions:

.. math::
   f(x): \mathbb{R}^n \to \mathbb{R}, \quad g(y) = (g_1, \ldots, g_m): \mathbb{R}^m \to \mathbb{R}^n

First, the gradient of :math:`f(g(y))` with respect to :math:`y` is given by:

.. math::
   \nabla_y f(g(y)) = \nabla_g f(g(y)) \cdot \nabla_y g(y)

Differentiating the above again yields:

.. math::
   \nabla_{yy} f(g(y)) = \nabla_y\left(\nabla_g f(g(y)) \cdot \nabla_y g(y)\right)

Expanding this expression:

.. math::
   \nabla_{yy} f(g(y)) = \left(\nabla_{gg} f(g(y)) : \nabla_y g(y)\right) : \nabla_y g(y) + \nabla_{g} f(g(y)) : \nabla_{yy} g(y)

Here, :math:`\nabla_{yy} g(y)` is a 3D tensor. The colons (:math:`:`) above represent **tensor contractions**, which are higher-dimensional generalizations of matrix products.

Tensor Contractions
-------------------

Tensor contractions allow us to multiply tensors of different dimensions. For example:

.. math::
   (X : Y)[i, j] = \sum_k X[i, k] Y[k, j]

Using subscript notation:

.. math::
   (X : Y)_{i, j} = \sum_k X_{i, k} Y_{k, j}

Applying this to the Hessian chain rule, we rewrite the expression:

.. math::
   \nabla_{yy} f(g(y)) = \left(\nabla_{gg} f(g(y)) : \nabla_y g(y)\right) : \nabla_y g(y) + \nabla_{g} f(g(y)) : \nabla_{yy} g(y)

Using derivative notation:

.. math::
   \nabla_{yy} f(g(y)) = (D^2[f] : D^1[g]) : D^1[g] + D^1[f] : D^2[g]

Here, :math:`D^s[f]` denotes the :math:`s`-th derivative of :math:`f`. For example:

- :math:`D^1[f] = J[f]` (Jacobian),
- :math:`D^2[f] = H[f]` (Hessian).

Tensor Contraction Example
--------------------------

The following is an example of a tensor contraction:

.. math::
   (X : Y)[i, j] = \sum_k X[i, k] Y[k, j]

Substituting into the Hessian chain rule:

.. math::
   H[f(g(y))] = (J[g])^\top \cdot H[f] \cdot J[g] + \sum_{k=1}^n \frac{\partial f}{\partial y^k} \cdot H[g^k]

This expression uses matrix multiplication and tensor contractions to compute the Hessian.

Hessian-Vector Product Example
------------------------------

Consider the residual:

.. math::
   r(p) = u f(p) \in \mathbb{R}^n

where :math:`u, g \in \mathbb{R}^n` and :math:`u` is not a function of :math:`p \in \mathbb{R}^m`. Let:

.. math::
   f = \exp(g(p)), \quad g(p) = \Phi \cdot p

We want to compute the Hessian of the Lagrange multiplier multiplied by the constraint :math:`r`:

.. math::
   c = \lambda^\top \cdot r(p) \in \mathbb{R}

with :math:`\lambda \in \mathbb{R}^n`. Let :math:`U = \mathrm{Diag}(u)`. Then:

.. math::
   D^1_p[c] = \lambda^\top \cdot U \cdot D^1[f]

Substituting the derivatives:

.. math::
   D^1_p[c] = \lambda^\top \cdot U \cdot \mathrm{Diag}[\exp(g)] \cdot D^1[g]
   = \lambda^\top \cdot U \cdot \mathrm{Diag}[\exp(g)] \cdot \Phi
   = (\lambda \circ u \circ \exp(g))^\top \cdot \Phi \in \mathbb{R}^{(1, p)}

Here, :math:`\circ` is the elementwise Hadamard product. For :math:`v \in \mathbb{R}^p`:

.. math::
   D^1_p[c] \cdot v = (\lambda \circ u \circ \exp(g))^\top \cdot \Phi \cdot v \in \mathbb{R}

Since this is a scalar, we can take the transpose:

.. math::
   D^1_p[c] \cdot v = v^\top \cdot \Phi^\top (\lambda \circ u \circ \exp(g)) \in \mathbb{R}

Finally:

.. math::
   D^2_p[c] \cdot v = D^1_p\left[D^1_p[c] \cdot v\right] = \left[(v^\top \cdot \Phi^\top) \circ (\lambda \circ u \circ \exp(g))^\top\right] \Phi \in \mathbb{R}

Miscellaneous
-------------

For :math:`\nabla^2_g[f(g)]`, we have:

.. math::
   J_p[g] = \Phi \in \mathbb{R}^{(n, p)}

.. math::
   J_g[f] = \mathrm{Diag}[\exp(g(p))] \in \mathbb{R}^{(n, n)}

.. math::
   H_g[f]_{i,j,k} =
   \begin{cases}
   \exp(g_i(p)), & \text{if } i = j = k \\
   0, & \text{otherwise}
   \end{cases}
   \in \mathbb{R}^{(n, n, n)}
"""
