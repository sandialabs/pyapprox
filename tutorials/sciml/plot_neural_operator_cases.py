r"""
CERTANN Special Cases
=====================


Kernel neural operators have the general from

.. math::

   y_{k+1}(z_{k+1})&=\sigma_k\left(\int_{\mathcal{D}_{k}} \mathcal{K}_{k}
   (z_{k+1}, z_{k}; \theta_{k}) y_{k}(z_{k}) \dx{\mu_{k}(z_{k})})\right)\\
   &=\sigma_k\left(u_{k+1}(z_{k+1})\right)


Dense Multi-layer Perceptron
----------------------------
Dense MLPs can be recovered by using a piecewise constant quadrature rule with

.. math::
    x^{(n)}=x^{(0)}+\Delta x, \quad n=1,\ldots,N-1, \qquad w^{(n)}=\Delta x

and the kernel

.. math::
    K(x,y) = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} \alpha_{mn}
    \chi_{x^{(n)}}(x)\chi_{y^{(n)}}(y)

where

.. math:: \chi_{x^{(n)}}(x)=\begin{cases}
   1 & x^{(n)}\le x < x^{(n)}+\Delta x\\
   0 & \text{otherwise}
  \end{cases}

Evaluating the kernel at the quadrature points for :math:`x` and :math:`y`
yields the typically dense weight matrics of neural networks where the weights
are statistically independent.


Fourier Neural Operator
-----------------------

Classic FNOs use the kernel

.. math::
    K(x-y) = \sum_{n=0}^{N-1}\alpha_n \phi_n(x-y) = \sum_{n=0}^{N-1}
    \alpha_n\exp\left(\mathrm{i}(x-y)\omega_n\right)

where the Fourier coefficients :math:`\alpha_n` are learnt directly in the
Fourier space. The Fourier convolution theorem is used to compute the integral
of the integral operator form.


Chebyshev Neural Operator
-------------------------

In line with classic FNOs, ChebNOs use the kernel

.. math::
    K(x,y) = \sum_{n=0}^{N-1} \alpha_n \phi_n(x-y) = \sum_{n=0}^{N-1}
    \alpha_n T_n(x-y)

where :math:`T_n` is the  Chebyshev polynomial of degree :math:`n`, and the
Chebyshev coefficients :math:`\alpha_n` are learnt directly in the Chebyshev
space.

The Chebyshev convolution theorem is used to compute the integral of the
integral operator form.


.. _tensor-product-kernel:

Tensor-Product Kernel
---------------------

A tensor-product kernel is useful for kernels that are not
translation-invariant:

.. math::
    K(x,y) = \mathbf{\Phi}^{\top} (x) \, \mathbf{A} \, \mathbf{\Phi}(y)

where :math:`\mathbf{\Phi}: \Omega \to \reals^N`, :math:`\Omega \subset
\reals`, and :math:`\mathbf{A} \in \reals^{N \times N}` is symmetric.
For each :math:`x \in \Omega`, :math:`\mathbf{\Phi}(x)` is a
vector of basis functions

.. math:: (\mathbf{\Phi}(x))_n = \phi_n(x) \, .

The matrix :math:`\mathbf{A}` determines the coefficients and basis
combinations that appear in :math:`K`. For computational efficiency, we choose
:math:`\phi_n(x)` to be orthogonal with respect to the integration measure
:math:`\dx{\mu(x)} = w(x) \dx{x}`. Importantly, one must multiply the final
output layer by :math:`w(x)` **even though no integral layers are left** since
the least-squares problem is in :math:`L^2_\mu(\Omega)`. If this is missing, we
observe degraded accuracy in practice.

Here, the coefficients :math:`a_{ij}` are learned in the original space, and we
only need the upper triangle since :math:`\mathbf{A}` is symmetric. In
contrast to convolutional kernels, which have :math:`O(N)` parameters, there
are in general :math:`O(N^2)` parameters for a tensor-product kernel. Problem
settings may allow sparsity assumptions that limit number of learnable
parameters:

* :math:`\mathbf{A}` is diagonal;
* :math:`\mathbf{A}` is banded;
* :math:`\mathbf{A}` is a lower-complete set (e.g., hyperbolic cross).


.. _chebyshev-tensor-product-kernel:

Chebyshev Tensor-Product Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case,

.. math::
    \phi_n(x) = T_n(x), \qquad \dx{\mu} = \frac{\dx{x}}{\sqrt{1-x^2}}, \qquad
    \Omega = [-1,1].

A single-layer CERTANN learns the map :math:`f \mapsto u`, given by

.. math::
    u(x) &= w(x) \int_{-1}^1 K(x,y) f(y) \dx{\mu(y)} \\
    &= w(x) \int_{-1}^1 \mathbf{\Phi}^\top (x) \mathbf{A} \mathbf{\Phi}(y) f(y)
        \dx{\mu(y)} \\
    &= w(x) \mathbf{\Phi}^\top (x) \mathbf{A} \int_{-1}^1 \mathbf{\Phi}(y) f(y)
        \dx{\mu} \, .

We can compute the integrals in :math:`\mathcal{O}(N \log N)` time with the
:ref:`inner product property <chebyshev-transform-inner-product>` of the
Chebyshev transform:

.. math::
    \int_{-1}^1 T_n(x) f(x) \dx{\mu} =
    \begin{cases} \pi \, \mathcal{T}[\mathbf{f}]_n, & n=0 \\
    (\pi/2) \, \mathcal{T}[\mathbf{f}]_n, & n>0 \end{cases} \ .
"""
