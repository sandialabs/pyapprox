Tensor product Lagrange interpolation
=====================================

Let :math:`\hat{f}_{\boldsymbol{\alpha},\boldsymbol{\beta}}(\mathbf{z})` be an M-point tensor-product interpolant of the function :math:`\hat{f}_{\boldsymbol{\alpha}}`. This interpolant is a weighted linear combination of tensor-product of univariate Lagrange polynomials

.. math:: \phi_{i,j}(z_i) = \prod_{k=1,k\neq j}^{m_{\beta_i}}\frac{z_i-z_i^{(k)}}{z_i^{(j)}-z_i^{(k)}}, \quad i\in[d],


defined on a set of univariate points :math:`z_{i}^{(j)},j\in[m_{\beta_i}]`  Specifically the multivariate interpolant is given by

.. math:: \hat{f}_{\boldsymbol{\alpha},\boldsymbol{\beta}}(\mathbf{z}) = \sum_{\boldsymbol{j}\le\boldsymbol{\beta}} \hat{f}_{\boldsymbol{\alpha}}(\mathbf{z}^{(\boldsymbol{j})})\prod_{i\in[d]}\phi_{i,j_i}(z_i).

The partial ordering :math:`\boldsymbol{j}\le\boldsymbol{\beta}` is true if all the component wise conditions are true.

Constructing the interpolant requires evaluating the function :math:`\hat{f}_{\boldsymbol{\alpha}}` on the grid of points

.. math::

   \mathcal{Z}_{\boldsymbol{\beta}} = \bigotimes_{i=1}^d \mathcal{Z}_{\beta_i}^i=\begin{bmatrix}\mathbf{z}^{(1)} & \cdots&\mathbf{z}^{(M_{\boldsymbol{\beta}})}\end{bmatrix}\in\mathbb{R}^{d\times M_{\boldsymbol{\beta}}}

	  
We denote the resulting function evaluations by

.. math:: \mathcal{F}_{\boldsymbol{\alpha},\boldsymbol{\beta}}=\hat{f}_{\boldsymbol{\alpha}}(\mathcal{Z}_{\boldsymbol{\beta}})=\begin{bmatrix}\hat{f}_{\boldsymbol{\alpha}}(\mathbf{z}^{(1)}) \quad \cdots\quad \hat{f}_{\boldsymbol{\alpha}}(\mathbf{z}^{(M_{\boldsymbol{\beta}})})\end{bmatrix}^T\in\mathbb{R}^{M_{\boldsymbol{\beta}}\times q},

where the number of points in the grid is :math:`M_{\boldsymbol{\beta}}=\prod_{i\in[d]} m_{\beta_i}`

It is often reasonable to assume that, for any :math:`\mathbf{z}`, the cost of each simulation is constant for a given :math:`\boldsymbol{\alpha}`. So letting :math:`W_{\boldsymbol{\alpha}}` denote the cost of a single simulation, we can write the total cost of evaluating the interpolant :math:`W_{\boldsymbol{\alpha},\boldsymbol{\beta}}=W_{\boldsymbol{\alpha}} M_{\boldsymbol{\beta}}`. Here we have assumed that the computational effort to compute the interpolant once data has been obtained is negligible, which is true for sufficiently expensive models :math:`\hat{f}_{\boldsymbol{\alpha}}`.
In this paper, we use the nested Clenshaw-Curtis points

.. math::
      
  z_{i}^{(j)}=\cos\left(\frac{(j-1)\pi}{m_{\beta_i}}\right),\qquad j=1,\ldots,m_{\beta_i}

to define the univariate Lagrange polynomials. The number of points :math:`m(l)` of this rule grows exponentially with the level :math:`l`, specifically
:math:`m(0)=1` and :math:`m(l)=2^{l}+1` for :math:`l\geq1`. The univariate Clenshaw-Curtis points, the tensor-product grid :math:`\mathcal{Z}_{\boldsymbol{\beta}}`, and two multivariate Lagrange polynomials with their corresponding univariate Lagrange polynomials are shown below for :math:`\boldsymbol{\beta}=(2,2)`.

.. plot::
      
   from pyapprox.examples.tensor_product_lagrange_interpolation import *
   fig = plt.figure(figsize=(2*8,6))
   ax=fig.add_subplot(1,2,1,projection='3d')
   level = 2; ii=1; jj=1
   plot_tensor_product_lagrange_basis_2d(level,ii,jj,ax)

   ax=fig.add_subplot(1,2,2,projection='3d')
   level = 2; ii=1; jj=3
   plot_tensor_product_lagrange_basis_2d(level,ii,jj,ax)
   plt.show()

   
