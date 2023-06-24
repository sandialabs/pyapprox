r"""
Sparse Grids
============
The number of model evaluations required by tensor product interpolation grows exponentitally with the number of model inputs. This tutorial introduces sparse grids [BNR2000]_, [BG2004]_ which can be used to overcome the so called curse of dimensionality faced by tensor-product methods.

Sparse grids approximate a model (function) :math:`f_\alpha` with :math:`D` inputs :math:`z=[z_1,\ldots,z_D]^\top` as a linear combination of low-resolution tensor product interpolantsm that is

.. math:: f_{\alpha, \mathcal{I}}(z) = \sum_{\beta\in \mathcal{I}} c_\beta f_{\alpha,\beta(z)},

where :math:`\beta=[\beta_1,\ldots,\beta_D]` is a multi-index controlling the number of samples in each dimension of the tensor-product interpolants, and the index set :math:`\mathcal{I}` controls the approximation accuracy and data-efficiency of the sparse grid. If the set :math:`\mathcal{I}` is downward closed, that is

.. math:: \gamma \le \beta \text{ and } \beta \in \mathcal{I} \implies \gamma \in \mathcal{I},

where the :math:`\le` is applied per entry, then the coefficients of the sparse grid are given by

.. math:: \sum_{i\in [0,1]^D, \alpha+i\in \mathcal{I}} (-1)^{\lVert i \rVert_1}.


While any tensor-product approximation can be used with sparse grids, e.g. based on piecewise-polynomials or splines, in this tutorial we will build sparse grids with Lagrange polynomials (see :ref:`sphx_glr_auto_tutorials_surrogates_plot_tensor_product_interpolation.py`).

The following code compares tensor-product interpolants of varying resolution and shows which interpolants are included in a so called level-:math:`l` isotropic sparse grid which sets

.. math:: \mathcal{I}(l)=\{\beta \mid (\max(0,l−1)\le \lVert\beta\rVert_1\le l+D−2\}, \quad l\ge 0

which leads to a simpler expression for the coefficients

.. math:: c_\beta = (-1)^{l-\lvert\beta\rvert_1} {D-1\choose l-\lvert\beta\rvert_1}.



"""

#%%
#There is no exact formula for the number of points in an isotropic sparse grid. #The following code can be used to determine the number of points in a sparse grid of any dimension or level. The number of points is much smaller than the number of points in a tensor-product grid, for a given level :math:`l`.

#%%
#For a function with :math:`r` continous mixed-derivatives, the isotropic level-:math:`l` sparse grid, based on 1D Clenshaw Curtis abscissa, with :math:`M_{\mathcal{I}(l)}` points satisfies
#
#.. math:: \lVert f-f_{\mathcal{I}(l)}\rVert_{L^\infty}\le C_{D,r} M^{−r}_{\mathcal{I}(l)}(\log M_{\mathcal{I}(l)})^{(r+2)(D−1)+1}.
#
#In contrast the tensor-product interpolant with :math:`M_l` points satifies
#
#.. math:: \lVert f-f_{\mathcal{I}(l)}\rVert_{L^\infty}\le  K_{D,r} M_l^{−r/D}.
#
#The following code compares the convergence of sparse grids and tensor-product lagrange interpolants.

#%%
#Remarks
#-------
#The efficiency of sparse grids can be improved using methods [GG2003]_, [H2003]_ that construct the index set :math:`\mathcal{I}` adaptively. This is the default behavior when using Pyapprox.
#
#Note, in this tutorial we used sparse grids based on Clenshaw-Curtis 1D quadrature rules. However other types of rules can be used. PyApprox uses 1D Leja sequences  [NJ2014]_.

#%%
#References
#^^^^^^^^^^
#.. [BNR2000] `V. Barthelmann, E. Novak and K. Ritter. High dimensional polynomial interpolation on sparse grid. Advances in Computational Mathematics (2000). <https://doi.org/10.1023/A:1018977404843>`_
#.. [BG2004] `H. Bungartz and  M. Griebel. Sparse grids. Acta Numerica (2004). <https://doi.org/10.1017/S0962492904000182>`_
#.. [GG2003] `T. Gerstner and M. Griebel. Dimension-adaptive tensor-product quadrature. Computing (2003). <https://doi.org/10.1007/s00607-003-0015-5>`_
#.. [H2003] `M. Hegland. Adaptive sparse grids. Proc. of 10th Computational Techniques and Applications Conference (2003). <https://doi.org/10.21914/anziamj.v44i0.685>`_
#.. [NJ2014] `A. Narayan and J.D. Jakeman. Adaptive Leja Sparse Grid Constructions for Stochastic Collocation and High-Dimensional Approximation. SIAM Journal on Scientific Computing (2014). <http://dx.doi.org/10.1137/140966368>`_
