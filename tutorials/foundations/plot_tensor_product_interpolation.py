r"""
Surrogate Modeling
==================
Many simulation models are extremely computationally expensive such that adequately understanding their behaviour and quantifying uncertainty can be computationally intractable for any of the aforementioned techniques. Various methods have been developed to produce surrogates of the model response to uncertain parameters, the most efficient are goal-oriented in nature and target very specific uncertainty measures. 

Generally speaking surrogates are built using a ``small'' number of model simulations and are then substituted in place of the expensive simulation models in future analysis. Some of the most popular surrogate types include polynomial chaos expansions (PCE) [XKSISC2002]_, Gaussian processes (GP) [RWMIT2006]_, and sparse grids (SG) [BGAN2004]_. 

Reduced order models (e.g. [SFIJNME2017]_) can also be used to construct surrogates and have been applied successfully for UQ on many applications. These methods do not construct response surface approximations, but rather solve the governing equations on a reduced basis. PyApprox does not currently implement reduced order modeling, however the modeling analyis tools found in PyApprox can easily be applied to assess or design systems based on reduced order models.


Example: Tensor-product Lagrange interpolation
----------------------------------------------
This tutorial demonstrates how to build an approximation of an expensive model
using tensor-product Lagrange interpolation.

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
Here we will use the nested Clenshaw-Curtis points

.. math::
      
  z_{i}^{(j)}=\cos\left(\frac{(j-1)\pi}{m_{\beta_i}}\right),\qquad j=1,\ldots,m_{\beta_i}

to define the univariate Lagrange polynomials. The number of points :math:`m(l)` of this rule grows exponentially with the level :math:`l`, specifically
:math:`m(0)=1` and :math:`m(l)=2^{l}+1` for :math:`l\geq1`. The univariate Clenshaw-Curtis points, the tensor-product grid :math:`\mathcal{Z}_{\boldsymbol{\beta}}`, and two multivariate Lagrange polynomials with their corresponding univariate Lagrange polynomials are shown below for :math:`\boldsymbol{\beta}=(2,2)`.
"""

from pyapprox.examples.tensor_product_lagrange_interpolation import *
fig = plt.figure(figsize=(2*8,6))
ax=fig.add_subplot(1,2,1,projection='3d')
level = 2; ii=1; jj=1
plot_tensor_product_lagrange_basis_2d(level,ii,jj,ax)

ax=fig.add_subplot(1,2,2,projection='3d')
level = 2; ii=1; jj=3
plot_tensor_product_lagrange_basis_2d(level,ii,jj,ax)

#%%
#To construct a surrogate using tensor product interpolation we simply multiply all such basis functions by the value of the function :math:`f_\ai` evaluated at the corresponding interpolation point. The following uses tensor product interpolation to approximate the simple function
#
#.. math:: f_\ai(\rv) = \cos(2\pi\rv_1)\cos(2\pi\rv_2), \qquad \rv\in\rvdom=[-1,1]^2

f = lambda z : (np.cos(2*np.pi*z[0,:])*np.cos(2*np.pi*z[1,:]))[:,np.newaxis]
def get_interpolant(function,level):
    level = np.asarray(level)
    univariate_samples_func = lambda l: pya.clenshaw_curtis_pts_wts_1D(l)[0]
    abscissa_1d = [univariate_samples_func(level[0]),
                   univariate_samples_func(level[1])]

    
    samples_1d = pya.get_1d_samples_weights(
        [pya.clenshaw_curtis_in_polynomial_order]*2,
        [pya.clenshaw_curtis_rule_growth]*2,level)[0]

    poly_indices = pya.get_subspace_polynomial_indices(
        level,[pya.clenshaw_curtis_rule_growth]*2,config_variables_idx=None)
    samples = pya.get_subspace_samples(level,poly_indices,samples_1d)
    fn_vals = function(samples)

    interp = lambda samples: pya.evaluate_sparse_grid_subspace(
        samples,level,fn_vals,samples_1d,None,False)
    hier_indices = pya.get_hierarchical_sample_indices(
        level,poly_indices,samples_1d,config_variables_idx=None)
    
    return interp, samples, hier_indices, abscissa_1d[0].shape[0], \
      abscissa_1d[1].shape[0]

import pyapprox as pya
level = [2,3]
interp,samples,_ = get_interpolant(f,level)[:3]

marker_color='k'
alpha=1.0
fig,axs = plt.subplots(1,1,figsize=(8,6))
axs.plot(samples[0,:],samples[1,:],'o',color=marker_color,ms=10,alpha=alpha)

plot_limits = [-1,1,-1,1]
num_pts_1d = 101
X,Y,Z = get_meshgrid_function_data(
    interp, plot_limits, num_pts_1d)

num_contour_levels=10
import matplotlib as mpl
cmap = mpl.cm.coolwarm
levels = np.linspace(Z.min(),Z.max(),num_contour_levels)
cset = axs.contourf(
    X, Y, Z, levels=levels,cmap=cmap,alpha=alpha)
plt.show()

#%%
# The error in the tensor product interpolant is given by
#.. math:: \lVert f_\ai-f_{\ai,\bi}\rVert_{L^\infty(\rvdom)} \le C_{d,r} N_{\bi}^{-s/d}


#%%
#References
#^^^^^^^^^^
#
#.. [XKSISC2002] `D. Xiu and G.E. Karniadakis. The Wiener-Askey Polynomial Chaos for stochastic differential equations. SIAM J. Sci. Comput., 24(2), 619-644, 2002. <http://dx.doi.org/10.1137/S1064827501387826>`_
#
#.. [RWMIT2006] `C.E Rasmussen and C. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006. <http://www.gaussianprocess.org/gpml/chapters/>`_
#
#.. [BGAN2004] `H. Bungartz and M. Griebel. Sparse Grids. Acta Numerica, 13, 147-269, 2004. <http://dx.doi.org/10.1017/S0962492904000182>`_
#
#.. [SFIJNME2017] `C Soize and C. Farhat. A nonparametric probabilistic approach for quantifying uncertainties in low-dimensional and high-dimensional nonlinear models. International Journal for Numerical Methods in Engineering, 109(6), 837-888, 2017. <https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.5312>`_
