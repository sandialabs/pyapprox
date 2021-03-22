r"""
Gaussian processes
==================
Gaussian processes (GPs) are an extremely popular tool for approximating multivariate functions from limited data. A GP is a distribution over a set of functions. Given a prior distribution on the class of admissible functions an approximation of a deterministic function is obtained by conditioning the GP on available observations of the function. 

Constructing a GP requires specifying a prior mean :math:`m(\rv)` and covariance kernel :math:`C(\rv, \rv^\star)`. The GP leverages the correlation between training samples to approximate the residuals between the training data and the mean function. In the following we set the mean to zero. The covariance kernel should be tailored to the smoothness of the class of functions under consideration.

.. math::
   C(\rv, \rv^\star; \ell)=\sigma^2 \frac{2^{1-\nu}}{\mathsf{\Gamma}(\nu)}\left(\frac{\sqrt{2\nu}d(\rv,\rv^\star; \ell)}{\ell}\right)^{\nu}K_{\nu}\left(\frac{\sqrt{2\nu}d(\rv,\rv^\star; \ell)}{\ell}\right).

Here :math:`d(\rv,\rv^\star; \ell)` is the weighted Euclidian distance between two points parameterized by the  vector hyper-parameters :math:`\ell=[\ell_1,\ldots,\ell_d]^\top` is. The variance of the kernel is determined by :mathL`\sigma^2` and we define :math:`K_{\nu}` as the modified Bessel function of the second 
kind of order :math:`\nu` and :math:`\mathsf{\Gamma}` as the gamma function.
Note that the parameter :math:`\nu` dictates for the smoothness of the 
kernel function. The analytic squared-exponential kernel can be obtained as
:math:`\nu\to\infty`.

Given a kernel and mean function, a Gaussian process approximation assumes that the joint prior distribution of :math:`f`, conditional on kernel hyper-parameters :math:`\theta=[\sigma^2,\ell^\top]^\top`,  is multivariate normal such that

.. math:: 

   f(\cdot) \mid \theta \sim \mathcal{N}\left(m(\cdot),C(\cdot,\cdot;\theta)+\epsilon^2I\right)

where :math:`\epsilon^2` is the variance of the mean zero white noise in the observations.
Given a set of training samples :math:`\mathcal{Z}=\{\rv^{(m)}\}_{m=1}^M` and associated values :math:`y=[y^{(1)}, \ldots, y^{(M)}]^\top` the posterior distibution of the GP is

.. math::  f(\cdot) \mid \theta,y \sim \mathcal{N}\left(m^\star(\cdot),C^\star(\cdot,\cdot;\theta)+\epsilon^2I\right)

where

.. math::  m^\star(\rv)=t(\rv)^TA^{-1}y \quad\quad C^\star(\rv,\rv^\prime)=C(\rv,\rv^\prime)-t(\rv)^TA^{-1}t(\rv^\prime)

with 

.. math::      t(\rv)=[C(\rv,\rv^{(1)}),\ldots,C(\rv,\rv^{(N)})]^T

and :math:`A` is a matrix with with elements :math:`A_{ij}=C(\rv^{(i)},\rv^{(j)})` for :math:`i,j=1,\ldots,M`.
"""

#%%
#References
#^^^^^^^^^^
#.. [RW2006] `C.E. Rasmussen and C. WIlliams. Gaussian Processes for Machine Learning. MIT Press (2006) <http://www.gaussianprocess.org/gpml/>`_
#
#.. [HJZ2021] `H. Harbrecht, J.D. Jakeman, P. Zaspel. Cholesky-based experimental design for Gaussian process and kernel-based emulation and calibration . Communications in Computational Physics (2021) In press <https://edoc.unibas.ch/79042/>`_
#
#.. [SGFSJ2020] `L.P. Swiler, M. Gulian, A. Frankel, C. Safta, J.D. Jakeman. A Survey of Constrained Gaussian Process Regression: Approaches and Implementation Challenges. Journal of Machine Learning for Modeling and Computing (2020) <http://www.dl.begellhouse.com/journals/558048804a15188a,2cbcbe11139f18e5,0776649265326db4.html>`_
