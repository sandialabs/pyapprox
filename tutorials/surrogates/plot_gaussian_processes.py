r"""
Gaussian processes
==================
Gaussian processes (GPs) are an extremely popular tool for approximating multivariate functions from limited data. A GP is a distribution over a set of functions. Given a prior distribution on the class of admissible functions an approximation of a deterministic function is obtained by conditioning the GP on available observations of the function.

Constructing a GP requires specifying a prior mean :math:`m(\rv)` and covariance kernel :math:`C(\rv, \rv^\star)`. The GP leverages the correlation between training samples to approximate the residuals between the training data and the mean function. In the following we set the mean to zero. The covariance kernel should be tailored to the smoothness of the class of functions under consideration.

.. math::
   C(\rv, \rv^\star; \ell)=\sigma^2 \frac{2^{1-\nu}}{\mathsf{\Gamma}(\nu)}\left(\frac{\sqrt{2\nu}d(\rv,\rv^\star; \ell)}{\ell}\right)^{\nu}K_{\nu}\left(\frac{\sqrt{2\nu}d(\rv,\rv^\star; \ell)}{\ell}\right).

Here :math:`d(\rv,\rv^\star; \ell)` is the weighted Euclidean distance between two points parameterized by the  vector hyper-parameters :math:`\ell=[\ell_1,\ldots,\ell_d]^\top`. The variance of the kernel is determined by :math:`\sigma^2` and :math:`K_{\nu}` is the modified Bessel function of the second
kind of order :math:`\nu` and :math:`\mathsf{\Gamma}` is the gamma function.
Note that the parameter :math:`\nu` dictates for the smoothness of the
kernel function. The analytic squared-exponential kernel can be obtained as
:math:`\nu\to\infty`.

Given a kernel and mean function, a Gaussian process approximation assumes that the joint prior distribution of :math:`f`, conditional on kernel hyper-parameters :math:`\theta=[\sigma^2,\ell^\top]^\top`,  is multivariate normal such that

.. math::

   f(\cdot) \mid \theta \sim \mathcal{N}\left(m(\cdot),C(\cdot,\cdot;\theta)+\epsilon^2I\right)

where :math:`\epsilon^2` is the variance of the mean zero white noise in the observations.
Given a set of training samples :math:`\mathcal{Z}=\{\rv^{(m)}\}_{m=1}^M` and associated values :math:`y=[y^{(1)}, \ldots, y^{(M)}]^\top` the posterior distribution of the GP is

.. math::  f(\cdot) \mid \theta,y \sim \mathcal{N}\left(m^\star(\cdot),C^\star(\cdot,\cdot;\theta)+\epsilon^2I\right)

where

.. math::  m^\star(\rv)=t(\rv)^\top A^{-1}y \quad\quad C^\star(\rv,\rv^\prime)=C(\rv,\rv^\prime)-t(\rv)^\top A^{-1}t(\rv^\prime)

with

.. math::      t(\rv)=[C(\rv,\rv^{(1)}),\ldots,C(\rv,\rv^{(N)})]^\top

and :math:`A` is a matrix with with elements :math:`A_{ij}=C(\rv^{(i)},\rv^{(j)})` for :math:`i,j=1,\ldots,M`. Here we dropped the dependence on the hyper-parameters :math:`\theta` for convenience.

Consider the univariate Runge function

.. math:: f(\rv) = \frac{1}{1+25\rv^2}, \quad \rv\in[-1,1]

Lets construct a GP with a fixed set of training samples and associated values we can train the Gaussian process. But first lets plot the true function and prior GP mean and plus/minus 2 standard deviations using the prior covariance
"""
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.surrogates import gaussianprocess as gps
np.random.seed(1)

lb, ub = -1, 1

def func(x):
    return 1/(1+25*x[0, :]**2)[:, np.newaxis]

kernel = gps.Matern(0.5, length_scale_bounds=(1e-1, 1e1), nu=np.inf)
gp = gps.GaussianProcess(kernel)

validation_samples = np.linspace(lb, ub, 101)[None, :]
validation_values = func(validation_samples)
plt.plot(validation_samples[0, :], validation_values[:, 0], 'r-', label='Exact')
gp_vals, gp_std = gp(validation_samples, return_std=True)
plt.plot(validation_samples[0, :], gp_vals[:, 0], 'b-', label='GP prior mean')
_ = plt.fill_between(validation_samples[0, :], gp_vals[:, 0]-2*gp_std,
                 gp_vals[:, 0]+2*gp_std,
                 alpha=0.2, color='blue', label='GP prior uncertainty')

#%%
#Now lets train the GP using a small number of evaluations and plot
#the posterior mean and variance.
ntrain_samples = 5
train_samples = np.linspace(lb, ub, ntrain_samples)[None, :]
train_values = func(train_samples)
gp.fit(train_samples, train_values)
print(gp.kernel_)
gp_vals, gp_std = gp(validation_samples, return_std=True)
plt.plot(validation_samples[0, :], validation_values[:, 0], 'r-', label='Exact')
plt.plot(train_samples[0, :], train_values[:, 0], 'or')
plt.plot(validation_samples[0, :], gp_vals[:, 0], '-k',
         label='GP posterior mean')
plt.fill_between(validation_samples[0, :], gp_vals[:, 0]-2*gp_std,
                 gp_vals[:, 0]+2*gp_std,
                 alpha=0.5, color='gray', label='GP posterior uncertainty')

_ = plt.legend()

#%% As we add more training data the posterior uncertainty will decrease and the mean will become a more accurate estimate of the true function.

#%%
#Experimental design
#-------------------
#The nature of the training samples significantly impacts the accuracy of a Gaussian process. Noting that the variance of a GP reflects the accuracy of a Gaussian process, [SWMW1989]_ developed an experimental design procedure which minimizes the average variance with respect to a specified measure. This measure is typically the probability measure :math:`\pdf(\rv)` of the random variables :math:`\rv`. Integrated variance designs, as they are often called, find a set of samples :math:`\mathcal{Z}\subset\Omega\subset\rvdom` from a set of candidate samples :math:`\Omega` by solving the minimization problem
#
#.. math:: \mathcal{Z}^\dagger=\argmin_{\mathcal{Z}\subset\Omega\subset\rvdom, \lvert\mathcal{Z}\rvert=M} \int_{\rvdom} C^\star(\rv, \rv\mid \mathcal{Z})\pdf(\rv)d\rv
#
#where we have made explicit the posterior variance dependence on :math:`\mathcal{Z}`.
#
#The variance of a GP is not dependent on the values of the training data, only the sample locations, and thus the procedure can be used to generate batches of samples. The IVAR criterion - also called active learning Cohn (ALC) - can be minimized over discrete [HJZ2021]_ or continuous [GM2016]_ design spaces :math:`\Omega`. When employing a discrete design space, greedy methods [C2006]_ are used to sample one at a time from a finite set of candidate samples to minimize the learning objective.  This approach requires a representative candidate set which, we have found, can be generated with low-discrepancy sequences, e.g. Sobol sequences. The continuous optimization optimization is non-convex and thus requires a good initial guess to start the gradient based optimization. Greedy methods can be used to produce the initial guess, however in certain situation optimizing from the greedy design resulted in minimal improvement.
#
#The following code plots the samples chosen by greedily minimizing IVAR
from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    IVARSampler, GreedyIntegratedVarianceSampler, CholeskySampler)
from pyapprox.variables.joint import IndependentMarginalsVariable, stats
variable = IndependentMarginalsVariable([stats.uniform(-1, 2)])
ncandidate_samples = 101
sampler = GreedyIntegratedVarianceSampler(
    1, 100, ncandidate_samples, variable.rvs, variable,
    use_gauss_quadrature=True, econ=False,
    candidate_samples=np.linspace(
       *variable.get_statistics("interval", 1)[0, :], 101)[None, :])

kernel = gps.Matern(0.5, length_scale_bounds="fixed", nu=np.inf)
sampler.set_kernel(kernel)


def plot_gp_samples(ntrain_samples, kernel, variable):
    axs = plt.subplots(1, ntrain_samples, figsize=(ntrain_samples*8, 6))[1]
    gp = gps.GaussianProcess(kernel)
    for ii in range(1, ntrain_samples+1):
        gp.plot_1d(101, variable.get_statistics("interval", 1)[0, :], ax=axs[ii-1])

    train_samples = sampler(ntrain_samples)[0]
    train_values = func(train_samples)*0
    for ii in range(1, ntrain_samples+1):
        gp.fit(train_samples[:, :ii], train_values[:ii])
        gp.plot_1d(101, variable.get_statistics("interval", 1)[0, :], ax=axs[ii-1])
        axs[ii-1].plot(train_samples[0, :ii], train_values[:ii, 0], 'ko', ms=15)


ntrain_samples = 5
plot_gp_samples(ntrain_samples, kernel, variable)

#%%
#The following plots the variance obtained by a global optimiaztion of IVAR,
#starting from the greedy IVAR sampls as the intial guess. The samples are plotted sequentially, however this is just for visualization as the global optimization does not produce a nested sequence of samples.
sampler = IVARSampler(
    1, 100, ncandidate_samples, variable.rvs, variable,
    'ivar', use_gauss_quadrature=True, nugget=1e-14)
sampler.set_kernel(kernel)
ntrain_samples = 5
plot_gp_samples(ntrain_samples, kernel, variable)

#%%
#Computing IVAR designs can be computationally expensive. An alternative cheaper algorithm called active learning Mckay (ALM) greedily chooses samples that minimizes the maximum variance of the Gaussian process. That is, given M training samples the next sample is chosen via
#
#.. math:: \rv^{(n+1)}=\argmax_{\mathcal{Z}\subset\Omega\subset\rvdom} C^\star(\rv, \rv\mid \mathcal{Z}_M)
#
#Although more computationally efficient than ALC, empirical studies suggest that ALM tends to produce GPs with worse predictive performance [GL2009]_.
#
#Accurately evaluating the ALC and ALM criterion is often challenging because inverting the covariance matrix :math:`C(\mathcal{Z}_M\cup \rv)` is poorly conditioned when :math:`\rv` is 'close' to a point in :math:`\mathcal{Z}_M`. Consequently a small constant (nugget) is often added to the diagonal of :math:`C(\mathcal{Z}_M\cup \rv)` to improve numerical stability [PW2014]_.
#
#Experimental design strategies similar to ALM and ALC have been developed for radial basis functions (RBFs). The strong connections between radial basis function and Gaussian process approximation mean that the RBF algorithms can often be used for constructing GPs. A popular RBF design strategy minimizes the worst case error function (power function) of kernel based approximations [SW2006]_. The minimization of the power function is equivalent to minimizing the ALM criteria [HJZ2021]_. As with ALM and ALC, evaluation of the power function is unstable [SW2006]_. However the authors of [PS2011]_ established that stability can be improved by greedily minimizing the power function using pivoted Cholesky factorization [PS2011]_. Specifically, the first :math:`M` pivots of the pivoted Cholesky factorization of a kernel (covariance matrix), evaluated a large set of candidate sample, define the :math:`M` samples which greedily minimize the power function (ALM criteria). Minimizing the power function does not take into account any available distribution information about the inputs :math:`\rv`. In [HJZ2021]_ this information was incorporated by weighting the power function by the density :math:`\pdf(\rv)` of the input variables. This procedure attempts to greedily minimizes the :math:`\pdf`-weighted :math:`L^2` error and produces GPs with predictive performance comparable to those based upon ALC designs while being much more computationally efficient because of its use of pivoted Cholesky factorization.
#
#Finally we remark that while ALM and ALC are the most popular experimental design strategies for GPs, alternative methods have been proposed. Of note are those methods which approximately minimize the mutual information between the Gaussian process evaluated at the training data and the Gaussian process evaluated at the remaining candidate samples [KSG2008]_, [BG2016]_. We do not consider these methods in our numerical comparisons.
#
#The following code shows how to use pivoted Cholesky factorization to greedily choose trainig samples for a GP.
sampler = CholeskySampler(1, 100, variable)
sampler.set_kernel(kernel)
ntrain_samples = 5
plot_gp_samples(ntrain_samples, kernel, variable)
plt.show()

#%%
#Active Learning
#---------------
# The samples selected by the aforementioned methods depends on the kernel specified. Change the length_scale of the kernel above to see how the selected samples changes. Active learning chooses a small initial sample set then trains the GP to learn the best kernel hyper-parameters. These parameters are then used to increment the training set and then used to train the GP hyper-parameters again and so until a sufficient accuracy or computational budget is reached. PyApprox's AdaptiveGaussianProcess implements this procedure [HJZ2021]_.

#%%
#References
#^^^^^^^^^^
#.. [RW2006] `C.E. Rasmussen and C. WIlliams. Gaussian Processes for Machine Learning. MIT Press (2006) <http://www.gaussianprocess.org/gpml/>`_
#
#.. [SWMW1989] `J. Sacks, W.J. Welch, T.J.Mitchell, H.P. Wynn Designs and analysis of computer experiments (with discussion). Statistical Science, 4:409-435 (1989) <http://www.jstor.org/stable/2245858>`_
#
#.. [HJZ2021] `H. Harbrecht, J.D. Jakeman, P. Zaspel. Cholesky-based experimental design for Gaussian process and kernel-based emulation and calibration . Communications in Computational Physics (2021) In press <https://edoc.unibas.ch/79042/>`_
#
#.. [SGFSJ2020] `L.P. Swiler, M. Gulian, A. Frankel, C. Safta, J.D. Jakeman. A Survey of Constrained Gaussian Process Regression: Approaches and Implementation Challenges. Journal of Machine Learning for Modeling and Computing (2020) <http://www.dl.begellhouse.com/journals/558048804a15188a,2cbcbe11139f18e5,0776649265326db4.html>`_
#
#.. [GM2016] `A. Gorodetsky, Y. Marzouk. Mercer kernels and integrated variance experimental design. Connec- tions between Gaussian process regression and polynomial approximation. SIAM/ASA J. Uncertain. Quantif., 4(1):796–828 (2016) <https://doi.org/10.1137/15M1017119>`_
#
#.. [C2006] `D. Cohn Neural network exploration using optimal experiment design, Neural Netw., 9 (1996), pp. 1071–1083. <https://proceedings.neurips.cc/paper/1993/file/d840cc5d906c3e9c84374c8919d2074e-Paper.pdf>`_
#
#.. [KSG2008] `A. Krause, A. Singh, C. Guestrin, Near-optimal sensor placements in Gaussian processes: Theory, efficient algorithms and empirical studies, J. Mach. Learn. Res., 9 (2008), pp. 235–284. <https://doi.org/10.1002/env.769>`_
#
#.. [PW2014] `C.Y. Peng, J. Wu, On the choice of nugget in kriging modeling for deterministic computer experiments, J. Comput. Graph. Statist., 23 (2014), pp. 151–168. <https://doi.org/10.1080/10618600.2012.738961>`_
#
#.. [BG2016] `J. Beck, S. Guillas, Sequential Design with Mutual Information for Computer Experiments (MICE): Emulation of a Tsunami Model, SIAM/ASA J. UNCERTAINTY QUANTIFICATION Vol. 4, pp. 739–766 (2016) <https://doi.org/10.1137/140989613>`_
#
#.. [GL2009] `R.B. Gramacy, H.K.H. Lee, Adaptive design and analysis of supercomputer experiments, Technometrics, 51 (2009), pp. 130–145. <https://doi.org/10.1198/TECH.2009.0015>`_
#
#.. [SW2006] `R. Schaback and H. Wendland. Kernel techniques: From machine learning to meshless methods. Acta Numer., 15:543–639 (2006). <https://doi.org/10.1017/S0962492906270016>`_
#
#.. [PS2011] `M. Pazouki and R. Schaback. Bases for kernel-based spaces. J. Comput. Appl. Math., 236:575–588 (2011). <https://doi.org/10.1016/j.cam.2011.05.021>`_
