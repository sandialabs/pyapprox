r"""
Monte Carlo Quadrature: Beyond Mean Estimation
==============================================
:ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_monte_carlo.py` discussed how to use Monte Carlo quadrature to compute the mean of a model. In contrast, this tutorial shows how to use MC to compute alternative statistics of a model, for example variance, and how to compute the MSE of MC estimates of such statistics.

In this tutorial we will discuss using MC to estimate the statistics

.. math:: Q_\alpha^\mu=\int_\rvdom f_\alpha(\rv)\pdf(\rv)\dx{\rv} \qquad Q_\alpha^{\sigma^2}=\int_\rvdom f_\alpha^2(\rv)\pdf(\rv)\dx{\rv}-\left(Q_\alpha^\mu\right)^2,

which are the mean and variance of a model, respectively. Moreover, we will show how to estimate a vector-valued statistis :math:`\mat{Q}_\alpha=[Q_{\alpha,1},\ldots,Q_{\alpha,K}]^\top` that may be comprised of multiple statistics of a scalar function, a single statistics of a vector-valued function, or a combination of both. For example, in the following we will show how to use MC to compute the mean and variance of a vector-valued function :math:`f_\alpha:\reals^D\to\reals^K`, that is

.. math:: \mat{Q}_\alpha=[Q_{\alpha,1}^\mu, \ldots, Q_{\alpha,K}^\mu, Q_{\alpha,1}^{\sigma^2}, \ldots, Q_{\alpha,K}^{\sigma^2}]^\top.

Specifically we will consider

.. math::   f_0(\rv) = [\sqrt{11}\rv^5, \rv^4, \sin(2\pi \rv)]^\top

To compute a vector-valued statistic we follow the same procedure introduced for estimating the mean of a scalar function but now compute MC estimators for each statistic and function, that is

.. math:: Q^\mu_{\alpha, k}(\rvset_N)= N^{-1}\sum_{n=1}^N f_{\alpha, k}^{(n)}, \quad Q^{\sigma^2}_{\alpha, k}(\rvset_N)= (N-1)^{-1}\sum_{n=1}^N \left( f_{\alpha, k}^{(n)}-Q^\mu_{\alpha,k}(\rvset_N)\right)^2, \quad k=1,\ldots, K.

The following implements this procedure.

First load the benchmark
"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.benchmarks import setup_benchmark
from pyapprox.multifidelity.multioutput_monte_carlo import get_estimator

np.random.seed(1)
shifts = [.1, .2]
benchmark = setup_benchmark("multioutput_model_ensemble")

#%%
#Now construct the estimator. The following requires the costs of the model
#the covariance and some other quantities W and B. These four quantities are not
#needed to compute the value of the estimator from a sample set, however they are needed to compute the MSE of the estimator. We load these quantities but ignore there meaning for the moment.
costs = [1]
nqoi = 3
cov = benchmark.fun.get_covariance_matrix()[:3, :3]
W = benchmark.fun.covariance_of_centered_values_kronker_product()[:9, :9]
B = benchmark.fun.covariance_of_mean_and_variance_estimators()[:3, :9]

nsamples = 10
samples = benchmark.variable.rvs(nsamples)
values = benchmark.fun.models[0](samples)
est = get_estimator("mc", "mean_variance", nqoi, costs, cov, W, B)
est.set_nsamples_per_model([nsamples])
stats = est(values)

#%%
#The following compares the estimated value with the true values. We are only extracting certain components from the benchmark because the benchmark is designed for estimating vector-valued statistics with multiple models, but we are ignoring the other models for now.
print(benchmark.fun.get_means()[0])
print(stats[:3])
print(benchmark.fun.get_covariance_matrix()[:3, :3])
print(stats[3:].reshape(3, 3))

#%%
#Similarly to a MC estimator for a scalar, a vector-valued MC estimator is a random variable. Assuming that we are not using an approximation of the model we care about, the error in the MC estimator is characterized completely by the covariance of the estimator statistics. The following shows how to compute the estimator covariance. We present general formula for the covariance between estimators of using different vector-valued models :math:`f_\alpha\text{ and }f_\beta` and using sample sets :math:`\rvset_N, \rvset_M` of (potentially) different sizes.
#
#Mean Estimation
#---------------
#First consider the estimation of the mean. The covariance between two estimates of the mean that share P samples, i.e. :math:`P=|\rvset_\alpha\cup \rvset_\beta|`, is
#
#.. math:: \covar{\mat{Q}^\mu_\alpha(\rvset_{N})}{\mat{Q}^\mu_\beta(\rvset_{M})} = \frac{P}{MN}\covar{f_\alpha}{f_\beta}
#
#Variance Estimation
#-------------------
#The covariance between two estimators of variance is
#
#.. math:: \covar{\mat{Q}_\alpha^{\sigma^2}(\rvset_{N})}{\mat{Q}_\beta^{\sigma^2}(\rvset_{M})} = \frac{P(P-1)}{M(M-1)N(N-1)}V_{\alpha,\beta}+\frac{P}{MN}W_{\alpha,\beta}
#
#where
#
#.. math:: V_{\alpha,\beta} = \covar{f_\alpha}{f_\beta}^{\otimes2}+(\mat{1}_K^\top\otimes\covar{f_\alpha}{f_\beta}\otimes\mat{1}_K)\circ(\mat{1}_K\otimes\covar{f_\alpha}{f_\beta}\otimes\mat{1}_K^\top),\quad \in \reals^{K^2\times K^2}
#
#.. math:: W_{\alpha,\beta} = \covar{(f_\alpha-\mean{f_\alpha})^{\otimes 2}}{(f_\beta-\mean{f_\beta})^{\otimes 2}}, \quad \in \reals^{K^2\times K^2}
#
#Above we used the notation, for general matrices :math:`\mat{X},\mat{Y}`,
#
#.. math:: X\otimes Y=\text{flatten}(\mat{X}\mat{Y}^\top)
#
#for a flattened outer product,
#
#.. math:: X\circ Y
#
#for an element wise product, and
#
#.. math:: X^{\otimes2}=X\otimes X
#
#Simultaneous Mean and Variance Estimation
#-----------------------------------------
#The covariance between two estimators of mean and variance of the form
#
#.. math:: \mat{Q}_\alpha^{\mu,\sigma^2}(\rvset_{N})=[\mat{Q}_\alpha^{\mu}(\rvset_{N}), \mat{Q}_\alpha^{\sigma^2}(\rvset_{N})]^\top
#
#is
#
#.. math:: \covar{\mat{Q}_\alpha^{\mu,\sigma^2}(\rvset_{N})}{\mat{Q}_\beta^{\mu,\sigma^2}(\rvset_{M})} = \begin{bmatrix}\covar{\mat{Q}_\alpha^\mu(\rvset_{N})}{\mat{Q}_\beta^\mu(\rvset_{N})} & \covar{\mat{Q}_\alpha^\mu(\rvset_{N})}{\mat{Q}_\beta^{\sigma^2}(\rvset_{M})}\\ \covar{\mat{Q}_\alpha^{\sigma^2}(\rvset_{M})}{\mat{Q}_\beta^\mu(\rvset_{N})}& \covar{\mat{Q}_\alpha^{\sigma^2}(\rvset_{M})}{\mat{Q}_\beta^{\sigma^2}(\rvset_{M})}\end{bmatrix}
#
#We have already shown the form of the upper and lower diagonal blocks. The off diagonal blocks are
#
#.. math:: \covar{\mat{Q}_\alpha^\mu(\rvset_{N})}{\mat{Q}_\beta^{\sigma^2}(\rvset_{M})}=\frac{P}{MN}B_{\alpha,\beta}
#
#where
#
#.. math:: B_{\alpha,\beta}=\covar{f_\alpha}{(f_\beta-\mean{f_\beta})^{\otimes 2}},\quad\in\reals^{K\times K^2}
#
#MSE
#---
#For vector valued statistics, the MSE formular presented in the previous tutorial no longer makes sense because the estimator is a multivariate random variable with a mean and covariance (not a mean and variance like for the scalar formulation). Consequently, for an unbiased estimator, the error in a vector-valued statistic is usually a scalar function of the estimator covariance. Two common choices are the determinant and trace of the estimator covariance.
#
#The estimator covariance can be obtained using
#
from pyapprox.multifidelity.visualize import plot_correlation_matrix
# plot correlation matrix can also be used for covariance matrics
est_cov = est._covariance_from_npartition_samples([nsamples])
ax = plt.subplots(1, 1, figsize=(2*8, 2*6))[1]
_ = plot_correlation_matrix(est_cov, ax=ax)

#%%
#We can print the diagonal, which contains the estimator variances that would be obtained if each statistic was treated individually.
print(np.diag(est_cov.numpy()))
