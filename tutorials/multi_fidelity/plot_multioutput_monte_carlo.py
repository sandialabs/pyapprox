r"""
Monte Carlo Quadrature: Beyond Mean Estimation
----------------------------------------------
:ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_monte_carlo.py` discussed how to use Monte Carlo quadrature to compute the mean of a model. In contrast, this tutorial shows how to use MC to compute alternative statistics of a model, for example variance, and how to compute the MSE of MC estimates of such statistics.

In this tutorial we will discuss using MC to estimate the statistics

.. math:: Q_\alpha^\mu=\int_\rvdom f_\alpha(\rv)\pdf(\rv)\dx{\rv} \qquad Q_\alpha^{\sigma^2}=\int_\rvdom f_\alpha^2(\rv)\pdf(\rv)\dx{\rv}-\left(Q_\alpha^\mu\right)^2,

which are the mean and variance of a model, respectively. Moreover, we will show how to estimate a vector-valued statistis :math:`\mat{Q}_\alpha=[Q_{\alpha,1},\ldots,Q_{\alpha,K}]^\top` that may be comprised of multiple statistics of a scalar function, a single statistics of a vector-valued function, or a combination of both. For example, in the following we will show how to use MC to compute the mean and variance of a vector-valued function :math:`f_\alpha:\reals^D\to\reals^K`, that is

.. math:: \mat{Q}_\alpha=[Q_{\alpha,1}^\mu, \ldots, Q_{\alpha,K}^\mu, Q_{\alpha,1}^{\sigma^2}, \ldots, Q_{\alpha,K}^{\sigma^2}]^\top.

Specifically we will consider

.. math::   f_0(\rv) = [\sqrt{11}\rv^5, \rv^4, \sin(2\pi \rv)]^\top

To compute a vector-valued statistic we follow the same procedure introduced for estimating the mean of a scalar function but now compute MC estimators for each statistic and function, that is

.. math:: Q^\mu_{\alpha, k}(\rvset_N)= N^{-1}\sum_{n=1}^N f_{\alpha, k}^{(n)}, \quad Q^{\sigma^2}_{\alpha, k}(\rvset_N)= (N-1)^{-1}\sum_{n=1}^N \left( f_{\alpha, k}^{(n)}-Q^\mu_{\alpha,k}(\rvset_N)\right)^2, \quad k=1,\ldots, K.

The following implements this procedure
"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.benchmarks import setup_benchmark
from pyapprox.multifidelity.multioutput_monte_carlo import get_estimator

np.random.seed(1)
shifts = [.1, .2]
benchmark = setup_benchmark("multioutput_model_ensemble")

costs = [1]
cov = benchmark.fun.get_covariance_matrix()
nsamples = 100
samples = benchmark.variable.rvs(nsamples)
values = benchmark.fun.models[0](samples)
est = get_estimator("mc", "mean_variance")
stats = est(values)


#%%
#.. math:: \covar{\mat{Q}_\alpha(\rvset_{N})}{\mat{Q}_\beta(\rvset_{M})} = \frac{P}{MN}\covar{f_\alpha}{f_\beta}
#
#.. math:: \covar{\mat{Q}_\alpha(\rvset_{N})}{\mat{Q}_\beta(\rvset_{M})} = \frac{P(P-1)}{M(M-1)N(N-1)}V_{\alpha,\beta}+\frac{P}{MN}W_{\alpha,\beta}
#
#.. math:: \covar{\mat{Q}_\alpha(\rvset_{N})}{\mat{Q}_\beta(\rvset_{M})} = \frac{P(P-1)}{M(M-1)N(N-1)}V_{\alpha,\beta}+\frac{P}{MN}W_{\alpha,\beta}

