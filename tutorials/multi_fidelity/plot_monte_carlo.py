r"""
Monte Carlo Quadrature
======================
This tutorial describes how to use Monte Carlo sampling to compute the expectations of the output of a model. Specifically, given a function :math:`f_\alpha(\rv):\reals^{d}\to\reals` parameterized by a set of variables :math:`\rv=[\rv_1,\ldots,\rv_d]^T` with joint density given by :math:`\rho(\rv):\reals^{d}\to\reals`, our goal is to approximate the integral

.. math:: Q_\alpha=\int_\rvdom f_\alpha(\rv)\pdf(\rv)\dx{\rv}

We can approximate the integral :math:`Q_\alpha` using Monte Carlo quadrature by drawing :math:`N` random samples of :math:`\rv` from :math:`\pdf` and evaluating the function at each of these samples to obtain the data pairs :math:`\{(\rv^{(n)},f^{(n)}_\alpha)\}_{n=1}^N`, where :math:`f^{(n)}_\alpha=f_\alpha(\rv^{(n)})` and computing

.. math::

   Q_{\alpha,N}=N^{-1}\sum_{n=1}^N f^{(n)}_\alpha

The mean squared error (MSE) of this estimator can be expressed as

.. math::

   \mean{\left(Q_{\alpha,N}-\mean{Q}\right)^2}&=\mean{\left(Q_{\alpha,N}-\mean{Q_{\alpha,N}}+\mean{Q_{\alpha,N}}-\mean{Q}\right)^2}\\
   &=\mean{\left(Q_{\alpha,N}-\mean{Q_{\alpha,N}}\right)^2}+\mean{\left(\mean{Q_{\alpha,N}}-\mean{Q}\right)^2}\\
   &\qquad\qquad+\mean{2\left(Q_{\alpha,N}-\mean{Q_{\alpha,N}}\right)\left(\mean{Q_{\alpha,N}}-\mean{Q}\right)}\\
   &=\var{Q_{\alpha,N}}+\left(\mean{Q_{\alpha,N}}-\mean{Q}\right)^2\\
   &=\var{Q_{\alpha,N}}+\left(\mean{Q_{\alpha,N}}-Q\right)^2

Here we used that :math:`\mean{\left(Q_{\alpha,N}-\mean{Q_{\alpha,N}}\right)}=0` so the third term on the second line is zero and :math:`\mean{Q}=Q` since the exact value of Q is deterministic. Now using the well known result that for random variable :math:`X_n`

.. math:: \var{\sum_{n=1}^N X_n} = \sum_{n=1}^N \var{X_n} + \sum_{n\neq p}\covar{X_n}{X_p}

and the result for a scalar :math:`a`

.. math:: \var{aX_n} =a^2\var{X_n}

yields

.. math::

   \var{Q_{\alpha,N}}=\var{N^{-1}\sum_{n=1}^N f^{(n)}_\alpha}=N^{-2}\sum_{n=1}^N \var{f^{(n)}_\alpha}=N^{-1}\var{Q_\alpha}

where :math:`\covar{f^{(n)}}{f^{(p)}}=0, n\neq p` because the samples are drawn independently.

Finally, substituting :math:`\var{Q_{\alpha,N}}` into the expression for MSE yields

.. math::

   \mean{\left(Q_{\alpha, N}-\mean{Q}\right)^2}=\underbrace{N^{-1}\var{Q_\alpha}}_{I}+\underbrace{\left(\mean{Q_{\alpha}}-Q\right)^2}_{II}

From this expression we can see that the MSE can be decomposed into two terms;
a so called stochastic error (I) and a deterministic bias (II). The first term is the variance of the Monte Carlo estimator which comes from using a finite number of samples. The second term is due to using an approximation of :math:`f`. These two errors should be balanced, however in the vast majority of all MC analyses a single model :math:`f_\alpha` is used and the choice of :math:`\alpha`, e.g. mesh resolution, is made a priori without much concern for the balancing bias and variance.

Given a fixed :math:`\alpha` the modelers only recourse to reducing the MSE is to reduce the variance of the estimator. In the following we plot the variance of the MC estimate of a simple algebraic function :math:`f_1` which belongs to an ensemble of models

.. math::

   f_0(\rv) &= A_0 \left(\rv_1^5\cos\theta_0 +\rv_2^5\sin\theta_0\right), \\
   f_1(\rv) &= A_1 \left(\rv_1^3\cos\theta_1 +\rv_2^3\sin\theta_1\right)+s_1,\\
   f_2(\rv) &= A_2 \left(\rv_1  \cos\theta_2 +\rv_2  \sin\theta_2\right)+s_2


where :math:`\rv_1,\rv_2\sim\mathcal{U}(-1,1)` and all :math:`A` and :math:`\theta` coefficients are real. We choose to set :math:`A=\sqrt{11}`, :math:`A_1=\sqrt{7}` and :math:`A_2=\sqrt{3}` to obtain unitary variance for each model. The parameters :math:`s_1,s_2` control the bias between the models. Here we set :math:`s_1=1/10,s_2=1/5`. Similarly we can change the correlation between the models in a systematic way (by varying :math:`\theta_1`. We will levarage this later in the tutorial.
"""

#%%
#Lets setup the problem
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pyapprox import variables
from pyapprox.analysis import visualize
from pyapprox.benchmarks import setup_benchmark

np.random.seed(1)
shifts = [.1, .2]
benchmark = setup_benchmark(
    "tunable_model_ensemble", theta1=np.pi/2*.95, shifts=shifts)
print(benchmark.variable)

#%%
#Now let us compute the mean of :math:`f_1` using Monte Carlo
nsamples = int(1e3)
samples = benchmark.variable.rvs(nsamples)

model = benchmark.fun
values = model.m1(samples)
variables.print_statistics(samples, values)

#%%
#We can compute the exact mean using sympy and compute the MC MSE
z1, z2 = sp.Symbol('z1'), sp.Symbol('z2')
ranges = [-1, 1, -1, 1]
exact_integral_f1 = benchmark.means[1]
print('MC difference squared =', (values.mean()-exact_integral_f1)**2)

#%%
#.. _estimator-histogram:
#
#Now let us compute the MSE for different sample sets of the same size for :math:`N=100,1000` and plot the distribution of the MC estimator :math:`Q_{\alpha,N}`
#

ntrials = 1000
means = np.empty((ntrials, 2))
for ii in range(ntrials):
    samples = benchmark.variable.rvs(nsamples)
    values = model.m1(samples)
    means[ii] = values[:100].mean(), values.mean()
fig, ax = plt.subplots()
textstr = '\n'.join(
    [r'$\mathbb{E}[Q_{1,100}]=\mathrm{%.2e}$' % means[:, 0].mean(),
     r'$\mathbb{V}[Q_{1,100}]=\mathrm{%.2e}$' % means[:, 0].var(),
     r'$\mathbb{E}[Q_{1,1000}]=\mathrm{%.2e}$' % means[:, 1].mean(),
     r'$\mathbb{V}[Q_{1,1000}]=\mathrm{%.2e}$' % means[:, 1].var()])
ax.hist(means[:, 0], bins=ntrials//100, density=True)
ax.hist(means[:, 1], bins=ntrials//100, density=True, alpha=0.5)
ax.axvline(x=shifts[0], c='r', label=r'$\mathbb{E}[Q_1]$')
ax.axvline(x=0, c='k', label=r'$\mathbb{E}[Q_0]$')
props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 1}
ax.text(0.65, 0.8, textstr, transform=ax.transAxes, bbox=props)
ax.set_xlabel(r'$\mathbb{E}[Q_N]$')
ax.set_ylabel(r'$\mathbb{P}(\mathbb{E}[Q_N])$')
_ = ax.legend(loc='upper left')

#%%
#The numerical results match our theory. Specifically the estimator is unbiased( i.e. mean zero, and the variance of the estimator is :math:`\var{Q_{0,N}}=\var{Q_{0}}/N=1/N`.
#
#The variance of the estimator can be driven to zero by increasing the number of samples :math:`N`. However when the variance becomes less than the bias, i.e. :math:`\left(\mean{Q_{\alpha}-Q}\right)^2>\var{Q_{\alpha}}/N`, then the MSE will not decrease and any further samples used to reduce the variance are wasted.
#
#Let our true model be :math:`f_0` above. The following code compues the bias induced by using :math:`f_\alpha=f_1` and also plots the contours of :math:`f_0(\rv)-f_1(\rv)`.

integrand_f0 = model.A0*(sp.cos(model.theta0)*z1**5 +
                         sp.sin(model.theta0)*z2**5)*0.25
exact_integral_f0 = float(
    sp.integrate(integrand_f0, (z1, ranges[0], ranges[1]), (z2, ranges[2], ranges[3])))
bias = (exact_integral_f0-exact_integral_f1)**2
print('MC f1 bias =', bias)
print('MC f1 variance =', means.var())
print('MC f1 MSE =', bias+means.var())

fig, ax = plt.subplots()
X, Y, Z = visualize.get_meshgrid_function_data_from_variable(
    lambda z: model.m0(z)-model.m1(z), benchmark.variable, 50)
cset = ax.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 20))
_ = plt.colorbar(cset, ax=ax)
# plt.show()

#%%
#As :math:`N\to\infty` the MSE will only converge to the bias (:math:`s_1`). Try this by increasing :math:`\texttt{nsamples}`.

#%%
#We can produced unbiased estimators using the high fidelity model. However if this high-fidelity model is more expensive then this comes at the cost of the estimator having larger variance. To see this the following plots the distribution of the MC estimators using 100 samples of the :math:`f_1` and 10 samples of :math:`f_0`. The cost of constructing these estimators would be equivalent if the high-fidelity model is 10 times more expensive than the low-fidelity model.
ntrials = 1000
m0_means = np.empty((ntrials, 1))
for ii in range(ntrials):
    samples = benchmark.variable.rvs(nsamples)
    values = model.m0(samples)
    m0_means[ii] = values[:10].mean()

fig, ax = plt.subplots()
textstr = '\n'.join(
    [r'$\mathbb{E}[Q_{1,100}]=\mathrm{%.2e}$' % means[:, 0].mean(),
     r'$\mathbb{V}[Q_{1,100}]=\mathrm{%.2e}$' % means[:, 0].var(),
     r'$\mathbb{E}[Q_{0,10}]=\mathrm{%.2e}$' % m0_means[:, 0].mean(),
     r'$\mathbb{V}[Q_{0,10}]=\mathrm{%.2e}$' % m0_means[:, 0].var()])
ax.hist(means[:, 0], bins=ntrials//100, density=True)
ax.hist(m0_means[:, 0], bins=ntrials//100, density=True, alpha=0.5)
ax.axvline(x=shifts[0], c='r', label=r'$\mathbb{E}[Q_1]$')
ax.axvline(x=0, c='k', label=r'$\mathbb{E}[Q_0]$')
props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 1}
ax.text(0.65, 0.8, textstr, transform=ax.transAxes, bbox=props)
ax.set_xlabel(r'$\mathbb{E}[Q_N]$')
ax.set_ylabel(r'$\mathbb{P}(\mathbb{E}[Q_N])$')
_ = ax.legend(loc='upper left')

#%%
#In a series of tutorials starting with :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_control_variate_monte_carlo.py` we show how to produce an unbiased estimator with small variance using both these models.
