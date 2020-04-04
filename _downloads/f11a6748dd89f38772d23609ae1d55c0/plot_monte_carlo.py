r"""
Monte Carlo Quadrature
======================
This tutorial describes how to use Monte Carlo sampling to compute the expectations of the output of an model. The following tutorials on multi-fidelity Monate Carlo methods assume the reader has understanding of the material presented here.

We can approximate th integral :math:`Q_\alpha` using Monte Carlo quadrature by drawing :math:`N` random samples of :math:`\rv` from :math:`\pdf` and evaluating the function at each of these samples to obtain the data pairs :math:`\{(\rv^{(n)},f^{(n)}_\alpha)\}_{n=1}^N`, where :math:`f^{(n)}_\alpha=f_\alpha(\rv^{(n)})`

.. math::

   Q_{\alpha,N}=N^{-1}\sum_{n=1}^N f^{(n)}_\alpha

The mean squared error (MSE) of this estimator can be expressed as

.. math::
   
   \mean{\left(Q_{\alpha,N}-\mean{Q}\right)^2}&=\mean{\left(Q_{\alpha,N}-\mean{Q_{\alpha,N}}+\mean{Q_{\alpha,N}}-\mean{Q}\right)^2}\\
   &=\mean{\left(Q_{\alpha,N}-\mean{Q_{\alpha,N}}\right)^2}+\mean{\left(\mean{Q_{\alpha,N}}-\mean{Q}\right)^2}\\
   &\qquad\qquad+\mean{2\left(Q_{\alpha,N}-\mean{Q_{\alpha,N}}\right)\left(\mean{Q_{\alpha,N}}-\mean{Q}\right)}\\
   &=\var{Q_{\alpha,N}}+\left(\mean{Q_{\alpha,N}}-\mean{Q}\right)^2
   
Here we used that :math:`Q_{\alpha,N}` is an unbiased estimator, i.e. :math:`\mean{Q_{\alpha,N}}=\mean{Q}` so the third term on the second line is zero. Now using

.. math::

   \var{Q_{\alpha,N}}=\var{N^{-1}\sum_{n=1}^N f^{(n)}_\alpha}=N^{-1}\sum_{n=1}^N \var{f^{(n)}_\alpha}=N^{-1}\var{Q_\alpha}

yields

.. math::

   \mean{\left(Q_{\alpha}-\mean{Q}\right)^2}=\underbrace{N^{-1}\var{Q_\alpha}}_{I}+\underbrace{\left(\mean{Q_{\alpha}}-\mean{Q}\right)^2}_{II}

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
# Lets setup the problem
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import TunableModelEnsemble
from scipy.stats import uniform

np.random.seed(1)
univariate_variables = [uniform(-1,2),uniform(-1,2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)
print(variable)
shifts=[.1,.2]
model = TunableModelEnsemble(np.pi/2*.95,shifts=shifts)

#%%
# Now let us compute the mean of :math:`f_1` using Monte Carlo
nsamples = int(1e2)
samples = pya.generate_independent_random_samples(
    variable,nsamples)
values = model.m1(samples)
pya.print_statistics(samples,values)

#%%
# We can compute the exact mean using sympy and compute the MC MSE
import sympy as sp
z1,z2 = sp.Symbol('z1'),sp.Symbol('z2')
ranges = [-1,1,-1,1]
integrand_f1=model.A1*(sp.cos(model.theta1)*z1**3+sp.sin(model.theta1)*z2**3)+shifts[0]*0.25
exact_integral_f1 = float(
    sp.integrate(integrand_f1,(z1,ranges[0],ranges[1]),(z2,ranges[2],ranges[3])))

print('MC difference squared =',(values.mean()-exact_integral_f1)**2)

#%%
#.. _estimator-histogram:
#
#Now let us compute the MSE for different sample sets of the same size and plot the distribution of the MC estimator :math:`Q_{\alpha,N}`
#

ntrials=1000
means = np.empty(ntrials)
for ii in range(ntrials):
    samples = pya.generate_independent_random_samples(
        variable,nsamples)
    values = model.m1(samples)
    means[ii] = values.mean()
fig,ax = plt.subplots()
textstr = '\n'.join([r'$E[Q_{1,N}]=\mathrm{%.2e}$'%means.mean(),
                     r'$V[Q_{1,N}]=\mathrm{%.2e}$'%means.var()])
ax.hist(means,bins=ntrials//100,density=True)
ax.axvline(x=shifts[0],c='r',label=r'$E[Q_1]$')
ax.axvline(x=0,c='k',label=r'$E[Q_0]$')
props = {'boxstyle':'round','facecolor':'white','alpha':1}
ax.text(0.65,0.9,textstr,transform=ax.transAxes,bbox=props)
_ = ax.legend(loc='upper left')

#%%
#The numerical results match our theory. Specifically the estimator is unbiased( i.e. mean zero, and the variance of the estimator is :math:`\var{Q_{0,N}}=\var{Q_{0}}/N=1/N`.
#
#The variance of the estimator can be driven to zero by increasing the number of samples :math:`N`. However when the variance becomes less than the bias, i.e. :math:`\left(\mean{Q_{\alpha}-Q}\right)^2>\var{Q_{\alpha}}/N`, then the MSE will not decrease and any further samples used to reduce the variance are wasted.
#
#Let our true model be :math:`f_0` above. The following code compues the bias induced by using :math:`f_\alpha=f_1` and also plots the contours of :math:`f_0(\rv)-f_1(\rv)`.

integrand_f0 = model.A0*(sp.cos(model.theta0)*z1**5+
                         sp.sin(model.theta0)*z2**5)*0.25
exact_integral_f0 = float(
    sp.integrate(integrand_f0,(z1,ranges[0],ranges[1]),(z2,ranges[2],ranges[3])))
bias = (exact_integral_f0-exact_integral_f1)**2
print('MC f1 bias =',bias)
print('MC f1 variance =',means.var())
print('MC f1 MSE =',bias+means.var())

fig,ax = plt.subplots()
X,Y,Z = pya.get_meshgrid_function_data(
    lambda z: model.m0(z)-model.m1(z),[-1,1,-1,1],50)
cset = ax.contourf(X, Y, Z, levels=np.linspace(Z.min(),Z.max(),20))
_ = plt.colorbar(cset,ax=ax)
#plt.show()

#%%
#As :math:`N\to\infty` the MSE will only converge to the bias (:math:`s_1`). Try this by increasing :math:`\texttt{nsamples}`.
