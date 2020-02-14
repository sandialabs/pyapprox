r"""
Control Variate Monte Carlo
===========================
This tutorial describes how to implement and deploy control variate Monte Carlo sampling to compute expectations of model output from two models.

Let :math:`f(\rv):\reals^d\to\reals` be a function of :math:`d` random variables :math:`\rv=[\rv_1,\ldots,\rv_d]^T` with joint density :math:`\pdf(\rv)`. Our goal is to compute the expectation of an approximation :math:`f_\alpha` of the function :math:`f`, e.g.

.. math::

   Q_\alpha = \int_{\reals^d} f_\alpha(\rv)\,\pdf(\rv)\,d\rv

The approximation :math:`f_\alpha` typically arises from the need to numerically compute :math:`f`. For example :math:`f` may be a functional of a finite element solution of a system of partial differential equations that cannot be solved analytically.


Monte Carlo
-----------

We can approximate th integral :math:`Q` using Monte Carlo quadrature by drawing :math:`N` random samples of :math:`\rv` from :math:`\pdf` and evaluating the function at each of these samples to obtain the data pairs :math:`\{(\rv^{(n)},f^{(n)}_\alpha)\}_{n=1}^N`, where :math:`f^{(n)}=f_\alpha(\rv^{(n)})`

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
a so called stochastic error (I) and a deterministic bias (II). The first term is the error between the variance in the Monte Carlo estimator due to using a finite number of samples. The second term is due to using an approximation of :math:`f`. These two errors should be balanced, however in the vast majority of all MC analyses a single model $f_\alpha$ is used and the choice of $\alpha$, e.g. mesh resolution, is made a priori without much concern for the balancing bias and variance. 

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
from pyapprox.tests.test_control_variate_monte_carlo import TunableExample
from scipy.stats import uniform

np.random.seed(1)
univariate_variables = [uniform(-1,2),uniform(-1,2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)
print(variable)
shifts=[.1,.2]
model = TunableExample(np.pi/2*.95,shifts=shifts)

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
#Now let us compute the MSE for different sample sets of the same size and plot
#the distribution of the MC estimator :math:`Q_{\alpha,N}`

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

#%%
#Control-variate Monte Carlo (CVMC)
#----------------------------------
#
#Let us introduce a model :math:`Q_\V{\kappa}` with known mean :math:`\mu_{\V{\kappa}}`. We can use this model to estimate the mean of :math:`Q_{\V{\alpha}}` via [LMWOR1982]_
#
#.. math::
#
#   Q_{\V{\alpha},N}^{\text{CV}} = Q_{\V{\alpha},N} + \eta \left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}} \right) 
#
#Where :math:`\eta` is a free parameter which can be optimized to the reduce the variance of this so called control variate estimator
#
#.. math::
#
#   \var{Q_{\V{\alpha},N}^{\text{CV}}} &= \var{Q_{\V{\alpha},N} + \eta \left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}\\
#    &=\var{Q_{\V{\alpha},N}} + \eta^2\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}+ 2\eta^2\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}\\
#    &=\var{Q_{\V{\alpha},N}}\left(1+\eta^2\frac{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}+ 2\eta^2\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}\right)
#
#where the first line follows from the variance of sums of random variables. The variance reduction ratio is
#
#.. math::
#
#   \gamma=\frac{\var{Q_{\V{\alpha},N}^{\text{CV}}}}{\var{Q_{\V{\alpha},N}}}=\left(1+\eta^2\frac{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}+ 2\eta\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}\right)
# 
#The variance reduction is minimized by setting its gradient to zero, i.e.
#
#.. math::
#
#    \frac{d}{d\eta}\gamma  &= 2\eta\frac{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}+ 2\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}} = 0\\
#   &\implies \eta\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}+ \covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)} = 0\\
#   &\implies \eta=-\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}\\
#   &=-\frac{\covar{Q_{\V{\alpha},N}}{Q_{\V{\kappa},N}}}{\var{Q_{\V{\kappa},N}}}
#
#With this choice
#
#.. math::
#
#   \gamma &= 1+\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2}\frac{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}-2\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}\\
#   &= 1+\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}\var{Q_{\V{\alpha},N}}}-2\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}\var{Q_{\V{\alpha},N}}}\\
#    &= 1-\corr{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2\\
#    &= 1-\corr{Q_{\V{\alpha},N}}{Q_{\V{\kappa},N}}^2

#
#Thus if a two highly correlated models (one with a known mean) are available then we can drastically reduce the MSE of our estimate of the unknown mean.
#
#The correlation between the models :math:`f_0` and :math:`f_1` can be tuned by varying :math:`\theta_1`. For a given choice of theta lets compute a single relization of the CVMC estimate of :math:`\mean{f_0}`

samples = pya.generate_independent_random_samples(
    variable,nsamples)
values0 = model.m0(samples)
values1 = model.m1(samples)
cov = model.get_covariance_matrix()
eta = -cov[0,1]/cov[0,0]
#cov_mc = np.cov(values0,values1)
#eta_mc = -cov_mc[0,1]/cov_mc[0,0]
cv_mean = values0.mean()+eta*(values1.mean()-exact_integral_f1)
print('MC difference squared =',(values0.mean()-exact_integral_f0)**2)
print('CVMC difference squared =',(cv_mean-exact_integral_f0)**2)

#%%
# Now lets look at the statistical properties of the CVMC estimator

ntrials=1000
means = np.empty((ntrials,2))
for ii in range(ntrials):
    samples = pya.generate_independent_random_samples(
        variable,nsamples)
    values0 = model.m0(samples)
    values1 = model.m1(samples)
    means[ii,0] = values0.mean()
    means[ii,1] = values0.mean()+eta*(values1.mean()-exact_integral_f1)

print("Theoretical variance reduction",
      1-cov[0,1]**2/(cov[0,0]*cov[1,1]))
print("Achieved variance reduction",
      means[:,1].var(axis=0)/means[:,0].var(axis=0))

#%%
# The following plot shows that unlike :math:`\mean{f_1}` the CVMC estimator is unbiased and has a smaller variance.

fig,ax = plt.subplots()
textstr = '\n'.join([r'$E[Q_{0,N}]=\mathrm{%.2e}$'%means[:,0].mean(),
                     r'$V[Q_{0,N}]=\mathrm{%.2e}$'%means[:,0].var(),
                     r'$E[Q_{0,N}^\mathrm{CV}]=\mathrm{%.2e}$'%means[:,1].mean(),
                     r'$V[Q_{0,N}^\mathrm{CV}]=\mathrm{%.2e}$'%means[:,1].var()])
ax.hist(means[:,0],bins=ntrials//100,density=True,alpha=0.5,
        label=r'$Q_{0,N}$')
ax.hist(means[:,1],bins=ntrials//100,density=True,alpha=0.5,
        label=r'$Q_{0,N}^\mathrm{CV}$')
ax.axvline(x=0,c='k',label=r'$E[Q_0]$')
props = {'boxstyle':'round','facecolor':'white','alpha':1}
ax.text(0.6,0.75,textstr,transform=ax.transAxes,bbox=props)
_ = ax.legend(loc='upper left')
plt.show()

#%%
#Change :math:`\texttt{eta}` to :math:`\texttt{eta\_mc}` to see how the variance reduction changes when the covariance between models is approximated

#%%
#References
#^^^^^^^^^^
#.. [LMWOR1982] `S.S. Lavenberg, T.L. Moeller, P.D. Welch, Statistical results on control variables with application to queueing network simulation, Oper. Res. 30 (1982) 45 182–202. <https://doi.org/10.1287/opre.30.1.182>`_
