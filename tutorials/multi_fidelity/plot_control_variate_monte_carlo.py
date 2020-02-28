r"""
Control Variate Monte Carlo
===========================
This tutorial describes how to implement and deploy control variate Monte Carlo sampling to compute the expectations of the output of a high-fidelity model using a lower-fidelity model with a known mean. The information presented here builds upon the tutorial :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_monte_carlo.py`.

Let us introduce a model :math:`Q_\V{\kappa}` with known mean :math:`\mu_{\V{\kappa}}`. We can use this model to estimate the mean of :math:`Q_{\V{\alpha}}` via [LMWOR1982]_

.. math::

  Q_{\V{\alpha},N}^{\text{CV}} &= Q_{\V{\alpha},N} + \eta \left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}} \right) \\

Here :math:`\eta` is a free parameter which can be optimized to the reduce the variance of this so called control variate estimator, which is given by

.. math::

  \var{Q_{\V{\alpha},N}^{\text{CV}}} &= \var{Q_{\V{\alpha},N} + \eta \left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}\\
   &=\var{Q_{\V{\alpha},N}} + \eta^2\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}+ 2\eta^2\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}\\
   &=\var{Q_{\V{\alpha},N}}\left(1+\eta^2\frac{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}+ 2\eta^2\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}\right).

The first line follows from the variance of sums of random variables.

We can measure the change in MSE bys using the control variate estimator, by looking at the ratio of the CVMC and MC estimator variances. The variance reduction ratio is

.. math::

  \gamma=\frac{\var{Q_{\V{\alpha},N}^{\text{CV}}}}{\var{Q_{\V{\alpha},N}}}=\left(1+\eta^2\frac{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}+ 2\eta\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}\right)

The variance reduction can be minimized by setting its gradient to zero and solving for :math:`\eta`, i.e.

.. math::

   \frac{d}{d\eta}\gamma  &= 2\eta\frac{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}+ 2\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}} = 0\\
  &\implies \eta\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}+ \covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)} = 0\\
  &\implies \eta=-\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}\\
  &=-\frac{\covar{Q_{\V{\alpha},N}}{Q_{\V{\kappa},N}}}{\var{Q_{\V{\kappa},N}}}

With this choice

.. math::

  \gamma &= 1+\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2}\frac{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}-2\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}}{\var{Q_{\V{\alpha},N}}}\\
  &= 1+\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}\var{Q_{\V{\alpha},N}}}-2\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}\var{Q_{\V{\alpha},N}}}\\
   &= 1-\corr{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa}}\right)}^2\\
   &= 1-\corr{Q_{\V{\alpha},N}}{Q_{\V{\kappa},N}}^2


Thus if a two highly correlated models (one with a known mean) are available then we can drastically reduce the MSE of our estimate of the unknown mean.

Again consider the tunable model ensemble. The correlation between the models :math:`f_0` and :math:`f_1` can be tuned by varying :math:`\theta_1`. For a given choice of theta lets compute a single relization of the CVMC estimate of :math:`\mean{f_0}`
"""

#%%
# First let us setup the problem and compute a single estimate using CVMC
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import TunableModelEnsemble
from scipy.stats import uniform

np.random.seed(1)
shifts=[.1,.2]
model = TunableModelEnsemble(np.pi/2*.95,shifts=shifts)

nsamples = int(1e2)
samples = model.generate_samples(nsamples)
values0 = model.m0(samples)
values1 = model.m1(samples)
cov = model.get_covariance_matrix()
eta = -cov[0,1]/cov[0,0]
#cov_mc = np.cov(values0,values1)
#eta_mc = -cov_mc[0,1]/cov_mc[0,0]
exact_integral_f0,exact_integral_f1=0,shifts[0]
cv_mean = values0.mean()+eta*(values1.mean()-exact_integral_f1)
print('MC difference squared =',(values0.mean()-exact_integral_f0)**2)
print('CVMC difference squared =',(cv_mean-exact_integral_f0)**2)

#%%
# Now lets look at the statistical properties of the CVMC estimator

ntrials=1000
means = np.empty((ntrials,2))
for ii in range(ntrials):
    samples = model.generate_samples(nsamples)
    values0 = model.m0(samples)
    values1 = model.m1(samples)
    means[ii,0] = values0.mean()
    means[ii,1] = values0.mean()+eta*(values1.mean()-exact_integral_f1)

print("Theoretical variance reduction",
      1-cov[0,1]**2/(cov[0,0]*cov[1,1]))
print("Achieved variance reduction",
      means[:,1].var(axis=0)/means[:,0].var(axis=0))

#%%
# The following plot shows that unlike the :ref:`Monte Carlo estimator <estimator-histogram>`. :math:`\mean{f_1}` the CVMC estimator is unbiased and has a smaller variance.

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
#Change :math:`\texttt{eta}` to :math:`\texttt{eta_mc}` to see how the variance reduction changes when the covariance between models is approximated

#%%
#References
#^^^^^^^^^^
#.. [LMWOR1982] `S.S. Lavenberg, T.L. Moeller, P.D. Welch, Statistical results on control variables with application to queueing network simulation, Oper. Res., 30, 45, 182-202, 1982. <https://doi.org/10.1287/opre.30.1.182>`_
