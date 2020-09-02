r"""
Approximate Control Variate Monte Carlo
=======================================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_control_variate_monte_carlo.py` and describes how to implement and deploy *approximate* control variate Monte Carlo (ACVMC) sampling to compute expectations of model output from multiple low-fidelity models with unknown means. 

CVMC is often not useful for practical analysis of numerical models because typically the mean of the lower fidelity model, i.e. :math:`\mu_\V{\kappa}`, is unknown and the cost of the lower fidelity model is non trivial. These two issues can be overcome by using approximate control variate Monte Carlo.

Let the cost of the high fidelity model per sample be :math:`C_\alpha` and let the cost of the low fidelity model be :math:`C_\kappa`. Now lets use :math:`N` samples to estimate :math:`Q_{\V{\alpha},N}` and :math:`Q_{\V{\kappa},N}` and these  :math:`N` samples plus another :math:`(r-1)N` samples to estimate :math:`\mu_{\V{\kappa}}` so that

.. math::

   Q_{\V{\alpha},N,r}^{\text{ACV}}=Q_{\V{\alpha},N} + \eta \left( Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r} \right)

and 

.. math::

   \mu_{\V{\kappa},N,r}=\frac{1}{rN}\sum_{i=1}^{rN}Q_\V{\kappa}

With this sampling scheme we have

.. math::

  Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r}&=\frac{1}{N}\sum_{i=1}^N f_\V{\kappa}^{(i)}-\frac{1}{rN}\sum_{i=1}^{rN}f_\V{\kappa}^{(i)}\\
  &=\frac{1}{N}\sum_{i=1}^N f_\V{\kappa}^{(i)}-\frac{1}{rN}\sum_{i=1}^{N}f_\V{\kappa}^{(i)}-\frac{1}{rN}\sum_{i=N}^{rN}f_\V{\kappa}^{(i)}\\
  &=\frac{r-1}{rN}\sum_{i=1}^N f_\V{\kappa}^{(i)}-\frac{1}{rN}\sum_{i=N}^{rN}f_\V{\kappa}^{(i)}\\

where for ease of notation we write :math:`r_\V{\kappa}N` and :math:`\lfloor r_\V{\kappa}N\rfloor` interchangibly.
Using the above expression yields

.. math::
   \var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r}\right)}&=\mean{\left(\frac{r-1}{rN}\sum_{i=1}^N f_\V{\kappa}^{(i)}-\frac{1}{rN}\sum_{i=N}^{rN}f_\V{\kappa}^{(i)}\right)^2}\\
  &=\frac{(r-1)^2}{r^2N^2}\sum_{i=1}^N \var{f_\V{\kappa}^{(i)}}+\frac{1}{r^2N^2}\sum_{i=N}^{rN}\var{f_\V{\kappa}^{(i)}}\\
  &=\frac{(r-1)^2}{r^2N^2}N\var{f_\V{\kappa}}+\frac{1}{r^2N^2}(r-1)N\var{f_\V{\kappa}}\\
  %&=\left(\frac{(r-1)^2}{r^2N}+\frac{(r-1)}{r^2N}\right)\var{f_\V{\kappa}}\\
  &=\frac{r-1}{r}\frac{\var{f_\V{\kappa}}}{N}

where we have used the fact that since the samples used in the first and second term on the first line are not shared, the covariance between these terms is zero. Also we have

.. math::

  \covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r}\right)}=\covar{\frac{1}{N}\sum_{i=1}^N f_\V{\alpha}^{(i)}}{\frac{r-1}{rN}\sum_{i=1}^N f_\V{\kappa}^{(i)}-\frac{1}{rN}\sum_{i=N}^{rN}f_\V{\kappa}^{(i)}}

The correlation between the estimators :math:`\frac{1}{N}\sum_{i=1}^{N}Q_\V{\alpha}` and :math:`\frac{1}{rN}\sum_{i=N}^{rN}Q_\V{\kappa}` is zero because the samples used in these estimators are different for each model. Thus

.. math::

   \covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r}\right)} &=\covar{\frac{1}{N}\sum_{i=1}^N f_\V{\alpha}^{(i)}}{\frac{r-1}{rN}\sum_{i=1}^N f_\V{\kappa}^{(i)}}\\
  &=\frac{r-1}{r}\frac{\covar{f_\V{\alpha}}{f_\V{\kappa}}}{N}

Recalling the variance reduction of the CV estimator using the optimal :math:`\eta` is

.. math::

   \gamma &= 1-\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{ \V{\kappa},N,r}\right)}^2}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r}\right)}\var{Q_{\V{\alpha},N}}}\\
   &=1-\frac{N^{-2}\frac{(r-1)^2}{r^2}\covar{f_\V{\alpha}}{f_\V{\kappa}}}{N^{-1}\frac{r-1}{r}\var{f_\V{\kappa}}N^{-1}\var{f_\V{\alpha}}}\\
   &=1-\frac{r-1}{r}\corr{f_\V{\alpha}}{f_\V{\kappa}}^2

which is found when

.. math::

   \eta&=-\frac{\covar{Q_{\V{\alpha},N}}{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r}\right)}}{\var{\left( Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r}\right)}}\\
  &=-\frac{N^{-1}\frac{r-1}{r}\covar{f_\V{\alpha}}{f_\V{\kappa}}}{N^{-1}\frac{r-1}{r}\var{f_\V{\kappa}}}\\
  &=-\frac{\covar{f_\V{\alpha}}{f_\V{\kappa}}}{\var{f_\V{\kappa}}}

"""
#%%
# Lets setup the problem and compute an ACV estimate of :math:`\mean{f_0}`
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import TunableModelEnsemble
from scipy.stats import uniform

np.random.seed(1)
shifts= [.1,.2]
model = TunableModelEnsemble(1,shifts=shifts)
exact_integral_f0=0

#%%
#Before proceeding to estimate the mean using ACVMV we must first define how to generate samples to estimate :math:`Q_{\V{\alpha},N}` and :math:`\mu_{\V{\kappa},N,r}`. To do so clearly we must first introduce some additional notation. Let :math:`\mathcal{Z}_0` be the set of samples used to evaluate the high-fidelity model and let :math:`\mathcal{Z}_\alpha=\mathcal{Z}_{\alpha,1}\cup\mathcal{Z}_{\alpha,2}` be the samples used to evaluate the low fidelity model. Using this notation we can rewrite the ACV estimator as
#
#.. math::
#
#   Q_{\V{\alpha},\mathcal{Z}}^{\text{ACV}}=Q_{\V{\alpha},\mathcal{Z}_0} + \eta \left( Q_{\V{\kappa},\mathcal{Z}_{\alpha,1}} - \mu_{\V{\kappa},\mathcal{Z}_{\alpha,2}} \right)
#
#where :math:`\mathcal{Z}=\bigcup_{\alpha=0}^M Z_\alpha`. The nature of these samples can be changed to produce different ACV estimators. Here we choose  :math:`\mathcal{Z}_{\alpha,1}\cap\mathcal{Z}_{\alpha,2}=\emptyset` and :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z_0}`. That is we use the set a common set of samples to compute the covariance between all the models and a second independent set to estimate the lower fidelity mean. The sample partitioning for :math:`M` models is  shown in the following Figure. We call this scheme the ACV IS sampling stratecy where IS indicates that the second sample set :math:`\mathcal{Z}_{\alpha,2}` assigned to each model are not shared.
#
#.. list-table::
#
#   * - .. _acv-is-sample-allocation:
#
#       .. figure:: ../../figures/acv_is.png
#          :width: 50%
#          :align: center
#
#          ACV IS sampling strategy
#
#The following code generates samples according to this strategy

nhf_samples = int(1e1)
nsample_ratio = 10
samples_shared = model.generate_samples(nhf_samples)
samples_lf_only =  model.generate_samples(nhf_samples*nsample_ratio-nhf_samples)
values0 = model.m0(samples_shared)
values1_shared = model.m1(samples_shared)
values1_lf_only = model.m1(samples_lf_only)

#%%
#Now lets plot the samples assigned to each model.

fig,ax = plt.subplots()
ax.plot(samples_shared[0,:],samples_shared[1,:],'ro',ms=12,
        label=r'$\mathrm{Low\ and\  high\  fidelity\  models}$')
ax.plot(samples_lf_only[0,:],samples_lf_only[1,:],'ks',
        label=r'$\mathrm{Low\  fidelity\  model\ only}$')
ax.set_xlabel(r'$z_1$')
ax.set_ylabel(r'$z_2$',rotation=0)
_ = ax.legend(loc='upper left')

#%%
#The high-fidelity model is only evaluated on the red dots. Now lets use these samples to estimate the mean of :math:`f_0`.


cov = model.get_covariance_matrix()
gamma = 1-(nsample_ratio-1)/nsample_ratio*cov[0,1]**2/(cov[0,0]*cov[1,1])
eta = -cov[0,1]/cov[1,1]
print(values1_shared.shape,values1_lf_only.shape)
acv_mean = values0.mean()+eta*(values1_shared.mean()-np.concatenate(
    [values1_shared[:,0],values1_lf_only[:,0]]).mean())
print('MC difference squared =',(values0.mean()-exact_integral_f0)**2)
print('ACVMC difference squared =',(acv_mean-exact_integral_f0)**2)

#%%
#Note here we have arbitrarily set the number of high fidelity samples :math:`N` and the ratio :math:`r`. In practice one should choose these in one of two ways: (i) for a fixed budget choose the free parameters to minimize the variance of the estimator; or (ii) choose the free parameters to achieve a desired MSE (variance) with the smallest computational cost. Note the cost of computing the two model ACV estimator is
#
#.. math::
#
#   C_\mathrm{cv} = NC_\alpha + r_\V{\kappa}NC_\kappa
#

#%%
#Now lets compute the variance reduction for different sample sizes
def compute_acv_two_model_variance_reduction(nsample_ratios,functions):
    M = len(nsample_ratios) # number of lower fidelity models
    assert len(functions)==M+1
    
    ntrials=int(1e3)
    means = np.empty((ntrials,2))
    for ii in range(ntrials):
        samples_shared = model.generate_samples(nhf_samples)
        # length M
        samples_lf_only =[
            model.generate_samples(nhf_samples*r-nhf_samples)
            for r in nsample_ratios]
        values_lf_only  =  [
            f(s) for f,s in zip(functions[1:],samples_lf_only)]
        # length M+1
        values_shared  = [f(samples_shared) for f in functions]
        #cov_mc  = np.cov(values_shared,rowvar=False)
        # compute mean using only hf data
        hf_mean = values_shared[0].mean()
        means[ii,0]= hf_mean
        # compute ACV mean
        gamma=1-(nsample_ratios[0]-1)/nsample_ratios[0]*cov[0,1]**2/(
            cov[0,0]*cov[1,1])
        eta = -cov[0,1]/cov[1,1]
        means[ii,1]=hf_mean+eta*(values_shared[1].mean()-
            np.concatenate([values_shared[1],values_lf_only[0]]).mean())

    print("Theoretical ACV variance reduction",
          1-(nsample_ratios[0]-1)/nsample_ratios[0]*cov[0,1]**2/(
              cov[0,0]*cov[1,1]))
    print("Achieved ACV variance reduction",
         means[:,1].var(axis=0)/means[:,0].var(axis=0))
    return means

r1,r2=10,100
print(f'Two model: r={r1}')
means1 = compute_acv_two_model_variance_reduction([r1],[model.m0,model.m1])
print(f'Three model: r={r2}')
means2 = compute_acv_two_model_variance_reduction([r2],[model.m0,model.m1])
print("Theoretical CV variance reduction",1-cov[0,1]**2/(cov[0,0]*cov[1,1]))

#%%
#Let us also plot the distribution of these estimators

ntrials = means1.shape[0]
fig,ax = plt.subplots()
ax.hist(means1[:,0],bins=ntrials//100,density=True,alpha=0.5,
        label=r'$Q_{0,N}$')
ax.hist(means1[:,1],bins=ntrials//100,density=True,alpha=0.5,
        label=r'$Q_{0,N,%d}^\mathrm{CV}$'%r1)
ax.hist(means2[:,1],bins=ntrials//100,density=True,alpha=0.5,
        label=r'$Q_{0,N,%d}^\mathrm{CV}$'%r2)
ax.axvline(x=0,c='k',label=r'$E[Q_0]$')
_ = ax.legend(loc='upper left')

#%%
#For a fixed number of high-fidelity evaluations :math:`N` the ACVMC variance reduction will converge to the CVMC variance reduction. Try changing :math:`N`.

#%%
#References
#^^^^^^^^^^
#.. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification, Journal of Computational Physics, 408:109257, 2020. <https://doi.org/10.1016/j.jcp.2020.109257>`_
