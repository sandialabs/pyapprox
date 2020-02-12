r"""
Approximate Control Variate Monte Carlo
=======================================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_plot_control_variate_monte_carlo.py` and describes how to implement and deploy *approximate* control variate Monte Carlo (ACVMC) sampling to compute expectations of model output from multiple models. CVMC is often not useful for practical analysis of numerical models because typically the mean of the lower fidelity model, i.e. :math:`\mu_\V{\kappa}` is unknown and the cost of the lower fidelity model is non trivial. These two issues can be overcome by using approximate control variate Monte Carlo.

Two models
^^^^^^^^^^

Let the cost of the high fidelity model per sample be 1 and let the cost of the low fidelity model be :math:`r_\V{\kappa}\ge1`. Now lets use :math:`N` samples to estimate :math:`Q_{\V{\alpha},N}` and :math:`Q_{\V{\kappa},N}` and these  :math:`N` samples plus another :math:`rN` samples to estimate :math:`\mu_{\V{\kappa}}` so that

.. math::

   \mu_{\V{\kappa},N,r}=\frac{1}{rN}\sum_{i=1}^{rN}Q_\V{\kappa}

and

.. math::

   Q_{\V{\alpha},N,r}^{\text{ACV}}=Q_{\V{\alpha},N} + \eta \left( Q_{\V{\kappa},N} - \mu_{\V{\kappa},N,r} \right)


The cost of computing the ACV estimator is

.. math::

   C_\mathrm{cv} = N + (1+r_\V{\kappa})N

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

where we have used the fact that since the samples between the first and second term on the first line are not shared the covariance is zero. Also we have

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

   \eta&=\frac{r(\gamma-1)\var{f_\V{\alpha}}}{(r-1)\covar{f_\V{\alpha}}{f_\V{\kappa}}}\\
   &=\frac{\covar{f_\V{\alpha}}{f_\V{\kappa}}}{\var{f_\V{\alpha}}}

"""
#%%
# Lets setup the problem and compute an ACV estimate of :math:`\mean{f_0}`
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import TunableExample
from scipy.stats import uniform

np.random.seed(1)
univariate_variables = [uniform(-1,2),uniform(-1,2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)
shifts= [.1,.2]
model = TunableExample(1,shifts=shifts)
exact_integral_f0=0

nhf_samples = int(1e1)
nsample_ratio = 10
samples_shared = pya.generate_independent_random_samples(
    variable,nhf_samples)
samples_lf_only = pya.generate_independent_random_samples(
    variable,nhf_samples*nsample_ratio)
values0 = model.m0(samples_shared)
values1_shared = model.m1(samples_shared)
values1_lf_only = model.m1(samples_lf_only)

cov = model.get_covariance_matrix()
gamma = 1-(nsample_ratio-1)/nsample_ratio*cov[0,1]**2/(cov[0,0]*cov[1,1])
eta = -cov[0,1]/cov[0,0]
acv_mean = values0.mean()+eta*(values1_shared.mean()-values1_lf_only.mean())
print('MC difference squared =',(values0.mean()-exact_integral_f0)**2)
print('ACVMC difference squared =',(acv_mean-exact_integral_f0)**2)

#%%
#The high-fidelity model is only evaluated on the red dots.
#
#Now lets plot the samples assigned to each model

fig,ax = plt.subplots()
ax.plot(samples_shared[0,:],samples_shared[1,:],'ro',ms=12,
        label=r'$\mathrm{Low\ and\  high\  fidelity\  models}$')
ax.plot(samples_lf_only[0,:],samples_lf_only[1,:],'ks',
        label=r'$\mathrm{Low\  fidelity\  model\ only}$')
ax.set_xlabel(r'$z_1$')
ax.set_ylabel(r'$z_2$',rotation=0)
_ = ax.legend(loc='upper left')

#%%
#Now lets compute the variance reduction for different sample sizes
def compute_acv_two_model_variance_reduction(nsample_ratios,functions):
    M = len(nsample_ratios) # number of lower fidelity models
    assert len(functions)==M+1
    
    ntrials=1000
    means = np.empty((ntrials,2))
    for ii in range(ntrials):
        samples_shared = pya.generate_independent_random_samples(
            variable,nhf_samples)
        # length M
        samples_lf_only =[
            pya.generate_independent_random_samples(variable,nhf_samples*r)
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
        eta = -cov[0,1]/cov[0,0]
        means[ii,1]=hf_mean+eta*(
            values_shared[1].mean()-values_lf_only[0].mean())

    print("Theoretical ACV variance reduction",
          1-(nsample_ratios[0]-1)/nsample_ratios[0]*cov[0,1]**2/(
              cov[0,0]*cov[1,1]))
    print("Achieved ACV variance reduction",
         means[:,1].var(axis=0)/means[:,0].var(axis=0))
    return means

r1,r2=10,100
means1 = compute_acv_two_model_variance_reduction([r1],[model.m0,model.m1])
means2 = compute_acv_two_model_variance_reduction([r2],[model.m0,model.m1])
print("Theoretical CV variance reduction",1-cov[0,1]**2/(cov[0,0]*cov[1,1]))

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
#Two or more models
#^^^^^^^^^^^^^^^^^^
#
#Control variate Monte Carlo can be easily extended and applied to more than two models. Consider :math:`M` lower fidelity models with sample ratios :math:`r_\alpha>=1`, for :math:`\alpha=1,\ldots,M`. The approximate control variate estimator of the mean of the high-fidelity model :math:`Q_0=\mean{f_0}` is
#
#.. math::
#   Q^{\text{ACV}} &= Q_{0,\mathcal{Z}_{0,1}} + \sum_{\alpha=1}^M \eta_\alpha \left( Q_{\alpha,\mathcal{Z}_{\alpha,1}} - \mu_{\alpha,\mathcal{Z}_{\alpha,2}} \right) =Q_{0,\mathcal{Z}_{0,1}} + \sum_{\alpha=1}^M \eta_\alpha \Delta_{\alpha,\mathcal{Z}_{\alpha,1},\mathcal{Z}_{\alpha,2}}\\&=Q_{0,N}+\V{\eta}\V{\Delta}
#
#Here :math:`\V{\eta}=[\eta_1,\ldots,\eta_M]^T`, :math:`\V{\Delta}=[\Delta_1,\ldots,\Delta_M]^T`, and :math:`\mathcal{Z}_{\alpha,1}`, :math:`\mathcal{Z}_{\alpha,2}` are sample sets that may or may not be disjoint. Specifying the exact nature of these sets, including their cardinality, can be used to design different ACV estimators which will discuss later.
#
#The variance of the ACV estimator is
#
#.. math::
#
#   \var{Q^{\text{ACV}}} = \var{Q_{0}}\left(1+\V{\eta}^T\frac{\covar{\V{\Delta}}{\V{\Delta}}}{\var{Q_0}}\V{\eta}+2\V{\eta}^T\frac{\covar{\V{\Delta}}{Q_0}}{\var{Q_0}}\right)
#
#The control variate weights that produce the minimum variance are given by
#
#.. math::
#
#   \V{\eta} = -\covar{\V{\Delta}}{\V{\Delta}}^{-1}\covar{\V{\Delta}}{Q_0}
#
#The resulting variance reduction is
#
#.. math::
#
#   \gamma =1-\covar{\V{\Delta}}{Q_0}^T\frac{\covar{\V{\Delta}}{\V{\Delta}}^{-1}}{\var{Q_0}}\covar{\V{\Delta}}{Q_0}
#
#The previous formulae require evaluating covarices with the discrepancies :math:`\Delta`. To avoid this we write
#
#.. math::
#
#   \covar{\V{\Delta}}{Q_0}&=N^{-1}\left(\mathrm{diag}\left(F\right)\circ \covar{\V{Q}_\mathrm{LF}}{Q_0}\right)\\
#   \covar{\V{\Delta}}{\V{\Delta}}&=N^{-1}\left(\covar{\V{Q}_\mathrm{LF}}{\V{Q}_\mathrm{LF}}\circ\mathrm{diag}\left(F\right)\right)\\
#
#where :math:`\V{Q}_\mathrm{LF}=[Q_1,\ldots,Q_M]^T` and :math:`\circ` is the Hadamard  (element-wise) product. The matrix :math:`F` is dependent on the sampling scheme used to generate the sets :math:`\mathcal{Z}_{\alpha,1}`, :math:`\mathcal{Z}_{\alpha,2}`. We discuss one useful sampling scheme here [GGEJJCP2020]_.
#
#The most straightforward way to obtain an ACV estimator with the same covariance structure of an CV estimator is to evaluate each model (including the high-fidelity model) at a set of :math:`N` samples  :math:`\mathcal{Z}_{\alpha,1}`. We then evaluate each low fidelity model at an additional :math:`N(1-r_\alpha)` samples :math:`\mathcal{Z}_{\alpha,2}`. That is the sample sets satisfy :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{0,1}\;\forall\alpha>0` and :math:`\mathcal{Z}_{\alpha,2}\setminus\mathcal{Z}_{\alpha,1}\cap\mathcal{Z}_{\kappa,2}\setminus\mathcal{Z}_{\kappa,1}=\emptyset\;\forall\kappa\neq\alpha`.

#Lets apply ACV to three models and this time use some helper functions
# to reduce the amount of code we have to write
def compute_acv_many_model_variance_reduction(nsample_ratios,functions):
    M = len(nsample_ratios) # number of lower fidelity models
    assert len(functions)==M+1
    
    ntrials=1000
    means = np.empty((ntrials,2))
    for ii in range(ntrials):
        samples1,samples2,values1,values2 =\
            pya.generate_samples_and_values_acv_IS(
                nhf_samples,nsample_ratios,functions,variable)
        #cov_mc  = np.cov(values1,rowvar=False)
        # compute mean using only hf data
        hf_mean = values1[0].mean()
        means[ii,0]= hf_mean
        # compute ACV mean
        eta = pya.get_approximate_control_variate_weights(
            cov[:M+1,:M+1],nsample_ratios,pya.get_discrepancy_covariances_IS)
        means[ii:,1] = pya.compute_control_variate_mean_estimate(
            eta,values1,values2)

    print("Theoretical ACV variance reduction",
          1-pya.get_rsquared_acv1(
              cov[:M+1,:M+1],nsample_ratios,pya.get_discrepancy_covariances_IS))
    print("Achieved ACV variance reduction",
          means[:,1].var(axis=0)/means[:,0].var(axis=0))
    return means
print('Two models')
means1 = compute_acv_many_model_variance_reduction([10],[model.m0,model.m1])
print('Three models')
means2 = compute_acv_many_model_variance_reduction([10,10],[model.m0,model.m1,model.m2])

#%%
#The benefit of using three models over two models depends on the correlation between each low fidelity model and the high-fidelity model. The benefit on using more models also depends on the relative cost of evaluating each model, however here we will just investigate the effect of changing correlation. The following code shows the variance reduction (relative to standard Monte Carlo) obtained using CVMC (not approximate CVMC) using 2 (OCV1) and three models (OCV). ACVMC will achieve these variance reductions in the limit as the number of samples of the low fidelity models goes to infinity.

theta1 = np.linspace(model.theta2*1.05,model.theta0*0.95,5)
covs = []
var_reds = []
for th1 in theta1:
    model.theta1=th1
    covs.append(model.get_covariance_matrix())
    OCV_var_red = pya.get_variance_reduction(
        pya.get_control_variate_rsquared,covs[-1],None)
    # use model with largest covariance with high fidelity model
    idx = [0,np.argmax(covs[-1][0,1:])+1]
    assert idx == [0,1] #it will always be the first model
    OCV1_var_red = pya.get_variance_reduction(
        pya.get_control_variate_rsquared,covs[-1][np.ix_(idx,idx)],None)
    var_reds.append([OCV_var_red,OCV1_var_red])
covs = np.array(covs)
var_reds = np.array(var_reds)

fig,axs = plt.subplots(1,2,figsize=(2*8,6))
for ii,jj, in [[0,1],[0,2],[1,2]]:
    axs[0].plot(theta1,covs[:,ii,jj],'o-',
                label=r'$\rho_{%d%d}$'%(ii,jj))
axs[1].plot(theta1,var_reds[:,0],'o-',label=r'$\mathrm{OCV}$')
axs[1].plot(theta1,var_reds[:,1],'o-',label=r'$\mathrm{OCV1}$')
axs[1].plot(theta1,var_reds[:,0]/var_reds[:,1],'o-',
            label=r'$\mathrm{OCV/OCV1}$')
axs[0].set_xlabel(r'$\theta_1$')
axs[0].set_ylabel(r'$\mathrm{Correlation}$')
axs[1].set_xlabel(r'$\theta_1$')
axs[1].set_ylabel(r'$\mathrm{Variance reduction ratio} \ \gamma$')
axs[0].legend()
_ = axs[1].legend()

#%%
#The variance reduction clearly depends on the correlation between all the models.

#%%
#Multi-level Monte Carlo (MLMC)
#------------------------------
#
#Total cost is
#
#.. math::
#
#   C_{\mathrm{tot}}=\sum_{l=1}^L C_lr_lN_1
#   
#Variance of estimator is
#
#.. math::
#  
#   \var{Q_L}=\sum_{l=1}^L \var{Y_l}r_lN_1
#   
#Treating :math:`r_l` as a continuous variable the variance of the MLMC estimator is minimized for a fixed budget :math:`C` by setting
#
#.. math::
#
#   N_l=r_lN_1=\sqrt{\var{Y_l}/C_l}
#   
#Choose L so that
#
#.. math::
#   
#   \left(\mean{Q_L}-\mean{Q}\right)^2<\frac{1}{2}\epsilon^2
#   
#Choose :math:`N_l` so total variance
#
#.. math::
#   \var{Q_L}<\frac{1}{2}\epsilon^2
#
#Multi-fidelity Monte Carlo (MFMC)
#---------------------------------
#
#.. math::
#   
#   r_i=\left(\frac{C_1(\rho^2_{1i}-\rho^2_{1i+1})}{C_i(1-\rho^2_{12})}\right)^{\frac{1}{2}}
#   
#Let :math:`C=(C_1\cdots C_L)^T r=(r_1\cdots r_L)^T` then
#
#.. math::
#
#   N_1=\frac{C_{\mathrm{tot}}}{C^Tr} & & N_i=r_iN_1\\
#
#  
#The control variate weights are
#
#.. math::
#   
#   \alpha_i=\frac{\rho_{1i}\sigma_1}{\sigma_i}

#%%
#References
#^^^^^^^^^^
#.. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification, Journal of Computational Physics, In press, (2020) <https://doi.org/10.1016/j.jcp.2020.109257>`_
