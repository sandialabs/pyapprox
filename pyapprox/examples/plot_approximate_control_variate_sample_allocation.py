r"""
Sampling Allocation for Approximate Control Variate Monte Carlo Methods
=======================================================================
This tutorial builds upon the tutorials :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py` and :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_recursive_control_variate_monte_carlo.py`.

In the previous tutorials we investigated the peformance of different estimators when the number of high-fidelity samples is fixed. This can be useful when one has no ability to generate more high-fidelity data. However in situations when such data can be generated we should choose the number of low fidelity samples and the number of high-fidelity samples in one of two ways: 

1. Minimze the variance of the estimator for a fixed budget; or 
2. Minimize the computational cost of an estimator with a fixed variance.

In the following we define demonstrate how to determine the optimal sample allocations that satisfy these goals, for a number of different ACV estimators.

Multilevel Monte Carlo
----------------------
In this section we follow [GAN2015]_ and show how to determine the optimal number of samples assigned to each model used in a MLMC estimator.

Let :math:`C_\alpha` be the cost of evaluating the function :math:`f_\alpha` at a single sample, then the total cost of the MLMC estimator is

.. math::

   C_{\mathrm{tot}}=\sum_{l=0}^M C_\alpha N_\alpha
   
and the variance of the estimator is

.. math:: \var{Q_0^\mathrm{ML}}=\sum_{\alpha=0}^M \var{Y_\alpha}N_\alpha,

where :math:`Y_\alpha` is the disrepancy between two consecutive models, e.g. :math:`f_{\alpha-1}-f_\alpha` and :math:`N_\alpha` be the number of samples allocated to resolving the discrepancy, i.e. :math:`N_\alpha=\lvert\hat{\mathcal{Z}}_\alpha\rvert`

For a fixed variance :math:`\epsilon^2` the cost of the MLMC estimator can be minimized, by solving

.. math::

  \min_{N_0,\ldots,N_M} & \sum_{\alpha=0}^M\left(N_\alpha C_\alpha\right)\\
  \mathrm{subject}\; \mathrm{to} &\sum_{\alpha=0}^M\left(N_\alpha^{-1}\var{Y_\alpha}\right)=\epsilon^2

or alternatively by introducing the lagrange multiplier :math:`\lambda^2` we can minimize

.. math:: 

   \mathcal{J}(N_0,\ldots,N_M,\lambda)&=\sum_{\alpha=0}^M\left(N_\alpha C_\alpha\right)+\lambda^2\left(\sum_{\alpha=0}^M\left(N_\alpha^{-1}\var{Y_\alpha}\right)-\epsilon^2\right)\\
   &=\sum_{\alpha=0}^M\left(N_\alpha C_\alpha+\lambda^2N_\alpha^{-1}\var{Y_\alpha}\right)-\lambda^2\epsilon^2

To find the minimum we set the gradient of this expression to zero:

.. math::

  \frac{\partial \mathcal{J}^\mathrm{ML}}{N_\alpha}&=C_\alpha-\lambda^2N_\alpha^{-2}\var{Y_\alpha}=0\\
  \implies C_\alpha&=\lambda^2N_\alpha^{-2}\var{Y_\alpha}\\
  \implies N_\alpha&=\lambda\sqrt{\var{Y_\alpha}C_\alpha^{-1}}

The constraint is satisifed by noting 

.. math:: \frac{\partial \mathcal{J}}{\lambda^2}=\sum_{\alpha=0}^M N_\alpha^{-1}\var{Y_\alpha}-\epsilon^2=0

Recalling that we can write the total variance as

.. math::

  \var{Q_{0,\mathcal{Z}}^\mathrm{ML}}&=\sum_{\alpha=0}^M N_\alpha^{-1} \var{Y_\alpha}\\
  &=\sum_{\alpha=0}^M \lambda^{-1}\var{Y_\alpha}^{-\frac{1}{2}}C_\alpha^{\frac{1}{2}}\var{Y_\alpha}\\
  &=\lambda^{-1}\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}=\epsilon^2\\
  \implies \lambda &= \epsilon^{-2}\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}

Then substituting :math:`\lambda` into the following

.. math::

  N_\alpha C_\alpha&=\lambda\sqrt{\var{Y_\alpha}C_\alpha^{-1}}C_\alpha\\
  &=\lambda\sqrt{\var{Y_\alpha}C_\alpha}\\
  &=\epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)\sqrt{\var{Y_\alpha}C_\alpha}



allows us to determine the smallest total cost that generates and estimator with the desired variance.

.. math::

  C_\mathrm{tot}&=\sum_{\alpha=0}^M N_\alpha C_\alpha\\
  &=\sum_{\alpha=0}^M \epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)\sqrt{\var{Y_\alpha}C_\alpha}\\
  &=\epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)^2

Again consider the model ensemble

.. math:: f_\alpha(\rv)=\rv^{5-\alpha}, \quad \alpha=0,\ldots,4

where each model is the function of a single uniform random variable defined on the unit interval :math:`[0,1]`.

The following code computes the variance of the MLMC estimator for different target costs using the optimal sample allocation using an exact estimate of the covariance between models and an approximation.
"""

import numpy as np
import pyapprox as pya
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import \
    PolynomialModelEnsemble
np.random.seed(1)

poly_model = PolynomialModelEnsemble()
model_ensemble = pya.ModelEnsemble(poly_model.models)
cov = poly_model.get_covariance_matrix()
target_costs = np.array([1e1,1e2,1e3,1e4],dtype=int)
costs = np.asarray([10**-ii for ii in range(cov.shape[0])])
model_labels=[r'$f_0$',r'$f_1$',r'$f_2$',r'$f_3$',r'$f_4$']

def plot_mlmc_error():
    """
    Define function to create plot so we can create it later and add
    other estimator error curves.

    Note this is only necessary for sphinx-gallery docs. If just using the 
    jupyter notebooks they create then we do not need this function definition 
    and simply need to call fig at end of next cell to regenerate plot.
    """
    variances, nsamples_history = [],[]
    npilot_samples = 5
    estimator = pya.MLMC
    for target_cost in target_costs:
        # compute variance  using exact covariance for sample allocation
        est = estimator(cov,costs)
        nhf_samples,nsample_ratios = est.allocate_samples(target_cost)[:2]
        variances.append(est.get_variance(nhf_samples,nsample_ratios))
        nsamples_history.append(est.get_nsamples(nhf_samples,nsample_ratios))
        # compute single fidelity Monte Carlo variance
        total_cost = nsamples_history[-1].dot(costs)
        variances.append(cov[0,0]/int(total_cost/costs[0]))
        nsamples_history.append(int(total_cost/costs[0]))
        # compute variance using approx covariance for sample allocation
        # use nhf_samples from previous target_cost as npilot_samples.
        # This way the pilot samples are only an additional cost at the first
        # step. This code does not do this though for simplicity
        cov_approx = pya.estimate_model_ensemble_covariance(
            npilot_samples,poly_model.generate_samples,model_ensemble)[0]
        est = estimator(cov_approx,costs)
        nhf_samples,nsample_ratios = est.allocate_samples(target_cost)[:2]
        variances.append(est.get_variance(nhf_samples,nsample_ratios))
        nsamples_history.append(est.get_nsamples(nhf_samples,nsample_ratios))
        npilot_samples = nhf_samples

    fig,axs=plt.subplots(1,2,figsize=(2*8,6))
    pya.plot_acv_sample_allocation(nsamples_history[::3],costs,model_labels,axs[1])
    total_costs = np.array(nsamples_history[::3]).dot(costs)
    axs[0].loglog(total_costs,variances[::3],label=r'$\mathrm{MLMC}$')
    mc_line = axs[0].loglog(total_costs,variances[1::3],label=r'$\mathrm{MC}$')
    total_costs = np.array(nsamples_history[2::3]).dot(costs)
    axs[0].loglog(total_costs,variances[2::3],'--',
                  label=r'$\mathrm{MLMC^\dagger}$')
    axs[0].set_xlabel(r'$\mathrm{Total}\;\mathrm{Cost}$')
    axs[0].set_ylabel(r'$\mathrm{Variance}$')
    _ = axs[0].legend()
    return fig,axs

fig,axs = plot_mlmc_error()
    
#%%
#The left plot shows that the variance of the MLMC estimator is over and order of magnitude smaller than the variance of the single fidelity MC estimator for a fixed cost. The impact of using the approximate covariance is more significant for small samples sizes.
#
#The right plot depicts the percentage of the computational cost due to evaluating each model. The numbers in the bars represent the number of samples allocated to each model. Relative to the low fidelity models only a small number of samples are allocated to the high-fidelity model, however evaluating these samples represents approximately 50\% of the total cost.

#%%
#Multi-fidelity Monte Carlo
#--------------------------
#Similarly to MLMC, the optimal number of samples that minimize the variance of the MFMC estimator can be determined analytically (see [PWGSIAM2016]_). Recalling that :math:`C_\mathrm{tot}` is the total budget then the optimal number of high fidelity samples is
#
#.. math:: N_0 = \frac{C_\mathrm{tot}}{\V{w}^T\V{r}}
#
#where :math:`\V{r}=[r_0,\ldots,r_M]^T` are the sample ratios defining the number of samples assigned to each level, i.e. :math:`N_\alpha=r_\alpha N_0`. The sample ratios are
#
#.. math::
#   
#   r_\alpha=\left(\frac{w_0(\rho^2_{0,\alpha}-\rho^2_{0,\alpha+1})}{w_\alpha(1-\rho^2_{0,1})}\right)^{\frac{1}{2}}
#
#where :math:`\V{w}=[w_0,w_M]^T` are the relative costs of each model, and :math:`\rho_{j,k}` is the correlation between models :math:`j` and :math:`k`.
#

#
#Now lets us compare MLMC with ACV-MF, MFMC and ACV-KL.
variances, nsamples_history = [],[]
npilot_samples = 5
estimators = [pya.MFMC,pya.ACVMF]
for target_cost in target_costs:
    for estimator in estimators:
        est = estimator(cov,costs)
        nhf_samples,nsample_ratios = est.allocate_samples(target_cost)[:2]
        variances.append(est.get_variance(nhf_samples,nsample_ratios))
        nsamples_history.append(est.get_nsamples(nhf_samples,nsample_ratios))
variances = np.asarray(variances)
nsamples_history = np.asarray(nsamples_history)

fig,axs = plot_mlmc_error()
del axs[0].lines[1] # delete single MC curve
del axs[0].lines[1] # delete single MLMC approx cov curve
axs[1].clear()
# plot ACVMF sample allocation
pya.plot_acv_sample_allocation(nsamples_history[1::2],costs,model_labels,axs[1])
mfmc_total_costs = np.array(nsamples_history[::2]).dot(costs)
mfmc_variances = variances[::2]
axs[0].loglog(mfmc_total_costs,mfmc_variances,':',label=r'$\mathrm{MFMC}$')
acvmf_total_costs = np.array(nsamples_history[1::2]).dot(costs)
acvmf_variances = variances[1::2]
axs[0].loglog(acvmf_total_costs,acvmf_variances,label=r'$\mathrm{ACV}-\mathrm{MF}$')
axs[0].set_ylim(axs[0].get_ylim()[0],1e-3)
_ = axs[0].legend()
#fig # necessary for jupyter notebook to reshow plot in new cell

#%%
#In this example ACV-KL is a more efficient estimator, i.e. it has a smaller variance for a fixed cost. However this improvement is problem dependent. For other model ensembles another estimator may be more efficient. Modify the above example to use another model to explore this more. The left plot shows the relative costs of evaluating each model using the ACVMF sampling strategy. Compare this to the MLMC sample allocation. Also edit above code to plot the MFMC sample allocation.

#%%
#Before this tutorial ends it is worth noting that a section of the MLMC literature explores adaptive methods which do not assume there is a fixed high-fidelity model but rather attempt to balance the estimator variance with the deterministic bias. These methods add a higher-fidelity model, e.g. a finer finite element mesh, when the variance is made smaller than the bias. We will not explore this here, but an example of this is shown in the tutorial on multi-index collocation.

#%%
#..
# nmodels  = 3
# num_vars = 100
# max_eval_concurrency = 1
# from pyapprox.examples.multi_index_advection_diffusion import *
# base_model = setup_model(num_vars,max_eval_concurrency)
# from pyapprox.models.wrappers import MultiLevelWrapper
# multilevel_model=MultiLevelWrapper(
#     base_model,base_model.base_model.num_config_vars,
#     base_model.cost_function)
# from scipy.stats import uniform
# import pyapprox as pya
# variable = pya.IndependentMultivariateRandomVariable(
#     [uniform(-np.sqrt(3),2*np.sqrt(3))],[np.arange(num_vars)])
#
# npilot_samples = 10
# pilot_samples = pya.generate_independent_random_samples(
#     variable,npilot_samples)
# config_vars = np.arange(nmodels)[np.newaxis,:]
# pilot_samples = pya.get_all_sample_combinations(pilot_samples,config_vars)
# pilot_values = multilevel_model(pilot_samples)
# assert pilot_values.shape[1]==1
# pilot_values = np.reshape(pilot_values,(npilot_samples,nmodels))
# # mlmc requires model accuracy to decrease with index
# # but model assumes the opposite. so reverse order here
# pilot_values = pilot_values[:,::-1]
# cov = np.cov(pilot_values,rowvar=False)
# print(pya.get_correlation_from_covariance(cov))
# for ii in range(nmodels-1):
#     vardelta = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
#     print(vardelta)
#
# target_cost = 10
# # mlmc requires model accuracy to decrease with index
# # but model assumes the opposite. so reverse order here
# costs = [multilevel_model.cost_function(ii) for ii in range(nmodels)][::-1]
# print(costs)
# nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
#     cov, costs, target_cost)[:2]
#
# import seaborn as sns
# from pandas import DataFrame
# df = DataFrame(
#     index=np.arange(pilot_values.shape[0]),
#     data=dict([(r'$f_%d$'%ii,pilot_values[:,ii])
#                for ii in range(pilot_values.shape[1])]))
# # heatmap does not currently work with matplotlib 3.1.1 downgrade to
# # 3.1.0 using pip install matplotlib==3.1.0
# #sns.heatmap(df.corr(),annot=True,fmt='.2f',linewidth=0.5)
# #plt.show()

#%%
#References
#^^^^^^^^^^
#.. [PWGSIAM2016] `B. Peherstorfer, K. Willcox, M. Gunzburger, Optimal model management for multifidelity Monte Carlo estimation, SIAM J. Sci. Comput., 38(5), A3163–A3194, 2016. <https://doi.org/10.1137/15M1046472>`_
#
#.. [GAN2015] `M. Giles, Multilevel Monte Carlo methods, Acta Numerica, 24, 259-328, 2015. <https://doi.org/10.1017/S096249291500001X>`_
