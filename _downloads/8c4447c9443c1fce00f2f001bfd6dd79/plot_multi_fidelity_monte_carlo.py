r"""
Multi-fidelity Monte Carlo
==========================
This tutorial builds on from :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_level_monte_carlo.py` and :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py` and introduces an approximate control variate estimator called Multi-fidelity Monte Carlo (MFMC). Unlike MLMC this method does not assume a strict ordering of models.

Many Model MFMC
---------------

To derive the MFMC estimator first recall the two model ACV estimator

.. math:: Q_{0,\mathcal{Z}}^\mathrm{MF}=Q_{0,\mathcal{Z}_{0}} + \eta\left(Q_{1,\mathcal{Z}_{0}}-\mu_{1,\mathcal{Z}_{1}}\right)

The MFMC estimator can be derived with the following recursive argument. Partition the samples assigned to each model such that
:math:`\mathcal{Z}_\alpha=\mathcal{Z}_{\alpha,1}\cup\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{\alpha,1}\cap\mathcal{Z}_{\alpha,2}=\emptyset`. That is the samples at the next lowest fidelity model are the samples used at all previous levels plus an additional independent set, i.e. :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{\alpha-1}`. See :ref:`mfmc-sample-allocation`. Note the differences between this scheme and the MLMC scheme.

.. list-table::

   * - 
       .. _mfmc-sample-allocation:

       .. figure:: ../../figures/mfmc.png
          :width: 100%
          :align: center

          MFMC sampling strategy

     -
       .. _mlmc-sample-allocation-mfmc-comparison:

       .. figure:: ../../figures/mlmc.png
          :width: 100%
          :align: center

          MLMC sampling strategy

Starting from two models we introduce the next low fidelity model in a way that reduces the variance of the estimate :math:`\mu_{\alpha}`, i.e.

.. math::

   Q_{0,\mathcal{Z}}^\mathrm{MF}&=Q_{0,\mathcal{Z}_{0}} + \eta_1\left(Q_{1,\mathcal{Z}_{1}}-\left(\mu_{1,\mathcal{Z}_{1}}+\eta_2\left(Q_{2,\mathcal{Z}_1}-\mu_{2,\mathcal{Z}_2}\right)\right)\right)\\
   &=Q_{0,\mathcal{Z}_{0}} + \eta_1\left(Q_{1,\mathcal{Z}_{1}}-\mu_{1,\mathcal{Z}_{1}}\right)+\eta_1\eta_2\left(Q_{2,\mathcal{Z}_1}-\mu_{2,\mathcal{Z}_2}\right)

We repeat this process for all low fidelity models to obtain

.. math:: Q_{0,\mathcal{Z}}^\mathrm{MF}=Q_{0,\mathcal{Z}_{0}} + \sum_{\alpha=1}^M\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{\alpha,1}}-\mu_{\alpha,\mathcal{Z}_{\alpha}}\right)

The optimal control variate weights for the MFMC estimator, which minimize the variance of the estimator, are :math:`\eta=(\eta_1,\ldots,\eta_M)^T`, where for :math:`\alpha=1\ldots,M`

.. math:: \eta_\alpha = -\frac{\covar{Q_0}{Q_\alpha}}{\var{Q_\alpha}}

With this choice of weights the variance reduction obtained is given by

.. math:: \gamma = 1-\rho_1^2\left(\frac{r_1-1}{r_1}+\sum_{\alpha=2}^M \frac{r_\alpha-r_{\alpha-1}}{r_\alpha r_{\alpha-1}}\frac{\rho_\alpha^2}{\rho_1^2}\right)

Let us use MFMC to estimate the mean of our high-fidelity model.
"""
import numpy as np
import matplotlib.pyplot as plt
import pyapprox as pya
from functools import partial
from pyapprox.tests.test_control_variate_monte_carlo import \
    TunableModelEnsemble, ShortColumnModelEnsemble, PolynomialModelEnsemble
np.random.seed(1)


short_column_model = ShortColumnModelEnsemble()
model_ensemble = pya.ModelEnsemble(
    [short_column_model.m0,short_column_model.m1,short_column_model.m2])

costs = np.asarray([100, 50, 5])
target_cost = int(1e4)
idx = [0,1,2]
cov = short_column_model.get_covariance_matrix()[np.ix_(idx,idx)]

# define the sample allocation
nhf_samples,nsample_ratios = pya.allocate_samples_mfmc(
    cov, costs, target_cost)[:2]
# generate sample sets
samples,values =pya.generate_samples_and_values_mfmc(
    nhf_samples,nsample_ratios,model_ensemble,
    short_column_model.generate_samples)
# compute mean using only hf data
hf_mean = values[0][0].mean()
# compute mlmc control variate weights
eta = pya.get_mfmc_control_variate_weights(cov)
# compute MLMC mean
mfmc_mean = pya.compute_approximate_control_variate_mean_estimate(eta,values)

# get the true mean of the high-fidelity model
true_mean = short_column_model.get_means()[0]
print('MLMC error',abs(mfmc_mean-true_mean))
print('MC error',abs(hf_mean-true_mean))

#%%
#Optimal Sample Allocation
#-------------------------
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
#Now lets us compare MC with MFMC using optimal sample allocations
poly_model = PolynomialModelEnsemble()
model_ensemble = pya.ModelEnsemble(poly_model.models)
cov = poly_model.get_covariance_matrix()
target_costs = np.array([1e1,1e2,1e3,1e4],dtype=int)
costs = np.asarray([10**-ii for ii in range(cov.shape[0])])
model_labels=[r'$f_0$',r'$f_1$',r'$f_2$',r'$f_3$',r'$f_4$']
print(pya.compute_correlations_from_covariance(cov))
variances, nsamples_history = [],[]
npilot_samples = 5
estimators = [pya.MC,pya.MFMC]
for target_cost in target_costs:
    for estimator in estimators:
        est = estimator(cov,costs)
        nhf_samples,nsample_ratios = est.allocate_samples(target_cost)[:2]
        variances.append(est.get_variance(nhf_samples,nsample_ratios))
        nsamples_history.append(est.get_nsamples(nhf_samples,nsample_ratios))
        print(nsamples_history[-1])
variances = np.asarray(variances)
nsamples_history = np.asarray(nsamples_history)

fig,axs=plt.subplots(1,2,figsize=(2*8,6))
# plot sample allocation
pya.plot_acv_sample_allocation(nsamples_history[1::2],costs,model_labels,axs[1])
mfmc_total_costs = np.array(nsamples_history[1::2]).dot(costs)
mfmc_variances = variances[1::2]
axs[0].loglog(mfmc_total_costs,mfmc_variances,':',label=r'$\mathrm{MFMC}$')
mc_total_costs = np.array(nsamples_history[::2]).dot(costs)
mc_variances = variances[::2]
axs[0].loglog(mc_total_costs,mc_variances,label=r'$\mathrm{MC}$')
axs[0].set_ylim(axs[0].get_ylim()[0],1e-2)
_ = axs[0].legend()
#plt.show()
#fig # necessary for jupyter notebook to reshow plot in new cell
