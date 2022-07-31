r"""
Multi-fidelity Monte Carlo
==========================
This tutorial builds on from :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_level_monte_carlo.py` and :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py` and introduces an approximate control variate estimator called Multi-fidelity Monte Carlo (MFMC) [PWGSIAM2016]_. Unlike MLMC this method does not assume a strict ordering of models.

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
from functools import partial
from pyapprox.benchmarks import setup_benchmark
from pyapprox import interface
from pyapprox import multifidelity

benchmark = setup_benchmark("short_column_ensemble")
short_column_model = benchmark.fun
model_costs = np.asarray([100, 50, 5, 1, 0.2])

target_cost = int(1e4)
idx = [0, 1, 2]
cov = short_column_model.get_covariance_matrix()[np.ix_(idx, idx)]
model_ensemble = interface.ModelEnsemble(
    [short_column_model.models[ii] for ii in idx])
costs = model_costs[idx]

# define the sample allocation
est = multifidelity.get_estimator("mfmc", cov, costs, benchmark.variable)
nsample_ratios, variance, rounded_target_cost = est.allocate_samples(
    target_cost)
acv_samples, acv_values = est.generate_data(model_ensemble)
mlmc_mean = est(acv_values)
hf_mean = acv_values[0][1].mean()

# get the true mean of the high-fidelity model
true_mean = short_column_model.get_means()[0]
print('MC error', abs(hf_mean-true_mean))
print('MFMC error', abs(mlmc_mean-true_mean))

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
np.random.seed(1)
benchmark = setup_benchmark("polynomial_ensemble")
poly_model = benchmark.fun
model_ensemble = interface.ModelEnsemble(poly_model.models)
cov = poly_model.get_covariance_matrix()
target_costs = np.array([1e1, 1e2, 1e3, 1e4], dtype=int)
costs = np.asarray([10**-ii for ii in range(cov.shape[0])])
model_labels = [r'$f_0$', r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$']
npilot_samples = 10
cov_mc = multifidelity.estimate_model_ensemble_covariance(
    npilot_samples, benchmark.variable.rvs, model_ensemble,
    model_ensemble.nmodels)[0]

from pyapprox.util.configure_plots import mathrm_labels, mathrm_label
estimators = [
    multifidelity.get_estimator("mfmc", cov, costs, poly_model.variable),
    multifidelity.get_estimator("mc", cov, costs, poly_model.variable)]
est_labels = mathrm_labels(["MFMC", "MC"])
optimized_estimators = multifidelity.compare_estimator_variances(
    target_costs, estimators)

fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
multifidelity.plot_estimator_variances(
    optimized_estimators, est_labels, axs[0],
    ylabel=mathrm_label("Relative Estimator Variance"))
axs[0].set_xlim(target_costs.min(), target_costs.max())
multifidelity.plot_acv_sample_allocation_comparison(
    optimized_estimators[0], model_labels, axs[1])
plt.show()
#fig # necessary for jupyter notebook to reshow plot in new cell


#%%
#References
#^^^^^^^^^^
#.. [PWGSIAM2016] `{Peherstorfer, B., Willcox, K.,  Gunzburger, M., Optimal Model Management for Multifidelity Monte Carlo Estimation, 2016. <https://doi.org/10.1137/15M1046472>`_