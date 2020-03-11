r"""
Multi-level Monte Carlo
=======================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_control_variate_monte_carlo.py` and describes how the pioneering work of Multi-level Monte Carlo can be used to estimate the mean of a high-fidelity model using multiple low fidelity models. CVMC can be extended to multiple models however it is often not useful for practical analysis of numerical models. CVMC requires the means of the low fidelity models to be known a priori, but typically the mean of the lower fidelity model, i.e. :math:`\mu_\V{\kappa}`, is unknown and the cost of the lower fidelity model is non trivial. 

Let :math:`f_0,\ldots,f_M` be an ensemble of :math:`M+1` models ordered by decreasing fidelity and cost (note typically MLMC literature reverses this order). MLMC provides a way to estimate the Before introducing many model MLMC estimator let us first discuss how to use MLMC with two models. 

The two model MLMC estimator is very similar to the CVMC estimator and is based upon the observations that we can estimate

.. math:: \mean{f_0}=\mean{f_1}+\mean{f_0-f_1}

using the following unbiased estimator

.. math:: Q_{0,\mathcal{Z}}^\mathrm{ML}=N_1^{-1}\sum_{i=1}^{N_1} f_{1,\mathcal{Z}_1}^{(i)}+N_0^{-1}\sum_{i=1}^{N_0} \left(f_{0,\mathcal{Z}_0}^{(i)}-f_{1,\mathcal{Z}_0}^{(i)}\right)

Where samples :math:`\mathcal{Z}=\mathcal{Z}_0\bigcup\mathcal{Z}_1` with :math:`\mathcal{Z}_0\bigcap\mathcal{Z}_1=\emptyset` and :math:`f_{\alpha,\mathcal{Z}_\kappa}^{(i)}` denotes an evaluation of :math:`f_\alpha` using a sample from the set :math:`\mathcal{Z}_\kappa`.
To simplify notation let

.. math:: Y_{\alpha,\mathcal{Z}_{\alpha-1}}=-N_{\alpha}^{-1}\sum_{i=1}^{N_{\alpha}} \left(f_{\alpha+1,\mathcal{Z}_{\alpha}}^{(i)}-f_{\alpha,\mathcal{Z}_{\alpha}}^{(i)}\right)

so we can write 

.. math:: Q_{0,\mathcal{Z}}^\mathrm{ML}=Y_{0,\mathcal{Z}_0}+Y_{1,\mathcal{Z}_1}

where :math:`f_{2}=0`.
The variance of this estimator is given by

.. math:: \var{Q_{0,\mathcal{Z}}^\mathrm{ML}}=\var{Y_{0,\mathcal{Z}_0}+Y_{1,\mathcal{Z}_1}}=N_0^{-1}\var{Y_{0,\mathcal{Z}_0}}+N_1^{-1}\var{Y_{1,\mathcal{Z}_1}}
  
since :math:`\mathcal{Z}_0\bigcap\mathcal{Z}_1=\emptyset`

MLMC works well if the variance of the differnce between models is smaller than the variance of either model.

MLMC can easily be extended to estimator based on :math:`M+1` models. We simply introduce estimates of the differences between the remaining models. That is

.. math :: Q_{0,\mathcal{Z}}^\mathrm{ML} = \sum_{\alpha=0}^M Y_{\alpha,\mathcal{Z}_\alpha}, \quad f_{M+1}=0

Again we use independent samples to estimate the expectation of each discrepancy so the variance of the estimator is

.. math:: \var{Q_{0,\mathcal{Z}}^\mathrm{ML}} = \sum_{\alpha=0}^M N_\alpha\var{Y_{\alpha,\mathcal{Z}_\alpha}}


 
.. list-table::

   * - 
       .. _mlmc-sample-allocation:

       .. figure:: ../../figures/mlmc.png
          :width: 100%
          :align: center

          MLMC sampling strategy

     - 
       .. _acv-is-sample-allocation-mlmc-comparison:

       .. figure:: ../../figures/acv_is.png
          :width: 100%
          :align: center

          ACV IS sampling strategy


Lets setup a problem to compute an MLMC estimate of :math:`\mean{f_0}`.
using the following ensemble of models, taken from [PWGSIAM2016]_, which simulate a short column with h rectangular cross-sectional area subject to bending and axial force.

.. math::

   f_0(z)&=(1 - 4M/(b(h^2)Y) - (P/(bhY))^2)\\
   f_1(z)&=(1 - 3.8M/(b(h^2)Y) - ((P(1 + (M-2000)/4000))/(bhY))^2)\\
   f_2(z)&=(1 - M/(b(h^2)Y) - (P/(bhY))^2)\\
   f_3(z)&=(1 - M/(b(h^2)Y) - (P(1 + M)/(bhY))^2)\\
   f_4(z)&=(1 - M/(b(h^2)Y) - (P(1 + M)/(hY))^2)

where :math:`z = (b,h,P,M,Y)^T`
"""
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import \
    TunableModelEnsemble, ShortColumnModelEnsemble, PolynomialModelEnsemble 
from scipy.stats import uniform
from functools import partial
from scipy.stats import uniform,norm,lognorm
np.random.seed(1)

short_column_model = ShortColumnModelEnsemble()
model_ensemble = pya.ModelEnsemble(
    [short_column_model.m0,short_column_model.m1,short_column_model.m2])

costs = np.asarray([100, 50, 5])
target_cost = int(1e4)
idx = [0,1,2]
cov = short_column_model.get_covariance_matrix()[np.ix_(idx,idx)]
# generate pilot samples to estimate correlation
# npilot_samples = int(1e4)
# cov = pya.estimate_model_ensemble_covariance(
#    npilot_samples,short_column_model.generate_samples,model_ensemble)[0]

# define the sample allocation
nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
    cov, costs, target_cost)[:2]
# generate sample sets
samples,values =pya.generate_samples_and_values_mlmc(
    nhf_samples,nsample_ratios,model_ensemble,
    short_column_model.generate_samples)
# compute mean using only hf data
hf_mean = values[0][0].mean()
# compute mlmc control variate weights
eta = pya.get_mlmc_control_variate_weights(cov.shape[0])
# compute MLMC mean
mlmc_mean = pya.compute_approximate_control_variate_mean_estimate(eta,values)

# get the true mean of the high-fidelity model
true_mean = short_column_model.get_means()[0]
print('MLMC error',abs(mlmc_mean-true_mean))
print('MC error',abs(hf_mean-true_mean))


#%%
#These errors are comparable. However these errors are only for one realiation of the samples sets. To obtain a clearer picture on the benefits of MLMC we need to look at the variance of the estimator.
#
#By viewing MLMC as a control variate we can derive its variance reduction [GGEJJCP2020]_
#
#.. math::  \gamma+1 = - \eta_1^2 \tau_{1}^2 - 2 \eta_1 \rho_{1} \tau_{1} - \eta_M^2 \frac{\tau_{M}}{\hat{r}_{M}} - \sum_{i=2}^M \frac{1}{\hat{r}_{i-1}}\left( \eta_i^2 \tau_{i}^2 + \tau_{i-1}^2 \tau_{i-1}^2 - 2 \eta_i \eta_{i-1} \rho_{i,i-1} \tau_{i} \tau_{i-1} \right),
#   :label: mlmc-variance-reduction
#
#where  :math:`\tau_\alpha=\left(\frac{\var{Q_\alpha}}{\var{Q_0}}\right)^{\frac{1}{2}}`. Recall that and :math:`\hat{r}_\alpha=\lvert\mathcal{Z}_{\alpha,2}\rvert/N` is the ratio of the cardinality of the sets :math:`\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{0,2}`. 
#
#The following code computes the variance reduction of the MLMC estimator, using the 2 models :math:`f_0,f_1`. The variance reduction is estimated numerically by  running MLMC repeatedly with different realizations of the sample sets. The function :math:`\texttt{get_rsquared_mlmc}` is used to return the theoretical variance reduction.
ntrials=1e1
get_cv_weights_mlmc = pya.get_mlmc_control_variate_weights_pool_wrapper
means1, numerical_var_reduction1, true_var_reduction1 = \
    pya.estimate_variance_reduction(
        model_ensemble, cov[:2,:2], short_column_model.generate_samples,
        pya.allocate_samples_mlmc,
        pya.generate_samples_and_values_mlmc, get_cv_weights_mlmc,
        pya.get_rsquared_mlmc,ntrials=ntrials,max_eval_concurrency=1,
        costs=costs[:2],target_cost=target_cost)
print("Theoretical 2 model MLMC variance reduction",true_var_reduction1)
print("Achieved 2 model MLMC variance reduction",numerical_var_reduction1)


#%%
# The numerical estimate of the variance reduction is consistent with the theory.
# Now let us compute the theoretical variance reduction using 3 models

true_var_reduction2 = 1-pya.get_rsquared_mlmc(cov,nsample_ratios)
print("Theoretical 3 model MLMC variance reduction",true_var_reduction2)

#%%
#The variance reduction obtained using three models is only slightly better than when using two models. The difference in variance reduction is dependent on the correlations between the models and the number of samples assigned to each model. However by looking at :eq:`mlmc-variance-reduction` we can see that the variance reduction is bounded by the CVMC estimator (see :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_control_variate_monte_carlo.py`) using the lowest fidelity model with the highest correlation with :math:`f_0`. 
#
#It is also worth empahsizing the MLMC only works when the variance between the model discrepancies decay. One needs to be careful that the variance between discrepancies does indeed decay. Sometimes when the models correspond to different mesh discretizations. Increasing the mesh resolution does not always produce a smaller discrepancy. The following example shows that, for this example, adding a third model actually increases the variance of the MLMC estimator. 

model_ensemble = pya.ModelEnsemble(
    [short_column_model.m0,short_column_model.m3,short_column_model.m4])
idx = [0,3,4]
cov3 = short_column_model.get_covariance_matrix()[np.ix_(idx,idx)]
true_var_reduction3 = 1-pya.get_rsquared_mlmc(cov3,nsample_ratios)
print("Theoretical 3 model MLMC variance reduction for a pathalogical example",true_var_reduction3)

#%%
#Using MLMC for this ensemble of models creates an estimate with a variance orders of magnitude larger than just using the high-fidelity model.

#%%
#Multi-index Monte Carlo
#-----------------------
#Multilevel Monte Carlo utilizes a sequence of models controlled by a single hyper-parameter, specifying the level of discretization for example, in a manner that balances computational cost with increasing accuracy. In many applications, however, multiple hyper-parameters may control the model discretization, such as the mesh and time step sizes. In these situations, it may not be clear how to construct a one-dimensional hierarchy represented by a scalar hyper-parameter. To overcome this limitation, a generalization of multi-level Monte Carlo, referred to as multi-index stochastic collocation (MIMC), was developed to deal with multivariate hierarchies with multiple refinement hyper-parameters [HNTNM2016]_.  PyApprox does not implement MIMC but a surrogate based version called Multi-index stochastic collocation (MISC) is presented in this tutorial :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_index_collocation.py`.

#%%
#References
#^^^^^^^^^^
#.. [HNTNM2016] `A. Haji-Ali, F. Nobile and R. Tempone. Multi-index Monte Carlo: when sparsity meets sampling. Numerische Mathematik, 132(4), 767-806, 2016. <https://doi.org/10.1007/s00211-015-0734-5>`_
