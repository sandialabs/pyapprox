r"""
Multi-level Monte Carlo
=======================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py` and describes how the pioneering work of Multi-level Monte Carlo [CGSTCVS2011]_, [GOR2008]_ can be used to estimate the mean of a high-fidelity model using multiple low fidelity models.

Two Model MLMC
--------------
The two model MLMC estimator is based upon the observations that we can estimate

.. math:: \mean{f_0}=\mean{f_1}+\mean{f_0-f_1}

using the following unbiased estimator

.. math:: Q_{0}^\mathrm{ML}(\rvset_0,\rvset_1)=N_1^{-1}\sum_{i=1}^{N_1} f_{1}^{(i)}(\rvset_1)+N_0^{-1}\sum_{i=1}^{N_0} \left(f_{0}^{(i)}(\rvset_0)-f_{1}^{(i)}(\rvset_0)\right),

where :math:`f_{\alpha}^{(i)}(\rvset_\kappa)` denotes an evaluation of :math:`f_\alpha` using a sample from the set :math:`\rvset_\kappa`. To evaluate this estimator we must evaluate the low fidelity model at a set of samples  :math:`\rvset_1` and then evaluate the difference between the two models at another independet set :math:`\rvset_0`.

To simplify notation let

.. math:: Y_{\alpha}(\rvset_{\alpha})=-N_{\alpha}^{-1}\sum_{i=1}^{N_{\alpha}} \left(f_{\alpha+1}^{(i)}(\rvset_{\alpha})-f_{\alpha}^{(i)}(\rvset_{\alpha})\right)

so we can write

.. math:: Q_{0}^\mathrm{ML}(\rvset_0,\rvset_1)=Y_{0}(\rvset_0)+Y_{1}(\rvset_1)

where :math:`f_{2}=0`.

Now it is easy to see that the variance of this estimator is given by

.. math:: \var{Q_{0}^\mathrm{ML}(\rvset_0, \rvset_1)}=\var{Y_{0}(\rvset_0)+Y_{1}(\rvset_1)}=N_0^{-1}\var{Y_{0}(,\rvset_0)}+N_1^{-1}\var{Y_{1}(\rvset_1)}

where :math:`N_\alpha = |\rvset_\alpha|` and we used the fact

:math:`\rvset_0\bigcap\rvset_1=\emptyset` so :math:`\covar{Y_{0}(\rvset_0)}{Y_{0}(\rvset_1)}=0`

From the previous equation we can see that MLMC works well if the variance of the difference between models is smaller than the variance of either model. Although the variance of the low-fidelity model is likely high, we can set :math:`N_0` to be large because the cost of evaluating the low-fidelity model is low. If the variance of the discrepancy is smaller we only need a smaller number of samples so that the two sources of error (the two terms on the RHS of the previous equation) are balanced.

Note above and below we assume that :math:`f_0` is the high-fidelity model, but typical multi-level literature assumes :math:`f_M` is the high-fidelity model. We adopt the reverse ordering to be consistent with control variate estimator literature.

MLMC estimators can be thought of a specific case of an ACV estimator. When using two models this can be seen by

.. math::

   Q_{0}^\mathrm{ML}(\rvset_0,\rvset_1)&=Y_{0}(\rvset_0)+Y_{1}(\rvset_1)\\
   &=Q_{0}(\rvset_0)-Q_{1}(\rvset_0)+Q_{1}(\rvset_1)\\
   &=Q_{0}(\rvset_0)-(Q_{1}(\rvset_0)-Q_{1}(\rvset_1))

which has the same form as the two model ACV estimator presented in :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py` where the control variate weight has been set to :math:`-1`.

Many Model MLMC
---------------

MLMC can easily be extended to estimator based on :math:`M+1` models. Letting :math:`f_0,\ldots,f_M` be an ensemble of :math:`M+1` models ordered by decreasing fidelity and cost (note typically MLMC literature reverses this order),  we simply introduce estimates of the differences between the remaining models. That is

.. math ::
   :label: eq:ml-estimator

   Q_{0}^\mathrm{ML}(\rvset) = \sum_{\alpha=0}^M Y_{\alpha}(\rvset_\alpha), \quad f_{M+1}=0

To compute this estimator we use the following algorithm, starting with :math:`\alpha=M`

#. Draw  :math:`N_\alpha` samples randomly from the PDF  :math:`\pdf` of the random variables.
#. Estimate :math:`f_{\alpha}` and :math:`f_{\alpha}` at the samples :math:`\rvset_\alpha`
#. Compute the discrepancies :math:`Y_{\alpha,\rvset_\alpha}` at each sample.
#. Decrease :math:`\alpha` and repeat steps 1-3. until :math:`\alpha=0`.
#. Compute the ML estimator using :eq:`eq:ml-estimator`

The sampling strategy used for MLMC is shown in figure :ref:`mlmc-sample-allocation`. Here it is clear that we evaluate only the lowest fidelity model with :math:`\rvset_M` (this follows from the assumption that :math:`f_{M+1}=0`) and to to evaluate each discrepancy we must evaluate two consecutive models at the sets :math:`\rvset_\alpha`.

.. list-table::

   * -
       .. _mlmc-sample-allocation:

       .. figure:: ../../figures/mlmc.png
          :width: 50%
          :align: center

          MLMC sampling strategy

Because we use independent samples to estimate the expectation of each discrepancy, the variance of the estimator is

.. math:: \var{Q_{0}^\mathrm{ML}(\rvset_0, \ldots, \rvset_M)} = \sum_{\alpha=0}^M N_\alpha\var{Y_{\alpha}(\rvset_\alpha)}


Lets setup a problem to compute an MLMC estimate of :math:`\mean{f_0}`
using the following ensemble of models, taken from [PWGSIAM2016]_, which simulate a short column with a rectangular cross-sectional area subject to bending and axial force.

.. math::

   f_0(\rv)&=(1 - 4M/(b(h^2)Y) - (P/(bhY))^2)\\
   f_1(\rv)&=(1 - 3.8M/(b(h^2)Y) - ((P(1 + (M-2000)/4000))/(bhY))^2)\\
   f_2(\rv)&=(1 - M/(b(h^2)Y) - (P/(bhY))^2)\\
   f_3(\rv)&=(1 - M/(b(h^2)Y) - (P(1 + M)/(bhY))^2)\\
   f_4(\rv)&=(1 - M/(b(h^2)Y) - (P(1 + M)/(hY))^2)

where :math:`\rv = (b,h,P,M,Y)^T`
"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.visualization import mathrm_labels, mathrm_label
from pyapprox.benchmarks import setup_benchmark
from pyapprox.multifidelity.multioutput_monte_carlo import (
    get_estimator, numerically_compute_estimator_variance,
    compare_estimator_variances)
from pyapprox.multifidelity.visualize import (
    plot_estimator_variance_reductions, plot_estimator_variances,
    plot_estimator_sample_allocation_comparison)

np.random.seed(1)
benchmark = setup_benchmark("short_column_ensemble")
short_column_model = benchmark.fun
model_costs = np.asarray([100, 50, 5, 1, 0.2])

idx = [0, 1]
cov = short_column_model.get_covariance_matrix()[np.ix_(idx, idx)]
funs = [short_column_model.models[ii] for ii in idx]
costs = model_costs[idx]

# define the sample allocation
target_cost = 10000
est = get_estimator("mlmc", "mean", 1, costs, cov)
est.allocate_samples(target_cost)
samples_per_model = est.generate_samples_per_model(benchmark.variable.rvs)
values_per_model = [
    fun(samples) for fun, samples in zip(funs, samples_per_model)]
mlmc_mean = est(values_per_model)

sfmc_est = get_estimator("mc", "mean", 1, costs, cov)
sfmc_est.allocate_samples(target_cost)
samples_per_model = sfmc_est.generate_samples_per_model(benchmark.variable.rvs)
values_per_model = [
    fun(samples) for fun, samples in zip(funs, samples_per_model)]
sfmc_mean = sfmc_est(values_per_model[0])

# get the true mean of the high-fidelity model
true_mean = short_column_model.get_means()[0]
print('MC error', abs(sfmc_mean-true_mean))
print('MLMC error', abs(mlmc_mean-true_mean))

#%%
#These errors are comparable. However these errors are only for one realization of the samples sets. To obtain a clearer picture on the benefits of MLMC we need to look at the variance of the estimator.
#
#By viewing MLMC as a control variate we can derive its variance reduction [GGEJJCP2020]_
#
#.. math::  \gamma+1 = - \eta_1^2 \tau_{1}^2 - 2 \eta_1 \rho_{1} \tau_{1} - \eta_M^2 \frac{\tau_{M}^2}{r_{M}} - \sum_{i=2}^M \frac{1}{r_{i-1}}\left( \eta_i^2 \tau_{i}^2 + \alpha_{i-1}^2 \tau_{i-1}^2 - 2 \eta_i \eta_{i-1} \rho_{i,i-1} \tau_{i} \tau_{i-1} \right),
#   :label: mlmc-variance-reduction
#
#where  :math:`\tau_\alpha=\left(\frac{\var{Q_\alpha}}{\var{Q_0}}\right)^{\frac{1}{2}}`. Recall that and :math:`r_\alpha=\lvert\rvset_{\alpha}\rvert/N` is the ratio of the cardinality of the sets :math:`\rvset_{\alpha}` and :math:`\rvset_{0}`.
#
#The following code computes the variance reduction of the MLMC estimator, using the 2 models :math:`f_0,f_1`. The variance reduction is estimated numerically by  running MLMC repeatedly with different realizations of the sample sets and compared with the analytical estimator variance.
ntrials = int(1e3)
numerical_var, true_var, means = (
    numerically_compute_estimator_variance(
        funs, benchmark.variable, est, ntrials,
        return_all=True))[2:5]
print("Theoretical 2 model MLMC variance", true_var)
print("Achieved 2 model MLMC variance", numerical_var)

sfmc_means = (
    numerically_compute_estimator_variance(
        funs, benchmark.variable, sfmc_est, ntrials,
        return_all=True))[5]

fig, ax = plt.subplots()
ax.hist(sfmc_means, bins=ntrials//100, density=True, alpha=0.5,
        label=r'$Q_{0}(\mathcal{Z}_N)$')
ax.hist(means, bins=ntrials//100, density=True, alpha=0.5,
        label=r'$Q_{0}^\mathrm{CV}(\mathcal{Z}_N,\mathcal{Z}_{1})$')
ax.axvline(x=0, c='k', label=r'$E[Q_0]$')
_ = ax.legend(loc='upper left')

#%%
#The variance reduction obtained using three models is only slightly better than when using two models. The difference in variance reduction is dependent on the correlations between the models and the number of samples assigned to each model.
#
#MLMC works extremely well when the variance between the model discrepancies decays with increaseing fidelity. However one needs to be careful that the variance between discrepancies does indeed decay. Sometimes when the models correspond to different mesh discretizations. Increasing the mesh resolution does not always produce a smaller discrepancy. The following example shows that, for this example, adding a third model actually increases the variance of the MLMC estimator.

funs = [short_column_model.m0, short_column_model.m3, short_column_model.m4]
idx = [0, 3, 4]
cov = short_column_model.get_covariance_matrix()[np.ix_(idx, idx)]
costs = model_costs[idx]
est = get_estimator("mlmc", "mean", 1, costs, cov)
est.allocate_samples(target_cost)

sfmc_est = get_estimator("mc", "mean", 1, costs, cov)
sfmc_est.allocate_samples(target_cost)
print("Theoretical 3 model MLMC estimator variance for a pathalogical example",
      est._optimized_covariance)
print("Single-fidelity MC variance", sfmc_est._optimized_covariance)

#%%
#Using MLMC for this ensemble of models creates an estimate with a variance orders of magnitude larger than just using the high-fidelity model. When using models that do not admit a hierarchical structure, alternative approaches are needed. We will introduce such estimators in future tutorials.
#

#%%
#Optimal Sample Allocation
#-------------------------
#In the numerical example we were calling est.allocate_samples, however we have yet to define what this function does. This function allocates the optimal number of sample evaluations of each model that minimizes the variance of the MLMC estimator. When estimating the mean, this optimal allocation can be determined analytically. The following follows closely the exposition in [GAN2015]_ to derive the optimal allocation.
#
#Let :math:`C_\alpha` be the cost of evaluating the function :math:`f_\alpha` at a single sample, then the total cost of the MLMC estimator is
#
#.. math::
#
#   C_{\mathrm{tot}}=\sum_{\alpha=0}^M C_\alpha N_\alpha
#
#Now recall that the variance of the estimator is
#
#.. math:: \var{Q_0^\mathrm{ML}}=\sum_{\alpha=0}^M \var{Y_\alpha}N_\alpha,
#
#where :math:`Y_\alpha` is the disrepancy between two consecutive models, e.g. :math:`f_{\alpha-1}-f_\alpha` and :math:`N_\alpha` be the number of samples allocated to resolving the discrepancy, i.e. :math:`N_\alpha=\lvert\hat{\rvset}_\alpha\rvert`. Then For a fixed variance :math:`\epsilon^2` the cost of the MLMC estimator can be minimized, by solving
#
#.. math::
#
#  \min_{N_0,\ldots,N_M} & \sum_{\alpha=0}^M\left(N_\alpha C_\alpha\right)\\
#  \mathrm{subject}\; \mathrm{to} &\sum_{\alpha=0}^M\left(N_\alpha^{-1}\var{Y_\alpha}\right)=\epsilon^2
#
#or alternatively by introducing the lagrange multiplier :math:`\lambda^2` we can minimize
#
#.. math::
#
#   \mathcal{J}(N_0,\ldots,N_M,\lambda)&=\sum_{\alpha=0}^M\left(N_\alpha C_\alpha\right)+\lambda^2\left(\sum_{\alpha=0}^M\left(N_\alpha^{-1}\var{Y_\alpha}\right)-\epsilon^2\right)\\
#   &=\sum_{\alpha=0}^M\left(N_\alpha C_\alpha+\lambda^2N_\alpha^{-1}\var{Y_\alpha}\right)-\lambda^2\epsilon^2
#
#To find the minimum we set the gradient of this expression to zero:
#
#.. math::
#
#  \frac{\partial \mathcal{J}^\mathrm{ML}}{N_\alpha}&=C_\alpha-\lambda^2N_\alpha^{-2}\var{Y_\alpha}=0\\
#  \implies C_\alpha&=\lambda^2N_\alpha^{-2}\var{Y_\alpha}\\
#  \implies N_\alpha&=\lambda\sqrt{\var{Y_\alpha}C_\alpha^{-1}}
#
#The constraint is satisifed by noting
#
#.. math:: \frac{\partial \mathcal{J}}{\lambda^2}=\sum_{\alpha=0}^M N_\alpha^{-1}\var{Y_\alpha}-\epsilon^2=0
#
#Recalling that we can write the total variance as
#
#.. math::
#
#  \var{Q_{0,\rvset}^\mathrm{ML}}&=\sum_{\alpha=0}^M N_\alpha^{-1} \var{Y_\alpha}\\
#  &=\sum_{\alpha=0}^M \lambda^{-1}\var{Y_\alpha}^{-\frac{1}{2}}C_\alpha^{\frac{1}{2}}\var{Y_\alpha}\\
#  &=\lambda^{-1}\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}=\epsilon^2\\
#  \implies \lambda &= \epsilon^{-2}\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}
#
#Then substituting :math:`\lambda` into the following
#
#.. math::
#
#  N_\alpha C_\alpha&=\lambda\sqrt{\var{Y_\alpha}C_\alpha^{-1}}C_\alpha\\
#  &=\lambda\sqrt{\var{Y_\alpha}C_\alpha}\\
#  &=\epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)\sqrt{\var{Y_\alpha}C_\alpha}
#
#allows us to determine the smallest total cost that generates and estimator with the desired variance.
#
#.. math::
#
#  C_\mathrm{tot}&=\sum_{\alpha=0}^M N_\alpha C_\alpha\\
#  &=\sum_{\alpha=0}^M \epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)\sqrt{\var{Y_\alpha}C_\alpha}\\
#  &=\epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)^2
#
#To demonstrate the effectiveness of MLMC consider the model ensemble
#
#.. math:: f_\alpha(\rv)=\rv^{5-\alpha}, \quad \alpha=0,\ldots,4
#
#where each model is the function of a single uniform random variable defined on the unit interval :math:`[0,1]`.
#
#The following code computes the variance of the MLMC estimator for different target costs using the optimal sample allocation using an exact estimate of the covariance between models and an approximation.

np.random.seed(1)
benchmark = setup_benchmark("polynomial_ensemble")
poly_model = benchmark.fun
cov = poly_model.get_covariance_matrix()
target_costs = np.array([1e1, 1e2, 1e3], dtype=int)
costs = np.asarray([10**-ii for ii in range(cov.shape[0])])
model_labels = [r'$f_0$', r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$']
estimators = [
    get_estimator("mc", "mean", 1, costs, cov),
    get_estimator("mlmc", "mean", 1, costs, cov)]
est_labels = mathrm_labels(["MC", "MLMC"])
optimized_estimators = compare_estimator_variances(
    target_costs, estimators)

#%%
#The following compares the estimator variance of MC with MLMC for a fixed
#target cost

axs = [plt.subplots(1, 1, figsize=(8, 6))[1]]
# get MC and MLMC estmators for target cost = 100
ests_100 = [estlist[1:2] for estlist in optimized_estimators]
_ = plot_estimator_variance_reductions(
    ests_100, est_labels, axs[0])

#%%
#The following compares the estimator variance of MC with MLMC for a set of 
#target costs and plot the number of samples allocated to each model by MLMC

# get MLMC estimators for all target costs
mlmc_ests = optimized_estimators[1]
axs[0].set_xlim(target_costs.min(), target_costs.max())
fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
plot_estimator_sample_allocation_comparison(
    mlmc_ests, model_labels, axs[0])
_ = plot_estimator_variances(
    optimized_estimators, est_labels, axs[1],
    relative_id=0, cost_normalization=1)

#%%
#The left plot shows that the variance of the MLMC estimator is over and order of magnitude smaller than the variance of the single fidelity MC estimator for a fixed cost. The impact of using the approximate covariance is more significant for small samples sizes.
#
#The right plot depicts the percentage of the computational cost due to evaluating each model. The numbers in the bars represent the number of samples allocated to each model. Relative to the low fidelity models only a small number of samples are allocated to the high-fidelity model, however evaluating these samples represents approximately 50\% of the total cost.

#%%
#Multi-index Monte Carlo
#-----------------------
#Multi-Level Monte Carlo utilizes a sequence of models controlled by a single hyper-parameter, specifying the level of discretization for example, in a manner that balances computational cost with increasing accuracy. In many applications, however, multiple hyper-parameters may control the model discretization, such as the mesh and time step sizes. In these situations, it may not be clear how to construct a one-dimensional hierarchy represented by a scalar hyper-parameter. To overcome this limitation, a generalization of multi-level Monte Carlo, referred to as multi-index stochastic collocation (MIMC), was developed to deal with multivariate hierarchies with multiple refinement hyper-parameters [HNTNM2016]_.  PyApprox does not implement MIMC but a surrogate based version called Multi-index stochastic collocation (MISC) is presented in this tutorial :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multiindex_collocation.py`.


#%%
#Remark
#------
#MLMC was originally developed to estimate the mean of a function, but adaptations MMLC have since ben developed to estimate other statistics, e.g. [MD2019]_. PyApprox, however, does not implement these specific methods, because it implements a more flexible way to compute multiple statistics which we will describe in later tutorials.

#%%
#References
#^^^^^^^^^^
#.. [CGSTCVS2011] `K.A. Cliffe, M.B. Giles, R. Scheichl, A.L. Teckentrup, Multilevel Monte Carlo methods and applications to elliptic PDEs with random coefficients, Comput. Vis. Sci., 14, 3-15, 2011. <https://doi.org/10.1007/s00791-011-0160-x>`_
#
#.. [GOR2008] `M.B. Giles, Multilevel Monte Carlo path simulation, Oper. Res., 56(3), 607-617, 2008. <https://doi.org/10.1287/opre.1070.0496>`_
#
#.. [GAN2015] `M. Giles, Multilevel Monte Carlo methods, Acta Numerica, 24, 259-328, 2015. <https://doi.org/10.1017/S096249291500001X>`_
#
#.. [HNTNM2016] `A. Haji-Ali, F. Nobile and R. Tempone. Multi-index Monte Carlo: when sparsity meets sampling. Numerische Mathematik, 132(4), 767-806, 2016. <https://doi.org/10.1007/s00211-015-0734-5>`_
#
#.. [MD2019] `P. Mycek, M. De Lozzo. Multilevel Monte Carlo Covariance Estimation for the Computation of Sobol' Indices. 2019. <https://doi.org/10.1137/18M1216389>`_
