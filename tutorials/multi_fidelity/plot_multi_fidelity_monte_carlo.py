r"""
Multi-fidelity Monte Carlo
==========================
This tutorial builds on from :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_level_monte_carlo.py` and :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py` and introduces an approximate control variate estimator called Multi-fidelity Monte Carlo (MFMC) [PWGSIAM2016]_.

To derive the MFMC estimator first recall the two model ACV estimator

.. math:: Q_{0}^\mathrm{MF}(\rvset_\text{ACV})=Q_{0}(\rvset_{0}) + \eta\left(Q_1(\rvset_{1}^*)-Q_{1}(\rvset_{1})\right)

The MFMC estimator can be derived with the following recursive argument. Partition the samples assigned to each model such that
:math:`\rvset_\alpha=\rvset_{\alpha}^*\cup\rvset_{\alpha}` and :math:`\rvset_{\alpha}^*\cap\rvset_{\alpha}=\emptyset`. That is the samples at the next lowest fidelity model are the samples used at all previous levels plus an additional independent set, i.e. :math:`\rvset_{\alpha}^*=\rvset_{\alpha-1}`. 

Starting from two models we introduce the next low fidelity model in a way that reduces the variance of the estimate :math:`Q_{\alpha}(\rvset_\alpha)`, i.e.

.. math::

   Q_{0}^\mathrm{MF}(\rvset_\text{ACV})&=Q_{0}(\rvset_{0}) + \eta_1\left(Q_{1}(\rvset_{1}^*)-\left(Q_{1}(\rvset_{1})+\eta_2\left(Q_{2}(\rvset_2^*)-Q_{2}(\rvset_{2})\right)\right)\right)\\
   &=Q_{0}(\rvset_{0}) + \eta_1\left(Q_{1}(\rvset_{1}^*)-Q_{1}(\rvset_{1})\right)+\eta_1\eta_2\left(Q_{2}(\rvset_2^*)-Q_{2}(\rvset_{2})\right)

We repeat this process for all low fidelity models to obtain

.. math:: Q_{0}^\mathrm{MF}(\rvset_\text{ACV})=Q_{0}(\rvset_{0}) + \sum_{\alpha=1}^M\eta_\alpha\left(Q_{\alpha}(\rvset_{\alpha}^*)-Q_{\alpha}(\rvset_{\alpha})\right)

The allocation matrix for three models is

.. math::

   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

The optimal control variate weights and the covariance of the MFMC estimator can be computed with the general formulas presented in :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_many_model_approximate_control_variate_monte_carlo.py`.

Optimal Sample Allocation
-------------------------
Similarly to MLMC, the optimal number of samples that minimize the variance of a single scalar mean computed using MFMC can be determined analytically (see [PWGSIAM2016]_) provided the following condition is met

.. math:: \frac{C_{\alpha-1}}{C_\alpha} > \frac{\rho^2_{0,\alpha-1}-\rho^2_{0,\alpha}}{\rho^2_{0,\alpha}-\rho^2_{0,\alpha+1}}

When this condition is met the optimal number of high fidelity samples is

.. math:: N_0 = \frac{C_\mathrm{tot}}{\V{C}^T\V{r}}

where :math:`\V{C}=[C_0,\cdots,C_M]^T` are the costs of each model and :math:`\V{r}=[r_0,\ldots,r_M]^T` are the sample ratios defining the number of samples assigned to each model, i.e.

.. math:: N_\alpha=r_\alpha N_0.

The number in each partition can be determined from the number of samples per model using

.. math:: p_0 = N_0, \qquad p_\alpha=N_\alpha-N_{\alpha-1}, \;\alpha > 0.

Recalling that :math:`\rho_{j,k}` denotes the correlation between models :math:`j` and :math:`k`, the optimal sample ratios are

.. math:: r_\alpha=\left(\frac{C_0(\rho^2_{0,\alpha}-\rho^2_{0,\alpha+1})}{C_\alpha(1-\rho^2_{0,1})}\right)^{\frac{1}{2}}.

Now lets us compare MC with MFMC using optimal sample allocations
"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.visualization import mathrm_labels
from pyapprox.benchmarks import setup_benchmark
from pyapprox.multifidelity.multioutput_monte_carlo import (
    get_estimator, compare_estimator_variances)
from pyapprox.multifidelity.visualize import (
    plot_estimator_variance_reductions, plot_estimator_variances,
    plot_estimator_sample_allocation_comparison)

np.random.seed(1)
benchmark = setup_benchmark("polynomial_ensemble")
poly_model = benchmark.fun
cov = poly_model.get_covariance_matrix()
target_costs = np.array([1e1, 1e2, 1e3, 1e4], dtype=int)
costs = np.asarray([10**-ii for ii in range(cov.shape[0])])
model_labels = [r'$f_0$', r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$']

estimators = [
    get_estimator("mc", "mean", 1, costs, cov),
    get_estimator("mlmc", "mean", 1, costs, cov),
    get_estimator("mfmc", "mean", 1, costs, cov)]
est_labels = mathrm_labels(["MC", "MLMC", "MFMC"])
optimized_estimators = compare_estimator_variances(
    target_costs, estimators)

axs = [plt.subplots(1, 1, figsize=(8, 6))[1]]

# get estimators for target cost = 100
ests_100 = [optimized_estimators[1][1], optimized_estimators[2][1]]
_ = plot_estimator_variance_reductions(
    ests_100, est_labels[1:3], axs[0])

#%%
#For this problem there is little difference between the MFMC and MLMC estimators but this is not always the case.
#
#The following plots the sample allocation matrices of the MLMC and MFMC estimators
axs = plt.subplots(1, 2, figsize=(2*8, 6))[1]
print(ests_100)
ests_100[0].plot_allocation(axs[0], True)
axs[0].set_title(est_labels[1])
ests_100[1].plot_allocation(axs[1], True)
_ = axs[1].set_title(est_labels[2])

#%%
#References
#^^^^^^^^^^
#.. [PWGSIAM2016] `Peherstorfer, B., Willcox, K.,  Gunzburger, M., Optimal Model Management for Multifidelity Monte Carlo Estimation, 2016. <https://doi.org/10.1137/15M1046472>`_
