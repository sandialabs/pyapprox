r"""
Multilevel Best Linear Unbiased estimators (MLBLUE)
===================================================
This tutorial introduces Multilevel Best Linear Unbiased estimators (MLBLUE) and compares its characteristics and performance with the previously introduced
multi-fidelity estimators.

.. list-table::

   * - .. _MLBLUE-sample-allocation:

       .. figure:: ../../figures/MLBLUE-sets.png
          :width: 50%
          :align: center

          MLBLUE sample allocations.

MLBLUE assumes that each model :math:`f_1,\ldots,f_M` is linearly dependent on the means of each model :math:`q=[Q_1,\ldots,Q_M]^\top`, where :math:`M` is the number of low-fidelity models. Specifically, MLBLUE assumes that

.. math:: Y=Hq+\epsilon,

where :math:`Y=[Y_1,\ldots,Y_{K}]` are evaluations of models within all :math:`K=2^{M}-1` nonempty subset :math:`S^k\in2^{\{1,\ldots,M\}}\setminus\emptyset` , that is :math:`Y_k=[f_{S_k, 1}^{(1)},\ldots,f_{S_k, |S_k|}^{(1)},\ldots,f_{S_k, 1}^{(m_k)},\ldots,f_{S_k, |S_k|}^{(m_k)}]`. An example of such sets is shown in the figure above where :math:`m_k` denotes the number of samples of each model in the subset :math:`S_k`. 

Here :math:`H` is a restriction opeator that specifies the means that produce the data in each set, that is

.. math:: H=[R_1^\top,\ldots,R_K^\top]^\top, \qquad R_kq = [Q_{S_k,1},\ldots,Q_{S_k,|S_k|}]^\top

MLBLUE then finds the means of all models bys solving the generalized least squares problem

.. math:: \min_{q\in\reals^M} \lVert Y-Hq\rVert^2_{{\covar{\epsilon}{\epsilon}}^{-1}}

Here ${\covar{\epsilon}{\epsilon}$ is a matrix that is dependent on the covariance between the models and is given by

.. math:: \covar{\epsilon}{\epsilon}&=\mathrm{BlockDiag}(G_1,\ldots, G_K),\quad G_k = \mathrm{BlockDiag}((C_k)_{i=1}^{m_k}), \\ C_k &= \covar{Y-H_kq}{Y-H_kq}.

The figure below gives an example of the construction of the least squares system when only three subsets are active, :math:`S_2, S_3, S_5`; one, one and two samples of the models in each subset are taken, respectively. :math:`S_2` only contributes on equation because it consists one model that is only sampled once. :math:`S_3` contributes two equations, because it consists of two models sampled once each. Finally, :math:`S_5` contributes four equations because it consists of two models sampled twice.

.. list-table::

   * - .. _MLBLUE-sample-example:

       .. figure:: ../../figures/MLBLUE_sample_example.png
          :width: 100%
          :align: center

          Example of subset data construction.


The figure below depicts the structure of the :math:`\covar{\epsilon}{\epsilon}` for the same example, where :math:`\sigma_{ij}^2` denotes the covariance between the models :math:`f_i,f_j` and must be computed from a pilot study.

.. list-table::

   * - .. _MLBLUE-covariance-example:

       .. figure:: ../../figures/MLBLUE_covariance_example.png
          :width: 100%
          :align: center

          Example of the covariance structure.

The solution to the generalized least-squares problem can be found by solving the sustem of linear equations

.. math:: \Psi q^B=y,

where q^B denotes the MLBLUE estimate of the model means :math:`q` and

.. math:: \Psi = \sum_{k=1}^K m_k R_k^\top C_k^{-1} R_k, \qquad y = \sum_{k=1}^K R^\top_K C_k^{-1} \sum_{i=1}^{m_k} Y_{k}^{(i)}

The vector :math:`q^B` is an estimate of all model means, however one is often only interested in a linear combination of means, i.e.

.. math:: q^B_\beta = \beta^\top q^B.

For example, :math:`\beta=[1, 0, \ldots, 0]^\top` can be used to estimate only the high-fidelity mean.

Given $\beta$ the variance of the MLBLU estimator is

.. math:: \var{Q^B_\beta}=\beta^\top\Psi^{-1}\beta

The following code compares MLBLUE to other multif-fidelity esimators when the number of high-fidelity samples is fixed and the number of low-fidelity samples increases. This example shows that unlike MLMC and MFMC, MLBLUE like ACV obtains the optimal variance reduction obtained by control variate MC, using known means,  as the number of low-fidelity samples becomes very large.

First setup the polynomial benchmark
"""
import numpy as np
from functools import partial
from pyapprox.util.configure_plots import plt, mathrm_labels, mathrm_label
from pyapprox import multifidelity
from pyapprox.multifidelity.control_variate_monte_carlo import (
    get_control_variate_rsquared)
from pyapprox.benchmarks import setup_benchmark

plt.figure()
benchmark = setup_benchmark("polynomial_ensemble")
poly_model = benchmark.fun
cov = poly_model.get_covariance_matrix()
model_costs = np.asarray([10**-ii for ii in range(cov.shape[0])])
nhf_samples = 10
nsample_ratios_base = np.array([2, 4, 8, 16])

#%%
#First, plot the variance reduction of the optimal control variates using known low-fidelity means.
#
#Second, plot the variance reduction of multi-fidelity estimators that do not assume known low-fidelity means. The code below repeatedly doubles the number of low-fidelity samples according to the initial allocation defined by nsample_ratios_base=[2,4,8,16].

# plot optimal control variate variance reduction
cv_labels = mathrm_labels(["OCV-1", "OCV-2", "OCV-4"])
cv_rsquared_funcs = [
    lambda cov: get_control_variate_rsquared(cov[:2, :2]),
    lambda cov: get_control_variate_rsquared(cov[:3, :3]),
    lambda cov: get_control_variate_rsquared(cov)]
cv_gammas = [1-f(cov) for f in cv_rsquared_funcs]
xloc = -.35
for ii in range(len(cv_gammas)):
    plt.axhline(y=cv_gammas[ii], linestyle='--', c='k')
    plt.text(xloc, cv_gammas[ii]*1.1, cv_labels[ii], fontsize=16)
plt.axhline(y=1, linestyle='--', c='k')
plt.text(xloc, 1, mathrm_label("MC"), fontsize=16)

# plot multi-fidelity estimator variance reduction
est_labels = mathrm_labels(["MLMC", "MFMC", "ACVMF", "ACVGMFB", "MLBLUE"])
estimator_types = ["mlmc", "mfmc", "acvmf", "acvgmfb", "mlblue"]
estimators = [
    multifidelity.get_estimator(t, cov, model_costs, poly_model.variable)
    for t in estimator_types]
# acvgmfb and mlblue require nhf_samples so create wrappers of that does not
estimators[-2]._get_rsquared = partial(
    estimators[-2]._get_rsquared_from_nhf_samples, nhf_samples)
estimators[-1]._get_rsquared = partial(
    estimators[-1]._get_rsquared_from_nhf_samples, nhf_samples)
nplot_points = 20
acv_gammas = np.empty((nplot_points, len(estimators)))
for ii in range(nplot_points):
    nsample_ratios = np.array([r*(2**ii) for r in nsample_ratios_base])
    acv_gammas[ii, :] = [1-est._get_rsquared(cov, nsample_ratios)
                         for est in estimators]
for ii in range(len(est_labels)):
    plt.semilogy(np.arange(nplot_points), acv_gammas[:, ii],
                 label=est_labels[ii])
plt.legend()
plt.xlabel(r'$\log_2(r_i)-i$')
_ = plt.ylabel(mathrm_label('Variance reduction ratio ')+ r'$\gamma$')

#%%
#Optimal sample allocation
#-------------------------
#In the following we show how to optimize the sample allocation of MLBLUE and compare the optimized estimator to other alternatives.
#
#The optimal sample allocation is obtained by solving
#
#.. math:: \min_{m\in\mathbb{N}_0^K}⁡ \var{Q^B_\beta(m)} \quad \text{such that}\quad  \sum_{k=1}^K m_k \sum_{j=1}^{|S_k|} W_j \le W_\max,
#
#where \(W_j\) denotes the cost of evaluating the jth model and \(W_\max\) is the total budget.
#
# We will again use the polynomail model and to be consistent with previous tutoriuals, we will only use the first 4 models
cov = cov[:4, :4]
model_costs = model_costs[:4]

# Add MC estimator to comparison
estimator_types = ["mc", "mlmc", "mfmc", "acvmf", "acvgmfb", "mlblue"]
estimators = [
    multifidelity.get_estimator(t, cov, model_costs, benchmark.variable)
    for t in estimator_types]
est_labels = mathrm_labels(
    ["MC", "MLMC", "MFMC", "ACVMF", "ACVGMFB", "MLBLUE"])

target_costs = np.array([1e1, 1e2, 1e3, 1e4], dtype=int)
optimized_estimators = multifidelity.compare_estimator_variances(
    target_costs, estimators)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
multifidelity.plot_estimator_variances(
    optimized_estimators, est_labels, ax,
    ylabel=mathrm_label("Relative Estimator Variance"))
ax.set_xlim(target_costs.min(), target_costs.max())

ocv_variances = np.empty_like(target_costs, dtype=float)
for ii, target_cost in enumerate(target_costs):
    # ocv uses one sample for each model (but does not need additional
    # values for estimating low fidelity means
    nocv_samples_per_model = int(target_cost//model_costs.sum())
    ocv_variances[ii] = cov[0, 0]/nocv_samples_per_model*(
        1-get_control_variate_rsquared(cov))
rel_ocv_variances = (
    ocv_variances/optimized_estimators[0][0].optimized_variance)
ax.loglog(target_costs, rel_ocv_variances, ':o',
          label=mathrm_label("OCV"))
ax.legend()

#%%
#Now plot the number of samples allocated for each target cost
model_labels = [r"$M_{0}$".format(ii) for ii in range(cov.shape[0])]
multifidelity.plot_acv_sample_allocation_comparison(
    optimized_estimators[-1], model_labels, plt.figure().gca())
plt.show()


#%%
#References
#^^^^^^^^^^
#.. [SUSIAMUQ2020] `D Schaden, E Ullmann. On multilevel best linear unbiased estimators, SIAM/ASA J. Uncertainty Quantification 8 (2), 601 - 635, 2020. <https://doi.org/10.1137/19M1263534>`_

