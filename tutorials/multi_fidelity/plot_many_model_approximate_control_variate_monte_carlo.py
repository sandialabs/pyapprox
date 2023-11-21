r"""
Parametrically Defined Approximate Control Variate Monte Carlo
==============================================================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py`, :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_level_monte_carlo.py`, and :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_fidelity_monte_carlo.py`. MLMC and MFMC are two approaches which can utilize an ensemble of models of vary cost and accuracy to efficiently estimate the expectation of the highest fidelity model. In this tutorial we introduce a general framework for ACVMC when using two or more models to estimate vector-valued statistics.

The approximate control variate estimator of a vector-valued statistic :math:`\mat{Q}_0\in\reals^{S}` that uses  :math:`M` lower fidelity models is

.. math::

  \mat{Q}_{\text{ACV}}(\rvset_0,\rvset_1^*,\rvset_1,\ldots,\rvset^*_M, \rvset_M) &= \mat{Q}_{0}(\rvset_0)+\begin{bmatrix}\eta_{1,1} & \cdots &\eta_{1, SM}\\
  \eta_{2,1} & \cdots &\eta_{2, SM}\\
  \vdots\\
  \eta_{S,1} & \cdots &\eta_{S, SM}
  \end{bmatrix}\begin{bmatrix} \mat{Q}_{1}(\rvset_1^*)-\mat{Q}_{1}(\rvset_1) \\ \mat{Q}_{2}(\rvset_2^*)-\mat{Q}_{2}(\rvset_2) \\ \vdots \\ \mat{Q}_{M}(\rvset_M^*)-\mat{Q}_{M}(\rvset_M)\end{bmatrix}\in\reals^{S}

or in more compact notation

.. math:: \mat{Q}_{\text{ACV}}(\rvset_\text{ACV})&=\mat{Q}_{0}(\rvset_0)+\mat{\eta}\mat{\Delta}(\rvset_{\Delta}), \quad \mat{\Delta}(\rvset_\Delta) = \begin{bmatrix}\mat{\Delta}_1(\rvset_1^*, \rvset_1)\\ \vdots\\ \mat{\Delta}_M(\rvset_M^*,\rvset_M)\end{bmatrix}\in\reals^{SM}, \quad \mat{\eta}\in\reals^{S\times SM},
where the entries of :math:`\mat{\eta}` are called control variate weights, :math:`\rvset_\Delta=\{\rvset_1^*, \rvset_1, \ldots, \rvset_M^*, \rvset_M\}`, and :math:`\rvset_\text{ACV}=\{\rvset_0, \rvset_\Delta\}`.

Here :math:`\mat{\eta}=[\eta_1,\ldots,\eta_M]^T`, :math:`\mat{\Delta}=[\Delta_1,\ldots,\Delta_M]^T`, and :math:`\rvset_{\alpha}^*`, :math:`\rvset_{\alpha}` are sample sets that may or may not be disjoint. Specifying the exact nature of these sets, including their cardinality, can be used to design different ACV estimators which will discuss later.

This estimator is constructed by evaluating each model at two sets of samples :math:`\rvset_{\alpha}^*=\{\rv^{(n)}\}_{n=1}^{N_{\alpha^*}}` and :math:`\rvset_{\alpha}=\{\rv^{(n)}\}_{n=1}^{N_\alpha}` where some samples may be shared between sets such that in some cases :math:`\rvset_{\alpha}^*\cup\rvset_{\beta}\neq \emptyset`.

For any :math:`\mat{\eta}(\rvset_\text{ACV})`, the covariance of the ACV estimator is

.. math::

   \var{\mat{Q}^{\text{ACV}}} = \var{\mat{Q}_{0}}+\mat{\eta}\covar{\mat{\Delta}}{\mat{\Delta}}\mat{\eta}^\top+\mat{\eta}\covar{\mat{\Delta}}{\mat{Q}_0}+\covar{\mat{\Delta}}{\mat{Q}_0}\mat{\eta}^\top, \qquad\in\reals^{S\times S}

The control variate weights that minimize the determinant of the ACV estimator covariance are

.. math:: \mat{\eta} = -\covar{\mat{\Delta}}{\mat{\Delta}}^{-1}\covar{\mat{\Delta}}{\mat{Q}_0}

The ACV estimator covariance using the optimal control variate weights is

.. math:: \covar{\mat{Q}_{\text{ACV}}}{\mat{Q}_{\text{ACV}}}(\rvset_\text{ACV})=\covar{\mat{Q}_0}{ \mat{Q}_0}-\covar{\mat{Q}_0}{\mat{\Delta}}\covar{\mat{\Delta}}{\mat{\Delta}}^{-1}\covar{\mat{Q}_0}{\mat{\Delta}}^\top

Computing the ACV estimator covariance requires computing :math:`\covar{\mat{Q}_0}{\mat{\Delta}}\text{ and} \covar{\mat{\Delta}}{\mat{\Delta}}`.

First

.. math::

  \covar{\mat{\Delta}}{\mat{\Delta}} =
  \begin{bmatrix}\covar{\mat{\Delta}_1}{\mat{\Delta}_1} & \covar{\mat{\Delta}_1}{\mat{\Delta}_2} & \cdots & \covar{\mat{\Delta}_1}{\mat{\Delta}_M}\\
  \covar{\mat{\Delta}_2}{\mat{\Delta}_1} & \covar{\mat{\Delta}_1}{\mat{\Delta}_2} &  & \vdots\\
  \vdots & & \ddots & \vdots \\
  \covar{\mat{\Delta}_M}{\mat{\Delta}_1} & \cdots & \cdots & \covar{\mat{\Delta}_M}{\mat{\Delta}_M}
  \end{bmatrix}

and second

.. math::     \covar{\mat{Q}_0}{\mat{\Delta}} = [\covar{\mat{Q}_0}{\mat{\Delta}_1}, \ldots, \covar{\mat{Q}_0}{\mat{\Delta}_M}]


where

.. math::     \covar{\mat{Q}_0}{\mat{\Delta}_\alpha} = \covar{\mat{Q}_0(\rvset_0)}{\mat{\Delta}_\alpha(\rvset_\alpha^*)}-\covar{\mat{Q}_0(\rvset_0)}{\mat{\Delta}_\alpha(\rvset_\alpha)}]

When estimating the statistics in :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multioutput_monte_carlo.py`  the above coavariances involving :math:`\Delta` can be computed using the formula in :ref:`sphx_glr_auto_tutorials_multi_fidelity_acv_covariances.py`.

The variance of an ACV estimator is dependent on how samples are allocated to the sets :math:`\rvset_\alpha,\rvset_\alpha^*`, which we call the sample allocation :math:`\mathcal{A}`. Specifically, :math:`\mathcal{A}` specifies: the number of samples in the sets :math:`\rvset_\alpha,\rvset_\alpha^*, \forall \alpha`, denoted by :math:`N_\alpha` and :math:`N_{\alpha^*}`, respectively; the number of samples in the intersections of pairs of sets, that is :math:`N_{\alpha\cap\beta} =|\rvset_\alpha \cap \rvset_\beta|`, :math:`N_{\alpha^*\cap\beta} =|\rvset_\alpha^* \cap \rvset_\beta|`, :math:`N_{\alpha^*\cap\beta^*} =|\rvset_\alpha^* \cap \rvset_\beta^*|`; and the number of samples in the union of pairs of sets :math:`N_{\alpha\cup\beta} = |\rvset_\alpha\cup \rvset_\beta|` and similarly :math:`N_{\alpha^*\cup\beta}`, :math:`N_{\alpha^*\cup\beta^*}`. Thus, finding the best ACV estimator can be theoretically be found by solving the constrained non-linear optimization problem

.. math:: \min_{\mathcal{A}\in\mathbb{A}}\mathrm{Det}\left[\covar{\mat{Q}_{\text{ACV}}}{\mat{Q}_{\text{ACV}}}\right](\mathcal{A}) \qquad \mathrm{s.t.} \qquad C(\mat{c},\tilde{\mathcal{A}})\le C_\mathrm{max},

Here, :math:`\tilde{\mathcal{A}}` is the set of all possible sample allocations, and given the computational costs of evaluating each model once :math:`c^\top=[c_0, c_1, \ldots, c_M]`, the constraint ensures that the computational cost of computing the ACV estimator

.. math:: C(\mat{c},\mathcal{A})=\sum_{\alpha=0}^M N_{\alpha^*\cup\alpha}c_\alpha

is smaller than a user-specified computational budget :math:`C_\mathrm{max}`.

Unfortunately, to date, no method has been devised to solve the above optimization problem for all possible allocations :math:`\tilde{\mathcal{A}}`. Consequently, all existing ACV methods restrict the optimization space to :math:`\mathcal{A}\subset\tilde{\mathcal{A}}`. These restricted search spaces are formulated in terms of a set of :math:`M+1` independent sample partitions :math:`\mathcal{P}_m` that contain :math:`p_m` samples drawn indepedently from the PDF of the random variable :math:`\rv`. Each subset :math:`\rvset_\alpha` is then assumed to consist of a combinations of these indepedent partitions. ACV methods encode the relationship between the sets :math:`\rvset_\alpha` and :math:`\mathcal{P}_m` via allocation matrices :math:`\mat{A}`. For example, the allocation matrix used by MFMC, which is an ACV method, when applied to three models is given by

.. math::

  \mat{A} = \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

An entry of one indicates in the ith row of the jth column indicates that the ith independent partition :math:`\mathcal{P}_i` is used in the corresponding set :math:`\rvset_j` if j is odd or :math:`\rvset_j^*` if j is even. The first column will always only contain zeros because the set :math:`\rvset_0^*` is never used by ACV estimators.

Once an allocation matrix is specified, the optimal sample allocation can be obtained by optimizing the number of samples in each partition :math:`\mat{p}=[p_0,\ldots,p_M]^\top`, that is

.. math:: \min_{\mat{p}}\mathrm{Det}\left[\covar{\mat{Q}_{\text{ACV}}}{\mat{Q}_{\text{ACV}}}\right](\mat{p}; \mat{A}) \qquad \mathrm{s.t.} \qquad C(\mat{c},\mat{p};\mat{A})\le C_\mathrm{max},

A detailed discussion of how the allocation matrix can be used to parameterize the allocation :math:`\mathcal{A}` can be found in :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_pacv.py`

The following tutorials introduce different ACV methods and their allocation matrices that have been introduced in the literature.

Generalized Multifidelity Sampling (GMF)
----------------------------------------
"""

#%%
#Lets apply ACV to the tunable model ensemble
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from pyapprox.benchmarks import setup_benchmark
from pyapprox.multifidelity.multioutput_monte_carlo import (
    get_estimator, numerically_compute_estimator_variance)

np.random.seed(2)
shifts = [.1, .2]
benchmark = setup_benchmark(
    "tunable_model_ensemble", theta1=np.pi/2*.95, shifts=shifts)
model = benchmark.fun
funs = model.models
model_costs = 10.**(-np.arange(3))
cov = model.get_covariance_matrix()

#%%
# First let us just use 2 models

print('Two models')
ntrials = 1000
target_cost = 100
est = get_estimator(
    "gis", "mean", 1, model.costs()[:2], benchmark.model_covariance[:2, :2])

est.allocate_samples(target_cost)
numerical_var, true_var, means = (
    numerically_compute_estimator_variance(
        funs[:2], benchmark.variable, est, ntrials,
        return_all=True))[2:5]

print("Theoretical ACV variance", true_var)
print("Achieved ACV variance", numerical_var)

#%%
# Now let us use 3 models
ntrials = 1000
target_cost = 100
est = get_estimator(
    "gis", "mean", 1, model.costs()[:3], benchmark.model_covariance[:3, :3])
est.allocate_samples(target_cost)

ax = plt.subplots(1, 1, figsize=(8, 6))[1]
est.plot_allocation(ax, True)
plt.show()

numerical_var, true_var, means = (
    numerically_compute_estimator_variance(
        funs[:3], benchmark.variable, est, ntrials,
        return_all=True))[2:5]

print('Three models')
print("Theoretical ACV variance reduction", true_var)
print("Achieved ACV variance reduction", numerical_var)

#%%
#The benefit of using three models over two models depends on the correlation between each low fidelity model and the high-fidelity model. The benefit on using more models also depends on the relative cost of evaluating each model, however here we will just investigate the effect of changing correlation. The following code shows the variance reduction (relative to standard Monte Carlo) obtained using CVMC (not approximate CVMC) using 2 (OCV1) and three models (OCV2). Unlike MLMC and MFMC, ACV-IS will achieve these variance reductions in the limit as the number of samples of the low fidelity models goes to infinity.

from pyapprox.multifidelity.control_variate_monte_carlo import (
    get_control_variate_rsquared
)
theta1 = np.linspace(model.theta2*1.05, model.theta0*0.95, 5)
covs = []
var_reds = []
for th1 in theta1:
    model.theta1 = th1
    covs.append(model.get_covariance_matrix())
    OCV2_var_red = 1-get_control_variate_rsquared(covs[-1])
    # use model with largest covariance with high fidelity model
    idx = [0, np.argmax(covs[-1][0, 1:])+1]
    assert idx == [0, 1] #it will always be the first model
    OCV1_var_red = get_control_variate_rsquared(covs[-1][np.ix_(idx, idx)])
    var_reds.append([OCV2_var_red, OCV1_var_red])
covs = np.array(covs)
var_reds = np.array(var_reds)

fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
for ii, jj in [[0, 1], [0, 2], [1, 2]]:
    axs[0].plot(theta1, covs[:, ii, jj], 'o-',
                label=r'$\rho_{%d%d}$' % (ii, jj))
axs[1].plot(theta1, var_reds[:, 0], 'o-', label=r'$\mathrm{OCV2}$')
axs[1].plot(theta1, var_reds[:, 1], 'o-', label=r'$\mathrm{OCV1}$')
axs[1].plot(theta1, var_reds[:, 0]/var_reds[:, 1], 'o-',
            label=r'$\mathrm{OCV2/OCV1}$')
axs[0].set_xlabel(r'$\theta_1$')
axs[0].set_ylabel(r'$\mathrm{Correlation}$')
axs[1].set_xlabel(r'$\theta_1$')
axs[1].set_ylabel(r'$\mathrm{Variance\;reduction\;ratio} \; \gamma$')
axs[0].legend()
_ = axs[1].legend()

#%%
#The variance reduction clearly depends on the correlation between all the models.
#
#Let us now compare the variance reduction obtained by MLMC, MFMC and ACV with the MF sampling scheme as we increase the number of samples assigned to the low-fidelity models, while keeping the number of high-fidelity samples fixed. Here we will use the model ensemble
#
#.. math:: f_\alpha(\rv)=\rv^{5-\alpha}, \quad \alpha=0,\ldots,4
#
#where each model is the function of a single uniform random variable defined on the unit interval :math:`[0,1]`.

from pyapprox.util.visualization import mathrm_labels, mathrm_label
plt.figure()
benchmark = setup_benchmark("polynomial_ensemble")
poly_model = benchmark.fun
cov = poly_model.get_covariance_matrix()
model_costs = np.asarray([10**-ii for ii in range(cov.shape[0])])
nhf_samples = 10
nsample_ratios_base = np.array([2, 4, 8, 16])
cv_labels = mathrm_labels(["OCV-1", "OCV-2", "OCV-4"])
cv_rsquared_funcs = [
    lambda cov: get_control_variate_rsquared(cov[:2, :2]),
    lambda cov: get_control_variate_rsquared(cov[:3, :3]),
    lambda cov: get_control_variate_rsquared(cov)]
cv_gammas = [1-f(cov) for f in cv_rsquared_funcs]
for ii in range(len(cv_gammas)):
    plt.axhline(y=cv_gammas[ii], linestyle='--', c='k')
    xloc = -.35
    plt.text(xloc, cv_gammas[ii]*1.1, cv_labels[ii], fontsize=16)
plt.axhline(y=1, linestyle='--', c='k')
plt.text(xloc, 1, r'$\mathrm{MC}$', fontsize=16)

est_labels = mathrm_labels(["MLMC", "MFMC", "ACVMF"])
estimators = [
    multifidelity.get_estimator("mlmc", cov, model_costs, poly_model.variable),
    multifidelity.get_estimator("mfmc", cov, model_costs, poly_model.variable),
    multifidelity.get_estimator("acvmf", cov, model_costs, poly_model.variable)
]
acv_rsquared_funcs = [est._get_rsquared for est in estimators]

nplot_points = 20
acv_gammas = np.empty((nplot_points, len(acv_rsquared_funcs)))
for ii in range(nplot_points):
    nsample_ratios = np.array([r*(2**ii) for r in nsample_ratios_base])
    acv_gammas[ii, :] = [1-f(cov, nsample_ratios) for f in acv_rsquared_funcs]
for ii in range(len(est_labels)):
    plt.semilogy(np.arange(nplot_points), acv_gammas[:, ii],
                 label=est_labels[ii])
plt.legend()
plt.xlabel(r'$\log_2(r_i)-i$')
_ = plt.ylabel(mathrm_label("Variance reduction ratio ")+r"$\gamma$")

#%%
#As the theory suggests MLMC and MFMC use multiple models to increase the speed to which we converge to the optimal 2 model CV estimator OCV-2. These two approaches reduce the variance of the estimator more quickly than the ACV estimator, but cannot obtain the optimal variance reduction.

#%%
#Accelerated Approximate Control Variate Monte Carlo
#---------------------------------------------------
#The recursive estimators work well when the number of low-fidelity samples are smal but ACV can achieve a greater variance reduction for a fixed number of high-fidelity samples. In this section we present an approach called ACV-GMFB that combines the strengths of these methods [BLWLJCP2022]_.
#
#This estimator differs from the previous recursive estimators because it uses some models as control variates and other models to estimate the mean of these control variates recursively. This estimator optimizes over the best use of models and returns the best model configuration.
#
#Let us add the ACV-GMFB estimator to the previous plot


plt.figure()
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

from pyapprox.multifidelity.monte_carlo_estimators import (
    get_acv_recursion_indices
)
est_labels = mathrm_labels(["MLMC", "MFMC", "ACVMF", "ACVGMFB"])
estimator_types = ["mlmc", "mfmc", "acvmf", "acvgmfb"]
estimators = [
    multifidelity.get_estimator(t, cov, model_costs, poly_model.variable)
    for t in estimator_types]
# acvgmfb requires nhf_samples so create wrappers of that does not
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
#The variance of the best ACV-GMFB still converges to the lowest possible variance. But its variance at small sample sizes is better than ACV-MF  and comparable to MLMC.
#

#%%
#Generalized Recursive Difference Sampling (GRD)
#-----------------------------------------------

#%%
#Generalized Independent Sampling (GIS)
#--------------------------------------


#%%
#Optimal Sample Allocation
#-------------------------
#
#The previous results compared MLMC with MFMC and ACV-MF when the number of high-fidelity samples were fixed. In the following we compare these methods when the number of samples are optimized to minimize the variance of each estimator. We will only use the first 4 models
estimator_types = ["mc", "mlmc", "mfmc", "acvmf", "acvgmfb"]
estimators = [
    multifidelity.get_estimator(t, cov[:4, :4], model_costs[:4], poly_model.variable)
    for t in estimator_types]
est_labels = mathrm_labels(["MC", "MLMC", "MFMC", "ACVMF", "ACVGMFB"])
target_costs = np.array([1e1, 1e2, 1e3, 1e4], dtype=int)
optimized_estimators = multifidelity.compare_estimator_variances(
    target_costs, estimators)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
multifidelity.plot_estimator_variances(
    optimized_estimators, est_labels, ax,
    ylabel=mathrm_label("Relative Estimator Variance"))
_ = ax.set_xlim(target_costs.min(), target_costs.max())

#fig # necessary for jupyter notebook to reshow plot in new cell

#%%
#In this example ACVGMFB is the most efficient estimator, i.e. it has a smaller variance for a fixed cost. However this improvement is problem dependent. For other model ensembles another estimator may be more efficient. Modify the above example to use another model to explore this more. The left plot shows the relative costs of evaluating each model using the ACVMF sampling strategy. Compare this to the MLMC sample allocation. Also edit above code to plot the MFMC sample allocation.

#%%
#Before this tutorial ends it is worth noting that a section of the MLMC literature explores adaptive methods which do not assume there is a fixed high-fidelity model but rather attempt to balance the estimator variance with the deterministic bias. These methods add a higher-fidelity model, e.g. a finer finite element mesh, when the variance is made smaller than the bias. We will not explore this here, but an example of this is shown in the tutorial on multi-index collocation.

#%%
#References
#^^^^^^^^^^
#.. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification, Journal of Computational Physics, 408:109257, 2020. <https://doi.org/10.1016/j.jcp.2020.109257>`_
#
#.. [BLWLJCP2022] `On the optimization of approximate control variates with parametrically defined estimators, Journal of Computational Physics,451:110882, 2022 <https://doi.org/10.1016/j.jcp.2021.110882>`_
