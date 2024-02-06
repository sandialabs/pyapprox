r"""
Approximate Control Variate Monte Carlo
==============================================================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variates.py`, :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_level_monte_carlo.py`, and :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_fidelity_monte_carlo.py`. MLMC and MFMC are two approaches which can utilize an ensemble of models of vary cost and accuracy to efficiently estimate the expectation of the highest fidelity model. In this tutorial we introduce a general framework for ACVMC when using two or more models to estimate vector-valued statistics.

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

The following tutorials introduce different ACV methods and their allocation matrices that have been introduced in the literature.

The following compares the estimator variances of three different ACV estimators. No one estimator type is best for all problems. Consequently for any given problem all possible estimators should be constructed. This requires estimates of the model covariance using a pilot study see :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_pilot_studies.py`
"""
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.visualization import mathrm_labels
from pyapprox.benchmarks import setup_benchmark
from pyapprox.multifidelity.factory import (
    get_estimator, compare_estimator_variances)
from pyapprox.multifidelity.visualize import (
    plot_estimator_variance_reductions)

np.random.seed(1)
benchmark = setup_benchmark("polynomial_ensemble")
model = benchmark.fun
cov = model.get_covariance_matrix()
nmodels = cov.shape[0]
target_costs = np.array([1e2], dtype=int)
costs = np.asarray([10**-ii for ii in range(nmodels)])
model_labels = [r'$f_0$', r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$']

estimators = [
    get_estimator("mlmc", "mean", 1, costs, cov),
    get_estimator("mfmc", "mean", 1, costs, cov),
    get_estimator("gmf", "mean", 1, costs, cov,
                  recursion_index=np.zeros(nmodels-1, dtype=int))]
est_labels = mathrm_labels(["MLMC", "MFMC", "ACVMF"])
optimized_estimators = compare_estimator_variances(
    target_costs, estimators)

axs = [plt.subplots(1, 1, figsize=(8, 6))[1]]

# get estimators for target cost = 100
ests_100 = [ests[0] for ests in optimized_estimators]
_ = plot_estimator_variance_reductions(
    ests_100, est_labels, axs[0])
