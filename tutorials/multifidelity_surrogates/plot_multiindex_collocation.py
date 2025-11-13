r"""
Multi-level and Multi-index Collocation
=======================================
Thsi tutorial introduces multi-level  [TJWGSIAMUQ2015]_ and multi-index collocation [HNTTCMAME2016]_. It assumes knowledge of the material covered in :ref:`sphx_glr_auto_tutorials_surrogates_plot_tensor_product_interpolation.py` and :ref:`sphx_glr_auto_tutorials_surrogates_plot_sparse_grids.py`.

Models often utilize numerical discretizations to solve the equations governing the system dynamics. For example, finite-elements, spectral collocation, etc are often used to solve partial differential equations. :ref:`sphx_glr_auto_examples_plot_pde_convergence.py` demonstrates how the numerical discretization, specifically the spatial mesh discretization and time-step, of a spectral collocation model of transient advection diffusion effects the accuracy of the model output.

.. list-table::

   * - .. _multilevel_hierarchy:

       .. figure:: ../../figures/multilevel-hierarchy.png
          :width: 50%
          :align: center

          A multi-level hierarchy formed by increasing mesh discretizations.

An observation
--------------
Multilevel collocation was introduced to reduce the cost of building surrogates of models when a one-dimensional hierarchy of numerical discretizations of a model  :math:`f_\alpha(\rv), \alpha=0,1,\ldots` are available such that

.. math:: \lVert f-f_\alpha\rVert \le \lVert f-f_{\alpha^\prime}\rVert

if :math:`\alpha^\prime < \alpha.` and the work :math:`W_\alpha` increases with fidelity.

Multilevel collocation can be implemented by modifying sparse grid interplation developed for a single model fidelity. The modification is based on the observation that the discrepancy between two consecutive models and the lower-fidelity model will be computationally cheaper to approximate than higher-fidelity model.

The following code demonstrates this observation for a simple 1D model with two numerical discretizations

.. math:: f_\alpha = \cos(\pi (\rv+1)/2+\epsilon_\alpha)
"""

from functools import partial

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pyapprox.interface.model import ModelFromVectorizedCallable
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.benchmarks import MultiLevelCosineBenchmark
from pyapprox.surrogates.sparsegrids.combination import (
    AdaptiveCombinationSparseGrid,
    Max1DLevelSparseGridSubSpaceAdmissibilityCriteria,
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria,
    TensorProductRefinementCriteria,
    LevelRefinementCriteria,
    MultiIndexLejaLagrangeAdaptiveCombinationSparseGrid,
    MultipleSparseGridSubSpaceAdmissibilityCriteria,
    MaxNSamplesSparseGridSubspaceAdmissibilityCriteria,
    VarianceRefinementCriteria,
    CostFunction,
)
from pyapprox.surrogates.univariate.base import (
    ClenshawCurtisQuadratureRule,
)
from pyapprox.surrogates.univariate.lagrange import (
    UnivariateLagrangeBasis,
)
from pyapprox.surrogates.affine.multiindex import (
    DoublePlusOneIndexGrowthRule,
)
from pyapprox.util.backends.numpy import NumpyMixin as bkd

benchmark = MultiLevelCosineBenchmark(bkd)
variable = IndependentMarginalsVariable([stats.uniform(-1, 2)])
ranges = benchmark.prior().interval(1.0).flatten()
# Set the univariate quarature rules and bases
quad_rule = ClenshawCurtisQuadratureRule(store=True, bounds=[-1, 1])
# Set the univriate bases
bases_1d = [
    UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(variable.nvars())
]
growth_rule = DoublePlusOneIndexGrowthRule()


def build_tp(model, max_level_1d):
    tp = AdaptiveCombinationSparseGrid(model.nqoi(), variable.nvars())
    tp.setup(
        Max1DLevelSparseGridSubSpaceAdmissibilityCriteria(max_level_1d),
        TensorProductRefinementCriteria(),
        bases_1d,
        growth_rule,
    )
    tp.build(model)
    return tp


model_0 = benchmark.models().get_model(np.array([0]))
model_1 = benchmark.models().get_model(np.array([1]))
model_2 = benchmark.models().get_model(np.array([2]))
axs = plt.subplots(1, 3, figsize=(3 * 8, 6), sharey=True)[1]
lf_approx = build_tp(model_0, 2)
model_0.plot_surface(axs[0], ranges, "k", label=r"$f_0$")
model_1.plot_surface(axs[0], ranges, "r", label=r"$f_1$")
lf_approx.plot_surface(axs[0], ranges, "g:", label=r"$f_{0,\mathcal{I}_0}$")
axs[0].plot(lf_approx.train_samples()[0], lf_approx.train_values()[:, 0], "ko")
axs[0].legend(fontsize=18)

hf_approx = build_tp(model_1, 1)
model_1.plot_surface(axs[1], ranges, "r", label=r"$f_1$")
axs[1].plot(hf_approx.train_samples()[0], hf_approx.train_values()[:, 0], "ro")
hf_approx.plot_surface(
    axs[1], ranges, ":", color="gray", label=r"$f_{1,\mathcal{I}_1}$"
)
axs[1].legend(fontsize=18)


def discrepancy_fun(fun1, fun0, zz):
    return fun1(zz) - fun0(zz)


discrapancy_model = ModelFromVectorizedCallable(
    1, 1, partial(discrepancy_fun, model_1, model_0)
)


discp_approx = build_tp(discrapancy_model, 1)
model_1.plot_surface(axs[2], ranges, "r", label=r"$f_1$")
discrapancy_model.plot_surface(axs[2], ranges, "k", label=r"$\delta=f_1-f_0$")
axs[2].plot(
    discp_approx.train_samples()[0], discp_approx.train_values()[:, 0], "ko"
)
discp_approx.plot_surface(
    axs[2], ranges, "g:", label=r"$\delta_{\mathcal{I}_1}$"
)
additive_model = ModelFromVectorizedCallable(
    1, 1, lambda x: lf_approx(x) + discp_approx(x)
)
additive_model.plot_surface(
    axs[2], ranges, "b:", label=r"$f_{0,\mathcal{I}_1}+\delta_{\mathcal{I}_1}$"
)
[ax.set_xlabel(r"$z$") for ax in axs]
_ = axs[2].legend(fontsize=18)

# %%
# The left plot shows that using 5 samples of the low-fidelity model produces an accurate approximation of the low-fidelity model, but it will be a poor approximation of the high fidelity model in the limit of infinite low-fidelity data. The middle plot shows three samples of the high-fidelity model also produces a poor approximation, but if more samples were added the approximation would coverge to the high-fidelity model. In contrast the right plot shows that 5 samples of the low-fideliy model plus three samples of the high-fidelity model produces a good approximation of the high-fidelity model.

# %%
# The following code plots different interpolations :math:`f_{\alpha,\beta}` of :math:`f_\alpha` for various :math:`\alpha` and number of interpolation points (controled by :math:`\beta`). Instead of building each interpolant with a custom function, we just build a multi-index sparse grid that uses a tensor-product refinement criterion to define the set :math:`\mathcal{I}=\{[\alpha,\beta]:\alpha \le l_0, \; \beta\le l_1\}`
max_level = 2

# cannot let max_level_1d for configure variables be larger
# than number of configure variables
max_level_1d = [max_level, 1]  # [l_0, l_1]

fig, axs = plt.subplots(
    max_level_1d[1] + 1,
    max_level_1d[0] + 1,
    figsize=((max_level_1d[0] + 1) * 8, (max_level_1d[1] + 1) * 6),
    sharey=True,
)

model_ensemble = benchmark.models()
tp = MultiIndexLejaLagrangeAdaptiveCombinationSparseGrid(
    benchmark.prior(),
    benchmark.nqoi(),
    benchmark.nrefinement_vars(),
    benchmark.models()._index_bounds,
)
tp.setup(
    Max1DLevelSparseGridSubSpaceAdmissibilityCriteria(max_level_1d),
    TensorProductRefinementCriteria(),
)
tp.build(model_ensemble)


fun_colors = ["r", "k", "cyan"]
approx_colors = ["b", "g", "pink"]
subspace_indices = tp._subspace_gen.get_indices()
for ii, subspace_index in enumerate(subspace_indices.T):
    jj, kk = subspace_index
    ax = axs[max_level_1d[1] - kk, jj]
    for ll in range(max_level_1d[1] + 1):
        model_ll = benchmark.models().get_model(np.array([ll]))
        model_ll.plot_surface(
            ax, ranges, "-", color=fun_colors[ll], label=r"$f_{%d}$" % (jj)
        )
    tp._subspace_surrogates[ii].plot_surface(
        ax,
        ranges,
        "--",
        color=approx_colors[kk],
        label=r"$f_{%d,%d}$" % (kk, jj),
    )
    ax.plot(
        tp._subspace_surrogates[ii].get_train_samples()[0],
        tp._subspace_surrogates[ii].get_train_values(),
        "o",
        color=fun_colors[kk],
    )
    ax.legend(fontsize=18)
_ = [[ax.set_ylim([-1, 1]), ax.set_xlabel(r"$z$")] for ax in axs.flatten()]


# %%
# Multi-level Collocation
# -----------------------
# Similar to sparse grids, multi-index collocation is a weighted combination of
# low-resolution tensor products, like those shown in the last plot
#
# .. math:: f_{\mathcal{I}}(z) = \sum_{[\alpha,\beta]\in \mathcal{I}} c_{[\alpha,\beta]} f_{\alpha,\beta}(z),
#
# where the Smolay coefficients can be computed using the same formula used for traditional sparse grids. However, unlike sparse grids, we now have introduced configuration variables that change what model discretization is being evaluated.
#
# A level-one isotropic collocation algorithm uses top-left, bottom-left and bottom middle interpolants in the previous plot, i.e. :math:`f_{1, 0}, f_{0, 0}, f_{0, 1}`, respectively. The level 2 approximation is plotted below. Note it is not a true isotropic grid because it cannot reach level 2 in the configuration variable which only uses two models. This is not true if more than 2 models are provided.

sg = MultiIndexLejaLagrangeAdaptiveCombinationSparseGrid(
    benchmark.prior(),
    benchmark.nqoi(),
    benchmark.nrefinement_vars(),
    benchmark.models()._index_bounds,
)
sg.setup(
    MultipleSparseGridSubSpaceAdmissibilityCriteria(
        [
            Max1DLevelSparseGridSubSpaceAdmissibilityCriteria(max_level_1d),
            MaxLevelSparseGridSubSpaceAdmissibilityCriteria(2, 1.0),
        ]
    ),
    LevelRefinementCriteria(),
)
sg.build(model_ensemble)


ax = plt.subplots(figsize=(8, 6))[1]
sg.plot_surface(ax, ranges, "--", label=r"$f_\mathcal{I}$")
for ll in range(max_level_1d[1] + 1):
    model_ll = benchmark.models().get_model(np.array([ll]))
    model_ll.plot_surface(
        ax, ranges, "-", color=fun_colors[ll], label=r"$f_{%d}$" % (ll)
    )
ax.legend()
_ = ax.set_xlabel(r"$z$")

# %%
# The approximation is close to the accuracy of :math:`f_{1, 2}` without needing as many evaluations of :math:`f_{1}`

# %%
# Adaptivity
# ----------
# The the algorithm that adapts the sparse grid index set :math:`\mathcal{I}` to the importance of each variable be modified for use with multi-index collocation [JEGG2019]_. The algorithm is highly effective as it balances the interpolation error due to using a finite number of training points with the cost of evaluating the models of varying accuracy.
#
# Lets build a multi-level sparse grid
# The sparse grid uses the wall time of the model execution as a cost function by default, but here we will use a custom cost function because all models are trivial to evaluate.


class CustomCostFunction(CostFunction):
    def __call__(self, subspace_index):
        model_id = subspace_index[-1]
        return (1 + model_id) ** 2


import copy


class AdaptiveCallback:
    def __init__(self, validation_samples, validation_values):
        self._validation_samples = validation_samples
        self._validation_values = validation_values

        self._nsamples = []
        self._errors = []
        self._sparse_grids = []

    def __call__(self, approx):
        self._nsamples.append(approx.train_samples().shape[1])
        approx_values = approx.values_using_all_subspaces(
            self._validation_samples
        )
        error = (
            np.linalg.norm(self._validation_values - approx_values)
            / self._validation_samples.shape[1]
        )
        self._errors.append(error)
        self._sparse_grids.append(copy.deepcopy(approx))

    def errors(self):
        return self._errors

    def nsamples(self):
        return self._nsamples

    def sparse_grids(self):
        return self._sparse_grids


validation_samples = variable.rvs(100)
validation_values = model_ensemble.highest_fidelity_model()(validation_samples)
adaptive_callback = AdaptiveCallback(validation_samples, validation_values)
sg = MultiIndexLejaLagrangeAdaptiveCombinationSparseGrid(
    benchmark.prior(),
    benchmark.nqoi(),
    benchmark.nrefinement_vars(),
    benchmark.models()._index_bounds,
)
sg.setup(
    MultipleSparseGridSubSpaceAdmissibilityCriteria(
        [
            Max1DLevelSparseGridSubSpaceAdmissibilityCriteria([10, 2]),
            MaxNSamplesSparseGridSubspaceAdmissibilityCriteria(20),  # 80
        ]
    ),
    VarianceRefinementCriteria(CustomCostFunction()),
)
while sg.step(model_ensemble):
    adaptive_callback(sg)
adaptive_callback(sg)
# todo activate cost function

# %%
# Now plot the adaptive algorithm
fig, axs = plt.subplots(1, 3, sharey=False, figsize=(3 * 8, 6))


def animate(ii):
    [ax.clear() for ax in axs]
    sg = adaptive_callback.sparse_grids()[ii]
    sel_samples = sg.selected_train_samples()
    cand_samples = sg.candidate_train_samples()
    axs[1].plot(*sel_samples, "ko", ms=20)
    axs[1].plot(*cand_samples, "rX", ms=20)
    # axs[1].plot(*sg.train_samples(), "o")
    sg._subspace_gen.plot_indices(axs[0])
    sg.plot_surface(axs[2], ranges, ls="--", color="b", label=r"$f_{I}(z_1)$")
    axs[0].set_xlim([0, 10])
    axs[0].set_ylim([0, 2])
    axs[1].set_ylim([0, 2])
    axs[1].set_ylabel(r"$\alpha_1$")
    model_0.plot_surface(axs[2], ranges, color="r", label=r"$f_0(z_1)$")
    model_1.plot_surface(axs[2], ranges, color="g", label=r"$f_1(z_1)$")
    model_2.plot_surface(axs[2], ranges, color="k", label=r"$f_2(z_1)$")
    axs[2].set_xlabel(r"$z_1$")
    axs[2].legend(fontsize=18)


import matplotlib.animation as animation

ani = animation.FuncAnimation(
    fig,
    animate,
    interval=500,
    frames=len(adaptive_callback.sparse_grids()),
    repeat_delay=1000,
)
ani.save("adaptive_misc.gif", dpi=100)

# %%
# The lower fidelity models are evaluated more until they can no longer reduce the error in the sparse grid. At this point the interpolation error of the low-fidelity models is dominated by the bias in the exact low-fidelity models. Changing the cost_function will change how many samples are used to evaluate each model.

# %%
# Three or more models
# --------------------
# This tutorial only demonstrates the use of multi-level collocation with two models, but it can easily be used with three or more models. Try setting config_values = [np.asarray([0.25, 0.125, 0])]

# %%
# Multi-index collocation
# -----------------------
# Multi-index collocation is an extension of mulit-level collocation that can be used with models that have multiple parameters controlling the numerical discretization, for example, the spatial and temporal resoutions of a finite element solver. While this tutorial does not demonstrate multi-index collocation it is supported by PyApprox.
#
# .. list-table::
#
#   * - .. _multiindex_hierarchy:
#
#       .. figure:: ../../figures/multiindex-hierarchy.png
#          :width: 50%
#          :align: center
#
#          A multi-index hierarchy formed by increasing mesh discretizations in two different spatial directions.


# %%
# References
# ^^^^^^^^^^
# .. [HNTTCMAME2016] `A. Haji-Ali, F. Nobile, L. Tamellini, and R. Tempone. Multi-index stochastic collocation for random pdes. Computer Methods in Applied Mechanics and Engineering, 306:95 – 122, 2016. <http://www. sciencedirect.com/science/article/pii/S0045782516301141, doi:10.1016/j.cma.2016.03.029>`_
#
# .. [TJWGSIAMUQ2015] `A. Teckentrup, P. Jantsch, C. Webster, and M. Gunzburger. A multilevel stochastic collocation method for partial differential equations with random input data. SIAM/ASA Journal on Uncertainty Quantification, 3(1):1046-1074, 2015. <https://doi.org/10.1137/140969002>`_
#
# .. [JEGG2019] `J.D. Jakeman, M.S. Eldred, G. Geraci, and A. Gorodetsky. Adaptive Multi-index Collocation for Uncertainty Quantification and Sensitivity Analysis. International Journal for Numerical Methods in Engineering (2019). <https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.6268>`_
