r"""
Sparse Grids
============
The number of model evaluations required by tensor product interpolation grows exponentitally with the number of model inputs. This tutorial introduces sparse grids [BNR2000]_, [BG2004]_ which can be used to overcome the so called curse of dimensionality faced by tensor-product methods.

Sparse grids approximate a model (function) :math:`f_\alpha` with :math:`D` inputs :math:`z=[z_1,\ldots,z_D]^\top` as a linear combination of low-resolution tensor product interpolantsm that is

.. math:: f_{\alpha, \mathcal{I}}(z) = \sum_{\beta\in \mathcal{I}} c_\beta f_{\alpha,\beta}(z),

where :math:`\beta=[\beta_1,\ldots,\beta_D]` is a multi-index controlling the number of samples in each dimension of the tensor-product interpolants, and the index set :math:`\mathcal{I}` controls the approximation accuracy and data-efficiency of the sparse grid. If the set :math:`\mathcal{I}` is downward closed, that is

.. math:: \gamma \le \beta \text{ and } \beta \in \mathcal{I} \implies \gamma \in \mathcal{I},

where the :math:`\le` is applied per entry, then the (Smolyak) coefficients of the sparse grid are given by

.. math:: \sum_{i\in [0,1]^D, \alpha+i\in \mathcal{I}} (-1)^{\lVert i \rVert_1}.


While any tensor-product approximation can be used with sparse grids, e.g. based on piecewise-polynomials or splines, in this tutorial we will build sparse grids with Lagrange polynomials (see :ref:`sphx_glr_auto_tutorials_surrogates_plot_tensor_product_interpolation.py`).

The following code compares a tensor-product interpolant with a level-:math:`l` isotropic sparse grid which sets

.. math:: \mathcal{I}(l)=\{\beta \mid (\max(0,l-1)\le \lVert\beta\rVert_1\le l+D-2\}, \quad l\ge 0

which leads to a simpler expression for the coefficients

.. math:: c_\beta = (-1)^{l-\lvert\beta\rvert_1} {D-1\choose l-\lvert\beta\rvert_1}.

First import the necessary modules and define the function we will approximate and its variable :math:`\rv`.

.. math:: f(\rv) = \cos(\pi\rv_1)\cos(\pi\rv_2/2)
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.utilities import nchoosek

from pyapprox.surrogates.sparsegrids.combination import (
    AdaptiveCombinationSparseGrid,
    Max1DLevelSparseGridSubSpaceAdmissibilityCriteria,
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria,
    TensorProductRefinementCriteria,
    LevelRefinementCriteria,
    VarianceRefinementCriteria,
    MaxNSamplesSparseGridSubspaceAdmissibilityCriteria,
)
from pyapprox.surrogates.bases.multiindex import (
    DoublePlusOneIndexGrowthRule,
)
from pyapprox.surrogates.bases.univariate.base import (
    ClenshawCurtisQuadratureRule,
)
from pyapprox.surrogates.bases.univariate.lagrange import (
    UnivariateLagrangeBasis,
)
from pyapprox.interface.model import ModelFromVectorizedCallable
from pyapprox.benchmarks import GenzBenchmark

variable = IndependentMarginalsVariable([stats.uniform(-1, 2)] * 2)


def fun(zz):
    return (np.cos(np.pi * zz[0]) * np.cos(np.pi * zz[1] / 2))[:, None]


model = ModelFromVectorizedCallable(1, variable.nvars(), fun)


# %%
# Now plot the tensor product interpolants and the Smolyak coefficients that make up the sparse grid. The coefficients are in the upper left corner of each subplot.
max_level = 2
fig, axs = plt.subplots(
    max_level + 1,
    max_level + 1,
    figsize=((max_level + 1) * 8, (max_level + 1) * 6),
)
ranges = variable.get_statistics("interval", 1.0).flatten()

# Set the univariate quarature rules and bases
quad_rule = ClenshawCurtisQuadratureRule(store=True, bounds=[-1, 1])
# Set the univriate bases
bases_1d = [
    UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(variable.nvars())
]
# Define how the number of univariate rules grow
growth_rule = DoublePlusOneIndexGrowthRule()


def build_tp(max_level_1d):
    tp = AdaptiveCombinationSparseGrid(model.nqoi(), variable.nvars())
    tp.setup(
        Max1DLevelSparseGridSubSpaceAdmissibilityCriteria(max_level_1d),
        TensorProductRefinementCriteria(),
        bases_1d,
        growth_rule,
    )
    tp.build(model)
    return tp


levels = np.linspace(-1.1, 1.1, 31)
text_props = dict(boxstyle="round", facecolor="white", alpha=0.5)
for ii in range(0, max_level + 1):
    for jj in range(max_level + 1):
        tp = build_tp([ii, jj])
        ax = axs[max_level - jj][ii]
        tp.plot_contours(ax, ranges, npts_1d=71, cmap="coolwarm")
        ax.plot(*tp.train_samples(), "ko")
        coef = int(
            (-1) ** (max_level - (ii + jj))
            * nchoosek(variable.nvars() - 1, max_level - (ii + jj))
        )
        ax.text(
            0.05,
            0.95,
            r"${:d}$".format(coef),
            transform=ax.transAxes,
            fontsize=24,
            verticalalignment="top",
            bbox=text_props,
        )


# %%
# Now compare the sparse grid with a tensor product interpolant of the same level.
fig, axs = plt.subplots(1, 3, figsize=(3 * 8, 6))
tp = build_tp([max_level] * variable.nvars())
sg = AdaptiveCombinationSparseGrid(model.nqoi(), variable.nvars())
sg.setup(
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria(max_level, 1.0),
    LevelRefinementCriteria(),
    bases_1d,
    growth_rule,
)
sg.build(model)
X, Y, pts = model.meshgrid_samples(ranges, npts_1d=51)
vals = [model(pts), tp(pts), sg(pts)]
lb = np.min([v.min() for v in vals])
ub = np.max([v.max() for v in vals])
levels = np.linspace(lb, ub, 21)
im = model.plot_contours(axs[0], ranges, levels=levels, cmap="coolwarm")
tp.plot_contours(axs[1], ranges, levels=levels, cmap="coolwarm")
axs[1].plot(*tp.train_samples(), "ko")
sg.plot_contours(axs[2], ranges, levels=levels, cmap="coolwarm")
axs[2].plot(*sg.train_samples(), "ko")
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.9125, 0.125, 0.025, 0.75])
_ = fig.colorbar(im, cax=cbar_ax)

# %%
# The sparse grid is slightly less accurate than the tensor product interpolant, but uses fewer points. There is no exact formula for the number of points in an isotropic sparse grid. The following code can be used to determine the number of points in a sparse grid of any dimension or level. The number of points is much smaller than the number of points in a tensor-product grid, for a given level :math:`l`.


def get_isotropic_sparse_grid_nsamples(nvars, max_level):
    """
    Get the number of points in an isotropic sparse grid
    """
    from pyapprox.surrogates.sparsegrids.combination import (
        IsotropicCombinationSparseGrid,
        TensorProductInterpolatingBasis,
    )

    bases_1d = [
        UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(nvars)
    ]
    sg = IsotropicCombinationSparseGrid(
        1,
        nvars,
        max_level,
        DoublePlusOneIndexGrowthRule(),
        TensorProductInterpolatingBasis(bases_1d),
    )
    return sg.train_samples().shape[1]


dims = [2, 3, 4]
levels = np.arange(1, 5)
sg_nsamples = [
    [get_isotropic_sparse_grid_nsamples(nvars, level) for level in levels]
    for nvars in dims
]
tp_nsamples = [
    [DoublePlusOneIndexGrowthRule()(level) ** nvars for level in levels]
    for nvars in dims
]
print("Growth of number of sparse grid points")
print(sg_nsamples)
print("Growth of number of tensor-product points")
print(tp_nsamples)

axs = plt.subplots(1, 3, figsize=(3 * 8, 6), sharey=True)[1]
for ii in range(len(sg_nsamples)):
    axs[ii].semilogy(levels, tp_nsamples[ii], "-o", label="tensor product")
    axs[ii].semilogy(levels, sg_nsamples[ii], "--s", label="sparse_grid")
    axs[ii].set_xlabel("Level")
    axs[ii].set_title(f"Dimensions: {dims[ii]}")
axs[0].set_ylabel("Number of points")
axs[0].legend()

# %%
# For a function with :math:`r` continous mixed-derivatives, the isotropic level-:math:`l` sparse grid, based on 1D Clenshaw Curtis abscissa, with :math:`M_{\mathcal{I}(l)}` points satisfies
#
# .. math:: \lVert f-f_{\mathcal{I}(l)}\rVert_{L^\infty}\le C_{D,r} M^{-r}_{\mathcal{I}(l)}(\log M_{\mathcal{I}(l)})^{(r+2)(D-1)+1}.
#
# In contrast the tensor-product interpolant with :math:`M_l` points satifies
#
# .. math:: \lVert f-f_{\mathcal{I}(l)}\rVert_{L^\infty}\le K_{D,r} M_l^{-r/D}.
#
# The following code compares the convergence of sparse grids and tensor-product lagrange interpolants. A callback is used to compute the error as the level of the approximations increases


class IsotropicCallback:
    def __init__(self, validation_samples, validation_values, istp):
        self._level = -1
        self._errors = []
        self._nsamples = []
        self._validation_samples = validation_samples
        self._validation_values = validation_values
        self._istp = istp

    def __call__(self, approx):
        if self._istp:
            approx_level = approx._subspace_gen.get_selected_indices().max()
        else:
            approx_level = (
                approx._subspace_gen.get_selected_indices().sum(axis=0).max()
            )
        if self._level != approx_level:
            # only compute error when all subspaces of the current
            # approximation level are added to the sparse grid.
            # The number of sparse grid points will be slightly larger
            # than an isotoropic grid of level=approx_level because
            # points associated with active indices will be included here.
            self._level = approx_level
            self._nsamples.append(approx.train_samples().shape[1])
            approx_values = approx.values_using_all_subspaces(
                self._validation_samples
            )
            error = (
                np.linalg.norm(self._validation_values - approx_values)
                / self._validation_samples.shape[1]
            )
            self._errors.append(error)

    def errors(self):
        return self._errors

    def nsamples(self):
        return self._nsamples


def fun(xx):
    return np.exp(-0.05 * (((xx + 1) / 2 - 0.5) ** 2).sum(axis=0))[:, None]


nvars = 4
variable = IndependentMarginalsVariable([stats.uniform(-1, 2)] * nvars)
model = ModelFromVectorizedCallable(1, nvars, fun)


validation_samples = variable.rvs(1000)
validation_values = model(validation_samples)

bases_1d = [UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(nvars)]

tp_max_level = 4
tp_callback = IsotropicCallback(validation_samples, validation_values, True)

tp = AdaptiveCombinationSparseGrid(model.nqoi(), variable.nvars())
tp.setup(
    Max1DLevelSparseGridSubSpaceAdmissibilityCriteria(tp_max_level),
    TensorProductRefinementCriteria(),
    bases_1d,
    growth_rule,
)
tp_callback = IsotropicCallback(validation_samples, validation_values, True)
while tp.step(model):
    tp_callback(tp)
# compute error at final level
tp_callback._level = tp_max_level  # set so callback computes error
tp_callback(tp)

sg_max_level = 6
sg_callback = IsotropicCallback(validation_samples, validation_values, False)
sg = AdaptiveCombinationSparseGrid(model.nqoi(), variable.nvars())
sg.setup(
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria(sg_max_level, 1.0),
    LevelRefinementCriteria(),
    bases_1d,
    growth_rule,
)
while sg.step(model):
    sg_callback(sg)
# compute error at final level
sg_callback._level = sg_max_level  # set so callback computes error
sg_callback(sg)

ax = plt.subplots(1, 1, figsize=(8, 6))[1]
ax.loglog(tp_callback.nsamples(), tp_callback.errors(), "-o", label="TP")
ax.loglog(sg_callback.nsamples(), sg_callback.errors(), "--s", label="SG")
ax.set_title(f"Dimensions: {nvars}")
_ = ax.legend()

# %%
# Experiment with changing nvars, e.g. try nvars = 2,3,4. Sparse grids become more effective as nvars increases.
#
# So far we have used sparse grids based on Clenshaw-Curtis 1D quadrature rules. However other types of rules can be used. PyApprox also supports 1D Leja sequences  [NJ2014]_ (see :ref:`sphx_glr_auto_tutorials_surrogates_plot_adaptive_leja_interpolation.py`). Change univariate_quad_rule=None to use Leja rules and observe the difference in convergence.

# %%
# Dimension adaptivity
# --------------------
# The efficiency of sparse grids can be improved using methods [GG2003]_, [H2003]_ that construct the index set :math:`\mathcal{I}` adaptively. This is the default behavior when using Pyapprox. The following applies the adaptive algorithm to an anisotropic function, where one variable impacts the function much more than the other.
#
# Finding an efficient index set can be cast as an optimization problem. With this goal, let the difference in sparse grid error before and after the interpolant :math:`f_{\alpha,\beta}` and the work from adding the new interpolant respectively be
#
# .. math::
#    \begin{align*}\Delta E_\beta = \lVert f_{\alpha, \mathcal{I}\cup\beta}-f_{\alpha, \mathcal{I}}\rVert && \Delta W_\beta = \lVert W_{\alpha, \mathcal{I}\cup\beta}-W_{\alpha, \mathcal{I}}\rVert\end{align*}
#
# Then noting that the error in the sparse grid satisfies, we can formulate finding a quasi-optimal index set as a binary knapsack problem
#
# .. math:: \max \sum_{\beta}\Delta E_\beta\delta_\beta \text{ such that }\sum_{\beta}\Delta W_\beta\delta_\beta \le W_{\max},
#
# for a total work budget :math:`W_{\max}`. The solution to this problem balances the computational work of adding a specific interpolant with the reduction in error that would be achieved.
#
# The isotropic index set represents a solution to this knapsack problem under certain conditions on the smoothness of the function being approximated. However, for weaker conditions, finding optimal index sets by solving the knapsack problem is typically intractable. Consequently, we use a greedy adaptive procedure.
#
# The algorithm begins with the index set :math:`\mathcal{I}=\{\beta\mid \beta=[0, \ldots, 0]\}` then identifies, so called active indices, which are candidates for refinement. The active indices satisfy the downward closed admissibility criteria above. The function is evaluated at the points assoicated with the active indices and error indicators, similar to :math:`\Delta W_\beta`, are computed. The active index with the largest error indicator is then added to :math:`\mathcal{I}` and the active set is updated. This procedure is repeated until and error threshold is met or a computational budget reached.
#
# First set up the benchmark

benchmark = GenzBenchmark(
    "oscillatory",
    2,
    coefs=(np.atleast_2d([2, 0.2]).T, np.atleast_2d([0, 0]).T),
)


# %%
# Now build a sparse grid using a callback that tracks important properties
# of the sparse grid and its adaptation.
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

    def sparse_grids(self):
        return self._sparse_grids


validation_samples = benchmark.variable().rvs(1000)
validation_values = benchmark.model()(validation_samples)
adaptive_callback = AdaptiveCallback(validation_samples, validation_values)
sg = AdaptiveCombinationSparseGrid(
    benchmark.model().nqoi(), benchmark.variable().nvars()
)
bases_1d = [
    UnivariateLagrangeBasis(quad_rule, 3)
    for dim_id in range(benchmark.variable().nvars())
]
sg.setup(
    MaxNSamplesSparseGridSubspaceAdmissibilityCriteria(100),
    VarianceRefinementCriteria(),
    bases_1d,
    growth_rule,
)
while sg.step(benchmark.model()):
    adaptive_callback(sg)
# compute error at final level
adaptive_callback._level = sg_max_level  # set so callback computes error
adaptive_callback(sg)

# %%
# The following visualizes the adaptive algorithm.
#
# The left plot depicts the multi-index of each tensor product interpolant. They gray boxes represent indices added to the sparse grid and the red boxes represent the active indices. The numbers in the boxes represent the Smolyak coefficients.
#
# The middle plot shows the grid points associated with the gray boxes (black dots) and the grid points associated with the active indices (red dots).
#
# The left plot depicts the sparse grid approximation at each iteration.
#
# The sparse grid spends more points resolving the function variation in the horizontal direction, associated with the most sensitive function input.

fig, axs = plt.subplots(1, 3, sharey=False, figsize=(3 * 8, 6))
ranges = benchmark.variable().get_statistics("interval", 1.0).flatten()
X, Y, pts = sg.meshgrid_samples(ranges, npts_1d=51)
data = [sg(pts) for sg in adaptive_callback.sparse_grids()]
Z_min = np.min([d for d in data])
Z_max = np.max([d for d in data])
levels = np.linspace(Z_min, Z_max, 21)


def animate(ii):
    [ax.clear() for ax in axs]
    sg = adaptive_callback.sparse_grids()[ii]
    sg._subspace_gen.plot_indices(axs[0])
    sg.plot_grid(axs[1])
    sg.plot_contours(axs[2], ranges, levels=levels)
    axs[0].set_xlim([0, 10])
    axs[0].set_ylim([0, 10])


import matplotlib.animation as animation

ani = animation.FuncAnimation(
    fig,
    animate,
    interval=500,
    frames=len(adaptive_callback.sparse_grids()),
    repeat_delay=1000,
)
ani.save("adaptive_sparse_grid.gif", dpi=100)

# %%
# References
# ^^^^^^^^^^
# .. [BNR2000] `V. Barthelmann, E. Novak and K. Ritter. High dimensional polynomial interpolation on sparse grid. Advances in Computational Mathematics (2000). <https://doi.org/10.1023/A:1018977404843>`_
# .. [BG2004] `H. Bungartz and  M. Griebel. Sparse grids. Acta Numerica (2004). <https://doi.org/10.1017/S0962492904000182>`_
# .. [GG2003] `T. Gerstner and M. Griebel. Dimension-adaptive tensor-product quadrature. Computing (2003). <https://doi.org/10.1007/s00607-003-0015-5>`_
# .. [H2003] `M. Hegland. Adaptive sparse grids. Proc. of 10th Computational Techniques and Applications Conference (2003). <https://doi.org/10.21914/anziamj.v44i0.685>`_
# .. [NJ2014] `A. Narayan and J.D. Jakeman. Adaptive Leja Sparse Grid Constructions for Stochastic Collocation and High-Dimensional Approximation. SIAM Journal on Scientific Computing (2014). <http://dx.doi.org/10.1137/140966368>`_

raise NotImplementedError("Fix sparse grid tests")
