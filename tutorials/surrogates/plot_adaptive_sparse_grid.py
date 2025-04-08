r"""
Adaptive Sparse Grids
=====================
The number of model evaluations required by tensor product interpolation grows exponentitally with the number of model inputs. In constrast, the error of isotropic sparse grids only depends logarthmically on dimension. However, further improvements can be obtained by adapting the sparse grid.

The efficiency of sparse grids can be improved using methods [GG2003]_, [H2003]_ that construct the index set :math:`\mathcal{I}` adaptively. This is the default behavior when using Pyapprox. The following applies the adaptive algorithm to an anisotropic function, where one variable impacts the function much more than the other.

Finding an efficient index set can be cast as an optimization problem. With this goal, let the difference in sparse grid error before and after the interpolant :math:`f_{\alpha,\beta}` and the work from adding the new interpolant respectively be

.. math::
   \begin{align*}\Delta E_\beta = \lVert f_{\alpha, \mathcal{I}\cup\beta}-f_{\alpha, \mathcal{I}}\rVert && \Delta W_\beta = \lVert W_{\alpha, \mathcal{I}\cup\beta}-W_{\alpha, \mathcal{I}}\rVert\end{align*}

Then noting that the error in the sparse grid satisfies, we can formulate finding a quasi-optimal index set as a binary knapsack problem

.. math:: \max \sum_{\beta}\Delta E_\beta\delta_\beta \text{ such that }\sum_{\beta}\Delta W_\beta\delta_\beta \le W_{\max},

for a total work budget :math:`W_{\max}`. The solution to this problem balances the computational work of adding a specific interpolant with the reduction in error that would be achieved.

The isotropic index set represents a solution to this knapsack problem under certain conditions on the smoothness of the function being approximated. However, for weaker conditions, finding optimal index sets by solving the knapsack problem is typically intractable. Consequently, we use a greedy adaptive procedure.

The algorithm begins with the index set :math:`\mathcal{I}=\{\beta\mid \beta=[0, \ldots, 0]\}` then identifies, so called active indices, which are candidates for refinement. The active indices satisfy the downward closed admissibility criteria above. The function is evaluated at the points assoicated with the active indices and error indicators, similar to :math:`\Delta W_\beta`, are computed. The active index with the largest error indicator is then added to :math:`\mathcal{I}` and the active set is updated. This procedure is repeated until and error threshold is met or a computational budget reached.

"""

import copy
import numpy as np
import matplotlib.pyplot as plt

from pyapprox.surrogates.sparsegrids.combination import (
    AdaptiveCombinationSparseGrid,
    VarianceRefinementCriteria,
    MaxNSamplesSparseGridSubspaceAdmissibilityCriteria,
)
from pyapprox.surrogates.affine.multiindex import (
    DoublePlusOneIndexGrowthRule,
)
from pyapprox.surrogates.univariate.base import (
    ClenshawCurtisQuadratureRule,
)
from pyapprox.surrogates.univariate.lagrange import (
    UnivariateLagrangeBasis,
)
from pyapprox.benchmarks import GenzBenchmark

# %%
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
quad_rule = ClenshawCurtisQuadratureRule(store=True, bounds=[-1, 1])
bases_1d = [
    UnivariateLagrangeBasis(quad_rule, 3)
    for dim_id in range(benchmark.variable().nvars())
]
growth_rule = DoublePlusOneIndexGrowthRule()
sg.setup(
    MaxNSamplesSparseGridSubspaceAdmissibilityCriteria(100),
    VarianceRefinementCriteria(),
    bases_1d,
    growth_rule,
)
while sg.step(benchmark.model()):
    adaptive_callback(sg)
# compute error at final level
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
# .. [GG2003] `T. Gerstner and M. Griebel. Dimension-adaptive tensor-product quadrature. Computing (2003). <https://doi.org/10.1007/s00607-003-0015-5>`_
# .. [H2003] `M. Hegland. Adaptive sparse grids. Proc. of 10th Computational Techniques and Applications Conference (2003). <https://doi.org/10.21914/anziamj.v44i0.685>`_
# .. [NJ2014] `A. Narayan and J.D. Jakeman. Adaptive Leja Sparse Grid Constructions for Stochastic Collocation and High-Dimensional Approximation. SIAM Journal on Scientific Computing (2014). <http://dx.doi.org/10.1137/140966368>`_
