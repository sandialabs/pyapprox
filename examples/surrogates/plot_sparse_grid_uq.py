"""
Sparse Grid Interpolation
=========================

We will use the Genz benchmark to illustrate how to use a sparse grid
as a surrogate for uncertainty quantification and sensitivity analysis

First we must load the benchmark
"""

import numpy as np
from scipy import stats
from pyapprox.benchmarks.genz import GenzBenchmark
from pyapprox.surrogates.bases.multiindex import (
    DoublePlusOneIndexGrowthRule,
    IterativeIndexGenerator,
)
from pyapprox.surrogates.bases.univariate.base import (
    ClenshawCurtisQuadratureRule,
)
from pyapprox.surrogates.bases.univariate.lagrange import (
    UnivariateLagrangeBasis,
)
from pyapprox.surrogates.sparsegrids.combination import (
    AdaptiveCombinationSparseGrid,
    MaxNSamplesSparseGridSubspaceAdmissibilityCriteria,
    L2NormRefinementCriteria,
)
from pyapprox.surrogates.bases.basis import TensorProductInterpolatingBasis
import matplotlib.pyplot as plt

# define the model to be approximated
# Increasing cfactor makes the function harder to approximate
nvars = 2
benchmark = GenzBenchmark(name="oscillatory", nvars=nvars, cfactor=10)
np.random.seed(1)

# %%
# Define the sparse grid properties

# Set the univariate quarature rules and bases
quad_rule = ClenshawCurtisQuadratureRule(store=True, bounds=[-1, 1])
# Set the univriate bases
bases_1d = [UnivariateLagrangeBasis(quad_rule, 3) for dim_id in range(nvars)]
# Define how the number of univariate rules grow
growth_rule = DoublePlusOneIndexGrowthRule()

# %%
#  Build the sparse grid with at approximately 100 samples. Sparse grids
# cannot guarantee budget will be satisfied exactly
max_nsamples = 100
sg = AdaptiveCombinationSparseGrid(
    benchmark.model().nqoi(), benchmark.variable().nvars()
)
basis = TensorProductInterpolatingBasis(bases_1d)
sg.set_basis(basis)
subspace_gen = IterativeIndexGenerator(nvars)
sg.set_subspace_generator(subspace_gen, growth_rule)
sg.set_subspace_admissibility_criteria(
    MaxNSamplesSparseGridSubspaceAdmissibilityCriteria(max_nsamples)
)
sg.set_refinement_criteria(L2NormRefinementCriteria())
sg.set_initial_subspace_indices()
sg.set_verbosity(0)
sg.build(benchmark.model())

# %%
# Plot the sparse grid samples and subspace indices
axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
sg.plot_grid(axs[0])
sg._subspace_gen.plot_indices(axs[1])

# %%
# We can estimate the error in the surrogate using some validation samples.
validation_samples = benchmark.variable().rvs(100)
validation_values = benchmark.model()(validation_samples)
approx_values = sg(validation_samples)
error = np.linalg.norm(validation_values - approx_values, axis=0) / np.sqrt(
    validation_values.shape[0]
)
print(f"The RMSE error is {error}")

# %% We can estimate the PDF of the two function outputs by sampling the surrogate
# and building a kernel density estimator. Lets first just plot the marginal
# PDFs of the output using the surrogate and the real model
samples = benchmark.variable().rvs(10000)
sg_values = sg(samples)
true_values = benchmark.model()(samples)

ax = plt.figure().gca()
sg_kde = stats.gaussian_kde(sg_values[:, 0])
true_kde = stats.gaussian_kde(true_values[:, 0])
yy = np.linspace(true_values.min(), true_values.max(), 101)
ax.plot(yy, true_kde(yy), "-k", label="True PDF")
ax.plot(yy, sg_kde(yy), "--r", label="Sparse Grid PDF")
_ = ax.legend()
