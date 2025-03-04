r"""
Lagrange Sparse Grid Sensitivity Analysis
-----------------------------------------
First Build the sparse grid surrogate
"""

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.benchmarks import IshigamiBenchmark
from pyapprox.analysis.sensitivity_analysis import (
    LagrangeSparseGridSensitivityAnalysis,
    plot_main_effects,
    plot_total_effects,
)
from pyapprox.surrogates.sparsegrids.combination import (
    MultipleSparseGridSubSpaceAdmissibilityCriteria,
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria,
    MaxErrorSparseGridSubspaceAdmissibilityCriteria,
    LejaLagrangeAdaptiveCombinationSparseGrid,
)

np.random.seed(1)

benchmark = IshigamiBenchmark(a=7, b=0.1)
level = 7
admissibility_criteria = MultipleSparseGridSubSpaceAdmissibilityCriteria(
    [
        MaxLevelSparseGridSubSpaceAdmissibilityCriteria(level, 1.0),
        MaxErrorSparseGridSubspaceAdmissibilityCriteria(1e-8),
    ]
)
sg = LejaLagrangeAdaptiveCombinationSparseGrid(
    benchmark.variable(), benchmark.model().nqoi()
)
# Must not use default of mean or the refinement will terminate early
# in the 3rd dimension
init_sequences = [
    np.array([[marginal.ppf(0.6)]])
    for marginal in benchmark.variable().marginals()
]
univariate_quad_rules = sg.unique_univariate_leja_quadrature_rules(
    init_sequences
)
sg.setup(admissibility_criteria, univariate_quad_rules=univariate_quad_rules)
sg.build(benchmark.model())

# %%
# Compute the sensivitity indices
analyzer = LagrangeSparseGridSensitivityAnalysis(benchmark.variable())
analyzer.set_interaction_terms_of_interest(
    benchmark.sobol_interaction_indices()
)
analyzer.compute(sg)

# %%
# Plot the sensitivity indices
axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
plot_main_effects(analyzer.main_effects(), axs[0])
_ = plot_total_effects(analyzer.total_effects(), axs[1])
plt.show()

# %%
# The benchmark contains the exact values of these indices which can be used
# to validate the answers obtained.
# Change the number of training samples and PCE degree to explore their impact on the accuracy of the sensitivity indices.
