r"""
Polynomial Chaos Expansion Sensitivity Analysis
-----------------------------------------------
First Build the PCE surrogate
"""

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.benchmarks import IshigamiBenchmark
from pyapprox.analysis.sensitivity_analysis import (
    PolynomialChaosSensitivityAnalysis,
    plot_main_effects,
    plot_total_effects,
)
from pyapprox.surrogates.affine.basisexp import (
    setup_polynomial_chaos_expansion_from_variable,
)
from pyapprox.util.backends.numpy import NumpyMixin as bkd

np.random.seed(1)
benchmark = IshigamiBenchmark(bkd, a=7, b=0.1)
nsamples = 1000
degree = 15
pce = setup_polynomial_chaos_expansion_from_variable(
    benchmark.prior(), benchmark.model().nqoi()
)
pce.basis().set_hyperbolic_indices(degree, 1.0)
samples = benchmark.prior().rvs(nsamples)
values = benchmark.model()(samples)
pce.fit(samples, values)

# %%
# Now compute the sensitivity indices
analyzer = PolynomialChaosSensitivityAnalysis(benchmark.prior().nvars())
analyzer.set_interaction_terms_of_interest(
    benchmark.sobol_interaction_indices()
)
analyzer.compute(pce)

# %%
# Plot the sensitivity indices
axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
plot_main_effects(analyzer.main_effects(), axs[0])
_ = plot_total_effects(analyzer.total_effects(), axs[1])
# plt.show()

# %%
# The benchmark contains the exact values of these indices which can be used
# to validate the answers obtained.
# Change the number of training samples and PCE degree to explore their impact on the accuracy of the sensitivity indices.
