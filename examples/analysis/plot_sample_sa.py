r"""
Sample-Based Sensitivity Analysis
---------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from pyapprox.benchmarks import IshigamiBenchmark
from pyapprox.analysis.sensitivity_analysis import (
    SobolSequenceBasedSensitivityAnalysis,
    plot_main_effects,
    plot_total_effects,
)
from pyapprox.util.backends.numpy import NumpyMixin as bkd

np.random.seed(1)

benchmark = IshigamiBenchmark(bkd, a=7, b=0.1)
nsamples = 10000
analyzer = SobolSequenceBasedSensitivityAnalysis(benchmark.prior(), 100)
analyzer.set_interaction_terms_of_interest(
    benchmark.sobol_interaction_indices()
)
samples = analyzer.generate_samples(nsamples)
values = benchmark.model()(samples)
analyzer.compute(values)

# %%
# Plot the sensitivity indices
axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
plot_main_effects(analyzer.main_effects(), axs[0])
_ = plot_total_effects(analyzer.total_effects(), axs[1])

# %%
# The benchmark contains the exact values of these indices which can be used
# to validate the answers obtained
