"""
Multi-index Collocation
-----------------------
The following provides an example of how to use multivariate quadrature, e.g. multilevel Monte Carlo, control variates to estimate the mean of a high-fidelity model from an ensemble of related models of varying cost and accuracy. Refer to :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multiindex_collocation.py` for a detailed tutorial on the theory behind multi-index collocation.
"""

# %%
# Load the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.benchmarks import MultiLevelCosineBenchmark
from pyapprox.surrogates.sparsegrids.combination import (
    MultiIndexLejaLagrangeAdaptiveCombinationSparseGrid,
    MultipleSparseGridSubSpaceAdmissibilityCriteria,
    MaxErrorSparseGridSubspaceAdmissibilityCriteria,
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria,
    VarianceRefinementCriteria,
)
from pyapprox.util.backends.numpy import NumpyMixin as bkd

# set seed for reproducibility
np.random.seed(1)

# %%
# Set up a :py:class:`~pyapprox.interface.model.MultiIndexModelEnsemble` that takes samples :math:`x=[z_1,\ldots,z_D,v_1,\ldots, v_C]` which is the concatenation of a random sample `z` and configuration values specifying the discretization parameters of the numerical model.

# load the benchmark
benchmark = MultiLevelCosineBenchmark(bkd)
# extract the model enemble
model_ensemble = benchmark.models()

# %%
# Now set up and run the multi-index algorithm
sg = MultiIndexLejaLagrangeAdaptiveCombinationSparseGrid(
    benchmark.prior(),
    benchmark.nqoi(),
    benchmark.nrefinement_vars(),
    benchmark.models()._index_bounds,
)
admissibility_criteria = MultipleSparseGridSubSpaceAdmissibilityCriteria(
    [
        MaxErrorSparseGridSubspaceAdmissibilityCriteria(1e-5),
        MaxLevelSparseGridSubSpaceAdmissibilityCriteria(6, 1.0),
    ]
)
sg.setup(
    admissibility_criteria,
    refinement_criteria=VarianceRefinementCriteria(),
)
sg.set_verbosity(0)
sg.build(model_ensemble)

# %%
# The following can be used to plot the approximation and high-fidelity target
# function if there is only one random variable z and one configuration variable
ax = plt.subplots()[1]
model_ensemble.highest_fidelity_model().plot_surface(
    ax, plot_limits=[-1, 1], label="True"
)
sg.plot_surface(ax, plot_limits=[-1, 1], label="SG", ls="--")
_ = ax.legend()
