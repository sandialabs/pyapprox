"""Variance-based sensitivity analysis methods.

This submodule provides implementations of variance-based (Sobol) sensitivity
analysis using different surrogate models:
- PCE (Polynomial Chaos Expansion)
- Sparse grids
- Gaussian processes
- Sample-based (Monte Carlo)
"""

from pyapprox.typing.sensitivity.variance_based.base import (
    VarianceBasedSensitivityAnalysis,
)
from pyapprox.typing.sensitivity.variance_based.pce import (
    PolynomialChaosSensitivityAnalysis,
)
from pyapprox.typing.sensitivity.variance_based.sparsegrid import (
    SparseGridSensitivityAnalysis,
)
from pyapprox.typing.sensitivity.variance_based.sample_based import (
    SampleBasedSensitivityAnalysis,
    MonteCarloSensitivityAnalysis,
    SobolSequenceSensitivityAnalysis,
    HaltonSequenceSensitivityAnalysis,
)
from pyapprox.typing.sensitivity.variance_based.bin_based import (
    BinBasedSensitivityAnalysis,
)

__all__ = [
    "VarianceBasedSensitivityAnalysis",
    "PolynomialChaosSensitivityAnalysis",
    "SparseGridSensitivityAnalysis",
    "SampleBasedSensitivityAnalysis",
    "MonteCarloSensitivityAnalysis",
    "SobolSequenceSensitivityAnalysis",
    "HaltonSequenceSensitivityAnalysis",
    "BinBasedSensitivityAnalysis",
]
