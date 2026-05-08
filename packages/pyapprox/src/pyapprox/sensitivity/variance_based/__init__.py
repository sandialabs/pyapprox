"""Variance-based sensitivity analysis methods.

This submodule provides implementations of variance-based (Sobol) sensitivity
analysis using different surrogate models:
- PCE (Polynomial Chaos Expansion)
- Sparse grids
- Gaussian processes
- Sample-based (Monte Carlo)
"""

from pyapprox.sensitivity.variance_based.base import (
    VarianceBasedSensitivityAnalysis,
)
from pyapprox.sensitivity.variance_based.bin_based import (
    BinBasedSensitivityAnalysis,
)
from pyapprox.sensitivity.variance_based.pce import (
    PolynomialChaosSensitivityAnalysis,
)
from pyapprox.sensitivity.variance_based.sample_based import (
    HaltonSequenceSensitivityAnalysis,
    MonteCarloSensitivityAnalysis,
    SampleBasedSensitivityAnalysis,
    SobolSequenceSensitivityAnalysis,
)
from pyapprox.sensitivity.variance_based.sparsegrid import (
    SparseGridSensitivityAnalysis,
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
