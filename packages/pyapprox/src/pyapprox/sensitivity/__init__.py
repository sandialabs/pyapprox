"""Sensitivity analysis module for PyApprox.

This module provides variance-based sensitivity analysis methods including:
- PCE-based sensitivity analysis
- Sparse grid sensitivity analysis
- Gaussian process sensitivity analysis
- Sample-based (Monte Carlo) sensitivity analysis
- Morris screening
"""

from pyapprox.sensitivity.plots import (
    plot_interaction_values,
    plot_main_effects,
    plot_morris_screening,
    plot_sensitivity_indices_with_confidence_intervals,
    plot_sensitivity_summary,
    plot_total_effects,
)
from pyapprox.sensitivity.protocols import (
    SensitivityAnalysisProtocol,
    SensitivityAnalysisWithMomentsProtocol,
    SensitivityAnalysisWithSobolIndicesProtocol,
)
from pyapprox.sensitivity.screening import MorrisSensitivityAnalysis
from pyapprox.sensitivity.variance_based import (
    BinBasedSensitivityAnalysis,
    HaltonSequenceSensitivityAnalysis,
    MonteCarloSensitivityAnalysis,
    PolynomialChaosSensitivityAnalysis,
    SampleBasedSensitivityAnalysis,
    SobolSequenceSensitivityAnalysis,
    SparseGridSensitivityAnalysis,
    VarianceBasedSensitivityAnalysis,
)

__all__ = [
    # Protocols
    "SensitivityAnalysisProtocol",
    "SensitivityAnalysisWithSobolIndicesProtocol",
    "SensitivityAnalysisWithMomentsProtocol",
    # Base classes
    "VarianceBasedSensitivityAnalysis",
    # Surrogate-based implementations
    "PolynomialChaosSensitivityAnalysis",
    "SparseGridSensitivityAnalysis",
    # Sample-based implementations
    "SampleBasedSensitivityAnalysis",
    "MonteCarloSensitivityAnalysis",
    "SobolSequenceSensitivityAnalysis",
    "HaltonSequenceSensitivityAnalysis",
    # Bin-based implementations
    "BinBasedSensitivityAnalysis",
    # Screening methods
    "MorrisSensitivityAnalysis",
    # Plotting functions
    "plot_main_effects",
    "plot_total_effects",
    "plot_interaction_values",
    "plot_sensitivity_indices_with_confidence_intervals",
    "plot_morris_screening",
    "plot_sensitivity_summary",
]
