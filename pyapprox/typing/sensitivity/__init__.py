"""Sensitivity analysis module for PyApprox.

This module provides variance-based sensitivity analysis methods including:
- PCE-based sensitivity analysis
- Sparse grid sensitivity analysis
- Gaussian process sensitivity analysis
- Sample-based (Monte Carlo) sensitivity analysis
- Morris screening
"""

from pyapprox.typing.sensitivity.protocols import (
    SensitivityAnalysisProtocol,
    SensitivityAnalysisWithSobolIndicesProtocol,
    SensitivityAnalysisWithMomentsProtocol,
)
from pyapprox.typing.sensitivity.variance_based import (
    VarianceBasedSensitivityAnalysis,
    PolynomialChaosSensitivityAnalysis,
    SparseGridSensitivityAnalysis,
    SampleBasedSensitivityAnalysis,
    MonteCarloSensitivityAnalysis,
    SobolSequenceSensitivityAnalysis,
    HaltonSequenceSensitivityAnalysis,
)
from pyapprox.typing.sensitivity.screening import MorrisSensitivityAnalysis
from pyapprox.typing.sensitivity.plots import (
    plot_main_effects,
    plot_total_effects,
    plot_interaction_values,
    plot_sensitivity_indices_with_confidence_intervals,
    plot_morris_screening,
    plot_sensitivity_summary,
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
