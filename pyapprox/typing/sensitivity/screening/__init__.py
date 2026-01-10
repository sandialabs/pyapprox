"""Screening methods for sensitivity analysis.

This submodule provides screening methods that efficiently identify
important parameters without full variance decomposition:
- Morris method (elementary effects)
"""

from pyapprox.typing.sensitivity.screening.morris import MorrisSensitivityAnalysis

__all__ = [
    "MorrisSensitivityAnalysis",
]
