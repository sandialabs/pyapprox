"""
Deviation measures for prediction OED.

Deviation measures quantify the spread or uncertainty of QoI predictions
conditional on observed data. They are used in the prediction OED objective.
"""

from .avar import AVaRDeviationMeasure
from .base import DeviationMeasure
from .entropic import EntropicDeviationMeasure
from .standard_deviation import StandardDeviationMeasure

__all__ = [
    "DeviationMeasure",
    "StandardDeviationMeasure",
    "EntropicDeviationMeasure",
    "AVaRDeviationMeasure",
]
