"""
Deviation measures for prediction OED.

Deviation measures quantify the spread or uncertainty of QoI predictions
conditional on observed data. They are used in the prediction OED objective.
"""

from .base import DeviationMeasure
from .standard_deviation import StandardDeviationMeasure
from .entropic import EntropicDeviationMeasure
from .avar import AVaRDeviationMeasure

__all__ = [
    "DeviationMeasure",
    "StandardDeviationMeasure",
    "EntropicDeviationMeasure",
    "AVaRDeviationMeasure",
]
