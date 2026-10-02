"""Evaluating observations and targets at the same samples.

A joint evaluator returns noise-free observations and one or more target
blocks from one set of input samples, so their joint moments can be
estimated. ``SeparateFunctions`` uses one function per block;
``SplitFunction`` runs one function once and splits its output rows;
``InputTarget`` makes part of the input itself a target.
"""

from .evaluators import SeparateFunctions, SplitFunction
from .input_target import InputTarget
from .outputs import JointOutputs
from .protocols import JointEvaluatorProtocol

__all__ = [
    "InputTarget",
    "JointEvaluatorProtocol",
    "JointOutputs",
    "SeparateFunctions",
    "SplitFunction",
]
