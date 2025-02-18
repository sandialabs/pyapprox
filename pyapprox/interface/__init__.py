"""The :mod:`pyapprox.interface` module implements a number of tools for
interfacing with numerical models
"""

from pyapprox.interface.model import (
    ModelFromVectorizedCallable,
    ModelFromSingleSampleCallable,
    ScipyModelWrapper,
    UmbridgeModelWrapper,
    UmbridgeIOModelWrapper,
    UmbridgeIOModelEnsembleWrapper,
    SerialIOModel,
    AsyncIOModel,
    ActiveSetVariableModel,
    ChangeModelSignWrapper,
    PoolModelWrapper,
)

__all__ = [
    "ModelFromVectorizedCallable",
    "ModelFromSingleSampleCallable",
    "ScipyModelWrapper",
    "UmbridgeModelWrapper",
    "UmbridgeIOModelWrapper",
    "UmbridgeIOModelEnsembleWrapper",
    "SerialIOModel",
    "AsyncIOModel",
    "ActiveSetVariableModel",
    "ChangeModelSignWrapper",
    "PoolModelWrapper",
]
