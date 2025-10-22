"""The :mod:`pyapprox.interface` module implements a number of tools for
interfacing with numerical models
"""

from pyapprox.interface.model import (
    ModelFromVectorizedCallable,
    ModelFromSingleSampleCallable,
    UmbridgeModelWrapper,
    UmbridgeIOModelWrapper,
    UmbridgeIOModelEnsembleWrapper,
    SerialIOModel,
    AsyncIOModel,
)
from pyapprox.interface.wrappers import (
    ScipyModelWrapper,
    create_active_set_variable_model,
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
    "create_active_set_variable_model",
    "ChangeModelSignWrapper",
    "PoolModelWrapper",
]
