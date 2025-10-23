"""The :mod:`pyapprox.interface` module implements a number of tools for
interfacing with numerical models
"""

from pyapprox.interface.model import (
    ModelWorkTracker,
    ModelDataBase,
    Model,
    ModelFromVectorizedCallable,
    ModelFromSingleSampleCallable,
    UmbridgeModel,
    UmbridgeIOModel,
    UmbridgeIOModelEnsemble,
    SerialIOModel,
    AsyncIOModel,
)
from pyapprox.interface.wrappers import (
    ScipyModelWrapper,
    create_active_set_variable_model,
    create_pool_model,
    ChangeModelSignWrapper,
)

__all__ = [
    "ModelWorkTracker",
    "ModelDataBase",
    "Model",
    "ModelFromVectorizedCallable",
    "ModelFromSingleSampleCallable",
    "ScipyModelWrapper",
    "UmbridgeModel",
    "UmbridgeIOModel",
    "UmbridgeIOModelEnsemble",
    "SerialIOModel",
    "AsyncIOModel",
    "create_active_set_variable_model",
    "ChangeModelSignWrapper",
    "create_pool_model",
]
