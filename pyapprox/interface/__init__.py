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
    IOModel,
    ActiveSetVariableModel,
    ChangeModelSignWrapper,
    PoolModelWrapper,
)
from pyapprox.interface.async_model import AsyncModel
from pyapprox.interface.file_io_model import FileIOModel

__all__ = [
    "ModelFromVectorizedCallable",
    "ModelFromSingleSampleCallable",
    "ScipyModelWrapper",
    "UmbridgeModelWrapper",
    "UmbridgeIOModelWrapper",
    "UmbridgeIOModelEnsembleWrapper",
    "IOModel",
    "ActiveSetVariableModel",
    "ChangeModelSignWrapper",
    "PoolModelWrapper",
    "AsyncModel",
    "FileIOModel",
]
