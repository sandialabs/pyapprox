"""The :mod:`pyapprox.interface` module implements a number of tools for
interfacing with numerical models
"""

from pyapprox.interface.wrappers import (
    evaluate_1darray_function_on_2d_array, WorkTrackingModel,
    TimerModel, ModelEnsemble, PoolModel, DataFunctionModel, MultiIndexModel
)
from pyapprox.interface.async_model import AynchModel
from pyapprox.interface.file_io_model import FileIOModel

__all__ = ["evaluate_1darray_function_on_2d_array", "WorkTrackingModel",
           "TimerModel", "ModelEnsemble", "PoolModel",
           "AynchModel", "FileIOModel", "DataFunctionModel", "MultiIndexModel"]
