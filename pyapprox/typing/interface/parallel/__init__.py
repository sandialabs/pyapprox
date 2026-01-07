"""Parallel execution capabilities for pyapprox.typing.

This module provides:
- ParallelConfig: Configuration for parallel execution
- make_parallel: Factory to wrap functions with parallel batch methods
- ParallelFunctionWrapper: Wrapper class for parallel batch methods
- ParallelJacobianMixin: Mixin for adding parallel jacobian_batch
- ParallelHessianMixin: Mixin for adding parallel hessian_batch
- ParallelHVPMixin: Mixin for adding parallel hvp_batch
- ParallelWHVPMixin: Mixin for adding parallel whvp_batch
- JoblibBackend: Parallel backend using joblib
- MpireBackend: Parallel backend using mpire
- SequentialBackend: Non-parallel backend for debugging
- ParallelBackendProtocol: Protocol for parallel backends
- BatchSplitter: Utilities for splitting/combining batches
- TensorTransfer: Utilities for numpy/tensor conversion

Examples
--------
>>> from pyapprox.typing.interface.parallel import make_parallel
>>> # Wrap a function with parallel support
>>> parallel_func = make_parallel(my_func, backend="joblib", n_jobs=4)
>>> jacobians = parallel_func.jacobian_batch(samples)

>>> # Or use mixins for class-based control
>>> from pyapprox.typing.interface.parallel import (
...     ParallelJacobianMixin,
...     ParallelConfig,
... )
>>> class MyFunction(ParallelJacobianMixin):
...     def jacobian(self, sample):
...         ...  # single sample implementation
>>> func = MyFunction()
>>> func.set_parallel_config(ParallelConfig(backend="mpire", n_jobs=4))
>>> jacobians = func.jacobian_batch(samples)
"""

from pyapprox.typing.interface.parallel.batch_utils import BatchSplitter
from pyapprox.typing.interface.parallel.config import (
    ParallelConfig,
    SequentialBackend,
)
from pyapprox.typing.interface.parallel.factory import (
    ParallelFunctionWrapper,
    make_parallel,
)
from pyapprox.typing.interface.parallel.function_protocols import (
    ParallelFunctionProtocol,
    ParallelFunctionWithHVPProtocol,
    ParallelFunctionWithJacobianProtocol,
    ParallelFunctionWithWHVPProtocol,
)
from pyapprox.typing.interface.parallel.joblib_backend import JoblibBackend
from pyapprox.typing.interface.parallel.mixins import (
    ParallelHessianMixin,
    ParallelHVPMixin,
    ParallelJacobianMixin,
    ParallelWHVPMixin,
)
from pyapprox.typing.interface.parallel.mpire_backend import MpireBackend
from pyapprox.typing.interface.parallel.protocols import (
    ParallelBackendProtocol,
)
from pyapprox.typing.interface.parallel.tensor_utils import TensorTransfer

__all__ = [
    # Configuration
    "ParallelConfig",
    # Factory
    "make_parallel",
    "ParallelFunctionWrapper",
    # Mixins
    "ParallelJacobianMixin",
    "ParallelHessianMixin",
    "ParallelHVPMixin",
    "ParallelWHVPMixin",
    # Backends
    "JoblibBackend",
    "MpireBackend",
    "SequentialBackend",
    # Protocols
    "ParallelBackendProtocol",
    "ParallelFunctionProtocol",
    "ParallelFunctionWithJacobianProtocol",
    "ParallelFunctionWithHVPProtocol",
    "ParallelFunctionWithWHVPProtocol",
    # Utilities
    "BatchSplitter",
    "TensorTransfer",
]
