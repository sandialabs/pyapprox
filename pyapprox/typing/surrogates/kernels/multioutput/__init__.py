"""
Multi-output kernels for modeling covariance across multiple outputs.

This module provides kernel implementations for multi-output Gaussian processes
and other multi-output surrogate models.

Key Protocols
-------------
- MultiOutputKernelProtocol: Protocol for multi-output kernel implementations
- ScalingFunctionProtocol: Protocol for spatially varying scaling functions

Kernel Implementations
---------------------
- IndependentMultiOutputKernel: Independent kernels per output (block-diagonal)
- LinearCoregionalizationKernel: Linear model of coregionalization (LMC)
- MultiLevelKernel: Sequential autoregressive multi-level kernel
- DAGMultiOutputKernel: General DAG-based autoregressive kernel

Supporting Classes
------------------
- PolynomialScaling: Polynomial scaling functions ρ(x) = c0 + c1*x1 + ... + cd*xd
- CovarianceHyperParameter: Learnable covariance matrices via hyperspherical parameterization

Examples
--------
Basic multi-output kernel:

>>> from pyapprox.typing.surrogates.kernels import MaternKernel
>>> from pyapprox.typing.surrogates.kernels.multioutput import IndependentMultiOutputKernel
>>> from pyapprox.typing.util.backends.numpy import NumpyBkd
>>> import numpy as np
>>> bkd = NumpyBkd()
>>> # Create independent kernels for 2 outputs
>>> kernel1 = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
>>> kernel2 = MaternKernel(1.5, [0.5, 0.5], (0.1, 10.0), 2, bkd)
>>> mo_kernel = IndependentMultiOutputKernel([kernel1, kernel2])

DAG-based autoregressive kernel:

>>> import networkx as nx
>>> from pyapprox.typing.surrogates.kernels import PolynomialScaling
>>> from pyapprox.typing.surrogates.kernels.multioutput import DAGMultiOutputKernel
>>> # Define structure: 0 -> 1 -> 2 <- 3 (diamond)
>>> dag = nx.DiGraph()
>>> dag.add_nodes_from([0, 1, 2, 3])
>>> dag.add_edges_from([(0, 1), (1, 2), (3, 2)])
>>> # Create kernels and scalings...
>>> kernel = DAGMultiOutputKernel(dag, discrepancy_kernels, edge_scalings)
"""

from .protocols import MultiOutputKernelProtocol
from .independent import IndependentMultiOutputKernel
from .linear_coregionalization import LinearCoregionalizationKernel
from .multilevel import MultiLevelKernel
from ..scalings import (
    ScalingFunctionProtocol,
    PolynomialScaling,
)
from .covariance_hyperparameter import CovarianceHyperParameter
from .dag_kernel import DAGMultiOutputKernel

__all__ = [
    # Protocols
    "MultiOutputKernelProtocol",
    "ScalingFunctionProtocol",
    # Kernel implementations
    "IndependentMultiOutputKernel",
    "LinearCoregionalizationKernel",
    "MultiLevelKernel",
    "DAGMultiOutputKernel",
    # Supporting classes
    "PolynomialScaling",
    "CovarianceHyperParameter",
]
