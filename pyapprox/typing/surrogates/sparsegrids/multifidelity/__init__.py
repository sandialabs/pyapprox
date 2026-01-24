"""Multi-fidelity sparse grid module.

This module provides sparse grid surrogates for multi-fidelity models,
where configuration/fidelity variables select which model resolution to use.

Classes:
    MultiIndexAdaptiveCombinationSparseGrid: Multi-fidelity adaptive sparse grid
        with physical-only subspaces

Protocols:
    MultiFidelityModelFactoryProtocol: Factory for creating models per config

Usage:
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.sparsegrids.multifidelity import (
    ...     MultiIndexAdaptiveCombinationSparseGrid,
    ...     MultiFidelityModelFactoryProtocol,
    ... )
    >>> from pyapprox.typing.surrogates.sparsegrids import PrebuiltBasisFactory
    >>> bkd = NumpyBkd()
    >>> # Create grid with model factory for auto mode
    >>> grid = MultiIndexAdaptiveCombinationSparseGrid(
    ...     bkd, physical_basis_factories, nconfig_vars=1,
    ...     config_bounds=config_bounds, nqoi=1,
    ...     model_factory=my_factory
    ... )
    >>> # Refine until convergence
    >>> while grid.step():
    ...     if grid.error_estimate() < 1e-6:
    ...         break
"""

from pyapprox.typing.surrogates.sparsegrids.multifidelity.protocols import (
    MultiFidelityModelFactoryProtocol,
)
from pyapprox.typing.surrogates.sparsegrids.multifidelity.grid import (
    MultiIndexAdaptiveCombinationSparseGrid,
)

__all__ = [
    "MultiFidelityModelFactoryProtocol",
    "MultiIndexAdaptiveCombinationSparseGrid",
]
