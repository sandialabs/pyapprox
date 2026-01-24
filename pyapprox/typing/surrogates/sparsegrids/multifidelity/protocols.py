"""Protocols for multi-fidelity sparse grids.

This module defines protocols for model factories used in multi-fidelity
sparse grid construction.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols import FunctionProtocol


@runtime_checkable
class MultiFidelityModelFactoryProtocol(Protocol, Generic[Array]):
    """Factory that returns a model for a given configuration index.

    Configuration indices are tuples of integers that select model fidelity.
    For example, in a PDE context:
    - (0,) might select a coarse mesh
    - (3,) might select a fine mesh
    - (2, 1) might select x-mesh=2, y-mesh=1

    The sparse grid passes config values as floats internally (to keep samples
    as uniform float arrays). This protocol expects integer tuples since the
    caller (MultiFidelitySparseGrid) converts floats to ints before calling.

    Examples
    --------
    >>> class MyModelFactory:
    ...     def __init__(self, bkd):
    ...         self._bkd = bkd
    ...         self._models = {}
    ...
    ...     def get_model(self, config_index):
    ...         if config_index not in self._models:
    ...             # Create model for this fidelity level
    ...             self._models[config_index] = create_model(config_index)
    ...         return self._models[config_index]
    ...
    ...     def nconfig_vars(self):
    ...         return 1
    ...
    ...     def bkd(self):
    ...         return self._bkd
    """

    def get_model(
        self, config_index: Tuple[int, ...]
    ) -> FunctionProtocol[Array]:
        """Return the model for a specific configuration.

        Parameters
        ----------
        config_index : Tuple[int, ...]
            Tuple of integers, one per config dimension.
            Values are non-negative integers representing fidelity levels.

        Returns
        -------
        FunctionProtocol[Array]
            A function satisfying FunctionProtocol that evaluates the model.
            The function should accept samples of shape (nvars_physical, nsamples)
            and return values of shape (nqoi, nsamples).
        """
        ...

    def nconfig_vars(self) -> int:
        """Return number of configuration variables.

        Returns
        -------
        int
            Number of config dimensions. Each dimension represents a
            fidelity parameter (e.g., mesh resolution, time step size).
        """
        ...

    def bkd(self) -> Backend[Array]:
        """Return the computational backend.

        Returns
        -------
        Backend[Array]
            The backend used for array operations.
        """
        ...
