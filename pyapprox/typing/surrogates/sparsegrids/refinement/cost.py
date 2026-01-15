"""Cost functions for sparse grid refinement criteria.

This module provides cost functions that estimate the computational
cost of evaluating subspaces, used for cost-aware adaptive refinement.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class UnitCostFunction(Generic[Array]):
    """Unit cost function: all subspaces have cost 1.

    This is the default cost function, treating all subspaces equally
    regardless of their level or size.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> cost_fn = UnitCostFunction(bkd)
    >>> index = bkd.asarray([1, 2, 3])
    >>> cost_fn(index)
    1.0
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, subspace_index: Array) -> float:
        """Return unit cost for any subspace.

        Parameters
        ----------
        subspace_index : Array
            Multi-index of the subspace (ignored).

        Returns
        -------
        float
            Always returns 1.0.
        """
        return 1.0

    def __repr__(self) -> str:
        return "UnitCostFunction()"


class LevelCostFunction(Generic[Array]):
    """Cost proportional to L1 norm of subspace index.

    Cost = sum(index) + 1

    Higher-level subspaces are assigned higher costs, which biases
    refinement toward lower-level subspaces when combined with
    cost-normalized priorities.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> cost_fn = LevelCostFunction(bkd)
    >>> index = bkd.asarray([1, 2, 0])
    >>> cost_fn(index)
    4.0
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, subspace_index: Array) -> float:
        """Compute cost based on L1 norm of index.

        Parameters
        ----------
        subspace_index : Array
            Multi-index of the subspace. Shape: (nvars,)

        Returns
        -------
        float
            Cost = sum(index) + 1.
        """
        return float(self._bkd.sum(subspace_index)) + 1.0

    def __repr__(self) -> str:
        return "LevelCostFunction()"
