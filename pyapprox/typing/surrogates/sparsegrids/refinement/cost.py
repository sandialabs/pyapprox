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
        return float(self._bkd.sum(subspace_index).item()) + 1.0

    def __repr__(self) -> str:
        return "LevelCostFunction()"


class ConfigIndexCostFunction(Generic[Array]):
    """Cost function based on configuration/fidelity index only.

    Computes cost as base^(sum(config_levels)) for multi-fidelity models
    where higher config indices mean more expensive evaluations.

    Only the config part of the subspace index is used for cost calculation.
    The physical variable levels are ignored since they don't affect
    model evaluation cost.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars_physical : int
        Number of physical variables. These dimensions are skipped when
        computing cost from the subspace index.
    base : float, optional
        Exponential base for cost scaling. Default: 2.0.
        Cost = base^(sum of config levels).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> cost_fn = ConfigIndexCostFunction(bkd, nvars_physical=2, base=2.0)
    >>> # Subspace index: [phys1_level, phys2_level, config1_level]
    >>> index = bkd.asarray([3, 2, 1])  # config_level = 1
    >>> cost_fn(index)
    2.0
    >>> index = bkd.asarray([0, 0, 3])  # config_level = 3
    >>> cost_fn(index)
    8.0
    """

    def __init__(
        self, bkd: Backend[Array], nvars_physical: int, base: float = 2.0
    ) -> None:
        if nvars_physical < 0:
            raise ValueError(
                f"nvars_physical must be non-negative, got {nvars_physical}"
            )
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self._bkd = bkd
        self._nvars_physical = nvars_physical
        self._base = base

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars_physical(self) -> int:
        """Return number of physical variables."""
        return self._nvars_physical

    def base(self) -> float:
        """Return exponential base for cost scaling."""
        return self._base

    def __call__(self, subspace_index: Array) -> float:
        """Compute cost based on config part of subspace index.

        Parameters
        ----------
        subspace_index : Array
            Multi-index of the subspace. Shape: (nvars_physical + nconfig_vars,).
            The first nvars_physical elements are physical variable levels,
            the remaining are config levels.

        Returns
        -------
        float
            Cost = base^(sum of config levels). Always >= 1.0.
        """
        config_levels = subspace_index[self._nvars_physical:]
        config_sum = float(self._bkd.sum(config_levels).item())
        return self._base**config_sum

    def __repr__(self) -> str:
        return (
            f"ConfigIndexCostFunction(nvars_physical={self._nvars_physical}, "
            f"base={self._base})"
        )
