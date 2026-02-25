"""Factory for creating tensor product subspaces.

Extracts subspace creation logic from the old CombinationSparseGrid class
into a standalone factory, enabling clean separation of build and evaluate.
"""

from typing import Generic, List, Protocol, Union, runtime_checkable

from pyapprox.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SubspaceFactoryProtocol(Protocol[Array]):
    """Protocol for creating tensor product subspaces from approx indices."""

    def __call__(self, approx_index: Array) -> TensorProductSubspace[Array]:
        """Create a subspace for the given multi-index.

        Parameters
        ----------
        approx_index : Array
            Multi-index for the subspace. For single-fidelity, shape
            (nvars_physical,). For multi-fidelity, shape
            (nvars_physical + nconfig_vars,) — only the first
            nvars_physical dimensions are used for basis construction.

        Returns
        -------
        TensorProductSubspace[Array]
            The newly created subspace.
        """
        ...

    def nvars_physical(self) -> int:
        """Return number of physical variables."""
        ...

    def is_nested(self) -> bool:
        """Return whether all basis factories produce nested rules."""
        ...

    def growth_rules(self) -> List[IndexGrowthRuleProtocol]:
        """Return the growth rules (one per physical dimension)."""
        ...


class TensorProductSubspaceFactory(Generic[Array]):
    """Create TensorProductSubspace instances from multi-indices.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis_factories : List[BasisFactoryProtocol[Array]]
        One factory per physical dimension.
    growth_rules : IndexGrowthRuleProtocol or List[IndexGrowthRuleProtocol]
        Growth rule(s) for mapping level to number of points.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis_factories: List[BasisFactoryProtocol[Array]],
        growth_rules: Union[IndexGrowthRuleProtocol, List[IndexGrowthRuleProtocol]],
    ) -> None:
        self._bkd = bkd
        self._basis_factories = basis_factories
        self._nvars_physical = len(basis_factories)

        # Normalize growth_rules to list
        if isinstance(growth_rules, list):
            self._growth_rules = growth_rules
        else:
            self._growth_rules = [growth_rules for _ in range(self._nvars_physical)]

        # Detect nested
        self._is_nested = all(
            getattr(f, "is_nested", lambda: False)() for f in basis_factories
        )

    def __call__(self, approx_index: Array) -> TensorProductSubspace[Array]:
        """Create a subspace using only the physical dimensions of the index.

        Parameters
        ----------
        approx_index : Array
            Full index, shape (nvars_physical,) or
            (nvars_physical + nconfig_vars,).

        Returns
        -------
        TensorProductSubspace[Array]
            The newly created subspace.
        """
        physical_index = approx_index[: self._nvars_physical]
        return TensorProductSubspace(
            self._bkd,
            physical_index,
            self._basis_factories,
            self._growth_rules,
        )

    def nvars_physical(self) -> int:
        """Return number of physical variables."""
        return self._nvars_physical

    def is_nested(self) -> bool:
        """Return whether all basis factories produce nested rules."""
        return self._is_nested

    def growth_rules(self) -> List[IndexGrowthRuleProtocol]:
        """Return the growth rules (one per physical dimension)."""
        return list(self._growth_rules)
