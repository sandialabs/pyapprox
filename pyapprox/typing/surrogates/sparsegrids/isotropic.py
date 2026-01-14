"""Isotropic combination sparse grid.

Pre-computed sparse grid with fixed level in all dimensions.
Uses HyperbolicIndexGenerator with pnorm=1.0 for index generation.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    Basis1DProtocol,
    IndexGrowthRuleProtocol,
)
from pyapprox.typing.surrogates.affine.indices import HyperbolicIndexGenerator

from .combination import CombinationSparseGrid


class IsotropicCombinationSparseGrid(CombinationSparseGrid[Array], Generic[Array]):
    """Isotropic sparse grid with fixed level.

    Creates a sparse grid using all subspace indices k with |k|_1 <= level.
    Uses HyperbolicIndexGenerator with pnorm=1.0 (total degree) for index
    generation.

    This class creates a fixed sparse grid at construction time. For
    incremental refinement, use AdaptiveCombinationSparseGrid instead.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    univariate_bases : List[Basis1DProtocol[Array]]
        Univariate bases for each dimension.
    growth_rule : IndexGrowthRuleProtocol
        Rule mapping level to number of points.
    level : int
        Maximum level (L1 norm bound for subspace indices).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule
    >>> bkd = NumpyBkd()
    >>> bases = [LegendrePolynomial1D(bkd) for _ in range(2)]
    >>> growth = LinearGrowthRule()
    >>> grid = IsotropicCombinationSparseGrid(bkd, bases, growth, level=3)
    >>> print(f"Subspaces: {grid.nsubspaces()}, Samples: {grid.nsamples()}")
    """

    def __init__(
        self,
        bkd: Backend[Array],
        univariate_bases: List[Basis1DProtocol[Array]],
        growth_rule: IndexGrowthRuleProtocol,
        level: int,
    ):
        super().__init__(bkd, univariate_bases, growth_rule)
        self._level = level

        # Create index generator with pnorm=1.0 for isotropic (total degree)
        self._index_gen = HyperbolicIndexGenerator(
            nvars=self._nvars,
            max_level=level,
            pnorm=1.0,
            bkd=bkd,
        )

        # Add all subspaces from generator
        indices = self._index_gen.get_selected_indices()
        for j in range(indices.shape[1]):
            self._add_subspace(indices[:, j])

        # Precompute Smolyak coefficients
        self._update_smolyak_coefficients()

        # Collect unique samples
        self._collect_unique_samples()

    def level(self) -> int:
        """Return the sparse grid level."""
        return self._level

    def __repr__(self) -> str:
        return (
            f"IsotropicCombinationSparseGrid(nvars={self._nvars}, "
            f"level={self._level}, nsubspaces={self.nsubspaces()}, "
            f"nsamples={self.nsamples()})"
        )
