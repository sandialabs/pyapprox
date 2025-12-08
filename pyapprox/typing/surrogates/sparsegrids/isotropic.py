"""Isotropic combination sparse grid.

Pre-computed sparse grid with fixed level in all dimensions.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    Basis1DProtocol,
    IndexGrowthRuleProtocol,
)

from .combination import CombinationSparseGrid


def _generate_isotropic_indices(
    nvars: int,
    level: int,
    bkd: Backend,
) -> Array:
    """Generate indices for isotropic sparse grid.

    Returns all multi-indices k with |k|_1 <= level.

    Parameters
    ----------
    nvars : int
        Number of variables
    level : int
        Maximum level (L1 norm bound)
    bkd : Backend
        Computational backend

    Returns
    -------
    Array
        Multi-indices of shape (nvars, nindices)
    """
    indices_list = []

    def recurse(current: List[int], remaining: int, dim: int):
        if dim == nvars - 1:
            # Last dimension: use all remaining
            for val in range(remaining + 1):
                indices_list.append(current + [val])
        else:
            for val in range(remaining + 1):
                recurse(current + [val], remaining - val, dim + 1)

    recurse([], level, 0)

    # Convert to array
    nindices = len(indices_list)
    indices = bkd.zeros((nvars, nindices), dtype=bkd.int64_dtype())
    for j, idx in enumerate(indices_list):
        for i in range(nvars):
            indices[i, j] = idx[i]

    return indices


class IsotropicCombinationSparseGrid(CombinationSparseGrid[Array], Generic[Array]):
    """Isotropic sparse grid with fixed level.

    Creates a sparse grid using all subspace indices k with |k|_1 <= level.

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

        # Generate and add all subspace indices
        indices = _generate_isotropic_indices(
            self._nvars, level, bkd
        )

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
