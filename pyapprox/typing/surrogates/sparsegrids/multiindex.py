"""Multi-index adaptive sparse grid.

This module provides sparse grid construction that supports multi-fidelity
models through refinement variables. The sparse grid is built over both
the physical variables and fidelity/refinement indices.
"""

from typing import Generic, List, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
)
from pyapprox.typing.surrogates.affine.indices import (
    AdmissibilityCriteria,
    Max1DLevelsCriteria,
    LinearGrowthRule,
)
from pyapprox.typing.surrogates.affine.univariate import LagrangeBasis1D

from .adaptive import AdaptiveCombinationSparseGrid
from .basis_factory import BasisFactoryProtocol, PrebuiltBasisFactory
from .refinement.protocols import SparseGridRefinementCriteriaProtocol


class MultiIndexAdaptiveCombinationSparseGrid(
    AdaptiveCombinationSparseGrid[Array], Generic[Array]
):
    """Multi-fidelity adaptive sparse grid.

    This sparse grid supports multi-fidelity models by extending the
    sparse grid index set to include refinement variables. The first
    `nvars` dimensions correspond to physical input variables, and the
    remaining `nrefinement_vars` dimensions correspond to model fidelity
    levels.

    NOTE: The same growth rule is applied to all dimensions (physical and
    refinement). For refinement variables, integer indices 0, 1, 2, ... are
    used as sample points. The refinement_bounds controls the maximum level
    allowed for each refinement dimension.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    physical_basis_factories : List[BasisFactoryProtocol[Array]]
        Factories for creating univariate bases for physical input variables.
    nrefinement_vars : int
        Number of refinement/fidelity variables.
    refinement_bounds : Array
        Maximum level for each refinement variable. Shape: (nrefinement_vars,).
        Note: This should be at least as large as the growth_rule output at
        the desired max sparse grid level. For LinearGrowthRule(scale=2, shift=1)
        at level k, you need refinement_bounds >= 2*k+1.
    growth_rule : IndexGrowthRuleProtocol, optional
        Rule mapping level to number of points.
        Default: LinearGrowthRule(scale=2, shift=1).
    admissibility : AdmissibilityCriteria[Array], optional
        Criteria for admissible subspace indices.
        If None, uses Max1DLevelsCriteria with refinement_bounds.
    refinement_priority : SparseGridRefinementCriteriaProtocol[Array], optional
        Criteria for computing refinement priorities. Default: L2NormRefinementCriteria.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> from pyapprox.typing.surrogates.sparsegrids import PrebuiltBasisFactory
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>>
    >>> # 2 physical variables with Legendre bases
    >>> physical_bases = [LegendrePolynomial1D(bkd) for _ in range(2)]
    >>> physical_factories = [PrebuiltBasisFactory(b) for b in physical_bases]
    >>>
    >>> # 1 fidelity variable - need at least 5 levels for level 2 sparse grid
    >>> # with LinearGrowthRule(scale=2, shift=1): 2*2+1 = 5 points
    >>> refinement_bounds = bkd.asarray([5])
    >>> grid = MultiIndexAdaptiveCombinationSparseGrid(
    ...     bkd, physical_factories, nrefinement_vars=1,
    ...     refinement_bounds=refinement_bounds
    ... )
    >>>
    >>> # Samples have shape (nvars + nrefinement_vars, nsamples)
    >>> samples = grid.step_samples()
    >>> physical_samples = samples[:2, :]  # Shape: (2, nsamples)
    >>> fidelity_indices = samples[2:, :]  # Shape: (1, nsamples) - integers
    """

    def __init__(
        self,
        bkd: Backend[Array],
        physical_basis_factories: List[BasisFactoryProtocol[Array]],
        nrefinement_vars: int,
        refinement_bounds: Array,
        growth_rule: Optional[IndexGrowthRuleProtocol] = None,
        admissibility: Optional[AdmissibilityCriteria[Array]] = None,
        refinement_priority: Optional[
            SparseGridRefinementCriteriaProtocol[Array]
        ] = None,
    ):
        self._nvars_physical = len(physical_basis_factories)
        self._nrefinement_vars = nrefinement_vars
        self._refinement_bounds = bkd.copy(refinement_bounds)

        if refinement_bounds.shape[0] != nrefinement_vars:
            raise ValueError(
                f"refinement_bounds must have length {nrefinement_vars}"
            )

        # Default growth rule
        if growth_rule is None:
            growth_rule = LinearGrowthRule(scale=2, shift=1)

        # For refinement variables, create discrete level bases wrapped in factories
        refinement_factories: List[BasisFactoryProtocol[Array]] = []
        for dim in range(nrefinement_vars):
            max_level = int(refinement_bounds[dim])
            refinement_basis = self._create_discrete_level_basis(bkd, max_level)
            refinement_factories.append(PrebuiltBasisFactory(refinement_basis))

        # Combine all factories
        all_factories = list(physical_basis_factories) + refinement_factories

        # Default admissibility with refinement bounds
        if admissibility is None:
            # No limit on physical vars, bounded on refinement vars
            # Use a large value for "unbounded" since inf doesn't work with
            # integer comparisons in Max1DLevelsCriteria
            max_levels = bkd.hstack(
                [
                    bkd.full((self._nvars_physical,), 1000.0),
                    bkd.flatten(refinement_bounds),
                ]
            )
            admissibility = Max1DLevelsCriteria(max_levels, bkd)

        super().__init__(
            bkd,
            all_factories,
            growth_rule,
            admissibility,
            refinement_priority,
        )

    def _create_discrete_level_basis(
        self, bkd: Backend[Array], max_level: int
    ) -> LagrangeBasis1D[Array]:
        """Create a basis for discrete level indices.

        For refinement variables, we use integer indices as sample points.
        Level k gives points 0, 1, ..., k (i.e., k+1 points).
        This is a linear growth: npoints = level + 1.
        """

        def level_quadrature_rule(npoints: int) -> Tuple[Array, Array]:
            """Return level indices as sample points."""
            samples = bkd.arange(npoints)[None, :].astype(bkd.double_dtype())
            weights = bkd.ones((npoints, 1))
            return samples, weights

        return LagrangeBasis1D(bkd, level_quadrature_rule)

    def nvars_physical(self) -> int:
        """Return number of physical input variables."""
        return self._nvars_physical

    def nrefinement_vars(self) -> int:
        """Return number of refinement variables."""
        return self._nrefinement_vars

    def get_refinement_bounds(self) -> Array:
        """Return refinement variable bounds."""
        return self._bkd.copy(self._refinement_bounds)

    def split_samples(self, samples: Array) -> Tuple[Array, Array]:
        """Split samples into physical and refinement components.

        Parameters
        ----------
        samples : Array
            Full sample array. Shape: (nvars + nrefinement_vars, nsamples)

        Returns
        -------
        physical_samples : Array
            Physical variable samples. Shape: (nvars, nsamples)
        refinement_indices : Array
            Refinement indices. Shape: (nrefinement_vars, nsamples)
        """
        return (
            samples[: self._nvars_physical, :],
            samples[self._nvars_physical :, :],
        )

    def __repr__(self) -> str:
        return (
            f"MultiIndexAdaptiveCombinationSparseGrid("
            f"nvars_physical={self._nvars_physical}, "
            f"nrefinement_vars={self._nrefinement_vars}, "
            f"nsubspaces={self.nsubspaces()}, "
            f"nselected={self._index_gen.nselected_indices()}, "
            f"ncandidates={self._index_gen.ncandidate_indices()})"
        )
