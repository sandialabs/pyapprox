"""Each sparse-grid fitter returns the result class matching its surrogate.

Tests run on both NumPy and PyTorch backends.
"""

from pyapprox.interface.functions.fromcallable.function import FunctionFromCallable
from pyapprox.surrogates.affine.indices.admissibility import MaxLevelCriteria
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_1d import (
    HierarchicalBasis1D,
)
from pyapprox.surrogates.sparsegrids.fit_result import (
    HierarchicalSparseGridFitResult,
)
from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_fitter import (
    SingleFidelityHierarchicalFitter,
)
from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_surrogate import (
    HierarchicalSurrogate,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestHierarchicalFitResult:
    def test_result_carries_hierarchical_surrogate(self, bkd: Backend[Array]) -> None:
        bases_1d = [HierarchicalBasis1D(bkd, p_max=1, boundary_mode="include")]
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityHierarchicalFitter(bkd, bases_1d, admis)

        def linear(samples: Array) -> Array:
            return 3.0 * samples + 1.0

        result = fitter.refine_to_tolerance(
            FunctionFromCallable(1, 1, linear, bkd), tol=1e-15, max_steps=50
        )
        assert isinstance(result, HierarchicalSparseGridFitResult)
        assert isinstance(result.surrogate, HierarchicalSurrogate)
        # One row per refinement variable, one column per promoted subspace.
        assert result.indices.shape[0] == 1
        assert result.indices.shape[1] > 0
        assert result.nsamples > 0
