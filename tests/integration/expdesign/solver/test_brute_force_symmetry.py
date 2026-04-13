"""
Standalone tests for brute-force OED utility symmetry.

PERMANENT - no legacy imports.

Replicates legacy test_brute_force_d_optimal_oed from test_bayesoed.py:575-649.
Verifies that symmetric design configurations produce equal utility values
and that the optimal design is found correctly.
"""

from typing import Any, Generic, List, Tuple

import numpy as np
import pytest

from pyapprox_benchmarks.instances.oed.linear_gaussian import (
    LinearGaussianKLOEDBenchmark,
    build_linear_gaussian_kl_benchmark,
)
from pyapprox.expdesign.objective import (
    create_kl_oed_objective,
)
from pyapprox.expdesign.solver import BruteForceKLOEDSolver
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend

# =============================================================================
# Test Utilities
# =============================================================================


class LinearForwardModel(Generic[Array]):
    """Test utility: Wrap a design matrix as a FunctionProtocol.

    Forward model: y = A @ theta
    """

    def __init__(self, design_matrix: Array, bkd: Backend[Array]) -> None:
        self._design_matrix = design_matrix
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._design_matrix.shape[1]

    def nqoi(self) -> int:
        return self._design_matrix.shape[0]

    def __call__(self, samples: Array) -> Array:
        return self._bkd.dot(self._design_matrix, samples)


def create_kl_oed_objective_from_benchmark(
    benchmark: LinearGaussianKLOEDBenchmark[Array],
    bkd: Backend[Array],
    outer_sampler_type: str = "gauss",
    inner_sampler_type: str = "gauss",
    nouter_approx: int = 100000,
    ninner_approx: int = 1000,
    outer_seed: int | None = None,
    inner_seed: int | None = None,
) -> Any:
    """Test utility: Create KLOEDObjective from a LinearGaussianKLOEDBenchmark."""
    nobs = benchmark.problem().nobs()
    nparams = benchmark.problem().nparams()
    prior_std = np.sqrt(benchmark.prior_var())

    # Create prior distribution
    prior_marginals = [GaussianMarginal(0.0, prior_std, bkd) for _ in range(nparams)]
    prior = IndependentJoint(prior_marginals, bkd)

    # Wrap design matrix as forward model
    forward_model = LinearForwardModel(benchmark.design_matrix(), bkd)

    # Get noise variances
    noise_variances = bkd.asarray(np.full(nobs, benchmark.noise_var()))

    return create_kl_oed_objective(
        prior,
        forward_model,
        noise_variances,
        bkd,
        outer_sampler_type=outer_sampler_type,
        inner_sampler_type=inner_sampler_type,
        nouter_approx=nouter_approx,
        ninner_approx=ninner_approx,
        outer_seed=outer_seed,
        inner_seed=inner_seed,
    )


# =============================================================================
# Tests
# =============================================================================


class TestBruteForceSymmetryStandalone:
    """Standalone tests for brute-force OED utility symmetry.

    Replicates legacy test_brute_force_d_optimal_oed:
    - Verify symmetric design pairs have equal utility values
    - Verify optimal k-subset design matches expected
    """

    @pytest.mark.parametrize(
        "nobs,min_degree,degree,noise_std,prior_std,k,expected_optimal",
        [
            # Original legacy test case: nobs=5, degree=2, k=3
            # Optimal design for symmetric polynomial regression is [0, 2, 4]
            # (endpoints and center of [-1, 1])
            (5, 0, 2, 0.5, 0.5, 3, [0, 2, 4]),
        ],
    )
    def test_brute_force_utility_symmetry(
        self,
        bkd: Backend[Array],
        nobs: int,
        min_degree: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        k: int,
        expected_optimal: List[int],
    ) -> None:
        """Verify symmetric design pairs have equal utility values.

        For symmetric design locations in [-1, 1], pairs of designs that
        are mirror images should have equal utility (EIG) values.

        Replicates legacy test at test_bayesoed.py:625-647.
        """
        # Create benchmark
        benchmark = build_linear_gaussian_kl_benchmark(
            nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree
        )

        # Gauss quadrature: symmetry is a structural property that holds
        # regardless of quadrature accuracy, so modest sample counts suffice.
        nouter = 1000
        ninner = 100

        objective = create_kl_oed_objective_from_benchmark(
            benchmark,
            bkd,
            outer_sampler_type="gauss",
            inner_sampler_type="gauss",
            nouter_approx=nouter,
            ninner_approx=ninner,
        )

        # Solve with store_all=True to get all utility values
        solver = BruteForceKLOEDSolver(objective)
        _, _, optimal_indices = solver.solve(k, store_all=True)

        all_indices = solver.all_indices()
        all_eigs = solver.all_eigs()

        # Create mapping from indices tuple to position
        indices_to_pos = {indices: ii for ii, indices in enumerate(all_indices)}

        # Symmetric pairs for nobs=5, k=3
        # These pairs are mirror images around the center of [-1, 1]
        # [0,1,2] <-> [2,3,4]: left side <-> right side
        # [0,1,3] <-> [1,3,4]: etc.
        pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = [
            ((0, 1, 2), (2, 3, 4)),
            ((0, 1, 3), (1, 3, 4)),
            ((0, 1, 4), (0, 3, 4)),
            ((1, 2, 4), (0, 2, 3)),
        ]

        # Verify each symmetric pair has equal utility
        for pair1, pair2 in pairs:
            pos1 = indices_to_pos[pair1]
            pos2 = indices_to_pos[pair2]
            eig1 = all_eigs[pos1]
            eig2 = all_eigs[pos2]

            # Use relative tolerance for comparison
            bkd.assert_allclose(
                bkd.asarray([eig1]),
                bkd.asarray([eig2]),
                rtol=1e-6,
            )

    @pytest.mark.parametrize(
        "nobs,min_degree,degree,noise_std,prior_std,k,expected_optimal",
        [
            (5, 0, 2, 0.5, 0.5, 3, [0, 2, 4]),
        ],
    )
    def test_brute_force_optimal_design(
        self,
        bkd: Backend[Array],
        nobs: int,
        min_degree: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        k: int,
        expected_optimal: List[int],
    ) -> None:
        """Verify optimal k-subset design matches expected.

        For polynomial regression on [-1, 1] with nobs=5 equally spaced
        points and degree=2, the optimal 3-observation design selects
        the endpoints and center: indices [0, 2, 4].

        Replicates legacy test at test_bayesoed.py:649.
        """
        # Create benchmark
        benchmark = build_linear_gaussian_kl_benchmark(
            nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree
        )

        # Gauss quadrature: optimal design is a structural property that holds
        # regardless of quadrature accuracy, so modest sample counts suffice.
        nouter = 1000
        ninner = 100

        objective = create_kl_oed_objective_from_benchmark(
            benchmark,
            bkd,
            outer_sampler_type="gauss",
            inner_sampler_type="gauss",
            nouter_approx=nouter,
            ninner_approx=ninner,
        )

        # Solve brute-force
        solver = BruteForceKLOEDSolver(objective)
        _, _, optimal_indices = solver.solve(k)

        # Verify optimal design
        bkd.assert_allclose(
            bkd.asarray(optimal_indices),
            bkd.asarray(expected_optimal),
        )

    def test_store_all_populates_lists(
        self, bkd: Backend[Array],
    ) -> None:
        """Test that store_all=True populates all_indices and all_eigs."""
        # Simple setup with nobs=4, k=2
        nobs = 4
        benchmark = build_linear_gaussian_kl_benchmark(
            nobs, 2, 0.5, 0.5, bkd, min_degree=0
        )

        objective = create_kl_oed_objective_from_benchmark(
            benchmark,
            bkd,
            outer_sampler_type="mc",
            inner_sampler_type="mc",
            nouter_approx=100,
            ninner_approx=50,
            outer_seed=42,
            inner_seed=43,
        )

        solver = BruteForceKLOEDSolver(objective)

        # Before solve, lists should be empty
        assert len(solver.all_indices()) == 0
        assert len(solver.all_eigs()) == 0

        # Solve with store_all=False (default)
        solver.solve(k=2, store_all=False)
        assert len(solver.all_indices()) == 0
        assert len(solver.all_eigs()) == 0

        # Solve with store_all=True
        solver.solve(k=2, store_all=True)

        # Should have C(4,2) = 6 entries
        from math import comb

        expected_count = comb(nobs, 2)
        assert len(solver.all_indices()) == expected_count
        assert len(solver.all_eigs()) == expected_count

        # All EIGs should be finite
        for eig in solver.all_eigs():
            assert np.isfinite(eig)
