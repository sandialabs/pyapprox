"""
Standalone tests for brute-force OED utility symmetry.

PERMANENT - no legacy imports.

Replicates legacy test_brute_force_d_optimal_oed from test_bayesoed.py:575-649.
Verifies that symmetric design configurations produce equal utility values
and that the optimal design is found correctly.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.benchmarks import LinearGaussianOEDBenchmark
from pyapprox.typing.expdesign.objective import KLOEDObjective, create_kl_oed_objective
from pyapprox.typing.expdesign.solver import BruteForceKLOEDSolver
from pyapprox.typing.probability.joint.independent import IndependentJoint
from pyapprox.typing.probability.univariate.gaussian import GaussianMarginal


# =============================================================================
# Test Utilities
# =============================================================================

class LinearForwardModel(Generic[Array]):
    """Test utility: Wrap a design matrix as a FunctionProtocol.

    Forward model: y = A @ theta
    """

    def __init__(self, design_matrix: Array, bkd: Backend[Array]):
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
    benchmark: LinearGaussianOEDBenchmark[Array],
    bkd: Backend[Array],
    outer_sampler_type: str = "gauss",
    inner_sampler_type: str = "gauss",
    nouter_approx: int = 100000,
    ninner_approx: int = 1000,
    outer_seed: int = None,
    inner_seed: int = None,
) -> KLOEDObjective[Array]:
    """Test utility: Create KLOEDObjective from a LinearGaussianOEDBenchmark."""
    nobs = benchmark.nobs()
    nparams = benchmark.nparams()
    prior_std = np.sqrt(benchmark.prior_var())

    # Create prior distribution
    prior_marginals = [
        GaussianMarginal(0.0, prior_std, bkd) for _ in range(nparams)
    ]
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

class TestBruteForceSymmetryStandalone(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Standalone tests for brute-force OED utility symmetry.

    Replicates legacy test_brute_force_d_optimal_oed:
    - Verify symmetric design pairs have equal utility values
    - Verify optimal k-subset design matches expected
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    @parametrize(
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
        nobs: int,
        min_degree: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        k: int,
        expected_optimal: list,
    ):
        """Verify symmetric design pairs have equal utility values.

        For symmetric design locations in [-1, 1], pairs of designs that
        are mirror images should have equal utility (EIG) values.

        Replicates legacy test at test_bayesoed.py:625-647.
        """
        # Create benchmark
        benchmark = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, self._bkd, min_degree=min_degree
        )

        # Gauss quadrature: symmetry is a structural property that holds
        # regardless of quadrature accuracy, so modest sample counts suffice.
        nouter = 1000
        ninner = 100

        objective = create_kl_oed_objective_from_benchmark(
            benchmark, self._bkd,
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
        indices_to_pos = {
            indices: ii for ii, indices in enumerate(all_indices)
        }

        # Symmetric pairs for nobs=5, k=3
        # These pairs are mirror images around the center of [-1, 1]
        # [0,1,2] <-> [2,3,4]: left side <-> right side
        # [0,1,3] <-> [1,3,4]: etc.
        pairs = [
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
            self._bkd.assert_allclose(
                self._bkd.asarray([eig1]),
                self._bkd.asarray([eig2]),
                rtol=1e-6,
            )

    @parametrize(
        "nobs,min_degree,degree,noise_std,prior_std,k,expected_optimal",
        [
            (5, 0, 2, 0.5, 0.5, 3, [0, 2, 4]),
        ],
    )
    def test_brute_force_optimal_design(
        self,
        nobs: int,
        min_degree: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        k: int,
        expected_optimal: list,
    ):
        """Verify optimal k-subset design matches expected.

        For polynomial regression on [-1, 1] with nobs=5 equally spaced
        points and degree=2, the optimal 3-observation design selects
        the endpoints and center: indices [0, 2, 4].

        Replicates legacy test at test_bayesoed.py:649.
        """
        # Create benchmark
        benchmark = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, self._bkd, min_degree=min_degree
        )

        # Gauss quadrature: optimal design is a structural property that holds
        # regardless of quadrature accuracy, so modest sample counts suffice.
        nouter = 1000
        ninner = 100

        objective = create_kl_oed_objective_from_benchmark(
            benchmark, self._bkd,
            outer_sampler_type="gauss",
            inner_sampler_type="gauss",
            nouter_approx=nouter,
            ninner_approx=ninner,
        )

        # Solve brute-force
        solver = BruteForceKLOEDSolver(objective)
        _, _, optimal_indices = solver.solve(k)

        # Verify optimal design
        self._bkd.assert_allclose(
            self._bkd.asarray(optimal_indices),
            self._bkd.asarray(expected_optimal),
        )

    def test_store_all_populates_lists(self):
        """Test that store_all=True populates all_indices and all_eigs."""
        # Simple setup with nobs=4, k=2
        nobs = 4
        benchmark = LinearGaussianOEDBenchmark(
            nobs, 2, 0.5, 0.5, self._bkd, min_degree=0
        )

        objective = create_kl_oed_objective_from_benchmark(
            benchmark, self._bkd,
            outer_sampler_type="mc",
            inner_sampler_type="mc",
            nouter_approx=100,
            ninner_approx=50,
            outer_seed=42,
            inner_seed=43,
        )

        solver = BruteForceKLOEDSolver(objective)

        # Before solve, lists should be empty
        self.assertEqual(len(solver.all_indices()), 0)
        self.assertEqual(len(solver.all_eigs()), 0)

        # Solve with store_all=False (default)
        solver.solve(k=2, store_all=False)
        self.assertEqual(len(solver.all_indices()), 0)
        self.assertEqual(len(solver.all_eigs()), 0)

        # Solve with store_all=True
        solver.solve(k=2, store_all=True)

        # Should have C(4,2) = 6 entries
        from math import comb
        expected_count = comb(nobs, 2)
        self.assertEqual(len(solver.all_indices()), expected_count)
        self.assertEqual(len(solver.all_eigs()), expected_count)

        # All EIGs should be finite
        for eig in solver.all_eigs():
            self.assertTrue(np.isfinite(eig))


class TestBruteForceSymmetryStandaloneNumpy(
    TestBruteForceSymmetryStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBruteForceSymmetryStandaloneTorch(
    TestBruteForceSymmetryStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
