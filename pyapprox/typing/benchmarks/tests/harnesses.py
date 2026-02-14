"""Generic test harnesses with pluggable strategies for benchmark verification.

Harnesses accept pluggable estimation strategies, time budgets, and
caller-specified tolerances. Tolerances are NOT a benchmark property ---
they depend on user requirements and computational resources.

Strategy Protocols
------------------
MeanEstimationStrategy, VarianceEstimationStrategy

Built-in Strategies
-------------------
MCMeanEstimator, MCVarianceEstimator

Harness Functions
-----------------
verify_jacobian_fd
    Check Jacobian via finite-difference convergence (DerivativeChecker).
verify_forward_uq_mean
    Compare estimated mean against reference value.
verify_forward_uq_variance
    Compare estimated variance against reference value.

Auto-selection
--------------
select_mean_verifier, select_variance_verifier
    Choose estimation strategy from benchmark properties (budget-adapted).
"""

from typing import Protocol, Optional, Any

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.protocols import (
    HasEstimatedEvaluationCost,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


# ---------------------------------------------------------------------------
# Strategy protocols
# ---------------------------------------------------------------------------


class MeanEstimationStrategy(Protocol):
    """Pluggable strategy for estimating the mean of a benchmark function."""

    def estimate_mean(self, benchmark: Any, bkd: Backend[Array]) -> float:
        """Return an estimate of the benchmark function's mean."""
        ...


class VarianceEstimationStrategy(Protocol):
    """Pluggable strategy for estimating the variance of a benchmark function."""

    def estimate_variance(self, benchmark: Any, bkd: Backend[Array]) -> float:
        """Return an estimate of the benchmark function's variance."""
        ...


# ---------------------------------------------------------------------------
# Built-in MC strategies
# ---------------------------------------------------------------------------


class MCMeanEstimator:
    """Monte Carlo mean estimator.

    Parameters
    ----------
    n_samples : int
        Number of MC samples.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_samples: int = 50000, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def estimate_mean(self, benchmark: Any, bkd: Backend[Array]) -> float:
        np.random.seed(self.seed)
        samples = benchmark.prior().rvs(self.n_samples)
        values = benchmark.function()(samples)
        return float(bkd.to_numpy(bkd.mean(values)))


class MCVarianceEstimator:
    """Monte Carlo variance estimator.

    Parameters
    ----------
    n_samples : int
        Number of MC samples.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_samples: int = 50000, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def estimate_variance(self, benchmark: Any, bkd: Backend[Array]) -> float:
        np.random.seed(self.seed)
        samples = benchmark.prior().rvs(self.n_samples)
        values = benchmark.function()(samples)
        return float(bkd.to_numpy(bkd.var(values)))


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------


def select_mean_verifier(
    benchmark: Any,
    budget_seconds: float = 60.0,
) -> MCMeanEstimator:
    """Auto-select mean estimation strategy from benchmark properties.

    Uses ``HasEstimatedEvaluationCost`` when available to adapt sample
    count to the time budget. Returns ``MCMeanEstimator`` with
    budget-adapted sample count.

    Parameters
    ----------
    benchmark
        Benchmark instance (may satisfy ``HasEstimatedEvaluationCost``).
    budget_seconds : float
        Wall-clock budget in seconds.

    Returns
    -------
    MCMeanEstimator
        Strategy with adapted sample count.
    """
    cost = 1e-6
    if isinstance(benchmark, HasEstimatedEvaluationCost):
        cost = benchmark.estimated_evaluation_cost()
    max_evals = int(budget_seconds / cost)
    return MCMeanEstimator(n_samples=min(max_evals, 50000))


def select_variance_verifier(
    benchmark: Any,
    budget_seconds: float = 60.0,
) -> MCVarianceEstimator:
    """Auto-select variance estimation strategy from benchmark properties.

    Parameters
    ----------
    benchmark
        Benchmark instance (may satisfy ``HasEstimatedEvaluationCost``).
    budget_seconds : float
        Wall-clock budget in seconds.

    Returns
    -------
    MCVarianceEstimator
        Strategy with adapted sample count.
    """
    cost = 1e-6
    if isinstance(benchmark, HasEstimatedEvaluationCost):
        cost = benchmark.estimated_evaluation_cost()
    max_evals = int(budget_seconds / cost)
    return MCVarianceEstimator(n_samples=min(max_evals, 50000))


# ---------------------------------------------------------------------------
# Harness functions
# ---------------------------------------------------------------------------


def verify_jacobian_fd(
    benchmark: Any,
    bkd: Backend[Array],
    n_tests: int = 5,
    seed: int = 42,
    rtol: float = 1e-4,
) -> None:
    """Verify benchmark Jacobian via finite-difference convergence.

    Requires the benchmark to satisfy ``HasJacobian`` (i.e. its
    ``function()`` must have a ``jacobian`` method).

    Parameters
    ----------
    benchmark
        Benchmark instance satisfying ``HasForwardModel`` and ``HasJacobian``.
    bkd
        Backend for array operations.
    n_tests : int
        Number of random sample points to test.
    seed : int
        Random seed for generating test points.
    rtol : float
        Maximum acceptable ``error_ratio`` from ``DerivativeChecker``.

    Raises
    ------
    AssertionError
        If the finite-difference convergence ratio exceeds *rtol* for
        any test point.
    """
    func = benchmark.function()
    checker = DerivativeChecker(func)

    domain_bounds = bkd.to_numpy(benchmark.domain().bounds())
    nvars = domain_bounds.shape[0]
    np.random.seed(seed)

    for ii in range(n_tests):
        # Sample uniformly within the domain
        u = np.random.rand(nvars, 1)
        lo = domain_bounds[:, 0:1]
        hi = domain_bounds[:, 1:2]
        sample_np = lo + u * (hi - lo)
        sample = bkd.array(sample_np)

        errors = checker.check_derivatives(sample)
        jac_errors = errors[0]
        ratio = checker.error_ratio(jac_errors)
        assert float(bkd.to_numpy(ratio)) <= rtol, (
            f"Jacobian FD check failed for {benchmark.name()!r} at test "
            f"point {ii}: error_ratio={float(bkd.to_numpy(ratio)):.2e} > "
            f"{rtol:.2e}"
        )


def verify_forward_uq_mean(
    benchmark: Any,
    bkd: Backend[Array],
    strategy: Optional[MeanEstimationStrategy] = None,
    budget_seconds: float = 60.0,
    rtol: float = 1e-2,
    atol: Optional[float] = None,
) -> None:
    """Verify that the estimated mean matches the reference value.

    Requires: ``HasForwardModel``, ``HasPrior``, ``HasReferenceMean``.

    Parameters
    ----------
    benchmark
        Benchmark instance.
    bkd
        Backend for array operations.
    strategy
        Mean estimation strategy. If *None*, auto-selected.
    budget_seconds : float
        Time budget for auto-selected strategy.
    rtol : float
        Relative tolerance (used when *atol* is None).
    atol : float or None
        Absolute tolerance. Overrides *rtol* when provided.

    Raises
    ------
    AssertionError
        If ``|estimate - reference| > tol``.
    """
    if strategy is None:
        strategy = select_mean_verifier(benchmark, budget_seconds)
    estimate = strategy.estimate_mean(benchmark, bkd)
    reference = benchmark.reference_mean()
    if atol is not None:
        tol = atol
    else:
        tol = max(abs(reference) * rtol, 1e-14)
    assert abs(estimate - reference) < tol, (
        f"Mean check failed for {benchmark.name()!r}: "
        f"estimate={estimate:.6e}, reference={reference:.6e}, "
        f"error={abs(estimate - reference):.6e}, tol={tol:.6e}"
    )


def verify_forward_uq_variance(
    benchmark: Any,
    bkd: Backend[Array],
    strategy: Optional[VarianceEstimationStrategy] = None,
    budget_seconds: float = 60.0,
    rtol: float = 1e-2,
    atol: Optional[float] = None,
) -> None:
    """Verify that the estimated variance matches the reference value.

    Requires: ``HasForwardModel``, ``HasPrior``, ``HasReferenceVariance``.

    Parameters
    ----------
    benchmark
        Benchmark instance.
    bkd
        Backend for array operations.
    strategy
        Variance estimation strategy. If *None*, auto-selected.
    budget_seconds : float
        Time budget for auto-selected strategy.
    rtol : float
        Relative tolerance (used when *atol* is None).
    atol : float or None
        Absolute tolerance. Overrides *rtol* when provided.

    Raises
    ------
    AssertionError
        If ``|estimate - reference| > tol``.
    """
    if strategy is None:
        strategy = select_variance_verifier(benchmark, budget_seconds)
    estimate = strategy.estimate_variance(benchmark, bkd)
    reference = benchmark.reference_variance()
    if atol is not None:
        tol = atol
    else:
        tol = max(abs(reference) * rtol, 1e-14)
    assert abs(estimate - reference) < tol, (
        f"Variance check failed for {benchmark.name()!r}: "
        f"estimate={estimate:.6e}, reference={reference:.6e}, "
        f"error={abs(estimate - reference):.6e}, tol={tol:.6e}"
    )
