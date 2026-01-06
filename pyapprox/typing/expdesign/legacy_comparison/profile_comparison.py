"""
Profiling script to identify performance differences between legacy and new OED.
"""
import time
import numpy as np

from pyapprox.expdesign.bayesoed import (
    KLBayesianOED,
    IndependentGaussianOEDInnerLoopLogLikelihood,
)
from pyapprox.util.backends.numpy import NumpyMixin

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.typing.expdesign.objective import KLOEDObjective


def profile_new(nobs, ninner, nouter, n_repeats=5):
    """Profile new implementation with component timing."""
    np.random.seed(42)
    bkd = NumpyBkd()

    noise_variances = bkd.asarray(np.random.uniform(0.1, 0.3, nobs))
    outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
    inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
    latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

    inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
    objective = KLOEDObjective(
        inner_likelihood,
        outer_shapes,
        latent_samples,
        inner_shapes,
        None, None, bkd,
    )
    weights = bkd.ones((nobs, 1)) / nobs

    # Warmup
    _ = objective(weights)

    # Profile individual components
    timings = {}

    # Total time
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective(weights)
    timings['total'] = (time.perf_counter() - start) / n_repeats * 1000

    # _update_observations
    start = time.perf_counter()
    for _ in range(n_repeats):
        objective._update_observations(weights)
    timings['update_observations'] = (time.perf_counter() - start) / n_repeats * 1000

    # _generate_observations
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective._generate_observations(weights)
    timings['generate_observations'] = (time.perf_counter() - start) / n_repeats * 1000

    # outer_loglike evaluation
    objective._update_observations(weights)
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective._outer_loglike(weights)
    timings['outer_loglike'] = (time.perf_counter() - start) / n_repeats * 1000

    # log_evidence evaluation
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective._log_evidence(weights)
    timings['log_evidence'] = (time.perf_counter() - start) / n_repeats * 1000

    # Inner likelihood logpdf_matrix
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective._inner_loglike.logpdf_matrix(weights)
    timings['logpdf_matrix'] = (time.perf_counter() - start) / n_repeats * 1000

    return timings


def profile_legacy(nobs, ninner, nouter, n_repeats=5):
    """Profile legacy implementation with component timing."""
    np.random.seed(42)
    bkd = NumpyMixin

    noise_variances = np.random.uniform(0.1, 0.3, nobs)
    outer_shapes = np.random.randn(nobs, nouter)
    inner_shapes = np.random.randn(nobs, ninner)
    latent_samples = np.random.randn(nobs, nouter)

    noise_cov = noise_variances[:, None]
    inner_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(noise_cov, backend=bkd)
    kl_oed = KLBayesianOED(inner_loglike)

    outloop_weights = np.ones((nouter, 1)) / nouter
    inloop_weights = np.ones((ninner, 1)) / ninner

    kl_oed.set_data(
        outer_shapes, latent_samples, outloop_weights,
        inner_shapes, inloop_weights,
    )

    objective = kl_oed.objective()
    weights = np.ones((nobs, 1)) / nobs

    # Warmup
    _ = objective(weights)

    # Profile
    timings = {}

    # Total time
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective(weights)
    timings['total'] = (time.perf_counter() - start) / n_repeats * 1000

    # _set_expanded_design_weights (observations setup)
    start = time.perf_counter()
    for _ in range(n_repeats):
        objective._set_expanded_design_weights(weights)
    timings['set_expanded_weights'] = (time.perf_counter() - start) / n_repeats * 1000

    # outloop loglike
    objective._set_expanded_design_weights(weights)
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective._outloop_loglike(weights)
    timings['outloop_loglike'] = (time.perf_counter() - start) / n_repeats * 1000

    # log_evidence
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective._log_evidence(weights)
    timings['log_evidence'] = (time.perf_counter() - start) / n_repeats * 1000

    # inloop_loglike (logpdf_matrix equivalent)
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective._inloop_loglike(weights)
    timings['inloop_loglike'] = (time.perf_counter() - start) / n_repeats * 1000

    return timings


def main():
    """Run profiling comparison."""
    print("=" * 90)
    print("COMPONENT-LEVEL PROFILING: Legacy vs New")
    print("=" * 90)
    print()

    sizes = [
        (10, 500, 500),
        (100, 1000, 1000),
    ]

    for nobs, ninner, nouter in sizes:
        print(f"Problem: nobs={nobs}, ninner={ninner}, nouter={nouter}")
        print("-" * 90)

        new_timings = profile_new(nobs, ninner, nouter)
        legacy_timings = profile_legacy(nobs, ninner, nouter)

        print(f"{'Component':<30} {'NEW (ms)':>12} {'LEGACY (ms)':>12} {'Ratio':>10}")
        print("-" * 90)

        # Match components
        component_pairs = [
            ('total', 'total'),
            ('update_observations', 'set_expanded_weights'),
            ('generate_observations', None),
            ('outer_loglike', 'outloop_loglike'),
            ('log_evidence', 'log_evidence'),
            ('logpdf_matrix', 'inloop_loglike'),
        ]

        for new_key, legacy_key in component_pairs:
            new_time = new_timings.get(new_key, 0)
            legacy_time = legacy_timings.get(legacy_key, 0) if legacy_key else 0
            ratio = new_time / legacy_time if legacy_time > 0 else float('inf')
            legacy_str = f"{legacy_time:.2f}" if legacy_key else "N/A"
            print(f"{new_key:<30} {new_time:>10.2f}  {legacy_str:>12}  {ratio:>8.2f}x")

        print()


if __name__ == "__main__":
    main()
