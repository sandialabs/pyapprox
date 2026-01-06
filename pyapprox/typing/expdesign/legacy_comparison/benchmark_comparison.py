"""
Direct comparison benchmark: Legacy vs New OED implementation.
Run this script to compare performance of the old and new implementations.
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


def setup_data(nobs, ninner, nouter, seed=42):
    """Generate test data for both implementations."""
    np.random.seed(seed)
    noise_variances = np.random.uniform(0.1, 0.3, nobs)
    outer_shapes = np.random.randn(nobs, nouter)
    inner_shapes = np.random.randn(nobs, ninner)
    latent_samples = np.random.randn(nobs, nouter)
    return noise_variances, outer_shapes, inner_shapes, latent_samples


def benchmark_new(noise_variances, outer_shapes, inner_shapes, latent_samples, n_repeats=10):
    """Benchmark new typing implementation. Returns timing and objective values."""
    bkd = NumpyBkd()
    nobs = outer_shapes.shape[0]

    inner_likelihood = GaussianOEDInnerLoopLikelihood(bkd.asarray(noise_variances), bkd)
    objective = KLOEDObjective(
        inner_likelihood,
        bkd.asarray(outer_shapes),
        bkd.asarray(latent_samples),
        bkd.asarray(inner_shapes),
        None, None, bkd,
    )

    weights = bkd.ones((nobs, 1)) / nobs

    # Warmup
    obj_val = objective(weights)
    jac_val = objective.jacobian(weights)

    # Time evaluation
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective(weights)
    eval_time = (time.perf_counter() - start) / n_repeats * 1000

    # Time jacobian
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective.jacobian(weights)
    jac_time = (time.perf_counter() - start) / n_repeats * 1000

    return eval_time, jac_time, float(obj_val[0, 0]), bkd.to_numpy(jac_val).flatten()


def benchmark_legacy(noise_variances, outer_shapes, inner_shapes, latent_samples, n_repeats=10):
    """Benchmark legacy implementation. Returns timing and objective values."""
    # Legacy uses backend class (mixin), not instance
    bkd = NumpyMixin
    nobs, nouter = outer_shapes.shape
    ninner = inner_shapes.shape[1]

    noise_cov = noise_variances[:, None]
    inner_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(noise_cov, backend=bkd)
    kl_oed = KLBayesianOED(inner_loglike)

    # Legacy API uses 2D arrays for weights with shape (nsamples, 1)
    outloop_weights = np.ones((nouter, 1)) / nouter
    inloop_weights = np.ones((ninner, 1)) / ninner

    kl_oed.set_data(
        outer_shapes, latent_samples, outloop_weights,
        inner_shapes, inloop_weights,
    )

    objective = kl_oed.objective()
    weights = np.ones((nobs, 1)) / nobs

    # Warmup
    obj_val = objective(weights)
    jac_val = objective.jacobian(weights)

    # Time evaluation
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective(weights)
    eval_time = (time.perf_counter() - start) / n_repeats * 1000

    # Time jacobian
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = objective.jacobian(weights)
    jac_time = (time.perf_counter() - start) / n_repeats * 1000

    return eval_time, jac_time, float(np.asarray(obj_val).flatten()[0]), jac_val.flatten()


def main():
    """Run the benchmark comparison."""
    print("=" * 100)
    print("LEGACY vs NEW IMPLEMENTATION BENCHMARK")
    print("=" * 100)
    print()

    sizes = [
        (5, 50, 50),
        (8, 100, 100),
        (10, 200, 200),
        (10, 500, 500),
        (10, 1000, 1000),
        (100, 1000, 1000),
    ]

    header = (
        f"{'Problem Size':<25} {'NEW Eval':>12} {'NEW Jac':>12} "
        f"{'LEGACY Eval':>12} {'LEGACY Jac':>12} {'Speedup':>10}"
    )
    print(header)
    print("-" * 100)

    results = []
    value_checks = []
    for nobs, ninner, nouter in sizes:
        noise_var, outer_shapes, inner_shapes, latent_samples = setup_data(nobs, ninner, nouter)

        new_eval, new_jac, new_obj, new_grad = benchmark_new(
            noise_var, outer_shapes, inner_shapes, latent_samples
        )
        legacy_eval, legacy_jac, legacy_obj, legacy_grad = benchmark_legacy(
            noise_var, outer_shapes, inner_shapes, latent_samples
        )

        size_str = f"nobs={nobs}, {ninner}x{nouter}"
        speedup_eval = legacy_eval / new_eval

        print(
            f"{size_str:<25} {new_eval:>10.2f}ms {new_jac:>10.2f}ms "
            f"{legacy_eval:>10.2f}ms {legacy_jac:>10.2f}ms {speedup_eval:>8.1f}x"
        )
        results.append((nobs, ninner, nouter, new_eval, new_jac, legacy_eval, legacy_jac, speedup_eval))

        # Check value consistency
        # Note: Legacy returns negative EIG (for minimization), new returns same
        obj_diff = abs(new_obj - legacy_obj)
        grad_diff = np.max(np.abs(new_grad - legacy_grad))
        value_checks.append((size_str, obj_diff, grad_diff, new_obj, legacy_obj))

    print()
    print("=" * 100)
    print("VALUE CONSISTENCY CHECK")
    print("=" * 100)
    print(f"{'Problem Size':<25} {'Obj Diff':>15} {'Grad Max Diff':>15} {'New Obj':>15} {'Legacy Obj':>15}")
    print("-" * 100)

    all_match = True
    for size_str, obj_diff, grad_diff, new_obj, legacy_obj in value_checks:
        obj_match = "✓" if obj_diff < 1e-10 else "✗"
        grad_match = "✓" if grad_diff < 1e-10 else "✗"
        print(
            f"{size_str:<25} {obj_diff:>13.2e} {obj_match} {grad_diff:>13.2e} {grad_match} "
            f"{new_obj:>15.6f} {legacy_obj:>15.6f}"
        )
        if obj_diff >= 1e-10 or grad_diff >= 1e-10:
            all_match = False

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    avg_speedup = np.mean([r[7] for r in results])
    print(f"Average speedup (evaluation): {avg_speedup:.1f}x")

    if avg_speedup > 1:
        print("New implementation is FASTER than legacy")
    elif avg_speedup < 1:
        print("New implementation is SLOWER than legacy")
    else:
        print("Performance is approximately equal")

    print()
    if all_match:
        print("VALUE CHECK: All objective and gradient values MATCH between implementations")
    else:
        print("VALUE CHECK: Some values DIFFER between implementations - investigation needed")


if __name__ == "__main__":
    main()
