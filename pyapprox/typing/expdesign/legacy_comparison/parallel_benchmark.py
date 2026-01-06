"""
Parallel benchmark: Compare speedup with different processor counts and problem sizes.

This script measures:
1. Speedup achieved with different numbers of processors
2. Scaling behavior across problem sizes
3. Comparison between sequential and parallel implementations
"""
import time
import numpy as np
from typing import List, Tuple

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.interface.parallel import ParallelConfig
from pyapprox.typing.expdesign.objective import KLOEDObjective, ParallelKLOEDObjective
from pyapprox.typing.expdesign.likelihood import GaussianOEDInnerLoopLikelihood


def setup_data(nobs: int, ninner: int, nouter: int, seed: int = 42):
    """Generate test data."""
    np.random.seed(seed)
    bkd = NumpyBkd()
    noise_variances = bkd.asarray(np.random.uniform(0.1, 0.3, nobs))
    outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
    inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
    latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
    return noise_variances, outer_shapes, inner_shapes, latent_samples


def benchmark_sequential(
    noise_variances,
    outer_shapes,
    inner_shapes,
    latent_samples,
    n_repeats: int = 5,
) -> Tuple[float, float, float, np.ndarray]:
    """Benchmark sequential implementation."""
    bkd = NumpyBkd()
    nobs = outer_shapes.shape[0]

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


def benchmark_parallel(
    noise_variances,
    outer_shapes,
    inner_shapes,
    latent_samples,
    n_jobs: int,
    backend: str = "mpire",  # Use mpire for better multiprocessing support
    n_repeats: int = 3,
    objective: ParallelKLOEDObjective = None,  # Reuse objective to avoid pool init overhead
) -> Tuple[float, float, float, np.ndarray]:
    """Benchmark parallel implementation."""
    bkd = NumpyBkd()
    nobs = outer_shapes.shape[0]

    if objective is None:
        parallel_config = ParallelConfig(backend=backend, n_jobs=n_jobs)
        objective = ParallelKLOEDObjective(
            noise_variances,
            outer_shapes,
            latent_samples,
            inner_shapes,
            bkd,
            parallel_config,
        )
    weights = bkd.ones((nobs, 1)) / nobs

    # Warmup (2 calls to initialize pool)
    _ = objective(weights)
    _ = objective(weights)
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


def run_processor_scaling(
    nobs: int,
    ninner: int,
    nouter: int,
    n_jobs_list: List[int],
    backend: str = "joblib",
) -> List[dict]:
    """Run scaling tests with different processor counts."""
    noise_var, outer_shapes, inner_shapes, latent_samples = setup_data(
        nobs, ninner, nouter
    )

    # Get sequential baseline
    seq_eval, seq_jac, seq_obj, seq_grad = benchmark_sequential(
        noise_var, outer_shapes, inner_shapes, latent_samples
    )

    results = [{
        "n_jobs": 1,
        "type": "sequential",
        "eval_time": seq_eval,
        "jac_time": seq_jac,
        "obj_val": seq_obj,
        "eval_speedup": 1.0,
        "jac_speedup": 1.0,
    }]

    for n_jobs in n_jobs_list:
        par_eval, par_jac, par_obj, par_grad = benchmark_parallel(
            noise_var, outer_shapes, inner_shapes, latent_samples,
            n_jobs=n_jobs,
            backend=backend,
        )

        # Verify values match
        obj_diff = abs(par_obj - seq_obj)
        grad_diff = np.max(np.abs(par_grad - seq_grad))

        results.append({
            "n_jobs": n_jobs,
            "type": f"parallel-{backend}",
            "eval_time": par_eval,
            "jac_time": par_jac,
            "obj_val": par_obj,
            "eval_speedup": seq_eval / par_eval,
            "jac_speedup": seq_jac / par_jac,
            "obj_diff": obj_diff,
            "grad_diff": grad_diff,
        })

    return results


def run_size_scaling(
    sizes: List[Tuple[int, int, int]],
    n_jobs: int,
    backend: str = "joblib",
) -> List[dict]:
    """Run scaling tests with different problem sizes."""
    results = []

    for nobs, ninner, nouter in sizes:
        noise_var, outer_shapes, inner_shapes, latent_samples = setup_data(
            nobs, ninner, nouter
        )

        # Sequential
        seq_eval, seq_jac, seq_obj, seq_grad = benchmark_sequential(
            noise_var, outer_shapes, inner_shapes, latent_samples
        )

        # Parallel
        par_eval, par_jac, par_obj, par_grad = benchmark_parallel(
            noise_var, outer_shapes, inner_shapes, latent_samples,
            n_jobs=n_jobs,
            backend=backend,
        )

        obj_diff = abs(par_obj - seq_obj)
        grad_diff = np.max(np.abs(par_grad - seq_grad))

        results.append({
            "nobs": nobs,
            "ninner": ninner,
            "nouter": nouter,
            "seq_eval": seq_eval,
            "seq_jac": seq_jac,
            "par_eval": par_eval,
            "par_jac": par_jac,
            "eval_speedup": seq_eval / par_eval,
            "jac_speedup": seq_jac / par_jac,
            "obj_diff": obj_diff,
            "grad_diff": grad_diff,
        })

    return results


def main():
    """Run parallel benchmark comparison."""
    import os
    n_cpus = os.cpu_count() or 4

    print("=" * 100)
    print("PARALLEL OED BENCHMARK")
    print("=" * 100)
    print(f"Available CPUs: {n_cpus}")
    print()

    # Test 1: Processor scaling for a fixed problem size
    print("=" * 100)
    print("TEST 1: PROCESSOR SCALING (nobs=50, ninner=1000, nouter=1000)")
    print("=" * 100)
    print()

    # Use processor counts up to max available CPUs
    n_jobs_list = [2]
    if n_cpus >= 4:
        n_jobs_list.append(4)
    if n_cpus >= 6:
        n_jobs_list.append(6)
    if n_cpus >= 8:
        n_jobs_list.append(8)

    results = run_processor_scaling(
        nobs=50, ninner=1000, nouter=1000,
        n_jobs_list=n_jobs_list,
        backend="mpire",
    )

    print(f"{'n_jobs':<10} {'Eval (ms)':>12} {'Jac (ms)':>12} {'Eval Speedup':>14} {'Jac Speedup':>12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['n_jobs']:<10} {r['eval_time']:>10.2f}  {r['jac_time']:>10.2f}  "
            f"{r['eval_speedup']:>12.2f}x {r['jac_speedup']:>10.2f}x"
        )

    # Check values match
    print()
    print("Value consistency check:")
    all_match = True
    for r in results:
        if r["type"] != "sequential":
            obj_match = "PASS" if r["obj_diff"] < 1e-10 else "FAIL"
            grad_match = "PASS" if r["grad_diff"] < 1e-10 else "FAIL"
            print(f"  n_jobs={r['n_jobs']}: obj_diff={r['obj_diff']:.2e} ({obj_match}), grad_diff={r['grad_diff']:.2e} ({grad_match})")
            if r["obj_diff"] >= 1e-10 or r["grad_diff"] >= 1e-10:
                all_match = False

    print()

    # Test 2: Problem size scaling
    print("=" * 100)
    print(f"TEST 2: PROBLEM SIZE SCALING (n_jobs={min(4, n_cpus)})")
    print("=" * 100)
    print()

    sizes = [
        (20, 500, 500),
        (50, 1000, 1000),
        (100, 1000, 1000),
        (100, 2000, 2000),
    ]

    results = run_size_scaling(
        sizes=sizes,
        n_jobs=min(4, n_cpus),
        backend="mpire",
    )

    print(
        f"{'Problem Size':<25} {'Seq Eval':>10} {'Par Eval':>10} "
        f"{'Seq Jac':>10} {'Par Jac':>10} {'Eval Spdup':>11} {'Jac Spdup':>10}"
    )
    print("-" * 100)
    for r in results:
        size_str = f"nobs={r['nobs']}, {r['ninner']}x{r['nouter']}"
        print(
            f"{size_str:<25} {r['seq_eval']:>8.1f}ms {r['par_eval']:>8.1f}ms "
            f"{r['seq_jac']:>8.1f}ms {r['par_jac']:>8.1f}ms "
            f"{r['eval_speedup']:>9.2f}x {r['jac_speedup']:>8.2f}x"
        )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Report average speedups
    avg_eval_speedup = np.mean([r["eval_speedup"] for r in results])
    avg_jac_speedup = np.mean([r["jac_speedup"] for r in results])

    print(f"Average evaluation speedup: {avg_eval_speedup:.2f}x")
    print(f"Average jacobian speedup: {avg_jac_speedup:.2f}x")

    if all_match:
        print("All parallel results match sequential (values verified)")
    else:
        print("WARNING: Some parallel results differ from sequential")


if __name__ == "__main__":
    main()
