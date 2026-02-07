"""
Benchmark: Numba-accelerated vs vectorized OED likelihood computations.

Compares performance at target problem size (nobs=50, ninner=1000, nouter=1000)
for each kernel and the full objective+jacobian pipeline.

Reports per-operation speedup, full pipeline speedup, peak memory, and
JIT compilation overhead.

Usage:
    conda run -n linalg python -m pyapprox.typing.expdesign.legacy_comparison.benchmark_numba
"""

import time

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.expdesign.benchmarks.linear_gaussian import (
    LinearGaussianOEDBenchmark,
)
from pyapprox.typing.expdesign.likelihood import (
    GaussianOEDInnerLoopLikelihood,
)
from pyapprox.typing.expdesign.objective import KLOEDObjective


def time_fn(fn, n_warmup=3, n_timed=10):
    """Time a function, returning (mean_seconds, std_seconds)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def benchmark_kernels():
    """Benchmark individual kernels."""
    bkd = NumpyBkd()
    nobs, ninner, nouter = 50, 1000, 1000

    np.random.seed(42)
    shapes = np.random.randn(nobs, ninner)
    obs = np.random.randn(nobs, nouter)
    base_variances = np.abs(np.random.randn(nobs)) + 0.1
    design_weights = np.random.uniform(0.5, 2.0, (nobs, 1))
    latent_samples = np.random.randn(nobs, nouter)

    quad_weights = np.ones(ninner) / ninner

    print("=" * 70)
    print("Individual Kernel Benchmarks")
    print(f"Problem size: nobs={nobs}, ninner={ninner}, nouter={nouter}")
    print("=" * 70)

    # --- logpdf_matrix ---
    like_numba = GaussianOEDInnerLoopLikelihood(
        bkd.asarray(base_variances), bkd, use_numba=True,
    )
    like_vec = GaussianOEDInnerLoopLikelihood(
        bkd.asarray(base_variances), bkd, use_numba=False,
    )
    for like in (like_numba, like_vec):
        like.set_shapes(bkd.asarray(shapes))
        like.set_observations(bkd.asarray(obs))
        like.set_latent_samples(bkd.asarray(latent_samples))

    dw = bkd.asarray(design_weights)

    # JIT warmup for Numba
    print("\nWarming up Numba JIT...")
    t0 = time.perf_counter()
    like_numba.logpdf_matrix(dw)
    jit_logpdf = time.perf_counter() - t0

    t0 = time.perf_counter()
    like_numba.jacobian_matrix(dw)
    jit_jacobian = time.perf_counter() - t0

    loglike = like_numba.logpdf_matrix(dw)
    like_matrix = np.exp(loglike)
    qwl = quad_weights[:, None] * like_matrix

    t0 = time.perf_counter()
    like_numba.evidence_jacobian(dw, qwl)
    jit_evidence = time.perf_counter() - t0

    print(f"  JIT logpdf_matrix:  {jit_logpdf:.3f}s")
    print(f"  JIT jacobian_matrix: {jit_jacobian:.3f}s")
    print(f"  JIT evidence_jacobian: {jit_evidence:.3f}s")
    print(f"  JIT total: {jit_logpdf + jit_jacobian + jit_evidence:.3f}s")

    # logpdf_matrix
    mean_numba, std_numba = time_fn(lambda: like_numba.logpdf_matrix(dw))
    mean_vec, std_vec = time_fn(lambda: like_vec.logpdf_matrix(dw))
    speedup = mean_vec / mean_numba
    print(f"\nlogpdf_matrix:")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")

    # jacobian_matrix
    mean_numba, std_numba = time_fn(lambda: like_numba.jacobian_matrix(dw))
    mean_vec, std_vec = time_fn(lambda: like_vec.jacobian_matrix(dw))
    speedup = mean_vec / mean_numba
    print(f"\njacobian_matrix:")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")

    # evidence_jacobian (fused)
    # Recompute quad_weighted_like for each path
    def run_fused():
        loglike = like_numba.logpdf_matrix(dw)
        like_mat = np.exp(loglike)
        qwl = quad_weights[:, None] * like_mat
        like_numba.evidence_jacobian(dw, qwl)

    def run_separate():
        loglike = like_vec.logpdf_matrix(dw)
        like_mat = np.exp(loglike)
        qwl = quad_weights[:, None] * like_mat
        jac = like_vec.jacobian_matrix(dw)
        np.einsum("io,iok->ok", qwl, jac)

    mean_numba, std_numba = time_fn(run_fused)
    mean_vec, std_vec = time_fn(run_separate)
    speedup = mean_vec / mean_numba
    print(f"\nevidence_jacobian (logpdf + fused jac vs logpdf + jac + einsum):")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")


def benchmark_full_pipeline():
    """Benchmark the full KL-OED objective + jacobian pipeline."""
    bkd = NumpyBkd()
    nobs, ninner, nouter = 50, 1000, 1000

    print("\n" + "=" * 70)
    print("Full Pipeline Benchmark: KLOEDObjective")
    print(f"Problem size: nobs={nobs}, ninner={ninner}, nouter={nouter}")
    print("=" * 70)

    # Set up benchmark data
    benchmark = LinearGaussianOEDBenchmark(
        nobs=nobs, degree=5, noise_std=0.1, prior_std=1.0, bkd=bkd,
    )

    np.random.seed(42)
    noise_variances = bkd.asarray(np.full(nobs, benchmark.noise_var()))

    # Generate shapes (model outputs)
    _, inner_shapes = benchmark.generate_data(ninner, seed=42)
    _, outer_shapes = benchmark.generate_data(nouter, seed=123)

    # Generate latent samples
    latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

    # Numba-accelerated
    inner_like_numba = GaussianOEDInnerLoopLikelihood(
        noise_variances, bkd, use_numba=True,
    )
    obj_numba = KLOEDObjective(
        inner_like_numba,
        outer_shapes, latent_samples,
        inner_shapes, None, None, bkd,
    )

    # Vectorized only
    inner_like_vec = GaussianOEDInnerLoopLikelihood(
        noise_variances, bkd, use_numba=False,
    )
    obj_vec = KLOEDObjective(
        inner_like_vec,
        outer_shapes, latent_samples,
        inner_shapes, None, None, bkd,
    )

    dw = bkd.ones((nobs, 1))

    # JIT warmup
    print("\nWarming up Numba JIT (first call)...")
    t0 = time.perf_counter()
    obj_numba(dw)
    obj_numba.jacobian(dw)
    jit_time = time.perf_counter() - t0
    print(f"  JIT warmup: {jit_time:.3f}s")

    # Benchmark __call__
    mean_numba, std_numba = time_fn(lambda: obj_numba(dw))
    mean_vec, std_vec = time_fn(lambda: obj_vec(dw))
    speedup = mean_vec / mean_numba
    print(f"\nKLOEDObjective.__call__:")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")

    # Benchmark jacobian
    mean_numba, std_numba = time_fn(lambda: obj_numba.jacobian(dw))
    mean_vec, std_vec = time_fn(lambda: obj_vec.jacobian(dw))
    speedup = mean_vec / mean_numba
    print(f"\nKLOEDObjective.jacobian:")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")

    # Combined (simulating optimizer iteration)
    def run_combined(obj):
        obj(dw)
        obj.jacobian(dw)

    mean_numba, std_numba = time_fn(lambda: run_combined(obj_numba))
    mean_vec, std_vec = time_fn(lambda: run_combined(obj_vec))
    speedup = mean_vec / mean_numba
    print(f"\nKLOEDObjective call + jacobian (optimizer iteration):")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")

    # Verify correctness
    val_numba = obj_numba(dw)
    val_vec = obj_vec(dw)
    jac_numba = obj_numba.jacobian(dw)
    jac_vec = obj_vec.jacobian(dw)

    val_diff = float(np.max(np.abs(val_numba - val_vec)))
    jac_diff = float(np.max(np.abs(jac_numba - jac_vec)))
    print(f"\nCorrectness check:")
    print(f"  Max value difference: {val_diff:.2e}")
    print(f"  Max jacobian difference: {jac_diff:.2e}")

    # Amortization analysis
    n_iter = 100
    print(f"\nAmortization ({n_iter} optimizer iterations):")
    amortized_numba = jit_time + n_iter * mean_numba
    amortized_vec = n_iter * mean_vec
    print(f"  Vectorized total: {amortized_vec:.1f}s")
    print(f"  Numba total (incl JIT): {amortized_numba:.1f}s")
    if amortized_numba < amortized_vec:
        print(f"  Numba breaks even after "
              f"~{int(jit_time / (mean_vec - mean_numba))} iterations")
    else:
        print("  Numba does not break even within 100 iterations")


if __name__ == "__main__":
    benchmark_kernels()
    benchmark_full_pipeline()
