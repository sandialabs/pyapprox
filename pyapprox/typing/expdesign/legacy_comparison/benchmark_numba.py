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
from pyapprox.typing.expdesign.likelihood.compute import (
    logpdf_matrix_vectorized,
    jacobian_matrix_vectorized,
    evidence_jacobian_vectorized,
)
from pyapprox.typing.expdesign.likelihood.compute_numba import (
    logpdf_matrix_numba,
    jacobian_matrix_numba,
    fused_evidence_jacobian_numba,
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
    """Benchmark individual kernels: numba vs vectorized."""
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
    print("Individual Kernel Benchmarks (raw functions)")
    print(f"Problem size: nobs={nobs}, ninner={ninner}, nouter={nouter}")
    print("=" * 70)

    # JIT warmup for Numba
    print("\nWarming up Numba JIT...")
    t0 = time.perf_counter()
    logpdf_matrix_numba(shapes, obs, base_variances, design_weights)
    jit_logpdf = time.perf_counter() - t0

    t0 = time.perf_counter()
    dummy = np.zeros_like(obs)
    jacobian_matrix_numba(shapes, obs, dummy, base_variances, design_weights, False)
    jit_jacobian = time.perf_counter() - t0

    loglike = logpdf_matrix_numba(shapes, obs, base_variances, design_weights)
    like_matrix = np.exp(loglike)
    qwl = quad_weights[:, None] * like_matrix

    t0 = time.perf_counter()
    fused_evidence_jacobian_numba(
        shapes, obs, dummy, base_variances, design_weights, qwl, False,
    )
    jit_evidence = time.perf_counter() - t0

    print(f"  JIT logpdf_matrix:  {jit_logpdf:.3f}s")
    print(f"  JIT jacobian_matrix: {jit_jacobian:.3f}s")
    print(f"  JIT evidence_jacobian: {jit_evidence:.3f}s")
    print(f"  JIT total: {jit_logpdf + jit_jacobian + jit_evidence:.3f}s")

    # logpdf_matrix
    mean_numba, std_numba = time_fn(
        lambda: logpdf_matrix_numba(shapes, obs, base_variances, design_weights)
    )
    mean_vec, std_vec = time_fn(
        lambda: logpdf_matrix_vectorized(shapes, obs, base_variances, design_weights, bkd)
    )
    speedup = mean_vec / mean_numba
    print(f"\nlogpdf_matrix:")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")

    # jacobian_matrix
    mean_numba, std_numba = time_fn(
        lambda: jacobian_matrix_numba(
            shapes, obs, latent_samples, base_variances, design_weights, True,
        )
    )
    mean_vec, std_vec = time_fn(
        lambda: jacobian_matrix_vectorized(
            shapes, obs, latent_samples, base_variances, design_weights, bkd,
        )
    )
    speedup = mean_vec / mean_numba
    print(f"\njacobian_matrix:")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")

    # evidence_jacobian (fused numba vs separate vectorized)
    def run_numba_fused():
        loglike = logpdf_matrix_numba(shapes, obs, base_variances, design_weights)
        like_mat = np.exp(loglike)
        qwl = quad_weights[:, None] * like_mat
        fused_evidence_jacobian_numba(
            shapes, obs, latent_samples, base_variances,
            design_weights, qwl, True,
        )

    def run_vectorized_separate():
        loglike = logpdf_matrix_vectorized(
            shapes, obs, base_variances, design_weights, bkd,
        )
        like_mat = np.exp(loglike)
        qwl = quad_weights[:, None] * like_mat
        jac = jacobian_matrix_vectorized(
            shapes, obs, latent_samples, base_variances, design_weights, bkd,
        )
        evidence_jacobian_vectorized(jac, qwl, bkd)

    mean_numba, std_numba = time_fn(run_numba_fused)
    mean_vec, std_vec = time_fn(run_vectorized_separate)
    speedup = mean_vec / mean_numba
    print(f"\nevidence_jacobian (logpdf + fused jac vs logpdf + jac + einsum):")
    print(f"  Vectorized: {mean_vec*1000:.1f} +/- {std_vec*1000:.1f} ms")
    print(f"  Numba:      {mean_numba*1000:.1f} +/- {std_numba*1000:.1f} ms")
    print(f"  Speedup:    {speedup:.1f}x")


def benchmark_full_pipeline():
    """Benchmark the full KL-OED objective + jacobian pipeline.

    With automatic dispatch, NumPy backend always uses Numba.
    This benchmarks the auto-dispatched pipeline.
    """
    bkd = NumpyBkd()
    nobs, ninner, nouter = 50, 1000, 1000

    print("\n" + "=" * 70)
    print("Full Pipeline Benchmark: KLOEDObjective (auto-dispatched)")
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

    # Auto-dispatched (NumPy → Numba)
    inner_like = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
    obj = KLOEDObjective(
        inner_like,
        outer_shapes, latent_samples,
        inner_shapes, None, None, bkd,
    )

    dw = bkd.ones((nobs, 1))

    # JIT warmup
    print("\nWarming up Numba JIT (first call)...")
    t0 = time.perf_counter()
    obj(dw)
    obj.jacobian(dw)
    jit_time = time.perf_counter() - t0
    print(f"  JIT warmup: {jit_time:.3f}s")

    # Benchmark __call__
    mean_val, std_val = time_fn(lambda: obj(dw))
    print(f"\nKLOEDObjective.__call__:")
    print(f"  {mean_val*1000:.1f} +/- {std_val*1000:.1f} ms")

    # Benchmark jacobian
    mean_jac, std_jac = time_fn(lambda: obj.jacobian(dw))
    print(f"\nKLOEDObjective.jacobian:")
    print(f"  {mean_jac*1000:.1f} +/- {std_jac*1000:.1f} ms")

    # Combined (simulating optimizer iteration)
    def run_combined():
        obj(dw)
        obj.jacobian(dw)

    mean_comb, std_comb = time_fn(run_combined)
    print(f"\nKLOEDObjective call + jacobian (optimizer iteration):")
    print(f"  {mean_comb*1000:.1f} +/- {std_comb*1000:.1f} ms")

    # Amortization analysis
    n_iter = 100
    print(f"\nAmortization ({n_iter} optimizer iterations):")
    total = jit_time + n_iter * mean_comb
    print(f"  Total (incl JIT): {total:.1f}s")
    print(f"  Per iteration (after JIT): {mean_comb*1000:.1f} ms")


if __name__ == "__main__":
    benchmark_kernels()
    benchmark_full_pipeline()
