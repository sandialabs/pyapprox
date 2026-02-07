"""
Benchmark: torch.compile-accelerated vs eager PyTorch OED computations.

Compares performance at target problem size (nobs=50, ninner=1000, nouter=1000)
for each kernel and the full objective+jacobian pipeline.

Reports per-operation speedup, warmup overhead, and correctness verification.

Usage:
    conda run -n linalg python -m pyapprox.typing.expdesign.legacy_comparison.benchmark_torch_compile
"""

import time

import numpy as np
import torch

from pyapprox.typing.util.backends.torch import TorchBkd
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
    """Benchmark individual kernels: compiled vs eager PyTorch."""
    torch.set_default_dtype(torch.float64)
    bkd = TorchBkd()
    nobs, ninner, nouter = 50, 1000, 1000

    np.random.seed(42)
    shapes = bkd.asarray(np.random.randn(nobs, ninner))
    obs = bkd.asarray(np.random.randn(nobs, nouter))
    base_variances = bkd.asarray(np.abs(np.random.randn(nobs)) + 0.1)
    design_weights = bkd.asarray(np.random.uniform(0.5, 2.0, (nobs, 1)))
    latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
    quad_weights = bkd.asarray(np.ones(ninner) / ninner)

    print("=" * 70)
    print("Individual Kernel Benchmarks (PyTorch CPU)")
    print(f"Problem size: nobs={nobs}, ninner={ninner}, nouter={nouter}")
    print(f"Device: {shapes.device}")
    print("=" * 70)

    # Create eager and compiled likelihoods
    like_eager = GaussianOEDInnerLoopLikelihood(
        base_variances, bkd, use_numba=False, use_torch_compile=False,
    )
    like_compiled = GaussianOEDInnerLoopLikelihood(
        base_variances, bkd, use_numba=False, use_torch_compile=True,
    )
    for like in (like_eager, like_compiled):
        like.set_shapes(shapes)
        like.set_observations(obs)
        like.set_latent_samples(latent_samples)

    dw = design_weights

    # torch.compile warmup
    print("\nWarming up torch.compile (first calls trigger compilation)...")
    t0 = time.perf_counter()
    like_compiled.logpdf_matrix(dw)
    warmup_logpdf = time.perf_counter() - t0

    t0 = time.perf_counter()
    like_compiled.jacobian_matrix(dw)
    warmup_jacobian = time.perf_counter() - t0

    loglike = like_compiled.logpdf_matrix(dw)
    like_matrix = torch.exp(loglike)
    qwl = quad_weights[:, None] * like_matrix

    t0 = time.perf_counter()
    like_compiled.evidence_jacobian(dw, qwl)
    warmup_evidence = time.perf_counter() - t0

    print(f"  Compile logpdf_matrix:    {warmup_logpdf:.3f}s")
    print(f"  Compile jacobian_matrix:  {warmup_jacobian:.3f}s")
    print(f"  Compile evidence_jacobian: {warmup_evidence:.3f}s")
    print(f"  Compile total: {warmup_logpdf + warmup_jacobian + warmup_evidence:.3f}s")

    # logpdf_matrix
    mean_compiled, std_compiled = time_fn(
        lambda: like_compiled.logpdf_matrix(dw),
    )
    mean_eager, std_eager = time_fn(
        lambda: like_eager.logpdf_matrix(dw),
    )
    speedup = mean_eager / mean_compiled
    print(f"\nlogpdf_matrix:")
    print(f"  Eager:    {mean_eager*1000:.1f} +/- {std_eager*1000:.1f} ms")
    print(f"  Compiled: {mean_compiled*1000:.1f} +/- {std_compiled*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # jacobian_matrix
    mean_compiled, std_compiled = time_fn(
        lambda: like_compiled.jacobian_matrix(dw),
    )
    mean_eager, std_eager = time_fn(
        lambda: like_eager.jacobian_matrix(dw),
    )
    speedup = mean_eager / mean_compiled
    print(f"\njacobian_matrix:")
    print(f"  Eager:    {mean_eager*1000:.1f} +/- {std_eager*1000:.1f} ms")
    print(f"  Compiled: {mean_compiled*1000:.1f} +/- {std_compiled*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # evidence_jacobian (compiled jac + compiled einsum vs eager jac + eager einsum)
    def run_compiled_fused():
        loglike = like_compiled.logpdf_matrix(dw)
        like_mat = torch.exp(loglike)
        qwl = quad_weights[:, None] * like_mat
        like_compiled.evidence_jacobian(dw, qwl)

    def run_eager_separate():
        loglike = like_eager.logpdf_matrix(dw)
        like_mat = torch.exp(loglike)
        qwl = quad_weights[:, None] * like_mat
        jac = like_eager.jacobian_matrix(dw)
        torch.einsum("io,iok->ok", qwl, jac)

    mean_compiled, std_compiled = time_fn(run_compiled_fused)
    mean_eager, std_eager = time_fn(run_eager_separate)
    speedup = mean_eager / mean_compiled
    print(f"\nevidence_jacobian (logpdf + fused jac vs logpdf + jac + einsum):")
    print(f"  Eager:    {mean_eager*1000:.1f} +/- {std_eager*1000:.1f} ms")
    print(f"  Compiled: {mean_compiled*1000:.1f} +/- {std_compiled*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # Correctness check
    val_eager = like_eager.logpdf_matrix(dw)
    val_compiled = like_compiled.logpdf_matrix(dw)
    diff = float(torch.max(torch.abs(val_eager - val_compiled)))
    print(f"\nCorrectness (logpdf max abs diff): {diff:.2e}")

    jac_eager = like_eager.jacobian_matrix(dw)
    jac_compiled = like_compiled.jacobian_matrix(dw)
    diff = float(torch.max(torch.abs(jac_eager - jac_compiled)))
    print(f"Correctness (jacobian max abs diff): {diff:.2e}")


def benchmark_full_pipeline():
    """Benchmark the full KL-OED objective + jacobian pipeline."""
    torch.set_default_dtype(torch.float64)
    bkd = TorchBkd()
    nobs, ninner, nouter = 50, 1000, 1000

    print("\n" + "=" * 70)
    print("Full Pipeline Benchmark: KLOEDObjective (PyTorch CPU)")
    print(f"Problem size: nobs={nobs}, ninner={ninner}, nouter={nouter}")
    print("=" * 70)

    # Set up benchmark data (noise_std=0.5 avoids numerical underflow
    # that produces NaN at this problem size with noise_std=0.1)
    benchmark = LinearGaussianOEDBenchmark(
        nobs=nobs, degree=5, noise_std=0.5, prior_std=1.0, bkd=bkd,
    )

    np.random.seed(42)
    noise_variances = bkd.asarray(np.full(nobs, benchmark.noise_var()))

    _, inner_shapes = benchmark.generate_data(ninner, seed=42)
    _, outer_shapes = benchmark.generate_data(nouter, seed=123)
    latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

    # Compiled
    inner_like_compiled = GaussianOEDInnerLoopLikelihood(
        noise_variances, bkd, use_numba=False, use_torch_compile=True,
    )
    obj_compiled = KLOEDObjective(
        inner_like_compiled,
        outer_shapes, latent_samples,
        inner_shapes, None, None, bkd,
    )

    # Eager
    inner_like_eager = GaussianOEDInnerLoopLikelihood(
        noise_variances, bkd, use_numba=False, use_torch_compile=False,
    )
    obj_eager = KLOEDObjective(
        inner_like_eager,
        outer_shapes, latent_samples,
        inner_shapes, None, None, bkd,
    )

    dw = bkd.ones((nobs, 1))

    # torch.compile warmup
    print("\nWarming up torch.compile (first calls trigger compilation)...")
    t0 = time.perf_counter()
    obj_compiled(dw)
    obj_compiled.jacobian(dw)
    warmup_time = time.perf_counter() - t0
    print(f"  Compile warmup: {warmup_time:.3f}s")

    # Benchmark __call__
    mean_compiled, std_compiled = time_fn(lambda: obj_compiled(dw))
    mean_eager, std_eager = time_fn(lambda: obj_eager(dw))
    speedup = mean_eager / mean_compiled
    print(f"\nKLOEDObjective.__call__:")
    print(f"  Eager:    {mean_eager*1000:.1f} +/- {std_eager*1000:.1f} ms")
    print(f"  Compiled: {mean_compiled*1000:.1f} +/- {std_compiled*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # Benchmark jacobian
    mean_compiled, std_compiled = time_fn(lambda: obj_compiled.jacobian(dw))
    mean_eager, std_eager = time_fn(lambda: obj_eager.jacobian(dw))
    speedup = mean_eager / mean_compiled
    print(f"\nKLOEDObjective.jacobian:")
    print(f"  Eager:    {mean_eager*1000:.1f} +/- {std_eager*1000:.1f} ms")
    print(f"  Compiled: {mean_compiled*1000:.1f} +/- {std_compiled*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # Combined (simulating optimizer iteration)
    def run_combined(obj):
        obj(dw)
        obj.jacobian(dw)

    mean_compiled, std_compiled = time_fn(lambda: run_combined(obj_compiled))
    mean_eager, std_eager = time_fn(lambda: run_combined(obj_eager))
    speedup = mean_eager / mean_compiled
    print(f"\nKLOEDObjective call + jacobian (optimizer iteration):")
    print(f"  Eager:    {mean_eager*1000:.1f} +/- {std_eager*1000:.1f} ms")
    print(f"  Compiled: {mean_compiled*1000:.1f} +/- {std_compiled*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # Verify correctness
    val_compiled = obj_compiled(dw)
    val_eager = obj_eager(dw)
    jac_compiled = obj_compiled.jacobian(dw)
    jac_eager = obj_eager.jacobian(dw)

    val_diff = float(torch.max(torch.abs(val_compiled - val_eager)))
    jac_diff = float(torch.max(torch.abs(jac_compiled - jac_eager)))
    print(f"\nCorrectness check:")
    print(f"  Max value difference: {val_diff:.2e}")
    print(f"  Max jacobian difference: {jac_diff:.2e}")

    # Amortization analysis
    n_iter = 100
    print(f"\nAmortization ({n_iter} optimizer iterations):")
    amortized_compiled = warmup_time + n_iter * mean_compiled
    amortized_eager = n_iter * mean_eager
    print(f"  Eager total: {amortized_eager:.1f}s")
    print(f"  Compiled total (incl warmup): {amortized_compiled:.1f}s")
    if amortized_compiled < amortized_eager:
        print(f"  Compiled breaks even after "
              f"~{int(warmup_time / (mean_eager - mean_compiled))} iterations")
    else:
        print("  Compiled does not break even within 100 iterations")


if __name__ == "__main__":
    benchmark_kernels()
    benchmark_full_pipeline()
