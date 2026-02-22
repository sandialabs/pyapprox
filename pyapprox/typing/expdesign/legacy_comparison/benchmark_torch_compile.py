"""
Benchmark: torch.compile-accelerated vs eager PyTorch OED computations.

Compares performance at target problem size (nobs=50, ninner=1000, nouter=1000)
for each kernel and the full objective+jacobian pipeline.

With automatic dispatch, PyTorch backend always uses torch.compile.
This benchmark compares the compiled path against the vectorized
(eager) functions called directly.

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
from pyapprox.typing.expdesign.likelihood.compute import (
    logpdf_matrix_vectorized,
    jacobian_matrix_vectorized,
    evidence_jacobian_vectorized,
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
    """Benchmark individual kernels: compiled (auto-dispatched) vs eager (vectorized)."""
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

    # Create auto-dispatched likelihood (gets torch.compile on TorchBkd)
    like_compiled = GaussianOEDInnerLoopLikelihood(base_variances, bkd)
    like_compiled.set_shapes(shapes)
    like_compiled.set_observations(obs)
    like_compiled.set_latent_samples(latent_samples)

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

    # logpdf_matrix: compiled vs vectorized (eager)
    mean_compiled, std_compiled = time_fn(
        lambda: like_compiled.logpdf_matrix(dw),
    )
    mean_eager, std_eager = time_fn(
        lambda: logpdf_matrix_vectorized(shapes, obs, base_variances, dw, bkd),
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
        lambda: jacobian_matrix_vectorized(
            shapes, obs, latent_samples, base_variances, dw, bkd,
        ),
    )
    speedup = mean_eager / mean_compiled
    print(f"\njacobian_matrix:")
    print(f"  Eager:    {mean_eager*1000:.1f} +/- {std_eager*1000:.1f} ms")
    print(f"  Compiled: {mean_compiled*1000:.1f} +/- {std_compiled*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # evidence_jacobian
    def run_compiled_fused():
        loglike = like_compiled.logpdf_matrix(dw)
        like_mat = torch.exp(loglike)
        qwl = quad_weights[:, None] * like_mat
        like_compiled.evidence_jacobian(dw, qwl)

    def run_eager_separate():
        loglike = logpdf_matrix_vectorized(shapes, obs, base_variances, dw, bkd)
        like_mat = torch.exp(loglike)
        qwl = quad_weights[:, None] * like_mat
        jac = jacobian_matrix_vectorized(
            shapes, obs, latent_samples, base_variances, dw, bkd,
        )
        evidence_jacobian_vectorized(jac, qwl, bkd)

    mean_compiled, std_compiled = time_fn(run_compiled_fused)
    mean_eager, std_eager = time_fn(run_eager_separate)
    speedup = mean_eager / mean_compiled
    print(f"\nevidence_jacobian (logpdf + fused jac vs logpdf + jac + einsum):")
    print(f"  Eager:    {mean_eager*1000:.1f} +/- {std_eager*1000:.1f} ms")
    print(f"  Compiled: {mean_compiled*1000:.1f} +/- {std_compiled*1000:.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # Correctness check
    val_eager = logpdf_matrix_vectorized(shapes, obs, base_variances, dw, bkd)
    val_compiled = like_compiled.logpdf_matrix(dw)
    diff = float(torch.max(torch.abs(val_eager - val_compiled)))
    print(f"\nCorrectness (logpdf max abs diff): {diff:.2e}")


def benchmark_full_pipeline():
    """Benchmark the full KL-OED objective + jacobian pipeline.

    With automatic dispatch, TorchBkd always gets torch.compile.
    """
    torch.set_default_dtype(torch.float64)
    bkd = TorchBkd()
    nobs, ninner, nouter = 50, 1000, 1000

    print("\n" + "=" * 70)
    print("Full Pipeline Benchmark: KLOEDObjective (PyTorch auto-dispatched)")
    print(f"Problem size: nobs={nobs}, ninner={ninner}, nouter={nouter}")
    print("=" * 70)

    benchmark = LinearGaussianOEDBenchmark(
        nobs=nobs, degree=5, noise_std=0.5, prior_std=1.0, bkd=bkd,
    )

    np.random.seed(42)
    noise_variances = bkd.asarray(np.full(nobs, benchmark.noise_var()))

    _, inner_shapes = benchmark.generate_data(ninner, seed=42)
    _, outer_shapes = benchmark.generate_data(nouter, seed=123)
    latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

    # Auto-dispatched (TorchBkd → torch.compile)
    inner_like = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
    obj = KLOEDObjective(
        inner_like,
        outer_shapes, latent_samples,
        inner_shapes, None, None, bkd,
    )

    dw = bkd.ones((nobs, 1))

    # torch.compile warmup
    print("\nWarming up torch.compile (first calls trigger compilation)...")
    t0 = time.perf_counter()
    obj(dw)
    obj.jacobian(dw)
    warmup_time = time.perf_counter() - t0
    print(f"  Compile warmup: {warmup_time:.3f}s")

    # Benchmark __call__
    mean_val, std_val = time_fn(lambda: obj(dw))
    print(f"\nKLOEDObjective.__call__:")
    print(f"  {mean_val*1000:.1f} +/- {std_val*1000:.1f} ms")

    # Benchmark jacobian
    mean_jac, std_jac = time_fn(lambda: obj.jacobian(dw))
    print(f"\nKLOEDObjective.jacobian:")
    print(f"  {mean_jac*1000:.1f} +/- {std_jac*1000:.1f} ms")

    # Combined
    def run_combined():
        obj(dw)
        obj.jacobian(dw)

    mean_comb, std_comb = time_fn(run_combined)
    print(f"\nKLOEDObjective call + jacobian (optimizer iteration):")
    print(f"  {mean_comb*1000:.1f} +/- {std_comb*1000:.1f} ms")


if __name__ == "__main__":
    benchmark_kernels()
    benchmark_full_pipeline()
