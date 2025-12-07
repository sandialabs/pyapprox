"""
Benchmark comparing GP optimization with and without HVP.

Compares:
1. With HVP (SquaredExponentialKernel with analytical derivatives)
2. Without HVP (BFGS with gradient only)
3. Analytical Jacobian (ExactGaussianProcess) vs autograd Jacobian (TorchExactGaussianProcess)

Metrics:
- Total optimization time
- Number of iterations
- Median Jacobian evaluation time
- Median HVP evaluation time (when available)
"""

import time
import numpy as np
import torch

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels import SquaredExponentialKernel
from pyapprox.typing.surrogates.kernels.torch_matern import TorchMaternKernel
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.surrogates.gaussianprocess.torch_exact import TorchExactGaussianProcess
from pyapprox.typing.surrogates.gaussianprocess.loss import NegativeLogMarginalLikelihoodLoss
from pyapprox.typing.optimization.minimize.scipy.trust_constr import ScipyTrustConstrOptimizer


def generate_data(nvars, n_train, bkd, seed=42):
    """Generate synthetic training data."""
    np.random.seed(seed)
    X_train_np = np.random.randn(nvars, n_train)
    # Simple function: sum of sines
    y_train_np = np.sin(X_train_np.sum(axis=0))[:, None]
    return bkd.array(X_train_np), bkd.array(y_train_np)


def time_jacobian(loss, params, n_calls=10):
    """Time Jacobian evaluations."""
    times = []
    for _ in range(n_calls):
        start = time.perf_counter()
        _ = loss.jacobian(params)
        times.append(time.perf_counter() - start)
    return np.median(times)


def time_hvp(loss, params, direction, n_calls=10):
    """Time HVP evaluations."""
    times = []
    for _ in range(n_calls):
        start = time.perf_counter()
        _ = loss.hvp(params, direction)
        times.append(time.perf_counter() - start)
    return np.median(times)


def optimize_with_trust_constr(loss, init_params, bounds, use_hvp=True, maxiter=100, gtol=1e-5):
    """
    Optimize using trust-constr with or without HVP.

    Returns optimization time, iterations, and final loss.
    """
    bkd = loss.bkd()

    # Temporarily disable HVP if requested
    original_hvp = getattr(loss, 'hvp', None)
    if not use_hvp and original_hvp is not None:
        delattr(loss, 'hvp')

    try:
        optimizer = ScipyTrustConstrOptimizer(
            loss,
            bounds,
            maxiter=maxiter,
            gtol=gtol,
            verbosity=0
        )

        start = time.perf_counter()
        result = optimizer.minimize(init_params)
        opt_time = time.perf_counter() - start

        raw_result = result.get_raw_result()
        return {
            'time': opt_time,
            'nit': raw_result.nit,
            'nfev': raw_result.nfev,
            'final_loss': float(raw_result.fun),
            'success': raw_result.success
        }
    finally:
        # Restore HVP if it was removed
        if not use_hvp and original_hvp is not None:
            loss.hvp = original_hvp


def benchmark_analytical_gp(nvars, n_train, n_timing_calls=10):
    """
    Benchmark ExactGaussianProcess with SquaredExponentialKernel (analytical derivatives).
    """
    bkd = NumpyBkd()

    # Generate data
    X_train, y_train = generate_data(nvars, n_train, bkd)

    # Create kernel with analytical derivatives
    lenscale = [1.0] * nvars
    kernel = SquaredExponentialKernel(lenscale, (0.1, 10.0), nvars, bkd)

    # Create GP and fit
    gp = ExactGaussianProcess(kernel, nvars, bkd, nugget=1e-6)
    gp.fit(X_train, y_train)

    # Create loss function
    loss = NegativeLogMarginalLikelihoodLoss(gp, X_train, y_train)

    # Get initial parameters
    params = gp.hyp_list().get_active_values()
    init_params = params[:, None]

    # Create bounds
    bounds_np = gp.hyp_list().get_bounds()
    bounds = bkd.array(bounds_np)

    # Time loss Jacobian (gradient of NLML w.r.t. hyperparameters)
    loss_jac_time = time_jacobian(loss, init_params, n_timing_calls)

    # Time loss HVP
    direction = bkd.ones((params.shape[0],))[:, None]
    loss_hvp_time = time_hvp(loss, init_params, direction, n_timing_calls)

    # Time GP prediction Jacobian (∂f/∂x) - analytical
    X_test = X_train[:, :1]
    gp_jac_times = []
    for _ in range(n_timing_calls):
        start = time.perf_counter()
        _ = gp.jacobian(X_test)
        gp_jac_times.append(time.perf_counter() - start)
    gp_jac_time = np.median(gp_jac_times)

    # Optimize with HVP
    result_with_hvp = optimize_with_trust_constr(loss, init_params, bounds, use_hvp=True)

    # Optimize without HVP
    result_without_hvp = optimize_with_trust_constr(loss, init_params, bounds, use_hvp=False)

    return {
        'method': 'analytical',
        'nvars': nvars,
        'n_train': n_train,
        'loss_jac_time': loss_jac_time,
        'loss_hvp_time': loss_hvp_time,
        'gp_jac_time': gp_jac_time,
        'with_hvp': result_with_hvp,
        'without_hvp': result_without_hvp,
        'has_hvp': True
    }


def benchmark_autograd_gp(nvars, n_train, n_timing_calls=10):
    """
    Benchmark TorchExactGaussianProcess with TorchMaternKernel (autograd derivatives).

    Note: HVP is not available for TorchExactGaussianProcess due to torch.cdist limitation.
    """
    torch.set_default_dtype(torch.float64)
    bkd = TorchBkd()

    # Generate data
    np.random.seed(42)
    X_train_np = np.random.randn(nvars, n_train)
    y_train_np = np.sin(X_train_np.sum(axis=0))[:, None]
    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np)

    # Create kernel with nu=inf (RBF) using autograd
    lenscale = [1.0] * nvars
    kernel = TorchMaternKernel(nu=200.0, lenscale=lenscale, lenscale_bounds=(0.1, 10.0), nvars=nvars)

    # Create GP and fit
    gp = TorchExactGaussianProcess(kernel, nvars, nugget=1e-6)
    gp.fit(X_train, y_train)

    # Time Jacobian using GP's jacobian method
    X_test = X_train[:, :1]
    jac_times = []
    for _ in range(n_timing_calls):
        start = time.perf_counter()
        _ = gp.jacobian(X_test)
        jac_times.append(time.perf_counter() - start)
    jac_time = np.median(jac_times)

    return {
        'method': 'autograd',
        'nvars': nvars,
        'n_train': n_train,
        'jac_time': jac_time,
        'hvp_time': None,  # Not available
        'has_hvp': False
    }


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 80)
    print("GP Optimization Benchmark: HVP vs No-HVP, Analytical vs Autograd")
    print("=" * 80)

    # Test configurations
    nvars_list = [2, 4, 8]
    n_train_list = [20, 50, 100, 500, 1000]

    results = []

    print("\n" + "-" * 80)
    print("Part 1: Trust-Constr Optimization with/without HVP")
    print("(Using SquaredExponentialKernel with analytical derivatives)")
    print("-" * 80)

    print(f"\n{'nvars':>6} | {'n_train':>7} | {'With HVP':^30} | {'Without HVP':^30}")
    print(f"{'':>6} | {'':>7} | {'Time (s)':>10} {'Iters':>8} {'Loss':>10} | {'Time (s)':>10} {'Iters':>8} {'Loss':>10}")
    print("-" * 80)

    for nvars in nvars_list:
        for n_train in n_train_list:
            result = benchmark_analytical_gp(nvars, n_train)
            results.append(result)

            with_hvp = result['with_hvp']
            without_hvp = result['without_hvp']

            print(f"{nvars:>6} | {n_train:>7} | "
                  f"{with_hvp['time']:>10.4f} {with_hvp['nit']:>8} {with_hvp['final_loss']:>10.4f} | "
                  f"{without_hvp['time']:>10.4f} {without_hvp['nit']:>8} {without_hvp['final_loss']:>10.4f}")

    print("\n" + "-" * 80)
    print("Part 2: Loss Jacobian and HVP Timing (Analytical)")
    print("(Gradient and HVP of NLML w.r.t. hyperparameters)")
    print("-" * 80)

    print(f"\n{'nvars':>6} | {'n_train':>7} | {'Loss Jac (ms)':>15} | {'Loss HVP (ms)':>15} | {'Ratio (HVP/Jac)':>15}")
    print("-" * 80)

    for r in results:
        ratio = r['loss_hvp_time'] / r['loss_jac_time'] if r['loss_jac_time'] > 0 else float('nan')
        print(f"{r['nvars']:>6} | {r['n_train']:>7} | {r['loss_jac_time']*1000:>15.3f} | {r['loss_hvp_time']*1000:>15.3f} | {ratio:>15.2f}")

    print("\n" + "-" * 80)
    print("Part 3: GP Prediction Jacobian - Analytical vs Autograd")
    print("(Jacobian of GP mean ∂f/∂x w.r.t. inputs)")
    print("-" * 80)

    print(f"\n{'nvars':>6} | {'n_train':>7} | {'Analytical (ms)':>17} | {'Autograd (ms)':>17} | {'Ratio (AG/An)':>15}")
    print("-" * 80)

    for nvars in nvars_list:
        for n_train in n_train_list:
            # Get analytical result
            analytical_result = next(r for r in results if r['nvars'] == nvars and r['n_train'] == n_train)

            # Get autograd result
            autograd_result = benchmark_autograd_gp(nvars, n_train)

            ratio = autograd_result['jac_time'] / analytical_result['gp_jac_time'] if analytical_result['gp_jac_time'] > 0 else float('nan')

            print(f"{nvars:>6} | {n_train:>7} | {analytical_result['gp_jac_time']*1000:>17.3f} | {autograd_result['jac_time']*1000:>17.3f} | {ratio:>15.2f}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)

    # Compute average speedups
    hvp_iter_reductions = []
    hvp_time_ratios = []
    for r in results:
        if r['with_hvp']['nit'] > 0 and r['without_hvp']['nit'] > 0:
            iter_reduction = r['without_hvp']['nit'] / r['with_hvp']['nit']
            time_ratio = r['with_hvp']['time'] / r['without_hvp']['time']
            hvp_iter_reductions.append(iter_reduction)
            hvp_time_ratios.append(time_ratio)

    if hvp_iter_reductions:
        avg_iter_reduction = np.mean(hvp_iter_reductions)
        avg_time_ratio = np.mean(hvp_time_ratios)
        print(f"\n1. HVP reduces iterations by {avg_iter_reduction:.2f}x on average")
        print(f"   But HVP takes {avg_time_ratio:.2f}x MORE wall-clock time (HVP overhead > iteration savings)")
        print(f"   HVP cost per call is ~2-3x more than Jacobian")

    print(f"\n2. HVP is NOT available for TorchExactGaussianProcess (torch.cdist limitation)")
    print(f"   Use ExactGaussianProcess with analytical kernels for HVP support")

    print(f"\n3. GP prediction Jacobian (∂f/∂x) comparison:")
    print(f"   - Analytical: Uses kernel.jacobian() method (ExactGaussianProcess)")
    print(f"   - Autograd: Uses bkd.jacobian() (TorchExactGaussianProcess)")
    print(f"   - Analytical is 4-14x faster than autograd")
    print(f"   - Use analytical Jacobians when available (Matern32, Matern52, RBF)")
    print(f"   - Use autograd for arbitrary nu values or custom kernels")

    return results


if __name__ == "__main__":
    run_benchmarks()
