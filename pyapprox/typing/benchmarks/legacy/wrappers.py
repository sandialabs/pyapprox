"""Wrappers for specific legacy benchmarks.

Each wrapper function creates a new-style Benchmark by wrapping
a legacy benchmark without modifying the original code.
"""

import math
from typing import Any

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import (
    Benchmark,
    BenchmarkWithPrior,
    BoxDomain,
)
from pyapprox.typing.benchmarks.ground_truth import (
    SensitivityGroundTruth,
    QuadratureGroundTruth,
)
from pyapprox.typing.benchmarks.legacy.adapter import (
    LegacyFunctionWithJacobianAdapter,
)
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry


def wrap_legacy_ishigami(
    bkd: Backend[Array],
    a: float = 7.0,
    b: float = 0.1,
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth]:
    """Wrap legacy IshigamiBenchmark.

    Creates a new-style benchmark by wrapping the legacy Ishigami
    implementation without modifying the original code.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    a : float, optional
        Parameter a for Ishigami function. Default is 7.0.
    b : float, optional
        Parameter b for Ishigami function. Default is 0.1.

    Returns
    -------
    BenchmarkWithPrior[Array, SensitivityGroundTruth]
        The wrapped benchmark.
    """
    # Import legacy (only when needed to avoid circular imports)
    from pyapprox.benchmarks.algebraic import IshigamiBenchmark
    from pyapprox.util.backends.numpy import NumpyMixin

    # Create legacy benchmark with numpy backend
    legacy_bm = IshigamiBenchmark(NumpyMixin, a=a, b=b)

    # Extract ground truth from legacy
    pi = math.pi

    # Analytical values from Ishigami & Homma (1990)
    variance = a**2 / 8 + b * pi**4 / 5 + b**2 * pi**8 / 18 + 0.5
    D1 = b * pi**4 / 5 + b**2 * pi**8 / 50 + 0.5
    D2 = a**2 / 8
    D3 = 0.0
    D13 = b**2 * pi**8 / 18 - b**2 * pi**8 / 50

    main_effects = np.array([D1 / variance, D2 / variance, D3 / variance])
    total_effects = np.array([
        (D1 + D13) / variance,
        D2 / variance,
        D13 / variance,
    ])

    ground_truth = SensitivityGroundTruth(
        mean=float(legacy_bm.mean()),
        variance=variance,
        main_effects=main_effects,
        total_effects=total_effects,
        sobol_indices={
            (0,): D1 / variance,
            (1,): D2 / variance,
            (2,): D3 / variance,
            (0, 2): D13 / variance,
        },
    )

    # Wrap the model
    adapted_function = LegacyFunctionWithJacobianAdapter(
        bkd=bkd,
        legacy_model=legacy_bm.model(),
        nvars=legacy_bm.nvars(),
        nqoi=legacy_bm.nqoi(),
    )

    # Create domain
    domain = BoxDomain(
        _bounds=bkd.array([[-pi, pi], [-pi, pi], [-pi, pi]]),
        _bkd=bkd,
    )

    # Create new benchmark with wrapped function
    return BenchmarkWithPrior(
        _name="ishigami_3d_legacy",
        _function=adapted_function,
        _domain=domain,
        _ground_truth=ground_truth,
        _description="Ishigami function (wrapped from legacy)",
        _reference="Ishigami, T. and Homma, T. (1990)",
        _prior=None,  # No prior in new style - would need probability module
    )


def wrap_legacy_genz(
    bkd: Backend[Array],
    nvars: int = 2,
    genz_type: str = "oscillatory",
) -> Benchmark[Array, QuadratureGroundTruth]:
    """Wrap legacy GenzBenchmark.

    Creates a new-style benchmark by wrapping the legacy Genz
    implementation without modifying the original code.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    nvars : int, optional
        Number of dimensions. Default is 2.
    genz_type : str, optional
        Type of Genz function. Options: 'oscillatory', 'product_peak',
        'corner_peak', 'gaussian'. Default is 'oscillatory'.

    Returns
    -------
    Benchmark[Array, QuadratureGroundTruth]
        The wrapped benchmark.
    """
    # Import legacy (only when needed)
    from pyapprox.benchmarks.genz import GenzBenchmark
    from pyapprox.util.backends.numpy import NumpyMixin

    # Map new names to legacy names (legacy uses underscores)
    genz_type_map = {
        "oscillatory": "oscillatory",
        "product_peak": "product_peak",
        "corner_peak": "corner_peak",
        "gaussian": "gaussian",
    }
    legacy_type = genz_type_map.get(genz_type, genz_type)

    # Create legacy benchmark - GenzBenchmark(name, nvars, ..., backend=...)
    legacy_bm = GenzBenchmark(legacy_type, nvars, backend=NumpyMixin)

    # Get integral from legacy
    integral = float(legacy_bm.integral())

    ground_truth = QuadratureGroundTruth(
        integral=integral,
    )

    # Wrap the model (Genz models typically don't have Jacobian)
    from pyapprox.typing.benchmarks.legacy.adapter import LegacyFunctionAdapter
    adapted_function = LegacyFunctionAdapter(
        bkd=bkd,
        legacy_model=legacy_bm.model(),
        nvars=legacy_bm.nvars(),
        nqoi=legacy_bm.nqoi(),
    )

    # Create domain
    domain = BoxDomain(
        _bounds=bkd.array([[0.0, 1.0]] * nvars),
        _bkd=bkd,
    )

    return Benchmark(
        _name=f"genz_{genz_type}_{nvars}d_legacy",
        _function=adapted_function,
        _domain=domain,
        _ground_truth=ground_truth,
        _description=f"Genz {genz_type} function (wrapped from legacy)",
        _reference="Genz, A. (1984)",
    )


# Register legacy wrappers
@BenchmarkRegistry.register(
    "ishigami_3d_legacy",
    category="sensitivity",
    description="Ishigami function (wrapped from legacy)",
)
def _ishigami_3d_legacy_factory(
    bkd: Backend[Array],
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth]:
    return wrap_legacy_ishigami(bkd)


@BenchmarkRegistry.register(
    "genz_oscillatory_2d_legacy",
    category="quadrature",
    description="Genz oscillatory 2D (wrapped from legacy)",
)
def _genz_oscillatory_2d_legacy_factory(
    bkd: Backend[Array],
) -> Benchmark[Array, QuadratureGroundTruth]:
    return wrap_legacy_genz(bkd, nvars=2, genz_type="oscillatory")


@BenchmarkRegistry.register(
    "genz_product_peak_2d_legacy",
    category="quadrature",
    description="Genz product peak 2D (wrapped from legacy)",
)
def _genz_product_peak_2d_legacy_factory(
    bkd: Backend[Array],
) -> Benchmark[Array, QuadratureGroundTruth]:
    return wrap_legacy_genz(bkd, nvars=2, genz_type="product_peak")


@BenchmarkRegistry.register(
    "genz_corner_peak_2d_legacy",
    category="quadrature",
    description="Genz corner peak 2D (wrapped from legacy)",
)
def _genz_corner_peak_2d_legacy_factory(
    bkd: Backend[Array],
) -> Benchmark[Array, QuadratureGroundTruth]:
    return wrap_legacy_genz(bkd, nvars=2, genz_type="corner_peak")


@BenchmarkRegistry.register(
    "genz_gaussian_2d_legacy",
    category="quadrature",
    description="Genz gaussian 2D (wrapped from legacy)",
)
def _genz_gaussian_2d_legacy_factory(
    bkd: Backend[Array],
) -> Benchmark[Array, QuadratureGroundTruth]:
    return wrap_legacy_genz(bkd, nvars=2, genz_type="gaussian")
