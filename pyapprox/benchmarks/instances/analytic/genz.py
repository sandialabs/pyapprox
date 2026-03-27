"""Genz benchmark instances.

Standard Genz family functions for quadrature/integration testing with
known analytical integrals.
"""

from typing import Generic, Optional

from pyapprox.benchmarks.benchmark import Benchmark, BoxDomain
from pyapprox.benchmarks.functions.genz import (
    CornerPeakFunction,
    GaussianPeakFunction,
    OscillatoryFunction,
    ProductPeakFunction,
)
from pyapprox.benchmarks.ground_truth import QuadratureGroundTruth
from pyapprox.benchmarks.protocols import DomainProtocol
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class GenzBenchmark(Generic[Array]):
    """Genz benchmark wrapper.

    Satisfies: HasForwardModel, HasJacobian, HasReferenceIntegral,
    HasSmoothness, HasEstimatedEvaluationCost.
    """

    def __init__(self, inner: Benchmark[Array, QuadratureGroundTruth]) -> None:
        self._inner = inner

    def name(self) -> str:
        return self._inner.name()

    def function(self) -> FunctionProtocol[Array]:
        return self._inner.function()

    def domain(self) -> DomainProtocol[Array]:
        return self._inner.domain()

    def ground_truth(self) -> QuadratureGroundTruth:
        return self._inner.ground_truth()

    def jacobian(self, sample: Array) -> Array:
        return self._inner.function().jacobian(sample)

    def smoothness(self) -> str:
        return "analytic"

    def estimated_evaluation_cost(self) -> float:
        return 8.0e-06

    def reference_integral(self) -> Optional[float]:
        return self._inner.ground_truth().integral


def _get_genz_coefficients(nvars: int, decay: str = "none") -> list[float]:
    """Compute standard Genz coefficients with specified decay.

    Parameters
    ----------
    nvars : int
        Number of variables.
    decay : str
        Decay type: 'none', 'quadratic', 'quartic'.

    Returns
    -------
    list[float]
        Coefficients normalized to sum to 1.
    """
    if decay == "none":
        c = [(i + 0.5) / nvars for i in range(nvars)]
    elif decay == "quadratic":
        c = [1.0 / (i + 1) ** 2 for i in range(nvars)]
    elif decay == "quartic":
        c = [1.0 / (i + 1) ** 4 for i in range(nvars)]
    else:
        raise ValueError(f"Unknown decay type: {decay}")

    # Normalize to sum to 1
    total = sum(c)
    return [ci / total for ci in c]


def genz_oscillatory_2d(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    """Create a 2D oscillatory Genz benchmark.

    Standard 2D oscillatory function for quadrature testing.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    GenzBenchmark
        The benchmark instance.
    """
    nvars = 2
    c = _get_genz_coefficients(nvars)
    w = [0.25] * nvars

    func = OscillatoryFunction(bkd, c=c, w=w)

    inner = Benchmark(
        _name="genz_oscillatory_2d",
        _function=func,
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=QuadratureGroundTruth(
            integral=func.integrate(),
        ),
        _description="2D oscillatory Genz function for quadrature",
        _reference="Genz, A. (1984)",
    )

    return GenzBenchmark(inner)


def genz_product_peak_2d(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    """Create a 2D product peak Genz benchmark.

    Standard 2D product peak function for quadrature testing.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    GenzBenchmark
        The benchmark instance.
    """
    nvars = 2
    c = [2.0, 3.0]  # Width coefficients
    w = [0.5, 0.5]  # Peak at center

    func = ProductPeakFunction(bkd, c=c, w=w)

    inner = Benchmark(
        _name="genz_product_peak_2d",
        _function=func,
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=QuadratureGroundTruth(
            integral=func.integrate(),
        ),
        _description="2D product peak Genz function for quadrature",
        _reference="Genz, A. (1984)",
    )

    return GenzBenchmark(inner)


def genz_corner_peak_2d(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    """Create a 2D corner peak Genz benchmark.

    Standard 2D corner peak function for quadrature testing.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    GenzBenchmark
        The benchmark instance.
    """
    nvars = 2
    c = _get_genz_coefficients(nvars)

    func = CornerPeakFunction(bkd, c=c)

    inner = Benchmark(
        _name="genz_corner_peak_2d",
        _function=func,
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=QuadratureGroundTruth(
            integral=func.integrate(),
        ),
        _description="2D corner peak Genz function for quadrature",
        _reference="Genz, A. (1984)",
    )

    return GenzBenchmark(inner)


def genz_gaussian_peak_2d(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    """Create a 2D Gaussian peak Genz benchmark.

    Standard 2D Gaussian peak function for quadrature testing.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    GenzBenchmark
        The benchmark instance.
    """
    nvars = 2
    c = [2.0, 3.0]  # Width coefficients
    w = [0.5, 0.5]  # Peak at center

    func = GaussianPeakFunction(bkd, c=c, w=w)

    inner = Benchmark(
        _name="genz_gaussian_peak_2d",
        _function=func,
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=QuadratureGroundTruth(
            integral=func.integrate(),
        ),
        _description="2D Gaussian peak Genz function for quadrature",
        _reference="Genz, A. (1984)",
    )

    return GenzBenchmark(inner)


def genz_oscillatory_5d(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    """Create a 5D oscillatory Genz benchmark.

    Higher-dimensional oscillatory function for quadrature testing.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    GenzBenchmark
        The benchmark instance.
    """
    nvars = 5
    c = _get_genz_coefficients(nvars, decay="quadratic")
    w = [0.25] * nvars

    func = OscillatoryFunction(bkd, c=c, w=w)

    inner = Benchmark(
        _name="genz_oscillatory_5d",
        _function=func,
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=QuadratureGroundTruth(
            integral=func.integrate(),
        ),
        _description="5D oscillatory Genz function with quadratic decay",
        _reference="Genz, A. (1984)",
    )

    return GenzBenchmark(inner)


def genz_gaussian_peak_5d(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    """Create a 5D Gaussian peak Genz benchmark.

    Higher-dimensional Gaussian peak function for quadrature testing.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    GenzBenchmark
        The benchmark instance.
    """
    nvars = 5
    c = [1.0, 1.5, 2.0, 2.5, 3.0]  # Varying widths
    w = [0.5] * nvars  # Peak at center

    func = GaussianPeakFunction(bkd, c=c, w=w)

    inner = Benchmark(
        _name="genz_gaussian_peak_5d",
        _function=func,
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=QuadratureGroundTruth(
            integral=func.integrate(),
        ),
        _description="5D Gaussian peak Genz function",
        _reference="Genz, A. (1984)",
    )

    return GenzBenchmark(inner)


# TODO: These are good examples of specific Benchmarks, hoewver we have
# to create a lot of code each time we want to just add a different
# genz function /nvars combination. Is there a better way
# to handle combinatorial explosion of benchmarks such as this one. E.g
# a way to register benchmarks by looping over combinations, would
# this bloat the registry?

# Register benchmarks
@BenchmarkRegistry.register(
    "genz_oscillatory_2d",
    category="analytic",
    description="2D oscillatory Genz function",
)
def _genz_oscillatory_2d_factory(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    return genz_oscillatory_2d(bkd)


@BenchmarkRegistry.register(
    "genz_product_peak_2d",
    category="analytic",
    description="2D product peak Genz function",
)
def _genz_product_peak_2d_factory(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    return genz_product_peak_2d(bkd)


@BenchmarkRegistry.register(
    "genz_corner_peak_2d",
    category="analytic",
    description="2D corner peak Genz function",
)
def _genz_corner_peak_2d_factory(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    return genz_corner_peak_2d(bkd)


@BenchmarkRegistry.register(
    "genz_gaussian_peak_2d",
    category="analytic",
    description="2D Gaussian peak Genz function",
)
def _genz_gaussian_peak_2d_factory(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    return genz_gaussian_peak_2d(bkd)


@BenchmarkRegistry.register(
    "genz_oscillatory_5d",
    category="analytic",
    description="5D oscillatory Genz function with quadratic decay",
)
def _genz_oscillatory_5d_factory(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    return genz_oscillatory_5d(bkd)


@BenchmarkRegistry.register(
    "genz_gaussian_peak_5d",
    category="analytic",
    description="5D Gaussian peak Genz function",
)
def _genz_gaussian_peak_5d_factory(
    bkd: Backend[Array],
) -> GenzBenchmark[Array]:
    return genz_gaussian_peak_5d(bkd)
