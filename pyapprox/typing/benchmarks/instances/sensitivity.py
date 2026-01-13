"""Fixed sensitivity analysis benchmark instances.

These are pre-configured benchmark instances with standard parameters
and known ground truth values.
"""

import math

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.typing.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
)
from pyapprox.typing.benchmarks.functions.algebraic.sobol_g import (
    SobolGFunction,
    sobol_g_indices,
)
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint


def ishigami_3d(
    bkd: Backend[Array],
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth]:
    """Create the standard Ishigami benchmark.

    Standard Ishigami benchmark with a=7, b=0.1 on [-pi, pi]^3.

    Returns fixed benchmark instance - NOT configurable.
    Includes uniform prior U[-pi, pi]^3 for sensitivity analysis.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    BenchmarkWithPrior[Array, SensitivityGroundTruth]
        The Ishigami benchmark instance.

    References
    ----------
    Ishigami, T. and Homma, T. (1990). "An importance quantification technique
    in uncertainty analysis for computer models."
    """
    a, b = 7.0, 0.1
    pi = math.pi

    # Analytical ground truth (from Ishigami & Homma, 1990)
    # Variance decomposition for f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1)
    # with U[-pi, pi]^3 prior
    variance = a**2 / 8 + b * pi**4 / 5 + b**2 * pi**8 / 18 + 0.5
    D1 = b * pi**4 / 5 + b**2 * pi**8 / 50 + 0.5
    D2 = a**2 / 8
    D3 = 0.0
    D13 = b**2 * pi**8 / 18 - b**2 * pi**8 / 50

    main_effects = bkd.array([[D1 / variance], [D2 / variance], [D3 / variance]])
    total_effects = bkd.array([
        [(D1 + D13) / variance],
        [D2 / variance],
        [D13 / variance],
    ])

    # Standard uniform prior for sensitivity analysis
    prior = IndependentJoint(
        [
            UniformMarginal(-pi, pi, bkd),
            UniformMarginal(-pi, pi, bkd),
            UniformMarginal(-pi, pi, bkd),
        ],
        bkd,
    )

    return BenchmarkWithPrior(
        _name="ishigami_3d",
        _function=IshigamiFunction(bkd, a=a, b=b),
        _domain=BoxDomain(
            _bounds=bkd.array([[-pi, pi], [-pi, pi], [-pi, pi]]),
            _bkd=bkd,
        ),
        _ground_truth=SensitivityGroundTruth(
            mean=a / 2,
            variance=variance,
            main_effects=main_effects,
            total_effects=total_effects,
            sobol_indices={
                (0,): D1 / variance,
                (1,): D2 / variance,
                (2,): D3 / variance,
                (0, 2): D13 / variance,
            },
        ),
        _prior=prior,
        _description="Ishigami function - standard sensitivity analysis benchmark",
        _reference="Ishigami, T. and Homma, T. (1990)",
    )


@BenchmarkRegistry.register(
    "ishigami_3d",
    category="sensitivity",
    description="Standard 3D Ishigami function for sensitivity analysis",
)
def _ishigami_3d_factory(bkd: Backend[Array]) -> BenchmarkWithPrior[
    Array, SensitivityGroundTruth
]:
    return ishigami_3d(bkd)


def sobol_g_6d(
    bkd: Backend[Array],
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth]:
    """Create the standard 6D Sobol G-function benchmark.

    Standard configuration with a = [0, 1, 4.5, 9, 99, 99].
    This gives decreasing importance from x1 to x6.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    BenchmarkWithPrior[Array, SensitivityGroundTruth]
        The Sobol G benchmark instance.

    References
    ----------
    Sobol, I.M. (1993). "Sensitivity analysis for non-linear mathematical
    models."
    """
    a = [0.0, 1.0, 4.5, 9.0, 99.0, 99.0]
    nvars = len(a)

    # Analytical Sobol indices (returns backend arrays with shape (nvars, 1))
    main_effects, total_effects, variance = sobol_g_indices(a, bkd)

    # Build Sobol indices dict for first-order
    sobol_dict = {(i,): float(main_effects[i, 0]) for i in range(nvars)}

    # Standard uniform prior on [0, 1]^6
    prior = IndependentJoint(
        [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
        bkd,
    )

    return BenchmarkWithPrior(
        _name="sobol_g_6d",
        _function=SobolGFunction(bkd, a=a),
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=SensitivityGroundTruth(
            mean=1.0,  # Product of means, each g_i has mean 1
            variance=float(variance),
            main_effects=main_effects,
            total_effects=total_effects,
            sobol_indices=sobol_dict,
        ),
        _prior=prior,
        _description="Sobol G-function - 6D with standard importance parameters",
        _reference="Sobol, I.M. (1993)",
    )


def sobol_g_4d(
    bkd: Backend[Array],
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth]:
    """Create a 4D Sobol G-function benchmark.

    Configuration with a = [0, 0, 0, 0] giving equal importance.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    BenchmarkWithPrior[Array, SensitivityGroundTruth]
        The Sobol G benchmark instance.
    """
    a = [0.0, 0.0, 0.0, 0.0]
    nvars = len(a)

    # Analytical Sobol indices (returns backend arrays with shape (nvars, 1))
    main_effects, total_effects, variance = sobol_g_indices(a, bkd)

    # Build Sobol indices dict for first-order
    sobol_dict = {(i,): float(main_effects[i, 0]) for i in range(nvars)}

    # Standard uniform prior on [0, 1]^4
    prior = IndependentJoint(
        [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
        bkd,
    )

    return BenchmarkWithPrior(
        _name="sobol_g_4d",
        _function=SobolGFunction(bkd, a=a),
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=SensitivityGroundTruth(
            mean=1.0,
            variance=float(variance),
            main_effects=main_effects,
            total_effects=total_effects,
            sobol_indices=sobol_dict,
        ),
        _prior=prior,
        _description="Sobol G-function - 4D with equal importance",
        _reference="Sobol, I.M. (1993)",
    )


@BenchmarkRegistry.register(
    "sobol_g_6d",
    category="sensitivity",
    description="Standard 6D Sobol G-function for sensitivity analysis",
)
def _sobol_g_6d_factory(bkd: Backend[Array]) -> BenchmarkWithPrior[
    Array, SensitivityGroundTruth
]:
    return sobol_g_6d(bkd)


@BenchmarkRegistry.register(
    "sobol_g_4d",
    category="sensitivity",
    description="4D Sobol G-function with equal importance",
)
def _sobol_g_4d_factory(bkd: Backend[Array]) -> BenchmarkWithPrior[
    Array, SensitivityGroundTruth
]:
    return sobol_g_4d(bkd)
