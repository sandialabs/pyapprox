"""Fixed sensitivity analysis benchmark instances.

These are pre-configured benchmark instances with standard parameters
and known ground truth values.
"""

import math
from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.typing.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
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

    main_effects = np.array([D1 / variance, D2 / variance, D3 / variance])
    total_effects = np.array([
        (D1 + D13) / variance,
        D2 / variance,
        D13 / variance,
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
