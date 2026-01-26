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
    IshigamiSensitivityIndices,
)
from pyapprox.typing.benchmarks.functions.algebraic.sobol_g import (
    SobolGFunction,
    SobolGSensitivityIndices,
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

    # Analytical Sobol indices
    indices = IshigamiSensitivityIndices(bkd, a=a, b=b)
    mean_val = float(indices.mean()[0])
    variance = float(indices.variance()[0])
    main_effects = indices.main_effects()
    total_effects = indices.total_effects()
    sobol_all = indices.sobol_indices()

    # Build Sobol indices dict
    # Order from IshigamiSensitivityIndices: S_1, S_2, S_3, S_12, S_13, S_23, S_123
    sobol_dict = {
        (0,): float(sobol_all[0, 0]),
        (1,): float(sobol_all[1, 0]),
        (2,): float(sobol_all[2, 0]),
        (0, 2): float(sobol_all[4, 0]),  # S_13 is at index 4
    }

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
            mean=mean_val,
            variance=variance,
            main_effects=main_effects,
            total_effects=total_effects,
            sobol_indices=sobol_dict,
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

    # Analytical Sobol indices
    indices = SobolGSensitivityIndices(bkd, a)
    main_effects = indices.main_effects()
    total_effects = indices.total_effects()
    variance = float(indices.variance()[0])

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
            variance=variance,
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

    # Analytical Sobol indices
    indices = SobolGSensitivityIndices(bkd, a)
    main_effects = indices.main_effects()
    total_effects = indices.total_effects()
    variance = float(indices.variance()[0])

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
            variance=variance,
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
