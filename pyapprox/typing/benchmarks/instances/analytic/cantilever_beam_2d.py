"""Analytical 2D cantilever beam benchmark instance.

Provides the analytical 2D cantilever beam model as a benchmark with
Gaussian marginals for loads (X, Y) and material properties (E, R),
and Uniform marginals for cross-section dimensions (w, t).

Matches the legacy CantileverBeamModel variable distributions.
"""

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.typing.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.functions.algebraic.cantilever_beam_2d import (
    CantileverBeam2DAnalytical,
)
from pyapprox.typing.probability.univariate.gaussian import GaussianMarginal
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint


def cantilever_beam_2d_analytical(
    bkd: Backend[Array],
    length: float = 100.0,
) -> BenchmarkWithPrior:
    """Create an analytical 2D cantilever beam benchmark.

    The model computes [max_stress, tip_displacement] from
    (X, Y, E, R, w, t) using closed-form Euler-Bernoulli beam theory.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    length : float
        Beam length.

    Returns
    -------
    BenchmarkWithPrior
        Benchmark instance.
    """
    func = CantileverBeam2DAnalytical(length=length, bkd=bkd)

    # Same distributions as the legacy CantileverBeamModel
    prior = IndependentJoint(
        [
            GaussianMarginal(500.0, 100.0, bkd),       # X: horizontal load
            GaussianMarginal(1000.0, 100.0, bkd),      # Y: vertical load
            GaussianMarginal(2.9e7, 1.45e6, bkd),      # E: Young's modulus
            GaussianMarginal(40000.0, 2000.0, bkd),     # R: yield stress
            UniformMarginal(1.0, 4.0, bkd),             # w: width
            UniformMarginal(1.0, 4.0, bkd),             # t: depth
        ],
        bkd,
    )

    # Bounds: 4-sigma for Gaussians, exact for Uniforms
    bounds = bkd.array([
        [100.0, 900.0],           # X
        [600.0, 1400.0],          # Y
        [2.32e7, 3.48e7],         # E
        [32000.0, 48000.0],       # R
        [1.0, 4.0],              # w
        [1.0, 4.0],              # t
    ])
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    return BenchmarkWithPrior(
        _name="cantilever_beam_2d_analytical",
        _function=func,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            "Analytical 2D cantilever beam: (X,Y,E,R,w,t) -> "
            "[max stress, tip displacement]"
        ),
    )


@BenchmarkRegistry.register(
    "cantilever_beam_2d_analytical",
    category="analytic",
    description=(
        "Analytical 2D cantilever beam with closed-form max stress "
        "and tip displacement"
    ),
)
def _cantilever_beam_2d_analytical_factory(
    bkd: Backend[Array],
) -> BenchmarkWithPrior:
    return cantilever_beam_2d_analytical(bkd)
