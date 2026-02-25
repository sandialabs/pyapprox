"""Analytical cantilever beam benchmark instance.

Provides the analytical 1D cantilever beam model as a benchmark with
Beta marginals for the Young's moduli E1 (skin) and E2 (core).
"""

from pyapprox.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.benchmarks.functions.algebraic.cantilever_beam import (
    CantileverBeam1DAnalytical,
)
from pyapprox.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.util.backends.protocols import Array, Backend


def cantilever_beam_1d_analytical(
    bkd: Backend[Array],
    length: float = 100.0,
    height: float = 30.0,
    skin_thickness: float = 5.0,
    q0: float = 10.0,
) -> BenchmarkWithPrior:
    """Create an analytical 1D cantilever beam benchmark.

    The model computes [tip_deflection, max_curvature] from (E1, E2)
    using closed-form formulas for a composite beam under linearly
    increasing load q(x) = q0*x/L.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    length : float
        Beam length.
    height : float
        Cross-section height.
    skin_thickness : float
        Thickness of each skin layer.
    q0 : float
        Load magnitude.

    Returns
    -------
    BenchmarkWithPrior
        Benchmark instance.
    """
    func = CantileverBeam1DAnalytical(
        length=length,
        height=height,
        skin_thickness=skin_thickness,
        q0=q0,
        bkd=bkd,
    )

    # Beta(2,5) marginals on [18000, 22000] for E1, [4500, 5500] for E2
    prior = IndependentJoint(
        [
            BetaMarginal(2.0, 5.0, bkd, lb=18000.0, ub=22000.0),
            BetaMarginal(2.0, 5.0, bkd, lb=4500.0, ub=5500.0),
        ],
        bkd,
    )

    bounds = bkd.array([[18000.0, 22000.0], [4500.0, 5500.0]])
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    return BenchmarkWithPrior(
        _name="cantilever_beam_1d_analytical",
        _function=func,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            "Analytical 1D cantilever beam: (E1, E2) -> [tip deflection, max curvature]"
        ),
    )


@BenchmarkRegistry.register(
    "cantilever_beam_1d_analytical",
    category="analytic",
    description=(
        "Analytical 1D cantilever beam with closed-form tip deflection "
        "and max curvature"
    ),
)
def _cantilever_beam_1d_analytical_factory(
    bkd: Backend[Array],
) -> BenchmarkWithPrior:
    return cantilever_beam_1d_analytical(bkd)
