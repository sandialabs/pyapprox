"""Analytical 2D cantilever beam forward UQ problem.

Provides the analytical 2D cantilever beam model as a ForwardUQProblem
with Gaussian marginals for loads (X, Y) and material properties (E, R),
and Uniform marginals for cross-section dimensions (w, t).

Matches the legacy CantileverBeamModel variable distributions.
"""

from pyapprox_benchmarks.functions.algebraic.cantilever_beam_2d import (
    CantileverBeam2DAnalytical,
)
from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def build_cantilever_beam_2d_analytical(
    bkd: Backend[Array],
    length: float = 100.0,
) -> ForwardUQProblem:
    """Create an analytical 2D cantilever beam forward UQ problem.

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
    ForwardUQProblem
        Problem instance.
    """
    func = CantileverBeam2DAnalytical(length=length, bkd=bkd)

    # Same distributions as the legacy CantileverBeamModel
    prior = IndependentJoint(
        [
            GaussianMarginal(500.0, 100.0, bkd),  # X: horizontal load
            GaussianMarginal(1000.0, 100.0, bkd),  # Y: vertical load
            GaussianMarginal(2.9e7, 1.45e6, bkd),  # E: Young's modulus
            GaussianMarginal(40000.0, 2000.0, bkd),  # R: yield stress
            UniformMarginal(1.0, 4.0, bkd),  # w: width
            UniformMarginal(1.0, 4.0, bkd),  # t: depth
        ],
        bkd,
    )

    return ForwardUQProblem(
        name="cantilever_beam_2d_analytical",
        function=func,
        prior=prior,
        description=(
            "Analytical 2D cantilever beam: (X,Y,E,R,w,t) -> "
            "[max stress, tip displacement]"
        ),
    )
