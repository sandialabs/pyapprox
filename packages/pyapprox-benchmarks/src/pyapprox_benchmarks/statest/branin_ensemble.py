"""Branin ensemble problem — no analytical ground truth."""

import math
from typing import Generic

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.functions.multifidelity.branin_ensemble import (
    BraninModelFunction,
)
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)


class BraninEnsembleProblem(Generic[Array]):
    """3-model Branin ensemble problem — no ground truth.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        b_std = 5.1 / (4 * math.pi**2)
        c_std = 5.0 / math.pi
        t_std = 1.0 / (8 * math.pi)

        models = [
            BraninModelFunction(bkd),
            BraninModelFunction(
                bkd, b=b_std * 1.2, c=c_std * 0.9,
                t=t_std * 1.1, shift=2.0,
            ),
            BraninModelFunction(
                bkd, b=b_std * 0.8, c=c_std * 1.2,
                t=t_std * 0.8, shift=5.0,
            ),
        ]
        costs = bkd.array([1.0, 0.1, 0.01])
        prior = IndependentJoint(
            [
                UniformMarginal(-5.0, 10.0, bkd),
                UniformMarginal(0.0, 15.0, bkd),
            ],
            bkd,
        )
        self._problem = MultifidelityForwardUQProblem(
            "branin_ensemble_3model",
            models,
            costs,
            prior,
            description="3-model multi-fidelity Branin ensemble",
        )

    def problem(
        self,
    ) -> MultifidelityForwardUQProblem[FunctionProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]
