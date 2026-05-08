"""Forrester ensemble problem — no analytical ground truth."""

from typing import Generic

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.functions.multifidelity.forrester_ensemble import (
    ForresterModelFunction,
)
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)


class ForresterEnsembleProblem(Generic[Array]):
    """2-model Forrester ensemble problem — no ground truth.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    A : float
        Scaling of HF function in LF model. Default 0.5.
    B : float
        Linear trend coefficient. Default 10.0.
    C : float
        Constant shift. Default -5.0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        A: float = 0.5,
        B: float = 10.0,
        C: float = -5.0,
    ) -> None:
        models = [
            ForresterModelFunction(bkd),
            ForresterModelFunction(bkd, A=A, B=B, C=C),
        ]
        costs = bkd.array([1.0, 0.1])
        prior = IndependentJoint(
            [UniformMarginal(0.0, 1.0, bkd)],
            bkd,
        )
        self._problem = MultifidelityForwardUQProblem(
            "forrester_ensemble_2model",
            models,
            costs,
            prior,
            description="2-model multi-fidelity Forrester ensemble",
        )

    def problem(
        self,
    ) -> MultifidelityForwardUQProblem[FunctionProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]
