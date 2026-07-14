"""Constraint protocols for optimizer binding.

``NonlinearConstraintProtocol`` lives at its canonical location,
``pyapprox.interface.functions.protocols.objective`` (evaluation, bounds,
and a ``derivatives()`` accessor — capability travels in the bundle,
never as protocol tiers); it is imported here only to build
``SequenceOfConstraintProtocols``. Import it from the canonical module.
"""

from typing import (
    Generic,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from scipy.optimize import LinearConstraint as ScipyLinearConstraint

from pyapprox.interface.functions.protocols.constraint import (
    NonlinearConstraintProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

__all__ = [
    "LinearConstraintProtocol",
    "SequenceOfConstraintProtocols",
]


@runtime_checkable
class LinearConstraintProtocol(Protocol, Generic[Array]):
    def to_scipy(self) -> ScipyLinearConstraint: ...

    def A(self) -> Array: ...

    def lb(self) -> Array: ...

    def ub(self) -> Array: ...

    def bkd(self) -> Backend[Array]: ...


SequenceOfConstraintProtocols = Sequence[
    Union[
        NonlinearConstraintProtocol[Array],
        LinearConstraintProtocol[Array],
    ]
]
