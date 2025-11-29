from typing import (
    Protocol,
    runtime_checkable,
    Generic,
    Union,
    Sequence,
    List,
    Any,
)

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)


from typing import Protocol, runtime_checkable, Generic
from pyapprox.typing.util.backend import Array, Backend
from scipy.optimize import LinearConstraint as ScipyLinearConstraint


@runtime_checkable
class LinearConstraintProtocol(Protocol, Generic[Array]):
    def to_scipy(self) -> ScipyLinearConstraint: ...

    def A(self) -> Array: ...

    def lb(self) -> Array: ...

    def ub(self) -> Array: ...

    def bkd(self) -> Backend[Array]: ...


@runtime_checkable
class NonlinearConstraintProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def lb(self) -> Array: ...

    def ub(self) -> Array: ...


@runtime_checkable
class NonlinearConstraintProtocolWithJacobian(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def lb(self) -> Array: ...

    def ub(self) -> Array: ...

    def jacobian(self, sample: Array) -> Array: ...


@runtime_checkable
class NonlinearConstraintProtocolWithJacobianAndHVP(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def lb(self) -> Array: ...

    def ub(self) -> Array: ...

    def jacobian(self, sample: Array) -> Array: ...

    def weighted_hvp(self, sample: Array, vec: Array) -> Array: ...


UnionOfNonlinearConstraintProtocols = Union[
    NonlinearConstraintProtocol[Array],
    NonlinearConstraintProtocolWithJacobian[Array],
    NonlinearConstraintProtocolWithJacobianAndHVP[Array],
]

SequenceOfUnionOfConstraintProtocols = Sequence[
    Union[
        UnionOfNonlinearConstraintProtocols[Array],
        LinearConstraintProtocol[Array],
    ]
]
