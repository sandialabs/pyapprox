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
from pyapprox.typing.optimization.linear_constraint import (
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


def validate_nonlinear_constraint(obj: object) -> None:
    """
    Validate that the given object satisfies one of the nonlinear constraint protocols.

    Parameters
    ----------
    obj : object
        The object to validate.

    Raises
    ------
    TypeError
        If the object does not satisfy any of the nonlinear constraint protocols.
    """
    if not isinstance(
        obj,
        (
            NonlinearConstraintProtocol,
            NonlinearConstraintProtocolWithJacobian,
            NonlinearConstraintProtocolWithJacobianAndHVP,
        ),
    ):
        raise TypeError(
            "The provided object must satisfy one of the following nonlinear constraint protocols: "
            "'NonlinearConstraintProtocol', "
            "'NonlinearConstraintProtocolWithJacobian', or "
            "'NonlinearConstraintProtocolWithJacobianAndHVP'. Got an object of"
            f" type {type(obj).__name__}."
        )


def validate_constraints(constraints: Sequence[Any]) -> None:
    """
    Validate that all objects in the list satisfy one of the constraint protocols
    (either nonlinear or linear).

    Parameters
    ----------
    constraints : List[object]
        The list of objects to validate.

    Raises
    ------
    TypeError
        If any object in the list does not satisfy the constraint protocols.
    """
    for idx, obj in enumerate(constraints):
        if not isinstance(
            obj,
            (
                NonlinearConstraintProtocol,
                NonlinearConstraintProtocolWithJacobian,
                NonlinearConstraintProtocolWithJacobianAndHVP,
                PyApproxLinearConstraint,
            ),
        ):
            raise TypeError(
                f"The object at index {idx} must satisfy one of the "
                "following protocols: "
                "'NonlinearConstraintProtocol', "
                "'NonlinearConstraintProtocolWithJacobian', "
                "'NonlinearConstraintProtocolWithJacobianAndHVP', or "
                "'PyApproxLinearConstraint'. "
                f"Got an object of type {type(obj).__name__}."
            )
