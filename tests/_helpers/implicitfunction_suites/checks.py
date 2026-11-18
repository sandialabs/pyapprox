"""Finite-difference checks of stack operators, for the suites.

``DerivativeChecker`` differentiates functions of backend arrays, so a
suite checks a stack function through the case's conversions: its
values as arrays, and an operator's matrix from its action on unit
vectors built through the case. These are test-side conversions; the
code under test still sees only stack vectors.
"""

from typing import Callable

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.util.linalg.operators import LinearMapProtocol
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V

FD_TOL = 1e-6


def operator_matrix(case: TypedStackCase[V, F, A], op: LinearMapProtocol[V]) -> A:
    """The matrix of ``op``, column by column from its action."""
    bkd, domain = case.bkd, op.domain()
    eye = bkd.eye(domain.size)
    return bkd.hstack(
        [
            case.array(op.apply(case.vector(eye[:, j : j + 1], domain)))
            for j in range(domain.size)
        ]
    )


def min_fd_error(
    case: TypedStackCase[V, F, A],
    nqoi: int,
    nvars: int,
    fun: Callable[[A], A],
    jacobian: Callable[[A], A],
    sample: A,
) -> float:
    """The smallest relative central-difference error of ``jacobian``."""
    errors = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=nvars, fun=fun, jacobian=jacobian, bkd=case.bkd
        )
    ).check_derivatives(sample, relative=True, central=True)[0]
    return case.bkd.to_float(case.bkd.min(errors))
