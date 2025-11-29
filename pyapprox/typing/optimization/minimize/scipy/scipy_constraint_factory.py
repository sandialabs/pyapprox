from typing import Sequence, List, Union, cast
import numpy as np
from scipy.optimize import (
    NonlinearConstraint,
    LinearConstraint as ScipyLinearConstraint,
)
from pyapprox.typing.util.backend import Array
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.typing.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
    NonlinearConstraintProtocol,
)
from pyapprox.typing.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)


def convert_constraints(
    constraints: SequenceOfConstraintProtocols[Array],
) -> List[Union[ScipyLinearConstraint, NonlinearConstraint]]:
    """
    Convert constraints into SciPy-compatible constraints.

    Parameters
    ----------
    constraints : SequenceOfUnionOfConstraintProtocols[Array]
        List of constraints to convert.

    Returns
    -------
    List[Union[ScipyLinearConstraint, NonlinearConstraint]]
        List of SciPy-compatible constraints.
    """
    converted_linear_constraints = []
    converted_nonlinear_constraints = []

    for constraint in constraints:
        if isinstance(constraint, PyApproxLinearConstraint):
            # Convert PyApproxLinearConstraint to SciPy LinearConstraint
            converted_linear_constraints.append(constraint.to_scipy())
        else:
            # Wrap nonlinear constraints using numpy_function_wrapper_factory
            con = numpy_function_wrapper_factory(
                cast(NonlinearConstraintProtocol[Array], constraint),
            )

            # Convert to SciPy NonlinearConstraint
            scipy_con = NonlinearConstraint(
                lambda x: con(x[:, None])[:, 0],
                constraint.bkd().to_numpy(constraint.lb()),
                constraint.bkd().to_numpy(constraint.ub()),
                lambda x: (
                    con.jacobian(x[:, None])
                    if hasattr(con, "jacobian")
                    else "2-point"
                ),
                con.whvp if hasattr(con, "whvp") else None,
                keep_feasible=getattr(con, "_keep_feasible", False),
            )
            print(scipy_con.hess)

            converted_nonlinear_constraints.append(scipy_con)

    return converted_linear_constraints + converted_nonlinear_constraints
