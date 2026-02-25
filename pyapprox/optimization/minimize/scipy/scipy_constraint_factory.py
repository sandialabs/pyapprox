from typing import Sequence, List, Union, cast, Any
from functools import partial

import numpy as np
from scipy.optimize import (
    NonlinearConstraint,
    LinearConstraint as ScipyLinearConstraint,
)

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
    NonlinearConstraintProtocol,
    NonlinearConstraintProtocolWithJacobianAndWHVP,
)
from pyapprox.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)


def _numpy_constraint_hess_from_whvp(
    constraint: Any,
    sample: Array,
    weights: Array,
) -> Array:
    nvars = sample.shape[0]
    actions = []
    for ii in range(nvars):
        vec = np.zeros((nvars, 1))
        vec[ii] = 1.0
        actions.append(
            constraint.whvp(sample[:, None], vec, weights[:, None])[:, 0]
        )
    return np.stack(actions, axis=1)


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
                (
                    partial(_numpy_constraint_hess_from_whvp, con)
                    if hasattr(con, "whvp")
                    else None
                ),
                keep_feasible=getattr(con, "_keep_feasible", False),
            )

            converted_nonlinear_constraints.append(scipy_con)

    return converted_linear_constraints + converted_nonlinear_constraints
