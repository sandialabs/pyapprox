from typing import List, Union, Optional, cast

from scipy.optimize import LinearConstraint as ScipyLinearConstraint

from pyapprox.typing.util.backend import Array
from pyapprox.typing.optimization.linear_constraint import (
    PyApproxLinearConstraint,
)
from pyapprox.typing.optimization.constraint_protocols import (
    UnionOfNonlinearConstraintProtocols,
    SequenceOfUnionOfConstraintProtocols,
)
from pyapprox.typing.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)


from typing import Sequence, List, Union, cast
import numpy as np
from scipy.optimize import (
    NonlinearConstraint,
    LinearConstraint as ScipyLinearConstraint,
)
from pyapprox.typing.util.backend import Array
from pyapprox.typing.optimization.linear_constraint import (
    PyApproxLinearConstraint,
)
from pyapprox.typing.optimization.constraint_protocols import (
    UnionOfNonlinearConstraintProtocols,
    SequenceOfUnionOfConstraintProtocols,
)
from pyapprox.typing.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)


def convert_constraints(
    constraints: SequenceOfUnionOfConstraintProtocols[Array],
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
                cast(UnionOfNonlinearConstraintProtocols[Array], constraint),
                sample_ndim=1,
            )

            # Convert to SciPy NonlinearConstraint
            scipy_con = NonlinearConstraint(
                con,
                constraint.bkd().to_numpy(constraint.lb()),
                constraint.bkd().to_numpy(constraint.ub()),
                con.jac if hasattr(con, "jacobian") else "2-point",
                con.weighted_hvp if hasattr(con, "weighted_hvp") else None,
                keep_feasible=getattr(con, "_keep_feasible", False),
            )
            converted_nonlinear_constraints.append(scipy_con)

    return converted_linear_constraints + converted_nonlinear_constraints
