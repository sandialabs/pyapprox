"""Convert PyApprox constraints into SciPy constraint objects.

Nonlinear constraints are read through their Derivatives bundle (via the
migration shim ``as_derivatives``): an absent jacobian hands SciPy the
string ``"2-point"`` so SciPy does its own finite differencing; an absent
whvp hands SciPy ``hess=None``. Module-level functions + ``partial`` (not
closures) keep the converted constraints picklable.
"""

from functools import partial
from typing import List, Union

import numpy as np
from scipy.optimize import (
    LinearConstraint as ScipyLinearConstraint,
)
from scipy.optimize import (
    NonlinearConstraint,
)

from pyapprox.interface.functions.legacy_adapter import (
    as_derivatives,
)
from pyapprox.interface.functions.numpy.adapter import (
    NumpyArray,
    NumpyDerivativesAdapter,
    NumpyFn,
    NumpyWHVPFn,
)
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    NonlinearConstraintProtocol,
    SequenceOfConstraintProtocols,
)
from pyapprox.util.backends.protocols import Array


def _nonlinear_constraint_fun(
    adapter: NumpyDerivativesAdapter[Array], x: NumpyArray
) -> NumpyArray:
    return np.asarray(adapter(x[:, None])[:, 0])


def _nonlinear_constraint_jac(np_jac: NumpyFn, x: NumpyArray) -> NumpyArray:
    return np_jac(x[:, None])


def _numpy_constraint_hess_from_whvp(
    np_whvp: NumpyWHVPFn, x: NumpyArray, weights: NumpyArray
) -> NumpyArray:
    """Dense constraint Hessian sum_i w_i * hess c_i, column by column."""
    nvars = x.shape[0]
    actions = []
    for ii in range(nvars):
        vec = np.zeros((nvars, 1))
        vec[ii] = 1.0
        actions.append(np_whvp(x[:, None], vec, weights[:, None])[:, 0])
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
            continue
        if not isinstance(constraint, NonlinearConstraintProtocol):
            raise TypeError(
                "constraint must satisfy NonlinearConstraintProtocol or be "
                f"a PyApproxLinearConstraint, got {type(constraint).__name__}"
            )
        adapter = NumpyDerivativesAdapter(
            constraint, as_derivatives(constraint)
        )
        # capability captured once; SciPy reacts to what is available
        np_jac = adapter.jacobian()
        np_whvp = adapter.whvp()  # resolved with the constraint's OWN nqoi
        bkd = constraint.bkd()
        converted_nonlinear_constraints.append(
            NonlinearConstraint(
                partial(_nonlinear_constraint_fun, adapter),
                bkd.to_numpy(constraint.lb()),
                bkd.to_numpy(constraint.ub()),
                jac="2-point"
                if np_jac is None
                else partial(_nonlinear_constraint_jac, np_jac),
                hess=None
                if np_whvp is None
                else partial(_numpy_constraint_hess_from_whvp, np_whvp),
            )
        )

    return converted_linear_constraints + converted_nonlinear_constraints
