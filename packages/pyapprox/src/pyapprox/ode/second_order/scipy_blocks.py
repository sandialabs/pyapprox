"""The second-order state storages on the SciPy CSR stack.

Kept apart from the stack-generic storages so importing those does not
import SciPy.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.second_order.assembled_blocks import AssembledStateBlocks
from pyapprox.util.linalg.operators import LinearSolverProtocol, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import (
    CSRFormat,
    NumpyRowPartition,
    NumpyVectorSpace,
    ScipyCSROperator,
    ScipySparseLU,
)

_Vector = NDArray[np.float64]


def scipy_csr_state_blocks(
    half: SpaceDescriptor,
    space: SpaceDescriptor,
    solver: Optional[LinearSolverProtocol[_Vector, CSRFormat]] = None,
) -> AssembledStateBlocks[_Vector, _Vector, CSRFormat]:
    """The assembled storage on the SciPy CSR stack, in float64.

    Parameters
    ----------
    half, space : SpaceDescriptor
        One block (size :math:`n`) and :math:`w` (:math:`2n`).
    solver : LinearSolverProtocol, optional
        Solves at size :math:`2n`; a sparse LU by default.
    """
    half_space = NumpyVectorSpace(half, np.float64)
    whole = NumpyVectorSpace(space, np.float64)
    return AssembledStateBlocks(
        half_space,
        whole,
        NumpyRowPartition(whole, [half_space, half_space]),
        ScipyCSROperator.identity(half, np.float64),
        ScipySparseLU() if solver is None else solver,
    )
