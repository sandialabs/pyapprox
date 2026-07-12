"""Optimizer-side protocol names.

``ObjectiveProtocol`` is the Derivatives-bundle protocol
(``pyapprox.interface.functions.protocols.objective``): evaluation plus a
``derivatives()`` accessor.

The two ``ObjectiveWith*`` aliases are legacy capability-tier names kept
only until the remaining importers are migrated (deleted in Phase 5 of the
derivatives refactor). Do not use them in new code.
"""

from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
)
from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)

__all__ = [
    "ObjectiveProtocol",
    "ObjectiveWithJacobianProtocol",
    "ObjectiveWithJacobianAndHVPProtocol",
]

# Legacy aliases — removal planned; do not use in new code.
ObjectiveWithJacobianProtocol = FunctionWithJacobianProtocol
ObjectiveWithJacobianAndHVPProtocol = FunctionWithJacobianAndHVPProtocol
