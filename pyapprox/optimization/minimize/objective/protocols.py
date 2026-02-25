from typing import Union

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
)

ObjectiveProtocol = FunctionProtocol
ObjectiveWithJacobianProtocol = FunctionWithJacobianProtocol
ObjectiveWithJacobianAndHVPProtocol = FunctionWithJacobianAndHVPProtocol
