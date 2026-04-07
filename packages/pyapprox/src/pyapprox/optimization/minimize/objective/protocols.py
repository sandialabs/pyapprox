from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
)
from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)

ObjectiveProtocol = FunctionProtocol
ObjectiveWithJacobianProtocol = FunctionWithJacobianProtocol
ObjectiveWithJacobianAndHVPProtocol = FunctionWithJacobianAndHVPProtocol
