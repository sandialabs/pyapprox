from typing import Union

from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)
from pyapprox.typing.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
)

ObjectiveProtocol = FunctionProtocol
ObjectiveWithJacobianProtocol = FunctionWithJacobianProtocol
ObjectiveWithJacobianAndHVPProtocol = FunctionWithJacobianAndHVPProtocol
